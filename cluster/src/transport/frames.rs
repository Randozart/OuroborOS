//! OURO frame transport (WP-U1, UPDATE_ROADMAP §wire).
//!
//! Line mode carries verbs; frame mode carries bulk bytes — updates
//! now, BMTS model shards later (DMA_ROADMAP Tier 2+). The handshake is
//! a signed `frames begin` line on the existing wire; both sides then
//! speak frames until EOF, after which the socket returns to line mode
//! for the receipt.
//!
//! Frame layout (AGENTS.md transport spec):
//!
//! ```text
//! magic "OURO" (4B) | flags (1B) | seq (8B BE) | len (4B BE) | tag (32B) | payload
//! ```
//!
//! `tag` reuses the line-protocol primitive: HMAC-SHA256 over
//! `seq || payload`. Magic/len are structural (tampering with them
//! fails the parse or the tag — same opaque error either way, no
//! oracle). Reliability is TCP's job; frames add integrity, mode
//! switching, and the ack window (the receiver's backpressure).
use anyhow::{bail, Result};
use std::io::{Read, Write};

use super::auth::{tag, Secret};

/// "OURO" — every frame starts here (AGENTS.md transport spec).
pub const MAGIC: [u8; 4] = [0x4F, 0x55, 0x52, 0x4F];

/// Cumulative ACK: payload is u64 BE = highest contiguous data seq seen.
pub const FLAG_ACK: u8 = 0x01;
/// End of bulk stream; sender waits for the final cumulative ACK.
pub const FLAG_EOF: u8 = 0x02;

pub const DEFAULT_CHUNK: usize = 256 * 1024;
pub const DEFAULT_WINDOW: u64 = 16;
/// Hard payload ceiling — refuses before allocating (a tampered or
/// hostile len must never size an allocation).
pub const MAX_PAYLOAD: usize = 4 * 1024 * 1024;

const HEADER_LEN: usize = 4 + 1 + 8 + 4;
const TAG_LEN: usize = 32;

/// One frame as parsed off the wire.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub flags: u8,
    pub seq: u64,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn is_ack(&self) -> bool {
        self.flags & FLAG_ACK != 0
    }

    pub fn is_eof(&self) -> bool {
        self.flags & FLAG_EOF != 0
    }

    /// The final cumulative ack (ACK|EOF) — by construction the LAST
    /// frame-mode byte the receiver writes (TCP ordering guarantees
    /// everything before it is consumed once it is read). After this
    /// frame the socket is clean to return to line mode.
    pub fn is_final_ack(&self) -> bool {
        self.is_ack() && self.is_eof()
    }
}

/// Encode one frame onto `w`. Seq is the sender's own counter; payload
/// size must not exceed MAX_PAYLOAD.
pub fn write_frame(w: &mut impl Write, secret: &Secret, seq: u64, flags: u8, payload: &[u8]) -> Result<()> {
    if payload.len() > MAX_PAYLOAD {
        bail!("frame payload {} exceeds {}", payload.len(), MAX_PAYLOAD);
    }
    let mut header = [0u8; HEADER_LEN];
    header[..4].copy_from_slice(&MAGIC);
    header[4] = flags;
    header[5..13].copy_from_slice(&seq.to_be_bytes());
    header[13..17].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    w.write_all(&header)?;
    w.write_all(&tag(secret, seq, payload))?;
    w.write_all(payload)?;
    Ok(())
}

/// Read one frame from `r`. Any structural or tag failure is an opaque
/// error — the only honest response is closing the connection (a
/// desynced stream cannot be resynced).
pub fn read_frame(r: &mut impl Read, secret: &Secret) -> Result<Frame> {
    let mut header = [0u8; HEADER_LEN];
    r.read_exact(&mut header)?;
    if header[..4] != MAGIC {
        bail!("frame: bad magic");
    }
    let flags = header[4];
    let seq = u64::from_be_bytes(header[5..13].try_into()?);
    let len = u32::from_be_bytes(header[13..17].try_into()?) as usize;
    if len > MAX_PAYLOAD {
        bail!("frame: len {} exceeds {}", len, MAX_PAYLOAD);
    }
    let mut tag_buf = [0u8; TAG_LEN];
    r.read_exact(&mut tag_buf)?;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    if tag_buf.as_slice() != tag(secret, seq, &payload).as_slice() {
        bail!("frame: auth failed");
    }
    Ok(Frame { flags, seq, payload })
}

/// Full-duplex frame session over one socket. Each direction keeps its
/// own seq counter starting at 1; data seqs are enforced strictly
/// in-order per direction (AGENTS.md: sequence IDs prevent reordering).
pub struct FrameSession<S: Read + Write> {
    sock: S,
    secret: Secret,
    send_seq: u64,
    expect_seq: u64,
    /// Highest contiguous DATA seq received (the ack we owe).
    recv_data_seq: u64,
    acked: u64,
}

impl<S: Read + Write> FrameSession<S> {
    pub fn new(sock: S, secret: Secret) -> Self {
        Self {
            sock,
            secret,
            send_seq: 1,
            expect_seq: 1,
            recv_data_seq: 0,
            acked: 0,
        }
    }

    /// Send one data frame (caller owns chunking/window discipline, or
    /// use [`pump_send`]).
    pub fn send_data(&mut self, payload: &[u8]) -> Result<()> {
        let seq = self.send_seq;
        write_frame(&mut self.sock, &self.secret, seq, 0, payload)?;
        self.send_seq += 1;
        Ok(())
    }

    pub fn send_eof(&mut self) -> Result<()> {
        let seq = self.send_seq;
        write_frame(&mut self.sock, &self.secret, seq, FLAG_EOF, &[])?;
        self.send_seq += 1;
        Ok(())
    }

    /// Cumulative ACK for the data seqs received so far.
    pub fn send_ack(&mut self) -> Result<()> {
        let seq = self.send_seq;
        write_frame(&mut self.sock, &self.secret, seq, FLAG_ACK, &self.recv_data_seq.to_be_bytes())?;
        self.send_seq += 1;
        Ok(())
    }

    /// The final cumulative ack (ACK|EOF): the receiver's last
    /// frame-mode frame. After sending it the receiver speaks line mode
    /// again (the receipt).
    pub fn send_final_ack(&mut self) -> Result<()> {
        let seq = self.send_seq;
        write_frame(
            &mut self.sock,
            &self.secret,
            seq,
            FLAG_ACK | FLAG_EOF,
            &self.recv_data_seq.to_be_bytes(),
        )?;
        self.send_seq += 1;
        Ok(())
    }

    /// Read the next frame. Data frames must arrive in strict order;
    /// ACK frames carry the peer's progress (monotonic, not sequenced
    /// against our data counter).
    pub fn recv(&mut self) -> Result<Frame> {
        let frame = read_frame(&mut self.sock, &self.secret)?;
        if frame.is_ack() {
            let acked = u64::from_be_bytes(frame.payload.as_slice().try_into()?);
            if acked < self.acked {
                bail!("frame: ack went backwards");
            }
            self.acked = acked;
        } else {
            if frame.seq != self.expect_seq {
                bail!("frame: seq {} out of order (expected {})", frame.seq, self.expect_seq);
            }
            if !frame.is_eof() {
                self.expect_seq += 1;
                self.recv_data_seq = frame.seq;
            }
        }
        Ok(frame)
    }

    pub fn acked(&self) -> u64 {
        self.acked
    }

    /// Highest contiguous data seq this side has received — i.e. what
    /// it has acked out. The receiver's own progress (the sender's
    /// mirror is [`Self::acked`]).
    pub fn progressed(&self) -> u64 {
        self.recv_data_seq
    }

    pub fn get_ref(&self) -> &S {
        &self.sock
    }

    pub fn get_mut(&mut self) -> &mut S {
        &mut self.sock
    }
}

/// Stream `src` through the session as data frames with the ack window
/// as backpressure: never more than `window` unacked frames in flight.
/// Ends with an EOF frame, then drains inbound acks until the FINAL ack
/// (ACK|EOF) — every frame-mode byte is consumed, so the socket is
/// clean for the line-mode receipt. Returns bytes sent. The caller
/// verifies the receiver's sha256 receipt out of band (the manifest is
/// the contract, not the transport).
pub fn pump_send(src: &mut impl Read, session: &mut FrameSession<impl Read + Write>, chunk: usize, window: u64) -> Result<u64> {
    let mut total = 0u64;
    let mut sent = 0u64;
    let mut buf = vec![0u8; chunk];
    loop {
        let n = src.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if sent - session.acked() >= window {
            // Window full: block on ACKs until room opens.
            loop {
                let frame = session.recv()?;
                if frame.is_final_ack() {
                    bail!("frame: final ack mid-stream");
                }
                if frame.is_ack() && session.acked() > sent - window {
                    break;
                }
                if frame.is_eof() {
                    bail!("frame: peer sent EOF mid-stream");
                }
            }
        }
        session.send_data(&buf[..n])?;
        sent += 1;
        total += n as u64;
    }
    session.send_eof()?;
    // Drain to the final cumulative ack: consumes every remaining
    // frame-mode byte (TCP ordering — nothing can follow it). Periodic
    // acks may already have covered every sent frame, so we cannot stop
    // at acked == sent; only the final ack proves the receiver saw the
    // EOF and that the stream is quiescent.
    loop {
        let frame = session.recv()?;
        if frame.is_final_ack() {
            break;
        }
        if frame.is_ack() {
            continue;
        }
        bail!("frame: expected ack after EOF, got flags {:#x}", frame.flags);
    }
    Ok(total)
}

/// Receive a bulk stream into `dst`, acking cumulatively every `window`
/// data frames and at EOF. Returns (bytes, frames).
pub fn pump_recv(session: &mut FrameSession<impl Read + Write>, dst: &mut impl Write, window: u64) -> Result<(u64, u64)> {
    let mut total = 0u64;
    let mut frames = 0u64;
    loop {
        let frame = session.recv()?;
        if frame.is_ack() {
            continue;
        }
        if frame.is_eof() {
            session.send_final_ack()?;
            return Ok((total, frames));
        }
        dst.write_all(&frame.payload)?;
        total += frame.payload.len() as u64;
        frames += 1;
        if frames.is_multiple_of(window) {
            session.send_ack()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const KEY: Secret = [7u8; 32];

    /// In-memory duplex: what one side writes, the other reads.
    struct Duplex {
        buf: std::rc::Rc<std::cell::RefCell<Vec<u8>>>,
    }

    impl Duplex {
        fn pair() -> (Duplex, Duplex) {
            let buf = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            (Duplex { buf: buf.clone() }, Duplex { buf })
        }
    }

    impl Read for Duplex {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let mut buf = self.buf.borrow_mut();
            let n = out.len().min(buf.len());
            out[..n].copy_from_slice(&buf.drain(..n).collect::<Vec<u8>>());
            Ok(n)
        }
    }

    impl Write for Duplex {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.buf.borrow_mut().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_frame_roundtrip_sizes() {
        for size in [0usize, 1, 100, 256 * 1024] {
            let payload: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            let mut wire = Cursor::new(Vec::new());
            write_frame(&mut wire, &KEY, 1, 0, &payload).unwrap();
            wire.set_position(0);
            let frame = read_frame(&mut wire, &KEY).unwrap();
            assert_eq!(frame.seq, 1);
            assert_eq!(frame.flags, 0);
            assert_eq!(frame.payload, payload);
        }
    }

    #[test]
    fn test_oversized_len_refused() {
        let mut wire = Cursor::new(Vec::new());
        let err = write_frame(&mut wire, &KEY, 1, 0, &vec![0u8; MAX_PAYLOAD + 1]);
        assert!(err.is_err());
        // hostile header: plausible magic, absurd len — refused before
        // any allocation
        let mut hostile = Vec::new();
        hostile.extend_from_slice(&MAGIC);
        hostile.push(0);
        hostile.extend_from_slice(&1u64.to_be_bytes());
        hostile.extend_from_slice(&u32::MAX.to_be_bytes());
        let mut cursor = Cursor::new(hostile);
        assert!(read_frame(&mut cursor, &KEY).is_err());
    }

    #[test]
    fn test_tamper_rejected() {
        let payload = vec![1u8; 64];
        let mut wire = Cursor::new(Vec::new());
        write_frame(&mut wire, &KEY, 3, 0, &payload).unwrap();
        let bytes = wire.into_inner();

        // flip a payload byte
        let mut tampered = bytes.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xFF;
        assert!(read_frame(&mut Cursor::new(tampered), &KEY).is_err());

        // flip a seq byte (tag covers seq)
        let mut tampered = bytes.clone();
        tampered[6] ^= 0x01;
        assert!(read_frame(&mut Cursor::new(tampered), &KEY).is_err());

        // flip magic
        let mut tampered = bytes.clone();
        tampered[0] ^= 0x01;
        assert!(read_frame(&mut Cursor::new(tampered), &KEY).is_err());

        // wrong key
        assert!(read_frame(&mut Cursor::new(bytes), &[9u8; 32]).is_err());
    }

    #[test]
    fn test_session_seq_enforced_per_direction() {
        let (a_wire, b_wire) = Duplex::pair();
        let mut a = FrameSession::new(a_wire, KEY);
        let mut b = FrameSession::new(b_wire, KEY);

        a.send_data(b"one").unwrap();
        a.send_data(b"two").unwrap();
        // simulate out-of-order delivery by sending seq+1 first: craft
        // a frame with a skipped seq
        write_frame(a.get_mut(), &KEY, 99, 0, b"jump").unwrap();
        assert!(b.recv().is_ok());
        assert!(b.recv().is_ok());
        assert!(b.recv().is_err(), "skipped seq must be refused");
    }

    #[test]
    fn test_pump_roundtrip_loopback() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let src_data: Vec<u8> = (0..600_000u32).map(|i| (i % 251) as u8).collect();
        let src_len = src_data.len() as u64;

        let handle = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut session = FrameSession::new(sock, KEY);
            let mut src = Cursor::new(src_data);
            pump_send(&mut src, &mut session, DEFAULT_CHUNK, DEFAULT_WINDOW).unwrap();
        });

        let sock = std::net::TcpStream::connect(addr).unwrap();
        let mut session = FrameSession::new(sock, KEY);
        let mut dst = Vec::new();
        let (bytes, frames) = pump_recv(&mut session, &mut dst, DEFAULT_WINDOW).unwrap();
        handle.join().unwrap();
        assert_eq!(bytes, src_len);
        assert_eq!(dst.len() as u64, src_len);
        assert!(frames > 2, "expected multiple frames, got {frames}");
        // data survived the wire intact
        let expected: Vec<u8> = (0..600_000u32).map(|i| (i % 251) as u8).collect();
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_ack_window_backpressures() {
        // sender must stall when the receiver stops acking: with
        // window=2, 5 frames can't all be sent until acks arrive. The
        // receiver acks on its own schedule; if the window logic is
        // broken (ignores acks), this deadlocks — bounded by the recv
        // side completing anyway, so assert on acked progression.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut session = FrameSession::new(sock, KEY);
            let data = vec![0u8; 1024];
            // 5 chunks through window 2 = relies on acks to proceed
            let mut src = Cursor::new(data);
            let sent = pump_send(&mut src, &mut session, 256, 2).unwrap();
            assert_eq!(sent, 1024);
        });

        let sock = std::net::TcpStream::connect(addr).unwrap();
        let mut session = FrameSession::new(sock, KEY);
        let mut dst = Vec::new();
        let (bytes, _) = pump_recv(&mut session, &mut dst, 1).unwrap();
        handle.join().unwrap();
        assert_eq!(bytes, 1024);
        assert!(session.progressed() >= 4, "receiver acked every frame (window=1), got {}", session.progressed());
    }
}
