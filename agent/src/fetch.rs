//! Fetch verb (Track C, the live side) — the agent serves a tensor's byte
//! span over the authenticated frame line.
//!
//! The head sends a signed line:
//!
//!     fetch <shard_path> <tensor> <offset> <length>
//!
//! The agent opens the BMTS shard on its own disk, reads the span
//! (size-stamp-checked — a span past the declared tensor length is refused,
//! never truncated), then switches the socket to frame mode and pumps the
//! bytes back. Bulk rides frames, never the line (Constitution Art. 6).
//! The size-stamp is the declared tensor length; the head cross-checks it.

use ouro_cluster::bmts::BmtsShard;
use ouro_cluster::op::Span;
use ouro_cluster::transport::auth::Secret;
use ouro_cluster::transport::frames::{pump_send, FrameSession, DEFAULT_CHUNK, DEFAULT_WINDOW};
use std::io::{Cursor, Write};
use std::net::TcpStream;

/// Parse a signed `fetch <shard> <tensor> <offset> <length>` line. Returns
/// (shard path, tensor, span) — `None` if not a fetch line or auth fails.
pub fn fetch_verb(secret: &Secret, line: &str) -> Option<(String, String, Span)> {
    let (_seq, body) = ouro_cluster::transport::auth::open_line(secret, line).ok()?;
    let rest = body.strip_prefix("fetch ")?;
    let mut parts = rest.split_whitespace();
    let shard = parts.next()?.to_string();
    let tensor = parts.next()?.to_string();
    let offset: u64 = parts.next()?.parse().ok()?;
    let length: u64 = parts.next()?.parse().ok()?;
    if length == 0 {
        return None;
    }
    Some((shard, tensor, Span { offset, length }))
}

/// Serve one fetch: open the shard, read the span, frame-mode the socket,
/// pump the bytes back. Returns bytes sent.
pub fn handle_fetch(
    secret: Secret,
    sock: TcpStream,
    shard_path: &str,
    tensor: &str,
    span: Span,
) -> anyhow::Result<u64> {
    let shard = BmtsShard::open(shard_path)?;
    let declared = shard.tensor_bytes(tensor)?.len() as u64;
    let payload = shard.read_span(tensor, span)?;

    let mut session = FrameSession::new(sock, secret);
    // Size-stamp line first: the declared tensor length (the integrity gate
    // the head cross-checks the fetch against).
    write_u64(session.get_mut(), declared)?;
    let mut cursor = Cursor::new(payload.bytes().to_vec());
    let sent = pump_send(&mut cursor, &mut session, DEFAULT_CHUNK, DEFAULT_WINDOW)?;
    Ok(sent)
}

/// Send a single u64 (big-endian) as a line-mode preamble to the socket.
fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    let buf = v.to_be_bytes();
    w.write_all(&buf)?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ouro_cluster::bmts::{write_shard, BmtsTensor};
    use ouro_cluster::transport::auth;
    use ouro_cluster::transport::frames::pump_recv;
    use std::io::Read;
    use std::net::TcpListener;

    const KEY: Secret = [9u8; 32];

    #[test]
    fn test_fetch_verb_parses() {
        let line = auth::sign_line(&KEY, 1, "fetch /s/draft.bmts blk.0.attn_q.weight 128 4096");
        let (shard, tensor, span) = fetch_verb(&KEY, &line).unwrap();
        assert_eq!(shard, "/s/draft.bmts");
        assert_eq!(tensor, "blk.0.attn_q.weight");
        assert_eq!(span, Span { offset: 128, length: 4096 });
    }

    #[test]
    fn test_fetch_verb_zero_length_refused() {
        let line = auth::sign_line(&KEY, 1, "fetch /s/draft.bmts blk.0.w 0 0");
        assert!(fetch_verb(&KEY, &line).is_none());
    }

    #[test]
    fn test_fetch_verb_auth_fail() {
        let line = auth::sign_line(&[10u8; 32], 1, "fetch /s/d.bmts t 0 4");
        assert!(fetch_verb(&KEY, &line).is_none());
    }

    /// The live fetch, on loopback: a head sends a signed fetch line, the
    /// agent serves the tensor span over frames, and the bytes arrive intact
    /// on the exact span requested.
    #[test]
    fn test_handle_fetch_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ouro-fetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard_path = dir.join("draft.bmts");
        let blob: Vec<u8> = (0..10_000u32).map(|i| (i % 256) as u8).collect();
        let tensor = "blk.0.attn_q.weight";
        write_shard(
            shard_path.to_str().unwrap(),
            1,
            &[BmtsTensor { name: tensor.into(), shape: vec![16, 625], dtype: 34, offset: 0, length: 10_000 }],
            &blob,
        )
        .unwrap();

        // Server: accept, read the fetch line, handle it.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let shard_path_str = shard_path.to_str().unwrap().to_string();
        let server = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut sock = sock;
            let mut line = String::new();
            // Manual line read (no BufReader: it would buffer frame bytes).
            let mut one = [0u8; 1];
            while !line.ends_with('\n') {
                sock.read_exact(&mut one).unwrap();
                line.push(one[0] as char);
                if line.len() > 64 * 1024 {
                    panic!("fetch line too long");
                }
            }
            let (shard, tensor, span) = fetch_verb(&KEY, line.trim()).unwrap();
            let sent = handle_fetch(KEY, sock, &shard, &tensor, span).unwrap();
            sent
        });

        // Head: connect, send the fetch line, read stamp + frames.
        let mut head = TcpStream::connect(addr).unwrap();
        let span = Span { offset: 1000, length: 4000 };
        let line = auth::sign_line(&KEY, 1, &format!("fetch {} {} {} {}", shard_path_str, tensor, span.offset, span.length));
        head.write_all(line.as_bytes()).unwrap();
        head.write_all(b"\n").unwrap();

        // Read the size-stamp (8 bytes, big-endian).
        let mut stamp_buf = [0u8; 8];
        head.read_exact(&mut stamp_buf).unwrap();
        let declared = u64::from_be_bytes(stamp_buf);
        assert_eq!(declared, 10_000, "size-stamp = declared tensor length");

        // Pump the span back.
        let mut session = FrameSession::new(head, KEY);
        let mut received = Vec::new();
        let (n, _) = pump_recv(&mut session, &mut received, DEFAULT_WINDOW).unwrap();

        let sent = server.join().unwrap();
        assert_eq!(sent as usize, 4000);
        assert_eq!(n as usize, 4000);
        // The bytes must be exactly the requested span of the tensor.
        assert_eq!(received, &blob[1000..5000]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
