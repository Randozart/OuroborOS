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

/// Parse a signed `fetch-tensor <tensor> <offset> <length>` line — the
/// tensor-by-name fetch (the tail resolves where it keeps the tensor).
/// Returns (tensor, span), or `None` if not a fetch-tensor line or auth
/// fails.
pub fn fetch_tensor_verb(secret: &Secret, line: &str) -> Option<(String, Span)> {
    let (_seq, body) = ouro_cluster::transport::auth::open_line(secret, line).ok()?;
    let rest = body.strip_prefix("fetch-tensor ")?;
    let mut parts = rest.split_whitespace();
    let tensor = parts.next()?.to_string();
    let offset: u64 = parts.next()?.parse().ok()?;
    let length: u64 = parts.next()?.parse().ok()?;
    if length == 0 {
        return None;
    }
    Some((tensor, Span { offset, length }))
}

/// Resolve which shard in the shard_map contains `tensor`, and return the
/// resolved file path. The agent's own shard_map is the source of truth for
/// where it keeps its tensors.
fn find_shard_for_tensor(shard_map_path: &str, tensor: &str) -> anyhow::Result<String> {
    use ouro_cluster::pipeline::PipelinePlan;
    let plan = PipelinePlan::load(shard_map_path)?;
    for stage in &plan.nodes {
        // The shard file path may be relative to the shard map's parent.
        let shard_path = if std::path::Path::new(&stage.file).is_absolute() {
            std::path::PathBuf::from(&stage.file)
        } else {
            std::path::Path::new(shard_map_path)
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join(&stage.file)
        };
        if !shard_path.exists() {
            continue;
        }
        let shard = match BmtsShard::open(shard_path.to_str().unwrap_or(&stage.file)) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if shard.tensors.iter().any(|t| t.name == tensor) {
            return Ok(shard_path.to_str().unwrap_or(&stage.file).to_string());
        }
    }
    anyhow::bail!("tensor {tensor} not found in any shard of {shard_map_path}")
}

/// Serve a fetch-tensor: resolve the shard from the agent's shard_map,
/// read the span, pump the bytes back over frames. Returns bytes sent.
pub fn handle_fetch_tensor(
    secret: Secret,
    sock: TcpStream,
    shard_map_path: &str,
    tensor: &str,
    span: Span,
) -> anyhow::Result<u64> {
    let shard_path = find_shard_for_tensor(shard_map_path, tensor)?;
    handle_fetch(secret, sock, &shard_path, tensor, span)
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

    #[test]
    fn test_fetch_tensor_verb_parses() {
        let line = auth::sign_line(&KEY, 1, "fetch-tensor blk.0.attn_q.weight 128 4096");
        let (tensor, span) = fetch_tensor_verb(&KEY, &line).unwrap();
        assert_eq!(tensor, "blk.0.attn_q.weight");
        assert_eq!(span, Span { offset: 128, length: 4096 });
        // zero-length refused
        let bad = auth::sign_line(&KEY, 2, "fetch-tensor t 0 0");
        assert!(fetch_tensor_verb(&KEY, &bad).is_none());
    }

    /// The tail resolves where it keeps a tensor: a shard_map with two
    /// shards, the tensor lives in shard 2, and find_shard_for_tensor
    /// returns shard 2's resolved path.
    #[test]
    fn test_find_shard_for_tensor() {
        let dir = std::env::temp_dir().join(format!("ouro-find-shard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let map_path = dir.join("shard_map.json");
        let blob = vec![0u8; 64];
        write_shard(
            dir.join("shard_1.bmts").to_str().unwrap(),
            1,
            &[BmtsTensor { name: "blk.0.attn_q.weight".into(), shape: vec![4, 4], dtype: 34, offset: 0, length: 64 }],
            &blob,
        )
        .unwrap();
        write_shard(
            dir.join("shard_2.bmts").to_str().unwrap(),
            2,
            &[BmtsTensor { name: "blk.0.attn_k.weight".into(), shape: vec![4, 4], dtype: 34, offset: 0, length: 64 }],
            &blob,
        )
        .unwrap();
        let map = serde_json::json!({
            "model": "test",
            "nodes": [
                { "node": 1, "file": "shard_1.bmts", "layers": [0], "tensors": 1, "bytes": 64 },
                { "node": 2, "file": "shard_2.bmts", "layers": [1], "tensors": 1, "bytes": 64 },
            ]
        });
        std::fs::write(&map_path, serde_json::to_string(&map).unwrap()).unwrap();

        let p = find_shard_for_tensor(map_path.to_str().unwrap(), "blk.0.attn_k.weight").unwrap();
        assert!(p.ends_with("shard_2.bmts"), "resolved to {p}");
        let missing = find_shard_for_tensor(map_path.to_str().unwrap(), "no.such.tensor");
        assert!(missing.is_err());

        std::fs::remove_dir_all(&dir).ok();
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
            handle_fetch(KEY, sock, &shard, &tensor, span).unwrap()
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
