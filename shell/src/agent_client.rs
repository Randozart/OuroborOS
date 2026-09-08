use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use ouro_cluster::transport::auth::{self, Secret};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

static REQUEST_SEQ: AtomicU64 = AtomicU64::new(1);
static SECRET_CACHE: OnceLock<Result<Secret, String>> = OnceLock::new();

/// Secret for signing the wire, loaded once from `OURO_SECRET_FILE`.
/// Missing or invalid secret → every wire call fails (mandatory gate,
/// no bypass).
fn cached_secret() -> Result<Secret> {
    let cached = SECRET_CACHE.get_or_init(|| {
        auth::secret_from_env().map_err(|e| format!("{:#}", e))
    });
    match cached {
        Ok(s) => Ok(*s),
        Err(e) => anyhow::bail!("{}", e),
    }
}

/// Telemetry received from a node agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTelemetry {
    pub hostname: String,
    pub cpu_model: String,
    pub cores: u32,
    pub threads: u32,
    #[serde(default)]
    pub has_avx: bool,
    pub has_avx2: bool,
    #[serde(default)]
    pub has_sse42: bool,
    pub ram_total_mib: u64,
    pub ram_used_mib: u64,
    pub power_watts: u32,
    pub temp_c: u32,
    pub load_avg: f64,
    #[serde(default)]
    pub gpus: Vec<GpuMini>,
    #[serde(default)]
    pub agent_version: String,
    #[serde(default)]
    pub image_rev: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct GpuMini {
    pub model: String,
    pub vram_mib: u64,
    #[serde(default)]
    pub driver: String,
}

/// Task sent to a node agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub name: String,
    pub payload: String,
    pub estimated_watts: u32,
    pub estimated_seconds: u32,
}

/// Result received from a node agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskResult {
    pub task_id: String,
    pub status: String,
    pub output: String,
    pub elapsed_ms: u64,
    pub peak_watts: u32,
}

/// Send one authenticated message and return the verified response body.
///
/// Wire: `<seq> <hex-tag> <body>` both ways. The response must carry a
/// valid tag over the request's seq — mismatch, tag failure, or unsigned
/// reply is an error.
fn send_raw_with(secret: &Secret, addr: &str, msg: &str, timeout: Duration) -> Result<String> {
    let seq = REQUEST_SEQ.fetch_add(1, Ordering::Relaxed);
    let stream = TcpStream::connect(addr)
        .with_context(|| format!("connect to {}", addr))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let mut writer = stream.try_clone()?;
    writer.write_all(auth::sign_line(secret, seq, msg).as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader
        .read_line(&mut response)
        .with_context(|| format!("read from {}", addr))?;

    let (resp_seq, body) = auth::open_line(secret, response.trim())
        .with_context(|| format!("unauthenticated reply from {}", addr))?;
    if resp_seq != seq {
        anyhow::bail!("reply seq {} != request seq {} from {}", resp_seq, seq, addr);
    }
    Ok(body.to_string())
}

/// Raw authenticated request returning the plain response body — for
/// non-JSON exchanges (e.g. the tagline registration echo).
pub fn raw_with(secret: &Secret, addr: &str, body: &str) -> Result<String> {
    send_raw_with(secret, addr, body, DEFAULT_TIMEOUT)
}

/// Ping an agent to check if it's alive.
pub fn ping(addr: &str) -> Result<bool> {
    ping_with(&cached_secret()?, addr)
}

/// `ping` with an explicit secret.
pub fn ping_with(secret: &Secret, addr: &str) -> Result<bool> {
    let resp = send_raw_with(secret, addr, "ping", DEFAULT_TIMEOUT)?;
    Ok(resp == "pong")
}

/// Request telemetry from an agent.
pub fn telemetry(addr: &str) -> Result<AgentTelemetry> {
    telemetry_with(&cached_secret()?, addr)
}

/// `telemetry` with an explicit secret (tests, multi-cluster tools).
pub fn telemetry_with(secret: &Secret, addr: &str) -> Result<AgentTelemetry> {
    let resp = send_raw_with(secret, addr, "telemetry", DEFAULT_TIMEOUT)?;
    let tel: AgentTelemetry =
        serde_json::from_str(&resp).with_context(|| "parse telemetry response")?;
    Ok(tel)
}

/// Execute a task on an agent (long timeout, for inference).
pub fn execute(addr: &str, task: &AgentTask) -> Result<AgentTaskResult> {
    execute_with(&cached_secret()?, addr, task)
}

/// `execute` with an explicit secret (tests, multi-cluster tools).
pub fn execute_with(secret: &Secret, addr: &str, task: &AgentTask) -> Result<AgentTaskResult> {
    execute_with_timeout(secret, addr, task, Duration::from_secs(120))
}

/// Execute a task on an agent with an explicit timeout.
pub fn execute_timeout(
    addr: &str,
    task: &AgentTask,
    timeout: Duration,
) -> Result<AgentTaskResult> {
    let secret = cached_secret()?;
    execute_with_timeout(&secret, addr, task, timeout)
}

/// `execute_timeout` with an explicit secret (tests, multi-cluster tools).
pub fn execute_with_timeout(
    secret: &Secret,
    addr: &str,
    task: &AgentTask,
    timeout: Duration,
) -> Result<AgentTaskResult> {
    let json = serde_json::to_string(task)?;
    let resp = send_raw_with(secret, addr, &json, timeout)?;
    let result: AgentTaskResult = serde_json::from_str(&resp)
        .with_context(|| format!("parse task result from {} (raw {:?})", addr, &resp[..resp.len().min(80)]))?;
    Ok(result)
}

/// Check if an agent is reachable, returning (addr, alive).
pub fn probe(addr: &str) -> (String, bool) {
    match ping(addr) {
        Ok(true) => (addr.to_string(), true),
        _ => (addr.to_string(), false),
    }
}

/// Fetch a tensor's byte span from an agent over the authenticated frame
/// line (Track C, the live side). Sends a signed
/// `fetch <shard> <tensor> <offset> <length>` line; the agent serves the
/// span back over frames. Returns the bytes (integrity-gated by the
/// size-stamp the agent sends first).
pub fn fetch_span(
    addr: &str,
    shard: &str,
    tensor: &str,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>> {
    fetch_span_with(&cached_secret()?, addr, shard, tensor, offset, length)
}

/// `fetch_span` with an explicit secret (tests, multi-cluster tools).
pub fn fetch_span_with(
    secret: &Secret,
    addr: &str,
    shard: &str,
    tensor: &str,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>> {
    use ouro_cluster::transport::frames::{pump_recv, FrameSession, DEFAULT_WINDOW};

    let seq = REQUEST_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut stream = TcpStream::connect(addr).with_context(|| format!("connect to {}", addr))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let line = auth::sign_line(
        secret,
        seq,
        &format!("fetch {shard} {tensor} {offset} {length}"),
    );
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;

    // Size-stamp: the agent sends the declared tensor length first.
    let mut stamp = [0u8; 8];
    stream.read_exact(&mut stamp).with_context(|| "read size-stamp")?;
    let declared = u64::from_be_bytes(stamp);
    if declared < length {
        anyhow::bail!(
            "size-stamp {declared} < requested {length} — corrupt or wrong tensor"
        );
    }

    // Pump the span back over frames.
    let mut session = FrameSession::new(stream, *secret);
    let mut received = Vec::new();
    let (n, _) = pump_recv(&mut session, &mut received, DEFAULT_WINDOW)?;
    if n != length {
        anyhow::bail!("fetched {n} bytes, wanted {length}");
    }
    Ok(received)
}

/// Fetch a tensor's byte span **across every live lane at once** (the bond
/// payoff, docs/AIR_PATH.md §4.2). The span is split into `edges.len()`
/// contiguous chunks; chunk i rides lane i (bandwidth-descending — the
/// biggest chunk takes the fastest lane), each on its own connection, in
/// parallel; the chunks are reassembled in order. This is "both paths at
/// once" for one tensor: a 4 MB tensor span streams over copper and air
/// simultaneously.
///
/// `edges` is the peer's priced edge set (lane order = bandwidth-descending);
/// `addr_for` maps an `iface` to the address to connect to (the multi-homed
/// address table).
pub fn fetch_span_bonded(
    secret: &Secret,
    edges: &[ouro_cluster::transport::edge::PricedEdge],
    addr_for: &dyn Fn(&str) -> Option<String>,
    shard: &str,
    tensor: &str,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>> {
    if edges.is_empty() {
        anyhow::bail!("bond has no live lanes");
    }
    // Lane order: bandwidth-descending (the bond's bulk stripe order).
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| edges[b].bw_mbps.cmp(&edges[a].bw_mbps));

    // Contiguous chunking: lane i (in bandwidth order) gets chunk i.
    let n = edges.len();
    let base = length / n as u64;
    let rem = (length % n as u64) as usize; // extra bytes to the first `rem` lanes
    let mut chunks: Vec<(u64, u64)> = Vec::with_capacity(n); // (offset, len)
    let mut cur = offset;
    for i in 0..n {
        let len = base + if i < rem { 1 } else { 0 };
        chunks.push((cur, len));
        cur += len;
    }

    // Fan out: one fetch per lane, in parallel, one connection each.
    let mut handles = Vec::with_capacity(n);
    for (i, &idx) in order.iter().enumerate() {
        let (chunk_off, chunk_len) = chunks[i];
        let iface = edges[idx].iface.clone();
        let addr = addr_for(&iface)
            .with_context(|| format!("lane {iface} has no address"))?;
        let secret_c = *secret;
        let shard_c = shard.to_string();
        let tensor_c = tensor.to_string();
        let handle = std::thread::spawn(move || {
            fetch_span_with(&secret_c, &addr, &shard_c, &tensor_c, chunk_off, chunk_len)
        });
        handles.push((i, handle));
    }

    // Reassemble in chunk order (chunks are contiguous, in order).
    let mut out = Vec::with_capacity(length as usize);
    for (_, handle) in handles {
        let chunk = handle
            .join()
            .map_err(|_| anyhow::anyhow!("lane thread panicked"))??;
        out.extend(chunk);
    }
    if out.len() != length as usize {
        anyhow::bail!("reassembled {} bytes, wanted {length}", out.len());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    const KEY: Secret = [7u8; 32];

    /// Fake agent speaking the authed wire: verify in, sign out.
    fn fake_agent_reject(unsigned: bool, key: Secret) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut writer = stream;
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) > 0 {
                    let resp = if unsigned {
                        "pong".to_string()
                    } else if let Ok((seq, _body)) = auth::open_line(&key, line.trim()) {
                        auth::sign_line(&key, seq, "pong")
                    } else {
                        "err auth".to_string()
                    };
                    writer.write_all(resp.as_bytes()).unwrap();
                    writer.write_all(b"\n").unwrap();
                }
            }
        });
        addr
    }

    #[test]
    fn test_send_raw_unreachable() {
        let result = send_raw_with(&KEY, "127.0.0.1:19999", "ping", DEFAULT_TIMEOUT);
        assert!(result.is_err());
    }

    #[test]
    fn test_signed_roundtrip_with_fake_agent() {
        let addr = fake_agent_reject(false, KEY);
        let resp = send_raw_with(&KEY, &addr, "ping", DEFAULT_TIMEOUT).unwrap();
        assert_eq!(resp, "pong");
    }

    #[test]
    fn test_unsigned_reply_rejected() {
        let addr = fake_agent_reject(true, KEY);
        assert!(send_raw_with(&KEY, &addr, "ping", DEFAULT_TIMEOUT).is_err());
    }

    #[test]
    fn test_wrong_key_exchange_rejected() {
        let other: Secret = [9u8; 32];
        let addr = fake_agent_reject(false, other);
        assert!(send_raw_with(&KEY, &addr, "ping", DEFAULT_TIMEOUT).is_err());
    }

    #[test]
    fn test_agent_task_serializes() {
        let task = AgentTask {
            id: "t1".into(),
            name: "echo".into(),
            payload: "hello".into(),
            estimated_watts: 10,
            estimated_seconds: 1,
        };
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("echo"));
        assert!(json.contains("hello"));
    }

    #[test]
    fn test_agent_telemetry_deserializes() {
        let json = r#"{
            "hostname": "test-node",
            "cpu_model": "i5-4590",
            "cores": 4,
            "threads": 4,
            "has_avx2": true,
            "ram_total_mib": 32768,
            "ram_used_mib": 8192,
            "power_watts": 35,
            "temp_c": 45,
            "load_avg": 0.5
        }"#;
        let tel: AgentTelemetry = serde_json::from_str(json).unwrap();
        assert_eq!(tel.hostname, "test-node");
        assert_eq!(tel.ram_total_mib, 32768);
    }

    /// The live fetch, head side: `fetch_span_with` talks to a minimal agent
    /// that serves a BMTS shard span over frames (mirroring the agent's
    /// `handle_fetch`), and the bytes arrive intact on the exact span.
    #[test]
    fn test_fetch_span_head_side() {
        use ouro_cluster::bmts::{write_shard, BmtsShard, BmtsTensor};
        use ouro_cluster::transport::frames::{pump_send, FrameSession, DEFAULT_CHUNK, DEFAULT_WINDOW};
        use std::io::Cursor;

        let dir = std::env::temp_dir().join(format!("ouro-hiss-fetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("draft.bmts");
        let blob: Vec<u8> = (0..8000u32).map(|i| (i % 253) as u8).collect();
        let tensor = "blk.0.attn_q.weight";
        write_shard(
            shard.to_str().unwrap(),
            1,
            &[BmtsTensor {
                name: tensor.into(),
                shape: vec![16, 500],
                dtype: 34,
                offset: 0,
                length: 8000,
            }],
            &blob,
        )
        .unwrap();
        let shard_path = shard.to_str().unwrap().to_string();

        // Minimal agent: read the fetch line, serve the span over frames.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                let mut sock = stream;
                let mut line = String::new();
                let mut one = [0u8; 1];
                while !line.ends_with('\n') {
                    sock.read_exact(&mut one).unwrap();
                    line.push(one[0] as char);
                }
                let (_seq, body) = auth::open_line(&KEY, line.trim()).unwrap();
                let rest = body.strip_prefix("fetch ").unwrap();
                let mut p = rest.split_whitespace();
                let shard = p.next().unwrap().to_string();
                let tensor = p.next().unwrap().to_string();
                let offset: u64 = p.next().unwrap().parse().unwrap();
                let length: u64 = p.next().unwrap().parse().unwrap();
                let bmts = BmtsShard::open(&shard).unwrap();
                let declared = bmts.tensor_bytes(&tensor).unwrap().len() as u64;
                let span = ouro_cluster::op::Span { offset, length };
                let payload = match bmts.read_span(&tensor, span) {
                    Ok(p) => p,
                    Err(_) => continue, // refuse: close, never truncate
                };
                let mut session = FrameSession::new(sock, KEY);
                let buf = declared.to_be_bytes();
                session.get_mut().write_all(&buf).unwrap();
                let mut cur = Cursor::new(payload.bytes().to_vec());
                pump_send(&mut cur, &mut session, DEFAULT_CHUNK, DEFAULT_WINDOW).unwrap();
            }
        });

        // Head: fetch a sub-span.
        let got = fetch_span_with(&KEY, &addr, &shard_path, tensor, 1000, 3000).unwrap();
        assert_eq!(got, &blob[1000..4000]);

        // A span past the declared length is refused, never truncated.
        let err = fetch_span_with(&KEY, &addr, &shard_path, tensor, 7000, 2000);
        assert!(err.is_err(), "overrun span must be refused");

        server.join().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The bond payoff: fetch one tensor span across TWO live lanes at once
    /// (copper + air, two loopback listeners each serving the same shard).
    /// The span is split, striped across both lanes in parallel, and
    /// reassembled intact. Both lanes must actually carry bytes.
    #[test]
    fn test_fetch_span_bonded_two_lanes() {
        use ouro_cluster::bmts::{write_shard, BmtsShard, BmtsTensor};
        use ouro_cluster::transport::edge::{EdgeKind, PricedEdge};
        use ouro_cluster::transport::frames::{pump_send, FrameSession, DEFAULT_CHUNK, DEFAULT_WINDOW};
        use std::io::Cursor;

        let dir = std::env::temp_dir().join(format!("ouro-hiss-bonded-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("draft.bmts");
        let blob: Vec<u8> = (0..64_000u32).map(|i| (i % 251) as u8).collect();
        let tensor = "blk.0.attn_q.weight";
        write_shard(
            shard.to_str().unwrap(),
            1,
            &[BmtsTensor {
                name: tensor.into(),
                shape: vec![16, 4000],
                dtype: 34,
                offset: 0,
                length: 64_000,
            }],
            &blob,
        )
        .unwrap();
        let shard_path = shard.to_str().unwrap().to_string();

        // One minimal-agent listener per lane, each serving one fetch.
        let listener_a = TcpListener::bind("127.0.0.1:0").unwrap();
        let listener_b = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr_a = listener_a.local_addr().unwrap().to_string();
        let addr_b = listener_b.local_addr().unwrap().to_string();

        fn serve_one(listener: TcpListener, key: Secret) -> thread::JoinHandle<u64> {
            thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                let mut sock = stream;
                let mut line = String::new();
                let mut one = [0u8; 1];
                while !line.ends_with('\n') {
                    sock.read_exact(&mut one).unwrap();
                    line.push(one[0] as char);
                }
                let (_seq, body) = auth::open_line(&key, line.trim()).unwrap();
                let rest = body.strip_prefix("fetch ").unwrap();
                let mut p = rest.split_whitespace();
                let shard = p.next().unwrap().to_string();
                let tensor = p.next().unwrap().to_string();
                let offset: u64 = p.next().unwrap().parse().unwrap();
                let length: u64 = p.next().unwrap().parse().unwrap();
                let bmts = BmtsShard::open(&shard).unwrap();
                let declared = bmts.tensor_bytes(&tensor).unwrap().len() as u64;
                let span = ouro_cluster::op::Span { offset, length };
                let payload = bmts.read_span(&tensor, span).unwrap();
                let mut session = FrameSession::new(sock, key);
                let buf = declared.to_be_bytes();
                session.get_mut().write_all(&buf).unwrap();
                let mut cur = Cursor::new(payload.bytes().to_vec());
                pump_send(&mut cur, &mut session, DEFAULT_CHUNK, DEFAULT_WINDOW).unwrap()
            })
        }

        let ha = serve_one(listener_a, KEY);
        let hb = serve_one(listener_b, KEY);

        // Two priced lanes: copper (fast) + air (slow).
        let copper = PricedEdge { iface: "enp3s0".into(), kind: EdgeKind::Tcp, bw_mbps: 1000, latency_us: 300, jitter_us: 10, watts: 0, signal_dbm: 0 };
        let air = PricedEdge { iface: "wlan0".into(), kind: EdgeKind::Air, bw_mbps: 60, latency_us: 2000, jitter_us: 500, watts: 0, signal_dbm: -60 };
        let edges = vec![copper, air];

        let span_len: u64 = 4000;
        let offset: u64 = 1000;
        let got = fetch_span_bonded(
            &KEY,
            &edges,
            &|iface| match iface {
                "enp3s0" => Some(addr_a.clone()),
                "wlan0" => Some(addr_b.clone()),
                _ => None,
            },
            &shard_path,
            tensor,
            offset,
            span_len,
        )
        .unwrap();
        assert_eq!(got, &blob[1000..5000], "bonded reassembly must be intact");

        let (na, nb) = (ha.join().unwrap(), hb.join().unwrap());
        assert_eq!(na + nb, span_len, "both lanes must carry bytes");
        assert!(na > 0 && nb > 0, "striping must use every lane");

        std::fs::remove_dir_all(&dir).ok();
    }
}
