use std::io::{BufRead, BufReader, Write};
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
}
