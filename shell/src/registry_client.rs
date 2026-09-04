//! Signed client for the ouro-registry daemon — the shell's read path
//! to the one source of truth (Registry↔HISS unification).
//!
//! One signed line out (`status`), one signed line in (JSON census).
//! Same wire discipline as the task channel: seq + HMAC both ways,
//! reply tag verified before anything is parsed. Any failure — no
//! daemon, no secret, bad tag, bad JSON — returns Err and the shell
//! falls back to its own topology (graceful, never a hang: 750ms
//! connect timeout).

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{Context, Result};
use ouro_cluster::transport::auth::{self, Secret};
use serde::Deserialize;

/// One node as the registry bus sees it — entry facts plus live state.
#[derive(Debug, Clone, Deserialize)]
pub struct RegistryNode {
    pub id: String,
    #[serde(default)]
    pub hostname: String,
    #[serde(default)]
    pub ip: String,
    #[serde(default)]
    pub cpu_model: String,
    #[serde(default)]
    pub cores: u32,
    #[serde(default)]
    pub threads: u32,
    #[serde(default)]
    pub ram_mib: u64,
    #[serde(default)]
    pub tdp_watts: u32,
    #[serde(default)]
    pub has_gpu: bool,
    #[serde(default)]
    pub gpu_model: String,
    #[serde(default)]
    pub gpu_vram_mib: u64,
    #[serde(default)]
    pub has_avx2: bool,
    #[serde(default)]
    pub has_avx: bool,
    #[serde(default)]
    pub has_sse42: bool,
    #[serde(default)]
    pub power_watts: u32,
    #[serde(default)]
    pub temp_c: u32,
    #[serde(default)]
    pub load_avg: f64,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub online: bool,
    #[serde(default)]
    pub last_seen: u64,
}

/// The daemon's answer to `status`.
#[derive(Debug, Clone, Deserialize)]
pub struct RegistryStatus {
    #[serde(default)]
    pub nodes: Vec<RegistryNode>,
    #[serde(default)]
    pub event_count: u64,
}

impl RegistryStatus {
    pub fn parse(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("registry status: bad json")
    }
}

/// Fetch the live census. `addr` is ip:port of the registry daemon.
pub fn fetch(addr: &str, secret: &Secret) -> Result<RegistryStatus> {
    let stream = TcpStream::connect_timeout(
        &addr
            .parse()
            .with_context(|| format!("registry addr {addr:?}"))?,
        Duration::from_millis(750),
    )
    .with_context(|| format!("registry daemon unreachable at {addr}"))?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_millis(1500))).ok();

    let mut stream = stream;
    // Seq 1: the connection closes after one exchange; ordering is
    // trivially safe per connection.
    let line = auth::sign_line(secret, 1, "status");
    stream
        .write_all(line.as_bytes())
        .and_then(|_| stream.write_all(b"\n"))
        .and_then(|_| stream.flush())
        .context("registry: write failed")?;

    let mut reader = BufReader::new(stream);
    let mut reply = String::new();
    reader
        .read_line(&mut reply)
        .context("registry: no reply")?;
    let (_seq, body) = auth::open_line(secret, reply.trim())?;
    RegistryStatus::parse(body)
}

/// The daemon address: config value, overridden by OURO_REGISTRY.
pub fn resolve_addr(configured: &str) -> String {
    std::env::var("OURO_REGISTRY").unwrap_or_else(|_| configured.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> String {
        serde_json::json!({
            "nodes": [{
                "id": "n1",
                "hostname": "pavilion",
                "ip": "192.168.1.114",
                "cpu_model": "Intel(R) Core(TM) i5-7200U",
                "cores": 2, "threads": 4, "ram_mib": 7829,
                "tdp_watts": 35, "has_gpu": true,
                "gpu_model": "NVIDIA GeForce GTX 1060 6GB",
                "gpu_vram_mib": 6144,
                "power_watts": 35, "temp_c": 46, "load_avg": 0.99,
                "status": "Idle", "online": true, "last_seen": 1,
            }],
            "event_count": 112
        })
        .to_string()
    }

    #[test]
    fn test_parse_status_census() {
        let st = RegistryStatus::parse(&sample_json()).unwrap();
        assert_eq!(st.nodes.len(), 1);
        let n = &st.nodes[0];
        assert_eq!(n.id, "n1");
        assert!(n.has_gpu);
        assert_eq!(n.gpu_model, "NVIDIA GeForce GTX 1060 6GB");
        assert!(n.online);
        assert_eq!(st.event_count, 112);
    }

    #[test]
    fn test_parse_tolerates_missing_fields() {
        let st = RegistryStatus::parse(r#"{"nodes":[{ "id": "n9" }]}"#).unwrap();
        assert_eq!(st.nodes[0].id, "n9");
        assert!(!st.nodes[0].has_gpu);
        assert!(!st.nodes[0].online);
        assert_eq!(st.nodes[0].power_watts, 0);
    }

    #[test]
    fn test_parse_rejects_garbage() {
        assert!(RegistryStatus::parse("not json").is_err());
    }

    #[test]
    fn test_resolve_addr_env_override() {
        // no env: config passes through
        std::env::remove_var("OURO_REGISTRY");
        assert_eq!(resolve_addr("127.0.0.1:9501"), "127.0.0.1:9501");
    }
}
