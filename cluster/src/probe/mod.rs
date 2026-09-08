pub mod cpu;
pub mod gpu;
pub mod energy;
pub mod memory;
pub mod network;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Complete hardware profile for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub hostname: String,
    pub ip: String,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub energy: EnergyInfo,
    pub network: Option<NetworkInfo>,
    pub status: NodeStatus,
    #[serde(default)]
    pub gpus: Vec<gpu::GpuInfo>,
    #[serde(default)]
    pub agent_version: String,
    #[serde(default)]
    pub image_rev: String,
    #[serde(default)]
    pub has_rdma: bool,
    #[serde(default)]
    pub rdma_gid: String,
    /// Priced lanes on this node (docs/AIR_PATH.md §1.1). Populated by the
    /// local probe; remote SSH probes leave empty until a tail-side agent
    /// reports its own lanes.
    #[serde(default)]
    pub edges: Vec<crate::transport::edge::PricedEdge>,
    /// Stable identity (SMBIOS-derived), the one anchor across many IPs
    /// (docs/AIR_PATH.md §4.4). Populated by the local probe.
    #[serde(default)]
    pub node_id: String,
}

/// Derive the node's stable identity from SMBIOS firmware (the one anchor a
/// tail keeps across IP changes and reflashes; docs/AIR_PATH.md §4.4).
/// Sources, in order: product UUID, board serial, product serial. Falls back
/// to a hostname hash when no readable firmware identity exists (VMs,
/// non-x86, minimal containers). Deterministic per box: the same machine
/// always derives the same id.
pub fn derive_node_id() -> String {
    const CANDIDATES: [&str; 3] = [
        "/sys/class/dmi/id/product_uuid",
        "/sys/class/dmi/id/board_serial",
        "/sys/class/dmi/id/product_serial",
    ];
    for path in CANDIDATES {
        if let Ok(raw) = std::fs::read_to_string(path) {
            let s = raw.trim();
            if !s.is_empty() && !s.to_lowercase().contains("to be filled") {
                return format!("b{}", stable_hash(s));
            }
        }
    }
    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    format!("b{}", stable_hash(&hostname))
}

/// A stable 8-hex hash of a string (std SipHash with default keys is
/// deterministic across runs — fixed keys, no randomness).
fn stable_hash(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:08x}", h.finish() as u32)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_sse42: bool,
    pub has_bmi1: bool,
    pub has_bmi2: bool,
    pub tdp_watts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mib: u64,
    pub speed_mhz: Option<u32>,
    pub memory_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyInfo {
    pub current_watts: u32,
    pub rapl_available: bool,
    pub power_limit_watts: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub latency_ms: f64,
    pub bandwidth_mbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Idle,
    Working,
    Offline,
    Sleeping,
}

/// Probe a remote node via SSH.
pub fn probe_node(hostname: &str, ip: &str) -> Result<NodeInfo> {
    let cpu = cpu::probe_remote(ip)?;
    let memory = memory::probe_remote(ip)?;
    let energy = energy::probe_remote(ip)?;
    let network = network::probe_remote(ip).ok();
    let gpus = gpu::detect_gpus();

    Ok(NodeInfo {
        hostname: hostname.to_string(),
        ip: ip.to_string(),
        cpu,
        memory,
        energy,
        network,
        status: NodeStatus::Idle,
        gpus,
        agent_version: String::new(),
        image_rev: String::new(),
        has_rdma: false,
        rdma_gid: String::new(),
        edges: Vec::new(),
        node_id: String::new(),
    })
}

/// Probe the local node (no SSH needed).
pub fn probe_local() -> Result<NodeInfo> {
    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let cpu = cpu::probe_local()?;
    let memory = memory::probe_local()?;
    let energy = energy::probe_local()?;
    let network = network::probe_agent("127.0.0.1:9500").ok();
    let gpus = gpu::detect_gpus();

    // Check for RDMA devices: look for /sys/class/infiniband/*/node_desc
    // or check if rdma_rxe is loaded.
    let (has_rdma, rdma_gid) = probe_rdma();

    // Enumerate + price every lane (link speed, iw signal, RTT/jitter).
    let edges = network::probe_lanes(None);

    Ok(NodeInfo {
        hostname,
        ip: "127.0.0.1".to_string(),
        cpu,
        memory,
        energy,
        network,
        status: NodeStatus::Idle,
        gpus,
        agent_version: String::new(),
        image_rev: String::new(),
        has_rdma,
        rdma_gid,
        edges,
        node_id: derive_node_id(),
    })
}

/// Detect RDMA devices by checking /sys/class/infiniband.
/// Returns (has_device, first_gid_hex).
fn probe_rdma() -> (bool, String) {
    use std::path::Path;

    let ib_dir = Path::new("/sys/class/infiniband");
    if !ib_dir.exists() {
        return (false, String::new());
    }

    // Find the first RDMA device. (The probe only ever used the first
    // entry — the previous `for` loop always returned on its first pass.)
    let entries = match std::fs::read_dir(ib_dir) {
        Ok(e) => e,
        Err(_) => return (false, String::new()),
    };
    if let Some(entry) = entries.flatten().next() {
        // Check if the device is a SoftRoCE (rxe) or real IB device.
        let dev_type_path = entry.path().join("node_type");
        if let Ok(node_type) = std::fs::read_to_string(&dev_type_path) {
            // node_type: 1 = CA (Channel Adapter), 2 = RNIC (RDMA NIC)
            let _ = node_type;
        }

        // Read port 1 GID if available.
        let gid_path = entry.path().join("ports/1/gids/0");
        let gid = std::fs::read_to_string(&gid_path)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        // SoftRoCE GID format: "fe80:0000:0000:0000:<mac>:0000:0000:0001"
        // Real IB GID: "0000:...:0001"
        return (true, gid);
    }

    (false, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable identity: deterministic per box, prefixed, non-empty, and the
    /// same every call (the one anchor across IPs — AIR_PATH §4.4).
    #[test]
    fn test_derive_node_id_is_stable_and_prefixed() {
        let id1 = derive_node_id();
        let id2 = derive_node_id();
        assert!(!id1.is_empty());
        assert!(id1.starts_with('b'));
        assert_eq!(id1, id2, "must be deterministic per box");
        assert_eq!(id1.len(), 9, "b + 8 hex chars");
    }

    #[test]
    fn test_stable_hash() {
        assert_eq!(stable_hash("same"), stable_hash("same"));
        assert_ne!(stable_hash("same"), stable_hash("other"));
        assert!(stable_hash("x").chars().all(|c| c.is_ascii_hexdigit()));
    }
}
