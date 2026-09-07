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

    // Find the first RDMA device.
    let entries = match std::fs::read_dir(ib_dir) {
        Ok(e) => e,
        Err(_) => return (false, String::new()),
    };

    for entry in entries.flatten() {
        let dev_name = entry.file_name();
        let dev_name_str = dev_name.to_string_lossy();

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
        // Real IB GID: "0000:0000:0000:0000:0000:0000:0000:0000:0000:0000:0000:0000:0000:0000:0000:0001"
        return (true, gid);
    }

    (false, String::new())
}
