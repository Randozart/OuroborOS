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

    Ok(NodeInfo {
        hostname: hostname.to_string(),
        ip: ip.to_string(),
        cpu,
        memory,
        energy,
        network,
        status: NodeStatus::Idle,
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

    Ok(NodeInfo {
        hostname,
        ip: "127.0.0.1".to_string(),
        cpu,
        memory,
        energy,
        network,
        status: NodeStatus::Idle,
    })
}
