use anyhow::Result;
use crate::beast;
use crate::probe::NodeInfo;
use serde::{Deserialize, Serialize};

/// A node in the cluster topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEntry {
    pub id: String,
    pub hostname: String,
    pub ip: String,
    pub cpu_model: String,
    pub cores: u32,
    pub threads: u32,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_sse42: bool,
    pub ram_mib: u64,
    pub tdp_watts: u32,
    #[serde(default)]
    pub has_gpu: bool,
    #[serde(default)]
    pub gpu_model: String,
    #[serde(default)]
    pub gpu_vram_mib: u64,
    #[serde(default)]
    pub gpu_driver: String,
}

impl Default for NodeEntry {
    fn default() -> Self {
        Self {
            id: String::new(),
            hostname: String::new(),
            ip: String::new(),
            cpu_model: String::new(),
            cores: 0,
            threads: 0,
            has_avx: false,
            has_avx2: false,
            has_sse42: false,
            ram_mib: 0,
            tdp_watts: 0,
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
        }
    }
}

/// A workload known to the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadEntry {
    pub name: String,
    pub source_file: String,
    pub class: String,
    pub est_time_seconds: u32,
    pub min_ram_mib: u64,
    pub requires_avx2: bool,
}

/// The complete cluster topology as Beast.
///
/// Serialized format:
/// ```text
/// (cluster
///   (nodes
///     (node n1 (hostname "laptop-1") (ip "192.168.1.101")
///       (cpu_model "Haswell i5-4200U") (cores 2) (threads 4)
///       (has_avx true) (has_avx2 true) (has_sse42 true)
///       (ram_mib 8192) (tdp_watts 35)))
///   (workloads
///     (workload branch_sort (source "branch_sort.bv") (class "BRANCH_HEAVY")
///       (est_time_seconds 45) (min_ram_mib 4096) (requires_avx2 true)))
///   (budget 500))
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTopology {
    pub nodes: Vec<NodeEntry>,
    pub workloads: Vec<WorkloadEntry>,
    pub power_budget_watts: u32,
}

impl ClusterTopology {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            workloads: Vec::new(),
            power_budget_watts: 500,
        }
    }

    /// Add a node from a probe result.
    pub fn add_node(&mut self, info: NodeInfo) -> NodeEntry {
        let id = node_id(&info.hostname, &info.ip, self.nodes.len());
        let entry = NodeEntry {
            id,
            hostname: info.hostname,
            ip: info.ip,
            cpu_model: info.cpu.model,
            cores: info.cpu.cores,
            threads: info.cpu.threads,
            has_avx: info.cpu.has_avx,
            has_avx2: info.cpu.has_avx2,
            has_sse42: info.cpu.has_sse42,
            ram_mib: info.memory.total_mib,
            tdp_watts: info.cpu.tdp_watts,
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
        };
        self.nodes.push(entry.clone());
        entry
    }

    pub fn add_workload(&mut self, workload: WorkloadEntry) {
        self.workloads.push(workload);
    }

    pub fn get_node(&self, id: &str) -> Option<&NodeEntry> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut NodeEntry> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn remove_node(&mut self, id: &str) -> Option<NodeEntry> {
        let idx = self.nodes.iter().position(|n| n.id == id)?;
        Some(self.nodes.remove(idx))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Serialize the topology to Beast text.
    pub fn to_beast(&self) -> Result<String> {
        beast::serialize(self)
    }

    /// Deserialize the topology from Beast text.
    pub fn from_beast(sexpr: &str) -> Result<Self> {
        beast::deserialize(sexpr)
    }

    /// Save topology to a JSON file.
    pub fn save_json(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load topology from a JSON file.
    pub fn load_json(path: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let topo: Self = serde_json::from_str(&data)?;
        Ok(topo)
    }
}

impl Default for ClusterTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a stable node ID from hostname/IP/index.
fn node_id(hostname: &str, _ip: &str, index: usize) -> String {
    // Prefer hostname-derived id: laptop-1 -> n1, thinkpad-x230 -> nx230
    let clean: String = hostname
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();

    if clean.len() >= 2 {
        let tail: String = clean.chars().skip(clean.len() - 2).collect();
        if let Ok(num) = tail.parse::<u32>() {
            return format!("n{}", num);
        }
    }
    format!("n{}", index + 1)
}
