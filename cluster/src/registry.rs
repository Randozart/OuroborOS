use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::beast::topology::NodeEntry;
use crate::beast::{NodeState, NodeStatus};
use crate::probe::NodeInfo;

/// Lifecycle event emitted by the registry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Event {
    NodeJoined { node_id: String, hostname: String },
    NodeLeft { node_id: String, reason: String },
    NodeUpdated { node_id: String, fields: Vec<String> },
    NodeStateChanged { node_id: String, from: String, to: String },
}

/// A single node record: static HW profile + live state + metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRecord {
    pub entry: NodeEntry,
    pub state: NodeState,
    pub registered_at: u64,
    pub last_seen: u64,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl NodeRecord {
    pub fn from_probe(info: &NodeInfo, id: &str) -> Self {
        let now = epoch_secs();
        let entry = NodeEntry {
            id: id.to_string(),
            hostname: info.hostname.clone(),
            ip: info.ip.clone(),
            cpu_model: info.cpu.model.clone(),
            cores: info.cpu.cores,
            threads: info.cpu.threads,
            has_avx: info.cpu.has_avx,
            has_avx2: info.cpu.has_avx2,
            has_sse42: info.cpu.has_sse42,
            ram_mib: info.memory.total_mib,
            tdp_watts: info.energy.current_watts.max(15),
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
        };
        Self {
            entry,
            state: NodeState::idle(id),
            registered_at: now,
            last_seen: now,
            tags: Vec::new(),
        }
    }

    pub fn is_alive(&self, threshold: Duration) -> bool {
        let now = epoch_secs();
        now.saturating_sub(self.last_seen) < threshold.as_secs()
    }
}

/// The cluster node registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    nodes: HashMap<String, NodeRecord>,
    #[serde(default)]
    events: Vec<Event>,
    persist_path: Option<PathBuf>,
    #[serde(default, skip)]
    heartbeat_threshold: Duration,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            events: Vec::new(),
            persist_path: None,
            heartbeat_threshold: Duration::from_secs(30),
        }
    }

    pub fn with_persist(path: impl Into<PathBuf>) -> Self {
        Self {
            persist_path: Some(path.into()),
            ..Self::new()
        }
    }

    pub fn with_heartbeat(self, threshold: Duration) -> Self {
        Self {
            heartbeat_threshold: threshold,
            ..self
        }
    }

    /// Load from disk, or start empty.
    pub fn load(path: &Path) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(reg) = serde_json::from_str(&data) {
                return reg;
            }
        }
        Self::with_persist(path.to_path_buf())
    }

    /// Persist to disk if a path is configured.
    pub fn save(&self) -> Result<()> {
        if let Some(path) = &self.persist_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let json = serde_json::to_string_pretty(self)?;
            std::fs::write(path, json)?;
        }
        Ok(())
    }

    /// Register a node from a probe result. Returns (node_id, events).
    pub fn register(&mut self, info: &NodeInfo) -> (String, Vec<Event>) {
        let id = self.next_id();
        let record = NodeRecord::from_probe(info, &id);
        let events = vec![Event::NodeJoined {
            node_id: id.clone(),
            hostname: info.hostname.clone(),
        }];
        self.nodes.insert(id.clone(), record);
        self.events.extend(events.clone());
        let _ = self.save();
        (id, events)
    }

    /// Update a node's live state from telemetry. Returns events.
    pub fn heartbeat(
        &mut self,
        node_id: &str,
        power_watts: u32,
        temp_c: u32,
        load_avg: f64,
        status: NodeStatus,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        if let Some(record) = self.nodes.get_mut(node_id) {
            record.last_seen = epoch_secs();
            let old_status = record.state.status.clone();
            record.state.power_watts = power_watts;
            record.state.thermal_c = temp_c;
            record.state.load_avg = load_avg;
            if old_status != status {
                events.push(Event::NodeStateChanged {
                    node_id: node_id.to_string(),
                    from: format!("{:?}", old_status),
                    to: format!("{:?}", status),
                });
                record.state.status = status;
            }
            events.push(Event::NodeUpdated {
                node_id: node_id.to_string(),
                fields: vec!["power".into(), "temp".into(), "load".into()],
            });
        }
        self.events.extend(events.clone());
        let _ = self.save();
        events
    }

    /// Unregister a node. Returns events.
    pub fn unregister(&mut self, node_id: &str, reason: &str) -> Vec<Event> {
        let mut events = Vec::new();
        if self.nodes.remove(node_id).is_some() {
            events.push(Event::NodeLeft {
                node_id: node_id.to_string(),
                reason: reason.to_string(),
            });
        }
        self.events.extend(events.clone());
        let _ = self.save();
        events
    }

    /// Get a node record by ID.
    pub fn get(&self, node_id: &str) -> Option<&NodeRecord> {
        self.nodes.get(node_id)
    }

    /// Backdate a node's last_seen (recovery drills, tests, offline marking).
    pub fn touch_last_seen(&mut self, node_id: &str, epoch_secs: u64) {
        if let Some(record) = self.nodes.get_mut(node_id) {
            record.last_seen = epoch_secs;
        }
    }

    /// Get all alive nodes (last seen within threshold).
    pub fn alive_nodes(&self) -> Vec<&NodeRecord> {
        self.nodes
            .values()
            .filter(|r| r.is_alive(self.heartbeat_threshold))
            .collect()
    }

    /// Get all offline nodes (last seen beyond threshold).
    pub fn offline_nodes(&self) -> Vec<&NodeRecord> {
        self.nodes
            .values()
            .filter(|r| !r.is_alive(self.heartbeat_threshold))
            .collect()
    }

    /// All node IDs.
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.keys().map(|s| s.as_str()).collect()
    }

    /// Node count.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Recent events (last N).
    pub fn recent_events(&self, n: usize) -> &[Event] {
        let start = self.events.len().saturating_sub(n);
        &self.events[start..]
    }

    /// Convert to ClusterTopology for scheduler/shell integration.
    pub fn to_topology(&self) -> crate::beast::topology::ClusterTopology {
        let mut topo = crate::beast::topology::ClusterTopology::new();
        for record in self.nodes.values() {
            topo.nodes.push(record.entry.clone());
        }
        topo
    }

    fn next_id(&self) -> String {
        let max_num = self.nodes.keys()
            .filter_map(|k| k.strip_prefix('n'))
            .filter_map(|s| s.parse::<u32>().ok())
            .max()
            .unwrap_or(0);
        format!("n{}", max_num + 1)
    }
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_info(hostname: &str, ip: &str) -> NodeInfo {
        NodeInfo {
            hostname: hostname.to_string(),
            ip: ip.to_string(),
            cpu: crate::probe::CpuInfo {
                model: "i5-4590".into(),
                cores: 4,
                threads: 4,
                has_avx: true,
                has_avx2: true,
                has_sse42: true,
                has_bmi1: false,
                has_bmi2: false,
                tdp_watts: 35,
            },
            memory: crate::probe::MemoryInfo {
                total_mib: 16384,
                speed_mhz: None,
                memory_type: None,
            },
            energy: crate::probe::EnergyInfo {
                current_watts: 35,
                rapl_available: true,
                power_limit_watts: Some(35),
            },
            network: None,
            status: crate::probe::NodeStatus::Idle,
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut reg = Registry::new();
        let info = test_info("node-a", "192.168.1.10");
        let (id, events) = reg.register(&info);
        assert_eq!(id, "n1");
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::NodeJoined { node_id, .. } if node_id == "n1"));
        assert_eq!(reg.len(), 1);
        let record = reg.get("n1").unwrap();
        assert_eq!(record.entry.hostname, "node-a");
        assert_eq!(record.entry.ip, "192.168.1.10");
    }

    #[test]
    fn test_heartbeat_updates_state() {
        let mut reg = Registry::new();
        let info = test_info("node-a", "192.168.1.10");
        reg.register(&info);
        let events = reg.heartbeat("n1", 42, 55, 1.5, NodeStatus::Working);
        assert!(events.iter().any(|e| matches!(e, Event::NodeStateChanged { .. })));
        let record = reg.get("n1").unwrap();
        assert_eq!(record.state.power_watts, 42);
        assert_eq!(record.state.thermal_c, 55);
        assert_eq!(record.state.status, NodeStatus::Working);
    }

    #[test]
    fn test_unregister() {
        let mut reg = Registry::new();
        let info = test_info("node-a", "192.168.1.10");
        reg.register(&info);
        let events = reg.unregister("n1", "manual");
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::NodeLeft { reason, .. } if reason == "manual"));
        assert!(reg.is_empty());
    }

    #[test]
    fn test_alive_vs_offline() {
        let mut reg = Registry::new().with_heartbeat(Duration::from_secs(1));
        let info = test_info("node-a", "192.168.1.10");
        reg.register(&info);
        assert_eq!(reg.alive_nodes().len(), 1);
        assert_eq!(reg.offline_nodes().len(), 0);
        // Simulate time passing by manipulating last_seen
        if let Some(record) = reg.nodes.get_mut("n1") {
            record.last_seen = 0;
        }
        assert_eq!(reg.alive_nodes().len(), 0);
        assert_eq!(reg.offline_nodes().len(), 1);
    }

    #[test]
    fn test_to_topology() {
        let mut reg = Registry::new();
        let info = test_info("node-a", "192.168.1.10");
        reg.register(&info);
        let topo = reg.to_topology();
        assert_eq!(topo.nodes.len(), 1);
        assert_eq!(topo.nodes[0].hostname, "node-a");
    }

    #[test]
    fn test_persistence_roundtrip() {
        let path = std::env::temp_dir().join("ouro_registry_test.json");
        {
            let mut reg = Registry::with_persist(&path);
            let info = test_info("node-a", "192.168.1.10");
            reg.register(&info);
        }
        let loaded = Registry::load(&path);
        assert_eq!(loaded.len(), 1);
        let record = loaded.get("n1").unwrap();
        assert_eq!(record.entry.hostname, "node-a");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_next_id_increments() {
        let mut reg = Registry::new();
        let info = test_info("a", "1.1.1.1");
        let (id1, _) = reg.register(&info);
        let info2 = test_info("b", "2.2.2.2");
        let (id2, _) = reg.register(&info2);
        assert_eq!(id1, "n1");
        assert_eq!(id2, "n2");
    }
}
