use serde::{Deserialize, Serialize};

/// Live status of a node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Idle,
    Working,
    Offline,
    Sleeping,
}

/// A workload assigned to a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadAssignment {
    pub workload: String,
    pub started_at: String,
    pub est_seconds: u32,
}

/// Live state of a single node, updated by telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    pub node_id: String,
    pub status: NodeStatus,
    pub power_watts: u32,
    pub thermal_c: u32,
    pub load_avg: f64,
    pub assignment: Option<WorkloadAssignment>,
}

impl NodeState {
    pub fn idle(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            status: NodeStatus::Idle,
            power_watts: 0,
            thermal_c: 0,
            load_avg: 0.0,
            assignment: None,
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self.status, NodeStatus::Idle)
    }
}
