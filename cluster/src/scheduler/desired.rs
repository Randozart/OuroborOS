//! Desired state — the declarative shell's target for the cluster.
//!
//! The shell is declarative: `n1 assign branch_sort`, `n1 sleep`, and
//! `budget 400w` do not one-shot mutate the cluster — they DECLARE the
//! state the cluster should converge toward. This store is the declaration.
//! A reconciliation pass (docs/PLAN9.md §6) diffs it against the live
//! cluster and executes the diff over the wire: dispatch workloads, suspend
//! nodes, enforce the budget. Convergence is the cluster adapting to the
//! declaration (Art. 3, Art. 4).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// What a node should be doing, per the declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DesiredNodeState {
    /// The node should be reachable and executing.
    Awake,
    /// The node should be suspended (systemctl suspend).
    Sleeping,
}

/// A declared workload: a task that should be running on a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredWorkload {
    pub node: String,
    /// The task name the agent executes (echo, load_shard, bitnet_generate,
    /// ...) — declared by name, executed by the tail.
    pub task: String,
    pub payload: String,
}

/// The declared state of the cluster. Shared (Arc<Mutex>) between the shell
/// (which mutates it) and the reconciliation loop (which converges toward it).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DesiredState {
    /// The cluster shall draw at most this many watts (None = unconstrained).
    pub budget_watts: Option<u32>,
    /// Node power-state declarations: node id -> desired state.
    pub node_states: HashMap<String, DesiredNodeState>,
    /// Declared workloads: node -> task that should be running.
    pub workloads: Vec<DeclaredWorkload>,
    /// Reconcile idempotency: declarations already acted upon this session
    /// ("n1:echo", "n1:sleep", "n1:wake"). A declaration is executed once;
    /// re-declaring (removing + re-adding) dispatches again.
    #[serde(default)]
    pub dispatched: Vec<String>,
}

impl DesiredState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a node's power state.
    pub fn declare_node(&mut self, node: &str, state: DesiredNodeState) {
        self.node_states.insert(node.to_string(), state);
    }

    /// Declare that a node should run a task.
    pub fn declare_workload(&mut self, node: &str, task: &str, payload: &str) {
        self.workloads.push(DeclaredWorkload {
            node: node.to_string(),
            task: task.to_string(),
            payload: payload.to_string(),
        });
    }

    /// Declare the cluster power budget (None lifts the constraint).
    pub fn declare_budget(&mut self, watts: Option<u32>) {
        self.budget_watts = watts;
    }

    /// The declared state of a node, defaulting to awake.
    pub fn node_desired(&self, node: &str) -> DesiredNodeState {
        self.node_states
            .get(node)
            .copied()
            .unwrap_or(DesiredNodeState::Awake)
    }

    /// Whether any declaration exists (reconcile is a no-op on empty desired).
    pub fn is_empty(&self) -> bool {
        self.budget_watts.is_none() && self.node_states.is_empty() && self.workloads.is_empty()
    }

    /// Mark a declaration as acted upon (idempotency guard).
    pub fn mark_dispatched(&mut self, key: &str) {
        if !self.dispatched.iter().any(|k| k == key) {
            self.dispatched.push(key.to_string());
        }
    }

    pub fn already_dispatched(&self, key: &str) -> bool {
        self.dispatched.iter().any(|k| k == key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_declare_and_read() {
        let mut d = DesiredState::new();
        assert!(d.is_empty());
        d.declare_node("n1", DesiredNodeState::Sleeping);
        d.declare_workload("n2", "echo", "hi");
        d.declare_budget(Some(400));
        assert_eq!(d.node_desired("n1"), DesiredNodeState::Sleeping);
        assert_eq!(d.node_desired("n2"), DesiredNodeState::Awake);
        assert_eq!(d.budget_watts, Some(400));
        assert_eq!(d.workloads.len(), 1);
        assert!(!d.is_empty());
    }

    #[test]
    fn test_roundtrip_json() {
        let mut d = DesiredState::new();
        d.declare_node("n3", DesiredNodeState::Awake);
        d.declare_workload("n1", "bitnet_generate", "hello|64|0.8");
        d.declare_budget(Some(120));
        let json = serde_json::to_string(&d).unwrap();
        let back: DesiredState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.node_desired("n3"), DesiredNodeState::Awake);
        assert_eq!(back.workloads[0].task, "bitnet_generate");
        assert_eq!(back.budget_watts, Some(120));
    }
}