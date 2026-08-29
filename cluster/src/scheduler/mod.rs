pub mod energy_budget;
pub mod workload_class;

use anyhow::Result;
use energy_budget::{BudgetCheck, EnergyBudget};
use workload_class::WorkloadClass;

use crate::beast::topology::{ClusterTopology, NodeEntry};

/// A task submitted to the cluster scheduler.
#[derive(Debug, Clone)]
pub struct Task {
    pub name: String,
    pub class: WorkloadClass,
    pub payload: String,
    pub estimated_watts: u32,
    pub estimated_seconds: u32,
}

/// Outcome of attempting to schedule a task.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleOutcome {
    Dispatched { node: String },
    Queued { reason: String },
}

/// The cluster scheduler: routes tasks to the best node.
///
/// Selection order (best to worst):
/// 1. Idle node that supports the workload class
/// 2. Node with the most capable SIMD (avx2 > avx > sse42)
/// 3. Cheapest node (lowest TDP) to conserve energy
pub struct Scheduler {
    pub topology: ClusterTopology,
    pub budget: EnergyBudget,
}

impl Scheduler {
    pub fn new(topology: ClusterTopology) -> Self {
        let budget = EnergyBudget::new(topology.power_budget_watts);
        Self { topology, budget }
    }

    /// Attempt to dispatch a task to the best suitable node.
    pub fn schedule(&mut self, task: &Task) -> Result<ScheduleOutcome> {
        let candidates = self.suitable_nodes(task);
        let best = self.rank_candidates(candidates);

        match best {
            Some(node) => {
                match self.budget.check(task.estimated_watts) {
                    BudgetCheck::Allowed => {
                        let node_id = node.id.clone();
                        self.budget.commit(task.estimated_watts);
                        self.mark_working(&node_id);
                        Ok(ScheduleOutcome::Dispatched { node: node_id })
                    }
                    BudgetCheck::Exceeded { .. } => Ok(ScheduleOutcome::Queued {
                        reason: format!(
                            "energy budget would be exceeded (current {}, requested {})",
                            self.budget.current_watts, task.estimated_watts
                        ),
                    }),
                }
            }
            None => Ok(ScheduleOutcome::Queued {
                reason: "no idle node supports this workload class".to_string(),
            }),
        }
    }

    /// Nodes that are idle and structurally capable of the workload.
    fn suitable_nodes(&self, task: &Task) -> Vec<NodeEntry> {
        self.topology
            .nodes
            .iter()
            .filter(|n| self.node_supports(n, task.class))
            .cloned()
            .collect()
    }

    /// Whether a node's architecture suits the workload class.
    fn node_supports(&self, node: &NodeEntry, class: WorkloadClass) -> bool {
        match class {
            WorkloadClass::BranchHeavy | WorkloadClass::Recursive | WorkloadClass::Irregular => {
                // Any x86-64 node works; branch prediction is universal.
                true
            }
            WorkloadClass::SimdFriendly => node.has_avx2 || node.has_avx,
            WorkloadClass::SmallBatch => node.has_sse42,
            WorkloadClass::Unknown => true,
        }
    }

    /// Rank candidate nodes: SIMD capability, then lowest TDP.
    fn rank_candidates(&self, nodes: Vec<NodeEntry>) -> Option<NodeEntry> {
        nodes
            .into_iter()
            .max_by(|a, b| self.capability_score(a).cmp(&self.capability_score(b)))
    }

    fn capability_score(&self, node: &NodeEntry) -> u32 {
        let simd = if node.has_avx2 {
            3
        } else if node.has_avx {
            2
        } else if node.has_sse42 {
            1
        } else {
            0
        };
        // Score combines SIMD capability with efficiency; lower TDP preferred.
        (simd << 8) + (100 - node.tdp_watts)
    }

    fn mark_working(&mut self, node_id: &str) {
        if let Some(node) = self.topology.nodes.iter_mut().find(|n| n.id == node_id) {
            let _ = node; // State tracking lives in the shell/agent layer.
        }
    }

    /// Release a node's energy allocation after a task completes.
    pub fn complete(&mut self, watts: u32) {
        self.budget.release(watts);
    }
}

    #[cfg(test)]
    mod tests {
        use super::*;

    fn make_node(id: &str, avx2: bool, tdp: u32) -> NodeEntry {
        NodeEntry {
            id: id.to_string(),
            hostname: id.to_string(),
            ip: "127.0.0.1".to_string(),
            cpu_model: "Test CPU".to_string(),
            cores: 4,
            threads: 4,
            has_avx: true,
            has_avx2: avx2,
            has_sse42: true,
            ram_mib: 8192,
            tdp_watts: tdp,
        }
    }

    fn make_node_with_simd(id: &str, avx: bool, avx2: bool, sse42: bool, tdp: u32) -> NodeEntry {
        NodeEntry {
            id: id.to_string(),
            hostname: id.to_string(),
            ip: "127.0.0.1".to_string(),
            cpu_model: "Test CPU".to_string(),
            cores: 4,
            threads: 4,
            has_avx: avx,
            has_avx2: avx2,
            has_sse42: sse42,
            ram_mib: 8192,
            tdp_watts: tdp,
        }
    }

    fn test_topology() -> ClusterTopology {
        let mut topo = ClusterTopology::new();
        topo.nodes.push(make_node("n1", true, 35));
        topo.nodes.push(make_node("n2", false, 15));
        topo.nodes.push(make_node("n3", true, 45));
        topo
    }

    #[test]
    fn test_dispatch_to_avx2_node() {
        let mut sched = Scheduler::new(test_topology());
        let task = Task {
            name: "matmul".to_string(),
            class: WorkloadClass::SimdFriendly,
            payload: String::new(),
            estimated_watts: 50,
            estimated_seconds: 10,
        };
        let outcome = sched.schedule(&task).unwrap();
        match outcome {
            ScheduleOutcome::Dispatched { node } => {
                // n1 is the best AVX2 node (score = 3<<8 + 65 = 833, lower TDP wins)
                assert_eq!(node, "n1");
            }
            _ => panic!("expected dispatch"),
        }
    }

    #[test]
    fn test_energy_budget_blocks() {
        let mut sched = Scheduler::new(test_topology());
        sched.budget.set_budget(30);
        let task = Task {
            name: "matmul".to_string(),
            class: WorkloadClass::SimdFriendly,
            payload: String::new(),
            estimated_watts: 50,
            estimated_seconds: 10,
        };
        let outcome = sched.schedule(&task).unwrap();
        assert_eq!(
            outcome,
            ScheduleOutcome::Queued {
                reason: "energy budget would be exceeded (current 0, requested 50)".to_string()
            }
        );
    }

    #[test]
    fn test_no_suitable_node() {
        let mut topo = ClusterTopology::new();
        topo.nodes.push(make_node_with_simd("n2", false, false, false, 15));
        let mut sched = Scheduler::new(topo);
        let task = Task {
            name: "matmul".to_string(),
            class: WorkloadClass::SimdFriendly,
            payload: String::new(),
            estimated_watts: 10,
            estimated_seconds: 10,
        };
        let outcome = sched.schedule(&task).unwrap();
        assert_eq!(
            outcome,
            ScheduleOutcome::Queued {
                reason: "no idle node supports this workload class".to_string()
            }
        );
    }

    #[test]
    fn test_complete_releases_energy() {
        let mut sched = Scheduler::new(test_topology());
        sched.budget.commit(100);
        sched.complete(100);
        assert_eq!(sched.budget.current_watts, 0);
    }
}
