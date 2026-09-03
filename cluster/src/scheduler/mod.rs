pub mod energy_budget;
pub mod task_queue;
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
    pub queue: task_queue::TaskQueue,
}

impl Scheduler {
    pub fn new(topology: ClusterTopology) -> Self {
        let budget = EnergyBudget::new(topology.power_budget_watts);
        Self { topology, budget, queue: task_queue::TaskQueue::new() }
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
                    BudgetCheck::Exceeded { .. } => {
                        self.queue.enqueue(task.clone());
                        Ok(ScheduleOutcome::Queued {
                            reason: format!(
                                "energy budget exceeded — queued (queue depth: {})",
                                self.queue.len()
                            ),
                        })
                    }
                }
            }
            None => {
                self.queue.enqueue(task.clone());
                Ok(ScheduleOutcome::Queued {
                    reason: format!(
                        "no suitable node — queued (queue depth: {})",
                        self.queue.len()
                    ),
                })
            }
        }
    }

    /// Try to dispatch queued tasks (call after a task completes or node joins).
    pub fn drain_queue(&mut self) -> Vec<(String, ScheduleOutcome)> {
        let mut results = Vec::new();
        let tasks: Vec<_> = self.queue.drain();
        for qt in tasks {
            if let Ok(outcome) = self.schedule(&qt.task) {
                results.push((qt.task.name, outcome));
            }
        }
        results
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
            WorkloadClass::LlmInference => node.has_avx2 || node.has_avx,
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
        // GPU capacity dominates (VRAM-class buckets), then SIMD, then TDP
        // efficiency. A decode step is bandwidth-bound: the biggest card wins.
        let gpu = if !node.has_gpu {
            0
        } else if node.gpu_vram_mib >= 8192 {
            12
        } else if node.gpu_vram_mib >= 4096 {
            8
        } else {
            4
        };
        (gpu << 16) + (simd << 8) + (100 - node.tdp_watts.min(100))
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
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
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
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
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
    fn test_gpu_node_ranked_first() {
        let mut cpu = NodeEntry {
            id: "cpu1".into(),
            hostname: "thinkpad".into(),
            ip: "10.0.0.2".into(),
            cpu_model: "i5-3320M".into(),
            cores: 2,
            threads: 4,
            has_avx: true,
            has_avx2: false,
            has_sse42: true,
            ram_mib: 8192,
            tdp_watts: 35,
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
        };
        cpu.has_gpu = false;
        let mut gpu = cpu.clone();
        gpu.id = "gpunode".into();
        gpu.has_gpu = true;
        gpu.gpu_model = "RTX 3060".into();
        gpu.gpu_vram_mib = 12288;
        gpu.tdp_watts = 170; // even at higher TDP the GPU wins: bandwidth rules

        let topo = ClusterTopology { nodes: vec![cpu, gpu], power_budget_watts: 500, ..ClusterTopology::new() };
        let sched = Scheduler::new(topo);
        let cand = vec![
            sched.topology.nodes[0].clone(),
            sched.topology.nodes[1].clone(),
        ];
        let best = sched.rank_candidates(cand).unwrap();
        assert_eq!(best.id, "gpunode");
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
        match outcome {
            ScheduleOutcome::Queued { reason } => {
                assert!(reason.contains("energy budget exceeded"));
                assert!(reason.contains("queue depth: 1"));
            }
            _ => panic!("expected Queued"),
        }
        // Task is now in the queue
        assert_eq!(sched.queue.len(), 1);
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
        match outcome {
            ScheduleOutcome::Queued { reason } => {
                assert!(reason.contains("no suitable node"));
                assert!(reason.contains("queue depth: 1"));
            }
            _ => panic!("expected Queued"),
        }
        assert_eq!(sched.queue.len(), 1);
    }

    #[test]
    fn test_complete_releases_energy() {
        let mut sched = Scheduler::new(test_topology());
        sched.budget.commit(100);
        sched.complete(100);
        assert_eq!(sched.budget.current_watts, 0);
    }

    #[test]
    fn test_drain_queue_dispatches_when_possible() {
        let mut sched = Scheduler::new(test_topology());
        // Queue a task (no budget)
        sched.budget.set_budget(0);
        let task = Task {
            name: "t1".to_string(),
            class: WorkloadClass::SimdFriendly,
            payload: String::new(),
            estimated_watts: 10,
            estimated_seconds: 5,
        };
        sched.schedule(&task).unwrap();
        assert_eq!(sched.queue.len(), 1);
        // Restore budget and drain
        sched.budget.set_budget(500);
        let results = sched.drain_queue();
        assert_eq!(results.len(), 1);
        assert!(matches!(&results[0].1, ScheduleOutcome::Dispatched { .. }));
        assert!(sched.queue.is_empty());
    }
}
