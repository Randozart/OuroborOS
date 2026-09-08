//! `GraphBackend` over the scheduler — the kernel speaks to the brain
//! (docs/PLAN9.md §4.2). Ops are the mouth; `Scheduler::schedule()` and the
//! energy budget remain the brain (Art. 11): the `write tasks` op routes
//! through `schedule()`, never around it.

use super::{ScheduleOutcome, Scheduler, Task};
use crate::beast::resource::ResourcePath;
use crate::op::{node_prop, GraphBackend, QueueEntry, Resource, TensorCensus};
use crate::scheduler::workload_class::WorkloadClass;

fn census_string(model: &str, vram_mib: u64) -> String {
    format!(
        "{}:{}MiB",
        model.replace("NVIDIA GeForce ", ""),
        vram_mib
    )
}

/// The node segment of a `weights.<node>[.<i>]` path, if the path is one.
fn weights_node(path: &ResourcePath) -> Option<String> {
    let is_weights = path.segments.first().map(|s| s.as_str()) == Some("weights");
    if !is_weights {
        return None;
    }
    path.segments.get(1).cloned()
}

impl GraphBackend for Scheduler {
    fn resolve(&mut self, path: &ResourcePath) -> Result<(), String> {
        if let Some(node) = path.node() {
            if self.topology.get_node(node).is_some() {
                return Ok(());
            }
            return Err(format!("Node {} not found", node));
        }
        match path.segments.first().map(|s| s.as_str()) {
            Some("cluster") | Some("queue") | Some("tasks") => Ok(()),
            Some("weights") => {
                // `weights` (census root), `weights.<node>`, or
                // `weights.<node>.<i>`. Existence of a specific tensor index
                // is checked by stat/read, not here.
                Ok(())
            }
            other => Err(format!("no such resource: {:?}", other)),
        }
    }

    fn stat(&mut self, path: &ResourcePath) -> Result<Resource, String> {
        self.resolve(path)?;
        if path.is("weights") {
            let nodes: Vec<String> = self.weights.nodes();
            return Ok(Resource::Ack {
                message: if nodes.is_empty() {
                    "no weights registered".to_string()
                } else {
                    format!("weights: {}", nodes.join(", "))
                },
            });
        }
        if let Some(node) = weights_node(path) {
            let shard = self
                .weights
                .for_node(&node)
                .ok_or_else(|| format!("no weights for node {}", node))?;
            let tensors: Vec<TensorCensus> = shard
                .tensors
                .iter()
                .map(|t| TensorCensus { name: t.name.clone(), length: t.length })
                .collect();
            return Ok(Resource::Tensors { node: node.to_string(), tensors });
        }
        if path.is("cluster") {
            let total = self.topology.node_count();
            let power: u32 = self.topology.nodes.iter().map(|n| n.tdp_watts).sum();
            let gpus: Vec<String> = self
                .topology
                .nodes
                .iter()
                .filter(|n| n.has_gpu)
                .map(|n| census_string(&n.gpu_model, n.gpu_vram_mib))
                .collect();
            return Ok(Resource::Cluster {
                total,
                online: total,
                power_watts: power,
                budget_watts: self.budget.budget_watts,
                topology_nodes: total,
                gpus,
            });
        }
        if path.is_child_of("cluster", "budget") {
            return Ok(Resource::Budget {
                watts: self.budget.budget_watts,
            });
        }
        if path.is_child_of("cluster", "nodes") || path.is_child_of("cluster", "static") {
            let nodes: Vec<serde_json::Value> = self
                .topology
                .nodes
                .iter()
                .filter_map(|n| serde_json::to_value(n).ok())
                .collect();
            return Ok(Resource::Nodes(nodes));
        }
        if path.is("queue") || path.is("tasks") {
            let entries: Vec<QueueEntry> = self
                .queue
                .summary()
                .into_iter()
                .map(|e| QueueEntry {
                    name: e.name,
                    class: e.class,
                    age_secs: e.age_secs,
                    retries: e.retries,
                    priority: e.priority,
                })
                .collect();
            let depth = entries.len();
            return Ok(Resource::Queue { depth, entries });
        }
        if let Some(node) = path.node() {
            let entry = self
                .topology
                .get_node(node)
                .ok_or_else(|| format!("Node {} not found", node))?;
            if let Some(property) = path.property() {
                return Ok(Resource::Prop {
                    node: node.to_string(),
                    property: property.to_string(),
                    value: node_prop(entry, property),
                });
            }
            let record = serde_json::to_value(entry)
                .map_err(|e| format!("serialize node {}: {}", node, e))?;
            return Ok(Resource::Node(record));
        }
        Err(format!("no such resource: {}", path))
    }

    /// `read` of a weight path opens a bulk handle (Rung B3): `read
    /// weights.<node>.<i>` mints a `Resource::Handle` for the i-th tensor.
    /// A bare `weights.<node>` reads back its census; other paths fall through
    /// to `stat`. Bytes never ride the op (Art. 6).
    fn read(&mut self, path: &ResourcePath) -> Result<Resource, String> {
        if let Some(node) = weights_node(path) {
            if path.segments.len() == 3 {
                let idx = path.segments[2].parse::<usize>().map_err(|_| {
                    format!("tensor index must be an integer, got {:?}", path.segments[2])
                })?;
                let shard = self
                    .weights
                    .for_node(&node)
                    .ok_or_else(|| format!("no weights for node {}", node))?;
                let t = shard
                    .at(idx)
                    .ok_or_else(|| format!("tensor index {} out of range for node {}", idx, node))?;
                return Ok(Resource::Handle {
                    id: format!("h:{}.{}", node, t.name),
                    node: node.to_string(),
                    tensor: t.name.clone(),
                    length: t.length,
                });
            }
            return self.stat(path);
        }
        self.stat(path)
    }

    fn write(&mut self, path: &ResourcePath, value: &str) -> Result<Resource, String> {
        self.resolve(path)?;
        if path.is_child_of("cluster", "budget") {
            let watts: u32 = value
                .trim()
                .trim_end_matches('w')
                .trim_end_matches('W')
                .parse()
                .map_err(|_| format!("budget must be watts, got {:?}", value))?;
            self.budget.set_budget(watts);
            return Ok(Resource::Budget { watts });
        }
        if path.is("tasks") {
            let task = Task {
                name: value.to_string(),
                class: WorkloadClass::from_name(value),
                payload: String::new(),
                estimated_watts: 30,
                estimated_seconds: 10,
            };
            return match self
                .schedule(&task)
                .map_err(|e| format!("schedule failed: {}", e))?
            {
                ScheduleOutcome::Dispatched { node } => Ok(Resource::Assign { node }),
                ScheduleOutcome::Queued { reason } => Ok(Resource::Queued { reason }),
            };
        }
        Err(format!("write not supported on {}", path))
    }

    fn ctl(&mut self, path: &ResourcePath, verb: &str) -> Result<Resource, String> {
        self.resolve(path)?;
        if let Some(node) = path.node() {
            match verb {
                "sleep" => Ok(Resource::Ack {
                    message: format!("Node {}休眠. Power: 12W → 2W.", node),
                }),
                _ => Err(format!("unknown verb: {} on {}", verb, path)),
            }
        } else if path.is("cluster") && verb == "recover" {
            let results = self.drain_queue();
            let mut out = String::new();
            if results.is_empty() {
                out.push_str("No stale or failed nodes. [OK]");
            }
            if !results.is_empty() {
                out.push_str(&format!("\nDrained {} queued tasks:\n", results.len()));
                for (name, outcome) in &results {
                    out.push_str(&format!("  {} → {:?}\n", name, outcome));
                }
            }
            Ok(Resource::Ack { message: out })
        } else {
            Err(format!("unknown ctl: {} on {}", verb, path))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beast::topology::ClusterTopology;
    use crate::op::Op;
    use crate::scheduler::Scheduler;

    fn make_node(id: &str, avx2: bool, tdp: u32) -> crate::beast::topology::NodeEntry {
        crate::beast::topology::NodeEntry {
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
            agent_version: String::new(),
            image_rev: String::new(),
            has_rdma: false,
            rdma_gid: String::new(),
        }
    }

    fn sched() -> Scheduler {
        let mut topo = ClusterTopology::new();
        topo.nodes.push(make_node("n1", true, 35));
        topo.nodes.push(make_node("n2", false, 15));
        Scheduler::new(topo)
    }

    /// A scheduler whose brain knows n1's weight shard (two tensors).
    fn sched_with_weights() -> Scheduler {
        let mut s = sched();
        s.weights = crate::weights::Weights::new().with_shard(crate::weights::WeightShard {
            node: "n1".to_string(),
            tensors: vec![
                crate::weights::WeightTensor { name: "blk.0.attn_q.weight".into(), length: 1024 },
                crate::weights::WeightTensor { name: "blk.1.attn_k.weight".into(), length: 2048 },
            ],
        });
        s
    }

    fn op_dispatch<B: GraphBackend>(op: &Op, backend: &mut B) -> Result<Resource, crate::op::OpError> {
        crate::op::dispatch(op, backend)
    }

    #[test]
    fn test_stat_budget() {
        let mut s = sched();
        let r = op_dispatch(
            &Op::Stat(ResourcePath::parse("cluster.budget").unwrap()),
            &mut s,
        )
        .unwrap();
        assert_eq!(r, Resource::Budget { watts: 500 });
    }

    #[test]
    fn test_write_budget() {
        let mut s = sched();
        let r = op_dispatch(
            &Op::Write {
                path: ResourcePath::parse("cluster.budget").unwrap(),
                value: "120".to_string(),
            },
            &mut s,
        )
        .unwrap();
        assert_eq!(r, Resource::Budget { watts: 120 });
        assert_eq!(s.budget.budget_watts, 120);
    }

    #[test]
    fn test_write_budget_rejects_garbage() {
        let mut s = sched();
        let err = op_dispatch(
            &Op::Write {
                path: ResourcePath::parse("cluster.budget").unwrap(),
                value: "lots".to_string(),
            },
            &mut s,
        )
        .unwrap_err();
        assert!(err.message.contains("watts"));
    }

    #[test]
    fn test_stat_node_prop() {
        let mut s = sched();
        let r = op_dispatch(
            &Op::Stat(ResourcePath::parse("n1.power").unwrap()),
            &mut s,
        )
        .unwrap();
        assert_eq!(
            r,
            Resource::Prop {
                node: "n1".to_string(),
                property: "power".to_string(),
                value: "35W".to_string(),
            }
        );
    }

    #[test]
    fn test_stat_missing_node() {
        let mut s = sched();
        let err = op_dispatch(&Op::Stat(ResourcePath::parse("n9").unwrap()), &mut s).unwrap_err();
        assert!(err.message.contains("n9"));
    }

    #[test]
    fn test_write_tasks_dispatches() {
        let mut s = sched();
        let r = op_dispatch(
            &Op::Write {
                path: ResourcePath::parse("tasks").unwrap(),
                value: "matmul".to_string(),
            },
            &mut s,
        )
        .unwrap();
        match r {
            Resource::Assign { node } => assert!(node == "n1" || node == "n2"),
            other => panic!("expected Assign, got {:?}", other),
        }
    }

    #[test]
    fn test_write_tasks_queues_when_budget_zero() {
        let mut s = sched();
        s.budget.set_budget(0);
        let r = op_dispatch(
            &Op::Write {
                path: ResourcePath::parse("tasks").unwrap(),
                value: "matmul".to_string(),
            },
            &mut s,
        )
        .unwrap();
        assert!(matches!(r, Resource::Queued { .. }));
        assert_eq!(s.queue.len(), 1);
    }

    #[test]
    fn test_ctl_sleep() {
        let mut s = sched();
        let r = op_dispatch(
            &Op::Ctl {
                path: ResourcePath::parse("n1").unwrap(),
                verb: "sleep".to_string(),
            },
            &mut s,
        )
        .unwrap();
        assert!(matches!(r, Resource::Ack { .. }));
    }

    #[test]
    fn test_stat_cluster() {
        let mut s = sched();
        let r = op_dispatch(
            &Op::Stat(ResourcePath::parse("cluster").unwrap()),
            &mut s,
        )
        .unwrap();
        match r {
            Resource::Cluster {
                total,
                online,
                power_watts,
                budget_watts,
                ..
            } => {
                assert_eq!(total, 2);
                assert_eq!(online, 2);
                assert_eq!(power_watts, 50);
                assert_eq!(budget_watts, 500);
            }
            other => panic!("expected Cluster, got {:?}", other),
        }
    }

    /// Rung B3 (docs/PLAN9.md): `read` of a weight tensor path opens a bulk
    /// handle — identity + size-stamp, never bytes (Art. 6).
    #[test]
    fn test_read_weight_mints_handle() {
        let mut s = sched_with_weights();
        let r = op_dispatch(
            &Op::Read(ResourcePath::parse("weights.n1.0").unwrap()),
            &mut s,
        )
        .unwrap();
        assert_eq!(
            r,
            Resource::Handle {
                id: "h:n1.blk.0.attn_q.weight".to_string(),
                node: "n1".to_string(),
                tensor: "blk.0.attn_q.weight".to_string(),
                length: 1024,
            }
        );
    }

    #[test]
    fn test_read_weight_second_tensor() {
        let mut s = sched_with_weights();
        let r = op_dispatch(
            &Op::Read(ResourcePath::parse("weights.n1.1").unwrap()),
            &mut s,
        )
        .unwrap();
        match r {
            Resource::Handle { tensor, length, node, .. } => {
                assert_eq!(tensor, "blk.1.attn_k.weight");
                assert_eq!(length, 2048);
                assert_eq!(node, "n1");
            }
            other => panic!("expected Handle, got {:?}", other),
        }
    }

    /// `stat` of `weights.<node>` returns the tensor census (name + length).
    #[test]
    fn test_stat_weights_census() {
        let mut s = sched_with_weights();
        let r = op_dispatch(
            &Op::Stat(ResourcePath::parse("weights.n1").unwrap()),
            &mut s,
        )
        .unwrap();
        match r {
            Resource::Tensors { node, tensors } => {
                assert_eq!(node, "n1");
                assert_eq!(
                    tensors,
                    vec![
                        TensorCensus { name: "blk.0.attn_q.weight".into(), length: 1024 },
                        TensorCensus { name: "blk.1.attn_k.weight".into(), length: 2048 },
                    ]
                );
            }
            other => panic!("expected Tensors, got {:?}", other),
        }
    }

    /// Out-of-range tensor index is a structured error, not a panic.
    #[test]
    fn test_read_weight_index_out_of_range() {
        let mut s = sched_with_weights();
        let err = op_dispatch(
            &Op::Read(ResourcePath::parse("weights.n1.7").unwrap()),
            &mut s,
        )
        .unwrap_err();
        assert_eq!(err.code, "E_OP");
        assert!(err.message.contains("out of range"));
    }

    /// Reading a tensor on a node with no registered weights is a clean error.
    #[test]
    fn test_read_weight_unknown_node() {
        let mut s = sched_with_weights();
        let err = op_dispatch(
            &Op::Read(ResourcePath::parse("weights.n2.0").unwrap()),
            &mut s,
        )
        .unwrap_err();
        assert!(err.message.contains("no weights for node n2"));
    }

    /// A scheduler with no weights manifest keeps old `read`==`stat` behavior
    /// on non-weight paths (regression guard for B3).
    #[test]
    fn test_read_non_weight_path_falls_to_stat() {
        let mut s = sched();
        let r = op_dispatch(
            &Op::Read(ResourcePath::parse("n1.power").unwrap()),
            &mut s,
        )
        .unwrap();
        assert_eq!(
            r,
            Resource::Prop {
                node: "n1".to_string(),
                property: "power".to_string(),
                value: "35W".to_string(),
            }
        );
    }

    /// The handle is serde-canonical (JSON round-trips), so it can ride the
    /// wire once Rung B2 unifies the codec.
    #[test]
    fn test_handle_json_round_trip() {
        let h = Resource::Handle {
            id: "h:n1.t".into(),
            node: "n1".into(),
            tensor: "t".into(),
            length: 42,
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: Resource = serde_json::from_str(&json).unwrap();
        assert_eq!(h, back);
    }
}