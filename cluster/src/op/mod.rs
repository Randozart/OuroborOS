//! The op kernel — the one grammar over the resource tree (docs/PLAN9.md §4).
//!
//! A fixed, Beast-serializable operation set (`attach|resolve|stat|read|
//! write|ctl`) dispatched over a `GraphBackend`. The kernel is the *mouth*:
//! it routes and returns typed results; the backend (Scheduler, Registry,
//! shell live graph) is the *brain* — ops never bypass scheduling (Art. 11).
//!
//! Bulk payloads never ride ops (Art. 6): `read` of a weight/tensor opens a
//! `Resource::Handle` (identity + size-stamp), not bytes (Rung B3). `bind`/
//! `revoke` are reserved for Phase C (fetch a range of a handle).

use serde::{Deserialize, Serialize};

use crate::beast::resource::ResourcePath;
use crate::beast::topology::NodeEntry;

/// One task-queue entry, as carried by `Resource::Queue`. Beast-serializable
/// so a queue stat rides the wire in Rung B2.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueueEntry {
    pub name: String,
    pub class: String,
    pub age_secs: u64,
    pub retries: u32,
    pub priority: u8,
}

/// One tensor in a node's weight census: name + declared byte length.
/// Metadata only — a handle, not bytes (Art. 6).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorCensus {
    pub name: String,
    pub length: u64,
}

/// A typed value read from the resource tree. Node/record payloads are
/// Beast values (the backend decides their shape); scalar resources are
/// typed variants. Never bytes (anti-table, docs/PLAN9.md §3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Resource {
    /// One node record (display-shaped, live-aware).
    Node(serde_json::Value),
    /// A list of node records.
    Nodes(Vec<serde_json::Value>),
    /// A resolved `node.property` value.
    Prop { node: String, property: String, value: String },
    /// The cluster energy budget, watts.
    Budget { watts: u32 },
    /// The task queue.
    Queue { depth: usize, entries: Vec<QueueEntry> },
    /// Cluster census (live-aware totals + GPU census strings).
    Cluster {
        total: usize,
        online: usize,
        power_watts: u32,
        budget_watts: u32,
        topology_nodes: usize,
        gpus: Vec<String>,
    },
    /// A placement outcome.
    Assign { node: String },
    /// A placement refused/queued.
    Queued { reason: String },
    /// A bulk reference opened by `read` of a weight/tensor path (Rung B3).
    /// Carries identity + size-stamp, never bytes (Art. 6). A client mints
    /// one, then `bind`s/fetches a range of it (Track C) and `revoke`s it.
    Handle {
        id: String,
        node: String,
        tensor: String,
        length: u64,
    },
    /// The weight census for one node: its tensors (name + byte length).
    Tensors { node: String, tensors: Vec<TensorCensus> },
    /// A bare acknowledgement with a display message.
    Ack { message: String },
}

/// The fixed operation set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Op {
    /// Authenticate + bind a client to the tree (Rung B2 wires this to the
    /// signed line; today it is an in-process no-op).
    Attach { node: Option<String> },
    /// Validate that a path resolves (existence check).
    Resolve(ResourcePath),
    /// Read a typed attribute/value at a path.
    Stat(ResourcePath),
    /// Read a value or open a bulk handle at a path. `read weights.<node>.<i>`
    /// opens a `Resource::Handle` (identity + size-stamp, never bytes, Art. 6);
    /// other paths fall through to a typed value.
    Read(ResourcePath),
    /// Set a value / enqueue work at a path (`cluster.budget`, `tasks`).
    Write { path: ResourcePath, value: String },
    /// Verb on a resource (`sleep`, `recover`, `register`, `unregister`).
    Ctl { path: ResourcePath, verb: String },
}

/// A structured op failure: wire-ready, never an exception.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpError {
    pub code: String,
    pub message: String,
}

/// The resource backend: one writer per store (Art. 3). `write`/`ctl` may
/// mutate; `stat`/`resolve` are pure reads.
pub trait GraphBackend {
    /// Existence check: is `path` reachable in this tree?
    fn resolve(&mut self, path: &ResourcePath) -> Result<(), String>;
    /// Read the typed value at `path`.
    fn stat(&mut self, path: &ResourcePath) -> Result<Resource, String>;
    /// Read a value or open a bulk handle at `path` (Rung B3). Defaults to
    /// `stat`; backends override to mint `Resource::Handle`s on weight paths.
    /// Bulk payloads never ride the op — a `read` returns a handle, not bytes
    /// (Art. 6).
    fn read(&mut self, path: &ResourcePath) -> Result<Resource, String> {
        self.stat(path)
    }
    /// Set a value / enqueue work at `path`.
    fn write(&mut self, path: &ResourcePath, value: &str) -> Result<Resource, String>;
    /// Run `verb` against the resource at `path`.
    fn ctl(&mut self, path: &ResourcePath, verb: &str) -> Result<Resource, String>;
}

/// Route one op through the backend. The kernel is the mouth: all ops enter
/// here, all results leave as typed `Resource`s or structured errors.
pub fn dispatch(op: &Op, backend: &mut dyn GraphBackend) -> Result<Resource, OpError> {
    let run = |res: Result<Resource, String>| {
        res.map_err(|message| OpError {
            code: "E_OP".to_string(),
            message,
        })
    };
    match op {
        Op::Attach { .. } => Ok(Resource::Ack {
            message: "attached".to_string(),
        }),
        Op::Resolve(path) => backend
            .resolve(path)
            .map(|()| Resource::Ack {
                message: format!("resolved {}", path),
            })
            .map_err(|message| OpError {
                code: "E_RESOLVE".to_string(),
                message,
            }),
        Op::Stat(path) => run(backend.stat(path)),
        Op::Read(path) => run(backend.read(path)),
        Op::Write { path, value } => run(backend.write(path, value)),
        Op::Ctl { path, verb } => run(backend.ctl(path, verb)),
    }
}

/// Resolve a node property to its display value (static source; no live
/// cache). Mirrors the shell resolvers so both backends agree. Unknown
/// properties resolve to the same "unknown property" value, never an error.
pub fn node_prop(node: &NodeEntry, property: &str) -> String {
    match property {
        "power" | "p" => format!("{}W", node.tdp_watts),
        "ram" | "r" => format!("{}MiB", node.ram_mib),
        "cpu" | "c" => node.cpu_model.clone(),
        "cores" => format!("{}", node.cores),
        "threads" => format!("{}", node.threads),
        "simd" | "s" => {
            let mut parts = Vec::new();
            if node.has_avx2 {
                parts.push("AVX2");
            }
            if node.has_avx {
                parts.push("AVX");
            }
            if node.has_sse42 {
                parts.push("SSE4.2");
            }
            if parts.is_empty() {
                "none".to_string()
            } else {
                parts.join(", ")
            }
        }
        "gpu" => {
            if node.has_gpu {
                format!("{} ({}MiB)", node.gpu_model, node.gpu_vram_mib)
            } else {
                "none".to_string()
            }
        }
        "status" => "IDLE".to_string(),
        "hostname" | "host" => node.hostname.clone(),
        "ip" | "addr" => node.ip.clone(),
        _ => format!("unknown property: {}", property),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted backend for dispatch tests.
    struct MockBackend {
        paths: Vec<String>,
        values: Vec<Resource>,
    }

    impl GraphBackend for MockBackend {
        fn resolve(&mut self, path: &ResourcePath) -> Result<(), String> {
            if self.paths.iter().any(|p| p == &path.to_string()) {
                Ok(())
            } else {
                Err(format!("no such path: {}", path))
            }
        }
        fn stat(&mut self, path: &ResourcePath) -> Result<Resource, String> {
            match self.paths.iter().position(|p| p == &path.to_string()) {
                Some(i) => Ok(self.values.get(i).cloned().unwrap_or(Resource::Ack {
                    message: "empty".to_string(),
                })),
                None => Err(format!("no such path: {}", path)),
            }
        }
        fn write(&mut self, _path: &ResourcePath, value: &str) -> Result<Resource, String> {
            if value == "boom" {
                Err("write rejected".to_string())
            } else {
                Ok(Resource::Ack {
                    message: "written".to_string(),
                })
            }
        }
        fn ctl(&mut self, _path: &ResourcePath, verb: &str) -> Result<Resource, String> {
            if verb == "sleep" {
                Ok(Resource::Ack {
                    message: "sleeping".to_string(),
                })
            } else {
                Err(format!("unknown verb: {}", verb))
            }
        }
    }

    fn mock() -> MockBackend {
        MockBackend {
            paths: vec!["n1".to_string()],
            values: vec![Resource::Prop {
                node: "n1".to_string(),
                property: "power".to_string(),
                value: "35W".to_string(),
            }],
        }
    }

    #[test]
    fn test_attach_is_ack() {
        let mut b = mock();
        let res = dispatch(&Op::Attach { node: None }, &mut b).unwrap();
        assert_eq!(res, Resource::Ack { message: "attached".into() });
    }

    #[test]
    fn test_resolve_ok_and_err() {
        let mut b = mock();
        let ok = dispatch(&Op::Resolve(ResourcePath::parse("n1").unwrap()), &mut b).unwrap();
        assert!(matches!(ok, Resource::Ack { .. }));
        let err = dispatch(&Op::Resolve(ResourcePath::parse("n9").unwrap()), &mut b).unwrap_err();
        assert_eq!(err.code, "E_RESOLVE");
        assert!(err.message.contains("n9"));
    }

    #[test]
    fn test_stat_and_read_agree() {
        let mut b = mock();
        let s = dispatch(&Op::Stat(ResourcePath::parse("n1").unwrap()), &mut b).unwrap();
        let mut b2 = mock();
        let r = dispatch(&Op::Read(ResourcePath::parse("n1").unwrap()), &mut b2).unwrap();
        assert_eq!(s, r);
    }

    #[test]
    fn test_write_and_ctl_error_mapping() {
        let mut b = mock();
        let err = dispatch(
            &Op::Write {
                path: ResourcePath::parse("tasks").unwrap(),
                value: "boom".to_string(),
            },
            &mut b,
        )
        .unwrap_err();
        assert_eq!(err.code, "E_OP");
        let ctl = dispatch(
            &Op::Ctl {
                path: ResourcePath::parse("n1").unwrap(),
                verb: "sleep".to_string(),
            },
            &mut b,
        )
        .unwrap();
        assert!(matches!(ctl, Resource::Ack { .. }));
    }

    #[test]
    fn test_resource_json_round_trip() {
        let res = Resource::Budget { watts: 400 };
        let json = serde_json::to_string(&res).unwrap();
        let back: Resource = serde_json::from_str(&json).unwrap();
        assert_eq!(res, back);
    }

    #[test]
    fn test_op_json_round_trip() {
        let op = Op::Write {
            path: ResourcePath::parse("cluster.budget").unwrap(),
            value: "120".to_string(),
        };
        let json = serde_json::to_string(&op).unwrap();
        let back: Op = serde_json::from_str(&json).unwrap();
        assert_eq!(op, back);
    }

    /// Regression guard for the Beast *text* codec gap: it does not yet
    /// round-trip objects/enums (serialize emits `(key value)` pairs,
    /// deserialize expects arrays — verified 2026-09-08 on ClusterTopology).
    /// Fixing the codec is a Rung B2 (wire unification) item; ops are
    /// serde-canonical (JSON) until then.
    #[test]
    fn test_beast_text_codec_object_gap_documented() {
        let topo = crate::beast::topology::ClusterTopology::new();
        let sexpr = crate::beast::serialize(&topo).unwrap();
        assert!(crate::beast::deserialize::<crate::beast::topology::ClusterTopology>(&sexpr).is_err());
    }

    #[test]
    fn test_node_prop_values() {
        let node = crate::beast::topology::NodeEntry {
            id: "n1".to_string(),
            hostname: "box".to_string(),
            ip: "127.0.0.1".to_string(),
            cpu_model: "i7-3770".to_string(),
            cores: 4,
            threads: 8,
            has_avx: true,
            has_avx2: true,
            has_sse42: true,
            ram_mib: 16384,
            tdp_watts: 77,
            has_gpu: true,
            gpu_model: "RTX 3060".to_string(),
            gpu_vram_mib: 12288,
            gpu_driver: String::new(),
            agent_version: String::new(),
            image_rev: String::new(),
            has_rdma: false,
            rdma_gid: String::new(),
            edges: Vec::new(),
            node_id: String::new(),
        };
        assert_eq!(node_prop(&node, "power"), "77W");
        assert_eq!(node_prop(&node, "simd"), "AVX2, AVX, SSE4.2");
        assert_eq!(node_prop(&node, "gpu"), "RTX 3060 (12288MiB)");
        assert_eq!(node_prop(&node, "bogus"), "unknown property: bogus");
    }
}