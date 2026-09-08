//! The weight manifest store — metadata only, never bytes.
//!
//! A `WeightShard` names one node's tensors (name + byte length). It is the
//! brain-side census that lets the op kernel mint `Resource::Handle`s (Rung
//! B3) and answer `stat weights.<node>`. It carries no payload: the bytes
//! live in the per-node `.bmts` shards (Phase D / Track C), reached later via
//! a handle, not through this store (Art. 6).
//!
//! A manifest is small (names + lengths), so it may be kept in the brain and
//! loaded from BMTS headers (cheap: the 16-byte header + JSON table, mmap'd)
//! or from a shard_map file.

use serde::{Deserialize, Serialize};

/// One tensor in a node's weight shard: identity + size-stamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightTensor {
    pub name: String,
    pub length: u64,
}

/// The brain's census of one node's weight shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WeightShard {
    pub node: String,
    pub tensors: Vec<WeightTensor>,
}

impl WeightShard {
    /// Total declared payload bytes for this node's shard.
    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.length).sum()
    }

    /// The tensor at index `i`, if in bounds.
    pub fn at(&self, i: usize) -> Option<&WeightTensor> {
        self.tensors.get(i)
    }
}

/// The brain's weight store: one shard per node, addressable by node id.
#[derive(Debug, Clone, Default)]
pub struct Weights {
    pub shards: Vec<WeightShard>,
}

impl Weights {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_shard(mut self, shard: WeightShard) -> Self {
        self.shards.push(shard);
        self
    }

    /// The shard for a node id, if registered.
    pub fn for_node(&self, node: &str) -> Option<&WeightShard> {
        self.shards.iter().find(|s| s.node == node)
    }

    /// All registered node ids.
    pub fn nodes(&self) -> Vec<String> {
        self.shards.iter().map(|s| s.node.clone()).collect()
    }

    /// Build a manifest from one BMTS shard's parsed header. Only the tensor
    /// table (name + length) is copied; the data section is never touched.
    pub fn from_bmts(shard: &crate::bmts::BmtsShard) -> WeightShard {
        WeightShard {
            node: format!("n{}", shard.node),
            tensors: shard
                .tensors
                .iter()
                .map(|t| WeightTensor {
                    name: t.name.clone(),
                    length: t.length,
                })
                .collect(),
        }
    }

    /// Build the manifest for every stage in a pipeline plan, mapping each
    /// stage's `.bmts` header. Stages whose file is missing are skipped (the
    /// manifest is best-effort metadata; a missing shard degrades to "no
    /// weights for that node", not an error). Returns the store and the list
    /// of files that could not be read (for a loud, non-panic report).
    pub fn from_pipeline_plan(
        plan: &crate::pipeline::PipelinePlan,
    ) -> (Self, Vec<String>) {
        let mut store = Self::new();
        let mut missing = Vec::new();
        for stage in &plan.nodes {
            match crate::bmts::BmtsShard::open(&stage.file) {
                Ok(shard) => {
                    store.shards.push(Self::from_bmts(&shard));
                }
                Err(_) => missing.push(stage.file.clone()),
            }
        }
        (store, missing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_addressing() {
        let w = Weights::new().with_shard(WeightShard {
            node: "n1".to_string(),
            tensors: vec![
                WeightTensor { name: "a".into(), length: 10 },
                WeightTensor { name: "b".into(), length: 20 },
            ],
        });
        assert_eq!(w.for_node("n1").unwrap().total_bytes(), 30);
        assert_eq!(w.for_node("n1").unwrap().at(1).unwrap().name, "b");
        assert!(w.for_node("n9").is_none());
        assert_eq!(w.nodes(), vec!["n1".to_string()]);
    }

    #[test]
    fn test_weights_empty_default() {
        let w = Weights::new();
        assert!(w.shards.is_empty());
        assert!(w.for_node("n1").is_none());
    }
}
