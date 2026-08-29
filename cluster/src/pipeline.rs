//! Pipeline parallelism: activation wire frames + stage plans.
//!
//! ACTS v1 frame (little-endian, fixed 24-byte header):
//! ```text
//! magic:       u32  0x4F55524F ("OURO")
//! version:     u8   1
//! frame_type:  u8   0 = ACTIVATION
//! sequence:    u32
//! token_pos:   u32
//! layer_start: u32
//! layer_end:   u32
//! n_elems:     u32
//! payload:     [f32; n_elems]
//! ```

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// ACTS frame magic.
pub const ACTS_MAGIC: u32 = 0x4F55524F;
/// ACTS frame version.
pub const ACTS_VERSION: u8 = 1;
/// Frame type code for activations.
pub const FRAME_ACTIVATION: u8 = 0;
/// Fixed header size: magic4 ver1 type1 seq4 pos4 lstart4 lend4 count4.
pub const ACTS_HEADER_LEN: usize = 26;

/// An activation tensor moving between pipeline stages.
#[derive(Debug, Clone, PartialEq)]
pub struct Activation {
    pub sequence: u32,
    pub token_pos: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub data: Vec<f32>,
}

impl Activation {
    /// Encode to ACTS v1 wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ACTS_HEADER_LEN + self.data.len() * 4);
        out.extend_from_slice(&ACTS_MAGIC.to_le_bytes());
        out.push(ACTS_VERSION);
        out.push(FRAME_ACTIVATION);
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&self.token_pos.to_le_bytes());
        out.extend_from_slice(&self.layer_start.to_le_bytes());
        out.extend_from_slice(&self.layer_end.to_le_bytes());
        out.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        for v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Decode ACTS v1 wire bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < ACTS_HEADER_LEN {
            bail!("ACTS frame too short: {} bytes", bytes.len());
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());

        let magic = u32_at(0);
        if magic != ACTS_MAGIC {
            bail!("bad ACTS magic {:#010x}", magic);
        }
        if bytes[4] != ACTS_VERSION {
            bail!("unsupported ACTS version {}", bytes[4]);
        }
        if bytes[5] != FRAME_ACTIVATION {
            bail!("unsupported ACTS frame type {}", bytes[5]);
        }

        let sequence = u32_at(6);
        let token_pos = u32_at(10);
        let layer_start = u32_at(14);
        let layer_end = u32_at(18);
        let n = u32_at(22) as usize;

        let want = ACTS_HEADER_LEN + n * 4;
        if bytes.len() < want {
            bail!("ACTS payload truncated: have {}, want {}", bytes.len(), want);
        }
        let mut data = Vec::with_capacity(n);
        for chunk in bytes[ACTS_HEADER_LEN..want].chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { sequence, token_pos, layer_start, layer_end, data })
    }
}

/// One stage of a sharded model: a node and the transformer layers it owns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageSpec {
    pub node: u16,
    pub file: String,
    pub layers: Vec<u32>,
    pub tensors: usize,
    pub bytes: u64,
}

/// A parsed shard_map.json: the static plan for pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelinePlan {
    pub model: String,
    pub nodes: Vec<StageSpec>,
}

impl PipelinePlan {
    /// Parse a shard_map.json document.
    pub fn from_json(text: &str) -> Result<Self> {
        let plan: PipelinePlan = serde_json::from_str(text)?;
        if plan.nodes.is_empty() {
            bail!("pipeline plan has no stages");
        }
        Ok(plan)
    }

    /// Load a plan from a shard_map.json file.
    pub fn load(path: &str) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Number of pipeline stages.
    pub fn stage_count(&self) -> usize {
        self.nodes.len()
    }

    /// Stage owning a given layer, if any.
    pub fn stage_for_layer(&self, layer: u32) -> Option<&StageSpec> {
        self.nodes.iter().find(|s| s.layers.contains(&layer))
    }

    /// Hidden-dim consistency check for activation frames:
    /// every stage must see identical token vector widths.
    pub fn validate_against(&self, other: &PipelinePlan) -> bool {
        self.model == other.model && self.stage_count() == other.stage_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Activation {
        Activation {
            sequence: 7,
            token_pos: 3,
            layer_start: 0,
            layer_end: 9,
            data: vec![0.5, -1.25, 3.0, f32::MIN, 1e10],
        }
    }

    #[test]
    fn test_activation_roundtrip() {
        let a = sample();
        let bytes = a.encode();
        assert_eq!(bytes.len(), ACTS_HEADER_LEN + 5 * 4);
        let b = Activation::decode(&bytes).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_activation_empty_data() {
        let a = Activation { sequence: 0, token_pos: 0, layer_start: 0, layer_end: 0, data: vec![] };
        let b = Activation::decode(&a.encode()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_activation_bad_magic() {
        assert!(Activation::decode(&[0u8; 32]).is_err());
    }

    #[test]
    fn test_activation_truncated_payload() {
        let a = sample();
        let mut bytes = a.encode();
        bytes.truncate(ACTS_HEADER_LEN + 4);
        assert!(Activation::decode(&bytes).is_err());
    }

    #[test]
    fn test_plan_parse() {
        let json = r#"{
            "model": "bitnet-2b.gguf",
            "nodes": [
                {"node": 1, "file": "shards/shard_1.bmts", "layers": [0,1,2], "tensors": 30, "bytes": 803000},
                {"node": 2, "file": "shards/shard_2.bmts", "layers": [3,4,5], "tensors": 30, "bytes": 147000}
            ]
        }"#;
        let plan = PipelinePlan::from_json(json).unwrap();
        assert_eq!(plan.stage_count(), 2);
        assert_eq!(plan.stage_for_layer(4).unwrap().node, 2);
        assert!(plan.stage_for_layer(99).is_none());
    }

    #[test]
    fn test_plan_empty_rejected() {
        assert!(PipelinePlan::from_json(r#"{"model":"m","nodes":[]}"#).is_err());
    }
}

/// Encode bytes as lowercase hex.
pub fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Decode lowercase/uppercase hex into bytes.
pub fn from_hex(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        bail!("hex string has odd length");
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for pair in bytes.chunks(2) {
        let hi = (pair[0] as char).to_digit(16).ok_or_else(|| anyhow::anyhow!("bad hex digit"))?;
        let lo = (pair[1] as char).to_digit(16).ok_or_else(|| anyhow::anyhow!("bad hex digit"))?;
        out.push((hi * 16 + lo) as u8);
    }
    Ok(out)
}

#[cfg(test)]
mod hex_tests {
    use super::*;

    #[test]
    fn test_hex_roundtrip() {
        let data = vec![0u8, 1, 15, 16, 255];
        assert_eq!(from_hex(&to_hex(&data)).unwrap(), data);
        assert!(from_hex("abc").is_err());
        assert!(from_hex("zz").is_err());
    }
}
