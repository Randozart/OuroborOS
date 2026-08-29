//! BMTS — BitNet Model Tensor Shard format (v1).
//!
//! Binary shard produced by `tools/shard_model.py`, one per pipeline stage.
//!
//! Layout (little-endian):
//! ```text
//! magic:     u32  0x4F55524F ("OURO")
//! version:   u16  1
//! node:      u16  node index (1-based)
//! n_tensors: u32
//! meta_len:  u32
//! meta:      JSON tensor table
//! data:      concatenated tensor bytes
//! ```

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;

/// BMTS file magic ("OURO" little-endian).
pub const BMTS_MAGIC: u32 = 0x4F55524F;
/// BMTS format version.
pub const BMTS_VERSION: u16 = 1;
/// Size of the fixed BMTS header in bytes.
pub const BMTS_HEADER_LEN: usize = 16;

/// One tensor inside a shard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BmtsTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: u32,
    /// Byte offset within the shard data section.
    pub offset: u64,
    /// Byte length of tensor data.
    pub length: u64,
}

/// Parsed BMTS shard (metadata only; payload read on demand).
#[derive(Debug, Clone)]
pub struct BmtsShard {
    pub node: u16,
    pub tensors: Vec<BmtsTensor>,
    /// Absolute file offset where the data section begins.
    pub data_start: u64,
    path: String,
}

impl BmtsShard {
    /// Parse header + tensor table from a .bmts file.
    pub fn open(path: &str) -> Result<Self> {
        let mut f = File::open(path)?;
        let mut header = [0u8; BMTS_HEADER_LEN];
        f.read_exact(&mut header)?;

        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != BMTS_MAGIC {
            bail!("bad BMTS magic {:#010x} in {}", magic, path);
        }
        let version = u16::from_le_bytes(header[4..6].try_into().unwrap());
        if version != BMTS_VERSION {
            bail!("unsupported BMTS version {} (expected {})", version, BMTS_VERSION);
        }
        let node = u16::from_le_bytes(header[6..8].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        let meta_len = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;

        let mut meta = vec![0u8; meta_len];
        f.read_exact(&mut meta)?;
        let tensors: Vec<BmtsTensor> = serde_json::from_slice(&meta)?;
        if tensors.len() != n_tensors {
            bail!("BMTS tensor count mismatch: header {} vs meta {}", n_tensors, tensors.len());
        }

        let data_start = (BMTS_HEADER_LEN + meta_len) as u64;
        Ok(Self { node, tensors, data_start, path: path.to_string() })
    }

    /// Total declared tensor payload bytes.
    pub fn data_len(&self) -> u64 {
        self.tensors.iter().map(|t| t.length).sum()
    }

    /// Read one tensor's raw bytes by name.
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let t = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("tensor {} not in shard", name))?;

        let mut f = File::open(&self.path)?;
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(self.data_start + t.offset))?;
        let mut buf = vec![0u8; t.length as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Serialize a minimal BMTS shard (used by tools and tests).
pub fn write_shard(path: &str, node: u16, tensors: &[BmtsTensor], data: &[u8]) -> Result<()> {
    let meta = serde_json::to_vec(tensors)?;
    let mut out = Vec::with_capacity(BMTS_HEADER_LEN + meta.len() + data.len());
    out.extend_from_slice(&BMTS_MAGIC.to_le_bytes());
    out.extend_from_slice(&BMTS_VERSION.to_le_bytes());
    out.extend_from_slice(&node.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    out.extend_from_slice(&meta);
    out.extend_from_slice(data);
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmts_roundtrip() {
        let dir = std::env::temp_dir().join("ouro_bmts_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_1.bmts");

        let blob: Vec<u8> = (0..64u8).collect();
        let tensors = vec![
            BmtsTensor {
                name: "blk.0.attn_q.weight".into(),
                shape: vec![8, 8],
                dtype: 34,
                offset: 0,
                length: 32,
            },
            BmtsTensor {
                name: "blk.0.attn_k.weight".into(),
                shape: vec![8, 8],
                dtype: 34,
                offset: 32,
                length: 32,
            },
        ];
        write_shard(path.to_str().unwrap(), 1, &tensors, &blob).unwrap();

        let shard = BmtsShard::open(path.to_str().unwrap()).unwrap();
        assert_eq!(shard.node, 1);
        assert_eq!(shard.tensors.len(), 2);
        assert_eq!(shard.data_len(), 64);

        let t0 = shard.read_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(t0, &blob[0..32]);
        let t1 = shard.read_tensor("blk.0.attn_k.weight").unwrap();
        assert_eq!(t1, &blob[32..64]);
        assert!(shard.read_tensor("missing").is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_bmts_bad_magic() {
        let dir = std::env::temp_dir().join("ouro_bmts_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.bmts");
        std::fs::write(&path, [0u8; 32]).unwrap();
        assert!(BmtsShard::open(path.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod real_shard_tests {
    use super::*;

    #[test]
    #[ignore] // Requires shards from `python3 tools/shard_model.py`
    fn test_read_real_shard() {
        let path = "/tmp/shards_test/shard_2.bmts";
        if !std::path::Path::new(path).exists() {
            eprintln!("no shard, skipping");
            return;
        }
        let shard = BmtsShard::open(path).unwrap();
        assert_eq!(shard.node, 2);
        assert!(shard.tensors.len() > 100);
        let t = &shard.tensors[0];
        let bytes = shard.read_tensor(&t.name).unwrap();
        assert_eq!(bytes.len() as u64, t.length);
        eprintln!("real shard ok: node {} tensors {} first {} bytes {}", shard.node, shard.tensors.len(), t.name, bytes.len());
    }
}
