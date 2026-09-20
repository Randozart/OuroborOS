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

/// Zero-copy or owned byte range of a shard payload.
#[derive(Clone)]
pub enum Payload {
    Owned(Vec<u8>),
    Mapped(std::sync::Arc<memmap2::Mmap>, usize, usize),
}

impl Payload {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Mapped(m, s, l) => &m[*s..*s + *l],
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(v) => v.len(),
            Self::Mapped(_, _, l) => *l,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for Payload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Payload({})", self.len())
    }
}

/// Parsed BMTS shard: the file is memory-mapped, tensors borrow it.
pub struct BmtsShard {
    pub node: u16,
    pub tensors: Vec<BmtsTensor>,
    /// Absolute file offset where the data section begins.
    pub data_start: u64,
    /// DUET epoch: sha256 hex of the data section ("" when unindexed).
    pub epoch: String,
    map: std::sync::Arc<memmap2::Mmap>,
    path: String,
}

impl std::fmt::Debug for BmtsShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BmtsShard")
            .field("node", &self.node)
            .field("tensors", &self.tensors.len())
            .field("path", &self.path)
            .finish()
    }
}

/// BMTS format version 2: metadata blob is Cap'n Proto (schemas/bmts.capnp)
/// instead of JSON. Data section layout unchanged; v1 files read forever.
pub const BMTS_VERSION_V2: u16 = 2;

/// DUET chunk size for the advisory chunk index (docs/DUET.md P1).
pub const DUET_CHUNK_SIZE: u32 = 64 * 1024;

/// sha256 hex of a byte slice — the shard epoch (DUET contract identity).
pub fn epoch_of(data: &[u8]) -> String {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(data);
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

/// Advisory per-chunk hash (SipHash-13, fixed key). NOT authoritative —
/// the epoch is. A 0 hash marks a short/empty chunk.
pub fn chunk_hash(chunk: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(chunk);
    h.finish()
}

/// Per-chunk hashes of a data section at `DUET_CHUNK_SIZE` granularity.
pub fn chunk_hashes(data: &[u8], chunk_size: u32) -> Vec<u64> {
    data.chunks(chunk_size.max(1) as usize).map(chunk_hash).collect()
}

/// Streaming DUET identity for shards too big to walk as one mmap slice
/// comfortably: one buffered read pass computes the epoch (sha256) and the
/// advisory chunk hashes together. Memory cost: one chunk-size buffer.
pub fn identity_of_stream(
    r: &mut impl std::io::Read,
    chunk_size: u32,
) -> Result<(String, Vec<u64>)> {
    use sha2::Digest;
    let cs = chunk_size.max(1) as usize;
    let mut hasher = sha2::Sha256::new();
    let mut hashes = Vec::new();
    let mut buf = vec![0u8; cs];
    loop {
        let mut done = 0;
        while done < cs {
            let n = r.read(&mut buf[done..])?;
            if n == 0 {
                break;
            }
            done += n;
        }
        if done == 0 {
            break;
        }
        hasher.update(&buf[..done]);
        hashes.push(chunk_hash(&buf[..done]));
        if done < cs {
            break;
        }
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    Ok((hex, hashes))
}

impl BmtsShard {
    /// Memory-map a .bmts file and parse its header + tensor table.
    ///
    /// Version 1 = serde_json meta; version 2 = Cap'n Proto meta
    /// (requires the `capnp2` feature). Tensor data layout is identical.
    ///
    /// # Safety notes
    /// The mapping assumes shard files are immutable once written
    /// (deploy-time writes, never concurrent truncation).
    pub fn open(path: &str) -> Result<Self> {
        let f = File::open(path)?;
        let map = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f) }?);
        let bytes: &[u8] = &map;
        if bytes.len() < BMTS_HEADER_LEN {
            bail!("BMTS file too short: {}", path);
        }
        let header = &bytes[..BMTS_HEADER_LEN];
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != BMTS_MAGIC {
            bail!("bad BMTS magic {:#010x} in {}", magic, path);
        }
        let version = u16::from_le_bytes(header[4..6].try_into().unwrap());
        let node = u16::from_le_bytes(header[6..8].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        let meta_len = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        if bytes.len() < BMTS_HEADER_LEN + meta_len {
            bail!("BMTS meta truncated: {}", path);
        }
        let meta = &bytes[BMTS_HEADER_LEN..BMTS_HEADER_LEN + meta_len];
        let mut epoch = String::new();
        let tensors: Vec<BmtsTensor> = match version {
            BMTS_VERSION => serde_json::from_slice(meta)?,
            BMTS_VERSION_V2 => {
                #[cfg(feature = "capnp2")]
                {
                    let (t, e) = parse_meta_v2(meta)?;
                    epoch = e;
                    t
                }
                #[cfg(not(feature = "capnp2"))]
                {
                    bail!(
                        "BMTS v{} (Cap'n Proto meta) requires the `capnp2` feature",
                        BMTS_VERSION_V2
                    )
                }
            }
            other => bail!("unsupported BMTS version {other} in {path}"),
        };
        if tensors.len() != n_tensors {
            bail!("BMTS tensor count mismatch: header {} vs meta {}", n_tensors, tensors.len());
        }
        let data_start = (BMTS_HEADER_LEN + meta_len) as u64;
        Ok(Self { node, tensors, data_start, epoch, map, path: path.to_string() })
    }

    /// DUET contract check: does the data section hash to the declared epoch?
    /// Shards without an epoch (`""`) vacuously verify — they carry no claim.
    /// Streams from the file (bounded memory — a full-mmap walk thrashes
    /// under memory pressure).
    pub fn verify_epoch(&self) -> bool {
        if self.epoch.is_empty() {
            return true;
        }
        let Ok(mut f) = std::fs::File::open(&self.path) else {
            return false;
        };
        use std::io::Seek;
        if f.seek(std::io::SeekFrom::Start(self.data_start)).is_err() {
            return false;
        }
        matches!(
            identity_of_stream(&mut f, DUET_CHUNK_SIZE),
            Ok((e, _)) if e == self.epoch
        )
    }

    /// The raw data section (everything after the meta blob).
    pub fn data_bytes(&self) -> &[u8] {
        &self.map[self.data_start as usize..]
    }

    /// Advisory chunk hashes of the data section, if the shard carries them.
    /// Streams from the file (bounded memory).
    pub fn chunk_hashes_recomputed(&self) -> Vec<u64> {
        let Ok(mut f) = std::fs::File::open(&self.path) else {
            return Vec::new();
        };
        use std::io::Seek;
        if f.seek(std::io::SeekFrom::Start(self.data_start)).is_err() {
            return Vec::new();
        }
        identity_of_stream(&mut f, DUET_CHUNK_SIZE).map(|(_, h)| h).unwrap_or_default()
    }

    /// Zero-copy byte range of a tensor.
    pub fn tensor_bytes(&self, name: &str) -> Result<Payload> {
        let t = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("tensor {} not in shard", name))?;
        let start = self.data_start as usize + t.offset as usize;
        let end = start + t.length as usize;
        if end > self.map.len() {
            bail!("tensor {} range {}..{} exceeds file {}", name, start, end, self.map.len());
        }
        Ok(Payload::Mapped(self.map.clone(), start, t.length as usize))
    }

    /// Total declared tensor payload bytes.
    pub fn data_len(&self) -> u64 {
        self.tensors.iter().map(|t| t.length).sum()
    }

    /// Read one tensor's raw bytes by name (copy).
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.tensor_bytes(name)?.bytes().to_vec())
    }

    /// Zero-copy byte range of a tensor — the streamed range-fetch unit
    /// (SwarmLLM borrow, AIR_PATH §2.2): fetch *only* a tensor's byte span.
    /// Bounds-checked against the tensor's declared length (the size-stamp);
    /// a span that exceeds it is refused, never truncated.
    pub fn read_span(&self, tensor: &str, span: crate::op::Span) -> Result<Payload> {
        let t = self
            .tensors
            .iter()
            .find(|t| t.name == tensor)
            .ok_or_else(|| anyhow::anyhow!("tensor {} not in shard", tensor))?;
        if span.end() > t.length {
            bail!(
                "span [{}, {}) exceeds tensor {} of {} bytes (size-stamp {})",
                span.offset,
                span.end(),
                tensor,
                t.length,
                t.length
            );
        }
        let start = self.data_start as usize + t.offset as usize + span.offset as usize;
        let end = start + span.length as usize;
        if end > self.map.len() {
            bail!("span range {}..{} exceeds file", start, end);
        }
        Ok(Payload::Mapped(self.map.clone(), start, span.length as usize))
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

/// Parse a v2 (Cap'n Proto) metadata blob into (tensor table, epoch).
#[cfg(feature = "capnp2")]
fn parse_meta_v2(meta: &[u8]) -> Result<(Vec<BmtsTensor>, String)> {
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut &meta[..],
        capnp::message::ReaderOptions::new(),
    )?;
    let header = reader.get_root::<crate::bmts_capnp::shard_header::Reader>()?;
    let list = header.get_tensors()?;
    let mut out = Vec::with_capacity(list.len() as usize);
    for t in list.iter() {
        out.push(BmtsTensor {
            name: t.get_name()?.to_str()?.to_string(),
            shape: t.get_shape()?.iter().collect(),
            dtype: t.get_dtype(),
            offset: t.get_offset(),
            length: t.get_length(),
        });
    }
    let epoch = header.get_epoch()?.to_str()?.to_string();
    Ok((out, epoch))
}

/// Build the v2 Cap'n Proto metadata blob (ShardHeader) for a shard.
#[cfg(feature = "capnp2")]
pub fn build_meta_v2(
    node: u16,
    tensors: &[BmtsTensor],
    epoch: &str,
    chunk_hashes: &[u64],
) -> Result<Vec<u8>> {
    let mut msg = capnp::message::Builder::new_default();
    let mut root = msg.init_root::<crate::bmts_capnp::shard_header::Builder>();
    root.set_version(BMTS_VERSION_V2);
    root.set_node(node);
    root.set_epoch(epoch);
    root.set_chunk_size(DUET_CHUNK_SIZE);
    let mut hl = root.reborrow().init_chunk_hashes(chunk_hashes.len() as u32);
    for (i, &h) in chunk_hashes.iter().enumerate() {
        hl.set(i as u32, h);
    }
    let mut list = root.reborrow().init_tensors(tensors.len() as u32);
    for (i, t) in tensors.iter().enumerate() {
        let mut ti = list.reborrow().get(i as u32);
        ti.set_name(&t.name);
        let mut shape = ti.reborrow().init_shape(t.shape.len() as u32);
        for (j, &d) in t.shape.iter().enumerate() {
            shape.set(j as u32, d);
        }
        ti.set_dtype(t.dtype);
        ti.set_offset(t.offset);
        ti.set_length(t.length);
    }
    // stage layer list from names (blk.N.*)
    let mut layers: Vec<u32> = tensors
        .iter()
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    layers.sort_unstable();
    layers.dedup();
    let mut ll = root.init_layers(layers.len() as u32);
    for (i, &l) in layers.iter().enumerate() {
        ll.set(i as u32, l);
    }
    Ok(capnp::serialize::write_message_to_words(&msg))
}

/// Serialize a BMTS v2 shard: fixed header identical to v1, but the meta
/// blob is a Cap'n Proto `ShardHeader` message (zero-copy readable, schema
/// evolvable). Layer list is derived from tensor names. Epoch + advisory
/// chunk hashes are computed here from `data` — the shard is born with its
/// DUET identity.
#[cfg(feature = "capnp2")]
pub fn write_shard_v2(path: &str, node: u16, tensors: &[BmtsTensor], data: &[u8]) -> Result<()> {
    let epoch = epoch_of(data);
    let hashes = chunk_hashes(data, DUET_CHUNK_SIZE);
    let meta = build_meta_v2(node, tensors, &epoch, &hashes)?;

    // Stream: never hold header+meta+data in one allocation (shards are
    // gigabytes; /tmp may be tmpfs).
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&BMTS_MAGIC.to_le_bytes())?;
    f.write_all(&BMTS_VERSION_V2.to_le_bytes())?;
    f.write_all(&node.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&(meta.len() as u32).to_le_bytes())?;
    f.write_all(&meta)?;
    for block in data.chunks(4 << 20) {
        f.write_all(block)?;
    }
    f.flush()?;
    f.into_inner()?.sync_all()?;
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

        // Track C range-fetch: a sub-span of a tensor, size-stamp checked.
        let sub = shard
            .read_span(
                "blk.0.attn_q.weight",
                crate::op::Span { offset: 8, length: 16 },
            )
            .unwrap();
        assert_eq!(sub.bytes(), &blob[8..24]);
        // Full-span read equals the tensor.
        let full = shard
            .read_span(
                "blk.0.attn_q.weight",
                crate::op::Span { offset: 0, length: 32 },
            )
            .unwrap();
        assert_eq!(full.bytes(), t0);
        // Overrunning the declared length (the size-stamp) refuses.
        assert!(
            shard
                .read_span(
                    "blk.0.attn_q.weight",
                    crate::op::Span { offset: 0, length: 33 },
                )
                .is_err()
        );

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

    /// L0 for the format: v2 (Cap'n Proto meta) shard round-trips the
    /// tensor table and data bit-exactly; v1 files keep reading.
    #[test]
    fn test_bmts_v2_roundtrip() {
        let dir = std::env::temp_dir().join("ouro_bmts_v2_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_1.bmts");

        let blob: Vec<u8> = (0..128u8).map(|b| b.wrapping_mul(7)).collect();
        let tensors = vec![
            BmtsTensor {
                name: "token_embd.weight".into(),
                shape: vec![64, 2],
                dtype: 143,
                offset: 0,
                length: 64,
            },
            BmtsTensor {
                name: "blk.0.ssm_out.weight".into(),
                shape: vec![64, 32],
                dtype: 143,
                offset: 64,
                length: 64,
            },
        ];

        write_shard_v2(path.to_str().unwrap(), 3, &tensors, &blob).unwrap();
        let shard = BmtsShard::open(path.to_str().unwrap()).unwrap();
        assert_eq!(shard.node, 3);
        assert_eq!(shard.tensors, tensors);
        assert_eq!(shard.read_tensor("token_embd.weight").unwrap(), &blob[..64]);
        assert_eq!(shard.read_tensor("blk.0.ssm_out.weight").unwrap(), &blob[64..]);
        assert!(shard.read_tensor("missing").is_err());

        // v1 (JSON meta) shards keep reading — old cards load unchanged.
        let v1_path = dir.join("shard_1_v1.bmts");
        write_shard(v1_path.to_str().unwrap(), 1, &tensors, &blob).unwrap();
        let v1 = BmtsShard::open(v1_path.to_str().unwrap()).unwrap();
        assert_eq!(v1.tensors, tensors);
        assert_eq!(v1.node, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// v2 meta should be materially smaller than the JSON blob it replaces
    /// (the plan's zero-copy pitch). 851-tensor tables are the real case;
    /// a 64-tensor synthetic table shows the direction.
    #[test]
    fn test_bmts_v2_meta_smaller_than_json() {
        let tensors: Vec<BmtsTensor> = (0..64)
            .map(|i| BmtsTensor {
                name: format!("blk.{i}.ffn_up.weight"),
                shape: vec![5120, 17408],
                dtype: 143,
                offset: i as u64 * 19496960,
                length: 19496960,
            })
            .collect();
        let json = serde_json::to_vec(&tensors).unwrap();

        let mut msg = capnp::message::Builder::new_default();
        let mut root = msg.init_root::<crate::bmts_capnp::shard_header::Builder>();
        root.set_version(2);
        root.set_node(1);
        let mut list = root.reborrow().init_tensors(tensors.len() as u32);
        for (i, t) in tensors.iter().enumerate() {
            let mut ti = list.reborrow().get(i as u32);
            ti.set_name(&t.name);
            let mut shape = ti.reborrow().init_shape(t.shape.len() as u32);
            for (j, &d) in t.shape.iter().enumerate() {
                shape.set(j as u32, d);
            }
            ti.set_dtype(t.dtype);
            ti.set_offset(t.offset);
            ti.set_length(t.length);
        }
        let meta = capnp::serialize::write_message_to_words(&msg);

        eprintln!("meta bytes: json={} capnp={}", json.len(), meta.len());
        assert!(meta.len() < json.len(), "capnp meta must be smaller than json");
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
