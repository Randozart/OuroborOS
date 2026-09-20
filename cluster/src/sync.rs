//! DUET P1 — shard delta sync (docs/DUET.md).
//!
//! Sender and receiver both hold chunk indexes; the pull plan names only
//! the chunks the receiver lacks. Rebuilds are staged and epoch-verified
//! before they become visible: speculation may fail, it may never lie.

use crate::bmts::{chunk_hashes, epoch_of, DUET_CHUNK_SIZE};
use anyhow::{bail, Context, Result};
use std::io::Write;

    /// A request for one byte-range of a shard's data section.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ChunkRequest {
        pub index: u32,
        /// The offset the sender must serve.
        pub offset: u64,
        pub length: u32,
    }

/// Which chunks of a shard does `local` lack, given `remote`'s length?
///
/// Requests run from the end of local data to the remote end. A partial
/// chunk at the local boundary is re-requested from its tail (the index
/// is bookkeeping; the offset is the truth). The epoch decides the final
/// verdict after rebuild.
pub fn plan_pull(local_data: &[u8], remote_len: u64) -> Result<Vec<ChunkRequest>> {
    let cs = DUET_CHUNK_SIZE as u64;
    if remote_len < local_data.len() as u64 {
        bail!("remote shard smaller than local ({} < {}): not a delta pull", remote_len, local_data.len());
    }
    let mut out = Vec::new();
    let mut next = local_data.len() as u64;
    while next < remote_len {
        let chunk_end = ((next / cs) + 1) * cs;
        let end = chunk_end.min(remote_len);
        out.push(ChunkRequest {
            index: (next / cs) as u32,
            offset: next,
            length: (end - next) as u32,
        });
        next = end;
    }
    Ok(out)
}

/// Chunks whose hashes differ between two same-length data slices — the
/// fast path when receiver and sender hold equal-length stale/fresh pairs.
pub fn plan_diff(local_data: &[u8], remote_data: &[u8]) -> Result<Vec<ChunkRequest>> {
    if local_data.len() != remote_data.len() {
        return plan_pull(local_data, remote_data.len() as u64);
    }
    let lh = chunk_hashes(local_data, DUET_CHUNK_SIZE);
    let rh = chunk_hashes(remote_data, DUET_CHUNK_SIZE);
    let cs = DUET_CHUNK_SIZE as u64;
    let mut out = Vec::new();
    for (i, (&a, &b)) in lh.iter().zip(&rh).enumerate() {
        if a != b {
            let offset = i as u64 * cs;
            out.push(ChunkRequest {
                index: i as u32,
                offset,
                length: (remote_data.len() as u64 - offset).min(cs) as u32,
            });
        }
    }
    Ok(out)
}

/// Staged rebuild: apply chunk replies (offset, payload) to the local tail
/// of a shard and atomically produce the full data section. The result is
/// written to a temp path and renamed only after the epoch verifies — no
/// partial state is ever observable (DUET golden rule).
pub fn rebuild_into(
    local_data: &[u8],
    chunks: &[(u64, Vec<u8>)],
    expected_epoch: &str,
    final_path: &str,
) -> Result<()> {
    let mut total = local_data.len() as u64;
    for (offset, payload) in chunks {
        total = total.max(offset + payload.len() as u64);
    }

    let mut buf = local_data.to_vec();
    buf.resize(total as usize, 0);
    for (offset, payload) in chunks {
        let start = *offset as usize;
        let end = start + payload.len();
        if end > buf.len() {
            bail!("chunk at {offset} overruns rebuild size");
        }
        buf[start..end].copy_from_slice(payload);
    }

    let epoch = epoch_of(&buf);
    if epoch != expected_epoch {
        bail!(
            "epoch mismatch after rebuild: expected {} got {}",
            &expected_epoch[..16.min(expected_epoch.len())],
            &epoch[..16]
        );
    }

    // Stage + atomic rename: readers see the whole shard or none of it.
    // create_new refuses to race a concurrent rebuilder of the same target.
    let stage = format!("{final_path}.duet-staging");
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage)
        .with_context(|| format!("staging {stage} (exists? another rebuild in flight?)"))?;
    {
        let mut f = std::io::BufWriter::new(f);
        f.write_all(&buf)?;
        f.flush()?;
        f.into_inner()?.sync_all()?;
    }
    std::fs::rename(&stage, final_path)
        .with_context(|| format!("publishing {final_path}"))?;
    Ok(())
}

/// Bytes the wire must carry for this plan — the DUET bill.
pub fn wire_bytes(plan: &[ChunkRequest]) -> u64 {
    plan.iter().map(|r| r.length as u64).sum()
}

// ---------------------------------------------------------------------------
// P1.5 — Deferred Unification: quiet-wire gossip + chunk staging.
// ---------------------------------------------------------------------------

/// An index advertisement gossiped on the quiet wire: tiny, idempotent,
/// stale-safe (epoch IS the freshness).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexAdvert {
    /// Shard identity (file name / deployment key).
    pub shard: String,
    /// sha256 hex of the data section — the freshness authority.
    pub epoch: String,
    /// Data-section length in bytes.
    pub len: u64,
}

/// One staged chunk waiting to be claimed by a rebuild.
#[derive(Debug, Clone)]
struct StagedChunk {
    data: Vec<u8>,
    epoch: String,
    staged_at: std::time::Instant,
}

/// Receiver-side chunk staging with a hard byte budget and TTL eviction
/// (docs/DUET.md guard 3): a chatty node cannot fill a tail's disk.
pub struct ChunkStage {
    max_bytes: u64,
    ttl: std::time::Duration,
    chunks: std::collections::HashMap<(String, u64), StagedChunk>,
    bytes: u64,
}

impl ChunkStage {
    pub fn new(max_bytes: u64, ttl: std::time::Duration) -> Self {
        Self { max_bytes, ttl, chunks: std::collections::HashMap::new(), bytes: 0 }
    }

    pub fn staged_bytes(&self) -> u64 {
        self.bytes
    }

    fn evict_expired(&mut self) {
        let now = std::time::Instant::now();
        let ttl = self.ttl;
        let before = self.chunks.len();
        self.chunks.retain(|_, c| now.duration_since(c.staged_at) < ttl);
        if self.chunks.len() != before {
            self.recount();
        }
    }

    fn recount(&mut self) {
        self.bytes = self.chunks.values().map(|c| c.data.len() as u64).sum();
    }

    /// Stage one chunk; LRU-evicts oldest staged chunks if over budget.
    /// Duplicate staging of the same (shard, chunk) overwrites idempotently.
    pub fn stage(&mut self, shard: &str, offset: u64, epoch: &str, data: Vec<u8>) {
        self.evict_expired();
        let key = (shard.to_string(), offset);
        if let Some(old) = self.chunks.get(&key) {
            self.bytes -= old.data.len() as u64;
        }
        self.chunks.insert(key, StagedChunk {
            data,
            epoch: epoch.to_string(),
            staged_at: std::time::Instant::now(),
        });
        self.recount();
        while self.bytes > self.max_bytes {
            // evict the OLDEST staged chunk (min staged_at)
            let Some(oldest) = self
                .chunks
                .iter()
                .min_by_key(|(_, c)| c.staged_at)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(c) = self.chunks.remove(&oldest) {
                self.bytes -= c.data.len() as u64;
            } else {
                break;
            }
        }
    }

    /// Claim staged chunks for a rebuild of `shard` at `epoch`; chunks from
    /// other epochs are not claimable (stale garbage, GC'd on sight).
    pub fn claim(&mut self, shard: &str, epoch: &str, plan: &[ChunkRequest]) -> Vec<(u64, Vec<u8>)> {
        self.evict_expired();
        let mut out = Vec::new();
        for r in plan {
            let key = (shard.to_string(), r.offset);
            if let Some(c) = self.chunks.get(&key) {
                if c.epoch == epoch {
                    out.push((r.offset, c.data.clone()));
                }
            }
        }
        out
    }
}

/// Ranks which shards a node should prefetch on the quiet wire
/// (docs/DUET.md guard 4: plan-ranked, never blind).
pub trait PrefetchRanker {
    /// Shard keys this node will need soon, best-first.
    fn planned_shards(&self, node: u16) -> Vec<String>;
}

/// Fallback when no fresh placement plan exists: pipeline adjacency —
/// a stage's neighbors are the most likely next homes.
pub struct AdjacentStages {
    pub total_stages: u16,
}

impl PrefetchRanker for AdjacentStages {
    fn planned_shards(&self, node: u16) -> Vec<String> {
        let mut out = Vec::new();
        if node > 1 {
            out.push(format!("shard_{}.bmts", node - 1));
        }
        if node < self.total_stages {
            out.push(format!("shard_{}.bmts", node + 1));
        }
        out
    }
}

#[cfg(test)]
mod p15_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_staging_budget_evicts_oldest() {
        let mut stage = ChunkStage::new(2000, Duration::from_secs(60));
        stage.stage("s.bmts", 0, "ep1", vec![7u8; 1000]);
        stage.stage("s.bmts", 65536, "ep1", vec![8u8; 1000]);
        assert_eq!(stage.staged_bytes(), 2000);
        // over budget: oldest (offset 0) must go
        stage.stage("s.bmts", 131072, "ep1", vec![9u8; 1000]);
        assert!(stage.staged_bytes() <= 2000, "budget enforced");
        let plan = vec![ChunkRequest { index: 2, offset: 131072, length: 1000 }];
        let claimed = stage.claim("s.bmts", "ep1", &plan);
        assert_eq!(claimed.len(), 1, "newest chunk survives");
        let plan0 = vec![ChunkRequest { index: 0, offset: 0, length: 1000 }];
        assert!(stage.claim("s.bmts", "ep1", &plan0).is_empty(), "oldest evicted");
    }

    #[test]
    fn test_claim_is_epoch_scoped() {
        let mut stage = ChunkStage::new(10_000, Duration::from_secs(60));
        stage.stage("s.bmts", 0, "epoch-old", vec![1u8; 100]);
        let plan = vec![ChunkRequest { index: 0, offset: 0, length: 100 }];
        assert!(stage.claim("s.bmts", "epoch-new", &plan).is_empty(), "stale epoch unclaimable");
        assert!(stage.claim("s.bmts", "epoch-old", &plan).len() == 1);
        assert!(stage.claim("other.bmts", "epoch-old", &plan).is_empty(), "shard-scoped");
    }

    #[test]
    fn test_duplicate_staging_is_idempotent() {
        let mut stage = ChunkStage::new(10_000, Duration::from_secs(60));
        stage.stage("s.bmts", 0, "ep", vec![1u8; 100]);
        stage.stage("s.bmts", 0, "ep", vec![2u8; 100]);
        assert_eq!(stage.staged_bytes(), 100, "overwrite, not accumulate");
    }

    #[test]
    fn test_adjacent_stages_ranking() {
        let r = AdjacentStages { total_stages: 4 };
        assert_eq!(r.planned_shards(2), vec!["shard_1.bmts", "shard_3.bmts"]);
        assert_eq!(r.planned_shards(1), vec!["shard_2.bmts"], "no left neighbor for stage 1");
        assert_eq!(r.planned_shards(4), vec!["shard_3.bmts"], "no right neighbor for last stage");
    }

    #[test]
    fn test_advert_roundtrip_serde() {
        let a = IndexAdvert { shard: "shard_1.bmts".into(), epoch: "c0ff ee".replace(' ', ""), len: 12345 };
        let bytes = serde_json::to_vec(&a).unwrap();
        let b: IndexAdvert = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_data(n: usize, seed: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        let mut s = seed as u32 | 1;
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((s >> 24) as u8);
        }
        v
    }

    /// The full P1 loop: local has a stale prefix of the remote shard,
    /// plan pulls only the missing tail, rebuild verifies epoch bit-exact.
    #[test]
    fn test_pull_rebuild_bit_exact() {
        let remote = lcg_data(DUET_CHUNK_SIZE as usize + 4096, 7);
        let local = remote[..DUET_CHUNK_SIZE as usize - 8192].to_vec(); // shorter stale

        let plan = plan_pull(&local, remote.len() as u64).unwrap();
        // unaligned local tail (57344) → partial-chunk request + aligned rest
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].offset, local.len() as u64);
        let wire = wire_bytes(&plan);
        assert_eq!(wire, (remote.len() - local.len()) as u64, "wire bill is exactly the gap");

        let chunks: Vec<(u64, Vec<u8>)> = plan
            .iter()
            .map(|r| {
                let start = r.offset as usize;
                let end = start + r.length as usize;
                (r.offset, remote[start..end].to_vec())
            })
            .collect();
        let dir = std::env::temp_dir().join("ouro_duet_p1");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("rebuilt.bmts");
        let epoch = epoch_of(&remote);
        rebuild_into(&local, &chunks, &epoch, out.to_str().unwrap()).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), remote, "rebuild must be bit-exact");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Wrong epoch expectation → refusal, nothing published.
    #[test]
    fn test_rebuild_refuses_epoch_mismatch() {
        let remote = lcg_data(4096, 9);
        let dir = std::env::temp_dir().join("ouro_duet_p1_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("rebuilt.bmts");
        let r = rebuild_into(&[], &[(0, remote)], "deadbeef", out.to_str().unwrap());
        assert!(r.is_err());
        assert!(!out.exists(), "no partial state may be visible");
        let stage = dir.join("rebuilt.bmts.duet-staging");
        assert!(!stage.exists(), "staging cleaned up");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Equal-length stale/fresh diff: only diverging chunks are pulled.
    #[test]
    fn test_diff_targets_only_changed_chunks() {
        let mut fresh = lcg_data(DUET_CHUNK_SIZE as usize * 3, 11);
        let stale = fresh.clone();
        // corrupt exactly chunk 1 (middle third)
        let mid = DUET_CHUNK_SIZE as usize;
        for b in &mut fresh[mid..mid + 64] {
            *b = b.wrapping_add(1);
        }
        let plan = plan_diff(&stale, &fresh).unwrap();
        assert_eq!(plan.len(), 1, "one diverging chunk, one request");
        assert_eq!(plan[0].index, 1, "the middle chunk");
    }

    /// Identical shards: zero-byte wire bill (the DUET promise).
    #[test]
    fn test_identical_shard_pulls_nothing() {
        let d = lcg_data(8192, 13);
        let plan = plan_diff(&d, &d).unwrap();
        assert!(plan.is_empty());
        assert_eq!(wire_bytes(&plan), 0);
    }
}
