# Cap'n Proto + Bonsai 2 27B Exploration Plan

**Date:** 2026-09-20
**Status:** Active
**Depends on:** Phase 2 (in progress)

---

## Overview

Two parallel tracks exploring efficient data sharing and distributed ternary
inference:

1. **Cap'n Proto** — Replace serde_json serialization with zero-copy Cap'n Proto
   schemas at the highest-impact boundaries (BMTS metadata, shard maps, registry bus)
2. **Bonsai 2 27B** — Run Ternary-Bonsai-2-27B-PTQ1_0 (5.6GB GGUF) across
   the cluster as a distributed pipeline-parallel model, testing 1/2/4 node
   configurations with differential validation against llama.cpp

---

## Track 1: Cap'n Proto

### Why Cap'n Proto

Current serialization is 100% serde_json. For BMTS shard metadata (parsed on
every `open()`), registry heartbeat telemetry (high-frequency wire traffic),
and deployment plans, Cap'n Proto offers:

- **44× faster deserialization** than JSON (zero-copy: field access = pointer
  arithmetic + bounds check)
- **True zero-copy**: wire format ≈ in-memory layout, mmap-friendly
- **Schema evolution**: versioned schemas with forward/backward compat
- **no_std + no-alloc**: fits the bare-metal aspirations

Trade-offs: ~20-30% larger messages vs FlatBuffers, ~55% larger vs Protobuf.
Acceptable for the use cases targeted (metadata, not bulk data).

### Audit: Where to Apply, Where to Skip

| Target | Current | Impact | Decision |
|--------|---------|--------|----------|
| BMTS tensor metadata | JSON blob in shard header | **High** — parsed on every open, mmap-able | **Replace** |
| Shard map (`shard_map.json`) | JSON file | **Medium** — deployment metadata | **Replace** |
| Model card (`model.json`) | JSON file | **Medium** — arch config | **Replace** |
| Registry bus telemetry | JSON heartbeat | **High** — high-frequency wire traffic | **Replace** |
| ACTS activation frames | Hand-packed binary (26-byte header) | **Low** — already zero-copy on wire | **Skip** |
| Frame transport (`frames.rs`) | HMAC-signed binary | **Low** — would fight auth layer | **Skip** |
| Op kernel types | Internal-only, small payloads | **Low** — not worth schema dependency | **Skip** |

### Cap'n Proto Ecosystem (Rust)

| Crate | Version | Purpose |
|-------|---------|---------|
| `capnp` | 0.26.2 | Runtime library (12.9M downloads) |
| `capnpc` | 0.26.0 | Code generator (.capnp → Rust) |
| `capnp-futures` | 0.26.0 | Async serialization |
| `capnp-rpc` | 0.26.x | Object-capability RPC (Level 1) |

Async support: Yes (since v0.22, Oct 2025). Generates `async fn` via Rust's
return-position impl Trait in traits. No direct Tokio dependency but integrates
naturally with Tokio runtimes.

### Phase 1: Schema Definitions + BMTS v2

**Status: WIRED (2026-09-20).** `capnp` 1.5 tool + capnp/capnpc 0.27 crates.
Schemas live in `schemas/{bmts,model}.capnp`; codegen runs from
`cluster/build.rs` under the default `capnp2` feature (`--no-default-features`
skips it; v1 JSON shards read either way). `BmtsShard::open` auto-detects
v1/v2; `write_shard_v2` emits Cap'n Proto meta. `Card::to_capnp` /
`Card::from_capnp` round-trip losslessly (incl. HadamardMeta + draft head).
Meta blob on a 64-tensor table: 6518 B JSON -> 5168 B capnp. Contract tests:
`test_bmts_v2_roundtrip`, `test_bmts_v2_meta_smaller_than_json`,
`card_capnp_roundtrip*`. Next: sharder emits v2, registry bus schema.

#### 1.1 Create schemas

```
schemas/
├── bmts.capnp      # TensorMeta, TensorTable, ShardHeader
└── model.capnp     # ModelCard, ShardMap, StageSpec
```

**bmts.capnp:**
```capnp
@0xb3c4f1e2d5a69708;  # unique file ID

struct TensorMeta @0xd4e8f2a1b3c5d7e9 {
  name @0 :Text;
  shape @1 :List(UInt64);
  dtype @2 :UInt32;
  offset @3 :UInt64;  # byte offset within data section
  length @4 :UInt64;  # byte length of tensor data
}

struct TensorTable @0xa1b2c3d4e5f60718 {
  tensors @0 :List(TensorMeta);
}

struct ShardHeader @0xe9f0a1b2c3d4e5f6 {
  node @0 :UInt16;
  nTensors @1 :UInt32;
  meta @2 :TensorTable;
}
```

**model.capnp:**
```capnp
@0xc7d8e9f0a1b2c3d4;

struct ModelCard @0xf1a2b3c4d5e6f708 {
  architecture @0 :Text;
  nLayer @1 :UInt32;
  nEmbd @2 :UInt32;
  nHead @3 :UInt32;
  nHeadKv @4 :UInt32;
  nFf @5 :UInt32;
  nVocab @6 :UInt32;
  eps @7 :Float32;
  ropeBase @8 :Float32;
  nRot @9 :UInt32;
  headDim @10 :UInt32;
  headVDim @11 :UInt32;
  fullAttentionInterval @12 :UInt32;
  nextn @13 :UInt32;
}

struct SsmConfig @0xa2b3c4d5e6f70819 {
  convKernel @0 :UInt32;
  dState @1 :UInt32;
  nKHeads @2 :UInt32;
  nVHeads @3 :UInt32;
  dInner @4 :UInt32;
}

struct StageSpec @0xb3c4d5e6f7081920 {
  node @0 :UInt16;
  file @1 :Text;
  layers @2 :List(UInt32);
  tensors @3 :UInt32;
  bytes @4 :UInt64;
}

struct ShardMap @0xc4d5e6f708192031 {
  model @0 :Text;
  card @1 :ModelCard;
  nodes @2 :List(StageSpec);
}
```

#### 1.2 Add dependencies

In root `Cargo.toml`:
```toml
[workspace.dependencies]
capnp = "0.26"
capnpc = "0.26"
```

In `cluster/Cargo.toml`:
```toml
[dependencies]
capnp = { workspace = true }

[build-dependencies]
capnpc = { workspace = true }
```

#### 1.3 Update `cluster/build.rs`

Add Cap'n Proto schema compilation:
```rust
fn main() {
    // Existing RDMA C shim
    cc::Build::new()
        .file("src/transport/ibv_poll_cq_shim.c")
        .compile("ibv_poll_cq_shim");

    // Cap'n Proto schemas
    capnpc::CompilerCommand::new()
        .src_prefix("schemas")
        .file("schemas/bmts.capnp")
        .run()
        .expect("capnp compile bmts.capnp");
    capnpc::CompilerCommand::new()
        .src_prefix("schemas")
        .file("schemas/model.capnp")
        .run()
        .expect("capnp compile model.capnp");
}
```

#### 1.4 Update `bmts.rs`

Key changes:
- `BmtsShard::open()` detects version field: v1 = JSON (existing), v2 = Cap'n Proto
- `BmtsTensor` becomes a Cap'n Proto reader (zero-copy field access)
- `write_shard()` defaults to v2 Cap'n Proto metadata
- `BmtsTensor` struct kept for backward compat, created from Cap'n Proto reader on access

```rust
// Version detection in open():
let version = u16::from_le_bytes(header[4..6].try_into().unwrap());
match version {
    1 => { /* existing JSON path */ }
    2 => { /* new Cap'n Proto zero-copy path */ }
    _ => bail!("unsupported BMTS version {}", version),
}
```

#### 1.5 Update `shard_model.py`

Python `capnp` library to emit Cap'n Proto metadata instead of JSON. Or:
- Keep JSON in shard for now (v1 compat)
- Add a `--capnp` flag to emit v2
- The Python side can use `pycapnp` or write raw Cap'n Proto encoding

### Phase 2: Registry Bus Upgrade

1. Create `schemas/registry.capnp` for `NodeRecord`, `Event`, heartbeat
2. Update bus protocol to serialize/deserialize with Cap'n Proto
3. Version byte on wire to distinguish JSON vs Cap'n Proto messages
4. Backward compat: old nodes still speak JSON, new nodes speak Cap'n Proto

### BMTS v2 Layout

```text
magic:     u32  0x4F55524F ("OURO")
version:   u16  2  (was 1)
node:      u16
n_tensors: u32
meta_len:  u32
meta:      Cap'n Proto TensorTable (zero-copy via mmap)
data:      concatenated tensor bytes (unchanged)
```

Tensor data section is unchanged. Only metadata encoding changes.
Existing v1 shards remain loadable via the JSON path.

---

## Track 2: Bonsai 2 27B Distributed Inference

### Model Facts

| Property | Value |
|----------|-------|
| File | `/home/randozart/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf` |
| Size | 5.6 GB (5,946,648,928 bytes) |
| Architecture | BitNet (ternary TQ1_0 weights) |
| Min CPU | SSSE3 (PSHUFB from 2006) — no AVX required |
| Dequant | TQ1_0 already in `cluster/src/infer/dequant.rs` |

### CPU Instruction Set Tiers

BitNet eliminates floating-point multiplication entirely. Arithmetic becomes
integer ADD/SUB + bitwise ops:

| Tier | Hardware | Mechanism | Throughput |
|------|----------|-----------|------------|
| 0 | 80386+ (1985) | C fallback loop: `if w==1 sum+=a; else if w==-1 sum-=a` | 1× |
| 1 | Core 2 Duo (2006) | SSSE3 PSHUFB LUT kernels | 16 parallel ternary dot/cycle |
| 2 | Haswell+ (2013) / ARM NEON | AVX2 `_mm256_shuffle_epi8` | 32 parallel ternary dot/cycle |
| 3 | AVX-512 / VNNI | 512-bit parallel | 64 parallel ternary dot/cycle |

Every decommissioned OptiPlex, ThinkCentre, ProDesk from the last 15 years
can run Bonsai 2 27B. The scheduler should tag nodes with SIMD capability.

### Implementation Steps

#### Step 1: Shard the GGUF

```bash
# 1-node (entire model on one machine, ~5.6GB RAM)
python3 tools/shard_model.py \
  /home/randozart/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf 1 \
  --output-dir shards_bonsai27_n1

# 2-node (~2.8GB per node, fits 8GB machines)
python3 tools/shard_model.py \
  /home/randozart/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf 2 \
  --output-dir shards_bonsai27_n2

# 4-node (~1.4GB per node, fits 4GB machines)
python3 tools/shard_model.py \
  /home/randozart/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf 4 \
  --output-dir shards_bonsai27_n4
```

#### Step 2: Extract ArchConfig from GGUF

The `shard_model.py` already parses GGUF KV metadata into `model_card`.
Run the 1-node shard to extract the card, then add:

```rust
// cluster/src/infer/mod.rs
impl ArchConfig {
    /// Bonsai 2 27B — ternary BitNet model (TQ1_0 weights).
    /// Extracted from GGUF: general.architecture = "bitnet"
    pub fn bitnet_27b() -> Self {
        Self {
            n_embd: ???,    // from GGUF KV
            n_head: ???,
            n_head_kv: ???,
            n_ff: ???,
            n_rot: ???,
            eps: ???,
            rope_base: ???,
            n_vocab: ???,
        }
    }
}
```

#### Step 3: Wire up pipeline runner

- `agent/src/stage.rs` already supports `Loaded::Bitnet` variant
- Add Bonsai 27B as a recognized model family
- Update shard_map loading for the new shard set

#### Step 4: Differential test (pure-Rust vs bitnet-rs FFI)

```rust
// bitnet-rs/tests/bonsai27_diff.rs
#[test]
#[ignore]
fn bonsai27_differential() {
    // 1. Load via bitnet-rs FFI (GGUF directly)
    let m_ffi = BitNetModel::load(
        "/home/randozart/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        256, 8
    );
    let ids = m_ffi.tokenize("The meaning of life is", true);
    let cap = m_ffi.decode_capture(&ids).unwrap();

    // 2. Load via pure-Rust engine (BMTS shards)
    let cfg = ArchConfig::bitnet_27b();
    let mut model = PipelineModel::load(
        &[
            "shards_bonsai27_n4/shard_1.bmts",
            "shards_bonsai27_n4/shard_2.bmts",
            "shards_bonsai27_n4/shard_3.bmts",
            "shards_bonsai27_n4/shard_4.bmts",
        ],
        cfg
    ).unwrap();
    let h = model.prefill(&ids).unwrap();

    // 3. Compare per-layer hidden states
    // Format cap output to match BMTS tensor names
    // Assert L1 parity within tolerance
}
```

#### Step 5: Test all node variations

| Config | Nodes | Shard Size | RAM/Node | Notes |
|--------|-------|------------|----------|-------|
| n1 | 1 | ~5.6 GB | ~6 GB | Simplest test, single-machine |
| n2 | 2 | ~2.8 GB | ~3 GB | Pipeline across 2 machines |
| n4 | 4 | ~1.4 GB | ~2 GB | Full pipeline, fits Kria-class |

### Scheduler Implications

BitNet runs on old hardware. The scheduler should detect and exploit this:

```rust
// NodeEntry gains SIMD capability tags:
pub has_ssse3: bool,  // minimum for BitNet (PSHUFB LUT kernels)
pub has_avx2: bool,   // 2× throughput via 256-bit shuffles
pub has_neon: bool,   // ARM equivalent (Cortex-A53+)
```

Placement rules:
- BitNet models prefer SSSE3+ nodes (all x86 since 2006)
- AVX2 nodes get priority for larger models (throughput bonus)
- Energy budget: BitNet on old CPU = ~65W burst, ~10W idle. Embodied carbon
  savings dominate operational cost (70-80% of lifetime carbon is fabrication)

---

## Sequencing

Both tracks are independent. Recommended order:

| Week | Cap'n Proto | Bonsai 2 27B |
|------|-------------|--------------|
| 1 | Schema files, BMTS v2 format | Shard GGUF → BMTS (all counts), extract ArchConfig |
| 2 | bmts.rs update, shard_model.py emit v2 | Pure-Rust pipeline test, single-node inference |
| 3 | Registry bus schemas + upgrade | bitnet-rs differential test, multi-node pipeline |
| 4 | Shard map/model card schemas | Full differential validation, benchmark all configs |

---

## Key Files

| File | Role |
|------|------|
| `cluster/src/bmts.rs` | BMTS shard format — primary Cap'n Proto target |
| `cluster/src/infer/mod.rs` | Pure-Rust inference engine — needs bitnet_27b() |
| `cluster/src/infer/dequant.rs` | TQ1_0 dequantization (already implemented) |
| `cluster/src/pipeline.rs` | PipelinePlan, ACTS frames |
| `cluster/src/registry.rs` | Node registry — Cap'n Proto target |
| `cluster/src/registry/bus.rs` | Registry bus protocol |
| `bitnet-rs/src/lib.rs` | llama.cpp FFI wrapper |
| `bitnet-rs/tests/cap9_test.rs` | Differential test pattern |
| `tools/shard_model.py` | GGUF → BMTS sharding tool |
| `agent/src/stage.rs` | Pipeline stage runner |
| `schemas/` | New: Cap'n Proto schema definitions |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Cap'n Proto 8-byte alignment adds ~20-30% to metadata size | Acceptable for metadata (not bulk data). Tensor data section unchanged. |
| Python capnp library may be immature | Can write raw Cap'n Proto encoding in Python, or keep JSON fallback. |
| Bonsai 27B ArchConfig unknown until GGUF is parsed | Step 1 (shard) extracts it automatically via model_card. |
| 4-node pipeline on 4GB machines may be tight | OS + runtime overhead. Test with swap or use 2-node config. |
| bitnet-rs FFI requires C++ build of bitnet-cpp | Already configured as git submodule with build scripts. |

---

*The tail feeds the head. The old silicon runs the new mind.*
