# BMTS v2 — Cap'n Proto shard header (replaces the JSON meta blob).
#
# Status: schema-only scaffold (2026-09-20). Wire-in requires the `capnp`
# tool for build-time codegen (pacman -S capnproto) and a `capnp` feature
# on ouro-cluster. See docs/CAPNP_BONSAI_PLAN.md Phase 1.
#
# v1 (current, serde_json): magic u32 | version u16 | node u16 |
# n_tensors u32 | meta_len u32 | JSON | data. The tensor DATA section is
# unchanged — only the metadata blob swaps to this schema, mmap-friendly.

@0xb3c4f1e2d5a69708;

using Model = import "model.capnp";

struct TensorMeta @0xd4e8f2a1b3c5d7e9 {
  name @0 :Text;
  shape @1 :List(UInt64);
  dtype @2 :UInt32;      # ggml type id (143 = PTQ1_0, 30 = BF16, 0 = F32)
  offset @3 :UInt64;     # byte offset within the shard data section
  length @4 :UInt64;     # byte length of tensor data
}

struct ShardHeader @0xe9f0a1b2c3d4e5f6 {
  version @0 :UInt16 = 2;
  node @1 :UInt16;                    # 1-based pipeline stage index
  tensors @2 :List(TensorMeta);
  # stage layer list derivable from tensor names; kept explicit so a
  # header read alone answers "what does this node own?"
  layers @3 :List(UInt32);

  # ---- DUET (docs/DUET.md): content identity for delta sync ----
  # epoch = sha256 hex over the shard DATA section. The contract: a rebuilt
  # shard must hash to exactly this. Present from v2 shards written by
  # `ouro-bmts upgrade` or future sharders; absent (empty) in bare v2.
  epoch @4 :Text = "";
  # advisory chunk hashes for delta diffing (SipHash-13, fixed key, u64 per
  # chunk_size block of the DATA section). NOT the authority — epoch is.
  # 0 = unindexed.
  chunkSize @5 :UInt32 = 0;           # bytes per chunk (0 = unindexed)
  chunkHashes @6 :List(UInt64);
}

# Pipeline deployment plan (shard_map.json today).
struct ShardMap @0xf6e5d4c3b2a1908f {
  architecture @0 :Text;              # "qwen35", "bitnet", ...
  nNodes @1 :UInt16;
  stages @2 :List(StageSpec);

  struct StageSpec {
    node @0 :UInt16;
    shardPath @1 :Text;
    layers @2 :List(UInt32);
    ownsEmbed @3 :Bool;
    ownsHead @4 :Bool;                # untied output.weight
    ownsOutputNorm @5 :Bool;
  }
}
