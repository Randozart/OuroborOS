# Model card + Hadamard fold metadata (model.json + hadamard.json today).
#
# Status: schema-only scaffold (2026-09-20). Wire-in needs the `capnp` tool
# (see bmts.capnp header). Fields mirror cluster/src/infer/qwen35.rs Card /
# HadamardCfg — keep in sync until codegen replaces the hand-rolled structs.

@0xc7d8e9f0a1b2c3d4;

struct SsmConfig @0xa2b3c4d5e6f70819 {
  convKernel @0 :UInt32;
  dState @1 :UInt32;
  nKHeads @2 :UInt32;
  nVHeads @3 :UInt32;
  dInner @4 :UInt32;
}

struct HadamardMeta @0xb8c9d0e1f2a30415 {
  # prism.hadamard.* — see docs/BONSAI_PTQ1.md for the conjugation rules.
  blockSize @0 :UInt32;               # 1024 for Bonsai-2
  signWidths @1 :List(UInt32);        # per-input-dim sign slices (5120/6144/17408)
  signs @2 :List(Int8);               # flat +/-1, concatenation of slices
  gdnVGrouped @3 :Bool = false;       # ssm_out grouped [hd, rep, nk] order
}

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
  headDim @10 :UInt32 = 0;
  headVDim @11 :UInt32 = 0;
  fullAttentionInterval @12 :UInt32 = 1;
  nextn @13 :UInt32 = 0;
  ssm @14 :SsmConfig;
  hadamard @15 :HadamardMeta;         # absent (null) = unfolded checkpoint

  # AIR_PATH Track B: MTP draft head placement (brain node).
  hasDraft @16 :Bool = false;
  draftLayerStart @17 :UInt32 = 0;    # inclusive
  draftLayerEnd @18 :UInt32 = 0;      # inclusive
  draftNode @19 :UInt16 = 0;
}
