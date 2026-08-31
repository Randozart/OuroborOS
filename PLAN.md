# OurobourOS Enhanced Plan

**Date:** 2026-08-29
**Status:** Phase 0 ✅, Phase 1 ✅, Phase 2 in progress — see §5 tables
**Founding document:** `CONSTITUTION.md` — all architecture here instantiates
its articles. When this plan and the constitution seem to conflict, the
constitution wins; when a design decision here cannot cite an article, it is
cargo cult and must be rejected at review (Constitution Art. 11).

**Reading guide:** §1-4 = why (the thesis), §5-9 = how (phases, protocol,
weights), §13 = the Qwen Program (the summit), §14 = heterogeneous weight
placement + HDMI modem (the constitution's first full applications), §15 = prior art
& lineage (provenance; read before claiming novelty), §16 = build schedule,
§17 = GPU substrate findings & corpus synthesis. Canonical architecture lives
in ARCHITECTURE.md; contracts in docs/CONTRACTS.md.

---

## 1. What the Discussion Gets Right

### 1.1 The VRAM Wall Argument

BitNet 1.58b compresses 400B params to ~100GB. Four cheap desktops with 32GB DDR4 each can hold a frontier-class model. This is the strongest motivation for the project.

| Model | FP16 Footprint | BitNet 1.58b Footprint |
|-------|---------------|----------------------|
| 8B | ~16 GB | ~2.0 GB |
| 70B | ~140 GB | ~17.5 GB |
| 400B | ~800 GB | ~100 GB |

### 1.2 Pipeline Parallelism Math

8KB activation tensors over 1GbE = ~64μs wire time. LLM layer compute on old GPUs = 20-50ms. Network is NOT the bottleneck for pipeline inference. The ring topology with hardcoded MAC forwarding is sound.

```
Physical Transmission Time = 8KB / 125,000 KB/s ≈ 64 microseconds (1GbE)
Physical Transmission Time ≈ 6.4 microseconds (10GbE)

GPU Compute Latency = 20-50ms per layer slice
Network Latency << GPU Compute Latency
```

### 1.3 KV-Cache Locality

Each node only stores KV-cache for its layers. Inter-node traffic stays tiny (8KB per token). This is why clustered inference works at all on old hardware.

### 1.4 Direct LBA Streaming

Eliminating ext4 overhead for weight loading is a real optimization. The BMTS (Bare-Metal Tensor Storage) format with 4KB-aligned tensors enables zero-copy DMA from NVMe to RAM.

### 1.5 Nix-Style Immutable Closures

Declarative node specs, cryptographic hash verification, reproducible builds. This solves the heterogeneous deployment problem cleanly.

---

## 2. Critical Gaps and Risks

### 2.1 Briev Compiler NOT Production-Ready

| Aspect | Status |
|--------|--------|
| GitHub stars | 38 |
| Maintainer | Single developer |
| LLVM backend | Functional for x86_64 |
| Freestanding targets | Implicit via LLVM, untested |
| Open issues | 8 |
| Pull requests | 0 |

**Risk:** Cannot build OurobourOS microkernels in Briev today. The compiler is still in completion phase (Phase 7 nested recursive types deferred, ForAll/Exists stubs only).

**Recommendation:** Build OurobourOS control plane in Rust (already started). Use Briev for workloads and future bare-metal nodes once LLVM backend stabilizes.

### 2.2 Unikraft Has NO GPU Support

| Feature | Status |
|---------|--------|
| VFIO passthrough | Not implemented |
| CUDA drivers | Not supported |
| GPU compute framework | None |
| Static linking | Required (blocks ML frameworks) |
| PXE boot | Feasible but not first-class |

**Risk:** The "sterile microkernel + JIT package delivery" architecture cannot use GPUs. The Alienware's GTX 960 cannot be accessed from Unikraft.

**Recommendation:** For GPU nodes (Alienware with GTX 960), use CachyOS with minimal systemd + custom daemon. For CPU-only nodes (BitNet inference), Unikraft is viable.

### 2.3 Alienware Alpha R2 Hardware Constraints

| Constraint | Impact |
|------------|--------|
| Single SODIMM slot | No dual-channel = halved memory bandwidth |
| 1GbE only | No 10GbE without PCIe expansion |
| 35W TDP CPU | Thermal throttling under sustained load |
| Linux suspend/resume | Broken, shutdown may sleep instead |
| AGA port | Windows-only, poorly documented on Linux |

**Recommendation:** Alpha R2 is a worker node, not a master. GTX 960 (4GB VRAM) can run small BitNet models natively. Add PCIe 2.5GbE adapter if pipeline bandwidth needed.

### 2.4 Kria K26 Constraints

| Constraint | Impact |
|------------|--------|
| 4GB DDR4 (soldered) | Cannot hold large model slices |
| No NVMe | Weight loading via SATA or network |
| FPGA transceivers | Require custom carrier board for 10/40GbE |
| 15W TDP | Limits compute throughput |

**Recommendation:** K26 is the FPGA interconnect node (NTB bridge or custom packet switch), not a compute node. ARM cores handle control plane; FPGA fabric handles data plane.

### 2.5 BitNet Model Ecosystem Is Tiny

| Model | Params | Status |
|-------|--------|--------|
| BitNet-b1.58-2B-4T | 2.4B | Official Microsoft |
| bitnet_b1_58-large | 0.7B | Community |
| bitnet_b1_58-3B | 3.3B | Community |
| Llama3-8B-1.58-100B-tokens | 8.0B | Community |
| Falcon3 Family | 1B-10B | TII |

**Critical:** Models MUST be trained from scratch with ternary weights. Cannot quantize existing Llama 3 to 1.58-bit. Custom GGUF format incompatible with stock llama.cpp.

**Recommendation:** Start with `bitnet-2b-tq1_0.gguf` (1.1GB, already in workspace). For larger models, use `ik_llama.cpp` fork with I2_S support.

### 2.6 Missing From Discussion

| Topic | What's Missing |
|-------|---------------|
| Error handling | What happens when a node mid-pipeline crashes? |
| Checkpointing | How to save/restore pipeline state? |
| Weight distribution | How to shard model across nodes with different RAM? |
| Weight update | How to hot-swap weights without restart? |
| Monitoring | Real-time pipeline health observation? |
| Security | Raw L2 Ethernet with no authentication |

---

## 3. Existing Codebase Analysis

### 3.1 What's Built (OurobourOS/cluster/)

| Module | Status | Lines |
|--------|--------|-------|
| `beast/mod.rs` | ✅ Complete | 159 |
| `beast/topology.rs` | ✅ Complete | 137 |
| `beast/node_state.rs` | ✅ Complete | 46 |
| `scheduler/mod.rs` | ✅ Complete + tests | 225 |
| `scheduler/energy_budget.rs` | ✅ Complete + tests | 103 |
| `scheduler/workload_class.rs` | ✅ Complete + tests | 101 |
| `transport/mod.rs` | ✅ Trait defined | 18 |
| `probe/mod.rs` | ✅ Complete | 98 |
| `probe/cpu.rs` | ✅ Complete | 86 |
| `probe/energy.rs` | ✅ Exists | - |
| `probe/memory.rs` | ✅ Exists | - |
| `probe/network.rs` | ✅ Exists | - |
| `error.rs` | ✅ Complete | 46 |
| `shell/` | 🔲 Empty | - |
| `agent/` | 🔲 Empty | - |

### 3.2 Reusable Code From Other Projects

| Source | What to Reuse | Location |
|--------|--------------|----------|
| VITRIOL | GPU probe (nvidia-smi) | `libvitriol/src/probe.rs` |
| VITRIOL | Hardware info struct | `libvitriol/src/probe.rs` |
| briev-backend-foundation | S-expression parser | `src/beast/sexpr.rs` |
| moore-kernel | msh parser (LL(1)) | `kernel/msh/src/parser.rs` |
| moore-kernel | REPL loop pattern | `kernel/msh/src/main.rs` |

### 3.3 Existing BitNet Model

```
/home/randozart/Desktop/Projects/bitnet-2b-tq1_0.gguf (1.1 GB)
```

This is the official Microsoft BitNet-b1.58-2B-4T model. Ready for single-node testing.

---

## 4. Revised Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OurobourOS Shell (Rust)                    │
│  Propositional interface │ Context memory │ Dot notation     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cluster Beast (Rust)                       │
│  Topology + node state as S-exprs                           │
│  Probed at boot │ Live-patchable │ Inspectable              │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Scheduler (Rust)                           │
│  BitNet inference routing │ Energy budget │ Fault tolerance  │
│  Workload: BRANCH_HEAVY │ SIMD_FRIENDLY │ LLM_INFERENCE     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Transport Layer                            │
│  MVP: TCP with custom protocol                              │
│  Phase 2: Raw L2 Ethernet (DPDK/uknetdev)                  │
│  Task dispatch │ Weight streaming │ Heartbeat               │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Worker Nodes                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Alienware R2     │  │ Old Desktop     │  │ Kria K26   │ │
│  │ GTX 960 4GB      │  │ CPU-only        │  │ FPGA fabric│ │
│  │ BitNet inference │  │ BitNet inference│  │ NTB bridge │ │
│  │ 16GB DDR4        │  │ 32GB DDR4       │  │ 4GB DDR4   │ │
│  └──────────────────┘  └──────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Control plane language | Rust | Already built, type-safe, ecosystem |
| Bare-metal language | Briev (future) | Contract enforcement, reactive nodes |
| Inference engine | BitNet.cpp | Only option for 1.58-bit ternary models |
| Transport (MVP) | TCP | Proven, works everywhere |
| Transport (Phase 2) | Raw L2 | Lower latency, zero-copy |
| Worker OS | CachyOS | GPU support, Arch-based, minimal |
| Unikraft | CPU-only workers only | No GPU support |
| Weight format | BMTS (custom) | Zero-copy DMA, no filesystem |
| Build system | Nix | Reproducible, immutable closures |
| Network boot | iPXE + dnsmasq | Standard, well-documented |

---

## 5. Implementation Phases

### Phase 0: Foundation (Weeks 1-4) — Partially Done

**Goal:** Working cluster probe + shell on existing Rust codebase.

| Step | Status | File | Description |
|------|--------|------|-------------|
| 0.1 | ✅ | `beast/mod.rs` | Beast serializer/deserializer |
| 0.2 | ✅ | `scheduler/mod.rs` | Scheduler with energy budget |
| 0.3 | ✅ | `scheduler/workload_class.rs` | Workload classification |
| 0.4 | ✅ | `transport/mod.rs` | Transport abstraction |
| 0.5 | ✅ | `probe/mod.rs` | Probe modules (CPU, memory, energy) |
| 0.6 | ✅ | `shell/src/` | Shell REPL with dot notation (parser, formatter, propositions, agent_client) |
| 0.7 | ✅ | `agent/src/` | Node agent daemon (TCP 9500, telemetry, heartbeat, tasks) |
| 0.8 | ✅ | `probe/network.rs` | ICMP + TCP ping/pong RTT measurement |
| 0.9 | 🔲 | `probe/gpu.rs` | GPU probe (adapt from VITRIOL) |
| 0.10 | 🔲 | `beast/topology.rs` | Add GPU fields to NodeEntry |

**GPU fields to add to NodeEntry:**
```rust
pub has_cuda: bool,
pub vram_mib: u64,
pub compute_cap: String,
pub pcie_gen: u32,
pub pcie_width: u32,
```

### Phase 1: BitNet Inference Engine (Weeks 5-8) — Mostly Done (2026-08-29)

**Goal:** Single-node BitNet inference working, then pipeline across nodes.

| Step | File | Description |
|------|------|-------------|
| 1.1 | ✅ `bitnet-cpp/` | BitNet.cpp submodule (isHuangXin/llama.cpp fork) |
| 1.2 | ✅ `bitnet-rs/` | bindgen FFI + safe wrapper; static-first linking |
| 1.3 | ✅ `workload_class.rs` | `LlmInference` class added (needs AVX/AVX2) |
| 1.4 | 🔲 `probe/gpu.rs` | GPU detection: `has_cuda`, `vram_mib` |
| 1.5 | ✅ `tools/bench_bitnet.sh` | Benchmark: ~4 tok/s @4 threads (i7-3770, no AVX2) |
| 1.6 | 🟡 `cluster/pipeline.rs` | ACTS frames + PipelinePlan + agent load_shard/acts_echo done; stage forward pending |
| 1.7 | 🔲 `transport/raw_l2.rs` | Custom L2 protocol with EtherType 0x88B5 |
| 1.8 | 🔲 `transport/jumbo_frame.rs` | MTU 9000 frame implementation |

**BitNet.cpp integration:**
```rust
// FFI bindings (simplified)
extern "C" {
    fn bitnet_init(model_path: *const c_char) -> *mut BitNetContext;
    fn bitnet_decode(ctx: *mut BitNetContext, tokens: *const i32, n_tokens: i32) -> BitNetResult;
    fn bitnet_free(ctx: *mut BitNetContext);
}
```

**Pipeline protocol (TCP MVP):**
```
[Frame Header]
  - magic: u32 = 0x4F55524F ("OURO")
  - version: u8 = 1
  - frame_type: u8 = (ACTIVATION | WEIGHT_UPDATE | HEARTBEAT | CHECKPOINT)
  - sequence: u32
  - payload_len: u32

[Activation Frame]
  - token_position: u32
  - layer_start: u32
  - layer_end: u32
  - tensor: [f16; 4096]  // 8KB for 8B model
```

### Phase 2: Pipeline Parallelism (Weeks 9-12) — started early

**Goal:** Multi-node LLM inference with pipeline parallelism.

| Step | File | Description |
|------|------|-------------|
| 2.1 | ✅ `tools/shard_model.py` | Writes byte-exact .bmts shards (30L/2B -> 3 nodes) |
| 2.2 | 🔲 `cluster/weight_dist.rs` | Weight distribution protocol (scp today) |
| 2.3 | ✅ `cluster/src/infer/` | **Pure-Rust forward validated: cos 0.99998 vs llama.cpp, top-1 match**, 1.6s/tok release-MT |
| 2.4 | ✅ (in infer) | Per-layer KV owned by stage, never on wire |
| 2.5 | `cluster/error_recovery.rs` | Node crash detection + restart |
| 2.6 | `cluster/checkpoint.rs` | Pipeline state checkpointing |
| 2.7 | `cluster/hot_swap.rs` | Weight update without restart |

**Stage execution design (BMTS runtime, step 2.3)**

llama.cpp's C API executes the full graph only — no partial-model entry point.
Rejected: patching graph-splitting into the fork (upstream churn, two forks to
maintain). Chosen: **pure-Rust forward over BMTS shards** (`cluster/src/infer/`):

1. `Tq1Block` — TQ1_0 dequant (superblock 256: packed trit qs + scales), scalar first, AVX2 later.
2. `run_stage(shard, x[2560]) -> x'[2560]` per owned layer: RMSNorm -> GQA attention
   (20 q / 5 kv heads) with `attn_sub_norm` sub-quadrant scaling -> FFN (gate/up/down,
   SiLU) with `ffn_sub_norm` — the b1.58 sub-quadrant norms are non-standard; match them exactly.
3. Orchestrator keeps `token_embd` (n1) + tied `lm_head` (same tensor, transposed)
   and loads vocab via `vocab_only=true` in llama.cpp — tokenizer stays borrowed, weights do not.
4. Each stage owns KV cache for its layers (local, never on the wire).
   Wire carries only hidden state: ACTS frame = 10.2 KB/token/hop -> ~20 KB/token over 1GbE: trivial.
5. Verification ladder (contract-first):
   a. in-process 3-stage pipeline vs full llama.cpp model: same greedy tokens, cosine > 0.999 per stage output;
   b. same over localhost TCP + ACTS;
   c. real hardware (Alpha R2 + ThinkPad + Kria A53, TL1 kernel path there).
6. Speedup model: single-stream latency scales with layers/stage (~3x on 3 nodes);
   micro-batched tokens-in-flight add throughput later.

Effort: dequant + scalar forward + tied head ≈ 600-900 LoC + per-layer test vectors
(dump reference activations from the controlled fork build with an env-gated writer).

**Model sharding strategy:**
```
Model: 2B params (1.1GB in 1.58-bit)
  → Single node, no sharding needed

Model: 8B params (~2GB in 1.58-bit)
  → 2 nodes: Layers 1-16, Layers 17-32

Model: 40B params (~10GB in 1.58-bit)
  → 4 nodes: Layers 1-10, 11-20, 21-30, 31-40
  → Each node needs ~2.5GB RAM for weights + working memory
```

**Pipeline error handling:**
```
Node Crash Detection:
  - Heartbeat timeout (3 seconds)
  - TCP connection reset

Recovery:
  1. Master detects node failure
  2. Master halts pipeline
  3. Master re-calculates layer distribution
  4. Master streams new weights to surviving nodes
  5. Master resumes pipeline from last checkpoint
  6. Total recovery time: < 5 seconds
```

### Phase 3: Network Boot + Deployment (Weeks 13-16)

**Goal:** Diskless worker nodes with automatic deployment.

| Step | File | Description |
|------|------|-------------|
| 3.1 | `nixos/dnsmasq.nix` | dnsmasq PXE configuration |
| 3.2 | `tools/build_ipxe.sh` | iPXE binary compilation |
| 3.3 | `nixos/worker-pxe.nix` | Worker PXE boot config |
| 3.4 | `storage/bmts_format.rs` | BMTS format parser |
| 3.5 | `cluster/registration.rs` | Worker registration protocol |
| 3.6 | `cluster/weight_push.rs` | Dynamic model update |

**dnsmasq configuration:**
```nix
# nixos/dnsmasq.nix
{ pkgs, ... }:
{
  services.dnsmasq = {
    enable = true;
    settings = {
      dhcp-range = "192.168.1.0,proxy";
      dhcp-match = "set:ipxe,175";
      dhcp-boot = "tag:!ipxe,undionly.kpxe";
      dhcp-boot = "tag:ipxe,http://192.168.1.50/menu.ipxe";
      enable-tftp = true;
      tftp-root = "/var/lib/tftpboot";
    };
  };
}
```

**iPXE boot menu:**
```
#!ipxe
set base http://192.168.1.50

menu OurobourOS Boot
item worker Worker Node
item master Master Node
item memtest Memtest86+
choose --timeout 30 target && goto ${target}

:worker
kernel ${base}/ouro-worker
boot

:master
kernel ${base}/ouro-master
boot
```

### Phase 4: FPGA Interconnect (Weeks 17-24, Optional)

**Goal:** Kria K26 as high-speed interconnect node.

| Step | Description |
|------|-------------|
| 4.1 | K26 carrier board design for dual PCIe endpoints |
| 4.2 | NTB bridge logic in Verilog/VHDL |
| 4.3 | Memory-mapped BAR translation between hosts |
| 4.4 | Hardware doorbells for interrupt notification |
| 4.5 | Scratchpad registers in BRAM for synchronization |

**This is the hardest phase.** Only pursue if 1GbE/2.5GbE pipeline performance is insufficient.

### Phase 5: Briev Integration (Weeks 25+, When Compiler Stabilizes)

**Goal:** Rewrite critical paths in Briev for contract-enforced safety.

| Step | Description |
|------|-------------|
| 5.1 | Port network polling loop to Briev |
| 5.2 | Port DMA memory management to Briev |
| 5.3 | Port scheduler to Briev |
| 5.4 | Compile to freestanding x86_64/aarch64 via LLVM |

**Depends on:** Briev LLVM backend completing freestanding target support.

---

## 6. BitNet.cpp Integration Details

### 6.1 Build Requirements

```
Python >= 3.9
CMake >= 3.22
Clang >= 18
```

### 6.2 Supported Kernels

| Kernel | Platform | SIMD | Use Case |
|--------|----------|------|----------|
| I2_S | x86_64 + ARM64 | Portable baseline | Fallback |
| TL1 | ARM64 only | NEON | Kria K26 ARM cores |
| TL2 | x86_64 only | AVX2 | Alienware, old desktops |

### 6.3 Performance Expectations

| Platform | Speedup vs FP16 | Energy Reduction |
|----------|----------------|-----------------|
| ARM CPUs | 1.37x - 5.07x | 55.4% - 70.0% |
| x86 CPUs | 2.37x - 6.17x | 71.9% - 82.2% |

### 6.4 Limitations

- No AVX-512 support in official kernels
- Custom GGUF format incompatible with stock llama.cpp
- Models must be trained from scratch (cannot quantize existing models)
- Pre-AVX2 x86 CPUs have bugs and under-tested paths

---

## 7. Network Protocol Design

### 7.1 Raw L2 Protocol (Phase 2)

```
[Ethernet Header]
  - Destination MAC: 6 bytes (hardcoded next-hop)
  - Source MAC: 6 bytes (current node)
  - EtherType: 2 bytes (0x88B5 = IEEE Local Experimental)

[OurobourOS Header]
  - Magic: 4 bytes (0x4F55524F = "OURO")
  - Version: 1 byte
  - Frame Type: 1 byte (ACTIVATION=1, WEIGHT=2, HEARTBEAT=3, CHECKPOINT=4)
  - Sequence ID: 4 bytes
  - Token Position: 4 bytes
  - Layer Start: 2 bytes
  - Layer End: 2 bytes
  - Payload Length: 4 bytes

[Payload]
  - Activation tensor: [f16; 4096] = 8192 bytes (for 8B model)
  - Or weight shard: variable size

[Frame Check Sequence]
  - CRC32: 4 bytes (verified by NIC hardware)

Total Frame: 8222 bytes (fits in 9000-byte Jumbo Frame)
```

### 7.2 MAC Address Assignment

```
Master Node:     00:00:00:00:00:01
Worker Node 1:   00:00:00:00:00:02
Worker Node 2:   00:00:00:00:00:03
Worker Node 3:   00:00:00:00:00:04
Broadcast:       FF:FF:FF:FF:FF:FF
```

### 7.3 Discovery Protocol

```
[Announce Frame]
  - Type: 0x01
  - MAC: source MAC
  - CPU: model name
  - RAM: total MiB
  - GPU: model, VRAM, compute capability
  - SIMD: AVX2, AVX, SSE4.2
  - Layers: assigned layer range

[Pipeline Map Frame]
  - Type: 0x02
  - Node Count: u8
  - Ordered MAC List: [MAC; N]
  - Layer Assignments: [(start, end); N]
```

---

## 8. Weight Format (BMTS)

**Status: v1 IMPLEMENTED** — `cluster/src/bmts.rs` (Rust read/write) + `tools/shard_model.py` (GGUF -> per-node shards).

### 8.1 v1 File Layout (implemented)

```
[magic]      u32  0x4F55524F ("OURO")
[version]    u16  1
[node]       u16  node index, 1-based
[n_tensors]  u32
[meta_len]   u32
[meta]       JSON tensor table: [{name, shape, dtype, offset, length}]
             (offset = bytes into data section)
[data]       concatenated raw tensor bytes
```

Verified on `bitnet-2b-tq1_0.gguf` (332 tensors, 30 layers) split 3 ways:
node 1 layers 0..9 = 803.8 MB (carries 656 MB f16 `token_embd`),
nodes 2-3 = 147.1 MB each; 1098.1/1105.9 MB accounted; residual = GGUF header + padding.

### 8.2 Activation Wire Format (ACTS v1, implemented)

`cluster/src/pipeline.rs`: 26-byte header (magic, ver, type, seq, token_pos,
layer_start, layer_end, n_elems) + f32 payload. `PipelinePlan` parses
`shard_map.json` into stage specs.

MVP transport is hex-in-JSON over the existing agent task channel (newline
delimited). Measured on localhost: 2560-dim frame (10 KB, 20.5 KB hex)
round-trips in ~5 ms. **v2 direction:** raw binary framing (or L2, section 7)
drops the 2x hex tax; 4 KB sector alignment for O_DIRECT deferred to v2.

### 8.3 Tuning findings (2026-08-29, i7-3770 dev box)

- TQ1_0 model runs on the native ggml path; TL2 LUT kernels NOT required
  (and their codegen is broken against the fork's current API).
- OpenMP oversubscription: threads=8 (all hw-threads) measured 5.8x SLOWER
  than threads=4 (physical cores). Default all inference threads to physical
  core count. `tools/bench_bitnet.sh` auto-detects.
- Greedy sampling on the 2.4B base model loops ("a small city, and...");
  temp=0.8 + top-k 40 + top-p 0.95 removes loops.
- Generation ~4 tok/s/node with two agents co-resident; E2E shell ->
  agent -> model -> shell proven over TCP.

---

## 9. Security Considerations

### 9.1 Raw L2 Risks

Raw L2 Ethernet has no authentication. Any device on the same switch can:
- Read all traffic
- Inject malicious frames
- Impersonate nodes

### 9.2 Mitigations

| Risk | Mitigation |
|------|------------|
| Eavesdropping | MAC whitelist on switch (if managed) |
| Frame injection | HMAC in frame header (shared secret) |
| Impersonation | Node registration with public key |
| Weight tampering | SHA-256 hash verification of weight shards |

### 9.3 HMAC Header Extension

```
[OurobourOS Header with HMAC]
  - Magic: 4 bytes
  - Version: 1 byte
  - Frame Type: 1 byte
  - Sequence ID: 4 bytes
  - HMAC-SHA256: 32 bytes (over header + payload)
  - ... rest of header
```

---

## 10. Monitoring and Observability

### 10.1 Metrics to Track

| Metric | Source | Interval |
|--------|--------|----------|
| Token throughput | Pipeline orchestrator | Per-token |
| Node latency | Heartbeat | 1s |
| Power draw | Intel RAPL | 1s |
| Temperature | /sys/class/thermal | 1s |
| GPU utilization | nvidia-smi | 5s |
| Memory usage | /proc/meminfo | 5s |
| Network errors | NIC counters | 5s |

### 10.2 Health Dashboard

```
> cluster.health?
CLUSTER HEALTH
  Nodes:     4/4 ONLINE
  Pipeline:  ACTIVE (token 1,247)
  Throughput: 12.3 tok/s
  Power:     89W / 500W (82% headroom)
  Errors:    0

> n2.health?
NODE_2 HEALTH
  Status:    ONLINE
  Pipeline:  ACTIVE (layers 13-24)
  Latency:   0.3ms
  Power:     28W
  Temp:      62°C
  Errors:    0
```

---

## 11. Recommended Next Steps

1. **Finish Phase 0:** Shell REPL + node agent. Foundation everything builds on.
2. **Integrate BitNet.cpp:** Single-node inference first, then pipeline.
3. **Skip Unikraft for now:** Use CachyOS with minimal setup.
4. **Skip FPGA for now:** 1GbE is sufficient for pipeline parallelism.
5. **Start with bitnet-2b:** 1.1GB model in workspace. Prove pipeline before scaling.

### Immediate Actions (2026-08-29 status)

| # | Action | Status |
|---|--------|--------|
| 1 | Shell REPL | ✅ + live telemetry cache, `generate`/`shards`/`save`/`load`/`deploy` |
| 2 | Node agent daemon | ✅ + bitnet_generate / load_shard / acts_echo tasks |
| 3 | GPU probe (VITRIOL adapt) | 🔲 next |
| 4 | GPU fields in NodeEntry | 🔲 blocked on 3 |
| 5 | BitNet.cpp submodule | ✅ (TL2 kernels unneeded — TQ1_0 native path) |
| 6 | Rust FFI wrapper | ✅ bindgen + safe BitNetModel (static-first) |
| 7 | Single-node benchmark | ✅ ~4 tok/s i7-3770; `tools/bench_bitnet.sh`; 5.8x oversubscription finding |
| 8 | Pipeline POC (TCP) | 🟡 framing/plan/transport-probe ✅; Rust stage forward = next big item (section 8.3) |

## 12. Open Decisions

| # | Decision | Options | Recommendation | Status |
|---|----------|---------|---------------|--------|
| 1 | Master node | One laptop vs main PC | Main PC | Decided |
| 2 | Network topology | Switch vs direct | Switch | Decided |
| 3 | IP addressing | DHCP vs static | Static | Decided |
| 4 | Node agent auth | SSH key vs mTLS | SSH key (MVP) | Decided |
| 5 | Beast format | Text S-exprs vs binary | Text (MVP) | Decided |
| 6 | Weight format | BMTS vs GGUF | BMTS for bare-metal | Decided |
| 7 | Transport MVP | SSH vs TCP | TCP | Changed from SSH |
| 8 | Worker OS | NixOS vs CachyOS | CachyOS (GPU support) | Changed from NixOS |
| 9 | Unikraft use | All nodes vs CPU-only | CPU-only workers | New decision |
| 10 | Briev timeline | Now vs later | Later (compiler not ready) | New decision |

## 13. The Qwen Program (2026-08-29) — THE SUMMIT

**Thesis statement:** a 27B-class model, bigger than any single card in the
cluster, streamed across four scrap GPUs by OurobourOS's own machinery,
under a power budget, governed by the shell.

### 13.1 Cluster Hardware (4 GPUs, 32 GB VRAM)

| Node | GPU | VRAM | Arch | Driver | Role |
|------|-----|------|------|--------|------|
| Master (this box, CachyOS) | RTX 3060 LHR | 12 GB | sm_86 | r610 live | orchestrator + biggest stage + lm_head |
| Master (second card) | GTX 1070 Ti | 8 GB | sm_61 | **needs r580** | stage |
| IdeaPad/ThinkPad slave | GTX 1080 | 8 GB | sm_61 | r580 | stage |
| Alienware Alpha R2 slave | GTX 960 | 4 GB | sm_52 | r580 (last Maxwell branch) | small stage |

**Driver finding:** r610 dropped Pascal/Maxwell (1070 Ti invisible to
nvidia-smi despite being bound). **r580 is the only branch covering
Maxwell -> Ampere** -> single driver cluster-wide. CUDA 12.8 arch list:
`52;61;86`. CUDA 13+ = dead end for the 960 -> our GPU kernels go **wgpu/
Vulkan compute** (portable to all four).

### 13.2 Target: Qwen3.8-27B (released 2026-08-14, Apache 2.0)

- Dense 27.78B (LM 27B + vision), hidden 5120, **64 layers**, vocab 248,320,
  **untied** lm_head, native 262K context, MTP heads.
- Layout: `16 x (3 x (GatedDeltaNet -> FFN) -> 1 x (GatedAttention -> FFN))`
  = 48 linear-attention + 16 softmax-attention layers.
- Gated Attention: 24 Q / 4 KV heads, head_dim 256, **partial RoPE (64 dims)**.
- Gated DeltaNet: 48 V / 16 QK heads, head_dim 128, constant-size state.
- FFN: SwiGLU, intermediate 17,408.
- **Vision tower skipped** (text-only).

**Quantization budget (Q4_K_M):** weights ~15.4 GB + lm_head ~0.64 GB +
DeltaNet state ~0.6 GB + attention KV ~0.06 GB/1K-tok -> **fits 32 GB
comfortably, fits no single card -> the pipeline IS the product.**

**Linear-attention gift:** constant per-layer state (no KV growth) makes
262K context ~1 GB -> old cards + long context finally compatible.
State lives on its stage; only hidden activations (20 KB/token) cross the wire.

**Fast early win:** Qwen3.6-35B-A3B MoE (19 GB total, ~3B active) -> high
tok/s achievable on this cluster before wgpu matures.

### 13.3 Compute-Weighted Layer Packing

Sequential pipeline: slowest stage sets tok/s. Pack by compute share, not
equal layers, subject to VRAM: 960 (2.2 TF) ~4 layers; Pascals (9-11 TF)
~17 each; 3060 (13 TF + tensor) ~26 layers incl. lm_head. Plan generator
reads per-node probe (FLOPS class + VRAM) and emits stage -> [layer list].

### 13.4 Milestones & Acceptance Contracts

| Milestone | Gate (contract) |
|-----------|-----------------|
| M1: big-model forward in pure-Rust on CPU | ✅ **9B achieved** (all-9-tensor layer-0 diff cos≥0.9998; full-logit cos 0.9994, top-1 match). 27B pending RAM/swap or r580 box |
| M2: bridge benchmark — llama.cpp CUDA tensor-split 27B on 3060+1070 Ti (post r580) | reference tok/s bar recorded (est. 10-15) |
| M3: model alive across chassis, CPU-mode | **architecture proven**: 9B over 4 localhost TCP agents == in-process stream exactly; remaining work is physical wiring |
| M4: wgpu GPU stages | **dense 27B >= 10 tok/s; 35B-A3B >= 30 tok/s**; shell reports W/token, budget never exceeded |

### 13.5 Remaining Work (full)

| # | Item | Where |
|---|------|-------|
| 1 | Q8_0/Q4_K (+later Q5_K/Q6_K, IQ) dequant + fused dot kernels | `cluster/src/infer/` |
| 2 | `StageExecutor` trait: swappable CPU-MT / wgpu backend | `cluster/src/infer/` |
| 3 | `ArchSpec`: sharder emits model card (hparams, family, layer types); graph templated by family (bitnet2b | qwen3.8 | llama-fam | qwen-moe) | tools + infer |
| 4 | **GatedDeltaNet recurrence kernel** (gated delta rule, per-head state matrix) | `infer/ops` |
| 5 | GatedAttention variant: head_dim 256, partial RoPE, QK-norm | `infer/ops` |
| 6 | Rung B TCP: agent `stage_setup`/`stage_step` (holds Stage+KV per shard), `ouro-pipeline` orchestrator binary | agent + bin |
| 7 | `probe/gpu.rs` (nvidia-smi csv, VITRIOL libvitriol/probe.rs to adapt) + NodeEntry GPU fields + scheduler VRAM/compute-aware packing | cluster |
| 8 | wgpu compute backend: TQ1_0 first, then Q4_K gemv + delta state ops | new crate |
| 9 | Shard shipping in `deploy.`: resume + checksum (15.4 GB over 1GbE ~25 min one-time) | shell/tools |
| 10 | llama.cpp (fork has fused GatedDeltaNet; verify/refresh vs upstream) as test oracle only | tests |
| 11 | Your hands: r580 on master (2 cards up), flash slaves (CachyOS minimal), wire LAN | - |
| 12 | MTP speculative decoding head (throughput bonus) | later |

**Non-goals (explicit):** training 20B+ BitNet (compute reality: ~10^22 FLOPs vs
cluster ~50 TFLOPS), FPGA interconnect, Unikraft until CPU stages prove out.

## 14. Heterogeneous Weight Placement (2026-08-29)

**Principle:** every weight is computed where it is *most easily calculated*.
The cluster does not assign layers to machines — it assigns **operations to
devices**, within the limits physics imposes.

### 14.1 The Evidence: real 27B quant mix

`Qwen3.8-27B-Q3_K_M.gguf` (parsed from disk): 866 tensors —
213× Q3_K (FFN gate/up, attn_gate, embeddings), 189× Q4_K (delta out-proj),
6× Q5_K (attention qkv + FFN down — the important ones), 1× Q6_K
(204 MB lm_head), 456× f32 vectors (SSM alpha/beta/dt, conv1d, norms;
KB-sized). Each layer = ~4 heavy matrices + ~10 tiny ops + recurrent state.
This is not one workload. It is three, wearing one model's clothes.

### 14.2 Placement math (decode, batch-1 = bandwidth-bound)

Every token reads every weight once; arithmetic intensity equalizes across
quant types (~2-4 MAC/byte) -> cost ~ bytes/token / device-GB/s:

| Device | ~Bandwidth | ~Watts | GB/s/W | Sweet spot |
|--------|-----------|--------|--------|-----------|
| RTX 3060 (master) | 360 GB/s | 170 W | 2.1 | heaviest layers + 204 MB lm_head |
| GTX 1080 (slave) | 320 GB/s | 180 W | 1.8 | heavy layers |
| GTX 1070 Ti (master) | 256 GB/s | 120 W | 2.1 | heavy layers |
| GTX 960 (slave) | 112 GB/s | ~120 W | ~1.0 | light layers or control plane |
| i7-3770 CPU | 20 GB/s | 45 W | 0.45 | f32 micro-ops (conv, gates, norms) — us each |

Spec numbers are placeholders: agents MEASURE bandwidth at registration
(memcpy GB/s probe), and the partitioner consumes measured profiles only.

27B = ~13.8 GB of weight reads per token. Aggregate GPU bandwidth ~1 GB/s/W:
theoretical ceiling ~75 tok/s; realistic M4 target **30 tok/s**. CPU-only:
~1.4 tok/s. The f32 SSM tensors (<2 MB total/layer) cost GPU launch overhead
(10-20 us) exceeding their compute, but run in us on the host CPU.

### 14.3 Two regimes (the honesty constraint)

- **Within one box** (CPU + 2 GPUs on master): per-tensor routing is nearly
  free — activations cross PCIe in us. GPUs eat big matvecs; CPU runs
  norms/gates/conv; recurrent state never leaves host RAM.
- **Across boxes** (1 GbE): each cut costs one 20 KB activation hop ~160 us.
  Per-tensor freedom across LAN = ~3000 hops/token = death. Cross-machine
  placement = contiguous layer groups, few cuts (classical pipeline).

### 14.4 The design: compile the model to the topology

`PlacementPlan` — static artifact computed from (model card + measured device
profiles), versioned like BMTS:
- per-tensor device assignment within each machine
- group cuts between machines (min-cut over byte-weighted op graph)
- per-cut activation buffer specs (ACTS frames carry them)

Algorithm: sort ops by bytes desc; greedy-place by GB/s-per-watt under VRAM
capacity; merge to minimize cuts; lm_head pinned to fastest card.

**The OS trick:** `budget 120w.` re-partitions *between requests*. Heavy
tensors migrate toward efficient devices, the 960 idles out, tok/s moves,
watts obey. The cluster recompiles itself to the power budget.

Objective: Pareto (tok/s vs W/token) with contracts per run. Default:
maximize tok/s under energy constraint.

MoE extension (later): same problem — router picks experts per token;
experts placed by aggregate touch-rate; hot experts replicated on the
highest-bandwidth device.

### 14.5 Plan amendments

| Was | Becomes |
|-----|---------|
| `StageExecutor` trait | **`OpExecutor`** trait: per-tensor op dispatch (CpuMt today, wgpu next) |
| QuantKind Tq1/Q8/Q4K | + Q3_K (84 B/blk), Q5_K (110 B), Q6_K (210 B); then IQ4_XS/IQ3_* — each with C-parity bit tests |
| Fixed packing [4,17,17,26] | partitioner output from measured profiles |
| Telemetry: power/temp/load | + measured memory bandwidth per device class |
| New qwen35 ops needed | gated delta-rule state update, causal conv1d(4), partial MRoPE (sections [11,11,10], base 1e7), double norms |
| Milestones | M1 gains ladder rung: **9B-Q6_K (7.5 GB) fits one 3060** = first single-GPU sanity run before 27B multi-card |

### 14.6 Open decisions

1. llama.cpp oracle for qwen35/delta arch (fork is Feb-2026; model is Aug) —
   which build ran the user's downloads?
2. Objective exposure: per-run contract (tok/s floor, W ceiling) — agreed.
3. 960 role: light stage member vs control plane — decide after M3 measurement.

### 14.7 HDMI Video Modem Transport (GPU-to-GPU over display cables)

**Question:** can we write GPU kernels to allow GPU-to-GPU contact over HDMI?
**Verdict: yes, with one hardware truth — GPU HDMI ports are TMDS
*transmitters* only. No receiver exists; wiring two outputs together fights
drivers. Put a capture receiver on the far end and the HDMI stream becomes a
bit-exact digital data link. This is a video modem, not an analog modem:
TMDS 8b/10b arrives error-free over quality cable <= 3 m.

**Receiver options (verified 2026-08-29):**
- MacroSilicon MS2130 USB3 UVC stick (~R250): HDMI-in up to 4K30,
  uncompressed **YUY2 1080p60 out = ~230 MB/s per direction**, driverless
  Linux UVC. YUV *conversion* is lossy (255,0,0 -> 135,86,54) but YUY2
  **passthrough preserves the Y channel per pixel -> one data byte per pixel**.
- YuzukiLOHCC-PRO (open-source MS2130+MS8003 board, loop-out) - buildable.
- Decklink-class PCIe capture: RGB444/4K60, 750-1500 MB/s, DMA near-VRAM.
- Kria K26 wildcard: the FPGA itself decodes TMDS (HP banks + DVI-RX shield),
  microsecond sync, no USB chain: GPU -> FPGA direct data-over-display.

**Kernel design:**
- TX (any GPU): encode payload into Y channel of a scanout framebuffer,
  driven via DRM/KMS atomic plane from VRAM (zero copy). Frame layout:
  `[preamble][seq][len][payload][Reed-Solomon parity]` across the pixel grid,
  1080p60 -> ~230 MB/s effective after ~5% framing/parity tax.
- RX (other box): V4L2 buffer -> wgpu deframe kernel (preamble search via
  compute shader, payload extract, RS correct) -> VRAM destination.
- MUST force source resolution == capture resolution (any scaler interpolation
  destroys bits) and HDCP off (plain Linux framebuffers normally negotiate
  unprotected; MS2130 keys are HDCP 1.x only).

**Link comparison (per direction):**

| Transport | BW | Latency | Cost |
|-----------|-----|---------|------|
| 1 GbE (existing) | 125 MB/s | ~160 us/hop | R0 |
| HDMI modem (MS2130 stick) | ~230 MB/s | 33-66 ms (frame pipeline) | R250/box |
| HDMI modem (RGB444/Decklink) | 750-1500 MB/s | 16-33 ms | R1500+ |
| used 10 GbE (honest alternative) | 1250 MB/s | ~5 us | R150/card |

**Role in the placement architecture:**
- NOT for per-token activation hops: 33 ms frame latency vs 160 us Ethernet.
  Pipeline hops stay on GbE/L2. The modem does not beat Ethernet where the
  pipeline is latency-bound.
- **Weight streaming on budget re-partition** (14.4): shifting ~3.5 GB onto a
  card takes ~28 s over GbE, ~10 s over HDMI modem, ~2 s over Decklink.
  `budget 120w.` recompiles become interactive; the OS can move the model
  between devices between requests without stalling the cluster.
- **MoE expert cold-fetch**, checkpoint replication, initial shard deploy
  (15.4 GB one-time: 110 s -> 60 s per box).
- Natural topology: **star downlink** - master's 3060 + 1070Ti HDMI-out feed
  capture sticks in the two slave boxes (230 MB/s each way down); 1 GbE
  carries the small return/control flow. Display cable = data plane,
  Ethernet = signaling.

**Implementation:** third backend behind `cluster/src/transport/` `Transport`
trait (alongside TCP MVP + planned raw_l2): `HdmiModem` transport + framing/RS
in a `ouro-modem` module; PlacementPlan gains per-edge link properties
(bandwidth, latency, direction). Effort ~3-4 focused days incl. a throughput
calibration rig. Risks: R1 HDCP negotiation (test first), R2 hidden scaler
paths, R3 EDID limits (the stick's own EDID caps at 1080p60 - fine).

**Why it belongs:** it is the thesis in hardware - ports everyone dismissed as
display-only, repurposed as data planes by an OS that owns the whole stack.
Prior art validates the class: Interocitor (GPGPU<->FPGA over SDI, 2015).

## 15. Prior Art & Lineage (verified 2026-08-29, provenance below)

**Purpose of this section:** every claim we make is checked against what the
world has already built — so we cargo-cult neither OS conventions nor our own
originality. Items marked **[unverified]** are recalled, not confirmed this
pass. Confidence in the six surviving deltas (15.9) rises *because* the
surrounding space is occupied by excellent work.

### 15.1 Distributed LLM inference on commodity hardware

| Work | Provenance | What it establishes | Status vs us |
|------|-----------|--------------------|--------------|
| llama.cpp RPC backend | github.com/ggml-org/llama.cpp/tree/master/tools/rpc | Official distributed path: `ggml-rpc-server` exposes devices; weights+KV split proportionally to device memory; `--tensor-split` override; worker-side tensor cache (`-c`); **RDMA auto-negotiated over RoCEv2 when libibverbs present**; documented "proof-of-concept, fragile and insecure" | Reinvented at tool level; their worker cache is our shard-deploy resume design; their insecurity is our contract-Article-10 opening |
| Distributed-inference field survey + benchmarks | localaimaster.com/blog/distributed-inference-local-ai (2026-02-26) | 70B Q4_K_M 2-node: **2.8 tok/s @1GbE, 6.1 @2.5GbE, 7.4 @10GbE, 7.6 @TB4**; pipeline carries only ~8-16 KB activations/token; cross-machine vLLM TP "unusable on 1GbE"; tool comparison table (RPC 6.1, exo 4.3, Petals 0.9 tok/s); verdict: "distributed inference on old GPUs is a Saturday afternoon project" | Sets our honest baseline: capability is commodity — see 15.9 |
| Multi-node guide | fungies.io/multi-node-local-llm-inference-guide-2026 (2026-07-01) | 10GbE = practical home minimum ($200 used ConnectX pair); PP cross-node + TP intra-node doctrine; heterogeneous mixing "performance limited by slowest card — isolate via PP" | Confirms §14.3 two-regimes independently |
| exo | cited in llms.blog/decentralized-llm-inference (2026-08-23) | **Ring memory-weighted partitioning** across heterogeneous fleets (Mac/Linux/GPU); mDNS/UDP-multicast zero-config discovery; heartbeat + **cluster re-benchmark on node churn** | Our PlacementPlan = its compute-weighted generalization; steal discovery + churn re-plan |
| Petals (NeurIPS 2023) | llms.blog + petals papers | Public volunteer swarm: Hivemind/libp2p DHT advertises layer blocks; latency-aware path construction; **8-bit activation quantization on the wire**; redundancy per block | Wire-activation quant = optional ACTS v2 mode for bad links |
| CrossPipe (2025) | arxiv.org/html/2507.00217 (Hoefler group, ETH) | Latency-aware pipeline schedules via **MILP-solver or greedy** over bandwidth/latency models; 33.6% faster than naive under cross-DC constraints; MoE shifts preference further to PP | Validates §14.4 partitioner as solved-science shape; adopt solver-then-greedy pattern |
| Decentralized inference survey | llms.blog (2026-08-23) | 160 syncs/token at 80 layers; PP = the only viable WAN strategy at ms latencies; KV loss on disconnect => re-evaluate prompt from scratch | Our per-stage KV = accepted risk; §2.6 checkpointing addresses it |

### 15.2 Display links as data links (the HDMI modem question)

| Work | Provenance | What it establishes | Status vs us |
|------|-----------|--------------------|--------------|
| hdmifiletransporter | github.com/MrDesjardins/hdmifiletransporter + docs.rs 1.0.0 (real transfer logged June 2025) | **Rust file-over-HDMI-to-USB-capture end to end.** Solved exactly our listed risks: captured frames are offset/scaled/overscanned/recompressed → **calibration ring with 3 finder patterns + affine registration**; per-frame CRC32 headers; density ladder (bw = 1bit/cell robust ↔ quantized 8 levels ↔ RGB); loop-and-retransmit reliability; a **planner** that searches (cell size × levels × fps) for fastest byte-exact config | Prior art for the channel; its registration ring, density ladder, and planner fold into our §14.7 design (15.10) |
| vdxpy / VDX | github.com/blackocean-tech/vdxpy | Commercial-grade framing: Reed-Solomon per frame, SHA256 verified, profiled per capture device — **sold for air-gapped defense/OT/financial transfers** (display channel as data diode) | Second use case for our modem: one-way data diode falls out free |
| hdmiFileTransfer | github.com/yesyesno8/hdmiFileTransfer (2025-04) | 720p@5fps QR-style, ~4.4 Mbps, 1px=1bit, no FEC — fragile PoC tier | Floor, not ceiling |
| MS2130 capture IC | doc.ultrasemi.com/en/ic/macrosilicon/ms2130.html | HDMI-in ≤4K30; USB3.0 UVC; **YUV422 or MJPEG USB-out, default max 1080p60** (RGB444 needs MS2130S/MS2131) | Our 230-250 MB/s/dir figures grounded; YUY2 Y-passthrough = data channel confirmed by IC capability list |
| YuzukiLOHCC-PRO | github.com/YuzukiHD/YuzukiLOHCC-PRO | Open-source MS2130 board with HDMI **loop-out** (monitor + capture on one cable path) | Enables the "console visible while carrying weights" mode (§14.7) |
| SDI IP cores | Microchip SDI_TX user guide; AMD/Xilinx XAPP1290 | 270 Mbps–12 Gbps uncompressed serial-video is **standard FPGA transceiver protocol** with scrambling/CRC framing in silicon | Video-serial data planes are turnkey on the Kria; no capture chip needed FPGA↔FPGA |
| Numato Opsis / hdmi2usb + FOSS DisplayPort core | hackaday.com 2015-10-02 (+comments: Mike Field's open DP core, 4K30 working) | Open-source FPGA implementations of display protocols exist (HDMI & DisplayPort PHY in free gateware) | Our modem TX/RX can be pure FOSS incl. PHY |
| Nyuzi GPGPU | hackaday.com 2016-03-30 + github.com/jbush001/NyuziProcessor | Fully open 32-bit VLIW GPGPU SoC on FPGA, runs Quake | Deep-time proof of "FPGA as GPU" in our own lineage |
| Vortex | MICRO'21, doi 10.1145/3466752.3480128 | RISC-V ISA extended for GPGPU; PCIe soft-GPU with OpenCL, up to 32 cores on Stratix-10/Alveo | Kria long game: carrier becomes a compute stage with its own ISA |
| "Interocitor" (SDI GPGPU-FPGA link, ~2015) | **[unverified]** — searches resolve to the 1949 novel and unrelated repos; original project not confirmed this pass | Recalled as 1.5 Gbps SDI cross-machine GPGPU memory link | Cited as possibly-myth; verified substitutes: APEnet+, FPGA² (15.7) |

### 15.3 Operating-system lineage (our school of thought)

| Work | Provenance | Core idea | Relation to Constitution |
|------|-----------|----------|--------------------------|
| **Exokernel** (Aegis/ExOS) | Engler/Kaashoek/O'Toole, SOSP'95 PDF (research.cs.wisc.edu mirror; JHU mirror) + Engler thesis 1998 (hdl.handle.net/1721.1/16713) | "Operating systems limit performance/flexibility by *policy they impose*; **securely multiplex physical resources and let application-level software implement abstractions**." Principles: expose hardware, expose names, expose events, **visible revocation**, fine-grained protection. Measured: primitives 10-100× cheaper, web server ~10× faster | This *is* Articles 2/7/10 with 30 years of citations. Our delta: (a) the "application" is one model's op-graph, (b) resources span **multiple chassis**, (c) watts enter the policy, (d) parity contracts gate every re-binding |
| **Multikernel / Barrelfish** | Baumann et al., SOSP'09 (barrelfish.org/publications) | Treat one machine as a **network of independent cores**; message-passing, state replication, hardware-neutrality across ISA-heterogeneous cores | Art. 3 inverted: they federated one box's cores; we federated whole boxes — same doctrine, one octave up |
| **OpenSSI** | Wikipedia (verified 2026-08-29) | Single-system-image clustering: Compaq 2001, lineage LOCUS (UCLA, early '80s) → Locus Computing Corp → UnixWare NonStop Clusters → Linux port | "PCs that consider themselves separate work as one machine" is a named, 40-year-old program — we cite the line, and note none of the SSIs optimized *tensor placement* |
| **Kerrighed** | Wikipedia + Lottiaux et al. CCGRID'05 comparative study (hal-01271223) | SSI with **process migration** over cluster (INRIA 1998-2012) | Ouroboros clause (Art. 4) has precedent for processes; moving the *scheduler itself* with the graph's authority (bootstrap-seed invariant) remains ours |
| **HTCondor** | Litzkow et al., ICDCS 1988; Thain et al. "Cheap cycles…" + "Distributed Computing in Practice" (htcondor.org) | Opportunistic cycle scavenging of idle desktops; **ClassAd matchmaking language**; checkpoint-migrate; preemptive-resume | Art. 9-item-5 (idle=reserved) ancestor; ClassAd = ready-made formalism for our PlacementPlan request/offer matching (adopt, 15.10) |
| **Beowulf** | Sterling/Becker et al., ICPP'95 (webhome.phy.duke.edu mirror); NASA history (ntrs.nasa.gov 20150001285); beowulf.org | 16 commodity 486 boards + **two channel-bonded Ethernets** ("the network, even in its dual configuration, is inadequate" — same finding, 31 yrs old); origin quote: "Cheap high-performance computing systems are virtually non existent… PC-compatible hardware is cheap and supports… Linux" | Our thesis is Beowulf's thesis for the GPU era; our bonded GbE + HDMI downlink is Becker's bonding move applied to *display ports* |

### 15.4 GPU-OS integration & removing the host OS from the critical path

| Work | Provenance | Finding | Relation |
|------|-----------|---------|----------|
| **Singularity** | arxiv.org/abs/2202.07848 | Device-proxy intercepts CUDA via LD_PRELOAD; GPU state decoupled from host address space → **transparent checkpoint/restore + live migration + time-slicing of GPU DNN jobs** (2-3% context switch overhead) | Live state relocation for *accelerated jobs* is production-proven; our Art. 4 (moving the orchestrator itself) extends beyond it |
| **GPUVM** | arxiv.org/abs/2411.05309 | GPU threads drive paging **through the NIC** (one-sided RDMA), host OS removed from critical path, 4× UVM | Direct precedent for Art. 6: measured hop-killing, not vibes |
| CUDA unified memory / HMM / ATS | docs.nvidia.com CUDA Programming Guide §2.6 | Industry converging on "all memory is one pool" from above (Grace-Hopper C2C hardware coherence) | We do it from below, on the cards the above-market rejects |
| **sched_ext** | docs.kernel.org/scheduler/sched_ext.html + github.com/sched-ext/scx | Mainline Linux: BPF schedulers with DSQ queues, verifier safety, **watchdog auto-revert to fair scheduler**; Meta/Google production use; shipped by gaming distros incl. CachyOS | Article 6's first sanctioned lever: `scx` for stage-host isolation is a config file away on the master today |

### 15.5 Energy-first scheduling

| Work | Provenance | Finding | Relation |
|------|-----------|---------|----------|
| Black-box energy-aware CPU/GPU partitioning | Barik et al., doi 10.1145/2854038.2854052 (PODS'16) | Power-model + workload profiling partitions kernels across CPU/GPU to 93-96% of oracle **energy-delay product** | Same optimization at core level; nobody runs it over a LAN fabric |
| Co-Cap | CECS-TR-15-05 (UC Irvine) | Coordinated CPU+GPU frequency capping, 10-23% energy/frame | Confirms: independent governors leave energy on table — same logic as independent *schedulers* |
| CGM-DVFS | MDPI Future Internet 14(3):91, 2022 | DVFS extended to **memory** too: +26% power, +21% thermal efficiency | When we control clocks we'll price memory rails too |

### 15.6 Ternary/low-bit inference (our model class)

| Work | Provenance | Finding | Relation |
|------|-----------|---------|----------|
| Microsoft BitNet repo | github.com/microsoft/BitNet | **Official GPU kernel released 2025-05-20** ("extending 1-bit inference beyond CPUs"); CPU kernels: x86 up to 6.17× speedup, 82.2% energy cut; ARM 5.07×/70%; **100B b1.58 model at 5-7 tok/s on ONE CPU** | The industry's answer to "big BitNet" is a monster single CPU — our counter: price/W of the cards that CUDA-13 orphans, and 100B weights remain research-only |
| bitnet.cpp (ACL 2025) | arxiv.org/abs/2502.11880 | TL1 (ARM), **TL2 (x86 LUT)**, I2_S (MAD, lossless); TL2_0 beats TQ1_0 1.33-1.65×; model-support table: 2B-4T x86 = I2_S+TL2 | Confirms our fork finding (TQ1_0 native runs fine; TL2 optional); our Rust parity path must eventually benchmark vs TL2 not just TQ1 |

### 15.7 PCIe peer-to-peer: "policy, not a wall" case study (Article 2's proof)

| Step | Provenance | Fact |
|------|-----------|------|
| 1. The lock | NVIDIA drivers refuse P2P on GeForce | every GPU-GPU byte staged through host RAM |
| 2. The crack | github.com/tinygrad/open-gpu-kernel-modules (cited in forks as 565.57.01-era patch, George Hotz) | hand-built BAR1 page-table aliases: **P2P shown to be driver policy** |
| 3. Simplification | github.com/aikitoria/open-gpu-kernel-modules (610.43.03) | clean port; `RMForceP2PType=1`; requires `iommu=pt` + **ACS override** ("ACS on root ports forces all GPU-to-GPU traffic through the CPU root complex") |
| 4. Productionization | github.com/QuixiAI/open-gpu-kernel-modules (Eric Hartford, 610.57.04) | 610 driver ships NVIDIA's own BAR1-P2P path, never selected on GeForce; one-commit force-enable over Turing/Ampere/Ada/Blackwell (open kernel module floor = **Turing**; Pascal/Maxwell stay locked out — legacy driver branch has no open modules). Verified: **NCCL all-reduce busbw 2.7 → 24.7 GB/s** (8×3090); needs BAR1 ≥ VRAM (ReBAR/resize dance) |
| 5. FPGA analogues | FPGA² (doi 10.1109/reconfig.2013.6732296): open-source **direct FPGA↔GPU DMA**, >5 GB/s, needed gdev/nouveau to even read GPU buffer physical addresses; APEnet+ (INFN): FPGA PCIe board, GPU BAR access, custom torus fabric, 34 Gbps/link, RDMA semantics in fabric | Direct-DMA bypass of host staging across vendors is a decade-old open practice |

**Consequence:** our master's 3060+1070Ti pair cannot use this unlock (mixed generations + Pascal lacks open modules + no P2P need at 1 GPU per box today) — but the case is *exhibit A* for Article 8's distinction and Article 2's program. Documented so future silicon (a pair of 3090s from the e-waste stream) activates a known path.

### 15.8 Optical/side channels (lowered expectations)

LiFi-class: OpenVLC ~150 Kbps @4m; LiFOD 400 Kbps (doi in TOSN; MCU-rate-limited, "FPGA could reach MHz/GHz"). Verdict: hobby optical = Kbps tier; interesting only as the audio-jack **clock-distribution** microsecond idea (no prior art found either way — [open probe], cheap to test, tiny reward).

### 15.9 What survives scrutiny — the six deltas

The capability ("run 70B across old machines") is **commodity**: llama.cpp RPC
does it tonight with no auth and no contracts. OurobourOS is differentiated by
the combination, none of which we found together in any system:

1. **Contract-gated re-placement** — no prior live-migration (exo churn,
   Singularity checkpoint, llama RPC split) re-verifies model *equivalence*
   (bit-identity parity ladder) after every hardware change. Contracts are the
   safety case for radical flexibility (Art. 10).
2. **Watts as a fabric-level recompile axis** — energy scheduling exists at
   DVFS/core-partition level; `budget 120w.` physically **relocating tensors
   between cards and regenerating PlacementPlan** we found nowhere.
3. **Online bonded affordance fabric** — HDMI-modem prior art is all
   *offline file transfer*; carrying a *running pipeline's* weight streams over
   bonded display+Ethernet edges chosen by measured per-flow price is new
   territory (Beowulf's bonding, applied to purpose-free ports).
4. **Ouroboros control plane** — SSI/Condor/exo migrate applications; the OS
   migrating **its own scheduler with bootstrap-seed invariant** we did not
   find.
5. **CUDA-orphan compute pooling** — bitnet.cpp's GPU kernel + vLLM/Triton
   ecosystems assume CUDA≥Turing/Volta-class or recent ROCm. Deliberately
   building the long tail of Maxwell/Pascal *because* the industry just
   deprecated them, via wgpu-first kernels: unclaimed ground.
6. **Ownership of the whole path** — display server, NIC queues, 1588,
   revocation, scanout DMA in one arbiter (Art. 3). Tools above the OS can't
   reach these levers; that is structural, not effort.

### 15.10 Adoption list (concrete, from provenance above)

1. **Modem v1 design inputs** (hdmifiletransporter/vdxpy): calibration ring +
   affine registration; per-frame CRC + sequence; Reed-Solomon option;
   density ladder (robust bw ↔ dense quantized); **encoding-speed planner**
   that searches (resolution×levels×fps) for max byte-exact throughput —
   fold all into §14.7 implementation.
2. **Worker-side tensor cache** (llama RPC `-c` pattern) → `deploy.` shard
   resume: hash BMTS tensor table per node, transfer only deltas.
3. **ClassAd-style match language** (Condor) → PlacementPlan v2 request/offer
   predicates (device ads: bw, vram, watts; op ads: bytes, MACs, cuts).
4. **Solve-then-greedy schedule search** (CrossPipe) for stage packing.
5. **RDMA path reservation**: ConnectX-3/4 used NICs (R150-400) would make
   llama-RPC bridge benchmarks RDMA-fast AND are graph edges for us
   (kernel-bypass, GPUVM-adjacent). Add to hardware shopping list as optional
   measurement instrument, not dependency.
6. **sched_ext (scx) lever** on master now: CachyOS ships it; stage-host
   cores under `scx` with pinned/busy-poll class = Art. 6's first priced hop-kill.
7. **8-bit activation wire mode** (Petals) as ACTS v2 flag for 2.5G-class
   links (halves activation bytes if we ever bottleneck).
8. **BitNet kernel target shift**: benchmark our Rust Q4/TQ1 path not just
   vs TQ1_0 but vs **TL2/I2_S** (their tables: 1.33-1.65× and lossless-1.58×
   expectations) so our contracts cite the current SOTA bar.

## 16. Build Schedule (approved 2026-08-29)

**Track R — Rung B, cluster-as-one-machine (FIRST, everything else rides it):**
1. Agent stage slot: `stage_setup|shard_path`, `stage_token|pos|id`,
   `stage_step|acts_hex` (ACTS carries pos; out-of-order rejected against
   kv.seq), `stage_sample|hidden_hex` (argmax over tied head, token id out —
   logits never cross the wire), `stage_reset`.
2. `ouro-pipeline` binary (shell crate): shard_map -> stage/node plan ->
   prefill tokenwise -> greedy generate -> token-id + hop-timing report;
   text via `tokenize|detok` tasks (agent's vocab_only-side bitnet slot).
3. Tests: synthetic 3-shard toy model (f32/tq1/f16, tiny dims) — TCP result
   == in-process PipelineModel exactly; `#[ignore]` real 3×TQ1 shards TCP ==
   rung-A greedy ids. Gate = token-id equality (text is convenience, not
   contract).
4. M3 readiness: same binary against LAN addresses when slaves arrive.

**Track Q — ladder to the 27B summit:**
5. cb_eval oracle harness (fork exposes `llama_context_params.cb_eval`) —
   validate dump mechanism on the proven TQ1_0 model first.
6. Q5_K + Q3_K + Q6_K dequant, C-parity bit tests (27B's real quant mix).
7. Shard Qwen3.8-9B-Q6_K (7.56 GB — fits this 16 GB box; 27B after r580).
8. ArchSpec: sharder emits model card (arch/dims/layer pattern/partial RoPE
   [11,11,10]/untied head); graph templated by family.
9. GatedDeltaNet + conv1d + gated attention ops — each differential vs
   cb_eval dumps (cos > 0.999), the discipline that caught everything so far.


## 16.1 Status (2026-08-29, end of session)

| Item | State |
|------|-------|
| Rung B agent stage tasks + ouro-pipeline | ✅ commits e70f7dc..81d96fc |
| Q1 oracle harness (cb_eval capture) | ✅ deterministic, 846/1492-node maps |
| Q2 kernels Q3_K/Q5_K/Q6_K | ✅ bit-exact vs C |
| Q3 9B model card + shards (vision/nextn filtered) | ✅ 4 stages, 7548 MB accounted |
| Q4 delta + gated-attn in Rust | ✅ layer-0 9/9 tensors + full 32-layer logits |
| M3-sim: 9B x 4 TCP agents == in-process | ✅ [17018, 7529, 998, 14541, 364] |
| Gates | 119 fast tests, clippy 0 |

Next: r580 driver (2 GPUs live here) → bridge benchmark M2; 27B forward
(its Q3_K/Q4_K/Q5_K mix already executes — needs swap-friendly loading or
the slave RAM); slave bring-up; scx + ClassAd + modem per §16 deferred list.

## 16.2 Build-Now Backlog (settled 2026-08-29 — this box, zero user action)

Order: **A → B → E → C → D → F** (prove-it-then-speed-it).

| # | Build | Success rationale | Gate |
|---|-------|-------------------|------|
| A | mmap shard loading → **27B full-model Rust differential** | The mountain itself runs on our kernels; page cache does the paging; oracle+Rust share clean pages of the same file | 27B logits vs llama.cpp: cos>0.999, top-1 equal |
| B | GPU probe (nvidia-smi/Vulkan) + NodeEntry vram/compute fields + scheduler ranking + `n1.gpu?` | 3060 live NOW; slaves join a GPU-literate graph | probe test vs recorded CSV + real 3060 |
| E | `tools/m2_bridge.sh` — pre-staged CUDA build + benchmark script (needs user `pacman -S cuda`) | One user command turns into M2 baseline + wgpu-priority data | produces PLAN results table |
| C | AVX1 fused dequant-dot (Q6_K/Q4_K hot loops) + head-parallel delta recurrence | measurable here (i7-3770 has AVX1); runtime-gated AVX2 later; kills "glacial CPU" objection | parity cos>0.999 vs scalar; tok/s reported |
| D | bring-up kit: `discover.` subnet sweep, `deploy --shards` (checksum+resume), `--packing w1,w2,..` weighted layers | hardware weekend becomes hours; kills hardcoded IPs | synthetic + loopback tests |
| F | CI: fast tier on commit, `--ignored` parity ladder nightly | the discipline becomes infrastructure | scripts/ci |

**Not now (by decision):** wgpu kernels (wait for E's numbers), HDMI modem,
chunked-delta prefill, 27B *throughput*.

## 17. GPU Substrate Findings & Corpus Synthesis (2026-08-30 research session)

### 17.1 The GPU floor under orphaned silicon — better than feared

| Finding | Provenance | Consequence |
|---|---|---|
| **NVK (Mesa FOSS Vulkan) conformant Vulkan 1.3 on Maxwell, Pascal, Volta** — default since Mesa 25.1 (Apr 2025, Faith Ekstrand/Collabora) | collabora.com/news-and-blog: "NVK enabled for Maxwell, Pascal, and Volta" | every card we own speaks modern Vulkan with zero proprietary help |
| NVK caveat: pre-Turing lacks GSP -> **stuck at boot clocks** (no reclocking) | same post | prefer NVIDIA proprietary ICD for perf; NVK = purity fallback + Plan B |
| NVIDIA proprietary provides **full Vulkan 1.2 on Maxwell-2/Pascal/Volta** | nvidia developer Vulkan support page (archived matrix) | r580 ICD meets wgpu's floor on all four cards |
| **`nvidia-580xx-dkms` exists, actively maintained** — AUR 580.178.04 (updated 2026-08-13, maintainer ptr1337/CachyOS) and **in CachyOS own repo** (580.173.02, built Jun 2026) | aur.archlinux.org/packages/nvidia-580xx-dkms; packages.cachyos.org | the driver question that gated GPU-first is a one-line pacman on every machine. Trap: avoid linux-lts (AUR note re 6.12.x) |
| Arch advisory: driver 590 dropped Pascal/Maxwell -> "switch to nvidia-580xx-dkms" | bbs.archlinux.org id=311143 | the whole cluster converges on ONE branch (r580 covers Maxwell->Ampere incl. the 3060) — **master must swap r610 -> 580xx to see the 1070 Ti** |
| **llama.cpp Vulkan scoreboard**: GTX 1080 Ti tg 67.8–71.6 t/s (L2-7B-Q4_0); 1070 Ti tg 42.9–43.4 (eGPU); 1070 tg 41–43 | github ggml-org/llama.cpp discussion #10879 (2026-era commits) | decode on Pascal desktop cards is 40–70 t/s on 7B — the M4 contract (30 tok/s on 27B across four) is bracketed by evidence |
| **GTX 1060: Vulkan tg BEATS CUDA tg** (90.6 vs 61.7 small; 28.1 vs 25.4 on 7B) | issue #19817 (2026-02) | on pre-tensor-core chips Vulkan is not consolation — it wins on decode. ggml reports Pascal `coopmat:none, int dot:1` -> general path is the hot path |
| 27B pipeline bound: spec aggregate ~1.04 TB/s (360+320+256+112); at ~45% eff. ÷ 13.8 GB/token | §14.2 math + scoreboard calibration | **~30–35 tok/s projected M4; plausible, not heroic** |

### 17.2 Deployment architecture (decisions taken)

- **Tiers**: T0 static musl agent + unit script (any Linux boots a node;
  qwen35 engine included — CPU stages proven) -> T1 dual-boot CachyOS-minimal
  recipe (R2: **single SATA bay — shrink existing disk, owner-approved**;
  IdeaPad: shrink NVMe; shared ESP; nvidia-580xx-dkms; headless boot =
  join-the-graph) -> T2 OurobourOS.iso (mkosi/archiso) post-M3.
- **Master btrfs** (CachyOS default `@,@home,@srv` on sda, 432 G free):
  `/srv/ouro/{repo,gen,shards}` subvolume tree; releases =
  `gen/<UTC>` + `current` symlink flip (atomic rollback);
  **qgroup fence proposed ~100 G** (subvolumes share the pool — capacity
  independence is a lie until qgroups exist); btrfs send deltas = LAN
  replication to slaves AND backup stream to sdb if owner elects.
- **sdb (931 G NTFS): owner's backup drive — hands-off, no reformat.**
- Honest limits recorded: subvolume independence = namespace/snapshot/
  replication depth only; fate (device, pool, power) is shared — mitigated
  by qgroup + send, not by naming.

### 17.3 What the corpus points at (integration, for future readers)

1. **Capability is solved; economics remain.** Every gate the industry put
   in front of this build — correctness, formats, drivers, Vulkan-on-old-
   NVIDIA, hybrid-attention ports — walked through this month. Remaining
   unknowns are speed-shaped, not possibility-shaped.
2. **The white space is the intersection.** Components all exist somewhere
   (§15): Exokernel's doctrine, multikernel's machine-as-network, Beowulf's
   economics, Condor's harvest, ggml-vulkan's kernels, video-modem hacks.
   Nobody binds them with a measured graph + contract-gated re-placement +
   watts as a recompile axis. That joint *is* the OS claim.
3. **The ecosystem is converging on our constraints.** Linear attention
   deleted KV from the wire (constant-state 48/64 layers); quantization
   shrinks frontier models into orphaned-VRAM envelopes; CUDA-13 keeps
   evicting old silicon, lowering our supply price and leaving Vulkan as
   the open door — which our architecture was built to walk through.
   Junk-tier models and junk-tier hardware are moving toward each other.
4. **The method is the artifact.** Differential-test-everything caught every
   real bug this project had (K_SCALE_SIZE, cursor, over-projection);
   provenance-check (§15) is the same instinct at project scale. Neither is
   decoration — both are the product's immune system.
5. **One measurement stands between blueprint and numbers**: Vulkan decode
   ceiling on our exact cards (llama-bench `-d vulkan -ngl 99`, §16.3
   Phase 0) -> wgpu Q6_K kernel (L1 rung) -> GPU-pipeline demo -> physical
   nodes. Everything else is machinery already built.

### 17.4 Standing decisions & pendings

| Item | State |
|---|---|
| GPU-first ordering | **DECIDED** (owner) |
| qwen35 in static agent | **DECIDED yes** |
| R2 disk shrink | **DECIDED OK** (single-bay chassis, recipe = gparted live) |
| VITRIOL llama-server kill for Phase 0 | **approved** (first act when executed) |
| Master driver swap r610 -> 580xx-dkms | **RESOLVED (2026-08-30)**: both GPUs live on 580.178.04 (1070 Ti enumerated). Caveat: CUDA 13.3 dropped sm_61 — 1070 Ti CUDA needs 12.x sidecar or Vulkan |
| qgroup size /srv/ouro | PENDING (proposed 100 G) |
| sdb usage for btrfs-send backup | PENDING (owner's drive) |
| ARCHITECTURE.md + docs/CONTRACTS.md | **written this session** — canonical |

### 16.3 M2 Bridge Results — CUDA on the 3060 (2026-08-30, llama-bench VITRIOL build)

| model | device | pp t/s | tg t/s | notes |
|-------|--------|--------|--------|-------|
| bitnet-2.4B TQ1_0 | CUDA -ngl 99 | 2.4 | **1.6** | **WORSE than our CPU (3.8).** llama CUDA has no efficient TQ1_0 gemv path; "bitnet GPU kernels" target I2_S/TL2, not TQ1. GPU story for ternary needs their kernels or ours |
| qwen35 9B Q6_K | CUDA -ngl 99 | 1249 | **22.9** | fits 11.9G; the bridge bar for our wgpu Q6_K kernel (L1 rung target >= ~15) |
| qwen35 9B Q8_0 | CUDA -ngl 99 | 1301 | 20.7 | +bytes -> -tg: bandwidth-bound law confirmed empirically |
| qwen35 27B Q3_K_M | CUDA -ngl 99 | — | — | **loader REFUSED: model > VRAM.** the single-card wall, literally enforced |
| qwen35 27B Q3_K_M | CUDA -ngl 25 | 40.7 | 1.10 | partial offload: CPU-dominated tail; our 4-card pipeline target 30-35 t/s now bracketed below (CPU 0.03-0.16) and above (1080Ti-class x4) |

Baselines for context: our Rust engine same box — 9B ~0.16 t/s, 27B ~0.03 t/s (CPU scalar+MT).
**Gap to close with wgpu: ~140x at 9B — that is Phase 3's entire purpose; the
22.9 t/s CUDA row on a 3060 (2021 card, $130) is the proof the GPU pool is
where efficiency lives.** Vulkan column: pending `vulkan-headers` install
(user action) + build-vk reconfigure already staged.

### 16.3b Vulkan bridge (same box, same binary: fork build-m2, CUDA+Vulkan, 2026-08-30)

| model | device | pp t/s | tg t/s |
|---|---|---|---|
| 9B Q6_K | **Vulkan0** | 1375.8 | **43.87** |
| 9B Q6_K | CUDA0 (same binary) | 1535.1 | 42.36 |
| 9B Q8_0 | Vulkan0 | 987.8 | 35.91 |
| bitnet-2.4B TQ1_0 | Vulkan0 | 4.2 | 2.0 |
| 27B Q3_K_M (-ngl 25) | Vulkan0 | 27.5 | 0.88 |

**Findings:**
1. **Vulkan tg BEATS CUDA on the 3060 (43.9 vs 42.4, same binary, same card)** —
   the GTX-1060 pattern generalizes to Ampere. Vulkan-first is not the
   fallback; it is the fast path. Wgpu session is fully justified.
2. **TQ1_0 is kernel-orphaned on every GPU backend** (CUDA 1.6, Vulkan 2.0
   vs our CPU 3.8). Ternary GPU path = I2_S repack or our own kernel.
3. 27B single-card partial (25/64 layers) is 0.9-1.1 t/s on either backend:
   the pipeline (>=30 tok/s contract) is the ONLY way 27B flies. Confirmed
   twice, by refusal (-ngl 99) and by crippled partial.
4. M4 arithmetic update: single 3060 does 43.9 t/s on 9B -> 4-card
   heterogeneous pipeline for 27B projected ~35-40 t/s (bytes-scaled +
   packing efficiency) -> contract comfortably plausible.
5. Anomaly RESOLVED (§16.3c): VITRIOL-tree CUDA measured 22.9 t/s on
   identical model; fork-tree CUDA measures 42.4. Decomposition: +17%
   toolchain (CUDA 13.3 rebuild), rest = VITRIOL's older ggml base kernels.
   Lesson: backend benchmarks MUST pin build provenance (§15 discipline
   applies to toolchains too).

### 16.3c Toolchain bisection — VITRIOL 2x anomaly (2026-08-30)

**Forensics (before any rebuild):**
- VITRIOL CUDA binary configured **Aug 18** with `/usr/bin/nvcc` — a path that
  **no longer exists**. Toolkit now `cuda 13.3.1-1` at `/opt/cuda` (nvcc 13.3).
- Fork `build-m2` configured Aug 30 with `/opt/cuda/bin/nvcc` (13.3).
- CMake flags otherwise **identical** (Release, FA=ON, GRAPHS=ON,
  FORCE_MMQ/CUBLAS=OFF). Only structural differences: VITRIOL arch list
  `61;86` vs fork `86`; VITRIOL ggml-cuda is an older base + its own CUDA
  patches (vitriol_copy_engine, vitriol-cuda-integration).
- So §16.3b's hypothesis ("CUDA 13.3 vs older") is now the **prime suspect**.

**Bisection protocol (cheapest first, one variable per step):**
1. `build-cu13/`: VITRIOL tree rebuilt with `-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`,
   own arch list `61;86` unchanged → isolates **toolchain**.
   Expect ~42 if toolchain was the whole story.
2. If still ~23: VITRIOL's ggml CUDA patches compiled out → isolates
   **patch cost** (copy-engine/DMA path on Ampere decode).
3. If still slow: ggml base age → rebase VITRIOL ggml onto newer upstream
   (server patches are separate files; contained risk).
4. `build-vk/` (Vulkan ON, already reconfigured Aug 30) built + benched —
   Vulkan is the proven-fast path (43.9) regardless of CUDA forensics.
5. Provenance recorded per row: nvcc path + version, commit, arch, flags.

**Results:**

| tree | commit base | nvcc/toolchain | backend | 9B Q6_K pp/tg | 9B Q8_0 pp/tg |
|------|-------------|----------------|---------|----------------|----------------|
| VITRIOL `build/` (ref, §16.3) | VITRIOL fork a3ee3be00 | `/usr/bin/nvcc` (gone, pre-13.3) | CUDA | 1249 / 22.9 | 1301 / 20.7 |
| fork `build-m2/` (ref, §16.3b) | bitnet fork 390c30775 | `/opt/cuda` 13.3 | CUDA | 1535 / 42.4 | — / 42.4 |
| fork `build-m2/` (ref, §16.3b) | bitnet fork 390c30775 | Vulkan | Vulkan | 1375.8 / 43.87 | 987.8 / 35.91 |
| VITRIOL `build-cu13/` | VITRIOL fork a3ee3be00 | `/opt/cuda` 13.3 | CUDA | 1149 / **26.89** | 1350 / **22.41** |
| VITRIOL `build-vk/` | VITRIOL fork a3ee3be00 | Vulkan (glslc) | Vulkan | 1178 / **27.81** | 1217 / **23.49** |

Bench invocation (VITRIOL rows): `llama-bench -m <gguf> -ngl 99 -p 512 -n 128
-fa 0 -r 2` (VITRIOL's `-m` takes no count arg; pp512/tg128 defaults).
Build quirks encountered: CUDA 13.3 **removed compute_61** (Pascal evicted —
§17.1 confirmed); VITRIOL's old ggml needed the upstream CUDA-13 compat shim
(`#include <cuda/iterator>` in argsort.cu + top-k.cu — applied); GCC 16
broke shared-lib link of cpp-httplib → `BUILD_SHARED_LIBS=OFF` for bench;
dual-GPU box: bench pinned via `CUDA_VISIBLE_DEVICES=0` / empty for VK runs.
**`vitriol-server.service` is a user systemd unit that auto-respawns — stop
the unit, never just kill the PID.**

**Findings (bisection verdict):**
1. **Toolchain alone: +17% tg** (22.9 → 26.9 on Q6_K CUDA 13.3 rebuild).
   Real, but not the story.
2. **VITRIOL's CUDA patches exonerated.** build-vk never executes
   ggml-cuda.cu hooks (vitriol pin/prefetch live only there), yet shows the
   same gap: VITRIOL Vulkan 27.8 vs fork Vulkan 43.9.
3. **The 1.6x tg gap is ggml base age.** It appears identically on both
   backends → shared kernel source. pp (compute-bound) gap is only 12-20%,
   tg (bandwidth/mmvq path) gap 60% — VITRIOL's decode vector kernels are
   stale, matching upstream's mmvq improvements landing after its base.
4. **Fold-back path**: rebase VITRIOL fork onto a newer upstream llama.cpp
   (patches are confined to server/* + ggml-cuda hook files — contained
   merge), then re-bench. Expected: VITRIOL ≈ fork ≈ 42-44 t/s.
5. **1070 Ti is LIVE on driver 580.178.04** (both GPUs enumerated) — §17.1
   driver question resolved by the 580xx branch. But CUDA 13.3 cannot emit
   sm_61 → **CUDA builds are now 3060-only; the 1070 Ti path is Vulkan**
   (NVK/proprietary ICD) or a CUDA 12.x sidecar toolkit.
6. Provenance recorded per row (nvcc path/version, commit, arch, flags) —
   §15 discipline extended to toolchains, now routine.

### 16.3d VITRIOL transplant — fold-back executed (2026-08-30)

Following §16.3c's verdict, VITRIOL got a fresh-transplant branch
`vitriol-ku` (upstream/master 9723942ad, no common ancestor — squash
history — so hand-port, not merge): VITRIOL's ggml-layer hooks (perf
diagnostics, LULL graph instrumentation + pool reset, buffer type, init)
re-applied onto 1572 commits of new upstream kernels. Expert-LRU hooks,
TQ3 stack, and server features = Phase 2 (VITRIOL SESSION_LOG_2026-08-30.md
holds the full record).

**Results (9B, pp512/tg128, -fa 0, 3060):**

| build | backend | Q6_K pp/tg | Q8_0 pp/tg |
|---|---|---|---|
| VITRIOL `build/` (old) | CUDA | 1249 / 22.9 | 1301 / 20.7 |
| VITRIOL `build-ku/` (transplant) | CUDA | 1509 / **42.08** | 1734 / **36.36** |
| VITRIOL `build-ku/` (transplant) | Vulkan | 1376 / **42.85** | — |
| fork `build-m2/` (reference) | CUDA/Vulkan | 1535 / 42.4 | 43.87 |

**Anomaly closed: 22.9 → 42.1 t/s (+84%), parity with the fork.**
Differential gate: greedy 48-token generation byte-identical to the fork
build. Production `vitriol-server.service` still on the old build until
Phase 2 ports the server flags it depends on.

## 18. Implementation Plan — GPU Rungs, Two-GPU Graph, Node-as-Device (2026-08-30)

Three interlocking tracks: **G** (wgpu perf ladder), **W** (two-GPU local
graph — first real heterogeneous pipeline), **T** (node-as-device TTY
front-end). G feeds W; T is the bring-up face for everything physical.

### 18.1 Track G — wgpu L2/L3 rungs

State: L1 done (`ouro-wgpu/src/lib.rs` — Q6_K gemv, parity PASSED on 3060,
cos > 0.9999). Current kernel is a correctness artifact: per-element
`byte_at` dequant, per-call buffer creation, one workgroup per row.

**G1 — persistent buffers + dispatch reuse** (biggest single win)
- Kill per-call `create_buffer_init` for x/y/params: persistent staging
  buffers, one reusable bind group per resident mat, write_buffer for x.
- Double-buffer readback (ring of 2 MAP_READ buffers), map-after-submit.
- Gate: unchanged parity test; matvec latency vs G1-before recorded.

**G2 — vectorized dequant kernel**
- Replace per-element byte reads with u32-aligned loads: one u32 = 8
  nibbles for ql/h assembly; `vec4<f32>` accumulate.
- Precompute `d*sc` per block into workgroup shared memory once per row
  (128-elem groups), not per element.
- Workgroup shape: reevaluate 64-thread/row tiling; consider 2D dispatch
  (row × block-chunk) + partial-sum reduce if rows are long.
- Gate: cos > 0.9999 vs CPU `matvec_q(Q6K)` (existing L1 test), and
  matvec throughput ≥ 8× CPU scalar on i7-3770.

**G3 — batched decode (ubatch 2-8)**
- Decode is bandwidth-bound; batching amortizes launch + readback.
- x becomes (tokens × n_embd) 2D storage; y likewise; dispatch z-dim.
- Gate: parity on batch of 8 vs 8 sequential CPU matvecs (greedy ids
  equal at sample).

**G4 — L3 integration: agent stage binds the pool**
- `OURO_GPU=1` on ouro-agent: `stage_setup` uploads the shard's Q6_K
  tensors (transformer attn.out + mlp.down first — the two hot matvecs);
  `stage_step` routes those matvecs through GpuPool, rest stays CPU.
- Parity contract per layer (cos > 0.9999), end-to-end greedy-id equality
  vs the same stage on CPU — the discipline that caught everything.
- Miss path (non-Q6_K tensor) = CPU fallback, never a failure.

### 18.2 Track W — two-GPU local graph (3060 + 1070 Ti)

Driver 580.178.04 live on both cards (§16.3c.5). No network involved —
this is M4 arithmetic measured locally before slaves exist.

**W1 — multi-adapter selection**
- `GpuPool::new()` currently takes `request_adapter(HighPerformance)` —
  ambiguous on dual-GPU. Add `enumerate_adapters` + explicit pick by
  index/name; env `OURO_GPU_NAME` (substring match) wins over index.
- Probe (`probe/gpu.rs`): add Vulkan adapter enumeration so NodeEntry
  reports per-GPU Vulkan capability (not just nvidia-smi CUDA-isms).
- 1070 Ti constraints recorded: 8.1 GB, sm_61, **Vulkan-only** (CUDA 13
  cannot emit it — §16.3c), int-dot present, coopmat2 absent (fine —
  our kernel uses neither).

**W2 — two-stage 9B Q6_K across two cards**
- Re-shard 9B Q6_K into 2 stages: 3060 = embed + layers 0..N + mid norm,
  1070 Ti = layers N..31 + output norm + head (`--packing` from bring-up
  kit D does this; VRAM split ~7.5 GB across 11.9/8.1 with headroom).
- Two ouro-agent processes on localhost (distinct ports, each pinned via
  `OURO_GPU_NAME=3060` / `=1070`), `ouro-pipeline` runs stage_step ACTS
  over TCP loopback — same topology as the M3-sim, now GPU-bound stages.
- Gates: greedy token ids == in-process CPU reference (the M3-sim gate,
  reused); measured t/s + watts table appended to §16.3c lineage.
  Power: sample nvidia-smi power draw per stage into the energy ledger
  (scheduler budget check runs for localhost nodes too — Art. 4 has no
  loopback exemption).

**W3 — 27B across two cards (gated)**
- 27B Q3_K_M ≈ 13.8 GB fits 20 GB aggregate, **blocked on Q3_K GPU
  gemv** — new kernel enters the same ladder (L1 parity vs CPU `matvec_q`
  → G2-style perf → G4 binding). No partial-offload hacks (§16.3b: they
  measured 0.9-1.1 t/s, the pipeline is the only honest path).
- Fallback meanwhile: 27B CPU-side with GPU mid-stages skipped (measured,
  documented, unglamorous).

**Deliverable**: measured two-GPU t/s + W/token row for 9B, extending the
M4 projection arithmetic (single 3060 43.9 t/s → 2-card split reality
check) — and the first GPU-literate graph entry for the 1070 Ti.

### 18.3 Track T — node-as-device (TTY front-end)

Doctrine (Art. 1): a node is an IO device, not a daemon we happen to have.
TTY = the device **face**; raw L2 (EtherType 0x88B5, Phase 2) = the wire
behind it. TTY is efficient for control/probes/bring-up/modem links; the
token/ACTS hot path stays on frames (seq ids, mux, ACKs — a tty has none).

**T1 — `ouro-ttyd` + FIFO device files**
- New bin in agent crate: bridges `/srv/ouro/tty/<node>.in` and `.out`
  FIFOs to the transport client for one node. Line in = one task
  (`stage_step <hex>`, `probe`, `budget 120w`, dot-form); line out = one
  response (status + optional hex continuation).
- Lockstep: one request in flight. This matches stage semantics exactly
  (positions strictly sequential — `stage.rs` already enforces), so the
  restriction costs nothing on the paths TTY is for.
- Contract: TTY path and TCP path and in-process produce token-id-equal
  results; every op still routes through scheduler + budget check (no
  device-file bypass of Art. 4).

**T2 — bootstrap: getty-shim agent (T0 tier)**
- `ouro-agent --stdio-tty`: speaks the same line protocol over
  stdin/stdout. A slave's getty line spawns it; master's ttyd connects
  via SSH pty (or raw serial). **Zero install**: any booted Linux with a
  login joins the graph — the bring-up recipe (§17.2 T0) becomes "boot,
  log in, done".
- Upgrade path: identical frames later ride raw L2; ttyd swaps transport
  behind the same FIFO face. Nothing above ttyd changes.
- Security: §9.3 HMAC header extension is MANDATORY before any
  cross-chassis ttyd — a TTY face must not become an unauthenticated
  execution orifice. Loopback demo first, HMAC before R2.

**T3 — modem bridge (deferred, integration point)**
- §14.7 HDMI video-modem stream lands as just another tty node — same
  FIFO face, exotic transport. The abstraction is the payoff: the OS
  cannot tell a slave from a display cable, and does not need to.

**Milestones**: T1 loopback demo → T2 = R2/IdeaPad bring-up (this IS the
M3 physical-wiring step) → T3 with modem hardware.

### 18.4 Sequencing and interlocks

```
G1 → G2 → G3 → G4 ─┐
                    ├→ W1 → W2 (two-GPU 9B demo, watts table)
T1 → (HMAC) → T2 ──┘         └→ W3 (Q3_K ladder first)
```

- G1-G2 first (days, self-contained, one file + one test file).
- T1 parallel-safe (different crate); T2 waits on HMAC decision.
- W2 is the convergence demo: G4 output + W1 selection + D-kit packing.
- W3 introduces the Q3_K GPU ladder — enter it only after W2 measures,
  so the 27B decision (2-card vs 4-card across slaves) is evidence-led.

Risks logged: 1070 Ti 8.1 GB stage budget (9B halves fit; 27B needs
Q3_K + tight packing); wgpu adapter ambiguity on dual-GPU (W1 pins by
name); VITRIOL llama-server holds ~5 GB on the 3060 — the budget ledger
must account it or benches lie; FIFO lockstep means TTY never carries
bulk prefill (route that through transport directly, as designed).

### 18.5 Session order (2026-08-31)

Executing §18.4 with gates; numbers land in this table as measured.

| # | Step | Gate | State |
|---|------|------|-------|
| S0 | Commit §16.3c/d + §18 (anomaly forensics, transplant, tracks) | provenance in history, not just on disk | ✅ 525c4af |
| S1 | G1: persistent x/y/params buffers, bind-group-per-mat, ring-of-2 readback | L1 parity tests unchanged-green; matvec latency before/after recorded | ✅ e2976ab: 9.76 → 3.3 ms (2.8 → 8.3 GB/s) on [8192,4096] |
| S2 | G2: vectorized dequant — u32 nibble loads (3 loads / 4 weights), workgroup-staged `d*sc`, vec4 accumulate | cos > 0.9999 vs `matvec_q`; ≥ 8× CPU scalar throughput on i7-3770 | ✅ ca5c043: cos 1.0 all tests; 31.5× (82.4 ms CPU → 2.61 ms GPU, 11 GB/s). Residual ≈ 2 ms = per-call submit+map overhead → G3 batching amortizes |
| S3 | W1: `enumerate_adapters` + `OURO_GPU_NAME`/index pick; Vulkan adapter fields in probe | 1070 Ti selectable by name on the dual-GPU box | ✅ b4079a7: both cards enumerate (3060=idx0, 1070 Ti=idx1, DiscreteGpu); `OURO_GPU_NAME=1070` picks it; probe merges `vulkaninfo --summary` → vulkan_api 1.4.312 both cards |
| S4 | T1: `ouro-ttyd` FIFO face, loopback demo (HMAC decision before any cross-chassis) | TTY == TCP == in-process token ids | queued next session |
| S5 | W2: two-GPU 9B Q6_K demo + watts table (stop `vitriol-server.service` unit first) | greedy ids equal; t/s + W/token row appended to §16.3c lineage | blocked-ish: agent shell cannot run systemctl — owner stops the unit before bench (server idle at 0% util during S1-S3, numbers believed clean) |
| S6 | W3: Q3_K gemv ladder → 27B across two cards, only after W2 measures | evidence decides 2-card now vs wait for slaves | gated on S5 |

G-rung ladder so far (same matvec, 3060, release): L1 kernel 9.8 ms →
G1 3.3 ms → G2 2.6 ms (31.5× scalar). Kernel time is now minor; the
floor is the synchronous submit+map round-trip. G3 (ubatch 2-8) and G4
(stage binds pool, one submit per token) are where the remaining 2 ms
goes away.

Hygiene: `bitnet-cpp` working tree carries auto-tuned LUT kernel configs
(generated artifacts, some degenerate) + untracked build dirs — left
uncommitted by design; VITRIOL transplant branch `vitriol-ku` lives in
the VITRIOL checkout, not this submodule.

