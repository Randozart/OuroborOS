# OurobourOS Enhanced Plan

**Date:** 2026-08-29
**Status:** Research complete, implementation pending

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
