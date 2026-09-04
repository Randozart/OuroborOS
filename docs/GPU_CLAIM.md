# GPU Claim — Making Tails' GPUs First-Class Devices

> Art. 1, taken literally: "A CPU is an IO device in this distributed
> system." A GPU is another IO device. The HP Pavilion's Intel HD 620 is
> the first test case. Companion: `DMA_ROADMAP.md` (the ladder),
> `FIRST_LIGHT.md` (the gauntlet).

## The HP's GPU

The HP Pavilion 15-bc500 has an **Intel HD Graphics 620** — Gen9 Kaby
Lake iGPU. Intel confirms Gen9 gets **OpenCL 3.0 + Level Zero 1.5**
through their open-source Compute Runtime (NEO), MIT licensed.

- Repo: `github.com/intel/compute-runtime` — the NEO driver
- NixOS package: `intel-compute-runtime` (already in nixpkgs)
- Rust bindings: `ocl` crate (785 stars, MIT/Apache) — high-level OpenCL API
- On NixOS: `hardware.opengl.extraPackages = [ pkgs.intel-compute-runtime ]`

## What exists (more than expected)

| Layer | Status | File |
|-------|--------|------|
| GPU probe (nvidia-smi + vulkaninfo) | Working for NVIDIA | `cluster/src/probe/gpu.rs` |
| Vulkan compute kernel (Q6_K gemv) | L1 parity + G2 gate passed | `ouro-wgpu/src/lib.rs` |
| Agent telemetry includes GPU data | Working (`gpus: Vec<GpuInfo>`) | `agent/src/telemetry.rs` |
| Scheduler ranks GPU nodes highest | Bits 16+ dominate | `cluster/src/scheduler/mod.rs` |
| Shell displays GPU info | `n1.gpu`, `cluster?` census | `shell/src/propositions.rs` |

## The five breaks (WP-G1 through WP-G5)

| WP | Break | Fix | Status |
|----|-------|-----|--------|
| G1 | `BusTelemetry` drops GPU data; `NodeRecord::from_probe` hardcodes `has_gpu: false`; `NodeInfo` lacks GPU fields; no `WorkloadClass::GpuCompute` | Wire GPU through bus → record; add GpuCompute class | ✅ done — 155 tests green |
| G2 | NixOS image has no NEO runtime; agent not in render group | `hardware.graphics.extraPackages`; `extraGroups = [ "video" "render" ]` | ✅ done — image builds |
| G3 | No OpenCL matvec kernel; `ocl` crate not wired | Port WGSL Q6_K to OpenCL C; add `ocl` dep; `GpuPool` struct | ✅ done — **cos 1.00000000 vs CPU ref on RTX 3060** (`agent/src/gpu.rs`) |
| G4 | Agent stage executor runs CPU-only `matvec_q()` | `gpu_selftest` task: on-node GPU-vs-CPU parity proof | ✅ done — `executor.rs` |
| G5 | Scheduler has no GPU workload class gate | `WorkloadClass::GpuCompute` + `node_supports()` | ✅ folded into G1 — dispatches only to `has_gpu` nodes |

The same `gpu` feature binary runs on any ICD: NVIDIA on the head
(development + parity proof), Intel NEO on the HP tail. The image build
enables it via `buildFeatures = [ "gpu" ]` with `ocl-icd` supplying the
loader (`nixos/agent.nix`).

## Open source repos we're pillaging

| Repo | License | What we take |
|------|---------|-------------|
| `intel/compute-runtime` | MIT | NEO runtime (OpenCL 3.0 + Level Zero for Gen9+) |
| `intel/gmmlib` | MIT | GPU memory management (NEO dependency, auto-pulled) |
| `intel/intel-graphics-compiler` | MIT | LLVM-based GPU compiler (NEO dependency) |
| `oneapi-src/level-zero` | MIT | Level Zero headers + loader (if L0 path chosen) |
| `cogciprocate/ocl` | MIT/Apache | Rust OpenCL bindings |
| `ouro-wgpu` (ours) | — | Vulkan compute kernel (Q6_K gemv, already proven) |

## The end state

After G1–G5:
- HP boots → NEO detects Intel HD 620 → agent reports `has_gpu: true`
- `hiss> n1.gpu` → `Intel(R) HD Graphics 620 · 0 MiB · intel · OpenCL 3.0`
- `hiss> n1 assign gpu_matvec.` → scheduler routes to HP → OpenCL dispatch → result
- The HD 620 is a compute peripheral — Tier 1 of the DMA ladder

## Timeline

| WP | Depends on | Estimate |
|----|-----------|----------|
| G1 (registry plumbing) | Nothing | 1 day |
| G2 (NEO on tail) | G1 | 0.5 day |
| G3 (OpenCL kernel) | G1 | 2 days |
| G4 (agent executor) | G2, G3 | 1 day |
| G5 (scheduler) | G1 | 0.5 day |
| **Total** | | **~4 days** |

---

## The HP Pavilion upgrade (WP-N)

**Hardware correction (2026-09-04)**: the HP is a Pavilion Power-class
15" — i5-7200U + **Intel HD 620 (iGPU) and NVIDIA GTX 1060 6GB
(dGPU, Pascal)**. It was never an iGPU-only tail. Fleet after this WP:
**26GB pooled VRAM across three CUDA-class dGPUs** — RTX 3060 12GB
(head), GTX 1060 6GB (HP), GTX 1080 8GB (laptop) — plus the HD 620 as
a bonus fourth compute device.

### N1 — `OURO_GPU_NAME` pinning (`agent/src/gpu.rs`, ~15 lines + test)

On the HP there will be **two OpenCL platforms** (NVIDIA CUDA + Intel
NEO). `GpuPool::new()` currently takes `Platform::first()` blindly.
Add the ouro-wgpu W1 pattern:

- `OURO_GPU_NAME=<substring>` — case-insensitive match across
  platform+device names; wins over default order
- `OURO_GPU_INDEX=<n>` still wins as the explicit index
- On no match: error listing every candidate — never a silent pick

This is what makes `gpu_selftest` target the 1060 vs the HD 620
deterministically on the same box.

### N2 — NVIDIA driver in the image (`node-image.nix`, image-only)

The image has no NVIDIA driver, and `detect_gpus()` keys on
`nvidia-smi` — the 1060 sits silent until this lands. Zero new code:

```nix
services.xserver.videoDrivers = [ "nvidia" ];  # triggers the module; no X runs
hardware.nvidia = {
  open = false;                    # Pascal needs the proprietary module
  modesetting.enable = false;      # headless compute
  powerManagement.enable = true;   # coarse Optimus sleep (no finegrained on Pascal)
  nvidiaSettings = false;
  package = config.boot.kernelPackages.nvidiaPackages.production;
};
```

- `nomodeset` stays — NVIDIA compute is KMS-free; proven pattern
- Driver registers the OpenCL ICD → our proven OpenCL path (cos 1.0 on
  the head's 3060, same Pascal driver family) runs on the 1060
  **unmodified**
- ISO grows ~400–600MB; stick is 16GB — fine
- QEMU: module finds no device in the VM, `detect_gpus()` already
  handles absent/broken nvidia-smi → prove unaffected
- Needs `config` in the module args if not already imported

### N3 — Doc corrections

FIRST_LIGHT.md hardware line + fleet VRAM table here (done with this
section).

### Verification sequence (after the next reflash)

1. HP boots → single crimson banner → registers with
   `has_gpu: true, gpu_model: "NVIDIA GeForce GTX 1060 6GB"`
   (telemetry → bus → registry — all WP-G1 plumbing, no new code)
2. `gpu_selftest` over the wire → 1060 proves parity on-node
3. `OURO_GPU_NAME=Intel` second run → HD 620 proves itself —
   **two GPUs, one tail, one code path**
4. Laptop boots the same stick later → third dGPU joins the graph

### Gates

clippy -D warnings · agent parity re-run (3060, expect cos 1.0) ·
nix build · QEMU prove ALL PASS · commit + push · one reflash

### Open (level 2, after both tails prove)

`MatVecBackend` trait — `Local(GpuPool)` + `Remote(node_id, wire)` —
so `pool.matvec(...)` call sites never learn which GPU answered;
scheduler + budget pick; Art. 10 parity per remote backend before
first use. Weights stay resident on tails (BMTS); only KB-sized
activations cross the 1GbE wire.
