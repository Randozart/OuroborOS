# Project Goal: The DMA Ladder — Tails as Peripheral Devices

> **Art. 1, taken literally**: "A CPU is an IO device in this distributed
> system." This document is that sentence made physical. A tail is not a
> smaller computer. It is device space: storage, memory, and silicon that
> the head addresses the way it addresses anything else it owns.

**Status**: goal, not yet scheduled. Ranked after R2 hardware bring-up.
Everything below is buildable on the current fleet; Tier 6 needs $25/tail
of used silicon.

---

## The ladder

| Tier | Technology | Fleet-ready | The head gains |
|------|-----------|-------------|----------------|
| **1. Compute peripheral** | Signed wire + executor tasks (WP2/WP3) | ✅ **Built** | Remote function calls into tail CPUs/GPUs |
| **2. Storage peripheral** | NBD / iSCSI / nvme-tcp: tail SSDs exported as head block devices | ✅ Works on 1GbE today | A tail's SATA bay as a local disk |
| **3. Memory tier** | NBD-swap / Infiniswap-class: tail RAM as the head's cold-page tier | ✅ Works, slow (~125MB/s on 1GbE — acceptable for *cold* pages) | ~100GB of extra RAM-space per tail |
| **4. RDMA verbs** | **SoftRoCE** (`rdma_rxe`, in-kernel): registered memory regions, zero-copy, true verbs semantics over ordinary NICs | ✅ Runs on any NIC; CPU-taxed | `ibv_*` DMA reads/writes into tail RAM behind memory windows |
| **5. Remote GPU** | rCUDA: head-side CUDA calls execute transparently on the tail's GPU | ⚠️ Works on 1GbE, latency-bound | The GTX 1060 (HP) + GTX 1080 (laptop) as CUDA devices on the head — 14GB of remote VRAM |
| **6. Physical DMA** | Used RDMA NICs — Mellanox ConnectX-3 class, **~$25/node** — RoCE at 10–40Gb/s, GPUDirect paths | 💰 Procurement | What HPC clusters actually run: DMA becomes *physical* |

**The pricing note (Art. 6)**: 1GbE is an inherited default, not a fact.
At ~$25 per tail, used RDMA NICs price that default away — the same move
the project makes everywhere: when "can't" cites hardware, buy better
hardware, not a smaller dream.

---

## Architecture: how it lands in OuroborOS

### The graph doesn't care what a node is
A storage-peripheral is a node whose capability class is `storage`. A
memory-tier is `memory`. A rCUDA host is `cuda-passthrough`. The
registry, scheduler, and budget gate already speak *capability*, not
*computer* — `NodeRecord` grows capability tags, placement rules grow a
few entries, and the Beast graph absorbs the rest unchanged.

### Control plane vs bulk plane
The signed line protocol stays the **control wire**: registration,
placement, contracts, budgets. Bulk DMA is a separate plane — a data
path cannot be a line protocol, and shouldn't try.

### Security model (Art. 10 — non-negotiable)
**DMA access means reading RAM, and the RAM holds the HMAC secret.**
The secret must never enter a DMA-visible window. Standard RDMA practice
is the mitigation:

- protection domains per peer
- registered-buffer whitelists only — never whole-RAM exposure
- keys live outside every registered region
- a node's DMA grants die with its registry lease (revocation on
  `NodeLeft`)

### The upgrade hop is already chosen
ACTS activations over 1GbE are the cluster's most bandwidth-critical
hop. RDMA upgrades *exactly that* — and the parity ladder (Art. 10)
re-runs over the new transport: token parity must hold over the DMA
path or the placement is rejected, not excused.

### Energy stays first-class (Art. 4)
A memory-tier tail idles at ~30W holding ~100GB. The budget scores it
as storage, not compute — a cheap shelf, not a slow server.

---

## First step (weekend WP, no purchases)

**SoftRoCE head↔tail proof**:

1. Load `rdma_rxe` on head + one tail (`rdma link add`)
2. Register a buffer on the tail; DMA-read it from the head (`ibv_` ping
   + throughput)
3. Measure: latency, throughput, CPU tax on both ends
4. Add `rdma` capability to `NodeRecord` + a placement rule
5. Re-run the ACTS parity ladder over the verbs path (Art. 10)

Deliverable: one measured row in this table, and the thesis — *node as
device* — demonstrated with hardware already owned.

---

## Provisioning and out-of-band rungs (2026-09-04)

Distilled from the "zero-touch provisioning" thread: the fleet workflow
should end with *no sticks, no monitors, no fingers on power buttons*.
Three rungs, honest versions.

### WP-PXE — born from the wire (near)

**NixOS netboot provisioning**: `config.system.build.netbootRamdisk`
+ dnsmasq (DHCP/TFTP) on the head. A tail powers on → PXE → iPXE
script from the head → kernel + initramfs fetched into RAM → boots
into the identical stateless node image.

- Kills the last manual fleet step: no USB sticks, no BIOS key mashing
  (PXE is a one-time boot-order entry per box), zero disk writes
- Enroll partition semantics change: identity arrives over the wire
  (head-link + signed admission) instead of the OURO partition — the
  stick workflow stays for boxes too ancient for PXE
- Same image, same gates; the QEMU prove grows a netboot variant

### WP-SER — out-of-band power authority (near)

**Serial power relay**: RTS or DTR pin → €0.50 optocoupler → the
motherboard's 2-pin `PWR_SW` header. Every board on earth has it.

- Short pulse (~100ms) = power on; hold (~4s) = hard cut; pulse again =
  cold boot. Total out-of-band authority, no WoL dependency on S5 state
- HISS grammar: `n4.power on | off | reset` — the propositions layer
  gains a power verb; the serial link is the same USB-UART class we
  already drive for consoles
- **Doubles as GENESIS's kill switch**: the watchdog that outranks the
  wyrm (GENESIS.md §IV). One relay, two constitutions served

### TERNARRAY — the silicon becomes the model (far, after Tier 6)

**FPGA ternary systolic array** — the corrected version of the hype:

Real and worth keeping:
- Ternary matmul needs **zero multipliers**: `{−1,0,+1}` weights are
  mux/negate/zero feeding an adder tree — pure LUTs, no DSP slices
- Block-scale distribution (`d · Σ wᵢ·aᵢ`) eliminates ~97% of FP
  multiplies — one DSP per block at the tree root
- Non-byte-aligned quants (3/5/6-bit) are *free*: a 5-bit bus is five
  wires; no shifting, no masking
- NF4-class non-linear dequant = one LUT6 per weight, ~0.2ns, zero
  clock cycles

The corrections (why this is a *long* rung):
- HDMI-deserializer activation transport on cheap boards is brutal
  serdes work — Artix-7/ECP5 do 1080p with effort, not casually; the
  honest transport is the same RDMA plane as Tier 6
- BRAM cannot hold 2B weights (~0.5GB packed); a DDR controller is
  mandatory, which means a real dev board, not a €30 module
- It is still a Von Neumann machine — but a *weight-stationary* one
  with a bespoke ALU, which for ternary inference is the winning shape

Sequencing: RDMA first (Tier 4–6 gives the fabric), then TERNARRAY as
a Tier 7 capability class (`fpga-ternary`) riding it.

---

| Item | ~Cost | Notes |
|------|-------|-------|
| Mellanox ConnectX-3 (PCIe, SFP+) | ~$25 each | RoCE v2, 10–40Gb/s; one per tail + head |
| SFP+ DAC cables | ~$8 each | Direct-attach, no switch needed for ≤4 nodes |
| Optional 10G switch | ~$60 used | Only if >4 nodes or copper transceivers preferred |

Physical layout + full wiring spec: **FLEET.md §8** (same-shelf layout,
dual-port head trick, USB-alternatives evaluation, open decisions).
The Alienware's single SATA bay and the head's wired link are unchanged;
NICs are PCIe add-ins — no chassis surgery. Laptops: no PCIe slots —
they ride Layer 1 (1GbE, Tier 1–3) or take an M.2 ConnectX-4 Lx
(~$40–60 used, spare NVMe slot) for Tier 5.

---

*The endgame, stated plainly: every functioning slab the world was going
to discard becomes addressable device space — storage, memory, silicon —
owned by one machine that remakes itself. The tail does not know it is
a peripheral. It knows what it is.*
