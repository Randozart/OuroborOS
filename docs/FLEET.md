# FLEET.md — Hardware roadmap & procurement notes

Session findings, 2026-09-02 (coding-agent session; sysinfo measured on head).
Statuses: `[x]` done/measured, `[ ]` pending decision, `[!]` blocked.

Companion docs: `CLUSTER.md` (topology), `PLAN.md` (order of work),
`ARCHITECTURE.md` (settled shape). This file covers the *physical* layer:
what to buy, what to skip, and why — so future sessions don't re-litigate.

---

## 1. Measured fleet (2026-09-02)

| Node | Silicon | Measured state |
|---|---|---|
| head (dev box) | i7-3770, 16GB DDR3, RTX 3060 12GB + GTX 1070 Ti 8GB, 2TB SATA SSD (btrfs, 81% full) + 1TB HDD | **6.3GB swap in use** — memory-starved; this is the `27B Rust forward pending RAM/swap` blocker in PLAN.md |
| tail: ThinkPad/IdeaPad | mobile GTX 1080 8GB (soldered — not removable) | per CLUSTER.md |
| tail: Alienware Alpha R2 | GTX 960 4GB, 16GB DDR4 | single SATA bay, needs disk shrink |
| [ ] node N+1 | OptiPlex, model TBD | see §3 |

Onboard NIC on head is dead/unclaimed (no `network` class in `lspci`); USB
WiFi/BT is the current link. Any new head board with onboard 2.5/5GbE is a
direct upgrade — keep the USB adapter as failover.

## 2. Head upgrade = the Summit unblock `[!]`

The AM5 platform swap is not a general "PC upgrade": it unblocks the
`27B mmap full-Rust forward` TODO (PLAN §16) and gives the orchestrator
headroom for shard packing. Same project as OurobourOS — treat it as such.

Working parts list (street prices observed 2026-09-02; verify before buying):

| Part | Pick | ~EUR |
|---|---|---|
| CPU | Ryzen 5 7600 boxed (cooler incl.) or 7500F | 139–189 |
| Board | MSI PRO B850-P WiFi ATX (5G LAN) or Gigabyte B650E Eagle WiFi6E | 148–168 |
| RAM | 32GB (2x16) DDR5-6000 CL30 | ~100 |
| PSU | be quiet! Pure Power 12 M 650W Gold | ~80 |
| Total | | **470–540** |

Carried over: case, 3060, 1070 Ti, both drives, USB WiFi/BT. AM5 keeps a
drop-in CPU upgrade path — supports the "capability no single purchase can
make" line in ARCHITECTURE.md §0.

**Procurement rule learned this session:** bol.com marks up RAM ~5x,
PSUs ~2x, SBCs 2–3x (Pi 5 16GB at €295 vs ~€100 street; Orin Nano Super
€549 vs ~€250). CPUs are the only fair category there. Buy parts via
Tweakers Pricewatch shops (Megekko/Alternate/Azerty/Informatique),
used market for DDR3/cards.

## 3. Node N+1: OptiPlex decision tree `[ ]`

The OptiPlex is a *stage node*, not an "inference box" — the 3060 on the
head already out-inferences anything under €200 (see §5).

| Option | Hardware | ~EUR | Verdict |
|---|---|---|---|
| **A. CPU stage node** | OptiPlex with i5-8500+ class CPU, no GPU | 80–150 | **most architecturally native**: rmsnorm/gates/delta-state already belong to CPU pools with host-RAM residency (ARCHITECTURE.md §3); also a control-plane candidate |
| **B. MT + Pascal stage** | **Mini-Tower** variant + used GTX 1080 Ti 11GB | 200–290 | new 11GB stage; r580/NVK Vulkan doctrine holds (CUDA 13 orphaned it); aggregate VRAM 32→43GB; llama.cpp scoreboard: 1080 Ti tg 67.8–71.6 t/s @ 7B (PLAN.md) |
| C. SFF + LP card | SFF + RTX 3050 6GB LP | ~310 | barely beats the captive 960 4GB; worst value; skip |

**Hard constraint:** SFF variants have proprietary ~260W PSUs, usually no
8-pin PCIe — option B is only real on the **MT** chassis (or accept PSU
surgery). If MT costs ~€20 more used, pay it: keeps option B open.

Working plan: **A first, B later** if 27B shard packing says the fifth
stage is needed.

## 4. Uniform nodes, non-interfering transports (session synthesis)

The model, confirmed: every box collapses to the same shape —

```
CachyOS-minimal headless + musl-static ouro-agent + systemd unit + btrfs gen
```

Two corrections to "just dumb boxes":

1. **Dumb about placement, loud about capabilities.** Only the arbiter
   decides (one graph, one authority, Art. 2/3); nodes self-measure
   (memcpy, gemv, RTT, RAPL) at registration — measured-only admission
   (Art. 6) is what makes the graph honest.

2. **Non-interference is structural, not negotiated.** Transports never
   share layers (ARCHITECTURE.md §1): TCP :9500 (L3/L4), raw EtherType
   0x88B5 (L2, kernel-blind via AF_XDP), HDMI modem (own wire), btrfs
   send (rides TCP, scheduled like any flow). Headroom math: 64µs/hop
   vs 20–50ms compute ⇒ ~300x margin ⇒ no QoS needed.

### Fabric pitfalls checklist (worth a PLAN section)

- [ ] **NIC queues**: raw 0x88B5 + TCP on one consumer NIC — single-queue
      parts can't RSS-separate; pin channels via ethtool or eat CPU cost.
- [ ] **Switch audit**: dumb switches pass 0x88B5 untouched; some managed
      switches filter unknown EtherTypes.
- [ ] **CPU contention**: frame processing vs GPU stage on one node —
      sched_ext pinned cores is the guard; verify during S5.
- [ ] **One flow, one transport** — never split a flow mid-flight;
      per-flow transport selection by measured price (Art. 5).

## 5. Rejected / deferred (so we don't re-open them)

| Idea | Verdict | Reason |
|---|---|---|
| KV260 as interconnect node | **deferred** (per CLUSTER.md) | correct: 20–50KB/token ACTS vs 1GbE = ~1000x headroom; revisit only if GbE proves insufficient |
| FPGA for LLM inference | rejected | tiny BRAM, single-digit TOPS; hosting a GPU off FPGA fabric = PCIe RC + driver research project for zero gain. FPGA stays control-plane/wildcard per CLUSTER.md |
| SBC inference box (Pi 5 16GB / Orange Pi 5 / Jetson) | rejected | Pi-class = 2–4 t/s on 7B vs 3060's 40+; Jetson Super is the only real one (~€250 street) and still slower + 8GB |
| NPU on desktop CPUs | rejected | Ryzen 7000/9000 none, 8000G weak, Arrow Lake thin software; local AI runs on VRAM, not NPU |
| Unified memory in DIY | rejected | Apple/Strix-Halo only; not in the cards for tower hardware |
| Custom kernel / "write an OS from scratch" | rejected at kernel level | NVIDIA GSP firmware handshake = decade-scale work; OurobourOS *is* the OS, CachyOS-minimal is substrate (see §6) |
| bol.com for parts | rejected except CPUs | markup data in §2 |

## 6. OS escalation ladder (when "custom OS" is wanted)

1. **Now**: musl-static agent + systemd unit — de-facto distroless nodes.
2. **Later**: Buildroot/Yocto minimal image per node (kernel + nvidia-580xx
   + the binary). Custom OS without writing a kernel. NixOS stays rejected
   for workers per PLAN (btrfs generations are the deployment model);
   `nixos/` dir stays empty until this rung, if ever.
3. **Never**: own kernel for GPU userspace (see §5).

## 7. Resume pointers

- `[!]` Head AM5 swap → unblocks 27B mmap forward (PLAN §16). Budget €470–540.
- `[ ]` OptiPlex: decide SFF vs MT **before** purchase (option B depends on it).
- `[ ]` §4 checklist items fold into PLAN §16.2/S5/S6 as verification steps.
- `[ ]` After head swap: re-run sysinfo (swap should be ~0), then 27B mmap task.
- Evidence thread this session: sysinfo (free/lspci/lsblk) → bol price survey
  → OurobourOS repo read → this file.
