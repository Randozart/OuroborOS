# OurobourOS Constitution

**The founding document. Everything else — PLAN.md, CLUSTER.md, AGENTS.md — instantiates these articles.**

**Purpose of this document.** To prevent the single most likely way this project
fails: building a cluster-inference system that quietly inherits the assumptions
of general-purpose operating systems, then wondering why it is a general-purpose
operating system wearing a costume. Every design decision in OurobourOS must be
checkable against a named article. Any "can't" must cite physics or a fuse
(Article 8). "Nobody does it that way" is not a limitation.

**The one-line test for every design review:**
*Is this a law of nature, or a default we inherited?*

---

## Article 1 — No Purpose But Use

Hardware has no semantics. It has capabilities. "GPU", "display port", "audio
jack", "network card", "the computer" — these are *userland opinions*, encoded
by drivers that answer to someone else's product roadmap. When we own the
driver, purpose becomes a software decision, revisable per request.

Corollaries in practice:

- An HDMI output is a 3.4–18 Gbps differential-pair transmitter we control
  (§14.7: it carries weights).
- An audio jack is a synchronized analog channel with microsecond timestamps
  (candidate: cluster clock distribution).
- An idle NVMe drive is a shared-memory tier (candidate: checkpoint plane).
- An FPGA's exposed TMDS pins are a soft PHY for any protocol we define.
- A GPU copy engine is a network processor that happens to live in silicon we
  already own.
- A compositor scanout path is a data-plane transmitter.
- **A CPU is not a CPU** (Article 5).

"Waste Is Fuel" is this article's economic face: old hardware is not weak
hardware, it is hardware whose *purpose was misfiled by its vendor*. We
refile. Where Waste Is Fuel says the machine should not be discarded, this
article says the machine's ports were never what they seemed.

## Article 2 — The Kernel Is a Constraint Factory

A general-purpose kernel is not neutral infrastructure. Every abstraction it
provides encodes a decision made for someone else's workload:

| Inherited default | Whose need it serves | OurobourOS stance |
|---|---|---|
| Time-sliced scheduler fairness | multiprogrammed desktops | stage hosts busy-poll pinned cores; workloads are long, cooperative, known |
| Interrupt-driven I/O | power + responsiveness for unknown devices | hot paths use polled/DMA rings; interrupts are a measured cost, not a given |
| Socket stacks (TCP/IP) as the network | universal compatibility | networking is any path with a framing rule: raw L2, video modem, shared storage, NIC multicast (Article 7) |
| VFS + page cache for storage | multipurpose file semantics | weight files are block streams; io_uring / O_DIRECT / raw-device reads where the profile says so |
| Process isolation everywhere | untrusted tenant safety | intra-cluster trust is total (private hardware, owned media); isolation remains exactly where the graph meets outsiders (Article 10) |
| One OS per machine | historical unit of the "computer" | the OS's unit is the *cluster*; a machine is a substrate holding devices, including a scheduler fragment (Article 4) |
| Vendor feature locks (GeForce P2P disabled, HDCP assertions, firmware fences) | product segmentation | locks are facts to inventory, not orders (Article 8); where a driver says no for market reasons, the silicon often says yes — we read the datasheets, not the marketing |

This is the "why a kernel?" answer: **to replace their defaults with ours,
item by item, where measurement proves the default costs us.** Not to
reinvent scheduling for pride. See Article 6's method for how we choose which
defaults to attack.

## Article 3 — One Graph, One OS

The central object of OurobourOS is the **cluster resource graph**. Not a
network of hosts; a single heterogeneous machine described once:

```
Graph =
  Devices     : { chip | bus | port | lane }  x { capability set }
                gpu(3060): scanout-tx, dma, compute@13TF, vram@360GB/s, 12GiB
                port(hdmi-out-0): differential-pair@18Gbps, purpose=UNSET
  Links       : { device_a, device_b } x { bandwidth, latency, direction,
                protocol_kind, watts }
                hdmi-out-0 -> usb-cap-1: 230MB/s, 40ms, down, video-modem
                enp0s25   -> enp0s25   : 125MB/s, 160us, both, l2
  Costs       : energy J/op, thermals °C telemetry, watts headroom per rail
  Roles       : assigned-per-run, revocable: {stage-host, brain-host, ...}
```

The **arbiter of that graph is the OS**. There is no second arbiter by which a
display server holds HDMI while NetworkManager holds the NIC while
pulseaudio holds the jack. One object, one writer.

Every subsystem speaks graph: probes *feed* it (Article 7), the shell *queries*
it (`n3.hdmi?` is a legal future sentence), the scheduler and PlacementPlan
*resolve* against it, transports *are* its edges. A capability absent from the
graph does not exist; a capability in the graph has no owner but the arbiter.

The graph self-describes in Beast (CLUSTER.md) — the system knows what it is,
down to the purpose-free port list.

## Article 4 — The Ouroboros Clause (self-scheduling)

The name is the architecture: **the control plane is a workload in its own
scheduler.** Shell, orchestrator, agents, modem encoders — all are graph
operators with resource demands, placeable, migratable, restartable like any
tensor op. `budget 30w.` may legitimately move the brain itself onto the Kria
and idle all x86 rails.

**Recursion guard.** Self-reference without a seed is a loop, not an OS. The
invariant: a minimal **bootstrap fragment** exists on the power-ordered node
(the machine that decides, via hardware, what stays energized — the rail
clock is the root trust). It holds: the graph's authoritative copy, the
schedule, and a heartbeat contract. It does NOT hold model state. The
bootstrap fragment itself is the only component that cannot be relocated
during a request, because its job is to decide relocations — after a planned
handover, it may migrate too. Brain-hopping is a transaction, not a paradox.

## Article 5 — A CPU Is Not a CPU

Identity is a role list. In the graph there are compute *pools*, each with a
cost function, and work *shapes*, each with a fit:

| Pool | Currency | Wins at | Loses at |
|---|---|---|---|
| Modern GPU | bytes/s from dedicated die-RAM | large matvecs (bandwidth-bound decode), batch | launch latency (<256 KB work), tiny vectors |
| Legacy GPU (Maxwell/Pascal) | bytes/s per watt, PCIe lane | heavy layers, weight streaming sinks | newer instruction mixes |
| CPU (this cluster: i7) | low per-op latency, DDR attach, state residency | f32 micro-ops (conv, alpha/beta, norms), recurrent state, glue, control | anything >1 GB/s sustained |
| FPGA (Kria) | *arbitrary* | custom PHY, wire-speed protocol, 1588 clock, glue logic | FLOPS, ease |

The scheduler therefore never asks "what code runs on what computer." It asks
**"what operation, on which data, over which path, at which joules?"** —
Article 14's PlacementPlan (`OpExecutor`, per-tensor within a box, group-cuts
across boxes) is the mechanism; this article is the justification.

A CPU is, as stated: an IO device in this distributed system — specifically
the pool attached to host RAM where recurrent state lives and where nothing
must ever cross a PCIe BAR. The fact that it can also run the scheduler is a
scheduling accident, not an identity.

## Article 6 — The Latency Inventory (method, not dogma)

How we decide when Article 2 means *replace* and when it means *accept*:

1. For every hot path, enumerate hops: IRQ, context switch, scheduler
   quantum, syscall, page fault, memcpy, copy engine, PCIe traversal, wire,
   protocol stack, scanout wait, USB transfer.
2. Price each hop in µs (measured on this hardware — spec sheets are
   candidates, probes are verdict).
3. Attack the largest priced hop. Stop when the next-largest hop is cheaper
   than the code that would remove it.

The menu we draw from, in order of expected price on our paths:
busy-poll + pinned cores > io_uring/O_DIRECT > AF_XDP > raw L2 (§7) > DMA
chains skipping CPU staging > Kria hardware 1588 as cluster clock >
frame-budgeted modem paths (HDMI latency is accepted by design where payload
is bulk) > **unikernel consolidation (Unikraft enters here — deferred by
inventory, never by principle: if agent stacks show scheduler noise, the
agent boots as a 2 MB image on the rail it serves)** > bare-metal scheduler
fragment (furthest cut; requires custom silicon drivers; only when the graph
itself becomes the bottleneck).

Nothing on the menu is ever "no". Everything is "not yet, for this price."

## Article 7 — Capability Registry (probes report what a thing can *become*)

Probing does not classify devices — it registers affordances:

```
port(hdmi-out-0): {tmds-pair, scanout-source, max-clock: measured}
nic(igb0): {l2-fd, hw-timestamp?, af_xdp?, wol}
nvme(sda): {o_direct, size, idle-tasks-writable}
usb(cap-stick-1): {uvc, yuy2-passthrough?, scaler: absent|present}
audio(jack-out): {clock-taps: measured jitter, codec sample clock}
```

Lifecycle: `discover -> advertise -> reserve -> assign(purpose, run-id) ->
revoke`. Assignment is per-run and revocable; the registry keeps a purpose
history (`hdmi-out-0: 3d idle | 2d boot console | 18h weight-modem`).
PlacementPlan edges consume the registry directly — an edge exists wherever
a reserved capability pair can be wired.

This is also the harvest backlog: every unclaimed affordance is a
§14.7-shaped opportunity awaiting measurement.

## Article 8 — The Limit Inventory (honesty about walls)

Permitted limitations cite a **number** or a **fuse**:

- Numbers: TMDS lane rate, PCIe 2.0 ×16 = 8 GB/s, DDR3-1600 dual channel =
  25.6 GB/s, 1080p60 = 249 MB/s/pixel-plane, thermal envelope, frame period
  16.7 ms, wire propagation.
- Fuses: HDCP keys (we never assert them — we own the source), secure-boot
  policy (we control both ends), vendor microcode locks, spectrum rules if
  we ever go radio.

The **anti-inventory** — things that are *not* limitations: "GPUs can't talk
to each other" (fused feature ≠ silicon limit), "HDMI is output" (connector
spec ≠ purpose law), "networks need IP" (a 1981 compromise became a habit),
"one kernel per box" (an accident of the 1970s). Any sentence containing
"normally" or "normally, an OS…" is a default, not a wall.

## Article 9 — Anti-Cargo-Cult Checklist (operational procedure)

At every design review, each item gets one of three answers:
**accepted** (justified by inventory price), **attacked** (scheduled work),
or **inverted** (made a feature).

1. The machine is the unit of scheduling. → invert: op/tensor is the unit.
2. Networking means the IP stack. → invert: any path with a framing rule.
3. The display is an output sink. → attack: scanout as TX, capture as RX.
4. A host runs one OS. → invert: a substrate hosts graph fragments.
5. Idle compute is waste. → invert: idle = reserved capacity, harvest tier.
6. Storage is for files. → attack: device tier for state and checkpoints.
7. System time is the clock. → attack: 1588/hardware-anchored cluster time.
8. Power is a facility bill. → invert: power is the scheduler's currency.
9. Processes are enemies to isolate. → accept inside, attack at edges:
   total trust within the owned cluster, contracts at the boundary.
10. Boot means reaching a desktop. → invert: boot means joining the graph
    (a slave's OurobourOS partition exists to register, not to present).
11. Failure is an exception. → invert: a node death is a graph edge loss;
    re-partition is the scheduler's ordinary Tuesday (§13 error recovery).
12. "A distributed AI cluster is an application." → **inversion of
    inversions: it is a machine. Applications do not get to own ports.**

## Article 10 — Contracts Survive Reassignment

This constitution does not weaken a single contract. It strengthens them,
because when purpose is software, **the contract is the only fixed point**.

- CONTRACT-FIRST governs what a placed/relocated/recompiled execution plan
  must still guarantee: product > 0, greedy-top1 equality vs oracle,
  cos > 0.999, budget never exceeded. Placement varies; proof stands.
- The parity ladder (bit-identical dequant, rung A→B→C token equality) is the
  constitution's enforcement mechanism: every recompile of the model onto new
  hardware re-passes the same contracts, automatically, or the recompile is
  rejected. **Contracts are what make radical flexibility safe.**
- External interface stability: the shell grammar (dot notation, propositions)
  never changes when the graph re-partitions — the user-visible machine is
  one machine precisely because its promises are one contract set.
- Security posture: total trust *within* the graph is an assumption of
  ownership, and ownership is enforced at the one boundary that remains
  standard — the outside. Node join = mTLS/SSH-key contract (§9); no
  anonymous edge ever becomes a purpose-free port.

## Article 11 — Governance

1. **Citation rule.** New subsystems and transports cite the article that
   licenses them ("modem: Art. 1 + 7; raw L2: Art. 2 + 6"). A design that
   cannot find its article is either novel — and gets one — or cargo-culted.
2. **Cargo-cult check.** Every PR answers one question: *what default did we
   accept in the last change, and which inventory priced it?* No pricing, no
   acceptance.
3. **The limit test.** Any "can't" names a number or a fuse, or it is a
   to-do item.
4. **Review ritual.** Article 9, item by item, quarterly — because the list
   is how the document stays alive; a constitution nobody re-reads is
   decoration.
5. Amendments arrive as new articles or new inventory rows, with the same
   proof standard they impose on everyone else.

---

## Appendix A — Worked examples (this repo, already justified by its articles)

| Work | Articles |
|---|---|
| TQ1_0/Q4_K dequant ported to Rust with C-parity bit tests | 5 (own the pool), 10 (contract enforces freedom) |
| OpenMP oversubscription measured at 5.8× cost → threads=physical | 6 (inventory, not folklore) |
| HDMI video modem as third `Transport` impl | 1, 2, 7 |
| Compute-weighted layer packing, lm_head pinned to 3060 | 5 |
| `budget 120w.` recompiling PlacementPlan between requests | 1, 3, 4, 8 |
| Kria deferred as "substrate, roles pending measurement" | 5, 6 |
| Shell as single-noun interface to 3 chassis | 3, 10 |

## Appendix B — Vocabulary discipline

Words that must never be load-bearing again in this codebase:
*naturally, normally, usually, obviously, standard, typically, you can't.*
Replacements: *measured, priced, reserved, revoked, fused, capable, chosen.*

When the i7 is an IO device, the vocabulary follows.
