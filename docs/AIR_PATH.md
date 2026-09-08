# Many Paths, One Protocol — the Air-Path / Dual-Lane Program

> **OUROBOROS**: **O**ne **U**nified **R**untime **O**rchestrating
> *a* **B**unch **O**f **R**andom **O**ld **S**ervers.
> The machine that remakes itself. The tail
> feeds the head.

Design authority is [`../CONSTITUTION.md`](../CONSTITUTION.md). This document is
the companion to [`docs/PLAN9.md`](PLAN9.md): where PLAN9 supplies the uniform
grammar over the graph (Phase B op kernel), this supplies the **multi-path
transport** — a node is reached over *every lane it owns* (cable, radio,
display port), and the lanes are bonded by the same arbiter. Air is a lane,
not a fallback.

Status: design + borrow program, adopted 2026-09-08. Provenance verified this
pass; implementation scheduled as tracks B–E in PLAN.md §19.

---

## 0. Origin — why this document exists

Session 2026-09-08. While planning the Plan 9 program, the owner surfaced
[`SwarmLLM`](https://github.com/Nehanth/swarmllm), a peer-to-peer LLM runtime
that splits a Qwen 3.8 27B over the devices in a room — over Wi-Fi, in browser
tabs. The owner's framing, preserved verbatim in intent:

> SwarmLLM shows we can also swarm over Wi-Fi. We could potentially even get
> speed by utilising both wired and unwired paths at the same time.
> OuroborOS is all about showing a distributed system is a protocol problem.

That is the thesis of this program. **A distributed system is a protocol
problem.** The arbiter (Art. 3) owns every path; the grammar (Phase B op
kernel) is the same over each; the lanes are priced (Art. 6), bonded, and
revocable (Art. 4/7).

---

## 1. Thesis

Two claims, one conclusion:

1. **The air path is a protocol problem, not a penalty.** SwarmLLM runs 27B
   decode at 7.7 tok/s across a MacBook + iPhone on the same Wi-Fi, and
   3.5–6 tok/s across the open internet — not by making Wi-Fi fast, but by
   making each network lap carry several tokens (batched prefill, MTP
   speculation, exact rollback). The slow link is absorbed by the protocol.
2. **Physical and air are simultaneous lanes, not alternatives.** A tail with
   a cable *and* a radio already owns two paths to the head. Bonding them is
   Beowulf's channel-bonding move (already in our lineage, CONSTITUTION
   Appendix C) applied to the second radio — with the LLM pipeline as the
   workload that decides, per frame, which lane (or both) carries it.

Conclusion: **bonded multi-path over priced edges, decided by the same
scheduler that places tensors.** The transport stops being a single wire and
becomes a set of `PricedEdge`s the arbiter schedules across. This is the Plan 9
thesis closed at the physical layer: one machine, many wires, one protocol.

---

## 1.1 The avenue doctrine — all lanes, copper and air

Owner's framing, preserved verbatim in intent (plan review, 2026-09-08):

> "If it can make a connection, whether through copper or air, it's an avenue,
> protocols be damned."

Two clauses, both binding.

**"It's an avenue."** The transport inventory is *any physical channel that can
carry a connection* — not the set of things with an IP stack. What a port was
*designed* to do is a vendor default (Art. 1): priced, and if it carries work,
exploited. A display port is an I/O lane (Art. 5: a CPU is an IO device — so is
a screen connector). A USB jack is a lane. A powerline is a lane. An audio jack,
an IR blaster, an idle NVMe, a spare radio — all avenues until priced out. The
cluster knows what it is (Art. 3, /proc-equivalent); the lane registry is part
of that self-knowledge.

**"Protocols be damned."** Two readings, both binding:

1. *Protocol is not the filter.* We do not ask "what speaks IP?" We ask "what
   connects?" and then price it. Enumerating an avenue is the work; giving it a
   protocol is a later, mechanical step. An avenue with no protocol yet is a
   capability, not a dead end.
2. *The grammar does not change per lane.* One op kernel over every avenue
   (PLAN9 §4); the bond layer (§4.2) is one policy — control to lowest-latency,
   bulk across all lanes, critical duplicated — applied identically whether the
   lanes are copper, air, or a display port. Lane-agnostic by construction.

**The lane inventory.** Every avenue is typed in `PricedEdge.kind`; the enum
grows as the registry does:

| Family | Avenue | Status | Note |
|---|---|---|---|
| Copper | TCP signed line + frames (`transport/auth.rs`, `frames.rs`) | live | control + frame carrier today; seq + `token_pos` already order cross-path |
| Copper | L2 raw Ethernet (EtherType 0x88B5, magic `0x4F55524F`) | Phase 2 spec | no IP in the middle; the cluster's own wire |
| Copper | DMA / PCIe bus-master (`transport/dma.rs`, `ouro-dma`) | code present | Art. 2 block-stream stance; the machine's internal spine |
| Copper | RDMA (`has_rdma`, `rdma_gid` on `NodeEntry`) | spec'd | zero-copy avenue for bulk |
| Copper | BMTS raw block device (`cluster/src/bmts.rs`) | live | weights on raw devices; the Phase D union store mounts the same lane |
| Copper | NVMe / SATA | spec'd | the checkpoint plane (Art. 1 idle-tier) is a transfer tier, not a tomb |
| Copper | USB / Thunderbolt | enumerated | a phone is a disk is a lane |
| Copper | Powerline (Ethernet over AC) | enumerated | the wall socket as a switch |
| Copper | Serial / UART / MIDI / any pin | enumerated | the nihilistic "any pin is a port" |
| Copper | GPU interconnect (NVLink/SLI) | enumerated | vram-to-vram lane between GPUs |
| Copper | Display port / HDMI | enumerated | invert the "output" default (Art. 5): an out-port is an avenue |
| Air | Wi-Fi (`wlan0`) | probe `iw dev` link | the original air lane; jitter is the live-or-die metric |
| Air | Radio / modem (MODEM) | spec'd | low-bandwidth survivable lane |
| Air | Bluetooth / BLE | enumerated | short-range avenue |
| Air | Cellular | enumerated | the internet join edge (Art. 10 mTLS), not owned transport |
| Air | LoRa / ISM band | enumerated | long-range, low-rate, high-survive |
| Air | IR / light (Li-Fi) | enumerated | the room as a bus |
| Air | Acoustic (sound card as modem) | enumerated | the acoustic coupler inverted |

**Pricing is the same for every avenue.** `{bw, latency, jitter, watts,
reliability}` — the §4.1 `PricedEdge` shape. The scheduler does not care which
family; it minimizes the priced cost. A lane that costs more than it carries is
dropped from the namespace by `budget` (Art. 4), not by taxonomy.

**Trust does not degrade per avenue.** Every lane rides the same HMAC-signed
frame layer (Art. 10); air means owned radio inside the trust boundary; a lane
never relaxes auth. "Avenue" is not "open door."

---

## 2. Provenance (verified 2026-09-08)

### 2.1 SwarmLLM

| Fact | Value |
|---|---|
| Repo | github.com/Nehanth/swarmllm |
| Author | Nehanth Narendrula |
| License | MIT (CITATION.cff; `@software{swarmllm2026,...}`) |
| Runtime | WebGPU (WGSL, ~50 kernels) + WebRTC room (PeerJS signaling only) |
| Model | Qwen 3.8 27B GGUF Q4_0 (15 GB), 64 layers = 48 Gated-DeltaNet + 16 attention, MTP `nextn` draft |
| Wire | hidden state = 10 KB f16 per token (`dim` 5,120); ordered reliable datachannels; frames correlated by position |
| Decode | 9.0 tok/s plain, 16.1 spec on NVIDIA GB10 (Vulkan, headless); llama.cpp native 8.0 on same file/GPU |
| Air | MacBook + iPhone, same Wi-Fi: 7.7 tok/s spec; cross-internet 3.5–6 tok/s |
| Prefill | up to 8 columns per GPU pass, 16 tokens per network round; no LM head during prefill |
| Bit-exactness | every optimization golden-gated; speculative output == plain decode for any sampler |

Source docs read this pass: `README.md`, `docs/protocol.md`, `docs/architecture.md`.

### 2.2 The mechanisms and what they establish

| Mechanism | Protocol detail | What it establishes for us |
|---|---|---|
| Batched prefill | `ai-hidden-b {basePos, n, spec?}`, ≤16 columns, recurrent state snapshotted per column | Prompt prefill needs ~1–2 laps on a slow link — the fill-the-caches pass never runs the LM head |
| MTP speculative draft | host chains `nextn` to predict K tokens; one batched trunk pass verifies 1+K (≤8 cols, one network lap); first mismatch ends acceptance | A slow lap moves 3–7 tokens; draft depth (3/5/7) chosen per room from measured tok/s |
| Exact rollback | `ai-rollback {k}` restores recurrent state to the between-column snapshot | Spec failure must be bit-reversible — our `kv.seq` out-of-order rejection is the seed; DeltaNet state snapshots extend it |
| Host owns the head | host = tokenizer, embed, final norm, LM head, sampler, draft; workers = contiguous layer ranges | Draft lives where sampling lives (our brain/control node); workers never carry it |
| Wire negotiation | `WIRE_F16` flag; decoders accept f32 for older peers | Art. 10 contract: frames negotiate format, old agents degrade, version mismatch fails loudly |
| Streamed range-fetch | each worker fetches *only its tensor byte spans*, repacks Q4_0/Q8_0 into nibble + f16-scale arrays, caches with a size stamp | `deploy shards` refines to fetch-only-your-spans + size-stamp cache + resume → Phase D union-store write path |
| Memory-roofline WGSL | cooperative GEMV family (generated), 64-thread row sweep, dequant in registers, reduce in shared memory, one command submit per token | Direct reference for ouro-wgpu G3/G4 (their measured 183/184 GB/s decode roofline confirms §14.2's bandwidth-bound thesis) |

### 2.3 Numbered deltas over SwarmLLM

1. **Carrier.** SwarmLLM is a browser mesh (WebRTC, PeerJS). We invert to one
   graph, one arbiter (Art. 3): workers are scheduler-assigned, never a room.
2. **Grammar.** Their protocol is LLM-specific (`ai-*` messages). Ours is the
   Phase B op kernel — `stat/read/write/ctl/bind/revoke` — the same over every
   lane. The LLM pipeline is a *client* of the grammar, not the grammar.
3. **Pricing.** Their slow-link mode is a global room setting. Ours is per-edge
   (`PricedEdge`), decided by the scheduler, feeding Plan-9 namespace slices
   (docs/PLAN9.md §5).
4. **Energy.** Air lanes cost watts (radio ~1–2 W active). The lane set is a
   revocable, budget-gated capability (Art. 4/7): `budget 120w.` can drop the
   radio from a namespace.
5. **Identity.** Their room is ephemeral and anonymous. Our multi-homed tail is
   one identity (SMBIOS `node_id`) with many edges (see §4.4).

---

## 3. The priced thesis (Art. 6)

"Both paths at once" wins differently per workload. No free lunch: each lane is
priced and the scheduler chooses.

| Workload | Payload | Where dual-path wins | Lane policy |
|---|---|---|---|
| Decode lap | 10 KB/token | **NOT bandwidth** — one lane exceeds need. Wins on latency + robustness: two independent paths kill head-of-line, air carries spec drafts | primary + draft over both |
| Prefill (batched) | 16 × 10 KB/lap | Stripes across both lanes → lap time halves | stripe |
| Weight / shard deploy | 15 GB | Aggregate bandwidth real: GbE ~125 MB/s + Wi-Fi 40–60 MB/s ≈ 1.6× | stripe |
| Checkpoint / brain-hop (Art. 4) | GBs | Same aggregate win | stripe |
| Steady decode, tight budget | 10 KB/token | Drop the air lane (radio ≈ 1–2 W); keep wired | wired only |

Honest reading: for steady decode the air lane buys latency-masking and
redundancy, not bandwidth; for prefill, deploy, and checkpoint it buys real
aggregate throughput. That split is the Art. 6 answer, and it is why ACTS v2
modes (§4.3) are per-edge, not global.

---

## 4. Multi-path architecture

### 4.1 Edges as first-class graph citizens

`NodeEntry` grows `edges: Vec<PricedEdge>`:

```
PricedEdge {
  iface: String,            // enp3s0 | wlan0 | hdmi0 | rdma0
  kind:   enum { L2, TCP, RDMA, MODEM, AIR },
  bw_mbps: u32,             // measured, never spec-sheet
  latency_us: u32,          // measured (ping-style, latest + p50/p95)
  jitter_us: u32,           // measured — air lives and dies on this
  watts: u32,               // radio/card active draw, Art. 4
  protocol_kind: enum,      // one line per lane; frames ride any
}
```

`kind` is the typed lane registry — the §1.1 avenue inventory, encoded. It
starts `L2 | TCP | RDMA | MODEM | AIR` and grows (USB, PWR, SERIAL, OPTICAL,
ACOUSTIC, GPU, BLK, ...) as avenues are enumerated and priced. The scheduler
only ever sees the price tuple; the enum is bookkeeping for humans and probes.

Probe additions (`cluster/src/probe/`): `iw dev <iface> link` (link speed,
signal dBm) + RTT/jitter measurement against the head. This is the Phase C
namespace prerequisite (docs/PLAN9.md §5) pulled forward by the air lane.

### 4.2 The bond layer (application-level MPTCP, no kernel work)

`frames.rs` already carries sequence numbers and `token_pos` correlation —
cross-path reordering is already handled. The bond layer opens **N channels
per peer** (one per live edge) and schedules each frame by policy:

- **control** (tiny, latency-bound ops) → lowest-latency lane
- **bulk** (prefill batches, shards, checkpoints) → stripe across all lanes
- **critical** (state snapshots, rebind acks) → duplicate on 2 lanes
- **lossy-lane backoff** → air lane carries spec drafts; wired carries primaries

No kernel changes, no MPTCP in-tree: the bond is a scheduler decision over
priced edges, exactly like tensor placement.

### 4.3 ACTS v2 modes (the SwarmLLM slow-link protocol, per-edge)

In `cluster/src/pipeline.rs`, ACTS gains modes that are *selected per edge by
the scheduler* (Art. 6), not globally:

- **`acts-batch`** — ≤16 hidden states per frame during prefill; columns
  processed strictly in order (causality preserved); no LM head on the wire.
- **`acts-spec`** — decode lap carries MTP draft tokens; verification runs ≤8
  columns in one batched pass; first mismatch ends acceptance; trunk token
  used. **Contract: speculative output == plain decode, token-exact** (Art. 10).
- **`acts-rollback {k}`** — exact recurrent-state restore (KV + Gated-DeltaNet
  state) to the snapshot after column k. Extends the existing `kv.seq`
  out-of-order rejection (PLAN §16 Track R).
- **`WIRE_F16`** negotiation — f16 activation frames by default, decoders
  accept f32 for older agents; version mismatch fails loudly at join.

Draft depth (3/5/7) and batch width become **scheduler outputs bound to the
namespace's edge set** — the per-room autotuning of SwarmLLM generalized to
per-edge, priced.

### 4.4 Multi-homed identity (a breaking consequence, handled)

`registry/bus.rs` today anchors node identity to the peer socket IP; idempotence
is per-IP. A tail with wired + Wi-Fi is two IPs → today two "nodes". Fix:

- `node_id` = SMBIOS-derived hash (already the node-image plan, R2_BRINGUP §8),
  the one anchor.
- Registration becomes `ctl nodes/<id>/edges/<iface>` — per-edge, not per-node.
- `find_by_ip` idempotence superseded by `find_by_node_id`; old single-IP
  agents keep working (verb mapping, Art. 10 migration).

This lands in the Phase B op kernel naturally: edges are resources under
`nodes/<id>/edges/`, registered with `ctl`.

### 4.5 Energy (Art. 4/7)

The air lane is a **priced, revocable capability**. WiFi radio ≈ 1–2 W active.
The energy budget scheduler decides whether the namespace includes the radio;
`budget 120w.` can drop it. The ouroboros clause (recompile the plan under a
budget) now bites the physical layer: a lane, not just an op, can be re-placed.

### 4.6 Security (Art. 10)

Air frames ride the **same HMAC-signed line** as every lane. Air means *owned*
Wi-Fi (WPA2/3, trusted AP) inside the cluster's trust boundary. Open/guest
Wi-Fi is a join boundary (mTLS), not a transport. No downgrade path: a lane
never relaxes auth.

### 4.7 Failure (Art. 9-11)

A lane dropping (cable pull, Wi-Fi fade) is an **edge loss in the graph**, not
a node loss. The arbiter rebinds the namespace over remaining edges; duplicated
critical frames make rebind acks survivable. Node death remains the Art. 9-11
recovery path, unchanged.

---

## 5. Anti-table (what we refuse, with citations)

| SwarmLLM / air habit | Why we refuse | Our stance |
|---|---|---|
| WebRTC/PeerJS mesh, room topology | Art. 3: one graph, one arbiter; workers are assigned, not self-organized | bond layer over frames; scheduler owns lanes |
| Guest-ask / shared conversation | not a compute model | one writer, one brain |
| Internet tails | owned-cluster trust boundary | air = owned Wi-Fi; internet stays the Art. 10 mTLS join edge |
| Kernel-level MPTCP / in-tree bonding | Art. 6: cheaper to schedule at the app layer over priced edges | app-level bond; frames seq already orders |
| Global room settings (one depth for all) | Art. 6: per-edge, priced | depth/batch per edge set |
| "Networking" defined as IP-only transport | Art. 6: curates the inventory to the vendor's menu | any physical avenue is a lane (§1.1); protocol follows price, never the reverse |

---

## 6. Tracks (implementation; scheduled in PLAN.md §19)

- **Track B — ACTS v2 air-path** (B1 MTP un-filter → B5 per-edge tuning).
  Gate: 9B Q6_K + `nextn` over TCP agents; **spec == plain token-id
  equality**; tok/s vs baseline over a simulated lossy lap.
- **Track C — streamed range-fetch + Phase D union store**. Gate: deploy
  resume + rejoin-fast cache test; union view bit-identical across rebinds.
- **Track D — SwarmLLM as second parity oracle**. Gate: greedy token equality
  vs cb_eval *and* SwarmLLM golden streams.
- **Track E — wgpu kernel reference for Track G**. Gate: G4 one-submit-per-
  token; measured bytes/s vs roofline.

Full specs, files, and gates: PLAN.md §19; borrow rationale above.

---

## 7. Governance and lineage

- Every PR in this program answers the Art. 11 cargo-cult question citing a
  row in this document or PLAN.md §19.
- Provenance rule: claims are verified, dated, and named — see §2.1 (verified
  2026-09-08).
- Cross-references: docs/PLAN9.md (grammar + namespace slices), PLAN.md §15.1
  (SwarmLLM lineage row), §15.10 (adoption items 13–18), §19 (session record +
  schedule), CONSTITUTION Appendix C (SwarmLLM row). Lane doctrine: §1.1.