# One Graph, One Grammar — the Plan 9 Program

> **OUROBOROS**: **O**ne **U**nified **R**untime **O**rchestrating
> *a* **B**unch **O**f **R**andom **O**ld **S**ervers.
> The machine that remakes itself. The tail
> feeds the head.

Design authority is [`../CONSTITUTION.md`](../CONSTITUTION.md). This document is
the doctrine for adopting Plan 9's distributed-OS ideas and the phase specs
(B–E) that instantiate them. **Phase A (this document + the lineage rows it
records) is adopted.** Phases B–E are specs awaiting schedule (§16 of PLAN.md);
each is independently shippable.

Status: 2026-09-08. Provenance verified this pass; see §2.

---

## 1. Thesis

Plan 9 was the last complete attempt to make a *distributed machine feel like
one machine* — not a cluster, not a network, one machine with no meaningful
answer to "where am I?" It failed to win for three reasons, none of which bind
us:

1. **Protocol interop tax** — POSIX, X11, TCP/IP had to be eaten to be usable.
   We own both ends of every wire (Art. 10); there is no legacy menu.
2. **9P generality cost** — every operation was a round trip; bulk payloads
   rode the same protocol as control. Our Art. 6 inventory already priced that
   and split control (signed line/frames) from bulk (frames, DMA, BMTS raw).
3. **All-or-nothing adoption** — every node ran the full Plan 9. Our tails are
   Linux agents remade from the graph at every boot; the unikernel path stays
   Art. 6-gated, never a romance.

What transfers is not "everything is a file." That was the surface. The
invention beneath it, in two ideas (Pike et al. 1993): **a small uniform
resource protocol** and **per-process name spaces** — every resource reached
by the same operations, and each process assembling its own private view of the
world. OuroborOS has been building the same thesis bottom-up from the graph
(Art. 3); Plan 9 built it top-down from the file server. This program joins the
two. The abstraction that transfers is *uniform access + private views*; the
carrier is our typed, priced, revocable graph edge (Art. 3, 7) — not the byte
stream (Art. 2).

### 1.1 Why we already had the debt

We were recapitulating Plan 9 without naming it. Honesty (Art. 11, Appendix C
lineage) requires the ledger row; the deltas below are why copying is not
cargo-cult.

| Plan 9 | OuroborOS today | Delta |
|---|---|---|
| cheap terminal + central cpu/file servers | head + tail (HANDBOOK §2) | tails are disposably rebuilt from the graph |
| `import`/`exportfs` — no local/remote | `discover`, `ouro-ttyd` FIFO face, dot-notation | edges are priced (bandwidth/latency/watts) |
| 9P one protocol for every resource | signed line + frames; registry bus + task + probe verbs | verbs are not yet *one* grammar (§3, Phase B) |
| per-process name space | per-task assignment to a node | binding a *view*, not a node (§4, Phase C) |
| union directories | BMTS shard deploy, `deploy shards` | synthetic union weight view (§5, Phase D) |
| `/proc` self-knowledge | Beast self-describing topology | already our claim (Art. 3) |
| `cpu(1)` — move the session | Art. 4 — move the *arbiter itself* | bootstrap-seed invariant exceeds Plan 9 |
| a terminal is not a computer | Art. 5 — a CPU is an IO device | Plan 9's terminals still ran a kernel; our "IO device" is deeper |

---

## 2. Lineage and provenance (verified 2026-09-08)

| Work | Provenance | What it establishes |
|---|---|---|
| **Plan 9 from Bell Labs** (survey) | Pike, Presotto, Dorward, Flandrena, Thompson, Trickey, Winterbottom, *Computing Systems* 8(3):221-254, Summer 1995. Mirrors: doc.cat-v.org/plan_9/4th_edition/papers/9; usenix.org/legacy/publications/compsystems/1995/sum_pike.pdf | The architecture: terminals + central cpu/file servers; 9P as the network-level resource protocol; per-process name spaces; no tty driver in kernel; "build a UNIX out of a lot of little systems, not a system out of a lot of little UNIXes." POSIX emulation explicitly a "backwater." |
| **The Use of Name Spaces in Plan 9** | Pike, Presotto, Thompson, Trickey, Winterbottom, *ACM SIGOPS Oper. Syst. Rev.* 27(2):72-76, Apr 1993, doi 10.1145/506378.506413. Mirror: plan9.io/sys/doc/names.html | The two foundations: per-process name space + message-oriented file protocol. `mount`/`bind`, union directories, `rfork` namespace inheritance, `import`, `cpu(1)` (recreate the namespace on a faster CPU). 9P = 17 messages (3 connection/auth, 14 object ops: attach/walk/open/read/write/create/remove/stat/wstat/clunk...). Network as files (`/net/tcp` clone + `ctl`). "There is no global name space; local name spaces must adhere to global conventions." |
| **Man pages** | factotum(4), secstore(4), import(4), exportfs(4), cpu(1); 9front maintained fork at git.9front.org | Identity service (factotum + secstore) predates and inspired ssh-agent; one delegated identity across machines. |
| Failure mode | Above papers + licensing history (free non-commercial 2000; open source 2002) | Interop tax (POSIX/X11/TCP), 9P per-op generality cost, all-or-nothing adoption. None bind us (Art. 8: numbers and fuses; Art. 6: priced hops). |

### 2.1 Numbered deltas over Plan 9

1. **Carrier.** Plan 9 made objects look like files. We make resources look
   like *typed, priced, revocable graph edges* (Art. 1, 3, 7). Files are an
   Art. 2 default we priced and declined; the uniform-access protocol survives
   the carrier.
2. **Two planes.** 9P carried control and payload on one message set and paid.
   Control ops are small signed lines; bulk is frames/DMA/raw device (Art. 6).
   "Uniform grammar" never means "bytes through the same verbs."
3. **Costed edges.** A Plan 9 bind knew nothing of bandwidth, latency, or watts.
   Our namespace binds price every edge (Art. 6) and is gated by the energy
   budget (Art. 4).
4. **Contract-gated rebind.** Plan 9 rebinding was unchecked. Our rebind must
   re-pass the parity ladder (Art. 10) or the new view is rejected. Contracts
   are the safety case that makes radical namespace flexibility safe.
5. **The namespace is the scheduling unit.** Plan 9 namespaces were per-process
   conveniences; ours is what a PlacementPlan binds, revokes, and re-verifies —
   the Art. 4 ouroboros loop applied to resource *views* rather than node IDs.

---

## 3. The anti-table (what we refuse, with citations)

Every item is a default we priced, not a wall (Art. 8, Art. 9).

| Plan 9 habit | Why we refuse it | OuroborOS stance |
|---|---|---|
| "Everything is a file" (bytes) | Art. 2 marks VFS/page-cache as an inherited default; the byte stream loses type, price, and revocation | Everything is a *named, typed, revocable resource edge*; ops, not bytes, are uniform |
| 9P for bulk payloads | Art. 6: per-op round-trip generality was Plan 9's measured tax | Payload rides frames/DMA/raw shards; control ops stay tiny |
| Every node runs the full OS | all-or-nothing adoption killed the ecosystem; Art. 6 gates our own unikernel cut | Tails boot a graph-defined agent image; Linux is substrate, not identity |
| Global naming tree as the one truth | Art. 3: the graph is one object with one arbiter; Plan 9's "global conventions, no global name space" is a weaker claim | Private namespace slices are *derived* from one authoritative graph, not competing roots |
| Files were the only way to be object-oriented | we are not bound to file-like read/write | op kernel over the graph (§3), namespace bind as a first-class transaction (§4) |

---

## 4. Phase B — the grammar: one op kernel on a named resource tree

**Current state:** `ClusterTopology` (`cluster/src/beast/topology.rs`) is a
host-centric list (`Vec<NodeEntry>` + workloads + budget). Verbs are ad hoc:
registry bus verbs (`cluster/src/registry/bus.rs`), task dispatch, probe reads,
ttyd FIFO requests (`shell/src/ttyd.rs`) — three grammars for one machine.
The Art. 3 graph (devices/links/ports as first-class) is aspirational in the
code; Phase B makes naming real before the full graph lands.

**Goal (Art. 1, 3, 7, 10):** one fixed operation set over every resource,
reachable identically from HISS, the ttyd FIFOs, the signed wire, and the
in-process scheduler API.

### 4.1 The resource tree

A single rooted naming tree, serialized as Beast, backed by `ClusterTopology`
(the store; the tree is a typed view — one writer, Art. 3):

```
cluster/            the root; attach point for every client
├── nodes/n1/       a NodeEntry: cpu, simd, ram, gpu, tdp, rdma...
│   ├── power       live property (live cache wins, suffixed "(live)")
│   └── gpu/vram
├── capabilities/   Art. 7 affordances (purpose history, revocable)
├── budget          the EnergyBudget; writable = re-place mandate
├── queue/          the TaskQueue
├── tasks/<id>/     one dispatched task (dispatch = write here)
├── weights/<model>/<tensor>/   union shard view (§5, Phase D)
└── links/          priced edges bandwidth/latency/watts (full Art. 3 graph)
```

Each entry is a typed `Resource` with an allowed op set. Nothing in the tree is
bytes; reads return Beast-typed values or frames handles for bulk.

### 4.2 The op kernel

A minimal message set, mirroring 9P's economy but typed:

| Op | Meaning | 9P cousin | Example target |
|---|---|---|---|
| `attach` | authenticate + bind a client to the root | `attach` | line auth (existing HMAC) |
| `resolve` | descend the name tree to a resource | `walk` | `resolve n1.gpu` |
| `stat` | read type + attributes (typed, not bytes) | `stat` | `n1.power?` |
| `read` | read a value / open a bulk handle | `read`/`open` | telemetry, tensor payload via frames |
| `write` | set a value / enqueue work | `write`/`create` | `assign`, `deploy shards` |
| `ctl` | verb on a resource (power, sleep, recover, budget) | writing `ctl` file | `n1休眠`, `recover` |
| `bind` | give a resource a view/namespace entry (§5) | `bind`/`mount` | placement — returns a Namespace |
| `revoke` | release a binding en bloc | `clunk` | deassign, rebind |

Bodies are Beast S-expressions over the existing signed line
(`transport/auth.rs`); bulk payloads are offered as frame-mode handles
(`transport/frames.rs`) — read/write of bytes never traverses ops (anti-table).

### 4.3 One face everywhere

- `shell/src/parser.rs` + `propositions.rs`: dot-notation compiles to op
  sequences (`n1.power?` → `attach; resolve n1; stat power`). Existing verbs
  become sugar; grammar unchanged for the user (Art. 10 interface stability).
- `shell/src/ttyd.rs`: FIFO requests become literal `read`/`write` on a node
  resource; lockstep stays.
- `cluster/src/registry/bus.rs`: `ping/register/heartbeat` → `ctl`/`write` on
  `nodes/<id>`; the peer-IP identity anchor is untouched.
- Scheduler stays the only route to placement (Art. 11); the op kernel is the
  mouth, `Scheduler::schedule()` the brain.

### 4.4 Files / gates / tests

- New: `cluster/src/beast/resource.rs` (Resource type + op enum + Beast codec);
  `cluster/src/op/` kernel. Touched: parser, propositions, ttyd, bus.rs.
- Gate: `cargo test --lib`; clippy `-D warnings`; ttyd `ping`/`echo` and bus
  register/heartbeat round-trip *unchanged* while moving through the kernel —
  proves grammar adoption is behavior-preserving.
- Refusal: no bulk through ops; no byte-file semantics introduced.

---

## 5. Phase C — namespace slices: placement binds a view, not a node

**Current state:** `ScheduleOutcome::Dispatched { node }`
(`cluster/src/scheduler/mod.rs`) names a host. The op that runs never sees the
edges it may touch, the watts it may spend, or the contracts that gate its
rebind. Plan 9's private-namespace idea, upgraded to the scheduling unit
(§2.1.5), makes the Art. 3 "one machine" claim literally true per op: an op
only ever sees its bound world.

### 5.1 The namespace

A placement returns, instead of a node id, a **Namespace** — an en-bloc
revocable binding:

```
Namespace {
  bound:  Vec<ResourceHandle>,   // lanes, vram, paths, queue slot, ...
  watts:  u32,                   // this view's slice of the budget
  edges:  Vec<PricedEdge>,       // measured bandwidth/latency/joules (Art. 6)
  contracts: Vec<Contract>,      // parity gate for any rebind (Art. 10)
}
```

- `budget 120w.` lowers the parent namespace cap → children must rebind;
  a rebind that fails its parity contract is rejected, not excused (Art. 10).
- `revoke` is en-bloc: killing a binding releases every edge in it.
- Node-failure (Art. 9-11) = a namespace's edges go stale; `recover` rebinds
  displaced namespaces against the live graph — the scheduler's ordinary
  Tuesday.

### 5.2 Sequencing

1. Design the graph-edge model: `Links` as first-class in the beast topology
   (Art. 3 graph becomes code, not prose).
2. ClassAd-style request/offer predicates (PLAN §15.10 #3) for namespace
   assembly.
3. `Scheduler::schedule()` returns a `Namespace`; the dispatch path consumes
   its handles.
4. Parity rebind: reuse `bitnet-rs` parity tests as the contract oracle on any
   rebind of a placed inference namespace.

### 5.3 Risk gate

The existing single-node dispatch path stays green throughout; namespaces are
opt-in per workload class, never a silent retrofit. **This is PlacementPlan
v2** — no earlier behavior is destroyed to build it.

---

## 6. Phase D — the union weight plane

Plan 9 union directories, applied to the weight store (Art. 1's
checkpoint/shared-memory tier; Art. 2's block-stream stance):

- BMTS shards (`cluster/src/bmts.rs`) register as resources under
  `weights/<model>/<tensor>/`. A tensor's view is the *union* of the shards
  that hold it, wherever they live — one name, many physical carriers.
- `deploy shards` (already checksum-aware) becomes the union store's write
  path; reads route to the owning node over frames/DMA (`transport/frames.rs`,
  `transport/dma.rs`), never per-op.
- Parity contract: the union view is bit-identical across rebinds (Art. 10);
  the parity ladder is the store's integrity gate.
- Checkpoint plane (idle-NVMe, Art. 1) mounts into the same union — a tensor
  name resolves whether its carrier is VRAM, RAM, or an idle disk tier.

---

## 7. Phase E — identity service (head-side factotum analog)

Art. 10's trust model is two-planed: HMAC for transport, ed25519 for content
(ARCHITECTURE §3.2). The gap Plan 9's factotum+secstore closed is *delegated,
rotatable* identity. Phase E: one head identity service that hands out
scoped, revocable proofs (a key ring + delegation, not scattered copies),
feeding the existing update-plane key rotation. Small; do last; do not invent a
new trust boundary where the two-plane model already holds.

---

## 8. Sequence, gates, governance

```
Phase A (done: this doc + lineage rows) 
   → Phase B (grammar)
   → Phase C design doc ∥ Phase D
   → Phase C impl
   → Phase E
```

Every phase:
- `cargo test --lib`, `cargo clippy -- -D warnings`, integration/prove where
  the touched seam has one.
- Answers the Art. 11 cargo-cult question citing a row in this document.
- Refuses the anti-table: no files-as-bytes, no bulk through ops, no
  full-OS-per-node requirement.

Phases B–E enter PLAN.md §16 (build schedule) only by approval; this document
records doctrine and specs so any later agent or reviewer can find the full
context, the numbered deltas, and the priced refusals in one place.

---

## 9. Phase B implementation rungs (session 2026-09-08 decisions)

The grammar (§4) is built in three rungs. Rung B1 is approved; B2/B3 deferred.
Scheduled in PLAN.md §19.

### 9.1 The verb→op map (every HISS command, categorized)

Decision: **kernel-first** (pure in-process dispatch, fixture-tested; wire
untouched this rung), **`GraphBackend` trait** (Scheduler now, Registry later),
**all query + placement verbs** routed through `dispatch`.

| Verb | Op | Note |
|---|---|---|
| `?` `cluster?` `n1?` `n1.prop?` `power?` `cluster.active?` `probe` `tasks` `drift` | `stat`/`read` | query grammar — one source of truth in the kernel |
| `budget Nw` | `write cluster.budget` | backend → `budget.set_budget` |
| `n3 assign wl.` `wl on?` | `write tasks/` + `resolve` | **must** route through `Scheduler::schedule()` — ops are the mouth, not the brain (Art. 11) |
| `n1休眠` `recover` `register` `unregister` | `ctl` | node/bus lifecycle |
| `discover.` `drift` | sugar this rung | network sweep / version table → B2 wire ops |
| `wl on?` | sugar this rung | dry-run placement feasibility → Phase C ClassAd stat |
| `save`/`load` `poetry` | sugar now | persistence + output register = shell-local; `attach`/`export` in B3 |
| `generate` `shards` `deploy` | sugar now | payload ops → `write tasks/` + frame handles in B3 |

The `Command` enum (`shell/src/parser.rs`) is **untouched** — user-visible
grammar identical (Art. 10 interface stability). Existing verbs become sugar;
dot-notation compiles to op sequences.

### 9.2 Rung B1 — the op kernel (build spec)

- NEW `cluster/src/beast/resource.rs` — `ResourcePath` (dot-path parser),
  `Resource` typed-value enum (nodes, properties, budget, queue, tasks).
- NEW `cluster/src/op/mod.rs` — `Op` (`attach|resolve|stat|read|write|ctl`;
  `bind`/`revoke` reserved for Phase C), `OpResult` (Beast-serializable),
  `trait GraphBackend` (`stat`/`resolve`/`write_budget`/`ctl`),
  `dispatch(op, &mut backend)`.
- `cluster/src/lib.rs` — register modules.
- `scheduler/mod.rs` — impl `GraphBackend` (write_budget → `set_budget`;
  ctl sleep → node state; stat → node lookup).
- `shell/propositions.rs` + `context.rs` — impl `GraphBackend` over the live
  cache; route op-verbs through `dispatch`; output byte-identical.
- `registry/bus.rs` — wire untouched this rung; document `status` ≡ stat.

Gates (met 2026-09-08): `cargo test --lib` (cluster 144 + shell 74 incl. 13
kernel-ops integration tests asserting byte-identical output) · clippy
`-D warnings` 0 · shell tests prove `n1.gpu?`/`budget`/`tasks`/`assign`/
`probe`/`cluster?`/`recover`/`unregister` unchanged.

**Found during B1 (recorded for B2):** the Beast *text* codec does not
round-trip objects/enums (serialize emits `(key value)` pairs; deserialize
expects arrays — verified on `ClusterTopology`, regression-tested in
`op::tests`). Ops are therefore serde-canonical (JSON) today; the Beast wire
encoding for op bodies is a B2 item (fix the codec, then ops ride it).

### 9.3 Rung B2 / B3 (deferred)

- **B2 — wire unification:** ttyd FIFO requests and registry bus bodies become
  Beast op bodies over the signed line (`transport/auth.rs`); `stat` becomes a
  discovery verb; old-agent verb mapping for migration (Art. 10).
- **B3 — bind/revoke + bulk:** `bind`/`revoke` op surface (Phase C hook);
  frame handles for bulk payloads (`read weights/...` returns a handle — bytes
  never traverse ops, anti-table §3).

### 9.4 Companion: the air-path program

The same grammar drives every lane. Edge pricing (`PricedEdge`: bw, latency,
jitter, watts) and the bonded physical+air transport are designed in
[`docs/AIR_PATH.md`](AIR_PATH.md) — Tracks B–E scheduled in PLAN.md §19.
