# OurobourOS — Architecture

The canonical blueprint. Governed by `CONSTITUTION.md`; every design here
cites its article. Implementation order and status live in `PLAN.md`
(§13–§17); this document describes the settled *shape*.

---

## 0. Mission (one line)

A private heterogeneous cluster of rejected hardware that presents itself as
one honest computer — running frontier-scale LLM inference that no single
member can run alone, metered in watts, verifiable in proofs.

## 1. The stack

```
+-- SURFACE ---------------------------------------------------+
| shell: dot-notation propositions, one noun, many parts      |
|   ?  . n3.gpu?  . generate <text>.  . budget 120w.          |
|   . discover.  . deploy [shards].  . save. / load.          |
+-- CONTROL ---------------------------------------------------+
| the graph (one arbiter, one authority)            [Art. 2/3]|
|   devices/ports/links/affordances/costs measured, not       |
|   assumed. PlacementPlan = compile of model -> topology.    |
|   Ouroboros clause: the brain is a schedulable workload.    |
+-- EXECUTION -------------------------------------------------+
| stage agents (musl-static binary + systemd unit per node)   |
| OpExecutor: tensor ops -> best-priced pool          [Art. 5]|
|   CPU pools: state, micro-ops, glue (host RAM residency)    |
|   wgpu/Vulkan: matvecs on every GPU class we own            |
|   FPGA (Kria): soft-PHY, 1588 clock, control-plane          |
| engine: pure-Rust model families (bitnet, qwen35, ...)      |
+-- DATA PLANE ------------------------------------------------+
| bonded transports, per-flow by measured price       [Art. 1]|
|   GbE raw/AF_XDP ....... activations (us-latency)           |
|   HDMI video-modem ..... bulk weights (230 MB/s, online)    |
|   btrfs send ........... state/release deltas over LAN      |
|   shared NVMe tier ..... checkpoint plane (candidate)       |
| formats: BMTS (immutable shards, mmap zero-copy)            |
|          ACTS (activations)  .  Beast (self-description)    |
+-- SUBSTRATE -------------------------------------------------+
| each node: CachyOS-minimal headless | nvidia-580xx-dkms     |
| (Maxwell -> Ampere, one branch) | vulkan-icd-loader         |
| sched_ext pinned busy-poll cores | RAPL power telemetry     |
| btrfs @ouro: gen/<ts> releases + current symlink + qgroup   |
+-- TRUST SPINE -----------------------------------------------+
| the parity ladder (docs/CONTRACTS.md): any placement, any   |
| backend, any recompile re-passes L0-L5 or is REJECTED.      |
+--------------------------------------------------------------+
```

## 2. The five load-bearing choices

| Choice | Rejected alternative | Why (article) |
|---|---|---|
| **Per-tensor placement inside a box; per-group cuts across boxes** | one-model-one-machine; llama-RPC-as-destination (tool, no contracts) | decode is bandwidth-bound; quant-mix workloads fit pools individually; LAN physics allow few cuts (Art. 5, §14.3) |
| **Vulkan/wgpu as the GPU substrate** | CUDA (vendor's current love) | CUDA 13 orphaned Maxwell/Pascal — our supply is orphans. NVIDIA r580 = Vulkan 1.2 on those cards, NVK conformant 1.3, ggml-vulkan *beats* CUDA on Pascal decode (§16.3, §17) |
| **State stays where it lives** (KV/delta-state on host RAM) | shipping state on the fabric | linear-attention models made the worst traffic vanish; activations only: 20–50 KB/token; 1 GbE becomes architecturally sufficient, not a compromise (Art. 5) |
| **The parity ladder as the immune system** | benchmark-and-pray | purpose is software -> contracts are the only fixed point; it already caught K_SCALE_SIZE, cursor, and over-projection bugs (Art. 10) |
| **btrfs generations as the deployment model** | package managers, containers, ISO re-flash | boot means joining the graph: node = subvolume + unit; delta replication = send stream; rollback = symlink flip; qgroups fence the shared pool (Art. 1, 3, 9-10) |

## 3. Data flow of the summit operation (`generate`, 27B across 4 cards)

```
prompt -> [stage0] tokenize (vocab-only) -> embed (CPU pool, host RAM)
  -> 64 layers in PlacementPlan order:
       micro-ops (rmsnorm/gates/conv/delta-state): CPU, zero traffic
       matvecs (qkv/ffn/lm_head): -> PCIe -> wgpu on stage card -> back
       card-internal only; nothing crosses the wire within a box
     -> 3 LAN cuts: ACTS frames ~20 KB, GbE, ~160 us each (0.5 ms/token)
  -> output_norm + untied lm_head on the 3060 (204 MB @ 360 GB/s ~ 0.6 ms)
  -> greedy sample -> text back through the shell
Budget monitor: RAPL per node, J/byte per pool. `budget 120w.` regenerates
PlacementPlan between requests; weights migrate on bonded transports;
the ladder re-passes before the first token of the new regime.
```

## 4. The Guarantee Chain

The federation is not a cluster we *hope* behaves as one machine; it is a
machine by three mechanically-checkable claims.

### I. Equivalence under relocation (the provable part)

> For the same input, the system emits the same tokens no matter where its
> parts sit.

Ladder: L0 dequant bit-exact -> L1 op parity across pools (cos >= 0.999) ->
L2 stage parity in-process vs TCP (token-exact) -> L3 system parity any
placement vs reference -> L4 performance contracts (measured, plans rejected
on violation) -> L5 energy contract (budget never breached, RAPL-verified).

A dead node is not failure; it is re-placement: edge lost -> plan
regenerated -> weights migrate cheapest measured path -> L1–L3 re-pass in
seconds -> tokens resume. Moving a layer is a rebinding, not a restart.
Full registry: `docs/CONTRACTS.md`.

### II. Max efficiency = bound-then-close, never hope

The ceiling is physics with a name: **bytes/token / effective aggregate
bandwidth**. The architecture guarantees approach, not assertion:

| Bound component | Guarantee mechanism |
|---|---|
| weight bytes read once/token | tensor-granular placement; giants->fat pipes, micro->CPU |
| no wasted wire | two-regime law; state-home placement; few-cut partitioner |
| no idle pool | compute-weighted packing equalizes stage times (slowest stage = optimized) |
| nothing assumed | self-measure at registration (memcpy, gemv, RTT); specs only `[unverified]`; TTL on plans |
| the trade is yours | budget = Pareto knob; measured W/token always reported |

Promise: achieved-vs-bound gap measured and displayed. Refuse: anything
that beats bytes/BW.

### III. Continuity = the Ouroboros operating its own body

Brain is a workload on its own graph (bootstrap-seed invariant, Art. 4);
releases roll back by symlink; Vulkan-first means CUDA 13 orphaning more
cards is a supply event, not a system event. A scheduler that cannot run on
its own cluster is a demo with a dependency.

### Failure inventory (guards named)

| Risk | Guard |
|---|---|
| thermal drift on 10-yr silicon skews plans | heartbeat re-benchmark; plan TTL; churn -> re-compile |
| fp-ordering greedy flips across backends | L3 on conditional streams; resync from checkpoints, not suspicion |
| compositor/contention stealing stage CPU | headless OurobourOS partitions; scx pinned cores |
| optimizer garbage-in (spec lies) | measured-only admission (Art. 6) |
| shared btrfs pool fate | qgroup fence + `btrfs send` deltas to backup device |
| pretending GbE is NVLink | cut cost is a hard term in the objective; TP-style chatter architecturally banned |

## 5. Explicit non-goals

- Distributed *training* (10^22 FLOPs reality)
- Kubernetes-shaped anything (no containers/etcd/YAML; one graph, one arbiter)
- llama.cpp wrapper (llama = oracle + measurement bar; runtime is ours — that
  is what makes GPU-on-Maxwell and budget-recompile exist at all)
- peak-perf single-box competition (we sell capability no single purchase
  can make: bigger-than-any-card, under-budget, self-describing, roll-back-able)

## 6. Position vs the blueprint (2026-08-30)

Surface/Control(partial)/Execution/DataPlane exist and **pass parity for the
CPU world** (27B in-process; 9B across 4 TCP nodes). Missing, in dependency
order: GPU pool in OpExecutor (wgpu Q6_K first) -> PlacementPlan compiler
(ClassAd-style) -> bonded fabric (HDMI modem online integration) -> physical
nodes (recipes ready; nvidia-580xx-dkms confirmed in CachyOS repos) ->
generations tooling. The last material unknown in this entire architecture
is the **Vulkan decode ceiling on orphaned NVIDIA silicon** — one afternoon
of llama-bench answers it.
