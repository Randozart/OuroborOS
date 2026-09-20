# DUET — Doctrine of Unified Expected Transport · Deferred Unification of Equivalent Transfers

> *Two machines holding one expectation move no data; they telegraph equivalence,
> and only surprise rides the wire.*

DUET is the data-sharing doctrine of OuroborOS: predictions substitute for
payloads. The wire carries fingerprints when prediction holds, quantized
residuals when prediction nearly holds, and full bytes only on miss. One
acronym, two faces — the doctrine (what travels) and the reconciler (when
unification runs): the protocol sings both parts.

- **DUET** = **D**octrine of **U**nified **E**xpected **T**ransport
- **DUET** = **D**eferred **U**nification of **E**quivalent **T**ransfers

Status: doctrine adopted 2026-09-20. Phases P1–P4 defined; P1 first landing.

---

## 1. The doctrine

Every DUET transfer is a three-way choice between:

| situation | wire carries | cost |
|---|---|---|
| prediction exact | fingerprint (hash) | ~0 bytes |
| prediction within margin | quantized residual | log2(range/margin) bits |
| prediction wrong | full payload + flag | full cost (bounded, never corrupted) |

**Golden rule (Art. 10)**: speculation may fail; it may never lie. Every
prediction is hash-verified by the receiver before it is trusted. A miss
costs bytes — never correctness. This is the cache-consistency protocol of
the distributed machine.

**Determinism boundary** (measured, not assumed — CONTRACTS.md ladder):

| compute context | agreement | DUET mode |
|---|---|---|
| same binary, same arch (L0/L2 in-process) | bit-exact | hash-first, zero-bit equivalence |
| same binary, cross-silicon (L1/L3: cos ≥ 0.999) | near, never exact | quantized residual |
| different binary/version | unknown | epoch mismatch → full fallback |

The fleet's oldest nodes (Ivy Bridge, pre-AVX2) and newest never agree
bit-for-bit on float math. DUET never assumes they do.

**Economics (Art. 1, Art. 4)**: recompute-for-bandwidth trades energy
against wire. Priced free at v1; the energy budget arbitrates once the
power model lands (L5). Prefetch is plan-ranked, never blind — utility
decays with plan TTL (CONTRACTS meta-rule 3).

---

## 2. Prior art (so we don't reinvent its failure modes)

| technique | what we took | what we left |
|---|---|---|
| Deterministic lockstep (RTS netcode) | input-only sync; peers simulate the full state | no fixed timestep — our "simulation" is the inference pipeline itself |
| rsync / zstd `--patch-from` | rolling chunk hashes, delta pull | no wire protocol — we add epochs + bond lanes |
| Video P-frames | predicted + quantized residual | no lossy allowance — activations must verify |
| Speculative decoding (Leviathan et al.) | draft K, verify in bulk, accept longest prefix | Bonsai ships no MTP head → prompt-lookup pivot (P3) |
| Outcome-index / selection codes | enumerated outcomes travel as indices (P4) | shared candidate-set registry + set-hash guard is ours |
| Nix closure pre-seeding | plan-informed pre-staging on idle links (P1.5) | continuous gossip instead of deploy-time copy |
| QUIC degradable flows | quiet frames die first under congestion | single-path bond lanes, not multipath |

## 3. The four guards (each maps to existing machinery)

1. **Never contend with token hops.** Reconciliation rides a degradable
   bond lane (`transport::bond`): token-bucket rate limit, dropped
   wholesale while a pipeline stream is live.
2. **Idempotent by construction.** Chunks are content-addressed (hash) and
   epoch-tagged; arrival order and duplication are irrelevant; stale-epoch
   chunks are GC'd on arrival. Reordering cannot corrupt.
3. **Staging budget.** Receiver caps unclaimed bytes per (peer, shard);
   LRU/TTL eviction. A chatty node cannot fill a tail's disk.
4. **Plan-ranked, never blind.** Prefetch queue = `PlannedChunks(node)`
   from the scheduler's placement plan; fallback heuristic = adjacent
   pipeline stages. Utility decays with plan TTL.

---

## 4. Phases

### P1 — Shard delta sync (core DUET)

Shard epochs and chunk indexes make re-placement pushes megabytes, not
gigabytes — and identical state pushes nothing.

- `schemas/bmts.capnp`: `ShardHeader` += `epoch` (sha256 hex of full shard
  data), `chunk_size` (64 KiB), `chunk_hashes` (u64 per chunk — *advisory*
  hints for diffing; the epoch is the authority)
- `cluster/src/bmts.rs`: `epoch_of(data)`, `verify_epoch(shard)`, chunk
  index (de)serialization through the v2 capnp path; v1 files read forever
- `cluster/src/sync.rs` (new): `plan_pull(local, remote) -> requests`;
  `rebuild(requests) -> shard` staged + atomic-renamed, epoch-verified
  before first visibility
- `cluster/src/bin/ouro-bmts.rs` (new): `info | verify | upgrade` — v1→v2
  conversion computing epoch + chunk index (sharder keeps emitting v1;
  pycapnp not assumed)
- Transport: chunk requests/replies ride `frames.rs` sessions (HMAC covers
  any payload shape)

**Gates**: rebuilt shard sha256 == source (bit-exact); index+epoch match →
push sends 0 data bytes; flipped byte in one chunk → detected → single-chunk
refetch → exact rebuild; v1 shard opens unchanged.

### P1.5 — Quiet-wire reconciler (Deferred Unification)

Solves P1's chicken-and-egg: the first field push pays full bytes because
the receiver lacks an index to diff against. Gossip indexes on idle wire so
the diff is ready before the transfer — ideally the transfer never happens.

- `cluster/src/transport/bond.rs`: **degradable lane class** — token-bucket
  rate limit, hard pause while any pipeline stream is live, frames marked
  droppable
- `cluster/src/sync.rs`: gossip loop (index/epoch advertisements), chunk
  staging with per-(peer, shard) budget + TTL/LRU eviction, `PlannedChunks`
  ranking hook
- `cluster/src/scheduler/mod.rs`: thin read-only `PlannedChunks(node)` hook
  over the current placement plan

**Gates**: live pipeline + reconciler → token stream unchanged, hop p99
unregressed; kill mid-transfer → no partial state observable (staging +
atomic rename); staging capped under synthetic flood; pre-gossiped nodes →
simulated placement moves 0–1 chunks.

### P2 — Speculative ACTS (hash-first activations)

Stage→stage hidden states become fingerprints when the receiver can
reconstruct them (same binary + order = bit-exact). Value: latency hiding +
WAN bandwidth + the contract pattern for prediction-before-trust at the
activation layer. On GigE LAN this is *not* a bandwidth win (20 KB/token is
noise) — the win is the pattern and the divergence telemetry.

- `cluster/src/pipeline.rs`: ACTS speculative mode — header flag +
  `state_hash`; receiver recomputes producing layers locally, compares
- Hit → payload skipped. Miss → NACK → full resend. Cross-silicon
  quantized-residual refinement is a documented future, not v1.
- `cluster/tests/duet_acts.rs`: in-process contract first; 2-node TCP per
  `shell/tests/qwen_tcp.rs` pattern

**Gates**: 1000-token same-binary replay → 100% hit rate (asserted);
injected one-ULP flip → miss → NACK → resend → stream token-exact (near-tie
rule); L2 ladder unchanged (4-agent token-exact).

### P3 — Speculative decode (pivoted: prompt-lookup)

**Fact**: Bonsai-2 GGUF carries `nextn=0`, zero draft tensors — no MTP head
exists in this model. The draft-head plan is impossible without
self-distillation, so we pivot: **prompt-lookup speculation** — n-grams from
prompt + generated text draft K continuations; the model verifies.

- `cluster/src/infer/qwen35.rs`: n-gram drafter + accept/reject loop on
  `Qwen35Model` (greedy longest-prefix acceptance)
- **Landed 2026-09-20 (batched verify)**: `matmul_tl1` K-column kernel
  (decode once, K dots — pure-matmul 2x at K=8), batched layer path
  (`run_layer_batched`: projections amortize, conv/recurrence/attention
  cores stay sequential-causal), block verification with recurrent-state
  snapshot/rollback (rejected drafts pollute delta-net state — restore
  on partial accept), batched lm_head. **Gates**: batched == sequential
  at cos 1.0000000 per position; speculative block stream token-identical
  to greedy. **Measured**: verify(4) = 1.45x vs 4 sequential steps
  end-to-end (sequential delta-net cores damp the 2x matmul win).
  One real bug found by the equivalence gate: the batched lm_head
  skipped the Hadamard pre-rotation the scalar path applies — caught at
  cos 0.037, fixed, gate green.

**Gates**: spec output stream token-identical to greedy stream (losslessness
on this model, this prompt set); miss path changes nothing observable; hit
statistics reported.

### P4 — Choice frames (DUET-CHOOSE: enumerated outcomes)

The generalization that started as "the other machine already predicted
this — it only needs a go command": when an outcome lands in a jointly
enumerable candidate set, the set travels as IDs (or not at all, if both
sides generated it from shared state), the outcome travels as an index, and
the reply travels as the pick.

- Protocol: `propose(set_id, set_hash, candidates) → pick(i) → verify →
  commit | fallback(full payload)`
- `set_hash` guards candidate-set divergence (version skew): mismatch →
  automatic fallback → correctness preserved
- **Flagship landing — Art. 4 budget reconciliation (hardens L5)**: the
  energy budget is literally a sum over nodes (RAPL per agent, scheduler
  sums). Scheduler predicts per-node draw from its power model; nodes reply
  `as-planned` (1 bit) or `drifted` + quantized delta; the budget sum runs
  on predictions + exceptions. L5 "budget exists; energy model pending"
  moves from GAP toward WIP — as *measurement*, not enforcement.
- `cluster/src/duet.rs` (new): choose-frame codec + candidate-set registry;
  telemetry first, activations later

**Gates**: injected drift beyond margin → reading path → budget decision
consistent with ground truth within the tolerance bound; simulated version
skew (set-hash mismatch) → fallback → correct; as-planned path measured at
O(1) bytes per reconcile.

**Precision note (learned during landing)**: naive deltas are lossy —
`p + (a-p) ≠ a` in f64. Drifted nodes therefore carry their exact reading;
reconstruction is bit-exact over communicated values and diverges from the
raw-actual sum by at most `n_as_planned × tolerance` (the designed price of
not shipping quiet nodes' readings).

---

## 5. Choices made (and why)

| choice | decision | rejected alternative |
|---|---|---|
| channel for reconciliation | existing bond lane, degradable class | plain background TCP (ships faster, no congestion semantics) |
| chunk hash function | SipHash-13 fixed-key u64 (std) | xxh3 crate dep — chunk hashes are *advisory*; epoch (sha256) is the authority, so no new dep for hints |
| sharder v2 emission | defer; `ouro-bmts upgrade` converts | pycapnp dependency in Python tooling |
| ACTS v1 speculative | hash-first, full payload on miss | quantized residual now (cross-silicon residual coding is a refinement with its own ladder) |
| P3 draft source | prompt-lookup n-grams | MTP head (impossible: `nextn=0`), self-distillation (out of scope) |
| P3 batched verify kernel | documented next step, not v1 | stepwise verify has no wall-clock win; landed only as losslessness scaffolding |
| P4 first landing | energy-budget telemetry (Art. 4) | activations (covered by P2 residual path) |
| speculation pricing | free at v1 | immediate Art. 4 arbitration (power model pending; pricing hooks logged) |

## 6. Uncertainties / open questions

- **Cross-silicon residual coding**: how many bits does a cos ≥ 0.999
  activation residual actually need per element? Measured only after P2
  telemetry accumulates. If large, the residual path stays WAN-only.
- **Gossip topology**: pairwise (pipeline neighbors) v1; mesh gossip is a
  future (AIR_PATH Track D adjacency may inform it).
- **Choice-frame candidate sources**: who generates candidate sets for
  sums-of-telemetry besides the scheduler's power model? Open.
- **Staging location**: disk (tails have slow disks) vs page-cache
  (evicted under memory pressure) — v1 uses temp file + budget; revisit
  with DMA_ROADMAP (RDMA staging could skip staging entirely).
- **Energy pricing**: the recompute-vs-wire decision needs a watts/byte
  model; logged as hooks, landed with L5.

## 7. Futures (not in scope)

- Quantized-residual ACTS for cross-silicon hops
- Batched verify kernel (`matmul_tl1` K-column) unlocking real P3 wins
- Mesh index gossip + AIR_PATH adjacency
- RDMA staging (zero-copy reconciliation into BMTS mmaps)
- Energy-priced speculation (Art. 4 arbitration)
- Registry-bus choice frames (heartbeat as-is-planned bit)
- **Optimistic mode (surrogate-then-correct)** — for REPLICA hops only
  (pipeline stages don't hold the producing weights, so they cannot
  recompute at all): receiver speculatively runs `A_surrogate · W` on idle
  cores while the exact activation crosses the wire, then applies the
  correction. The algebra ((A+Δ)·W = A·W + Δ·W) is exact, but the
  economics must be stated honestly: per-token activation deltas are
  *dense* (every element missed its rounding target), so the correction is
  a second full matmul — total compute INCREASES, latency DECREASES,
  funded by otherwise-idle cores (Art. 4 prices it). The sparse/low-rank
  structure in the literature belongs to *weight* residuals
  (GPTQ/AWQ/LoRC), which are static and compensable offline — and this
  repo already exploits that: ternary inference IS the cheap function
  with the correction absorbed at training time (scales, norms, baked
  rotation). Prior art: CPU value prediction (Lipasti/Shen '96,
  Sazeides/Smith '97), multigrid coarse-fine correction (Brandt '77),
  predictor-corrector ODE integrators.
- **Transport Hadamard codec** — a wire codec, distinct from the in-model
  QAT rotation: rotate the residual stream with the WHT (add/sub
  butterflies — `hadamard::rotate_fwd/inv` already exist), entropy-code
  the Gaussian-ized result, un-rotate on arrival. Motivated by massive
  activation outliers (Sun et al. 2024) and incoherence processing
  (QuaRot/QuIP#/SpinQuant).
- **Delta-coding the residual stream — MEASURED, REFUTED.** The proposal
  "transmit the per-layer delta (the derivative of the thought), not the
  state" fails on Bonsai-2: measured over 63 layer boundaries
  (`bonsai27_stream_geometry`), the per-layer delta costs *more* entropy
  than the state at every boundary (e.g. layer 5: 4.74 vs 2.60 bits/value
  on a 256-bin histogram), adjacent-layer cosine is mean 0.980 / min
  0.719 (layer 63 moves the state by 72% of its norm — the stream does
  NOT "barely change"), and the naive adjacent-layer delta-coding
  doubling applies only to replica hops anyway (pipeline receivers don't
  hold the reference state). The transport-codec idea survives only as
  *state* compression — which is lossy quantization again, not
  delta-coding. Measured 2026-09-20; the claim is buried here so nobody
  digs it up unmeasured.

## 8. Execution log

- 2026-09-20: doctrine adopted; P1–P4 scoped; this document written.
- 2026-09-20: **P1 landed** — epoch + chunk index in BMTS v2 capnp header;
  `sync.rs` (plan_pull / plan_diff / rebuild_into, staged + create_new
  against concurrent rebuilders); `ouro-bmts` CLI (info/verify/upgrade/plan).
  Tests: bit-exact rebuild, epoch-mismatch refusal publishes nothing,
  identical-shard zero-byte bill, one-chunk diff targeting. Note: upgrading
  the real 2.8 GB shard was demonstrated (`ouro-bmts upgrade` completed,
  epoch c0151733a64ca917…) but the box was memory-starved (multiple heavy
  sessions; /tmp is 16 GB tmpfs) — re-run `verify` on an idle machine.
- 2026-09-20: **P1.5 landed** — `FrameClass::Reconcile` rides the WORST
  lane (quiet wire); `ReconcileGate` token bucket + hard pause while a
  pipeline stream is live; `ChunkStage` staging with byte budget, TTL and
  LRU eviction, epoch-scoped claims; `IndexAdvert`; `PrefetchRanker` with
  `AdjacentStages` fallback (scheduler `PlannedChunks` wiring is the thin
  follow-on).
- 2026-09-20: **P2 landed** — `FRAME_SPECULATIVE` (hash-first ACTS, 26+8 B);
  `state_hash` fingerprint; contracts: 1000-token same-binary replay hits
  1000/1000, one-ULP flip caught → NACK → fallback stream exact.
- 2026-09-20: **P3 landed (scaffolding)** — `PromptLookup` drafter +
  `generate_speculative` accept/reject loop; losslessness gate passed on
  the real 27B (spec == greedy `[11, 353, 2688, 264]`). Wall-clock win
  awaits the K-column batched verify kernel (documented future).
- 2026-09-20: **P4 landed** — `duet.rs` choice frames + budget
  reconciliation; wire-bill collapse measured (1 bit/as-planned node vs
  8 B f64); set-hash skew → fallback; precision contract refined during
  landing (see §5 note).
- 2026-09-20: **P4 wired into the shell** — `energy?` reconciles
  per-node predicted draw (TDP estimate today; learned power model is
  Art. 4 pending) against live telemetry through `duet::reconcile`,
  prints per-node verdicts + the budget decision. RAPL metering
  upgraded from limit-as-draw to true interval deltas
  (`probe/energy.rs` EnergyMeter).
- 2026-09-20: **multi-turn `ask`** — the shell's Bonsai session is
  persistent across REPL turns (prompt tokens ingest into the
  recurrent state); `ask clear` resets. Demonstrated: turn 1 "Hello" →
  ", I'm a", turn 2 " University" → " of California".

## 9. Provenance (App. C standing rule)

Per CONTRACTS.md meta-rule 4: no novelty claim without a check. Tiered:

### Tier 1 — component techniques: all prior art, cited

| DUET piece | prior art | status |
|---|---|---|
| speculative decode (draft + verify) | Chen et al., *Accelerating LLM Decoding with Speculative Sampling*, arXiv:2302.01318 (DeepMind, 2023) — verified | confirmed |
| speculative decoding (independent) | Leviathan et al., arXiv:2211.17192 (Google, ICML'23) | confident, re-verify |
| prompt-lookup decoding (P3's draft source) | mainline HF transformers: `prompt_lookup_num_tokens`, `max_matching_ngram_size` in GenerationConfig — verified | confirmed |
| chunk-hash delta sync | rsync (Tridgell, 1996); BitTorrent piece hashes; Nix/ostree content stores | confirmed (textbook) |
| hash-verify before trust | content-addressed storage generally | confirmed (textbook) |
| quiet-lane background reconcile | DB replica lag management; kernel readahead; QUIC degradable datagrams | confirmed (textbook) |
| low-rank/sparse quantization residual | GPTQ arXiv:2210.17323; AWQ arXiv:2306.00978; LoRC (low-rank compensation) | confident, re-verify ids |
| massive activation outliers | Sun et al. 2024, *Massive Activations in LLMs*; QuaRot/QuIP#/SpinQuant incoherence processing | confident, re-verify |
| value prediction (surrogate-then-correct on a CPU) | Lipasti & Shen 1996; Sazeides & Smith 1997 (FCM) | confident, re-verify |
| coarse-then-fine correction | multigrid (Brandt 1977); predictor-corrector integrators | confirmed (textbook) |
| deterministic-inference concern | He, *Defeating Nondeterminism in LLM Inference* (Thinking Machines, 2025) | confident, re-verify |

### Tier 2 — composition: uncommon, plausibly unoccupied

No known system combines: (a) the determinism boundary as a *measured,
per-gate contract* selecting the wire mode (same-binary bit-exact vs
cross-silicon residual); (b) placement-plan-ranked epoch gossip on a
degradable lane; (c) speculative activation transfer between replicas of a
sharded LLM (Petals Borzunov et al. 2023 / NeurIPS arXiv:2312.08361 —
verified — loads weights from the HF hub and does no epoch-checked delta
sync; exo ships weights p2p without content-addressed dedup); (d) energy
budgets reconciled as choice frames.

### Tier 3 — the defensible claim

Not "we invented prediction-before-transmission." The defensible claim:
**per-gate falsified trust on mixed-silicon junk hardware** — every
speculative mechanism carries a measured boundary and a tested fallback
(bit-exact rebuild; 1000/1000 same-binary hit rate with ULP-miss NACK;
lossless spec-decode stream; bounded-divergence budget). The ladder is the
contribution; the primitives are the field's.
