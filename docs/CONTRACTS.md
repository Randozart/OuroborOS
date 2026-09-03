# Contract Registry — the Parity Ladder

The trust spine of OuroborOS (CONSTITUTION.md Art. 10). Any hardware
re-placement, backend swap, budget recompile, kernel rewrite, or node
failure/rejoin re-passes the rungs below, bottom-up. **A contract that
cannot be re-verified blocks the action it guards.**

Status key: PASSING | WIP | GAP (design settled, code pending)

---

## L0 — Kernel truth (bit-for-bit vs substrate)

*Every quantization kernel we write decodes identically to the C reference
in the vendored fork.*

| Contract | Threshold | Command | Status |
|---|---|---|---|
| TQ1_0 == `dequantize_row_tq1_0` | `to_bits` equality, 25,600 real-tensor elems | `cargo test -p bitnet-rs --test verify_infer test_tq1` | PASSING |
| Q8_0, Q4_K == C | bit-exact, 1024 pseudo-random blocks | `... test_kquant` | PASSING |
| Q3_K, Q5_K, Q6_K == C | bit-exact, 512 blocks each | `... test_q356` | PASSING |
| future: IQ types, BF16, etc. | bit-exact before any use | extend verify_infer | GAP |

Caught: K_SCALE_SIZE=12 stride trap; q3k output-cursor overwrite; a Rust
operator-precedence bug in the 2-bit extract. The ladder's record is the
argument for the ladder.

## L1 — Pool parity (same math, different silicon)

*CPU-MT, and later wgpu/Vulkan and CUDA pools, agree on op results.*

| Contract | Threshold | Status |
|---|---|---|
| fused/AVX dot kernels == scalar reference | cos > 0.9999 per output vector | GAP (AVX1/wgpu sessions) |
| wgpu Q6_K/Q4_K/Q3_K gemv == CPU engine | cos > 0.999, greedy-equal stage outputs | PARTIAL — Q6_K gemv cos 1.0 (G1/G2, 2026-08-31, 31.5x scalar); Q4_K/Q3_K + stage gate = G4/W3 |

## L2 — Stage parity (the same brain in different bodies)

| Contract | Threshold | Command | Status |
|---|---|---|---|
| synthetic toy 3-shard: TCP == in-process | token-exact | `cargo test -p ouro-hiss --test pipeline_test` (fast tier) | PASSING |
| BitNet-2.4B real shards, 3 agents == in-process | token-exact `[374, 264, 2678, 3363, 11]` | `--test pipeline_test -- --ignored` | PASSING |
| Qwen3.8-9B, 4 agents == in-process | token-exact `[17018, 7529, 998, 14541, 364]` | `--test qwen_tcp -- --ignored` | PASSING |
| out-of-order position contract per stage | hard reject | agent slot check | PASSING |

## L3 — System equivalence (any placement == reference placement)

*The machine's answer does not depend on where its parts sit.*

| Contract | Threshold | Status |
|---|---|---|
| Qwen3.8-9B full forward == llama.cpp oracle | logits cos >= 0.999 + greedy top-1 equal (measured 0.9994) | PASSING (`--test qwen_diff`) |
| Qwen3.8-27B full forward == oracle | cos >= 0.999 + top-1 (measured 0.99993 @2614) | PASSING |
| layer-0 9-tensor differential incl. 524K-float delta state | every tensor cos > 0.999 | PASSING |
| PlacementPlan X vs reference plan on identical tokens | streams identical, else resync from checkpoint | WIP (compiler pending) |
| post-relocation re-verify (node drop -> re-place) | L1-L2 re-pass within N seconds before first token | GAP (choreography pending) |

## L4 — Performance contracts (the plan must beat its bound, or explain)

| Contract | Threshold | Status |
|---|---|---|
| M4: dense 27B >= 10 tok/s on 4-GPU pipeline (est. ~30–35) | measured, PLAN §13.4 | GAP (post-wgpu) |
| M4: 35B-A3B MoE >= 30 tok/s | measured | GAP |
| achieved/bound-gap displayed (bytes/token / eff. BW) | every bench run | WIP — bridge benches (§16.3) report it |
| hop RTT vs schedule prediction within 2x | pipeline telemetry | WIP |

## L5 — Energy contracts (watts are law, not suggestion)

| Contract | Threshold | Status |
|---|---|---|
| admitted tasks never exceed budget sum | scheduler check, RAPL-reported by every agent | WIP (budget exists; placement-weighted energy model pending) |
| `budget 120w.` recompile: post-migration RAPL sum <= budget | measured during first request | GAP |
| W/token reported per node, per run | shell census | WIP (telemetry feeds it) |

---

## Meta-rules

1. **Order**: a run re-passes from the lowest rung its change touched
   (kernel edit -> L0; placement move -> L1-L3; backend swap -> L1-L3).
2. **Near-tie rule**: fp-order greedy flips are resolved by comparing
   *conditional streams* (same history -> same token), not raw equality;
   persistent divergence = resync from checkpoint (never "looks close").
3. **Freshness**: plans carry a TTL; measured profiles that fail
   re-benchmark on heartbeat invalidate plans -> recompile (Condor/exo
   churn pattern adopted, PLAN §15.10).
4. **Provenance**: no claim of novelty enters a README without a §15
   check (Constitution App. C standing rule) — the meta-contract.
5. **Gate**: `tools/ci.sh` = L0-L2 on every commit; `tools/ci.sh --heavy`
   runs L2-L3 ignorable tier nightly.
