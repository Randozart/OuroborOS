# GENESIS — Gate-Enforced Natural Evolution of the Self-Improving Substrate

> The governance constitution for the loop: a process running on
> OuroborOS that improves OuroborOS. The machine that remakes itself —
> with teeth that are contracts. Every clause cites an article of
> `CONSTITUTION.md` (Art. 11: design must cite the law first).
>
> **Status**: law before code. No loop process may be deployed until
> its mutation pathway satisfies every section here.

## The loop

```
SENSE      heartbeat telemetry + Praetor benchmarks + HISS logs
   ↓
PROPOSE    the loop emits a candidate: Nix derivation, kernel,
           scheduler policy, Briev pass, workload placement
   ↓
GATE       the ladder below — each rung can kill the candidate
   ↓
DEPOSE     a gate-passing candidate becomes the next generation
   ↓
MEASURE    the new generation's fitness feeds the next SENSE
```

The loop's capability already exists in pieces: telemetry flows
(heartbeats carry watts/load/temp), the scheduler routes, Praetor
benchmarks shadows, Nix builds hermetically, contracts gate placement.
GENESIS names the missing piece: **governance** — who may propose, who
may depose, and what may never be touched.

## I. Propose vs. depose (Art. 11)

**The loop may propose. Only the gates may depose.**

A self-emitted derivation becomes the running generation only by
passing the full ladder, in order:

1. **Build gate** — the derivation must build hermetically. A candidate
   that doesn't build is not a proposal; it is noise.
2. **Contract gate (Art. 10)** — the parity ladder re-runs against the
   candidate: cos > 0.999 on kernels, greedy-top1 equality on
   placements, wire-format round-trips. A recompiled system that fails
   a contract is rejected, not excused.
3. **Budget gate (Art. 4)** — the candidate's projected energy profile
   must fit the current budget. A faster machine that costs more watts
   than the budget allows is not an improvement.
4. **Benchmark gate (Praetor)** — the candidate must beat the incumbent
   in a shadow benchmark on the metric it claims to improve. Ties go to
   the incumbent.
5. **Human-signed promotion** — a human promotes the candidate to the
   new generation. This signature may be *delegated* later by a
   constitutional amendment creating a **change budget** — the way
   Art. 4 delegates watts — never silently.

**No candidate skips rungs. No gate trusts another gate's era**: gates
re-run against the candidate itself, not cached verdicts.

## II. The fitness function

Official improvement metrics, all *already flowing* through existing
telemetry and tooling:

| Metric | Source |
|--------|--------|
| tokens/sec per stage | ACTS stage timings |
| perplexity | model eval harness |
| watts/token | heartbeat power ÷ stage throughput |
| transport latency | activation-probe / wire RTT |
| scheduling p99 | task queue wait times |
| memory footprint | telemetry ram_used + weight residency |

A proposal names **which metric it improves** and **what it risks**.
The benchmark gate runs the named metric; the rollback SLA watches all
of them.

## III. Rollback SLA

Any generation that regresses any official metric beyond its claimed
trade-off is **atomically reverted** (Nix generations) — no debate, no
grace period. Reversion is an event on the bus like any other
(`NodeStateChanged` semantics: the cluster's state includes which
generation is running).

## IV. Out-of-band authority — the watchdog outranks the wyrm

The loop runs *on* the machine; therefore something not run by the
machine must be able to stop it:

- **Serial power authority (WP-SER, DMA_ROADMAP)**: RTS/DTR → PWR_SW
  relay. Out-of-band on/off/hard-reset, reachable when the OS is not.
- **`recover.` semantics**: stale-node sweep, cooldown, manual
  unregister — the head may always amputate a tail; the operator may
  always amputate the head's current generation.
- **The kill switch outranks every process** — including any process
  that edits the rules about kill switches. (That's Section V.)

## V. Prohibited mutations

The loop may never propose changes to:

- `CONSTITUTION.md` — the law is not self-amendable by its subject
- `enroll/`, the secret, any key material (Art. 10)
- The gate code itself — a checker that checks itself is not a checker
  ( Praetor, the contract ladder, and the budget gate are load-bearing
  walls; the resident may not renovate them)
- `.praetor/shadow-results.json` — benchmark history is append-only
  via Praetor, never hand-edited (existing anti-pattern rule)
- `enroll` admission semantics — measured admission is how the wyrm
  grows; the loop may not lower its own entrance bar

## VI. Why this is safe at all

The one-sentence version of the whole document:

> **NixOS made bricking the substrate impossible; the contracts made
> lying about improvement impossible; the out-of-band relay made
> ignoring the operator impossible.**

AutoGPT-class failures (the agent corrupts its own environment) are
structurally excluded: a bad proposal fails to build, fails a contract,
exceeds a budget, or loses a benchmark — and dies as a store path, not
as an incident.

*The snake bites its tail. The teeth are contracts. The hand that
feeds it holds the power pin.*
