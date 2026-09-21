# P5a–P5c: Appetite Protocol — Implementation Plan

The head declares what the cluster is optimized for; tails adapt
themselves to match. The appetite travels inside the heartbeat
response — zero new wire protocol.

## Wire Format

`AppetiteFrame` embedded in heartbeat: `"ok {id} {json}"`. Old agents
ignore the extra JSON (they check `starts_with("ok")`).

## Fast Path (<1s)

- RAPL power limit via sysfs write
- Bond lane repricing (pure function of edge set)
- Scheduler budget update + `drain_queue()`

## Slow Path (minutes, P5d)

- NixOS profile switch (driver swap, kernel module load)
- Deferred to a separate implementation phase

## Contract Gate

After reconfiguration, telemetry reports new `power_watts`.
The head's `energy?` verb runs `duet::reconcile` — existing
parity ladder + energy reconciliation catches drift.

## Steps

1. `cluster/src/duet.rs` — AppetiteFrame + MemoryProfile + hash
2. `cluster/src/probe/energy.rs` — set_rapl_limit
3. `cluster/src/registry.rs` — pending_appetite field + set/clear
4. `cluster/src/registry/bus.rs` — embed in heartbeat response
5. `agent/src/head_link.rs` — parse + apply
6. `shell/src/parser.rs` — appetite / app command
7. `shell/src/propositions.rs` — Appetite arm
8. Tests (all of the above)
9. `docs/DUET.md` — P5 spec
10. Full workspace test + clippy
11. Commit + push

## Provenance

- RAPL sysfs: Intel Powerclamp, standard Linux interface
- Heartbeat piggyback: zero-cost, existing wire
- NixOS profiles: declarative, reproducible, Art. 3
