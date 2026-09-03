# OuroborOS Architecture

> **OUROBOROS**: **O**ne **U**nified **R**untime **O**rchestrating
> *a* **B**unch **O**f **R**andom **O**ld **S**ervers.
> The machine that remakes itself. The tail
> feeds the head.

This document is the internals reference. For usage, see
[`HANDBOOK.md`](HANDBOOK.md). Design authority is
[`../CONSTITUTION.md`](../CONSTITUTION.md); every section cites its
article.

Terminology: **head** = control-plane machine (registry, shell); **tail**
= node running the agent. Tasks flow head → tail; telemetry flows
tail → head. (Historically "master/slave"; retired.)

---

## 1. Crate map

```
ouroboros/
├── cluster/          ouro-cluster — the machine's body
│   ├── beast/        S-expression graph: topology, node state, codec
│   ├── probe/        cpu, memory, energy (RAPL), gpu, network measurement
│   ├── scheduler/    placement: classes, capability score, budget, queue
│   ├── registry/     node records, lifecycle events, bus protocol (bus.rs)
│   ├── error_recovery.rs  failure tracking, cooldown, stale sweep
│   ├── transport/    auth.rs (the wire), tcp/ssh stubs
│   ├── pipeline.rs   ACTS activation framing + stage plans
│   ├── infer.rs      shard loading, dequant, layer forward
│   └── bmts.rs       BMTS weight shard format
├── shell/            ouro-hiss — the machine's mouth
│   ├── main.rs       HISS REPL (banner, prompt, verb loop)
│   ├── parser.rs     lexer + Command enum
│   ├── propositions.rs  verb handlers against topology + scheduler
│   ├── context.rs    sticky node context + live property cache
│   ├── formatter.rs  output shaping, poetry register
│   ├── agent_client.rs  signed-wire client (ping/telemetry/execute)
│   └── bin/          ouro-hiss, ouro-ttyd, ouro-registry, ouro-pipeline
├── agent/            ouro-agent — the machine's hands
│   ├── main.rs       TCP daemon + --stdio-tty shim + message dispatch
│   ├── head_link.rs  push client: register + heartbeat to the head
│   ├── telemetry.rs  live measurement (/proc, RAPL, thermal, nvidia-smi)
│   ├── executor.rs   task execution
│   └── stage.rs      pipeline stage runner
├── nixos/            node image: flake.nix, node-image.nix, agent.nix
└── tools/            flash.sh (USB), wp7_prove.py (QEMU acceptance)
```

## 2. Data flow

```
        tasks (head → tail)            telemetry (tail → head)
  ┌──────────────────────────┐   ┌──────────────────────────────┐
  │                          ▼   ▼                              │
┌─[ HEAD ]──────────────────────────────────────────┐
│  ouro-registry ←── head_link (register/heartbeat) │
│      │ Registry: records, events, JSON state      │
│      ▼                                            │
│  ouro-hiss: topology + scheduler + budget + queue │
│      │ agent_client: signed execute/telemetry     │
└──────┼────────────────────────────────────────────┘
       ▼ signed wire (§3)
┌─[ TAIL ]──────────────────────────────────────────┐
│  ouro-agent :9500  (or --stdio-tty via getty/ssh) │
│      ├── telemetry: /proc, RAPL, thermal, GPU     │
│      ├── executor: task JSON in → result JSON out │
│      └── stage: ACTS activations in → out         │
└───────────────────────────────────────────────────┘
```

Two channels, one wire format:
- **Push channel** (registry bus): tail announces and heartbeats; the
  head never sweeps. `Art. 6` — polling a sleeping machine is a default
  to attack.
- **Task channel** (agent :9500 / stdio shim): head dispatches, tail
  answers. Every task passes the budget gate first (`Art. 4`).

## 3. The wire (exact)

All channels share one authenticated line format
(`cluster/src/transport/auth.rs`):

```
<seq> SP <64-hex-tag> SP <body> \n
```

- `seq` — u64, decimal. Sender increments; receiver requires an exact
  echo of the request seq in the reply.
- `tag` — `HMAC-SHA256(secret, seq.to_be_bytes(8) || body_bytes)`,
  lowercase hex. The seq rides big-endian so every implementation hashes
  identically.
- `secret` — 32 bytes, from `OURO_SECRET_FILE` (64 hex chars text).
  Provisioned out-of-band (OURO partition / manual copy). It crosses no
  wire, ever.
- Verification is constant-time (`verify`), and `open_line` returns one
  opaque error for structural failure, bad hex, or tag mismatch — no
  oracle for guessers.

Rules that are load-bearing:
1. **Both directions are signed.** An unsigned reply is an error, not a
   warning.
2. **No bypass flag.** Agents and daemons refuse to start without the
   secret; there is no `--insecure` (`Art. 10` — contracts are the fixed
   point).
3. Body bytes are UTF-8; everything else on the line is ASCII.

## 4. Registry bus protocol (`cluster/src/registry/bus.rs`)

One signed exchange per TCP connection to `ouro-registry` (default
:9501). The peer IP of the socket *is* the node's identity anchor —
self-reported IPs are ignored (`Art. 9` — a node's claim about itself is
a default; the socket is a measurement).

| Request (body) | Reply (body) | Notes |
|---|---|---|
| `ping` | `pong` | liveness |
| `register <telemetry-JSON>` | `registered <id>` | idempotent per IP: re-registering keeps the id, refreshes profile + last_seen |
| `heartbeat <telemetry-JSON>` | `ok <id>` | updates power/temp/load/status, refreshes last_seen |
| `heartbeat …` (unregistered IP) | `unknown` | agent re-registers |
| `register/heartbeat <bad JSON>` | `err bad-json` | |
| anything else | `err unknown-verb <verb>` | |

Telemetry JSON mirrors agent telemetry (`hostname`, `cpu_model`, `cores`,
`threads`, `has_avx`, `has_avx2`, `has_sse42`, `ram_total_mib`,
`power_watts`, `temp_c`, `load_avg`); every field has `#[serde(default)]`
so older agents still parse. Status is derived, not claimed:
`load_avg > 2.0 → Working`, else `Idle`.

`ouro-agent --head <addr>` runs `head_link.rs`: register on boot, then a
heartbeat every 5s; any `unknown`, bad reply, or broken wire drops back
to re-register with a 5s backoff. The link is fire-and-forget from the
agent's perspective — the task server runs regardless.

## 5. Registry

`Registry` maps `node_id → NodeRecord`:

```
NodeRecord
├── entry: NodeEntry        static profile (CPU, SIMD flags, RAM, GPU, TDP)
├── state: NodeState        live (status, power_watts, thermal_c, load, assignment)
├── registered_at / last_seen (epoch seconds)
└── tags: Vec<String>
```

- **IDs**: `n1, n2, …` — max existing `n<num>` + 1; stable across
  re-registration (per-IP idempotence), freed on `unregister`.
- **Liveness**: a record is *alive* if `now - last_seen < 30s`
  (heartbeat window). Past that it is *offline* — the 10s daemon sweep
  reports it and `recover.` can act.
- **Events**: `NodeJoined`, `NodeLeft {reason}`, `NodeUpdated {fields}`,
  `NodeStateChanged {from, to}` — appended to an in-memory journal and
  consumed by error recovery.
- **Persistence**: `--state path.json` — pretty JSON, saved on every
  mutation, loaded on boot. Without it, state is honest and ephemeral.
- **Bridge**: `to_topology()` renders the graph for the scheduler and
  HISS.

## 6. Scheduler (`cluster/src/scheduler/`)

`Scheduler::schedule(task)` is the only route to a node (`Art. 11`):
nothing — HISS verb, FIFO line, future API — may place work around it.

1. **Filter** by workload class:
   `BranchHeavy | Recursive | Irregular | Unknown` run anywhere;
   `SimdFriendly | LlmInference` need AVX or AVX2;
   `SmallBatch` needs SSE4.2.
2. **Rank** by capability score:

   ```
   score = (gpu_bucket << 16) + (simd << 8) + (100 - min(tdp_watts, 100))
   gpu_bucket: 12 if vram ≥ 8 GiB, 8 if ≥ 4 GiB, 4 if any GPU, else 0
   simd:       avx2 → 3, avx → 2, sse42 → 1, else 0
   ```

   Decode is bandwidth-bound, so VRAM dominates; SIMD breaks ties; the
   cheapest adequate machine wins (`Waste is Fuel`).
3. **Gate** on `EnergyBudget`: commit only if
   `current + estimate ≤ budget`. Exceeded → task is enqueued, not
   dropped.
4. **Queue** (`task_queue.rs`): priority-ordered `VecDeque` (higher
   priority inserts ahead; retry pushes to the back), `max_retries = 3`
   per task, optional deadlines (`expire()`), hard cap 1000, `drain()`
   re-offers everything when conditions change (budget raised, node
   joined, `recover.`).
5. **Release**: `complete(watts)` returns energy to the budget; queued
   work is then drainable.

## 7. Error recovery (`cluster/src/error_recovery.rs`)

- `report_failure(node)` increments a per-node count; at
  `max_failures = 3` the node should be marked offline.
- Success (a heartbeat) clears the count.
- Failed nodes are in **cooldown** for 30s (not retried), and stale past
  a 300s recovery timeout.
- `sweep_stale(registry)` finds records past the heartbeat window;
  `cleanup` forgets tracking for nodes no longer registered.
- `recover.` (HISS) sweeps, reports, and drains the queue so displaced
  work finds a new tail.

## 8. Probes (`cluster/src/probe/`)

Truth comes from files and tools, never from claims (`Art. 1`):

| Field | Source |
|---|---|
| CPU model/cores/threads, SIMD flags | `/proc/cpuinfo` (`flags` line; end-of-line tokens handled) |
| TDP estimate | model-name heuristic (i9/i7 45W, i5 35W, i3/Cel 25W, Pent 15W) |
| RAM | `/proc/meminfo` `MemTotal` |
| Power, RAPL | `/sys/class/powercap/intel-rapl:0/{energy_uj,power_limit}`; fallback 35W estimate |
| Temperature | `/sys/class/thermal/thermal_zone*/temp` (agent) |
| GPU | `nvidia-smi --query-gpu=…` merged with `vulkaninfo --summary` by model name |
| Network | ICMP RTT (`ping -c3`), bandwidth (dd over ssh), TCP RTT to agent |

Remote probing shells out over SSH and parses the same files. Agent
telemetry reports `has_avx`, `has_avx2`, `has_sse42` explicitly;
nothing is inferred from its neighbor.

## 9. Node image boot sequence (`nixos/`)

Stateless by design — the tail is remade from the graph at every boot
(`Art. 3`; the OS is the agent):

```
ISO boot (console=ttyS0 primary; serial is a first-class console)
└── systemd:
    ouro-probe   node_id = sha256(SMBIOS uuid | MAC)[0..16]; WoL flag
    ouro-enroll  findfs LABEL=OURO (15×1s udev race retry) → mount ro
                 → /run/ouro/secret (0600, ouro:ouro)
                 → authorized_keys for the head
                 → breadcrumbs to console + /run/ouro/enroll-status
    ouro-brand   random motto from /etc/ouro/taglines → /run/ouro/issue
                 (backronym, motto, node_id, secret state, enroll trail)
    getty@tty1 + serial: autologin ouro, --no-issue, issue file override
    → ouro-agent --stdio-tty
         isatty(stdin) → prints the banner; piped → clean protocol
```

No partition / no secret ⇒ `secret: REFUSED` ⇒ the agent refuses the
wire. Enrollment never lies silently.

Acceptance is executable: `tools/wp7_prove.py` boots the image in QEMU,
attaches an OURO-labeled drive, drives the getty shim over raw serial
with signed traffic, and asserts brand + enrollment + ping→pong +
tagline under HMAC.

## 10. Inference stack

- **BMTS** (`bmts.rs`): weight shard format for raw placement; shards
  are pushed checksum-aware (`deploy shards.`).
- **ACTS** (`pipeline.rs`): activation framing between stages —
  `sequence, token_pos, layer bounds, dims, bytes`; the pipeline plan
  maps layer ranges to node ids.
- **infer**: loads a real shard, dequantizes rows, runs a layer forward;
  tests run against actual tensors, not mocks.
- Parity law: any recompiled placement must reproduce token-for-token
  output and the watts row, or it is rejected (`Art. 10`).

## 11. What is deliberately not here yet

Error-recovery-driven live reassignment (policy exists, executor wiring
pending), checkpoint/hot-swap of weights, raw L2 transport (EtherType
0x88B5), PXE boot, scx integration. The runbook
([`R2_BRINGUP.md`](R2_BRINGUP.md)) tracks hardware-day acceptance.
