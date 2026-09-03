# OurobourOS Handbook

> **OUROBOROS**: **O**ne **U**nified **R**untime **O**rchestrating
> **B**unch **O**f **R**andom **O**ld **S**ervers.
> The machine that remakes itself. The tail feeds the head.

**HISS** — the **H**ierarchical **I**nteractive **S**hell **S**ystem — is
how you drive the cluster. Everything else (agent, registry, ttyd) exists
so that HISS has something true to say.

Terminology: the cluster has a **head** (the machine you sit at; runs the
control plane) and **tails** (old boxes that booted the node image; run
the agent). Telemetry flows tail → head ("the tail feeds the head");
tasks flow head → tail. Older docs said master/slave; git history
preserves those words so you can diff, but the words themselves are
retired.

---

## 1. Quickstart (5 minutes, one machine)

Everything needs one shared secret: 64 hex chars, never on the wire.

```bash
# 1. Secret (once; same file on head and tails)
python3 -c "import secrets; print(secrets.token_hex(32))" > /etc/ouro/secret
chmod 600 /etc/ouro/secret
export OURO_SECRET_FILE=/etc/ouro/secret

# 2. Head side: registry daemon (push-based node bookkeeping)
cargo run --release --bin ouro-registry -- --addr 0.0.0.0:9501 --state registry.json

# 3. Tail side (or same box for a smoke test): agent, linked to the head
OURO_PORT=9500 cargo run --release --bin ouro-agent -- --head 192.168.1.10:9501

# 4. Drive the machine
cargo run --release --bin ouro-hiss
```

You should see the agent print `head-link: registered as n1 @ …` and the
registry persist `registry.json`. In HISS:

```
hiss> ?
hiss> n1?
hiss> n1.power?
```

---

## 2. The binaries

| Binary | Crate | Role |
|--------|-------|------|
| `ouro-hiss` | `ouro-hiss` | Interactive shell. Dot notation, propositions, budget, queue, recovery. |
| `ouro-registry` | `ouro-hiss` | Push-based registry daemon. Agents register + heartbeat here. |
| `ouro-agent` | `ouro-agent` | Node daemon: task execution, telemetry, getty-shim, head link. |
| `ouro-ttyd` | `ouro-hiss` | FIFO face for one node (scripting a node from a shell script). |
| `ouro-pipeline` | `ouro-hiss` | Pipeline/stage experiments (W3 lineage). |

### 2.1 `ouro-hiss`

```
cargo run --release --bin ouro-hiss
```

No flags. Starts with a demo topology; `discover.` or the bus fills it
with truth. Prompt is `hiss> `.

### 2.2 `ouro-registry`

```
ouro-registry [--addr 0.0.0.0:9501] [--state <path.json>]
```

- `--addr` — listen address (default `0.0.0.0:9501`).
- `--state` — persist the registry to JSON; load it back on restart.
- Env: `OURO_SECRET_FILE` (mandatory — no secret, no daemon).
- A 10s sweep logs how many nodes are past the heartbeat window (30s).

### 2.3 `ouro-agent`

```
ouro-agent [--head <ip:port>]          # TCP daemon + optional head link
ouro-agent --stdio-tty                 # getty-shim: signed line protocol on stdin/stdout
```

- TCP daemon listens on `0.0.0.0:9500` (override: `OURO_PORT`).
- `--head <ip:port>` — register with the registry daemon on boot, then
  heartbeat every 5s. Re-registers automatically if the daemon replies
  `unknown` (state lost) or the wire breaks (5s backoff).
- `--stdio-tty` — the getty shim: a tail's login line spawns this; the
  head reaches it over `ssh -T` or raw serial. When stdin is a TTY it
  prints the boot banner (brand + this boot's motto); when piped, the
  protocol stays clean.
- Env: `OURO_SECRET_FILE` (mandatory — the agent refuses to start, and
  refuses every unsigned line, without it).

### 2.4 `ouro-ttyd`

```
ouro-ttyd --node n1 --addr 127.0.0.1:9500 [--tty-dir /srv/ouro/tty]
ouro-ttyd --node n1 --pty-cmd "ssh -T ouro@192.168.1.20 -- ouro-agent --stdio-tty"
```

Exposes a node as two FIFOs: `<tty-dir>/<node>.in` and `.out`. One
request in flight (lockstep). Write a line to `.in`, read one line from
`.out`. Task lines route through the scheduler and energy budget — the
FIFO cannot bypass Art. 4.

```bash
echo 'ping' > /srv/ouro/tty/n1.in && cat /srv/ouro/tty/n1.out   # -> ok pong
echo 'budget 120w.' > /srv/ouro/tty/n1.in && cat /srv/ouro/tty/n1.out
```

In: `ping` | `echo <text>` | `stage_setup <path>` | `stage_reset` |
`tagline` | dot-notation shell lines (`budget 120w.`, `n1?`, `probe.`).
Out: `ok <text>` | `queued <reason>` | `err <msg>`.

---

## 3. HISS verbs (complete reference)

Everything is `subject.verb` or bare propositions. Context sticks: `n1`
selects node 1; `cluster` deselects.

### Queries

| Input | Meaning |
|-------|---------|
| `?` or `cluster?` | Cluster summary: nodes, power, budget, GPU census |
| `n1?` | Full node record: CPU, RAM, SIMD flags, GPU |
| `n1.power?` | One property (`power` `ram` `cpu` `cores` `threads` `simd` `gpu` `status`); live cache wins over static profile, suffixed `(live)` |
| `power?` | Same, on the context node |
| `cluster.active?` | Bulk query (`active` `idle` `offline` `sleeping`) |

### Placement (Art. 4 — energy is a first-class constraint)

| Input | Meaning |
|-------|---------|
| `n1 assign branch_sort.` | Route through `Scheduler::schedule()`: class match → capability rank → budget gate. Dispatch or queue, step by step. |
| `branch_sort on?` | Dry-run: would it place? where? why not? |
| `budget 400w.` | Set cluster power budget. May re-place anything, any time. |
| `tasks.` | Task queue: depth, per-task age, retries, priority |
| `recover.` | Sweep stale/failed nodes, drain the queue |

### Fleet truth

| Input | Meaning |
|-------|---------|
| `register.` | Probe this box locally, add it to the topology |
| `unregister n3.` | Remove a node |
| `discover. [cidr] [port]` | One-shot sweep: TCP-probe a /24, pull telemetry from live agents, absorb them (`discover. 127.0.0.1 9501` for localhost) |
| `probe.` | List all topology nodes |
| `save.` / `load.` | Topology ↔ JSON on disk |

### Payloads

| Input | Meaning |
|-------|---------|
| `generate <prompt>.` | BitNet generation on the target node |
| `shards.` | Pipeline plan + activation transport probe |
| `deploy.` | Ship the agent binary to known node addrs (SSH/SCP) |
| `deploy shards.` | Checksum-aware shard sync |
| `n1休眠.` or `n1 sleep.` | Sleep transition (stub) |
| `poetry on.` / `poetry off.` | Output register |

### Example session

```
hiss> ?
hiss> discover.
hiss> n2
hiss> power?
hiss> budget 120w.
hiss> n2 assign llm_decode.
hiss> tasks.
hiss> recover.
```

---

## 4. Enrollment: how a tail joins

A tail boots the NixOS node image (`nix build ./nixos#node-image`,
flash with `tools/flash.sh`). On boot, three services run in order:

1. **`ouro-probe`** — derives `node_id = sha256(SMBIOS-uuid | MAC)` to 16
   hex chars. Identity is *measured*, never stored. Also records Wake-on-LAN.
2. **`ouro-enroll`** — finds the partition labeled `OURO`, mounts it
   read-only (15s retry for the udev race), installs `secret` (0600,
   owner `ouro`) and the head's `authorized_keys`. Writes breadcrumbs to
   `/run/ouro/enroll-status` and the console at every step. No partition
   → `secret: REFUSED` → the agent refuses the wire. Enrollment never
   lies silently.
3. **`ouro-brand`** — picks this boot's motto at random from
   `/etc/ouro/taglines`, stamps `/run/ouro/issue`: backronym, motto in
   crimson, node id, secret state, enroll breadcrumbs.

Then getty auto-logins `ouro` and spawns `ouro-agent --stdio-tty`. The
banner prints on any real TTY (console, serial, SSH); over pipes the
line protocol stays clean. The head talks to it with the same signed
wire as the TCP daemon — proven end-to-end by `tools/wp7_prove.py`
(QEMU: brand, enrollment, ping→pong, tagline, all under HMAC).

To re-enroll or re-key: rewrite the OURO partition, reboot. The node
keeps no other state — it is remade from the graph, not from disk.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Error: OURO_SECRET_FILE not set` / agent refuses to start | Mandatory secret gate | Export it; generate §1 if needed |
| `stream did not contain valid UTF-8` on secret load | Secret file is raw bytes | Store 64 hex chars as text |
| Banner says `secret: REFUSED` | OURO partition missing/unreadable | Check `enroll:` line in banner + `ouro-enroll` console lines; re-flash |
| Registry replies `unknown` to heartbeat | Daemon restarted without `--state`, or node's IP changed | Agent re-registers automatically; nothing to do |
| `head-link: … — retry in 5s` | Daemon down / wrong `--head` address | Check registry is up; verify address |
| Task shows `queued (queue depth: N)` | Budget exceeded or no capable node | `tasks.` to inspect; `budget 600w.` or free a node; `recover.` drains |
| Node listed but `?` shows it idle with stale watts | Heartbeat gap > 30s | Node offline — `recover.`; check its link |
| `discover.` finds nothing | Agents not listening on that port | `discover. <cidr> 9500`; prefer the push bus over sweeps |
| FIFO `.out` shows `err empty line` | Wrote newline only | Each request must be one non-empty line |

---

## 6. Hygiene rules (from the constitution)

- Never bypass `Scheduler::schedule()` — not from HISS, not from the FIFO.
- Never hardcode node IPs in code; discovery and the bus exist for that.
- The secret crosses no wire — only HMAC tags do. `OURO_SECRET_FILE` is
  provisioned out-of-band (the OURO partition).
- A placement that fails its contract is rejected, not excused (Art. 10).
- If output looks wrong, suspect the default, not the machine (Art. 6).
