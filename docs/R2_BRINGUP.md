# R2_BRINGUP.md — Alienware Alpha R2 as first cross-chassis node

Plan of record, 2026-09-02 session. **Status: planning only — nothing
implemented.** Owner explicitly deferred implementation.

This is the concrete execution of the T-track in `PLAN.md` §18.3–§18.4
("T2 = R2/IdeaPad bring-up ... this IS the M3 physical-wiring step"),
specialized to the R2. Hardware facts live in `docs/FLEET.md` §1;
topology in `CLUSTER.md`.

---

## 1. Why R2 first

- Cheapest full test of the thesis: master (3060) + one slave over 1GbE,
  token-exact parity across a real wire (L3 contract, `docs/CONTRACTS.md`).
- Needs **no master changes**: the 3060 runs on r610 today; the r580 swap
  stays paired with the W2 two-GPU milestone (PLAN §18.5 S5), not with this.
- GTX 960 4GB = smallest stage; if the pipeline works at 4GB with a 35W
  TDP host CPU, it works everywhere else in the fleet.

## 2. Current state (what already exists)

| Piece | State |
|---|---|
| 9B token-exact over 4 TCP localhost agents | done (M3, in-process equivalent) |
| 27B pure-Rust CPU forward | **[!] blocked on master RAM/swap** — FLEET.md §2 (independent of R2) |
| wgpu Q6_K matvec, adapter pinning | done, S1–S3 committed (G-rung: 2.6 ms, 31.5x) |
| r580 driver decision | solved on paper: `nvidia-580xx-dkms` in CachyOS repo; **avoid linux-lts kernel** (AUR note re 6.12.x) |
| R2 disk shrink | owner-approved; recipe = gparted live (single SATA bay) |
| HMAC design | §9.3 exists: 32B HMAC-SHA256 over header+payload, shared secret |
| `ouro-ttyd`, getty-shim, HMAC impl | **all done (2026-09-02)** — WP1 ttyd FIFO face, WP2 HMAC, WP3 getty-shim + `--pty-cmd` wire |

## 3. Work packages (in dependency order)

### WP1 — S4/T1: `ouro-ttyd` FIFO face + loopback demo *(repo, testable today)*

Per PLAN §18.3 T1, restated as build spec:

- New bin in agent crate: bridges `/srv/ouro/tty/<node>.in` and `.out`
  FIFOs to the transport client for one node.
- Line protocol: in = one task (`stage_step <hex>`, `probe`,
  `budget 120w.`, dot-form); out = one response (status + optional hex
  continuation).
- Lockstep: one request in flight (matches `stage.rs` sequential
  semantics; costs nothing on TTY paths).
- **Gate**: TTY path == TCP path == in-process, token-id-equal
  (parity ladder L2 analog for the FIFO face).
- Every op still routes through scheduler + budget check — no
  device-file bypass of Art. 4.

### WP2 — HMAC §9.3 implementation *(repo; mandatory gate before R2)*

- Frame auth: HMAC-SHA256, 32 bytes, over header+payload, shared secret.
- **[ ] OPEN DECISION — key provisioning**: manual copy to the first
  slave (acceptable for node #1) vs. deploy tooling carries the secret
  from day one. Owner to decide before WP3 ships.
- Threat model note (§9.2): without this, the TTY face is an
  unauthenticated execution orifice on the LAN. Loopback demo first,
  **HMAC before R2** — already written into §18.3.

### WP3 — `ouro-agent --stdio-tty` getty-shim *(repo)* — **DONE 2026-09-02**

Per PLAN §18.3 T2:

- Same line protocol over stdin/stdout; a slave's getty line spawns it;
  master's `ouro-ttyd` connects via SSH pty (or raw serial).
- **Zero install**: any booted Linux with a login joins the graph.
- Upgrade path: identical frames ride raw L2 (EtherType 0x88B5) later;
  ttyd swaps transport behind the same FIFO face. Nothing above ttyd
  changes.

Plan of record (2026-09-02):

- **Same authed wire on stdio** (`seq tag body`, WP2 core). SSH
  authenticates the channel; our HMAC stays because the transport
  swap to raw L2 must be a no-op above the wire (Art. 10: same
  contract everywhere, no special cases) and it costs <1%.
- Agent: `ouro-agent --stdio-tty` — stdin line → `authed_process` →
  signed stdout line, flush per line. Auth fail → `err auth`, exit
  (getty respawns = fresh login). EOF → exit 0.
- Master: `ouro-ttyd --pty-cmd '<cmd>'` (replaces `--addr`) — spawns
  the command once per tty connection, persistent child, lockstep
  signed lines over its pipes. One SSH handshake per tty session, not
  per request. Respawns the child if it dies.
- Master uses `ssh -T` (no pty allocation — no echo corruption); raw
  serial paths must `stty -echo` for the same reason.
- Secrets: same `OURO_SECRET_FILE` on both ends (manual copy, §8).

### WP4 — D-kit minimum: `discover.` *(repo, optional for first test)*

- The getty path needs no subnet sweep, but `discover.` kills the
  hardcoded-IP anti-pattern the moment a second node exists. At minimum:
  registration must produce graph-attributes, not config entries
  (AGENTS.md anti-pattern list).

## 4. R2 physical checklist

| # | Task | Route |
|---|---|---|
| [ ] | LAN: R2 on master's subnet (switch or direct cable) | all |
| [ ] | **Node image stick** (§10): `tools/flash.sh` a USB, set boot order, boot. Zero disk changes, fully reversible. Plain CachyOS live = fallback | image |
| [x] | ~~B — durable dual-boot: gparted shrink~~ **cancelled** (§8: node image replaces it) | — |
| [ ] | `nvidia-580xx` driver in NixOS — deferred to its own WP; image v1 is CPU-only (smoke needs no GPU) | later |
| [ ] | Suspend masking — handled inside the image config (§10 WP5) | image |
| [ ] | `OURO_GPU_NAME=960` style pinning check once probe runs on R2 | later |

## 5. R2 constraints to respect (PLAN §2.3)

| Constraint | Consequence for this test |
|---|---|
| 35W TDP CPU | stage sizing small; CPU micro-op pools thin — keep heavy layers on master |
| GTX 960 4GB | light-layer group only (§14 packing already prices it lowest) |
| Single SODIMM | halved memory bandwidth — expect honest probe numbers to reflect it |
| 1GbE only | fine: ACTS frames are 20–50KB/token; wire is not the bottleneck (§14) |
| AGA port | ignore (Windows-only, undocumented on Linux) |

## 6. Acceptance test (definition of "tested with the Alienware")

1. R2 boots Linux (live USB counts), user logs in, getty spawns
   `ouro-agent --stdio-tty`.
2. Master `ouro-ttyd` connects; R2 registers; its probes (memcpy, gemv,
   RTT, power estimate) land in the Beast graph — measured-only
   admission, nothing from spec sheets.
3. Pipeline runs: bitnet-2.4B for first smoke, then 9B Q6_K split
   master-stage → LAN cut → R2-stage.
4. **Parity**: token ids equal to in-process reference (L3).
5. Watts row appended (RAPL on master; R2 estimate flagged as estimate
   until/unless a meter exists).
6. All traffic authenticated (WP2 live on the wire).

## 7. Dependency map

```
WP1 (ttyd + loopback) ──> WP2 (HMAC) ──> WP3 (getty-shim) ──> R2 test
                                     └─ WP4 (discover., parallel-safe)
R2 physical: LAN + live USB can be prepared any time; joins at WP3.
Master AM5 swap (FLEET.md §2) is independent — unblocks 27B, not this.
W2 two-GPU demo (S5) is independent — both feed the 27B decision.
```

## 8. Open decisions for owner

- [x] HMAC secret provisioning: **manual copy to node #1** (decided
      2026-09-02; deploy-carried revisited at fleet scale).
- [x] Live USB (A) first — **confirmed**; upgraded to the node image
      (§10). Plain CachyOS live demoted to 30-min fallback.
- [x] First model: bitnet-2.4B smoke → 9B Q6_K split — **confirmed as
      assumed**.
- [x] Boot OS: **USB-booted OurobourOS node image** (2026-09-02,
      §10). Route B (gparted shrink + CachyOS-minimal dual-boot) is
      **cancelled** — R2's disk is never touched; the owner's shrink
      approval goes unspent.
- [x] Taglines: all six mottos baked into the image; **each boot picks
      one at random** (kernel entropy, stateless-safe). Master echoes
      a node's tagline in crimson at registration — the boot reveal
      happens on the host screen.
- [x] Node identity: **derived, never stored** — `node_id` =
      hash(SMBIOS product_uuid, MAC fallback) computed by the boot
      probe. Roles ("host"/"sleeper") are never persisted; they are
      priced at schedule time (Art. 1, Art. 4). WoL/suspend support
      is measured (`ethtool wol`, R2 `sleep_ok=false`) and lands in
      the Beast graph as attributes.
- [ ] Where watts truth for R2 comes from (estimate vs USB meter) — only
      matters if the energy contract (L5) is asserted for the R2 stage.

## 9. Execution record

- 2026-09-02: decisions above locked; implementation started at WP1
  (`ouro-ttyd` FIFO face + loopback demo). Order: WP1 → WP2 (manual
  provisioning) → WP3 → physical (live USB).
- 2026-09-02: **WP1 done.** `ouro-ttyd` lives in the shell crate
  (master-side; `agent_client` is there — deviation from "agent crate"
  noted), module `shell/src/ttyd.rs`, bin `shell/src/bin/ouro-ttyd.rs`.
  Line protocol: `ping` / `echo` / `stage_* <payload>` / other agent
  tasks / dot-form (`budget 120w.`, `probe.`, `n1?`) in; `ok|queued|err`
  out. Lockstep (one request in flight). Every task routes through
  `Scheduler::schedule()` + budget check (Art. 4) — `budget 0w.`
  provably queues. Gate test green: FIFO path == TCP path == in-process
  (`test_fifo_loopback_end_to_end`, `test_echo_parity_tty_tcp_inprocess`,
  ACTS hex parity); verified again through real `ouro-agent` +
  `ouro-ttyd` processes on real FIFOs. Workspace `cargo test --lib` +
  `cargo clippy -- -D warnings` clean.
- Next: **WP2 HMAC §9.3** — plan of record (2026-09-02, owner-approved):
  - **Crypto core is frame-agnostic and frame-ready**: `tag(secret, seq,
    payload) -> [u8;32]` (HMAC-SHA256) + constant-time verify, pure
    functions in `cluster/src/transport/auth.rs`. Drops into the OURO
    frame trailer unchanged when L2/frames land.
  - **Line encoding (interim), flat, zero-copy**: `<seq> <64hex-tag>
    <body verbatim>\n`. Tag over `seq || body`. No JSON envelope — an
    envelope would re-serialize every 20–50KB ACTS payload per hop;
    flat prefix costs one `splitn(3)`.
  - **Cost**: HMAC-SHA256 software ~0.5–1 GB/s on R2-class CPU (no
    SHA-NI) = 25–100µs per 50KB frame vs ≥5–50ms token step → <1% of
    the token loop. Cheaper than debugging one silently corrupted
    activation (L3 parity contract).
  - **Both directions signed.** Client verifies the response tag over
    the request's seq. Auth failure → terse unsigned `err auth`, no
    detail (no oracle).
  - **Mandatory gate**: no `OURO_SECRET_FILE` → process refuses to
    start. No bypass flag. Secret file = 64 hex chars = 32 bytes;
    manual provisioning (§8, node #1).
  - **Honest limitation (Art. 10)**: the wire is connect-per-request,
    so seq gives correlation, not server-side anti-replay. HMAC buys
    integrity + authenticity now; replay protection arrives with
    persistent frame connections (WP3/L2 upgrade path, §3).
  - Deps: `hmac` + `sha2` crates (audited; hand-rolled MACs rejected).
- 2026-09-02: **WP2 done.** Core: `cluster/src/transport/auth.rs`
  (`tag`/`verify`/`sign_line`/`open_line`/`secret_from_env`). Agent:
  eager secret load, `authed_process` wraps `process_message`, auth
  failure = unsigned `err auth` + close. `agent_client`: seq counter +
  secret cache, signs every request, verifies reply tag over the
  request's seq; `*_with` variants expose explicit secrets (tests).
  `TtySession` owns a secret — construction is the gate. Note: the
  FIFO face itself stays master-local plaintext; auth is a wire (TCP)
  property — `agent_client` signs at the transport boundary. Tests:
  tag determinism/seq+key sensitivity, tamper/wrong-key/structural
  reject (opaque errors), unsigned-reply reject, wrong-key exchange
  reject, full ttyd parity ladder green on the authed wire. Real
  daemons: agent + ttyd with generated secret, FIFO → signed TCP →
  echo/ping/budget/stage_step/real-ACTS-frame all round-trip; raw
  unsigned TCP ping and zero-tag both bounce `err auth`. Workspace
  `cargo test --lib` (115) + `cargo clippy -- -D warnings` clean.
- Next: **WP3** — `ouro-agent --stdio-tty` getty-shim: same line
  protocol on stdin/stdout (authed the same way), getty spawns it on
  R2, master's ttyd connects via SSH pty.
- 2026-09-02: **WP3 done — repo work packages complete.** Agent:
  `ouro-agent --stdio-tty` (`serve_stdio`, agent/src/main.rs) — signed
  line in, signed line out, flush per line; auth fail → `err auth` +
  exit (getty respawn = fresh login); EOF → clean exit; refuses to
  start without `OURO_SECRET_FILE`. Master: `ouro-ttyd --pty-cmd
  '<cmd>'` (`TtyWire::{Tcp,Child}`, shell/src/ttyd.rs) — spawns the
  command once per tty connection (one SSH handshake per session, not
  per request), lockstep signed lines over its pipes, dead child
  detected + respawned on next use; `--addr` (TCP) and `--pty-cmd`
  are mutually exclusive; getty/stdio nodes get no `node_addrs`
  entries (proposition probing skips them honestly). Real-daemon
  smoke, full chain: FIFO → ttyd → scheduler/budget → `sh -c` child →
  real `ouro-agent --stdio-tty` → signed stdio → back: ping / echo /
  `budget 90w.` / bench_sum / real 42-byte ACTS frame (seq=9 pos=5
  intact) all round-trip. Negative: wrong-secret ttyd vs agent →
  every request `err auth` (dead wire, opaque). `cat`-bridge +
  Cursor-level unit tests cover both halves hermetically. Workspace
  `cargo test --lib` (117 lib + 19 agent) + `cargo clippy -- -D
  warnings` clean. **Next: physical R2 checklist §4** — LAN +
  CachyOS live USB (route A), getty spawns the shim, join at §6
  acceptance test. WP4 `discover.` stays deferred to the second-node
  milestone.

---

## 10. The node image — USB-booted OurobourOS (plan of record, 2026-09-02)

The zero-install thesis, generalized: the OS **is** the agent. One stick
per cluster device; adding a node = flash + boot-order + boot. Any x86
box in the pile joins the graph with zero disk changes. Route B is dead
(§8); plain CachyOS live is the 30-minute fallback only.

### WP5 — `nixos/` node image (build once, boot anywhere)

```
nixos/
├── flake.nix              # nixos-generators, iso/usb image format
├── node-image.nix         # the node config (everything below)
└── agent.nix              # package: ouro-agent, no-default-features, musl static
```

- **Pure-Rust static agent**: `ouro-agent` built `--no-default-features`.
  The R2 role (`stage_setup/step/token/sample`) runs on
  `ouro_cluster::infer` — pure Rust (agent/src/stage.rs). The `bitnet`
  feature (C++ llama.cpp via bindgen — musl-hostile) is only needed for
  full-model-on-node tasks, which are not in the §6 acceptance test.
  True single-binary image, no `.so` bundling.
- **Stateless cattle**: read-only squashfs root + tmpfs. Node identity
  comes from the boot probe each boot; nothing is stored, so nothing can
  drift (measured-only admission, Art. 10: nothing trusted carried).
- **Boot brand service**: reads `/etc/ouro/taglines`, picks one at
  random (kernel entropy), renders truecolor crimson
  (`\e[38;2;220;20;60m`, 256-color `\e[38;5;160m` fallback) on black.
  Pre-login `issue` banner + post-login state line:
  `node <id> · measured admission · secret: ok|REFUSED`.
- **Tagline pool** (§8: all six, random per boot):
  ```
  it knows what it is.
  devour the default.
  no purpose but use.
  the tail feeds the head.
  one wire. one budget. one machine.
  nothing declared. everything measured.
  ```
- **Boot probe service**: `node_id` = hash(SMBIOS
  `/sys/class/dmi/id/product_uuid`, MAC-of-lowest-NIC fallback);
  graph attributes: `wol` (ethtool), `sleep_ok` (R2 = false, measured),
  plus the standard memcpy/gemv/RTT/power probes at registration.
  **No role flags anywhere** — "host"/"sleeper" are prices, not
  identities (Art. 1); the control plane relocates per Art. 4.
- **Enrollment artifacts**: boot service mounts the labeled `OURO`
  partition → `/run/ouro/secret` (32B hex HMAC secret) + master SSH
  pubkey → `authorized_keys`. Missing partition = no secret = agent
  refuses the wire (WP2 gate holds, no bypass).
- sshd: key-only auth. getty autologin tty1 → `ouro-agent --stdio-tty`.
  sleep/suspend targets masked (absorbs route B's leftover task).
- **v1 is CPU-only**: `nvidia-580xx` is not packaged for NixOS; GPU
  driver in Nix = separate later WP. §6 smoke (bitnet-2.4B → 9B split)
  needs no GPU on R2.

### WP6 — `tools/flash.sh` (one command per stick)

`dd` the image → create the labeled `OURO` partition → write the secret
file + master pubkey → verify readback. Extends §8's manual-copy
decision to per-stick provisioning; sticks are keys — custody matters.

### WP7 — QEMU prove-out (before any physical USB)

Boot the image in QEMU on the master: banner + random tagline, `OURO`
partition consumed, signed wire up, `ouro-ttyd --pty-cmd` (QEMU as the
child) completes the loopback. The entire Nix ramp is debugged with
zero physical risk. **Gate: install Nix on master first** (none
present as of 2026-09-02); WP5 files are written to be
correct-by-inspection until then.

### Master-side cinematic echo

`ouro-ttyd` prints a node's tagline in crimson on the host terminal at
registration. Transport: the agent's first response carries
`tagline <text>` as its body — one line over the existing WP2 signed
wire, zero new protocol surface. Boot-up reveal happens where the
operator is looking.

### Deferred, explicit

- GPU driver in Nix (image v1 CPU-only)
- Persistence partition (until nodes run for weeks)
- Enrollment service (fleet scale; per-stick flash until then)
- netboot/iPXE (no USB at all) — pairs with WP4 `discover.`
- WoL wake-on-demand scheduling ("sleeper" as an energy state priced by
  the scheduler, watts vs capacity gained — Art. 4)

### Sequencing

1. WP5 scaffolding → WP7 QEMU loop (needs Nix on master) → WP6 flash
2. Physical (parallel): R2 LAN + boot-order
3. First light: image stick in R2 → §6 acceptance test
4. Owner works `docs/brand/` SVG/Braille logo in parallel — TTY banner
   renders whatever drops in (art is separate from the tagline pool).

- 2026-09-02: **§10 node image kickoff.** WP5 scaffolding written:
  `nixos/flake.nix` (nixos-generators, dd-able ISO), `nixos/agent.nix`
  (pure-Rust agent, no-default-features; honest note: workspace
  Cargo.lock vendoring unproven until Nix exists), `nixos/node-image.nix`
  (stateless config: getty autologin → agent shim, sshd keys-only,
  sleep targets masked, crimson console palette remap, brand/probe/
  enroll services, tagline pool). WP6: `tools/flash.sh` (dd + OURO
  partition + secret/pubkey + verify; syntax-checked only — destructive
  tool, needs hardware to truly prove). Master-side cinematic echo:
  agent answers `tagline` (OURO_TAGLINE env or /run/ouro/tagline);
  `TtySession::motto()` rides the signed wire (schedules 1W like any
  op); `serve_connection` writes the crimson banner as the first
  `.out` line before serving requests — consumers skip ANSI lines.
  Verified: full workspace `cargo test --lib` (137) + agent (20) +
  `cargo clippy -- -D warnings` clean; FIFO e2e test now asserts the
  banner line. **Blocked, explicit: no Nix on master** (checked
  2026-09-02) — WP7 QEMU prove-out gate. Owner: install Nix, then
  `nix build .#node-image` (expect one cargoLock hash dance), and
  work `docs/brand/` logo in parallel.

- 2026-09-02: **WP7 done — ALL PASS.** `tools/wp7_prove.py`: boots the
  node image under QEMU (TCG, 3G), OURO-labeled enrollment drive
  attached, drives the getty-spawned shim over the raw serial line with
  signed wire traffic. Accepts: brand banner + random tagline (differs
  per boot — pool pick proven), enroll breadcrumbs visible on console,
  secret consumed, `node_id` derived, ping→pong and tagline round-trips
  verified under HMAC. Two consecutive clean passes, different
  taglines. Debug findings baked back into the image:
  - enroll needs `-g ouro` → user now owns group `ouro`; enroll writes
    breadcrumbs (`enroll-status`, console echoes) — the banner shows
    the enroll reason, never a silent REFUSED
  - findfs by-label race at early boot → 15s retry loop
  - agetty skips banners under autologin → the agent prints the issue
    banner itself when stdin is a TTY (isatty); pipes (`ssh -T`, FIFO
    face) stay clean protocol — the banner travels with the agent
  - serial getty (ttyS0) = raw-serial join path, brand included
  Flakes fixed along the way: nixos-generators incompatible with
  unstable's customisation internals → dropped, ISO built directly via
  `iso-image.nix`; crates.io 403s curl's default UA → fetchurl overlay
  injects identifying UA; nixpkgs bumped to unstable for rustc ≥1.87
  (`is_multiple_of`). NOTE: the WP7 debug image bakes the test SSH key
  (`ouro-wp7-debug-shell`); production flash strips it (one line in
  `nixos/node-image.nix`). Rust: 137 lib + 20 agent tests, clippy
  `-D warnings` clean. **Next: physical R2** — flash a stick
  (`tools/flash.sh`), boot-order, §6 acceptance test. Brand `docs/brand/`
  has the crimson ouroboros SVG; Braille TTY variant pending from owner.
