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
| `ouro-ttyd`, getty-shim, HMAC impl | **not started** — this is the work |

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

### WP3 — `ouro-agent --stdio-tty` getty-shim *(repo)*

Per PLAN §18.3 T2:

- Same line protocol over stdin/stdout; a slave's getty line spawns it;
  master's `ouro-ttyd` connects via SSH pty (or raw serial).
- **Zero install**: any booted Linux with a login joins the graph.
- Upgrade path: identical frames ride raw L2 (EtherType 0x88B5) later;
  ttyd swaps transport behind the same FIFO face. Nothing above ttyd
  changes.

### WP4 — D-kit minimum: `discover.` *(repo, optional for first test)*

- The getty path needs no subnet sweep, but `discover.` kills the
  hardcoded-IP anti-pattern the moment a second node exists. At minimum:
  registration must produce graph-attributes, not config entries
  (AGENTS.md anti-pattern list).

## 4. R2 physical checklist

| # | Task | Route |
|---|---|---|
| [ ] | LAN: R2 on master's subnet (switch or direct cable) | both |
| [ ] | **A — live USB (recommended first test)**: boot CachyOS live session, log in, getty-shim joins. Zero disk changes, fully reversible | fast |
| [ ] | **B — durable dual-boot**: gparted live shrink of existing disk (owner-approved), CachyOS-minimal headless, shared ESP | after A passes |
| [ ] | `nvidia-580xx-dkms` install (route B only; live USB can defer) | B |
| [ ] | Mask sleep/suspend targets — R2 suspend/resume is broken (PLAN §2.3: "shutdown may sleep instead") | B |
| [ ] | `OURO_GPU_NAME=960` style pinning check once probe runs on R2 | both |

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

- [ ] HMAC secret provisioning: manual copy (node #1) vs deploy-carried.
- [ ] Live USB (A) first — assumed yes; confirm.
- [ ] First model: bitnet-2.4B smoke → 9B Q6_K split (assumed); veto to change.
- [ ] Where watts truth for R2 comes from (estimate vs USB meter) — only
      matters if the energy contract (L5) is asserted for the R2 stage.
