# WP-UPDATE — The Tail Rewrites Its Own Genesis

> **Art. 3, taken literally**: "The OS's unit is the cluster resource
> graph, not the host." A tail that must be walked a USB stick is a host
> with extra steps. This document removes the walk. The tail consumes
> its own tail and is reborn — the ouroboros move, made physical.

**Status**: approved 2026-09-06. Build order WP-U1 → U7; each rung is
permanent (no throwaway MVP — the first version of every layer *is* the
final contract, tested against reality at every rung).

---

## The problem

Every image change so far has meant: build → `flash.sh` → walk the stick
to a shelf box → reboot. The image is a live-ISO (tmpfs root,
`nix.enable = false`) by design — stateless cattle. NixOS's native
upgrade path (`nix copy` + `switch-to-configuration`) does not apply and
should not: it requires an installed system, a persistent store, and
imports store corruption as a failure mode.

**Decision**: keep the live-ISO. Two mechanisms replace the stick walk:

1. **Agent hot-swap** — push the 8MB agent binary over the wire; the
   running process `execv`s it in place. No reboot.
2. **Self-reflash** — push the full ISO; the tail writes it onto *its
   own boot medium* and reboots. The running system is RAM-resident; the
   ISO region is unmounted at runtime; this is safe.

## Trust planes (two authorities, never conflated)

| Plane | Key | Lives where | Answers |
|---|---|---|---|
| **Transport** | HMAC-SHA256 shared secret (existing wire) | OURO partition | "Who may speak to this tail?" |
| **Content** | **ed25519** (`ed25519-dalek`) | Private: head-only `keys/update.signing.key` (0600, gitignored — *never* in `enroll/`). Public: committed, baked into the image at `/etc/ouro/update.pub` | "What will this tail believe?" |

A tail verifies content signatures regardless of how the artifact
arrived — wire, USB, later PXE. Trust follows the *signature*, not the
channel. A compromised tail cannot forge updates for the fleet.

**Rotation**: `key_id` field reserved in v1. Rotation = bake the new
pubkey into an image push signed by the old key, then hand over.

## Manifest contract (`ouro-update/1`)

Canonical JSON (sorted keys, no whitespace — deterministic, the Beast
convention), detached ed25519 signature over the canonical bytes:

```
kind = "ouro-update/1"
key_id          # e.g. "2026-09-primary"
artifact        # "agent" | "image"
version         # "git:<rev>"
image_rev       # rev the artifact was built against
size            # bytes
sha256          # hex digest of the artifact
min_agent_protocol
reboot          # bool
created_utc
sig             # ed25519 over the canonical bytes of all of the above
```

**Closure-match rule (Art. 10 honesty)**: an `agent` push requires
`manifest.image_rev == tail.image_rev` — the pushed binary's rpaths
point into the image's nix store. Anything touching nix config is an
`image` push. One canonicalizer (Rust, shared by signer and verifier);
no cross-language canonicalization bugs.

## Wire: frame mode (the promised upgrade)

`auth.rs` has carried "frame upgrade path" in its docstring since WP2.
WP-U1 builds it:

- **Handshake**: signed line `frames begin` on the existing wire flips
  the socket to frame mode. Exit: receiver returns to line mode and
  answers with a signed TaskResult-shaped receipt.
- **Frame**: `magic 0x4F55524F ("OURO") | flags:u8 | seq:u64 BE |
  len:u32 BE | tag:32B | payload` — per the AGENTS.md transport spec.
  Tag reuses the existing `tag(secret, seq, payload)` primitive. Flags:
  `ACK` (payload = u64 BE, highest contiguous data seq received), `EOF`.
- **Flow control**: 256KiB default chunk, cumulative ACK every 16 data
  frames (window = the tail's backpressure; TCP owns reliability).
- **Dual-use**: this framing later carries BMTS model shards
  (DMA_ROADMAP Tier 2+ gets a cross-reference).

## Agent update module (`agent/src/update.rs`)

**Agent artifact** (no reboot):
1. Stream → `/run/ouro/updates/agent.new` → verify sig+sha+closure →
   atomic rename to `/run/ouro/agent-live`; copy binary + signed
   manifest to the OURO partition (survives reboot)
2. `execv` in place — same process, the login wire survives (stdio
   inherited). The 9500 listener dies with exec (CLOEXEC); clients
   reconnect — documented, not excused
3. **Crash-loop rollback**: boot counter on the OURO partition. Enroll
   increments at boot; the agent zeroes it after successful bus
   registration; enroll deletes `agent-live` at ≥3. The baked-in ISO
   binary is the permanent fallback slot. A/B semantics, zero extra
   partitions

**Image artifact** (full system):
1. Idle gate (no task running) + battery guard (discharging → refuse;
   image artifacts only)
2. Stream → OURO partition staging file (not tmpfs — no RAM pressure)
3. **Self-reflash rails**: boot device = the disk whose partition is
   labeled OURO; refuse if `iso_bytes > OURO partition start sector`;
   write only `[0, iso_bytes)`; fsync; **readback sha256 verify** (the
   same law `flash.sh` obeys on the head); delete staging; clean
   `systemctl reboot` (polkit rule ships in the image)
4. Post-reboot: same enroll flow, same secret/keys/head from the
   untouched partition; `find_by_ip` keeps identity (the stale-hostname
   fallback from `64fd26b` already covers DHCP changes)

## Version observability

`OURO_BUILD_REV` stamped by `agent.nix`; `/run/ouro/image-rev` stamped
in the image. Telemetry → `NodeEntry.agent_version` / `image_rev`,
reconciled by `refresh_entry` with audit events. HISS gauge shows
versions per node; the `drift` verb lists nodes behind head's build.

## Head tooling

`ouro-sign` (Rust bin: keygen / sign / verify, canonical JSON) and
`tools/ouro-update`:

- `agent` / `image` subcommands: build (nix) → manifest → sign →
  frame-push → await receipt → confirm the version change
- **Canary law**: image pushes require QEMU-prove PASS, then one canary
  node, rejoin + stable heartbeat 60s, explicit confirm before fleet.
  Agent pushes roll node-by-node. No fleet-wide blast, ever.

## Test ladder (fullest state, tested at every rung)

| Layer | Test |
|---|---|
| Unit | canonical JSON determinism; manifest verify (tamper / wrong key / wrong kind); frame codec roundtrip (random payloads); tamper rejection (payload / seq / magic / len); size-guard math; boot-counter transitions |
| Integration | frame pump over loopback TCP with a live ack window; verify+install flow over temp files |
| **QEMU prove-update** | **The decisive rung**: boot ISO_A in QEMU, push ISO_B over the real wire, tail reflashes its own virtual disk, reboots, reports version B. Full self-reflash loop, zero hardware |
| Hardware | HP: agent hot-swap receipt → HP: self-reflash receipt → laptop joins on a wire-pushed image |

## Build order

| Rung | Deliverable |
|---|---|
| **WP-U1** ✅ | Frame wire: `cluster/src/transport/frames.rs` + codec/pump/window tests — final-ack law proven live (break-on-first-ack RST'd the peer mid-receipt) |
| **WP-U2** ✅ | Signing: `ouro-sign` bin, ed25519-dalek, canonical JSON, manifest verify — ceremony rehearsed end-to-end on a real 8MB agent artifact; tamper rejected |
| **WP-U3** ✅ | Version plumbing: build stamps → telemetry → registry → HISS — drift verified live against the bus |
| **WP-U4** ✅ | Agent update module + image changes — full transaction proven live in sandbox (push → receipt → install → boot_check exec handoff) |
| **WP-U5** ✅ | `tools/ouro-update`: push, canary flow — `selftest` runs the whole U4 contract in a sandbox |
| **WP-U6** ✅ | QEMU prove-update: A (guard refuses a full-staged overrun) + B (frames → staging → guard → raw write → readback → reboot → rejoin, 360s) — ALL PASS |
| **WP-U7** | Hardware receipts + docs (HANDBOOK, ARCHITECTURE, FLEET §10) |

**Bootstrap honesty**: the first update-capable image still needs one
last physical flash. After that, the stick becomes a permanently
installed peripheral — never touched again.

**Cost note (Art. 4)**: image pushes respect the idle gate and the
battery guard; the canary law bounds blast radius. Update dispatch is a
control-plane operation and does not route through the energy-budget
scheduler — but a tail mid-task is never yanked.
