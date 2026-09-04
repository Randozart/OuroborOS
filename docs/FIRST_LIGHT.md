# FIRST LIGHT — 2026-09-03

> The report of the day the wyrm ate its first real slab, and the plan
> for the fixes it flushed out. Companion: `R2_BRINGUP.md` §11 (the
> runbook), `PROMPT.md` (the gauge), `DMA_ROADMAP.md` (the appetite).

## What happened

An **HP Pavilion Power** (i5-7200U, 2C/4T, 8GB DDR4, eno1 ethernet,
**Intel HD 620 iGPU + NVIDIA GTX 1060 6GB dGPU** — GPU_CLAIM.md WP-N
claims both) booted the OuroborOS node image from the flashed SanDisk
stick, enrolled from the OURO partition, joined the head's registry
bus across the LAN, and heartbeat-telemetred for the rest of the
evening:

```
n1: nixos @ 192.168.1.114
    Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz — 2C/4T
    7829 MiB RAM
    35W · 42→46°C · load 0.05–0.99 · [Idle]
    112+ events (NodeJoined + heartbeats) — persisted through registry reads
    ARP: REACHABLE; signed queries answered over SSH (port 22)
```

Zero bytes written to the HP's disk. Pull the stick, it's a normal PC.
Stick in, it's the wyrm's. Identity as a removable decision (Art. 1).

## The gauntlet (every blocker + its fix, in order)

| # | Blocker | Symptom | Fix |
|---|---------|---------|-----|
| 1 | flash.sh secret validator | died before dd | validator passed the hex as a grep *filename* arg — bash regex now (`89ef817`) |
| 2 | flash.sh free-space math | lsblk multi-line into `$(( ))` | parse one line; `ready` no longer lies (`3c9f3e4`) |
| 3 | flash.sh readback | mismatch after sfdisk | compare sectors 1–8, skip the MBR sfdisk rewrites (`2643c1c`) |
| 4 | Secure Boot | "Selected bootimage did not authenticate" | BIOS: Secure Boot OFF (one-time; `shutdown /r /fw` from Windows reaches setup without key combos) |
| 5 | Windows Fast Startup | Esc/F9 dead at power-on | hybrid shutdown hides the firmware; `/fw` reboots or Shift+Shut Down |
| 6 | Windows Update | stole boot turns | persistent boot order: SanDisk above Windows Boot Manager — stick wins when present, Windows boots normally when absent |
| 7 | **Boot loop on real HW** | GRUB → `_` → GRUB, forever | **`nomodeset`** (`ca17df9`): the minimal image's GPU drivers panicked during KMS init on the HP's iGPU. Tails are headless — KMS buys nothing |
| 8 | Head registry absent | "couldn't authenticate" = Connection refused | the daemon must actually run: `tools/ouro up` (the launcher exists because of this) |
| 9 | **Tail firewall** | task channel (9500) times out | **stock NixOS firewall is ON** in the image; outbound joins worked, inbound tasks died. Fix in this batch: `networking.firewall.enable = false` — the signed wire is the gatekeeper |
| 10 | **Double banner, literal ANSI** | colored banner, then a white one with `\e[31m` as text | two bugs, one reflash: agetty `-f issue` AND the agent's isatty banner both print (collision), and the issue file contains literal `\e` chars (nix `''` strings + heredocs don't interpret them; agetty happens to, the agent's raw print doesn't) — fix in this batch |

Lessons 1–3 were all caught by the gates *before touching hardware*;
7–10 could only be learned by booting real silicon. That's why both
halves exist.

## What works right now (no changes needed)

- Registry liveness + heartbeats: `ouro status`, `registry.json`
- **Signed queries over SSH** (port 22 is the one port the tail's
  firewall allows): ping/telemetry/diag/tagline piped through
  `ssh ouro@192.168.1.114` — verified end-to-end with HMAC both ways
- The hiss prompt gauge: first tail = one grey-red extra s

## The fix batch (this reflash)

1. **Single banner**: agetty loses `-f /run/ouro/issue`; the agent's
   isatty banner is the only one (portable: serial join path needs it)
2. **Real ESC bytes**: `ouro-brand` writes the issue via `printf`
   (which interprets `\e`) instead of a heredoc (which doesn't) —
   colors render everywhere, no literal-ANSI screens
3. **Tail firewall off**: `networking.firewall.enable = false` — the
   signed wire is the gatekeeper (Art. 10), the fabric is LAN-isolated
   (DMA_ROADMAP §procurement); port 9500 becomes reachable for tasks

## Opens after this batch

- **Registry↔HISS unification**: hiss still walks its own topology;
  the HP lives in the registry daemon. `discover.`/`probe` close it per
  session; the real fix is hiss consuming registry state
- **First distributed compute**: once 9500 is reachable,
  `n1 assign <workload>` becomes actual execution on the tail
- **Naming**: nodes answer to the ISO default `nixos`; measured-style
  or user-set hostnames are a one-liner for a future reflash
- **Alienware**: 180W Dell adapter outstanding; joins the same stick
- **Laptop**: same stick, F12, wired ethernet
