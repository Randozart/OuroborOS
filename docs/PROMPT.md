# The HISS Command Shape

> The prompt is not decoration. It is a gauge, a joke, and a hiss.

Every command you type into HISS is preceded by the shell hissing at
you. The hiss is precise: its **length is the size of your cluster**,
its **capitalisation is chaos**.

## Spec

```
\e[1;31mHiSs\e[0m\e[2;31mSsS\e[0m \e[31m»\e[0m
└─ base: bold crimson └─ tail: dim grey-red   └─ caret
```

### Length — literal topology gauge

```
total_s = 2 + topology.node_count()
```

The base word `hiss` carries two `s`s. Every tail device in the
topology adds exactly one more. The length is never random and never
capped — the prompt counts your cluster at you:

| Tails | Prompt (one render of many) |
|-------|------------------------------|
| 0 | `HiSs »` |
| 1 | `hisSs »` |
| 2 | `HisSss »` |
| 5 | `hiSssSss »` |
| 40 | `Hiss…` (42 s's — earned) |

An idle head hisses politely. A full rack hisses long. The prompt is
measured admission in miniature (Art. 6): the shell does not claim a
cluster size, it *sounds* one.

### Capitalisation — chaos

Every letter, base and tail, is independently upper- or lower-cased at
random, re-rolled on every render (`HiSs`, `hISS`, `Hiss`…). Length is
truth; caps is chaos. The entropy source is a 20-line xorshift64 seeded
from clock nanos — no `rand` dependency, because this is theatre, not
crypto.

### Colour

- Base word: **bold red** (`\e[1;31m`) — the brand crimson, loud.
- Tail `s`s: **dim red** (`\e[2;31m`) — greyed-out red; the hiss fades
  as it trails away.
- Caret: red `»`.

### Render cadence

One fresh render per command. Each Enter re-hisses: same length (the
topology did not change mid-keystroke), new caps (it might have).

### Fallbacks

| Environment | Prompt |
|-------------|--------|
| TTY, colours allowed | full hiss (ANSI, zero-width-wrapped for readline) |
| `NO_COLOR` set | `hiss{…} > ` — glyph and gauge intact, colour gone |
| stdin/stdout piped | *no prompt at all* — clean `printf '?\n' \| ouro-hiss` scripting |

The FIFO face (`ouro-ttyd`) and every wire protocol never see the
prompt — it exists only in the interactive REPL loop.

### History ("hisstory")

Interactive input goes through **rustyline**: up-arrow history, Ctrl-R
search, proper cursor handling. Entries persist to
`~/.ouro/hiss_history` (loaded at start, saved on exit). Ctrl-C clears
the current line; Ctrl-D exits cleanly.

## Decision log

- **Why a gauge?** Because the shell's first duty is honesty about the
  machine (CONSTITUTION Art. 1: hardware has no semantics, only
  capabilities — the prompt reflects measured topology, not vibes).
- **Why random caps?** A hiss is not a steady tone. Uniform text reads
  like a config file; wobbling case reads like breath.
- **Why dim grey-red for the tail?** One loud red brand tone, and the
  echo decaying — the hiss has dynamics instead of a second colour.
- **Why `»`?** Double angle, Latin-1: renders in every monospace font
  since 1990, no font roulette (see the glyph discussion this design
  was born from).
- **Why rustyline and not `read_line`?** A shell without history has
  amnesia. The prompt got teeth; the input line got a memory.
- **Why no cap on tail length?** Forty boxes should produce an absurd
  prompt. The absurdity is the audit.
