# Bonsai Tokenizer — Plan (text in, text out, no C)

**Status**: planned (this document is the plan; implementation lands under
`cluster/src/infer/tokenizer.rs`). Goal: HISS serves actual text on Bonsai
end-to-end — `ask "Hello"` → tokens → greedy generation → streamed text —
with zero oracle trust: every gate is anchored to token ids the PrismML
fork already emitted in verified capture runs.

## Ground truth (read from the GGUF, 2026-09-20)

- `tokenizer.ggml.model = gpt2` — byte-level BPE
- `tokenizer.ggml.pre = qwen35` — **custom pretokenizer**, implemented in
  the fork's `llama-vocab.cpp` as `LLAMA_VOCAB_PRE_TYPE_QWEN35`
- vocab: 248320 tokens, 247587 merges, token_type array
- `eos = 248046`, `bos = pad = 248044`, `add_bos_token = 0` (no BOS prepend)
- chat template present (vision + thinking Jinja) — **out of scope**; base
  completion only for now
- The vocab lives ONLY in the GGUF: BMTS shards carry no tokenizer today,
  and the Rust engine has no tokenizer at all (bitnet-rs tokenizes via the
  C fork — Bonsai doesn't use that path).

## The qwen35 pretokenizer (verbatim from the fork)

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

This is the Qwen2 pattern with `\p{L}` widened to `[\p{L}\p{M}]` (letters +
marks) in two places. It contains the negative lookahead `(?!\S)` — the
plain `regex` crate cannot compile it, so this plan pins **`fancy-regex`**
(backtracking engine). Hand-rolling lookahead semantics is exactly the
subtle-wrongness class the contract ladder exists to prevent; the dep is
the honest choice.

## Phases

### A — Extract (`tools/dump_tokenizer.py`)

GGUF → `shards_bonsai27_n*/tokenizer.json`: tokens, token_type, merges,
special ids (eos/bos/pad), `pre` type, `add_bos`. Rust never reads GGUF
for vocab (no Rust GGUF parser exists here; Python already has `read_kv`).
~15–20 MB JSON, loaded once per process.

### B — BPE engine (`cluster/src/infer/tokenizer.rs`, new)

- **Pretokenizer**: the regex above, compiled once; text → word pieces.
- **Byte-level alphabet**: GPT-2 `bytes_to_unicode` table (tokens are
  stored escaped — `Ġ` = space etc.). Encode text bytes → alphabet for
  matching; decode back.
- **Merge ranks**: position in the merges list; lowest rank wins per BPE
  step; then vocab string → id. (`scores` are irrelevant to BPE ranking
  for gpt2-style vocab; carried in JSON for completeness.)
- **Detokenizer**: id → string → bytes, with streaming partial-UTF-8
  buffering (a multi-byte char split across two tokens must not emit a
  replacement character mid-stream).
- Special tokens: `token_type` respected on decode; eos = generation stop.

### C — Wire into the model

- `Card::load_dir` picks up `tokenizer.json` when present.
- `Qwen35Model::tokenize(&str) -> Vec<usize>`,
  `detokenize(&[usize]) -> String`, streaming detokenizer handle for HISS.
- HISS: `ask <text>` — tokenize → greedy generate (existing
  `step`/`logits` loop) → stream decoded text; eos stops.

### D — Gates (all oracle-anchored; zero new trust)

| gate | assertion | anchor |
|---|---|---|
| encode | `tokenize("Hello") == [9419]` | ouro-capture printed token id (verified run) |
| decode | `detokenize([11, 353, 2688, 264, 5286, 303, 279, 3694]) == ", I'm a student in the University"` | the verified oracle greedy stream + captured pieces |
| roundtrip | ascii / CJK / emoji / contractions / digits / trailing spaces / newlines survive encode∘decode | tokenizer's own contract |
| end-to-end | text → tokens → 6-token greedy → text on real shards (`#[ignore]`, ~30 s release) | the whole stack |

Parity stretch: extend `ouro-capture` to tokenize a paragraph and diff
against ours token-for-token (the fork as oracle for arbitrary text).

## Choices

| choice | decision | rejected |
|---|---|---|
| vocab source | Python dumps `tokenizer.json` at shard time | Rust GGUF parser (doesn't exist here) / C fork call (defeats the no-C goal) |
| regex engine | `fancy-regex` (lookahead) | plain `regex` (cannot compile `(?!\S)`) / hand-rolled scanner (subtle wrongness) |
| chat template | out of scope | Jinja engine — its own session |
| BOS | none (`add_bos_token = 0`) | prepending (would change token stream vs oracle) |
