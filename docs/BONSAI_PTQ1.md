# Bonsai-2-27B (PrismML PTQ1_0) — Format & Oracle Notes

Hard-won facts from the differential bring-up (2026-09-20). The L3 gate
(CONTRACTS.md) passes at logits cos 0.999998, greedy top-1 equal.

## PTQ1_0 (GGUF dtype 143) — 28 bytes per 128 trits

NOT a base-3 big-integer. It is TQ1_0's byte magic re-staged at block 128
(`block_ptq1_0` in ggml-common.h):

```
struct { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; }  // 28 B
```

Element order per group (dequantize_row_ptq1_0, stages {32,16,8} — only
16 and 8 execute for qs[24]):

- elements `n*16 + m`  ← digit n of `qs[m]`,       n∈0..5, m∈0..16
- elements `80 + n*8 + m` ← digit n of `qs[16+m]`, n∈0..5, m∈0..8
- elements `120 + n*2 + h` ← digit n of `qh[h]`,   n∈0..4, h∈0..2

Digit extraction (per byte): `q = byte * 3^n (u8 wrap); xi = (q*3) >> 8;
value = (xi - 1) * d`. Rust: `tl1_group` in `cluster/src/infer/dequant.rs`.

`matvec_tl1` (ops.rs) fuses decode + dot with an SSE4.1 kernel — the
fleet floor (oldest node is pre-AVX2 Ivy Bridge): 8 bytes -> u16 lanes,
u8-wrap scale by 3^n, `(q*3)>>8` digit, two 4-lane f32 FMAs per group
(`__m128` is 4-wide; the "obvious" 8-lane f32 chain silently corrupts
lanes 4-7 — caught by the fused-vs-reference test). Scalar fallback kept.
Throughput on the lm_head shape (248320x5120): 1.75 GB/s, ~0.16 s;
whole-token step 17.5 s on an i7-3770 (8 threads).

## Hadamard conjugation — sign order matters

`H[i][j] = (-1)^popcount(i & j) / sqrt(N)`, block N=1024, `H_s = diag(s)·H`.
The fork uses the two conjugates (llama-graph.cpp):

| site | op | order |
|---|---|---|
| `build_lora_mm` (folded matmul) | `x' = H·(s ∘ x)` = H_s^T x | signs FIRST |
| embedding lookup recovery | `h = s ∘ (H·z)` = H_s z | signs AFTER |

Rust: `hadamard::rotate_fwd` / `hadamard::rotate_inv`. Composed they give
identity — unit-tested.

Sign metadata: `prism.hadamard.sign_mode=explicit`, per-width slices keyed
by the weight's input dim (Bonsai-2: 5120 / 6144 / 17408). Extracted by
`tools/extract_hadamard.py` into `hadamard.json` next to the shards;
consumed via `Card::load_dir`.

### GDN V-grouped ssm_out

`prism.hadamard.gdn_v_grouped = 1`: ssm_out rows are stored in grouped
[hd, rep, nk] feature order (hd=128, rep=3, nk=16). The ACTIVATION is
permuted before the transform:
`g[k*384 + r*128 + d] = o[(r*16 + k)*128 + d]`.

## dtype 30 = BF16 (not f16)

token_embd is PTQ1_0 ternary ("Hadamard-latent"); ssm_alpha/ssm_beta are
bf16. `QuantKind::from_dtype(30) → Bf16`.

## Oracle tooling

The vendored `bitnet-cpp/3rdparty/llama.cpp` does NOT speak PTQ1_0. Oracle
is the PrismML fork (`PrismML-Eng/llama.cpp`, branch `prism`):

```bash
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp
cp tools/prism-oracle/* <fork>/tools/ouro-capture/
echo "add_subdirectory(ouro-capture)" >> <fork>/tools/CMakeLists.txt
cmake -B <fork>/build -S <fork> -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build <fork>/build --target ouro-capture -j

<fork>/build/bin/ouro-capture -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -p "Hello" \
    -ngl 0 -t 8 -o /tmp/opencode/bonsai_oracle_logits.f32 \
    -d "model.input_embed,attn_norm-,linear_attn_qkv_mixed-,z-,attn_residual-,ffn_out-,post_ffn-,l_out-,result_norm"

cargo test --release -p ouro-cluster --test bonsai_diff -- --ignored --nocapture
```

`-n N` greedy-decodes N tokens (prints the stream; `-o` holds the last
step's logits) — used by `bonsai27_greedy_stream_diff`.

`-d` prefixes name graph tensors (llama-context `cb`) captured through the
ggml eval callback; useful drill-down taps for qwen35: `model.input_embed`,
`attn_norm-N`, `linear_attn_qkv_mixed-N`, `z-N`, `attn_residual-N`,
`ffn_out-N`, `post_ffn-N`, `l_out-N`, `result_norm`.
