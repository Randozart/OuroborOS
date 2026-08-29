# Qwen3.8 (arch `qwen35`) Port Spec — from vendored fork, 2026-08-29

Source of truth: `bitnet-cpp/3rdparty/llama.cpp/src/models/qwen35.cpp` (644 lines)
+ `src/models/delta-net-base.cpp` (606 lines). Fork **runs Qwen3.8-9B on CPU**
(1.3 tok/s measured) — oracle live via `BitNetModel::decode_capture`.

## Layer typing
- `full_attn_interval = 4` (KV, default): layer `il` is **delta/linear** if
  `(il+1) % 4 != 0`, **full attention** if `== 0`. (27B: 64 layers = 48 delta + 16 attn)

## Shared per-layer flow (qwen35.cpp build_layers loop)
```
h    = RMSNorm(x) * attn_norm            # LLM_NORM_RMS = x/sqrt(mean x^2 + eps) * w
o    = attn_kind(h)                      # delta or full-attn below
x1   = x + o                             # attn_residual
f    = RMSNorm(x1) * attn_post_norm      # post-attention norm gates FFN
x2   = x1 + FFN(f)                       # post_ffn -> l_out
```
FFN: standard PAR SwiGLU `down( silu(gate@u) * (up@u) )` (no MoE in dense).
Final: `output_norm` RMS then lm_head (`output` tensor; **tied to tok_embd when absent** — 27B GGUF has no `output.weight` → tied ✓).

## Delta layer (gated delta net) — exact ops
Hyperparams from GGUF KV (27B): `key_head=16, value_head=48,
head_dim_k=head_dim_v=128, conv_kernel=4, d_inner=6144`;
`conv_channels = d_inner + 2*16*128 = 10240`.

```
qkv = wqkv @ h            # [10240]
z   = wqkv_gate @ h       # [6144]
beta_raw = ssm_beta @ h   # [48];  beta = sigmoid(beta_raw)
alpha    = ssm_alpha @ h  # [48]; alpha = softplus(alpha + ssm_dt_bias)
gate     = ssm_a * alpha  # ssm_a stored as -exp(A_log); per v-head SCALAR (GDA)

# stateful causal conv, 4 taps, per channel, then SiLU:
conv_in  = push qkv into conv_state[c][4]; y_c = w[c]·conv_in + bias? (no bias)
qkv = SiLU(y)                                   # [10240]

q = qkv[0:2048]      -> 16 heads × 128
k = qkv[2048:4096]   -> 16 heads × 128
v = qkv[4096:10240]  -> 48 heads × 128
q = L2norm(q) ; k = L2norm(k)        # eps = rms eps (1e-6 per KV f_norm_rms_eps? USE 1e-5? -> verify vs capture)
q = repeat 16->48 (×3 interleave: head hv uses hk = hv / 3)   # GGML_ASSERT num_v%num_k
o = per-head gated-delta-recurrence (below)                   # 48 × [128]

attn_out = RMSNorm(o_perhead) * ssm_norm_w  *  SiLU(z)        # norm_gated
out      = ssm_out @ flatten(attn_out)      # [6144 -> 2560]
```

### Per-head recurrence (delta-net-base.cpp:289-371, AR path, single token)
State per v-head: `S ∈ R^{128×128}` (row-major; orientation TBD by differential —
C treats S as [S_v dim0, S_v dim1], `sk[j]=Σ_i S[i,j]·k[i]`, `S[i,j] += k[i]·d[j]`,
`o[j] = Σ_i S[i,j]·q[i]` ⇒ **o = Sᵀq** with that layout):

```
q *= 1/sqrt(128)
S *= exp(gate)                      # gate scalar per head: column scale == all
sk = einsum('ij,i->j', S, k)        # S^T k
d  = (v - sk) * beta                # [128]
S += outer(k, d)
o  = einsum('ij,i->j', S, q)        # S^T q
```
State storage per layer: `48*128*128` f32 = 3.1 MB; 48 delta layers ≈ 150 MB/seq
— host-resident on stage node (Art. 5: state never crosses wire nor PCIe).

## Full-attention layer (every 4th)
```
Qfull = wq @ h          # [24 heads × (256 q + 256 gate) interleaved]
Q  = interleave_view(Qfull, stride 512, take[0:256])   # 24×256
gate = same view, take[256:512]
Q = RMSNorm(Q) * q_norm ; K = RMSNorm(K) * k_norm      # per-head 256
K = wk @ h [4×256] ; V = wv @ h [4×256]
rope: n_rot=64 of 256, NEOX split-half within those 64 dims;
      MROPE sections [11,11,10] in TEXT mode = plain sequential inv_freq[0..31]
      base=1e7 ✓ verify against capture
attn = causal GQA(24/4, head 256, scale 1/16)
attn *= sigmoid(gate) elementwise            # "gated attention"
out  = wo @
```
KV cache: 4 heads×256×2 = 2 KB/token/layer → 16 attn layers = 33 KB/token.

## Tensor name map (27B GGUF, prefix `blk.N.`)
attn_norm.weight f32 | attn_post_norm.weight f32 |
**delta**: attn_qkv.weight q4_1?? (verify: 27B parse said 12×?→`ssm` names below)
`attn_qkv.weight [2560,10240]`, `attn_gate.weight [2560,6144]`(z),
`ssm_conv1d.weight f32 [4,10240]`, `ssm_dt.bias f32 [48]`,
`ssm_alpha.weight f32 [2560,48]`, `ssm_beta.weight f32 [2560,48]`,
`ssm_norm.weight f32 [128]`, `ssm_out.weight q4_K [6144,2560]`,
`ssm_a f32 [48]` (no ".weight" suffix — bare name!),
**attn**: `attn_q.weight [2560, 24*512]` (!interleaved), `attn_k/v [2560,1024]`,
`attn_out [1536? no: 24*256=6144 →2560]`, `attn_q_norm/attn_k_norm f32 [256]`;
ffn_{gate,up} [2560,17408], ffn_down [17408,2560].
(Confirm exact names by dumping GGUF tensor list — the parse showed these.)

## Differential test plan (M1 gate)
1. capture9b = decode_capture(single token t at pos p, prefill 2 tokens first)
2. Rust: build 9B model card from GGUF metadata (sharder step), load shard-0 layers
3. per node-name map {q_conv-l, k_conv-l, beta_sigmoid-l, gate-l, state_predelta-l,
   q/k/v_conv_predelta-l, dnet_add_ar_state-l? (fused!), attn_post_norm-l,
   linear_attn_out-l, ffn_out-l, post_ffn-l, l_out-l}
4. assert cos>0.999 per mapped tensor; orientation failures caught at o-step.
NOTE: fused GDN path (`fused_gdn_ar`) may replace node names with
`__fgdn__-l` — capture will show; disable fusion via cparam if needed
(`cparams.fused_gdn_ar=false` exists — expose via env if simpler).

## Sharding notes
- 64 layers / 4 stages ≈ [12,16,16,20] + head on last? — compute-weighted (§14.3).
- delta states live on each stage's host: **stage-local**, zero wire traffic.
- nextn/MTP + vision tensors: filter OUT at shard time (`--skip nextn,vit`).
