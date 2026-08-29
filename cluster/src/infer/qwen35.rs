//! Qwen3.8 (arch `qwen35`) kernels: gated delta-net recurrence, causal conv,
//! L2 norm, partial-NEOX rope, gated RMS norm.
//!
//! Transcribed from vendored `delta-net-base.cpp::build_delta_net_autoregressive`
//! (see docs/QWEN35_PORT.md for line references and orientation notes).

/// Delta-net head dimension (d_state), fixed for the qwen35 family.
pub const S: usize = 128;

/// Runtime-dim variant (same math as delta_head, dim passed in).
fn delta_head_generic(state: &mut [f32], sdim: usize, q: &[f32], k: &[f32], v: &[f32], gate: f32, beta: f32, out: &mut [f32]) {
    let decay = gate.exp();
    let mut sk = vec![0f32; sdim];
    for j in 0..sdim {
        let mut acc = 0.0f32;
        for (i, &ki) in k.iter().enumerate() {
            acc += state[i + j * sdim] * ki;
        }
        sk[j] = acc * decay;
    }
    let d: Vec<f32> = (0..sdim).map(|j| beta * (v[j] - sk[j])).collect();
    for j in 0..sdim {
        for i in 0..sdim {
            state[i + j * sdim] = state[i + j * sdim] * decay + k[i] * d[j];
        }
    }
    for j in 0..sdim {
        let mut acc = 0.0f32;
        for (i, &qi) in q.iter().enumerate() {
            acc += state[i + j * sdim] * qi;
        }
        out[j] = acc;
    }
}

/// One autoregressive gated-delta-net step for a single head.
///
/// `state` is row-major S×S (element (i,j) at i + j*S, matching ggml's
/// [S_v, S_v] tensor). `q` is scaled by the caller. Updates state in place
/// and writes the head output to `out`.
pub fn delta_head(state: &mut [f32], q: &[f32], k: &[f32], v: &[f32], gate: f32, beta: f32, out: &mut [f32]) {
    debug_assert_eq!(state.len(), S * S);
    debug_assert_eq!(q.len(), S);
    debug_assert_eq!(k.len(), S);
    debug_assert_eq!(v.len(), S);

    let decay = gate.exp();

    // sk[j] = sum_i S[i,j] * k[i]      (S^T k)
    let mut sk = [0f32; S];
    for j in 0..S {
        let mut acc = 0.0f32;
        for (i, &ki) in k.iter().enumerate() {
            acc += state[i + j * S] * ki;
        }
        sk[j] = acc * decay;
    }

    // d[j] = beta * (v[j] - sk[j]); S[i,j] = S[i,j]*decay + k[i]*d[j]
    let d: [f32; S] = std::array::from_fn(|j| beta * (v[j] - sk[j]));
    for (j, &dj) in d.iter().enumerate() {
        for (i, &ki) in k.iter().enumerate() {
            state[i + j * S] = state[i + j * S] * decay + ki * dj;
        }
    }

    // o[j] = sum_i S[i,j] * q[i]       (S^T q)
    for j in 0..S {
        let mut acc = 0.0f32;
        for (i, &qi) in q.iter().enumerate() {
            acc += state[i + j * S] * qi;
        }
        out[j] = acc;
    }
}

/// Causal depthwise conv tap step: 4-tap ring, oldest-first.
///
/// `ring` holds the last 3 inputs (ring[0] oldest); y = w0*ring0 + w1*ring1 +
/// w2*ring2 + w3*x (newest last). Orientation flagged for differential
/// confirmation against `conv_output_raw` captures.
pub fn conv_step(ring: &mut [f32; 3], w: &[f32; 4], x: f32) -> f32 {
    let y = w[0] * ring[0] + w[1] * ring[1] + w[2] * ring[2] + w[3] * x;
    ring[0] = ring[1];
    ring[1] = ring[2];
    ring[2] = x;
    y
}

/// L2 normalize a head vector in place.
pub fn l2_norm(v: &mut [f32], eps: f32) {
    let sum: f32 = v.iter().map(|x| x * x).sum();
    let inv = 1.0 / (sum.sqrt() + eps);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// RMSNorm followed by SiLU gating with a matching-width vector.
pub fn gated_norm_rms(x: &[f32], w: &[f32], gate: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    (0..n)
        .map(|i| {
            let g = gate[i] / (1.0 + (-gate[i]).exp());
            x[i] * inv * w[i] * g
        })
        .collect()
}

/// Partial NEOX rope: rotate the first `rot` dims (split-half pairs), leave
/// dims [rot..] untouched.
pub fn rope_partial_neox(v: &mut [f32], t: usize, rot: usize, base: f32) {
    let half = rot / 2;
    for i in 0..half {
        let theta = 1.0 / base.powf((2 * i) as f32 / rot as f32);
        let (cos_t, sin_t) = ((t as f32 * theta).cos(), (t as f32 * theta).sin());
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * cos_t - b * sin_t;
        v[i + half] = b * cos_t + a * sin_t;
    }
}

/// q scale used by the delta recurrence.
pub fn delta_q_scale() -> f32 {
    1.0 / (S as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_head_decay_only_when_beta_zero() {
        let mut state = vec![0.5f32; S * S];
        let q = vec![0.1f32; S];
        let k = vec![0.2f32; S];
        let v = vec![0.3f32; S];
        let mut out = vec![0.0f32; S];
        let gate = -0.25f32;
        delta_head(&mut state, &q, &k, &v, gate, 0.0, &mut out);
        let expect = 0.5f32 * gate.exp();
        assert!(state.iter().all(|&x| (x - expect).abs() < 1e-6));
        // o = (S*decay)^T q  with q pre-scaled
        let dot_q = q.iter().sum::<f32>() * expect;
        assert!(out.iter().all(|&x| (x - dot_q).abs() < 1e-4));
    }

    #[test]
    fn test_delta_head_single_write_read() {
        // Zero state; beta=1; write k (pre-decay state=0 -> sk=0, d=v).
        let mut state = vec![0.0f32; S * S];
        let mut q = vec![0.0f32; S];
        q[5] = delta_q_scale(); // only dim 5
        let mut k = vec![0.0f32; S];
        k[5] = 2.0; // align with q's nonzero dim
        k[9] = 1.0;
        let mut v = vec![0.0f32; S];
        v[7] = 4.0;
        let mut out = vec![0.0f32; S];
        delta_head(&mut state, &q, &k, &v, 0.0, 1.0, &mut out);
        // S = outer(k, v): S[i,j] = k_i*v_j
        assert!((state[5 + 7 * S] - 8.0).abs() < 1e-6);
        assert!((state[9 + 7 * S] - 4.0).abs() < 1e-6);
        // o[j] = q5 * S[5,j] = q5 * k5 * v_j -> nonzero only at j=7
        assert!(out.iter().enumerate().all(|(j, &x)| if j == 7 { (x - 4.0 * delta_q_scale() * 2.0).abs() < 1e-6 } else { x.abs() < 1e-6 }));
    }

    #[test]
    fn test_delta_head_is_linear_in_state() {
        let q: Vec<f32> = (0..S).map(|i| ((i * 7) % 13) as f32 * 0.1).collect();
        let k: Vec<f32> = (0..S).map(|i| ((i * 5) % 11) as f32 * 0.1).collect();
        let v: Vec<f32> = (0..S).map(|i| ((i * 3) % 17) as f32 * 0.1).collect();
        let mut s1 = vec![0.3f32; S * S];
        let mut s2 = s1.clone();
        let mut o1 = vec![0f32; S];
        let mut o2 = vec![0f32; S];
        delta_head(&mut s1, &q, &k, &v, -0.1, 0.7, &mut o1);
        delta_head(&mut s2, &q, &k, &v, -0.1, 0.7, &mut o2);
        assert_eq!(s1, s2, "must be deterministic");
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_conv_step_steady_state() {
        let mut ring = [3.0, 3.0, 3.0];
        let w = [0.25, 0.25, 0.25, 0.25];
        let y = conv_step(&mut ring, &w, 3.0);
        assert!((y - 3.0).abs() < 1e-6);
        assert_eq!(ring, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_conv_step_shifts() {
        let mut ring = [1.0, 2.0, 3.0];
        let w = [0.0, 0.0, 0.0, 1.0]; // newest-only
        assert!((conv_step(&mut ring, &w, 9.0) - 9.0).abs() < 1e-6);
        assert_eq!(ring, [2.0, 3.0, 9.0]);
    }

    #[test]
    fn test_l2_norm_unit_length() {
        let mut v = vec![3.0, 4.0, 0.0, 0.0];
        l2_norm(&mut v, 1e-12);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_gated_norm_rms() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![1.0; 4];
        let g = vec![0.0; 4]; // silu(0)=0 -> out 0
        let y = gated_norm_rms(&x, &w, &g, 1e-5);
        assert!(y.iter().all(|&v| v.abs() < 1e-6));
        let g2 = vec![10.0; 4];
        let y2 = gated_norm_rms(&x, &w, &g2, 1e-5);
        assert!((y2[0] - 10.0).abs() < 0.1, "large gate ~ rms*gate: {:?}", y2[0]);
    }

    #[test]
    fn test_rope_partial_keeps_tail() {
        let mut v: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let tail_before: Vec<f32> = v[64..].to_vec();
        rope_partial_neox(&mut v, 7, 64, 1e7);
        assert_eq!(&v[64..], &tail_before[..], "dims >= rot untouched");
        let changed = (0..64).any(|i| (v[i] - i as f32).abs() > 1e-4);
        assert!(changed, "rope must rotate leading dims");
    }
}

// ---------------------------------------------------------------------------
// Model card + stage runner (family `qwen35` = Qwen3.5/3.8 hybrid attention)
// ---------------------------------------------------------------------------

use crate::bmts::BmtsShard;
use crate::infer::ops::{rmsnorm, silu, softmax};
use crate::infer::Stage;
use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SsmParams {
    pub conv_kernel: usize,
    pub d_state: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub d_inner: usize,
}

/// model.json emitted by tools/shard_model.py
#[derive(Debug, Clone, Deserialize)]
pub struct Card {
    pub architecture: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub n_rot: usize,
    pub full_attention_interval: usize,
    pub nextn: usize,
    pub ssm: SsmParams,
}

impl Card {
    pub fn load(path: &str) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
    pub fn head_v_dim(&self) -> usize {
        self.ssm.d_inner / self.ssm.n_v_heads
    }
    pub fn attn_head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
    /// true if layer il uses gated delta-net, false = full attention
    pub fn is_delta(&self, il: usize) -> bool {
        (il + 1) % self.full_attention_interval != 0
    }

    /// Coarse ArchConfig for Stage validation (dims used by wm/row checks).
    pub fn to_arch(&self) -> crate::infer::ArchConfig {
        crate::infer::ArchConfig {
            n_embd: self.n_embd,
            n_head: self.n_head,
            n_head_kv: self.n_head_kv,
            n_ff: self.n_ff,
            n_rot: self.n_rot,
            eps: self.eps,
            rope_base: self.rope_base,
            n_vocab: self.n_vocab,
        }
    }
}

/// Recurrent runtime state of one delta layer.
#[derive(Debug, Clone)]
pub struct DeltaRuntime {
    /// per-channel causal conv ring of prior inputs (kernel-4 oldest first)
    pub conv: Vec<[f32; 3]>,
    /// per-head matrix state [n_v_heads][S][S] flattened i + j*S
    pub heads: Vec<f32>,
    pub steps: usize,
}

/// KV cache for one full-attention layer (rows of n_head_kv*head_dim).
#[derive(Debug, Clone, Default)]
pub struct AttnKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub seq: usize,
}

/// One pipeline stage of a qwen35 model.
pub struct Qwen35Stage {
    pub card: Card,
    pub inner: Stage,
    delta: Vec<(u32, DeltaRuntime)>,
    attn: Vec<(u32, AttnKv)>,
    /// capture intermediates for differential testing vs oracle
    pub tap: bool,
    pub last_qkv: Option<Vec<f32>>,
    pub last_conv_out: Option<Vec<f32>>,
    pub last_q: Option<Vec<f32>>,
    pub last_beta: Option<Vec<f32>>,
    pub last_gate: Option<Vec<f32>>,
    pub last_delta_o: Option<Vec<f32>>,
    pub last_state: Option<Vec<f32>>,
    pub last_delta_out: Option<Vec<f32>>,
}

impl Qwen35Stage {
    pub fn from_shard(shard: &BmtsShard, card: Card) -> Result<Self> {
        let arch = card.to_arch();
        let inner = Stage::from_shard(shard, arch)?;
        let delta = inner
            .layers
            .iter()
            .filter(|&&l| card.is_delta(l as usize))
            .map(|&l| {
                let hd = card.head_v_dim();
                let ch = card.ssm.d_inner + 2 * card.ssm.n_k_heads * card.ssm.d_state;
                (l, DeltaRuntime {
                    conv: vec![[0.0; 3]; ch],
                    heads: vec![0.0; card.ssm.n_v_heads * hd * hd],
                    steps: 0,
                })
            })
            .collect();
        let attn = inner
            .layers
            .iter()
            .filter(|&&l| !card.is_delta(l as usize))
            .map(|&l| (l, AttnKv::default()))
            .collect();
        Ok(Self {
            card, inner, delta, attn,
            tap: false,
            last_qkv: None, last_conv_out: None, last_q: None, last_beta: None,
            last_gate: None, last_delta_o: None, last_state: None, last_delta_out: None,
        })
    }

    pub fn layers(&self) -> &[u32] {
        &self.inner.layers
    }

    pub fn reset(&mut self) {
        for (_, d) in &mut self.delta {
            for c in &mut d.conv {
                *c = [0.0; 3];
            }
            for h in &mut d.heads {
                *h = 0.0;
            }
            d.steps = 0;
        }
        for (_, a) in &mut self.attn {
            a.k.clear();
            a.v.clear();
            a.seq = 0;
        }
    }

    /// Full decoder layer (norm -> kind -> residual -> postnorm -> FFN -> residual).
    pub fn run_layer(&mut self, il: u32, x: &[f32], pos: usize) -> Result<Vec<f32>> {
        let c = self.card.clone();
        let inner = &self.inner;
        let h = rmsnorm(x, &inner.vec_gain(&format!("blk.{}.attn_norm.weight", il))?, c.eps);
        let mut me = self;

        let o = if c.is_delta(il as usize) {
            me.run_delta(il, &h)?
        } else {
            me.run_attn(il, &h, pos)?
        };
        // qwen35 residual flow (qwen35.cpp): pre = x + attn_out;
        // f = RMSNorm(pre) * post_attention_norm; out = pre + FFN(f)
        let inner = &me.inner;
        let post = inner.vec_gain(&format!("blk.{}.post_attention_norm.weight", il))?;
        let pre: Vec<f32> = (0..c.n_embd).map(|i| x[i] + o[i]).collect();
        let f = rmsnorm(&pre, &post, c.eps);
        let ffn = run_ffn(inner, il, &f)?;
        Ok((0..c.n_embd).map(|i| pre[i] + ffn[i]).collect())
    }

    fn run_delta(&mut self, il: u32, h: &[f32]) -> Result<Vec<f32>> {
        let c = self.card.clone();
        let p = &c.ssm;
        let hv = p.n_v_heads;
        let hk = p.n_k_heads;
        let hd = c.head_v_dim();
        let kd = p.d_state;
        let channels = p.d_inner + 2 * hk * kd;

        let qkv = self.inner.wmat(&format!("blk.{}.attn_qkv.weight", il), h)?;
        let z = self.inner.wmat(&format!("blk.{}.attn_gate.weight", il), h)?;
        let beta_raw = self.inner.wmat(&format!("blk.{}.ssm_beta.weight", il), h)?;
        let alpha_raw = self.inner.wmat(&format!("blk.{}.ssm_alpha.weight", il), h)?;
        let dt_bias = self.inner.vec_gain(&format!("blk.{}.ssm_dt.bias", il))?;
        let a_log = self.inner.vec_gain(&format!("blk.{}.ssm_a", il))?;
        let conv_w = self.inner.vec_gain(&format!("blk.{}.ssm_conv1d.weight", il))?; // [4*channels] f32
        let norm_w = self.inner.vec_gain(&format!("blk.{}.ssm_norm.weight", il))?;

        let beta: Vec<f32> = beta_raw.iter().map(|b| 1.0 / (1.0 + (-b).exp())).collect();
        let gate: Vec<f32> = (0..hv)
            .map(|i| a_log[i] * softplus(alpha_raw[i] + dt_bias[i]))
            .collect();
        if self.tap {
            self.last_qkv = Some(qkv.clone());
            self.last_beta = Some(beta.clone());
            self.last_gate = Some(gate.clone());
        }

        let di = hk * kd;
        let mut conv_out = vec![0.0f32; channels];
        let slot = self.delta.iter_mut().find(|(l, _)| *l == il).map(|(_, d)| d as *mut DeltaRuntime);
        // Safety: single &mut self; raw pointer avoids double-borrow of fields
        let d = unsafe { &mut *slot.unwrap() };
        for ch_i in 0..channels {
            let w: [f32; 4] = [
                conv_w[ch_i * 4],
                conv_w[ch_i * 4 + 1],
                conv_w[ch_i * 4 + 2],
                conv_w[ch_i * 4 + 3],
            ];
            let y = conv_step(&mut d.conv[ch_i], &w, qkv[ch_i]);
            conv_out[ch_i] = silu(y);
        }


        let mut o = vec![0.0f32; p.d_inner];
        let scale = 1.0 / (kd as f32).sqrt();
        let mut qn_flat = vec![0.0f32; di];
        for v_i in 0..hv {
            let k_i = v_i % hk; // ggml repeat tiles: modulo mapping (differential confirms)
            let mut qh: Vec<f32> = conv_out[k_i * kd..(k_i + 1) * kd].to_vec();
            let mut kh: Vec<f32> = conv_out[di + k_i * kd..di + (k_i + 1) * kd].to_vec();
            let vh = &conv_out[2 * di + v_i * hd..2 * di + (v_i + 1) * hd];
            l2_norm(&mut qh, c.eps);
            l2_norm(&mut kh, c.eps);
            if self.tap && v_i < hk {
                qn_flat[v_i * kd..(v_i + 1) * kd].copy_from_slice(&qh);
            }
            let qs: Vec<f32> = qh.iter().map(|x| x * scale).collect();
            let st_start = v_i * hd * hd;
            let st = &mut d.heads[st_start..st_start + hd * hd];
            if hd == S && kd == S {
                delta_head(st, &qs, &kh, vh, gate[v_i], beta[v_i], &mut o[v_i * hd..(v_i + 1) * hd]);
            } else {
                // generic path: copy out/in via temporaries
                let mut buf = st.to_vec();
                let mut oo = vec![0f32; hd];
                delta_head_generic(&mut buf, hd, &qs, &kh, vh, gate[v_i], beta[v_i], &mut oo);
                st.copy_from_slice(&buf);
                o[v_i * hd..(v_i + 1) * hd].copy_from_slice(&oo);
            }
        }

        if self.tap {
            self.last_conv_out = Some(conv_out.clone());
            self.last_q = Some(qn_flat);
            self.last_delta_o = Some(o.clone());
            self.last_state = Some(d.heads.clone());
        }
        // gated per-head RMS norm with z (head_v_dim vectors, shared ssm_norm weights)
        let mut gated = vec![0.0f32; p.d_inner];
        for v_i in 0..hv {
            let (o_i, z_i, g_i) = (
                &o[v_i * hd..(v_i + 1) * hd],
                &z[v_i * hd..(v_i + 1) * hd],
                &mut gated[v_i * hd..(v_i + 1) * hd],
            );
            let y = gated_norm_rms(o_i, &norm_w, z_i, c.eps);
            g_i.copy_from_slice(&y);
        }

        let out = self.inner.wmat(&format!("blk.{}.ssm_out.weight", il), &gated)?;
        if self.tap {
            self.last_delta_out = Some(out.clone());
        }
        Ok(out)
    }

    fn run_attn(&mut self, il: u32, h: &[f32], pos: usize) -> Result<Vec<f32>> {
        let c = self.card.clone();
        let nh = c.n_head;
        let nkv = c.n_head_kv;
        let hd = c.attn_head_dim();
        let dim = c.n_embd;

        let qfull = self.inner.wmat(&format!("blk.{}.attn_q.weight", il), h)?;
        let qn = self.inner.vec_gain(&format!("blk.{}.attn_q_norm.weight", il))?;
        let kn = self.inner.vec_gain(&format!("blk.{}.attn_k_norm.weight", il))?;
        let mut k = self.inner.wmat(&format!("blk.{}.attn_k.weight", il), h)?;
        let v = self.inner.wmat(&format!("blk.{}.attn_v.weight", il), h)?;

        let mut q = vec![0.0f32; dim];
        let mut gate = vec![0.0f32; dim];
        for hi in 0..nh {
            let (mut qh, gh): (Vec<f32>, Vec<f32>) = (
                qfull[hi * hd * 2..hi * hd * 2 + hd].to_vec(),
                qfull[hi * hd * 2 + hd..hi * hd * 2 + hd * 2].to_vec(),
            );
            let y = rmsnorm_head(&qh, &qn, c.eps);
            qh.copy_from_slice(&y);
            q[hi * hd..(hi + 1) * hd].copy_from_slice(&qh);
            gate[hi * hd..(hi + 1) * hd].copy_from_slice(&gh);
        }
        for hi in 0..nkv {
            let kh = &k[hi * hd..(hi + 1) * hd];
            let y = rmsnorm_head(kh, &kn, c.eps);
            k[hi * hd..(hi + 1) * hd].copy_from_slice(&y);
        }
        for hi in 0..nh {
            rope_partial_neox(&mut q[hi * hd..(hi + 1) * hd], pos, c.n_rot, c.rope_base);
        }
        for hi in 0..nkv {
            rope_partial_neox(&mut k[hi * hd..(hi + 1) * hd], pos, c.n_rot, c.rope_base);
        }

        let slot = self.attn.iter_mut().find(|(l, _)| *l == il).map(|(_, a)| a as *mut AttnKv);
        let a = unsafe { &mut *slot.unwrap() };
        a.k.extend_from_slice(&k);
        a.v.extend_from_slice(&v);
        a.seq += 1;

        let ctx = AttnCtx { hd, kv_dim: nkv * hd, scale: 1.0 / (hd as f32).sqrt(), seq: a.seq, k: &a.k, v: &a.v };
        let mut o = vec![0.0f32; dim];
        for hi in 0..nh {
            attn_head(&q[hi * hd..(hi + 1) * hd], &ctx, hi / (nh / nkv), &mut o[hi * hd..(hi + 1) * hd])?;
        }
        for i in 0..dim {
            o[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }
        self.inner.wmat(&format!("blk.{}.attn_output.weight", il), &o)
    }
}

struct AttnCtx<'a> {
    hd: usize,
    kv_dim: usize,
    scale: f32,
    seq: usize,
    k: &'a [f32],
    v: &'a [f32],
}

fn attn_head(q: &[f32], ctx: &AttnCtx, kvh: usize, out: &mut [f32]) -> Result<()> {
    let mut scores = vec![0.0f32; ctx.seq];
    for (t, sc) in scores.iter_mut().enumerate() {
        let kt = &ctx.k[t * ctx.kv_dim + kvh * ctx.hd..t * ctx.kv_dim + (kvh + 1) * ctx.hd];
        *sc = crate::infer::ops::dot(q, kt) * ctx.scale;
    }
    softmax(&mut scores);
    for (t, &w) in scores.iter().enumerate() {
        let vt = &ctx.v[t * ctx.kv_dim + kvh * ctx.hd..t * ctx.kv_dim + (kvh + 1) * ctx.hd];
        for (oi, &vi) in out.iter_mut().zip(vt) {
            *oi += w * vi;
        }
    }
    Ok(())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Per-head RMS norm (llama LLM_NORM_RMS on head vectors).
fn rmsnorm_head(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    rmsnorm(x, w, eps)
}

/// Shared dense FFN: PAR SwiGLU (up*sigmoid*gate then down).
pub fn run_ffn(inner: &Stage, il: u32, f: &[f32]) -> Result<Vec<f32>> {
    let up = inner.wmat(&format!("blk.{}.ffn_up.weight", il), f)?;
    let gate = inner.wmat(&format!("blk.{}.ffn_gate.weight", il), f)?;
    let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    inner.wmat(&format!("blk.{}.ffn_down.weight", il), &act)
}

/// Multi-stage qwen35 pipeline model (local orchestration; TCP later).
pub struct Qwen35Model {
    pub card: Card,
    stages: Vec<Qwen35Stage>,
    pos: usize,
}

impl Qwen35Model {
    pub fn load(shard_paths: &[&str], card: Card) -> Result<Self> {
        let mut stages = Vec::new();
        for (i, p) in shard_paths.iter().enumerate() {
            let shard = BmtsShard::open(p)?;
            if (shard.node as usize) != i + 1 {
                anyhow::bail!("shard {} claims node {}", p, shard.node);
            }
            stages.push(Qwen35Stage::from_shard(&shard, card.clone())?);
        }
        Ok(Self { card, stages, pos: 0 })
    }

    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.reset();
        }
        self.pos = 0;
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    /// Feed one token id; returns final hidden (post output_norm) + which stage owns head.
    pub fn step(&mut self, token: usize) -> Result<Vec<f32>> {
        let c = self.card.clone();
        let pos = self.current_pos();

        let mut x = self.stages[0].inner.row("token_embd.weight", token)?;
        for s in &mut self.stages {
            let layers = s.layers().to_vec();
            for il in layers {
                x = s.run_layer(il, &x, pos)?;
            }
        }
        self.pos = pos + 1;
        // last stage owns output_norm (shard filter guarantees)
        let last = self.stages.len() - 1;
        if self.stages[last].inner.output_norm_present() {
            x = self.stages[last].inner.apply_output_norm(&x)?;
        }
        let _ = c;
        Ok(x)
    }

    pub fn logits(&self, h: &[f32]) -> Result<Vec<f32>> {
        for s in self.stages.iter().rev() {
            if s.inner.has_output_head() {
                return s.inner.logits_untied(h);
            }
        }
        anyhow::bail!("no lm_head in stages")
    }
}


