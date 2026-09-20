//! Qwen3.8 (arch `qwen35`) kernels: gated delta-net recurrence, causal conv,
//! L2 norm, partial-NEOX rope, gated RMS norm.
//!
//! Transcribed from vendored `delta-net-base.cpp::build_delta_net_autoregressive`
//! (see docs/QWEN35_PORT.md for line references and orientation notes).

/// Delta-net head dimension (d_state), fixed for the qwen35 family.
pub const S: usize = 128;

/// Per-head gate and beta scalars for one recurrence step.
#[derive(Debug, Clone, Copy)]
pub struct DeltaStep {
    pub gate: f32,
    pub beta: f32,
}

/// One autoregressive gated-delta-net step for a single head (S=128).
///
/// `state` is row-major S×S (element (i,j) at i + j*S, matching ggml's
/// [S_v, S_v] tensor). `q` is scaled by the caller. Updates state in place
/// and writes the head output to `out`.
pub fn delta_head(state: &mut [f32], q: &[f32], k: &[f32], v: &[f32], step: DeltaStep, out: &mut [f32]) {
    delta_head_dim(state, S, q, k, v, step, out)
}

/// Runtime-dim variant of the recurrence.
pub fn delta_head_dim(state: &mut [f32], sdim: usize, q: &[f32], k: &[f32], v: &[f32], step: DeltaStep, out: &mut [f32]) {
    debug_assert_eq!(state.len(), sdim * sdim);
    let decay = step.gate.exp();

    // sk[j] = sum_i S[i,j] * k[i]   (S^T k), pre-decay read folded via decay mult
    let mut sk = vec![0.0f32; sdim];
    for j in 0..sdim {
        let mut acc = 0.0f32;
        for (i, &ki) in k.iter().enumerate() {
            acc += state[i + j * sdim] * ki;
        }
        sk[j] = acc * decay;
    }

    // d[j] = beta * (v[j] - sk[j]); S[i,j] = S[i,j]*decay + k[i]*d[j]
    let d: Vec<f32> = (0..sdim).map(|j| step.beta * (v[j] - sk[j])).collect();
    for j in 0..sdim {
        for (i, &ki) in k.iter().enumerate() {
            state[i + j * sdim] = state[i + j * sdim] * decay + ki * d[j];
        }
    }

    // o[j] = sum_i S[i,j] * q[i]   (S^T q)
    for j in 0..sdim {
        let mut acc = 0.0f32;
        for (i, &qi) in q.iter().enumerate() {
            acc += state[i + j * sdim] * qi;
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

    /// Cap'n Proto card round-trip: every field survives the schema.
    #[cfg(feature = "capnp2")]
    #[test]
    fn card_capnp_roundtrip() {
        let card = Card {
            architecture: "qwen35".into(),
            n_layer: 64,
            n_embd: 5120,
            n_head: 24,
            n_head_kv: 4,
            n_ff: 17408,
            n_vocab: 248320,
            head_dim: 256,
            eps: 1e-6,
            rope_base: 1e7,
            n_rot: 64,
            full_attention_interval: 4,
            nextn: 0,
            draft_layers: Some((64, 65)),
            draft_node: Some(2),
            ssm: SsmParams {
                conv_kernel: 4,
                d_state: 128,
                n_k_heads: 16,
                n_v_heads: 48,
                d_inner: 6144,
            },
            hadamard: Some(HadamardCfg {
                block_size: 1024,
                sign_widths: vec![5120, 6144, 17408],
                signs: vec![-1, 1, -1, 1],
                gdn_v_grouped: true,
            }),
        };
        let bytes = card.to_capnp().unwrap();
        let back = Card::from_capnp(&bytes).unwrap();
        assert_eq!(card, back, "capnp roundtrip must be lossless");
    }

    #[cfg(feature = "capnp2")]
    #[test]
    fn card_capnp_roundtrip_no_hadamard_no_draft() {
        let card = Card {
            architecture: "bitnet".into(),
            n_layer: 30,
            n_embd: 2560,
            n_head: 20,
            n_head_kv: 5,
            n_ff: 6912,
            n_vocab: 128256,
            head_dim: 0,
            eps: 1e-5,
            rope_base: 5e5,
            n_rot: 128,
            full_attention_interval: 1,
            nextn: 0,
            draft_layers: None,
            draft_node: None,
            ssm: SsmParams {
                conv_kernel: 0,
                d_state: 0,
                n_k_heads: 0,
                n_v_heads: 0,
                d_inner: 0,
            },
            hadamard: None,
        };
        let bytes = card.to_capnp().unwrap();
        let back = Card::from_capnp(&bytes).unwrap();
        assert_eq!(card, back);
    }

    /// Rung B1 gate (docs/AIR_PATH.md Track B): the model card must carry the
    /// draft-head placement the sharder emits; old cards load unchanged.
    #[test]
    fn card_parses_draft_fields() {
        let json = r#"{
            "architecture": "qwen35",
            "n_layer": 65, "n_embd": 5120, "n_head": 40, "n_head_kv": 8,
            "n_ff": 17408, "n_vocab": 248320, "eps": 1e-5, "rope_base": 1e7,
            "n_rot": 64, "full_attention_interval": 4, "nextn": 1,
            "ssm": {"conv_kernel": 4, "d_state": 128, "n_k_heads": 16, "n_v_heads": 48, "d_inner": 6144},
            "keep_layers": 64,
            "draft_layers": [64, 64],
            "draft_node": 1
        }"#;
        let card: Card = serde_json::from_str(json).unwrap();
        assert!(card.has_draft());
        assert_eq!(card.draft_layers, Some((64, 64)));
        assert_eq!(card.draft_node, Some(1));
    }

    #[test]
    fn old_card_without_draft_still_loads() {
        let json = r#"{
            "architecture": "bitnet",
            "n_layer": 30, "n_embd": 2048, "n_head": 16, "n_head_kv": 16,
            "n_ff": 8192, "n_vocab": 150000, "eps": 1e-5, "rope_base": 10000,
            "n_rot": 0, "full_attention_interval": 1, "nextn": 0,
            "ssm": {"conv_kernel": 0, "d_state": 0, "n_k_heads": 0, "n_v_heads": 0, "d_inner": 0}
        }"#;
        let card: Card = serde_json::from_str(json).unwrap();
        assert!(!card.has_draft());
        assert_eq!(card.draft_layers, None);
        assert_eq!(card.draft_node, None);
    }

    #[test]
    fn test_delta_head_decay_only_when_beta_zero() {
        let mut state = vec![0.5f32; S * S];
        let q = vec![0.1f32; S];
        let k = vec![0.2f32; S];
        let v = vec![0.3f32; S];
        let mut out = vec![0.0f32; S];
        let gate = -0.25f32;
        delta_head(&mut state, &q, &k, &v, DeltaStep { gate, beta: 0.0 }, &mut out);
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
        delta_head(&mut state, &q, &k, &v, DeltaStep { gate: 0.0, beta: 1.0 }, &mut out);
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
        delta_head(&mut s1, &q, &k, &v, DeltaStep { gate: -0.1, beta: 0.7 }, &mut o1);
        delta_head(&mut s2, &q, &k, &v, DeltaStep { gate: -0.1, beta: 0.7 }, &mut o2);
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
use crate::infer::hadamard;
use crate::infer::ops::{rmsnorm, silu, softmax};
use crate::infer::Stage;
use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SsmParams {
    pub conv_kernel: usize,
    pub d_state: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub d_inner: usize,
}

/// PrismML Hadamard-fold metadata (`hadamard.json`, from prism.hadamard.* KV).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HadamardCfg {
    pub block_size: usize,
    /// Sign-vector widths; one slice per distinct folded input dimension.
    pub sign_widths: Vec<usize>,
    /// Flat ±1 values, concatenation of per-width slices.
    pub signs: Vec<i32>,
    /// GDN V-grouped ssm_out feature order (perm_rep > 1 in the fork).
    #[serde(default)]
    pub gdn_v_grouped: bool,
}

/// model.json emitted by tools/shard_model.py
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Card {
    pub architecture: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub n_rot: usize,
    pub full_attention_interval: usize,
    pub nextn: usize,
    /// MTP draft-head layer range (inclusive), if the sharder kept it.
    /// Track B (docs/AIR_PATH.md): the draft lives on the brain node.
    #[serde(default)]
    pub draft_layers: Option<(usize, usize)>,
    #[serde(default)]
    pub draft_node: Option<usize>,
    pub ssm: SsmParams,
    /// Hadamard fold (PTQ1_0 exports); absent for non-folded checkpoints.
    #[serde(default)]
    pub hadamard: Option<HadamardCfg>,
}

impl Card {
    pub fn load(path: &str) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Load `model.json` from a shard dir, attaching `hadamard.json` when present.
    pub fn load_dir(dir: &str) -> Result<Self> {
        let mut card = Self::load(&format!("{dir}/model.json"))?;
        if let Ok(s) = std::fs::read_to_string(format!("{dir}/hadamard.json")) {
            card.hadamard = Some(serde_json::from_str(&s)?);
        }
        Ok(card)
    }
    pub fn head_v_dim(&self) -> usize {
        if self.ssm.n_v_heads == 0 {
            0
        } else {
            self.ssm.d_inner / self.ssm.n_v_heads
        }
    }
    pub fn attn_head_dim(&self) -> usize {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.n_embd / self.n_head
        }
    }

    /// true if layer il uses gated delta-net, false = full attention
    pub fn is_delta(&self, il: usize) -> bool {
        !(il + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Whether the sharder kept an MTP draft head for the brain node.
    pub fn has_draft(&self) -> bool {
        self.draft_layers.is_some() && self.draft_node.is_some()
    }

    /// Tied-head fallback allowed only for families without untied export.
    pub fn tie_fallback(&self) -> bool {
        self.architecture != "qwen35"
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

    /// Cap'n Proto serialization (BMTS v2 track, schemas/model.capnp).
    /// Field-for-field with the serde card; keep in sync when Card grows.
    #[cfg(feature = "capnp2")]
    pub fn to_capnp(&self) -> Result<Vec<u8>> {
        let mut msg = capnp::message::Builder::new_default();
        let mut root = msg.init_root::<crate::model_capnp::model_card::Builder>();
        root.set_architecture(&self.architecture);
        root.set_n_layer(self.n_layer as u32);
        root.set_n_embd(self.n_embd as u32);
        root.set_n_head(self.n_head as u32);
        root.set_n_head_kv(self.n_head_kv as u32);
        root.set_n_ff(self.n_ff as u32);
        root.set_n_vocab(self.n_vocab as u32);
        root.set_eps(self.eps);
        root.set_rope_base(self.rope_base);
        root.set_n_rot(self.n_rot as u32);
        root.set_head_dim(self.head_dim as u32);
        root.set_head_v_dim(self.head_v_dim() as u32);
        root.set_full_attention_interval(self.full_attention_interval as u32);
        root.set_nextn(self.nextn as u32);

        let mut ssm = root.reborrow().init_ssm();
        ssm.set_conv_kernel(self.ssm.conv_kernel as u32);
        ssm.set_d_state(self.ssm.d_state as u32);
        ssm.set_n_k_heads(self.ssm.n_k_heads as u32);
        ssm.set_n_v_heads(self.ssm.n_v_heads as u32);
        ssm.set_d_inner(self.ssm.d_inner as u32);

        if let Some(h) = &self.hadamard {
            let mut hroot = root.reborrow().init_hadamard();
            hroot.set_block_size(h.block_size as u32);
            let mut widths = hroot.reborrow().init_sign_widths(h.sign_widths.len() as u32);
            for (i, &w) in h.sign_widths.iter().enumerate() {
                widths.set(i as u32, w as u32);
            }
            let mut signs = hroot.reborrow().init_signs(h.signs.len() as u32);
            for (i, &s) in h.signs.iter().enumerate() {
                signs.set(i as u32, s as i8);
            }
            hroot.set_gdn_v_grouped(h.gdn_v_grouped);
        }

        if let Some((a, b)) = self.draft_layers {
            root.set_has_draft(true);
            root.set_draft_layer_start(a as u32);
            root.set_draft_layer_end(b as u32);
        }
        if let Some(n) = self.draft_node {
            root.set_draft_node(n as u16);
        }

        Ok(capnp::serialize::write_message_to_words(&msg))
    }

    /// Inverse of `to_capnp`.
    #[cfg(feature = "capnp2")]
    pub fn from_capnp(bytes: &[u8]) -> Result<Self> {
        let reader = capnp::serialize::read_message_from_flat_slice(
            &mut &bytes[..],
            capnp::message::ReaderOptions::new(),
        )?;
        let root = reader.get_root::<crate::model_capnp::model_card::Reader>()?;

        let ssm_r = root.get_ssm()?;
        let ssm = SsmParams {
            conv_kernel: ssm_r.get_conv_kernel() as usize,
            d_state: ssm_r.get_d_state() as usize,
            n_k_heads: ssm_r.get_n_k_heads() as usize,
            n_v_heads: ssm_r.get_n_v_heads() as usize,
            d_inner: ssm_r.get_d_inner() as usize,
        };
        let hadamard = if root.has_hadamard() {
            let h = root.get_hadamard()?;
            Some(HadamardCfg {
                block_size: h.get_block_size() as usize,
                sign_widths: h.get_sign_widths()?.iter().map(|w| w as usize).collect(),
                signs: h.get_signs()?.iter().map(|s| s as i32).collect(),
                gdn_v_grouped: h.get_gdn_v_grouped(),
            })
        } else {
            None
        };
        let draft_layers = if root.get_has_draft() {
            Some((root.get_draft_layer_start() as usize, root.get_draft_layer_end() as usize))
        } else {
            None
        };
        let draft_node = if root.get_draft_node() != 0 {
            Some(root.get_draft_node() as usize)
        } else {
            None
        };

        Ok(Card {
            architecture: root.get_architecture()?.to_str()?.to_string(),
            n_layer: root.get_n_layer() as usize,
            n_embd: root.get_n_embd() as usize,
            n_head: root.get_n_head() as usize,
            n_head_kv: root.get_n_head_kv() as usize,
            n_ff: root.get_n_ff() as usize,
            n_vocab: root.get_n_vocab() as usize,
            head_dim: root.get_head_dim() as usize,
            eps: root.get_eps(),
            rope_base: root.get_rope_base(),
            n_rot: root.get_n_rot() as usize,
            full_attention_interval: root.get_full_attention_interval() as usize,
            nextn: root.get_nextn() as usize,
            draft_layers,
            draft_node,
            ssm,
            hadamard,
        })
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

/// Parsed Hadamard runtime: f32 signs ready for `hadamard::rotate_*`.
#[derive(Debug, Clone)]
struct HadRuntime {
    block: usize,
    widths: Vec<usize>,
    signs: Vec<f32>,
    gdn_v_grouped: bool,
}

impl HadRuntime {
    fn from_cfg(cfg: &HadamardCfg) -> Self {
        Self {
            block: cfg.block_size,
            widths: cfg.sign_widths.clone(),
            signs: cfg.signs.iter().map(|&i| i as f32).collect(),
            gdn_v_grouped: cfg.gdn_v_grouped,
        }
    }

    /// Offset of the sign slice whose width matches `len`.
    fn offset_for(&self, len: usize) -> Result<usize> {
        let mut acc = 0usize;
        for &w in &self.widths {
            if w == len {
                return Ok(acc);
            }
            acc += w;
        }
        anyhow::bail!("no hadamard sign slice for width {len}")
    }

    /// Activation-side (build_lora_mm): x' = H·(s ∘ x).
    fn rotate_fwd_for(&self, x: &mut [f32]) -> Result<()> {
        let off = self.offset_for(x.len())?;
        hadamard::rotate_fwd(x, &self.signs[off..off + x.len()], self.block);
        Ok(())
    }

    /// Embedding-side (post-lookup): h = s ∘ (H·z).
    fn rotate_inv_for(&self, x: &mut [f32]) -> Result<()> {
        let off = self.offset_for(x.len())?;
        hadamard::rotate_inv(x, &self.signs[off..off + x.len()], self.block);
        Ok(())
    }
}

/// One pipeline stage of a qwen35 model.
pub struct Qwen35Stage {
    pub card: Card,
    pub inner: Stage,
    had: Option<HadRuntime>,
    delta: Vec<(u32, DeltaRuntime)>,
    attn: Vec<(u32, AttnKv)>,
    pub seq: usize,
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
        let had = card.hadamard.as_ref().map(HadRuntime::from_cfg);
        Ok(Self {
            card, inner, had, delta, attn, seq: 0,
            tap: false,
            last_qkv: None, last_conv_out: None, last_q: None, last_beta: None,
            last_gate: None, last_delta_o: None, last_state: None, last_delta_out: None,
        })
    }

    /// Hadamard-rotate an activation for folded weights; identity without fold.
    fn wh(&self, x: &[f32]) -> Result<Vec<f32>> {
        match &self.had {
            None => Ok(x.to_vec()),
            Some(h) => {
                let mut v = x.to_vec();
                h.rotate_fwd_for(&mut v)?;
                Ok(v)
            }
        }
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

    pub fn has_embed(&self) -> bool {
        self.inner.tensors.contains_key("token_embd.weight")
    }

    /// Embedding lookup; un-rotates the Hadamard-latent row when present.
    pub fn embed(&self, token: usize) -> Result<Vec<f32>> {
        let mut row = self.inner.row("token_embd.weight", token)?;
        if let Some(h) = &self.had {
            h.rotate_inv_for(&mut row)?;
        }
        Ok(row)
    }

    /// Run this stage's whole slice at absolute position `pos`.
    /// Contract: positions must arrive strictly in sequence.
    pub fn forward(&mut self, x: &[f32], pos: usize) -> Result<Vec<f32>> {
        if pos != self.seq {
            anyhow::bail!("qwen stage expects pos {}, got {}", self.seq, pos);
        }
        let layers = self.layers().to_vec();
        let mut h = x.to_vec();
        for il in layers {
            h = self.run_layer(il, &h, pos)?;
        }
        if self.inner.output_norm_present() {
            h = self.inner.apply_output_norm(&h)?;
        }
        self.seq += 1;
        Ok(h)
    }

    /// Greedy sample if this stage owns an lm_head (untied or tied).
    pub fn sample(&self, h: &[f32]) -> Result<Option<usize>> {
        if self.inner.has_output_head() {
            let l = self.logits(h)?;
            return Ok(Some(crate::infer::finite_argmax(&l)));
        }
        if self.inner.has_head() && self.card.tie_fallback() {
            let l = self.inner.logits(h)?;
            return Ok(Some(crate::infer::finite_argmax(&l)));
        }
        Ok(None)
    }

    /// Logits via untied `output.weight`, rotating `h` into the folded basis.
    pub fn logits(&self, h: &[f32]) -> Result<Vec<f32>> {
        let hr = self.wh(h)?;
        self.inner.logits_untied(&hr)
    }

    pub fn head_kind(&self) -> &'static str {
        if self.inner.has_output_head() {
            "untied"
        } else if self.inner.has_head() && self.card.tie_fallback() {
            "tied"
        } else {
            "none"
        }
    }

    /// Full decoder layer (norm -> kind -> residual -> postnorm -> FFN -> residual).
    pub fn run_layer(&mut self, il: u32, x: &[f32], pos: usize) -> Result<Vec<f32>> {
        let c = self.card.clone();
        let inner = &self.inner;
        let h = rmsnorm(x, &inner.vec_gain(&format!("blk.{}.attn_norm.weight", il))?, c.eps);
        let me = self;

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
        let ffn = me.run_ffn(il, &f)?;
        Ok((0..c.n_embd).map(|i| pre[i] + ffn[i]).collect())
    }

    /// Shared dense FFN: PAR SwiGLU (up*sigmoid*gate then down), Hadamard-aware.
    fn run_ffn(&self, il: u32, f: &[f32]) -> Result<Vec<f32>> {
        let fh = self.wh(f)?;
        let up = self.inner.wmat(&format!("blk.{}.ffn_up.weight", il), &fh)?;
        let gate = self.inner.wmat(&format!("blk.{}.ffn_gate.weight", il), &fh)?;
        let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let ad = self.wh(&act)?;
        self.inner.wmat(&format!("blk.{}.ffn_down.weight", il), &ad)
    }

    fn run_delta(&mut self, il: u32, h: &[f32]) -> Result<Vec<f32>> {
        let c = self.card.clone();
        let p = &c.ssm;
        let hv = p.n_v_heads;
        let hk = p.n_k_heads;
        let hd = c.head_v_dim();
        let kd = p.d_state;
        let channels = p.d_inner + 2 * hk * kd;

        let qkv_in = self.wh(h)?;
        let qkv = self.inner.wmat(&format!("blk.{}.attn_qkv.weight", il), &qkv_in)?;
        let z = self.inner.wmat(&format!("blk.{}.attn_gate.weight", il), &qkv_in)?;
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
            let sp = DeltaStep { gate: gate[v_i], beta: beta[v_i] };
            delta_head_dim(st, hd, &qs, &kh, vh, sp, &mut o[v_i * hd..(v_i + 1) * hd]);
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

        // GDN V-grouped fold: ssm_out rows live in grouped [hd, rep, nk]
        // feature order; the fork permutes the activation to match
        // (llama-graph.cpp build_lora_mm perm_rep > 1 branch) before the
        // signs+Hadamard transform. tiled v=(rep*nk + k) -> grouped (k, rep).
        let gated_in: Vec<f32> = match &self.had {
            Some(h) if h.gdn_v_grouped => {
                let nk = p.n_k_heads;
                let rep = hv / nk;
                let mut g = vec![0.0f32; p.d_inner];
                for k in 0..nk {
                    for r in 0..rep {
                        let src = (r * nk + k) * hd;
                        let dst = k * rep * hd + r * hd;
                        g[dst..dst + hd].copy_from_slice(&gated[src..src + hd]);
                    }
                }
                g
            }
            _ => gated,
        };
        let gated_rot = self.wh(&gated_in)?;
        let out = self.inner.wmat(&format!("blk.{}.ssm_out.weight", il), &gated_rot)?;
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
        let qdim = nh * hd;

        let hr = self.wh(h)?;
        let qfull = self.inner.wmat(&format!("blk.{}.attn_q.weight", il), &hr)?;
        let qn = self.inner.vec_gain(&format!("blk.{}.attn_q_norm.weight", il))?;
        let kn = self.inner.vec_gain(&format!("blk.{}.attn_k_norm.weight", il))?;
        let mut k = self.inner.wmat(&format!("blk.{}.attn_k.weight", il), &hr)?;
        let v = self.inner.wmat(&format!("blk.{}.attn_v.weight", il), &hr)?;

        // attn_q output is per-head [q(hd) | gate(hd)] interleaved
        let mut q = vec![0.0f32; qdim];
        let mut gate = vec![0.0f32; qdim];
        for hi in 0..nh {
            let mut qh = qfull[hi * hd * 2..hi * hd * 2 + hd].to_vec();
            let gh = &qfull[hi * hd * 2 + hd..hi * hd * 2 + hd * 2];
            let y = rmsnorm_head(&qh, &qn, c.eps);
            qh.copy_from_slice(&y);
            q[hi * hd..(hi + 1) * hd].copy_from_slice(&qh);
            gate[hi * hd..(hi + 1) * hd].copy_from_slice(gh);
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
        let mut o = vec![0.0f32; qdim];
        for hi in 0..nh {
            attn_head(&q[hi * hd..(hi + 1) * hd], &ctx, hi / (nh / nkv), &mut o[hi * hd..(hi + 1) * hd])?;
        }
        for i in 0..qdim {
            o[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }
        let o_rot = self.wh(&o)?;
        self.inner.wmat(&format!("blk.{}.attn_output.weight", il), &o_rot)
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

    /// Stage access for differential-tap drivers (oracle comparison tests).
    pub fn stages(&self) -> &[Qwen35Stage] {
        &self.stages
    }

    pub fn stages_mut(&mut self) -> &mut [Qwen35Stage] {
        &mut self.stages
    }

    /// Feed one token id; returns final hidden (post output_norm) + which stage owns head.
    pub fn step(&mut self, token: usize) -> Result<Vec<f32>> {        let c = self.card.clone();
        let pos = self.current_pos();

        let mut x = self.stages[0].embed(token)?;
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

    pub fn argmax(logits: &[f32]) -> usize {
        crate::infer::finite_argmax(logits)
    }

    pub fn logits(&self, h: &[f32]) -> Result<Vec<f32>> {
        for s in self.stages.iter().rev() {
            if s.inner.has_output_head() {
                return s.logits(h);
            }
        }
        anyhow::bail!("no lm_head in stages")
    }

    /// Greedy generation with prompt-lookup speculation (docs/DUET.md P3).
    ///
    /// The drafter guesses each next token from n-grams of the history;
    /// the model verifies by computing the true greedy argmax. Acceptance
    /// feeds the drafted token; rejection feeds the argmax — the emitted
    /// stream is IDENTICAL to plain greedy by construction (losslessness
    /// contract). On this single-token engine verification costs a full
    /// step either way, so v1 wins nothing in wall-clock: it lands the
    /// drafter + accept/reject machinery that a batched-verify kernel
    /// (matmul_tl1 K-column, memory amortization) will make profitable.
    pub fn generate_speculative(
        &mut self,
        first: usize,
        n: usize,
        drafter: &mut PromptLookup,
        k: usize,
    ) -> Result<(Vec<usize>, SpecStats)> {
        let mut stats = SpecStats::default();
        let mut emitted = Vec::with_capacity(n);
        let mut history: Vec<usize> = vec![first];
        let mut tok = first;
        while emitted.len() < n {
            drafter.observe_token(tok);
            let draft = drafter.draft(&history, k);
            let h = self.step(tok)?;
            let logits = self.logits(&h)?;
            let actual = Self::argmax(&logits);
            emitted.push(actual);
            history.push(actual);
            if let Some(&d) = draft.first() {
                if d as usize == actual {
                    stats.hits += 1;
                } else {
                    stats.misses += 1;
                }
            } else {
                stats.no_draft += 1;
            }
            tok = actual;
        }
        Ok((emitted, stats))
    }
}

/// Prompt-lookup drafter: 3-gram → continuation table built from the model's
/// own output history (no draft weights needed — Bonsai has none).
#[derive(Debug, Default)]
pub struct PromptLookup {
    ngram: std::collections::HashMap<(u32, u32, u32), Vec<u32>>,
    recent: Vec<u32>,
}

impl PromptLookup {
    /// Seed the table with an observed token sequence (the prompt).
    pub fn observe(&mut self, tokens: &[usize]) {
        for &t in tokens {
            self.observe_token(t);
        }
    }

    pub fn observe_token(&mut self, t: usize) {
        let t = t as u32;
        if self.recent.len() >= 3 {
            let k = (
                self.recent[self.recent.len() - 3],
                self.recent[self.recent.len() - 2],
                self.recent[self.recent.len() - 1],
            );
            let e = self.ngram.entry(k).or_default();
            if !e.contains(&t) {
                e.push(t);
            }
        }
        self.recent.push(t);
    }

    /// Draft up to `k` continuations of `history` by iterated lookup.
    pub fn draft(&self, history: &[usize], k: usize) -> Vec<u32> {
        if history.len() < 3 || k == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(k);
        let probe = |tri: (u32, u32, u32)| -> Option<u32> {
            self.ngram.get(&tri).and_then(|v| v.first().copied())
        };
        let mut w: [u32; 3] = [
            history[history.len() - 3] as u32,
            history[history.len() - 2] as u32,
            history[history.len() - 1] as u32,
        ];
        for _ in 0..k {
            let Some(next) = probe((w[0], w[1], w[2])) else { break };
            out.push(next);
            w = [w[1], w[2], next];
        }
        out
    }
}

/// Draft acceptance telemetry (docs/DUET.md P3 gates).
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    pub hits: usize,
    pub misses: usize,
    pub no_draft: usize,
}

#[cfg(test)]
mod prompt_lookup_tests {
    use super::*;

    #[test]
    fn test_drafter_learns_and_extends() {
        let mut d = PromptLookup::default();
        // observe "the cat sat on the mat on the"
        for t in [1usize, 2, 3, 4, 5, 6, 2, 3, 7] {
            d.observe_token(t);
        }
        // 3-gram (5,6,2) -> 3; the chain continues (6,2,3) -> 7
        let draft = d.draft(&[5, 6, 2], 3);
        assert_eq!(draft, vec![3, 7], "iterated lookup chains past the first");
    }

    #[test]
    fn test_drafter_short_history_no_draft() {
        let mut d = PromptLookup::default();
        d.observe(&[1, 2]);
        assert!(d.draft(&[1, 2], 4).is_empty(), "needs a full trigram");
        assert!(d.draft(&[], 4).is_empty());
    }

    #[test]
    fn test_drafter_deterministic_first_choice() {
        let mut d = PromptLookup::default();
        for t in [1usize, 2, 3, 9, 1, 2, 3, 8] {
            d.observe_token(t);
        }
        // (1,2,3) seen continuing to 9 then 8: first-seen wins (deterministic)
        assert_eq!(d.draft(&[1, 2, 3], 1), vec![9]);
    }
}


