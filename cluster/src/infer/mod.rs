//! Pure-Rust BitNet forward pass over BMTS shards.
//!
//! Pipeline-parallel inference without llama.cpp at runtime: each stage owns
//! consecutive transformer layers loaded from `.bmts` files; hidden-state
//! activations (ACTS frames) flow stage to stage.
//!
//! Architecture (matches fork `src/models/bitnet.cpp`, LLM_TYPE_2B):
//! - RMSNorm -> GQA attention (RoPE NEOX, sub-norm before Wo) -> residual
//! - RMSNorm -> SwiGLU PAR (sub-norm before Wdown) -> residual
//! - final: output_norm -> tied lm_head (token_embd^T)

mod dequant;
mod ops;

pub use dequant::{
    dequant_f16, dequant_q4_k, dequant_q8_0, dequant_tq1_0, f16_to_f32, QuantKind,
};

use anyhow::{bail, Result};
use std::collections::HashMap;

use crate::bmts::BmtsShard;
use ops::{f16_row, matvec_f16, rmsnorm, rope_neox, silu, softmax};

/// Transformer hyper-parameters for a BitNet model.
#[derive(Debug, Clone, Copy)]
pub struct ArchConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub n_rot: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub n_vocab: usize,
}

impl ArchConfig {
    /// BitNet-b1.58-2B-4T (30 layers, GQA 20/5, tied embeddings).
    pub fn bitnet_2b() -> Self {
        Self {
            n_embd: 2560,
            n_head: 20,
            n_head_kv: 5,
            n_ff: 6912,
            n_rot: 128,
            eps: 1e-5,
            rope_base: 500000.0,
            n_vocab: 128256,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn kv_dim(&self) -> usize {
        self.head_dim() * self.n_head_kv
    }
}

const DTYPE_F16: u32 = 1;

/// KV cache for ONE layer (rows of n_embd_gqa, one row per token).
#[derive(Debug, Clone, Default)]
pub struct LayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub seq: usize,
}

/// A quantized weight: raw payload + shape.
#[derive(Clone)]
struct Weight {
    payload: Vec<u8>,
    kind: QuantKind,
    in_len: usize,
    out_len: usize,
}

/// A pipeline stage: dequantizable weights + the layers it owns.
#[derive(Clone)]
pub struct Stage {
    cfg: ArchConfig,
    tensors: HashMap<String, Weight>,
    pub layers: Vec<u32>,
}

impl Stage {
    /// Load all tensors of a shard (raw quantized payloads + shapes).
    pub fn from_shard(shard: &BmtsShard, cfg: ArchConfig) -> Result<Self> {
        let mut tensors = HashMap::new();
        for t in &shard.tensors {
            let kind = QuantKind::from_dtype(t.dtype)
                .ok_or_else(|| anyhow::anyhow!("unsupported dtype {} on {}", t.dtype, t.name))?;
            let bytes = shard.read_tensor(&t.name)?;
            let (in_len, out_len) = match t.shape.len() {
                1 => (t.shape[0] as usize, 1usize),
                2 => (t.shape[0] as usize, t.shape[1] as usize),
                n => bail!("tensor {} has rank {}", t.name, n),
            };
            tensors.insert(t.name.clone(), Weight { payload: bytes, kind, in_len, out_len });
        }
        let mut layers: Vec<u32> = tensors
            .keys()
            .filter_map(|k| {
                let rest = k.strip_prefix("blk.")?;
                rest.split('.').next()?.parse().ok()
            })
            .collect();
        layers.sort_unstable();
        layers.dedup();
        Ok(Self { cfg, tensors, layers })
    }

    /// y = W * x for a matrix tensor by name.
    fn wm(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let w = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {}", name))?;
        if w.in_len != x.len() {
            bail!("tensor {}: in_len {} != x len {}", name, w.in_len, x.len());
        }
        Ok(ops::matvec_q(&w.payload, w.kind, w.out_len, w.in_len, x))
    }

    /// 1-D vector tensor (norm gains) as f32.
    fn vec(&self, name: &str) -> Result<Vec<f32>> {
        let w = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {}", name))?;
        match w.kind {
            QuantKind::F32 => Ok(w.payload.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
            QuantKind::F16 => Ok(w.payload.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            other => bail!("tensor {} not a vector dtype {:?}", name, other),
        }
    }

    /// One decoder layer forward. `x` is the hidden state (n_embd).
    pub fn run_layer(&self, layer: u32, x: &[f32], pos: usize, kv: &mut LayerKv) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let p = format!("blk.{}.", layer);

        // --- attention block ---
        let a = rmsnorm(x, &self.vec(&format!("{}attn_norm.weight", p))?, c.eps);

        let mut q = self.wm(&format!("{}attn_q.weight", p), &a)?;
        let mut k = self.wm(&format!("{}attn_k.weight", p), &a)?;
        let v = self.wm(&format!("{}attn_v.weight", p), &a)?;

        for h in 0..c.n_head {
            rope_neox(&mut q[h * c.head_dim()..(h + 1) * c.head_dim()], pos, c.n_rot, c.rope_base);
        }
        for h in 0..c.n_head_kv {
            rope_neox(&mut k[h * c.head_dim()..(h + 1) * c.head_dim()], pos, c.n_rot, c.rope_base);
        }

        kv.k.extend_from_slice(&k);
        kv.v.extend_from_slice(&v);
        kv.seq += 1;

        let attn_out = self.attend(&q, kv)?;

        let b = rmsnorm(&attn_out, &self.vec(&format!("{}attn_sub_norm.weight", p))?, c.eps);
        let o = self.wm(&format!("{}attn_output.weight", p), &b)?;
        let h1: Vec<f32> = (0..c.n_embd).map(|i| x[i] + o[i]).collect();

        // --- FFN block ---
        let f = rmsnorm(&h1, &self.vec(&format!("{}ffn_norm.weight", p))?, c.eps);
        let up_y = self.wm(&format!("{}ffn_up.weight", p), &f)?;
        let gate_y = self.wm(&format!("{}ffn_gate.weight", p), &f)?;
        let act: Vec<f32> = gate_y.iter().zip(&up_y).map(|(g, u)| silu(*g) * u).collect();

        let s = rmsnorm(&act, &self.vec(&format!("{}ffn_sub_norm.weight", p))?, c.eps);
        let d = self.wm(&format!("{}ffn_down.weight", p), &s)?;

        Ok((0..c.n_embd).map(|i| h1[i] + d[i]).collect())
    }

    /// Causal GQA attention against the layer KV cache.
    fn attend(&self, q: &[f32], kv: &LayerKv) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let hd = c.head_dim();
        let groups = c.n_head / c.n_head_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let seq = kv.seq;
        let mut out = vec![0.0f32; c.n_embd];

        for h in 0..c.n_head {
            let qh = &q[h * hd..(h + 1) * hd];
            let kvh = h / groups;
            let mut scores = vec![0.0f32; seq];
            for t in 0..seq {
                let kt = &kv.k[t * c.kv_dim() + kvh * hd..t * c.kv_dim() + (kvh + 1) * hd];
                scores[t] = ops::dot(qh, kt) * scale;
            }
            softmax(&mut scores);
            let o = &mut out[h * hd..(h + 1) * hd];
            for t in 0..seq {
                let vt = &kv.v[t * c.kv_dim() + kvh * hd..t * c.kv_dim() + (kvh + 1) * hd];
                for i in 0..hd {
                    o[i] += scores[t] * vt[i];
                }
            }
        }
        Ok(out)
    }
}

/// A full sharded model: ordered stages + embedding/output tensors (stage 0/last own them).
pub struct PipelineModel {
    cfg: ArchConfig,
    pub stages: Vec<Stage>,
    kv: Vec<Vec<LayerKv>>,
    tok_embd: Vec<u8>,
    output_norm: Vec<f32>,
}

impl PipelineModel {
    /// Assemble a model from ordered shard paths (stage i handles earlier layers).
    pub fn load(paths: &[&str], cfg: ArchConfig) -> Result<Self> {
        if paths.is_empty() {
            bail!("no shards given");
        }
        let mut stages = Vec::new();
        let mut tok_embd = Vec::new();
        let mut output_norm = Vec::new();
        let mut seen_layers: Vec<u32> = Vec::new();

        for (i, p) in paths.iter().enumerate() {
            let shard = BmtsShard::open(p)?;
            if shard.node as usize != i + 1 {
                bail!("shard {} claims node {} (expected {})", p, shard.node, i + 1);
            }
            if i == 0 {
                let t = shard
                    .tensors
                    .iter()
                    .find(|t| t.name == "token_embd.weight")
                    .ok_or_else(|| anyhow::anyhow!("stage 0 lacks token_embd"))?;
                if t.dtype != DTYPE_F16 {
                    bail!("token_embd must be f16, got {}", t.dtype);
                }
                tok_embd = shard.read_tensor(&t.name)?;
            }
            if i + 1 == paths.len() {
                let t = shard
                    .tensors
                    .iter()
                    .find(|t| t.name == "output_norm.weight")
                    .ok_or_else(|| anyhow::anyhow!("last stage lacks output_norm"))?;
                let raw = shard.read_tensor(&t.name)?;
                output_norm = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
            }
            let stage = Stage::from_shard(&shard, cfg)?;
            seen_layers.extend(stage.layers.iter().copied());
            stages.push(stage);
        }

        seen_layers.sort_unstable();
        let want: Vec<u32> = (0..seen_layers.len() as u32).collect();
        if seen_layers != want {
            bail!("layer coverage broken: {:?}", seen_layers);
        }

        let kv = stages.iter().map(|st| vec![LayerKv::default(); st.layers.len()]).collect();
        Ok(Self {
            cfg,
            stages,
            kv,
            tok_embd,
            output_norm,
        })
    }

    /// Reset all layer KV caches.
    pub fn reset(&mut self) {
        for stage in &mut self.kv {
            for k in stage {
                k.k.clear();
                k.v.clear();
                k.seq = 0;
            }
        }
    }

    /// Next absolute position from stage 0 KV.
    pub fn current_pos(&self) -> usize {
        self.kv[0][0].seq
    }

    /// Feed one token id through all stages; returns final hidden state (pre-lm-head) for it.
    pub fn step_token(&mut self, token: usize, pos: usize) -> Result<Vec<f32>> {
        let c = self.cfg;
        let mut x = f16_row(&self.tok_embd, token, c.n_embd);
        for si in 0..self.stages.len() {
            let n_layers = self.stages[si].layers.len();
            for li in 0..n_layers {
                let layer = self.stages[si].layers[li];
                x = self.stages[si].run_layer(layer, &x, pos, &mut self.kv[si][li])?;
            }
        }
        Ok(rmsnorm(&x, &self.output_norm, c.eps))
    }

    /// Logits for the token just fed: tied lm_head (token_embd^T * h).
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let c = self.cfg;
        matvec_f16(&self.tok_embd, c.n_vocab, c.n_embd, hidden)
    }

    /// Greedy argmax over logits.
    pub fn argmax(logits: &[f32]) -> usize {
        let mut best = 0usize;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        best
    }

    /// Process a token sequence (prefill), return hidden state of the last token.
    pub fn prefill(&mut self, tokens: &[usize]) -> Result<Vec<f32>> {
        let mut h = Vec::new();
        for &t in tokens {
            let pos = self.current_pos();
            h = self.step_token(t, pos)?;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ops::matvec_q;

    #[test]
    fn test_arch_dims() {
        let c = ArchConfig::bitnet_2b();
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.kv_dim(), 640);
    }

    #[test]
    fn test_stage_loads_real_shard() {
        let path = "../shards/shard_2.bmts";
        if !std::path::Path::new(path).exists() {
            eprintln!("no shard, skipping");
            return;
        }
        let shard = BmtsShard::open(path).unwrap();
        let cfg = ArchConfig::bitnet_2b();
        let stage = Stage::from_shard(&shard, cfg).unwrap();
        assert_eq!(stage.layers, vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    }

    #[test]
    fn test_layer_forward_runs() {
        let path = "../shards/shard_2.bmts";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let shard = BmtsShard::open(path).unwrap();
        let cfg = ArchConfig::bitnet_2b();
        let stage = Stage::from_shard(&shard, cfg).unwrap();
        let x = vec![0.02; cfg.n_embd];
        let mut kv = LayerKv::default();
        let out = stage.run_layer(10, &x, 0, &mut kv).unwrap();
        assert_eq!(out.len(), cfg.n_embd);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(kv.seq, 1);
    }

    #[test]
    fn test_matvec_shapes_on_real_tensors() {
        let path = "../shards/shard_2.bmts";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let shard = BmtsShard::open(path).unwrap();
        let t = shard.tensors.iter().find(|t| t.name == "blk.10.attn_k.weight").unwrap();
        assert_eq!(t.shape, vec![2560, 640]);
        let payload = shard.read_tensor(&t.name).unwrap();
        let x = vec![0.1; 2560];
        let y = matvec_q(&payload, QuantKind::Tq1_0, 640, 2560, &x);
        assert_eq!(y.len(), 640);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
