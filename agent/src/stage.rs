//! Pipeline stage slot: one loaded BMTS shard + its per-layer KV caches,
//! executing stage_step commands from the cluster orchestrator.

use anyhow::{bail, Result};
use ouro_cluster::bmts::BmtsShard;
use ouro_cluster::infer::{ArchConfig, LayerKv, Stage};
use ouro_cluster::pipeline::{from_hex, to_hex, Activation};
use std::sync::{Mutex, OnceLock};

/// Loaded stage + its KV state.
pub struct StageSlot {
    pub stage: Stage,
    pub kv: Vec<LayerKv>,
}

fn slot() -> &'static Mutex<Option<StageSlot>> {
    static SLOT: OnceLock<Mutex<Option<StageSlot>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

fn arch_from_env() -> ArchConfig {
    match std::env::var("OURO_ARCH") {
        Ok(json) => serde_json::from_str(&json).unwrap_or_else(|e| {
            eprintln!("bad OURO_ARCH json ({}), using bitnet_2b default", e);
            ArchConfig::bitnet_2b()
        }),
        Err(_) => ArchConfig::bitnet_2b(),
    }
}

/// Handle a stage_* task. Payload formats:
/// - setup: `<shard_path>`
/// - token: `<pos>|<token_id>` (stage with token_embd only)
/// - step:  ACTS hex (pos taken from the frame)
/// - sample: ACTS hex of final hidden -> token id string
pub fn handle(kind: &str, payload: &str) -> Result<String> {
    let mut guard = slot().lock().map_err(|_| anyhow::anyhow!("stage slot poisoned"))?;

    match kind {
        "stage_setup" => {
            let path = payload.trim();
            let shard = BmtsShard::open(path)?;
            let cfg = arch_from_env();
            let stage = Stage::from_shard(&shard, cfg)?;
            let n = stage.layers.len();
            let summary = format!(
                "node={} layers={:?} tensors={}",
                shard.node,
                stage.layers,
                stage.tensor_count()
            );
            *guard = Some(StageSlot {
                stage,
                kv: vec![LayerKv::default(); n],
            });
            Ok(summary)
        }
        "stage_reset" => {
            let s = guard.as_mut().ok_or_else(|| anyhow::anyhow!("no stage loaded"))?;
            for k in &mut s.kv {
                k.k.clear();
                k.v.clear();
                k.seq = 0;
            }
            Ok("reset".to_string())
        }
        "stage_token" => {
            let (pos_s, tok_s) = payload
                .split_once('|')
                .ok_or_else(|| anyhow::anyhow!("stage_token wants pos|id"))?;
            let pos: usize = pos_s.parse()?;
            let token: usize = tok_s.trim().parse()?;
            let s = guard.as_mut().ok_or_else(|| anyhow::anyhow!("no stage loaded"))?;
            check_pos(s, pos)?;
            let mut x = s.stage.embed(token)?;
            x = run_layers(s, &x, pos)?;
            Ok(to_hex(&pack(x, pos)))
        }
        "stage_step" => {
            let act = Activation::decode(&from_hex(payload.trim())?)?;
            let s = guard.as_mut().ok_or_else(|| anyhow::anyhow!("no stage loaded"))?;
            check_pos(s, act.token_pos as usize)?;
            if act.data.len() != s.stage.cfg().n_embd {
                bail!("acts dim {} != n_embd {}", act.data.len(), s.stage.cfg().n_embd);
            }
            let x = run_layers(s, &act.data, act.token_pos as usize)?;
            Ok(to_hex(&pack(x, act.token_pos as usize)))
        }
        "stage_sample" => {
            let act = Activation::decode(&from_hex(payload.trim())?)?;
            let s = guard.as_ref().ok_or_else(|| anyhow::anyhow!("no stage loaded"))?;
            if !s.stage.has_head() {
                bail!("stage lacks token_embd; sample must target stage 0");
            }
            let tok = s.stage.argmax_token(&act.data)?;
            Ok(tok.to_string())
        }
        other => bail!("unknown stage task {}", other),
    }
}

/// Contract: stages execute positions strictly in sequence.
fn check_pos(s: &StageSlot, pos: usize) -> Result<()> {
    if s.kv.is_empty() {
        bail!("stage has no layers");
    }
    let expected = s.kv[0].seq;
    if pos != expected {
        bail!("out-of-order step: got pos {}, stage expects {}", pos, expected);
    }
    Ok(())
}

fn run_layers(s: &mut StageSlot, x: &[f32], pos: usize) -> Result<Vec<f32>> {
    let mut h = x.to_vec();
    for li in 0..s.kv.len() {
        let layer = s.stage.layers[li];
        h = s.stage.run_layer(layer, &h, pos, &mut s.kv[li])?;
    }
    // output_norm lives on the last stage (its shard owns the tensor)
    if s.stage.output_norm_present() {
        h = s.stage.apply_output_norm(&h)?;
    }
    Ok(h)
}

fn pack(data: Vec<f32>, pos: usize) -> Vec<u8> {
    Activation {
        sequence: 0,
        token_pos: pos as u32,
        layer_start: 0,
        layer_end: 0,
        data,
    }
    .encode()
}
