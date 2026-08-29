//! Pipeline stage slot: one loaded BMTS shard + its per-layer KV caches,
//! executing stage_step commands from the cluster orchestrator.

use anyhow::{bail, Result};
use ouro_cluster::bmts::BmtsShard;
use ouro_cluster::infer::qwen35::{Card, Qwen35Stage};
use ouro_cluster::infer::{ArchConfig, LayerKv, Stage};
use ouro_cluster::pipeline::{from_hex, to_hex, Activation};
use std::sync::{Mutex, OnceLock};

fn slot() -> &'static Mutex<Option<StageSlot>> {
    static SLOT: OnceLock<Mutex<Option<StageSlot>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Which model family a loaded slot speaks.
pub enum Loaded {
    Bitnet { stage: Stage, kv: Vec<LayerKv> },
    Qwen(Qwen35Stage),
}

/// Loaded stage + its KV state.
pub struct StageSlot {
    pub model: Loaded,
}

/// Architecture summary returned by stage_setup.
fn arch_from_env() -> Option<Card> {
    match std::env::var("OURO_ARCH") {
        Ok(json) => serde_json::from_str::<Card>(&json).ok(),
        Err(_) => None,
    }
}

fn bitnet_arch_from_env() -> ArchConfig {
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
            let (model, summary) = if let Some(card) = arch_from_env() {
                let st = Qwen35Stage::from_shard(&shard, card)?;
                let head = st.head_kind();
                let embed = st.has_embed();
                let summary = format!(
                    "node={} family=qwen35 layers={:?} tensors={} head={} embed={}",
                    shard.node, st.layers(), st.inner.tensor_count(), head, embed
                );
                (Loaded::Qwen(st), summary)
            } else {
                let cfg = bitnet_arch_from_env();
                let stage = Stage::from_shard(&shard, cfg)?;
                let n = stage.layers.len();
                let head = if stage.has_head() { "tied" } else { "none" };
                let summary = format!(
                    "node={} family=bitnet layers={:?} tensors={} head={} embed={}",
                    shard.node, stage.layers, stage.tensor_count(), head, stage.has_head()
                );
                (Loaded::Bitnet { stage, kv: vec![LayerKv::default(); n] }, summary)
            };
            *guard = Some(StageSlot { model });
            Ok(summary)
        }
        "stage_reset" => {
            let s = guard.as_mut().ok_or_else(|| anyhow::anyhow!("no stage loaded"))?;
            match &mut s.model {
                Loaded::Bitnet { kv, .. } => {
                    for k in kv {
                        k.k.clear();
                        k.v.clear();
                        k.seq = 0;
                    }
                }
                Loaded::Qwen(st) => st.reset(),
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
            let x = match &mut s.model {
                Loaded::Qwen(st) => {
                    let e = st.embed(token)?;
                    st.forward(&e, pos)?
                }
                Loaded::Bitnet { stage, kv } => {
                    check_pos_bitnet(&*kv, pos)?;
                    let mut x = stage.embed(token)?;
                    x = run_layers_bitnet(stage, kv, &x, pos)?;
                    x
                }
            };
            Ok(to_hex(&pack(x, pos)))
        }
        "stage_step" => {
            let act = Activation::decode(&from_hex(payload.trim())?)?;
            let s = guard.as_mut().ok_or_else(|| anyhow::anyhow!("no stage loaded"))?;
            let pos = act.token_pos as usize;
            let x = match &mut s.model {
                Loaded::Qwen(st) => st.forward(&act.data, pos)?,
                Loaded::Bitnet { stage, kv } => {
                    check_pos_bitnet(&*kv, pos)?;
                    if act.data.len() != stage.cfg().n_embd {
                        bail!("acts dim {} != n_embd {}", act.data.len(), stage.cfg().n_embd);
                    }
                    run_layers_bitnet(stage, kv, &act.data, pos)?
                }
            };
            Ok(to_hex(&pack(x, pos)))
        }
        "stage_sample" => {
            let act = Activation::decode(&from_hex(payload.trim())?)?;
            let s = guard.as_ref().ok_or_else(|| anyhow::anyhow!("no stage loaded"))?;
            let tok = match &s.model {
                Loaded::Qwen(st) => st.sample(&act.data)?,
                Loaded::Bitnet { stage, .. } => {
                    if !stage.has_head() {
                        bail!("stage lacks token_embd; sample must target a head stage");
                    }
                    Some(stage.argmax_token(&act.data)?)
                }
            };
            match tok {
                Some(t) => Ok(t.to_string()),
                None => bail!("this stage has no lm_head; route sample to a head=true stage"),
            }
        }
        other => bail!("unknown stage task {}", other),
    }
}

/// Contract: stages execute positions strictly in sequence.
fn check_pos_bitnet(kv: &[LayerKv], pos: usize) -> Result<()> {
    if kv.is_empty() {
        bail!("stage has no layers");
    }
    let expected = kv[0].seq;
    if pos != expected {
        bail!("out-of-order step: got pos {}, stage expects {}", pos, expected);
    }
    Ok(())
}

fn run_layers_bitnet(stage: &Stage, kv: &mut [LayerKv], x: &[f32], pos: usize) -> Result<Vec<f32>> {
    let mut h = x.to_vec();
    for li in 0..kv.len() {
        let layer = stage.layers[li];
        h = stage.run_layer(layer, &h, pos, &mut kv[li])?;
    }
    // output_norm lives on the last stage (its shard owns the tensor)
    if stage.output_norm_present() {
        h = stage.apply_output_norm(&h)?;
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
