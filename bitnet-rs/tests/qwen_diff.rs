//! Qwen3.8 differential: Rust qwen35 stage vs llama.cpp oracle capture.
//! Gate: attn_output-0 / new_state-0 / linear_attn_out-0 / l_out-0 cos > 0.999.

use ouro_cluster::bmts::BmtsShard;
use ouro_cluster::infer::qwen35::{Card, Qwen35Model, Qwen35Stage};

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        let (x, y) = (a[i] as f64, b[i] as f64);
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    (d / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32
}

#[test]
#[ignore] // heavy: 9B oracle + shard
fn test_qwen_layer0_delta_diff() {
    let r = root();
    let model = std::env::var("CAP_MODEL")
        .unwrap_or("/home/randozart/Downloads/Qwen3.8-9B-Q6_K.gguf".into());
    let card = Card::load(r.join("shards9b/model.json").to_str().unwrap()).unwrap();
    let shard_path = r.join("shards9b/shard_1.bmts");
    if !std::path::Path::new(&model).exists() || !shard_path.exists() {
        eprintln!("missing model or shards");
        return;
    }

    // Oracle: single-token prefill (BOS) — fused chunk == AR.
    let m = bitnet_rs::BitNetModel::load(&model, 64, 4).unwrap();
    let ids = m.tokenize("Hello", true);
    let bos = vec![ids[0]];
    let cap = m.decode_capture(&bos).unwrap();
    let get = |name: &str| -> Option<Vec<f32>> {
        cap.iter().find(|n| n.name == name).map(|n| n.data.clone())
    };

    // Rust: same token through stage 1 delta layer 0.
    let shard = BmtsShard::open(shard_path.to_str().unwrap()).unwrap();
    let mut stage = Qwen35Stage::from_shard(&shard, card.clone()).unwrap();
    stage.tap = true;
    let emb = stage.inner.row("token_embd.weight", ids[0] as usize).unwrap();
    let l0 = stage.run_layer(0, &emb, 0).unwrap();

    let checks: [(&str, Option<Vec<f32>>); 9] = [
        ("linear_attn_qkv_mixed-0", stage.last_qkv.clone()),
        ("conv_output_silu-0", stage.last_conv_out.clone()),
        ("q_conv_predelta-0", stage.last_q.clone()),
        ("beta_sigmoid-0", stage.last_beta.clone()),
        ("gate-0", stage.last_gate.clone()),
        ("attn_output-0", stage.last_delta_o.clone()),
        ("new_state-0", stage.last_state.clone()),
        ("linear_attn_out-0", stage.last_delta_out.clone()),
        ("l_out-0", Some(l0.clone())),
    ];
    for (name, mine) in checks {
        let Some(mine) = mine else { continue };
        let Some(refd) = get(name) else {
            eprintln!("{}: not in capture", name);
            continue;
        };
        let c = cos(&refd, &mine);
        eprintln!("{} cos={:.6} (ref n={} mine n={})", name, c, refd.len(), mine.len());
        assert!(c > 0.999, "{} diverged: cos {}", name, c);
    }
}

/// Full 9B: 4 stages, single-token prefill, logits vs oracle.
#[test]
#[ignore] // heavy: full model
fn test_qwen_full_logits_diff() {
    let r = root();
    let model = std::env::var("CAP_MODEL")
        .unwrap_or("/home/randozart/Downloads/Qwen3.8-9B-Q6_K.gguf".into());
    if !std::path::Path::new(&model).exists() {
        return;
    }
    let card = Card::load(r.join("shards9b/model.json").to_str().unwrap()).unwrap();

    // Oracle logits first (then free the mmap).
    let (ref_logits, tok) = {
        let m = bitnet_rs::BitNetModel::load(&model, 64, 4).unwrap();
        let ids = m.tokenize("Hello", true);
        let cap = m.decode_capture(&[ids[0]]).unwrap();
        let logits = cap.iter().find(|n| n.name == "result_output").map(|n| n.data.clone());
        (logits.expect("no result_output node"), ids[0] as usize)
    };

    // Rust: full 4-stage model.
    let paths: Vec<String> = (1..=4).map(|i| r.join(format!("shards9b/shard_{}.bmts", i)).to_str().unwrap().to_string()).collect();
    let refs: Vec<&str> = paths.iter().map(|p| p.as_str()).collect();
    let mut model = ouro_cluster::infer::qwen35::Qwen35Model::load(&refs, card).unwrap();
    let h = model.step(tok).unwrap();
    let mine = model.logits(&h).unwrap();

    let c = cos(&ref_logits, &mine);
    let ref_top = ref_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let mine_top = mine.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let maxd = ref_logits.iter().zip(&mine).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("9B logits: cos={:.6} ref_top={} mine_top={} max_delta={:.4}", c, ref_top, mine_top, maxd);
    assert!(c > 0.999, "logit cos {}", c);
    assert_eq!(ref_top, mine_top, "greedy token must match");
}

/// THE MOUNTAIN: Qwen3.8-27B (13.8 GB, bigger than every single card here)
/// forward-executed by the pure-Rust engine, mmap'd, vs llama.cpp logits.
#[test]
#[ignore] // very heavy: 27B x2 passes
fn test_qwen27b_logits_diff() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    let model = std::env::var("CAP_MODEL_27")
        .unwrap_or("/home/randozart/Downloads/Qwen3.8-27B-Q3_K_M.gguf".into());
    if !std::path::Path::new(&model).exists() || !r.join("shards27/model.json").exists() {
        eprintln!("27B model or shards absent");
        return;
    }
    let card = Card::load("shards27/model.json").unwrap();

    let (ref_logits, tok) = {
        let m = bitnet_rs::BitNetModel::load(&model, 64, 4).unwrap();
        let ids = m.tokenize("The capital of France", true);
        let first = vec![ids[0]];
        let cap = m.decode_capture(&first).unwrap();
        let l = cap.iter().find(|n| n.name == "result_output").map(|n| n.data.clone());
        (l.expect("no result_output"), ids[0] as usize)
    };

    let paths = ["shards27/shard_1.bmts", "shards27/shard_2.bmts", "shards27/shard_3.bmts", "shards27/shard_4.bmts"];
    let t0 = std::time::Instant::now();
    let mut m = Qwen35Model::load(&paths, card).unwrap();
    let h = m.step(tok).unwrap();
    let mine = m.logits(&h).unwrap();
    eprintln!("27B rust step: {:.0}s", t0.elapsed().as_secs_f64());

    let c = cos(&ref_logits, &mine);
    let rt = ref_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let mt = mine.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    eprintln!("27B logits: cos={:.6} ref_top={} rust_top={}", c, rt, mt);
    assert!(c > 0.999, "27B logit cos {}", c);
    assert_eq!(rt, mt, "27B greedy token must match");
}
