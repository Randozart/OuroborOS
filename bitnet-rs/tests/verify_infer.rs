extern "C" {
    // ggml-internal but exported from libggml-cpu (static-linked here).
    fn dequantize_row_tq1_0(x: *const u8, y: *mut f32, k: i64);
}

use ouro_cluster::bmts::BmtsShard;
use ouro_cluster::infer::{dequant_tq1_0, ArchConfig, PipelineModel};

fn shards_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("shards")
}

fn model_path() -> String {
    std::env::var("BITNET_MODEL")
        .unwrap_or_else(|_| "/home/randozart/Desktop/Projects/bitnet-2b-tq1_0.gguf".to_string())
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Rust TQ1_0 dequant must be bit-identical to ggml's C implementation.
#[test]
fn test_tq1_dequant_parity_with_c() {
    let dir = shards_dir();
    let path = dir.join("shard_1.bmts");
    if !path.exists() {
        eprintln!("run: python3 tools/shard_model.py ... first");
        return;
    }
    let shard = BmtsShard::open(path.to_str().unwrap()).unwrap();
    let t = shard
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .unwrap();
    let payload = shard.read_tensor(&t.name).unwrap();

    // first 10 rows (25600 elements)
    let elems = 2560 * 10;
    let bytes = elems / 256 * 54;

    let rust = dequant_tq1_0(&payload[..bytes]);
    let mut c = vec![0f32; elems];
    unsafe {
        dequantize_row_tq1_0(payload.as_ptr(), c.as_mut_ptr(), elems as i64);
    }
    for i in 0..elems {
        assert_eq!(rust[i].to_bits(), c[i].to_bits(), "mismatch at {}", i);
    }
}

/// Full 3-stage Rust pipeline must agree with llama.cpp reference logits.
#[test]
#[ignore] // needs model + shards
fn test_pipeline_logits_match_reference() {
    let mp = model_path();
    if !std::path::Path::new(&mp).exists() {
        return;
    }
    let dir = shards_dir();
    let sp: Vec<std::path::PathBuf> = (1..=3)
        .map(|i| dir.join(format!("shard_{}.bmts", i)))
        .collect();
    if !sp.iter().all(|p| p.exists()) {
        eprintln!("missing shards");
        return;
    }

    let prompt = "The capital of France";
    let reference = bitnet_rs::BitNetModel::load(&mp, 512, 4).expect("load");
    let tokens = reference.tokenize(prompt, true);
    eprintln!("tokens: {:?}", tokens);
    let ref_logits = reference.logits_for_tokens(&tokens).expect("logits");

    let sps: Vec<&str> = sp.iter().map(|p| p.to_str().unwrap()).collect();
    let cfg = ArchConfig::bitnet_2b();
    let t0 = std::time::Instant::now();
    let mut model = PipelineModel::load(&sps, cfg).expect("pipeline load");
    let ids: Vec<usize> = tokens.iter().map(|t| *t as usize).collect();
    let hidden = model.prefill(&ids).expect("prefill");
    let my_logits = model.logits(&hidden);
    eprintln!("rust pipeline prefill: {:.1}s", t0.elapsed().as_secs_f64());

    let cos = cos_sim(&ref_logits, &my_logits);
    let ref_top = ref_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let my_top = PipelineModel::argmax(&my_logits);
    let max_delta = ref_logits
        .iter()
        .zip(&my_logits)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let ref_abs = ref_logits.iter().fold(0.0f32, |m, a| m.max(a.abs()));
    eprintln!(
        "cos={:.5} top1_ref={} top1_rust={} max_delta={:.4} (ref_abs_max={:.1})",
        cos, ref_top, my_top, max_delta, ref_abs
    );

    assert!(cos > 0.99, "logit cosine too low: {}", cos);
    assert_eq!(ref_top, my_top, "greedy next-token must match");
}
