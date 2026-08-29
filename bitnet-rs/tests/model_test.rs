use bitnet_rs::BitNetModel;

fn model_path() -> String {
    std::env::var("BITNET_MODEL")
        .unwrap_or_else(|_| "/home/randozart/Desktop/Projects/bitnet-2b-tq1_0.gguf".to_string())
}

#[test]
#[ignore] // Requires actual model file
fn test_load_bitnet_model() {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("Model not found at {}, skipping", path);
        return;
    }

    let model = BitNetModel::load(&path, 2048, 4).expect("Failed to load model");
    assert!(model.n_ctx() > 0);

    let tokens = model.tokenize("Hello", true);
    assert!(!tokens.is_empty());
    eprintln!("Model loaded, n_ctx={}, tokens={}", model.n_ctx(), tokens.len());
}

#[test]
#[ignore] // Requires actual model file
fn test_bitnet_generation() {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        return;
    }

    let model = BitNetModel::load(&path, 2048, 4).expect("Failed to load model");
    let output = model.generate("The capital of France is", 16).expect("generate failed");
    eprintln!("Generated: '{}'", output);
    assert!(!output.is_empty());
}

#[test]
#[ignore] // Requires actual model file
fn test_bitnet_benchmark() {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        return;
    }

    let model = BitNetModel::load(&path, 2048, 4).expect("Failed to load model");
    let (pp, tg) = model.benchmark("The meaning of life is", 32).expect("benchmark failed");
    eprintln!("Benchmark: {:.2} tok/s prompt, {:.2} tok/s generate", pp, tg);
}
