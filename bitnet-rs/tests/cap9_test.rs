//! Dev tool: capture Qwen-family node names/values from the llama.cpp oracle
//! for differential testing of the Rust qwen35 forward.
//! Usage: CAP_MODEL=... CAP_OUT=... cargo test -p bitnet-rs --test cap9_test -- --ignored

#[test]
#[ignore]
fn capture_qwen_nodes() {
    let model = std::env::var("CAP_MODEL").unwrap_or("/home/randozart/Downloads/Qwen3.8-9B-Q6_K.gguf".into());
    let out = std::env::var("CAP_OUT").unwrap_or("/tmp/cap9_names.txt".into());
    if !std::path::Path::new(&model).exists() {
        eprintln!("no model at {}", model);
        return;
    }
    let m = bitnet_rs::BitNetModel::load(&model, 64, 4).unwrap();
    let ids = m.tokenize("Hello world", true);
    let c = m.decode_capture(&ids).unwrap();
    let mut text = String::new();
    for n in &c {
        text.push_str(&format!("{} {:?}\n", n.name, &n.data[..n.data.len().min(4)]));
    }
    std::fs::write(out, text).unwrap();
    eprintln!("{} nodes", c.len());
}
