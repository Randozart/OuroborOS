use bitnet_rs::BitNetModel;

#[test]
fn test_tokenizer_debug() {
    let model_path = "/home/randozart/Desktop/Projects/bitnet-2b-tq1_0.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("model missing, skipping");
        return;
    }

    let model = BitNetModel::load(model_path, 2048, 4).expect("Failed to load model");

    let t1 = model.tokenize("Hello", true);
    eprintln!("tokenize('Hello', true) -> {} tokens: {:?}", t1.len(), t1);
    assert!(!t1.is_empty(), "tokenize failed");

    let t2 = model.tokenize("The capital of France is", true);
    eprintln!("tokenize('The capital of France is') -> {} tokens", t2.len());

    let piece = model.token_to_piece(t1[t1.len() - 1]);
    eprintln!("token_to_piece({}) = {:?}", t1[t1.len() - 1], piece);
}
