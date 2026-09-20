//! Bonsai-2-27B (PTQ1_0 + Hadamard) differential vs the PrismML llama.cpp
//! fork oracle. CONTRACTS.md L3 gate: logits cos >= 0.999 + greedy top-1.
//!
//! Oracle produced by tools/ouro-capture in the prism fork clone:
//!   ouro-capture -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -p "Hello" -ngl 0 \
//!       -o /tmp/opencode/bonsai_oracle_logits.f32 -d "l_out-"
//!
//! Run: cargo test -p ouro-cluster --test bonsai_diff -- --ignored --nocapture

use ouro_cluster::infer::qwen35::{Card, Qwen35Model};

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

fn read_f32s(path: &str) -> std::io::Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Parse the ouro-capture .cap record stream into name -> data.
fn read_cap(path: &str) -> std::io::Result<Vec<(String, Vec<f32>)>> {
    let b = std::fs::read(path)?;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let nl = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        let name = String::from_utf8_lossy(&b[i..i + nl]).to_string();
        i += nl;
        let n = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8;
        let data: Vec<f32> = b[i..i + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        i += n * 4;
        out.push((name, data));
    }
    Ok(out)
}

/// Isolation probe: raw embedding vs oracle `model.input_embed`, then each
/// Hadamard variant through layer-0 attn_norm vs oracle `attn_norm-0`.
#[test]
#[ignore]
fn bonsai27_probe_embed() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();

    let cap_path = std::env::var("ORACLE_CAP")
        .unwrap_or_else(|_| "/tmp/opencode/bonsai_oracle_logits.f32.cap".into());
    let token: usize = std::env::var("ORACLE_TOKEN").ok().and_then(|s| s.parse().ok()).unwrap_or(9419);
    if !std::path::Path::new(&cap_path).exists() {
        eprintln!("no oracle cap at {cap_path}");
        return;
    }
    let cap = read_cap(&cap_path).unwrap();
    let get = |name: &str| -> Option<Vec<f32>> {
        cap.iter().find(|(n, _)| n == name).map(|(_, d)| d.clone())
    };
    let ref_embed = get("model.input_embed").expect("no model.input_embed in cap");
    let ref_norm = get("attn_norm-0").expect("no attn_norm-0 in cap");

    let model = Qwen35Model::load(
        &["shards_bonsai27_n1/shard_1.bmts"],
        Card::load_dir("shards_bonsai27_n1").unwrap(),
    )
    .unwrap();
    let stage = &model.stages()[0];
    let raw = stage.inner.row("token_embd.weight", token).unwrap();

    let c0 = cos(&ref_embed, &raw);
    eprintln!("raw embed:    cos={c0:.6} (n={} vs {})", ref_embed.len(), raw.len());

    // attn_norm weights + eps
    let w = stage.inner.vec_gain("blk.0.attn_norm.weight").unwrap();
    let eps = 1e-5f32;
    let rms = |x: &[f32]| -> Vec<f32> {
        let n = x.len();
        let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        (0..n).map(|i| x[i] * inv * w[i]).collect()
    };

    let none: Vec<f32> = raw.clone();
    let mut inv = raw.clone();
    let mut fwd = raw.clone();
    // replicate HadRuntime slice selection: width 5120 = first slice
    let card_had = Card::load_dir("shards_bonsai27_n1").unwrap().hadamard.unwrap();
    let block = card_had.block_size;
    let signs: Vec<f32> = card_had.signs[..5120].iter().map(|&i| i as f32).collect();
    ouro_cluster::infer::hadamard::rotate_inv(&mut inv, &signs, block);
    ouro_cluster::infer::hadamard::rotate_fwd(&mut fwd, &signs, block);
    for (name, v) in [("none", &none), ("inv", &inv), ("fwd", &fwd)] {
        let y = rms(v);
        eprintln!("attn_norm-0 via {name}: cos={:.6}", cos(&ref_norm, &y));
    }
}

/// Greedy stream vs oracle (near-tie rule: conditional stream equality).
/// Oracle: `ouro-capture -p "Hello" -n 4 -o /tmp/opencode/bonsai_greedy_logits.f32`
/// produced tokens 11 353 2688 264 ("Hello, I'm a").
#[test]
#[ignore]
fn bonsai27_greedy_stream_diff() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    let ref_path = std::env::var("ORACLE_GREEDY_LOGITS")
        .unwrap_or_else(|_| "/tmp/opencode/bonsai_greedy_logits.f32".into());
    if !std::path::Path::new(&ref_path).exists() {
        eprintln!("no oracle greedy logits at {ref_path}");
        return;
    }
    let ref_logits = read_f32s(&ref_path).unwrap();
    let expect: Vec<usize> = std::env::var("ORACLE_GREEDY_TOKENS")
        .ok()
        .map(|s| s.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![11, 353, 2688, 264]);

    let mut model = Qwen35Model::load(
        &["shards_bonsai27_n1/shard_1.bmts"],
        Card::load_dir("shards_bonsai27_n1").unwrap(),
    )
    .unwrap();

    let mut tok = 9419usize; // "Hello"
    for (step, &want) in expect.iter().enumerate() {
        let h = model.step(tok).unwrap();
        let logits = model.logits(&h).unwrap();
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        eprintln!("step {step}: fed={tok} top={top} (oracle {want})");
        assert_eq!(top, want, "stream diverged at step {step}");
        if step + 1 == expect.len() {
            let c = cos(&ref_logits, &logits);
            eprintln!("final-step logits cos={c:.6}");
            assert!(c > 0.999, "final logits cos {c}");
        }
        tok = top;
    }
}

/// L2: 2-node and 4-node pipeline shards reproduce the n1 reference token
/// (and hence the oracle). Feeds "Hello" through each split in-process.
#[test]
#[ignore]
fn bonsai27_pipeline_n2_n4_token_exact() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    let ref_path = std::env::var("ORACLE_LOGITS")
        .unwrap_or_else(|_| "/tmp/opencode/bonsai_oracle_logits.f32".into());
    let ref_logits: Option<Vec<f32>> = if std::path::Path::new(&ref_path).exists() {
        read_f32s(&ref_path).ok()
    } else {
        None
    };

    let cases: [(&str, usize); 2] = [("shards_bonsai27_n2", 2), ("shards_bonsai27_n4", 4)];
    for (dir, n) in cases {
        let paths: Vec<String> = (1..=n)
            .map(|i| format!("{dir}/shard_{i}.bmts"))
            .collect();
        let refs: Vec<&str> = paths.iter().map(|p| p.as_str()).collect();
        let mut model = Qwen35Model::load(&refs, Card::load_dir(dir).unwrap()).unwrap();
        let h = model.step(9419).unwrap();
        let logits = model.logits(&h).unwrap();
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        eprintln!("{dir}: top={top}");
        assert_eq!(top, 11, "{dir} must match oracle token 11");
        if let Some(ref l) = ref_logits {
            let c = cos(l, &logits);
            eprintln!("{dir}: cos={c:.6}");
            assert!(c > 0.999, "{dir} logit cos {c}");
        }
    }
}

/// P3 gate (docs/DUET.md): speculative generation must be token-identical
/// to plain greedy — speculation changes speed, never output.
#[test]
#[ignore]
fn bonsai27_speculative_lossless() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    if !std::path::Path::new("shards_bonsai27_n1/shard_1.bmts").exists() {
        eprintln!("no shards");
        return;
    }
    let n = std::env::var("SPEC_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize);

    let plain = || -> anyhow::Result<Vec<usize>> {
        let mut model = Qwen35Model::load(
            &["shards_bonsai27_n1/shard_1.bmts"],
            Card::load_dir("shards_bonsai27_n1").unwrap(),
        )?;
        let mut tok = 9419usize;
        let mut out = Vec::new();
        for _ in 0..n {
            let h = model.step(tok)?;
            let l = model.logits(&h)?;
            tok = l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            out.push(tok);
        }
        Ok(out)
    };

    let spec = || -> anyhow::Result<(Vec<usize>, ouro_cluster::infer::qwen35::SpecStats)> {
        let mut model = Qwen35Model::load(
            &["shards_bonsai27_n1/shard_1.bmts"],
            Card::load_dir("shards_bonsai27_n1").unwrap(),
        )?;
        let mut drafter = ouro_cluster::infer::qwen35::PromptLookup::default();
        drafter.observe(&[9419]);
        model.generate_speculative(9419, n, &mut drafter, 4)
    };

    let g = plain().unwrap();
    let (s, stats) = spec().unwrap();
    eprintln!("greedy: {g:?}");
    eprintln!("spec:   {s:?} (hits {} misses {} no_draft {})", stats.hits, stats.misses, stats.no_draft);
    assert_eq!(g, s, "speculative decode must be lossless");
}

/// Stream geometry (docs/DUET.md §7 transport-codec gate): measures the
/// claims behind "transmit the delta, not the state" —
/// (a) adjacent-layer cosine: does the residual stream barely move?
/// (b) delta entropy vs state entropy: is the per-layer delta cheaper to
///     entropy-code than the state itself?
/// Grounds the §2 discussion with numbers instead of vibes. NOT a pass/fail
/// contract: it prints measurements.
#[test]
#[ignore]
fn bonsai27_stream_geometry() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    if !std::path::Path::new("shards_bonsai27_n1/shard_1.bmts").exists() {
        eprintln!("no shards");
        return;
    }

    fn entropy_bits_per_value(v: &[f32]) -> f64 {
        // 256-bin histogram over the observed range; entropy of the bins.
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &x in v {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if hi <= lo {
            return 0.0;
        }
        let mut bins = [0u64; 256];
        for &x in v {
            let b = (((x - lo) / (hi - lo)) * 255.0) as usize;
            bins[b.min(255)] += 1;
        }
        let n = v.len() as f64;
        bins.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum()
    }

    let mut model = Qwen35Model::load(
        &["shards_bonsai27_n1/shard_1.bmts"],
        Card::load_dir("shards_bonsai27_n1").unwrap(),
    )
    .unwrap();
    let pos = model.current_pos();
    let mut x = model.stages_mut()[0].embed(9419).unwrap();
    let mut layer_outs: Vec<(u32, Vec<f32>)> = Vec::new();
    for s in model.stages_mut() {
        for il in s.layers().to_vec() {
            x = s.run_layer(il, &x, pos).unwrap();
            layer_outs.push((il, x.clone()));
        }
    }

    eprintln!("layer | adj_cos | state_bits | delta_bits | |dx|/|x|");
    let (mut cos_min, mut cos_sum, mut n) = (1.0f32, 0.0f64, 0usize);
    for w in layer_outs.windows(2) {
        let (_, x0) = &w[0];
        let (l1, x1) = &w[1];
        let d: Vec<f32> = x1.iter().zip(x0).map(|(a, b)| a - b).collect();
        let c = cos(x0, x1);
        let sb = entropy_bits_per_value(x1);
        let db = entropy_bits_per_value(&d);
        let rel = (d.iter().map(|v| v * v).sum::<f32>()).sqrt()
            / (x1.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-9);
        eprintln!("{l1:5} | {c:.6} | {sb:.3} | {db:.3} | {rel:.4}");
        cos_min = cos_min.min(c);
        cos_sum += c as f64;
        n += 1;
    }
    eprintln!(
        "adjacent-layer cos: min={cos_min:.6} mean={:.6} over {n} boundaries",
        cos_sum / n as f64
    );
}

/// Tokenizer gates (docs/BONSAI_TOKENIZER.md): oracle-anchored ids from
/// the verified capture runs, plus text→tokens→generation→text e2e.
#[test]
#[ignore]
fn bonsai27_tokenizer_gates() -> Result<(), Box<dyn std::error::Error>> {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    if !std::path::Path::new("shards_bonsai27_n1/tokenizer.json").exists() {
        eprintln!("no tokenizer.json (run tools/dump_tokenizer.py)");
        return Ok(());
    }
    let mut model = Qwen35Model::load(
        &["shards_bonsai27_n1/shard_1.bmts"],
        Card::load_dir("shards_bonsai27_n1").unwrap(),
    )
    .unwrap();

    // GATE 1: encode matches the fork's token id for "Hello"
    let ids = model.tokenize("Hello").unwrap();
    eprintln!("tokenize(\"Hello\") = {ids:?}");
    assert_eq!(ids, vec![9419u32], "oracle anchor: Hello == 9419");

    // GATE 2: decode of the verified oracle greedy stream
    let stream = [11u32, 353, 2688, 264, 5286, 303, 279, 3694];
    let text = model.detokenize(&stream).unwrap();
    eprintln!("detokenize(stream) = {text:?}");
    assert_eq!(text, ", I'm a student in the University");

    // GATE 3: streaming decoder == full decode on the same ids
    let mut sd = ouro_cluster::infer::tokenizer::StreamingDecoder::new(
        ouro_cluster::infer::tokenizer::Tokenizer::load_dir("shards_bonsai27_n1").unwrap(),
    );
    let mut streamed = String::new();
    for &id in &stream {
        streamed.push_str(&sd.push(id));
    }
    streamed.push_str(&sd.finish());
    assert_eq!(streamed, text, "streaming must equal full decode");

    // GATE 4: roundtrip on awkward text
    for sample in [
        "Hello",
        "I'm sure it's 42% done, isn't we'll've",
        "naïve café 中文 🦀 tab\there\nnewline  trailing  ",
    ] {
        let rt = model.tokenize(sample).and_then(|ids| model.detokenize(&ids))?;
        assert_eq!(rt, sample, "roundtrip failed for {sample:?}");
    }

    // GATE 5: end-to-end text→tokens→greedy gen→text, eos-aware
    let prompt = model.tokenize("Hello")?;
    let mut tok = prompt[0] as usize;
    let mut gen_ids: Vec<u32> = Vec::new();
    let eos = model.tokenizer().unwrap().eos().unwrap() as usize;
    for _ in 0..6 {
        let h = model.step(tok)?;
        let l = model.logits(&h)?;
        let next = l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if next == eos {
            eprintln!("eos hit");
            break;
        }
        gen_ids.push(next as u32);
        tok = next;
    }
    let out = model.detokenize(&gen_ids)?;
    eprintln!("generated: {out:?}");
    assert_eq!(gen_ids, vec![11, 353, 2688, 264, 5286, 303], "greedy stream must match oracle");
    Ok(())
}

/// DUET P3 kernel gates (docs/DUET.md):
/// (a) batched verify_block == K sequential steps, position by position;
/// (b) speculative block generation stream == greedy stream (lossless);
/// (c) amortization measured honestly: verify_block(K) time vs K steps.
#[test]
#[ignore]
fn bonsai27_block_verify_equivalence_and_speed() -> Result<(), Box<dyn std::error::Error>> {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    if !std::path::Path::new("shards_bonsai27_n1/shard_1.bmts").exists() {
        eprintln!("no shards");
        return Ok(());
    }

    let load = || {
        Qwen35Model::load(
            &["shards_bonsai27_n1/shard_1.bmts"],
            Card::load_dir("shards_bonsai27_n1").unwrap(),
        )
    };
    let block = [9419usize, 11, 353, 2688]; // "Hello" "," " I" "'m"

    // k=1 sanity: verify_block([t]) vs step(t) on fresh models
    {
        let mut m1 = load()?;
        let mut m2 = load()?;
        let h = m1.step(9419)?;
        let l1 = m1.logits(&h)?;
        let l2 = m2.verify_block(&[9419])?;
        let d = l1
            .iter()
            .zip(&l2[0])
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let n1: f32 = l1.iter().map(|v| v * v).sum();
        eprintln!("k=1 verify vs step: max_delta={d:.5} (logit l2^2={n1:.1})");
    }

    // (a) equivalence: sequential steps vs one batched verify
    let t_seq = std::time::Instant::now();
    let mut seq = load()?;
    let mut seq_logits: Vec<Vec<f32>> = Vec::new();
    for &t in &block {
        let h = seq.step(t)?;
        seq_logits.push(seq.logits(&h)?);
    }
    let seq_dt = t_seq.elapsed().as_secs_f64();

    let mut bat = load()?;
    let t_bat = std::time::Instant::now();
    let bat_logits = bat.verify_block(&block)?;
    let bat_dt = t_bat.elapsed().as_secs_f64();

    for (j, (s, b)) in seq_logits.iter().zip(&bat_logits).enumerate() {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..s.len() {
            let (x, y) = (s[i] as f64, b[i] as f64);
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        let c = (d / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32;
        let maxd = s.iter().zip(b).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("pos {j}: logits cos={c:.7} max_delta={maxd:.5}");
        assert!(c > 0.999, "batched position {j} diverged: cos {c}");
    }
    eprintln!(
        "sequential 4 steps: {seq_dt:.2}s | batched verify(4): {bat_dt:.2}s ({:.2}x)",
        seq_dt / bat_dt
    );

    // (b) losslessness: speculative block stream == greedy stream
    let greedy = || -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let mut m = load()?;
        let mut tok = 9419usize;
        let mut out = Vec::new();
        for _ in 0..6 {
            let h = m.step(tok)?;
            let l = m.logits(&h)?;
            tok = Qwen35Model::argmax(&l);
            out.push(tok);
        }
        Ok(out)
    };
    let spec = || -> Result<(Vec<usize>, ouro_cluster::infer::qwen35::SpecStats), Box<dyn std::error::Error>> {
        let mut m = load()?;
        let mut drafter = ouro_cluster::infer::qwen35::PromptLookup::default();
        // seed the drafter with the KNOWN continuation so drafts actually
        // hit: this measures the accept path, not the cold table
        drafter.observe(&[9419, 11, 353, 2688, 264, 5286, 303, 279]);
        Ok(m.generate_speculative_block(9419, 6, &mut drafter, 4)?)
    };
    let g = greedy()?;
    let (s, stats) = spec()?;
    eprintln!("greedy: {g:?}");
    eprintln!("spec:   {s:?} (hits {} misses {} no_draft {})", stats.hits, stats.misses, stats.no_draft);
    assert_eq!(g, s, "speculative block decode must be lossless");
    Ok(())
}

#[test]
#[ignore] // heavy: 27B oracle + full forward pass
fn bonsai27_oracle_diff() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();

    let logits_path = std::env::var("ORACLE_LOGITS")
        .unwrap_or_else(|_| "/tmp/opencode/bonsai_oracle_logits.f32".into());
    let cap_path = std::env::var("ORACLE_CAP")
        .unwrap_or_else(|_| "/tmp/opencode/bonsai_oracle_logits.f32.cap".into());
    let token: usize = std::env::var("ORACLE_TOKEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9419); // "Hello" via prism tokenizer (add_special)

    if !std::path::Path::new(&logits_path).exists() {
        eprintln!("no oracle logits at {logits_path}");
        return;
    }

    let ref_logits = read_f32s(&logits_path).unwrap();
    let ref_layers = read_cap(&cap_path).unwrap_or_default();
    eprintln!("oracle: {} logits, {} layer taps", ref_logits.len(), ref_layers.len());

    // --- Rust engine: mirror Qwen35Model::step but tap per-layer outputs ---
    let t_all = std::time::Instant::now();
    let mut model = Qwen35Model::load(
        &["shards_bonsai27_n1/shard_1.bmts"],
        Card::load_dir("shards_bonsai27_n1").unwrap(),
    )
    .unwrap();
    eprintln!("[t] load: {:.1}s", t_all.elapsed().as_secs_f64());
    let pos = model.current_pos();

    let mut x = model.stages_mut()[0].embed(token).unwrap();
    let t_step = std::time::Instant::now();
    let mut mine_layers: Vec<(u32, Vec<f32>)> = Vec::new();
    for s in model.stages_mut() {
        for il in s.layers().to_vec() {
            x = s.run_layer(il, &x, pos).unwrap();
            mine_layers.push((il, x.clone()));
        }
    }
    eprintln!("[t] step: {:.1}s", t_step.elapsed().as_secs_f64());
    let last = model.stages().len() - 1;
    if model.stages()[last].inner.output_norm_present() {
        x = model.stages()[last].inner.apply_output_norm(&x).unwrap();
    }
    let t_logits = std::time::Instant::now();
    let mine = model.logits(&x).unwrap();
    eprintln!("[t] logits: {:.1}s", t_logits.elapsed().as_secs_f64());

    // --- per-layer comparison (diagnostic, not the gate) ---
    let mut worst = (1.0f32, String::new());
    for (name, refd) in &ref_layers {
        let Some(il_str) = name.strip_prefix("l_out-") else {
            continue; // only l_out taps map onto our per-layer outputs
        };
        let il: u32 = il_str.parse().unwrap_or(u32::MAX);
        if let Some((_, mine_l)) = mine_layers.iter().find(|(l, _)| *l == il) {
            let c = cos(refd, mine_l);
            if c < worst.0 {
                worst = (c, name.clone());
            }
            if *name == "l_out-63" || c < 0.99 {
                eprintln!("{name}: cos={c:.6}");
            }
        }
    }
    eprintln!("layer cos: worst={:.6} @ {}", worst.0, worst.1);

    // --- THE GATE: logits cos + greedy top-1 (CONTRACTS.md L3) ---
    let c = cos(&ref_logits, &mine);
    let rt = ref_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let mt = mine
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let maxd = ref_logits.iter().zip(&mine).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("bonsai27 logits: cos={c:.6} ref_top={rt} rust_top={mt} max_delta={maxd:.4}");
    assert!(c > 0.999, "bonsai27 logit cos {c}");
    assert_eq!(rt, mt, "bonsai27 greedy token must match");
}

/// Isolation: per-layer, 4 sequential positions vs 1 batched call.
#[test]
#[ignore]
fn bonsai27_layer_batched_isolation() -> Result<(), Box<dyn std::error::Error>> {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    let load = || {
        Qwen35Model::load(
            &["shards_bonsai27_n1/shard_1.bmts"],
            Card::load_dir("shards_bonsai27_n1").unwrap(),
        )
    };
    let block = [9419usize, 11, 353, 2688];
    let mut sa = load()?;
    let mut sb = load()?;
    let layers = sa.stages()[0].layers().to_vec();
    // seed both with the same first token (embed only; run_layer takes normed input)
    let mut xs_seq: Vec<Vec<f32>> = Vec::new();
    let mut xs_bat: Vec<Vec<f32>> = Vec::new();
    for &t in &block {
        xs_seq.push(sa.stages_mut()[0].embed(t)?);
        xs_bat.push(sb.stages_mut()[0].embed(t)?);
    }
    for il in layers {
        // sequential: 4 run_layer calls (pos 0..3)
        let mut outs_seq = Vec::new();
        for (j, x) in xs_seq.iter().enumerate() {
            outs_seq.push(sa.stages_mut()[0].run_layer(il, x, j)?);
        }
        // batched: one call
        let outs_bat = sb.stages_mut()[0].run_layer_batched(il, &xs_bat.clone(), 0)?;
        let mut worst = (0.0f32, 0usize);
        for (j, (o1, o2)) in outs_seq.iter().zip(&outs_bat).enumerate() {
            let d: f32 = o1.iter().zip(o2).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
            if d > worst.0 {
                worst = (d, j);
            }
        }
        if worst.0 > 1e-2 {
            eprintln!("layer {il}: worst abs delta {:.4} at pos {} — FIRST DIVERGENCE", worst.0, worst.1);
            return Ok(());
        }
        xs_seq = outs_seq;
        xs_bat = outs_bat;
    }
    eprintln!("all layers match across 4 positions (sequential == batched)");
    // output norm + batched lm_head on the final positions
    let last = sa.stages().len() - 1;
    let mut h_seq = Vec::new();
    let mut h_bat = Vec::new();
    for x in &xs_seq {
        h_seq.push(sa.stages()[last].inner.apply_output_norm(x)?);
    }
    for x in &xs_bat {
        h_bat.push(sb.stages()[last].inner.apply_output_norm(x)?);
    }
    for j in 0..4 {
        let d = h_seq[j]
            .iter()
            .zip(&h_bat[j])
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("hidden[{j}] max_delta = {d:.6}");
    }
    let l_seq = sb.stages()[last].inner.logits_untied(&h_seq[0])?;
    let l_bat = &sb.stages()[last].inner.logits_untied_batched(&h_bat)?[0];
    let d = l_seq
        .iter()
        .zip(l_bat)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("logits[0] max_delta = {d:.5}");
    Ok(())
}



