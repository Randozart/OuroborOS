//! Rung-B validation: pipeline executed by three agents over TCP must
//! produce exactly the same greedy token stream as the in-process
//! PipelineModel over the same shards (same code, same bytes).

use ouro_cluster::bmts::{write_shard, BmtsTensor};
use ouro_cluster::infer::{ArchConfig, PipelineModel};
use ouro_shell::pipeline;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const DTYPE_F32: u32 = 0;
const DTYPE_F16: u32 = 1;
const DTYPE_TQ1: u32 = 34;

fn toy_cfg() -> ArchConfig {
    ArchConfig {
        n_embd: 256,
        n_head: 2,
        n_head_kv: 1,
        n_ff: 512,
        n_rot: 128,
        eps: 1e-5,
        rope_base: 10000.0,
        n_vocab: 64,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() & 0xFF) as u8).collect()
    }
    /// positive-ish f32 gains around 1.0
    fn vec_f32(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (0.75 + (self.next() % 1000) as f32 / 2000.0).to_le_bytes())
            .collect()
    }
    /// benign f16 (finite, no nan/inf patterns)
    fn vec_f16(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| ((self.next() % 0x7000) as u16).to_le_bytes())
            .collect()
    }
}

fn tq1(payload_len_rows: usize, in_len: usize, rng: &mut Rng) -> Vec<u8> {
    let mut b = rng.bytes(payload_len_rows * (in_len / 256) * 54);
    // Sanitize TQ1_0 block scales (bytes 52..54 of each block) to ~[0.25, 1.0)
    for blk in b.chunks_mut(54) {
        blk[52] = (rng.next() & 0xFF) as u8;
        blk[53] = (0x38 + (rng.next() % 4)) as u8;
    }
    b
}

/// Build one shard file; returns path.
fn make_shard(dir: &PathBuf, node: u16, layers: &[u32], cfg: &ArchConfig, rng: &mut Rng, first: bool, last: bool) -> PathBuf {
    let mut tensors: Vec<BmtsTensor> = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut off = 0u64;
    let push = |name: String, shape: Vec<u64>, dtype: u32, blob: Vec<u8>, tensors: &mut Vec<BmtsTensor>, data: &mut Vec<u8>, off: &mut u64| {
        tensors.push(BmtsTensor { name, shape, dtype, offset: *off, length: blob.len() as u64 });
        data.extend_from_slice(&blob);
        *off += blob.len() as u64;
    };

    if first {
        // embedding rows = n_embd f16, vocab rows
        push("token_embd.weight".into(), vec![cfg.n_embd as u64, cfg.n_vocab as u64], DTYPE_F16, rng.vec_f16(cfg.n_embd * cfg.n_vocab), &mut tensors, &mut data, &mut off);
    }
    for &l in layers {
        let p = format!("blk.{}.", l);
        push(format!("{}attn_norm.weight", p), vec![cfg.n_embd as u64], DTYPE_F32, rng.vec_f32(cfg.n_embd), &mut tensors, &mut data, &mut off);
        push(format!("{}attn_q.weight", p), vec![cfg.n_embd as u64, cfg.n_embd as u64], DTYPE_TQ1, tq1(cfg.n_embd, cfg.n_embd, rng), &mut tensors, &mut data, &mut off);
        push(format!("{}attn_k.weight", p), vec![cfg.n_embd as u64, (cfg.kv_dim()) as u64], DTYPE_TQ1, tq1(cfg.kv_dim(), cfg.n_embd, rng), &mut tensors, &mut data, &mut off);
        push(format!("{}attn_v.weight", p), vec![cfg.n_embd as u64, (cfg.kv_dim()) as u64], DTYPE_TQ1, tq1(cfg.kv_dim(), cfg.n_embd, rng), &mut tensors, &mut data, &mut off);
        push(format!("{}attn_sub_norm.weight", p), vec![cfg.n_embd as u64], DTYPE_F32, rng.vec_f32(cfg.n_embd), &mut tensors, &mut data, &mut off);
        push(format!("{}attn_output.weight", p), vec![cfg.n_embd as u64, cfg.n_embd as u64], DTYPE_TQ1, tq1(cfg.n_embd, cfg.n_embd, rng), &mut tensors, &mut data, &mut off);
        push(format!("{}ffn_norm.weight", p), vec![cfg.n_embd as u64], DTYPE_F32, rng.vec_f32(cfg.n_embd), &mut tensors, &mut data, &mut off);
        push(format!("{}ffn_up.weight", p), vec![cfg.n_embd as u64, cfg.n_ff as u64], DTYPE_TQ1, tq1(cfg.n_ff, cfg.n_embd, rng), &mut tensors, &mut data, &mut off);
        push(format!("{}ffn_gate.weight", p), vec![cfg.n_embd as u64, cfg.n_ff as u64], DTYPE_TQ1, tq1(cfg.n_ff, cfg.n_embd, rng), &mut tensors, &mut data, &mut off);
        push(format!("{}ffn_sub_norm.weight", p), vec![cfg.n_ff as u64], DTYPE_F32, rng.vec_f32(cfg.n_ff), &mut tensors, &mut data, &mut off);
        push(format!("{}ffn_down.weight", p), vec![cfg.n_ff as u64, cfg.n_embd as u64], DTYPE_TQ1, tq1(cfg.n_embd, cfg.n_ff, rng), &mut tensors, &mut data, &mut off);
    }
    if last {
        push("output_norm.weight".into(), vec![cfg.n_embd as u64], DTYPE_F32, rng.vec_f32(cfg.n_embd), &mut tensors, &mut data, &mut off);
    }

    let path = dir.join(format!("shard_{}.bmts", node));
    write_shard(path.to_str().unwrap(), node, &tensors, &data).unwrap();
    path
}

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

fn agent_bin() -> PathBuf {
    if let Ok(p) = std::env::var("OURO_AGENT_BIN") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/debug/ouro-agent")
}

fn start_agent(port: u16, arch_json: &str) -> Child {
    let bin = agent_bin();
    assert!(bin.exists(), "build ouro-agent first (make test-all)");
    Command::new(&bin)
        .env("OURO_PORT", port.to_string())
        .env("OURO_ARCH", arch_json)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn agent")
}

fn wait_ready(addr: &str) {
    for _ in 0..100 {
        if let Ok(mut s) = TcpStream::connect(addr) {
            s.set_read_timeout(Some(Duration::from_millis(250))).ok();
            if s.write_all(b"ping\n").is_ok() && s.flush().is_ok() {
                let mut buf = String::new();
                if BufReader::new(&mut s).read_line(&mut buf).is_ok() && buf.trim() == "pong" {
                    return;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("agent at {} never ready", addr);
}

fn in_process_greedy(paths: &[PathBuf], cfg: ArchConfig, ids: &[i32], n_gen: u32) -> Vec<i32> {
    let refs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();
    let mut model = PipelineModel::load(&refs, cfg).expect("load");
    let idus: Vec<usize> = ids.iter().map(|i| *i as usize).collect();
    let mut h = model.prefill(&idus).expect("prefill");
    let mut pos = ids.len();
    let mut out = Vec::new();
    for _ in 0..n_gen + 1 {
        let l = model.logits(&h);
        let t = PipelineModel::argmax(&l) as i32;
        out.push(t);
        h = model.step_token(t as usize, pos).expect("step");
        pos += 1;
    }
    out
}

#[test]
fn test_rung_b_tcp_matches_in_process() {
    let cfg = toy_cfg();
    let arch_json = serde_json::to_string(&cfg).unwrap();
    let dir = std::env::temp_dir().join("ouro_rungb_test");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();

    let mut rng = Rng(1234);
    let p1 = make_shard(&dir, 1, &[0, 1], &cfg, &mut rng, true, false);
    let p2 = make_shard(&dir, 2, &[2, 3], &cfg, &mut rng, false, false);
    let p3 = make_shard(&dir, 3, &[4, 5], &cfg, &mut rng, false, true);
    let paths = vec![p1.clone(), p2.clone(), p3.clone()];

    let ids = vec![5i32, 9, 2, 42];
    let n_gen = 4u32;

    // Oracle: in-process.
    let oracle = in_process_greedy(&paths, cfg, &ids, n_gen);
    assert!(!oracle.is_empty());

    // Subjects: three agents.
    let ports: Vec<u16> = (0..3).map(|_| free_port()).collect();
    let mut kids: Vec<Child> = Vec::new();
    for p in &ports {
        kids.push(start_agent(*p, &arch_json));
    }
    let addrs: Vec<String> = ports.iter().map(|p| format!("127.0.0.1:{}", p)).collect();
    for a in &addrs {
        wait_ready(a);
    }

    let nodes: Vec<pipeline::PipelineNode> = addrs
        .iter()
        .enumerate()
        .map(|(i, a)| pipeline::PipelineNode {
            node: (i + 1) as u16,
            addr: a.clone(),
            shard_path: paths[i].to_str().unwrap().to_string(),
        })
        .collect();

    let run = pipeline::run(&nodes, &ids, n_gen).expect("pipeline run");
    eprintln!("tcp run ids={:?} oracle={:?} hops={:?}", run.token_ids, oracle, run.hop_ms);

    assert_eq!(oracle, run.token_ids, "TCP pipeline must match in-process exactly");
    // hop latency sanity: activation round trips measured
    assert!(run.hop_ms.iter().any(|(n, _)| n == "stage_step"));

    for mut k in kids {
        k.kill().ok();
        k.wait().ok();
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore] // requires real 3xTQ1 shards + model: OURO_MODEL_PATH + ./shards
fn test_rung_b_real_model() {
    let model = std::env::var("OURO_MODEL_PATH").expect("OURO_MODEL_PATH");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&root).unwrap();
    let plan = ouro_cluster::pipeline::PipelinePlan::load("shards/shard_map.json").unwrap();
    let cfg = ArchConfig::bitnet_2b();

    // Oracle: in-process over the same shards.
    let refs: Vec<String> = plan.nodes.iter().map(|s| s.file.clone()).collect();
    let refsl: Vec<&str> = refs.iter().map(|r| r.as_str()).collect();
    let mut m = PipelineModel::load(&refsl, cfg).expect("oracle load");
    let mut mparams = None;
    {
        // tokenize once via a throwaway vocab-only load in THIS process
        let v = bitnet_rs::BitNetModel::load_vocab_only(&model).unwrap();
        mparams = Some(v.tokenize("The capital of France", true));
    }
    let toks = mparams.unwrap();
    assert!(!toks.is_empty(), "tokenizer produced nothing");
    let ids32: Vec<i32> = toks.iter().map(|t| *t as i32).collect();
    let idus: Vec<usize> = ids32.iter().map(|i| *i as usize).collect();
    let mut h = m.prefill(&idus).unwrap();
    let mut pos = idus.len();
    let mut oracle = Vec::new();
    for _ in 0..5 {
        let l = m.logits(&h);
        let t = PipelineModel::argmax(&l) as i32;
        oracle.push(t);
        h = m.step_token(t as usize, pos).unwrap();
        pos += 1;
    }

    // Subjects: 3 agents (bitnet default arch).
    let ports: Vec<u16> = (0..3).map(|_| free_port()).collect();
    let mut kids = Vec::new();
    for p in &ports {
        let bin = agent_bin();
        kids.push(
            Command::new(bin)
                .env("OURO_PORT", p.to_string())
                .env("OURO_MODEL_PATH", &model)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let addrs: Vec<(String, String)> = ports
        .iter()
        .enumerate()
        .map(|(i, p)| (format!("n{}", i + 1), format!("127.0.0.1:{}", p)))
        .collect();
    for (_, a) in &addrs {
        wait_ready(a);
    }

    let ids = pipeline::tokenize(&addrs[0].1, "The capital of France").unwrap();
    assert_eq!(ids, ids32, "vocab-only vs in-test tokenize must agree");

    let nodes = pipeline::plan_nodes(&plan, &addrs);
    assert_eq!(nodes.len(), 3);
    let run = pipeline::run(&nodes, &ids, 4).unwrap();
    eprintln!("real TCP ids={:?}\nreal oracle={:?}\ntext={:?}", run.token_ids, oracle, run.text);
    for (h, ms) in &run.hop_ms {
        eprintln!("  hop {}: {:.1} ms", h, ms);
    }
    assert_eq!(oracle, run.token_ids, "TCP pipeline must match in-process on the real model");

    for mut k in kids {
        k.kill().ok();
        k.wait().ok();
    }
}
