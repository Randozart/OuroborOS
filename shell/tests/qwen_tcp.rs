//! M3 simulation: Qwen3.8-9B running as a 4-stage TCP pipeline vs in-process.
//! Run: make test-all, then
//!   OURO_AGENT_BIN=.../target/release/ouro-agent \
//!   cargo test --release -p ouro-shell --test qwen_tcp -- --ignored

use ouro_cluster::infer::qwen35::{Card, Qwen35Model};
use ouro_hiss::pipeline;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn agent_bin() -> PathBuf {
    std::env::var("OURO_AGENT_BIN").map(PathBuf::from).unwrap_or_else(|_| root().join("target/debug/ouro-agent"))
}

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
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

#[test]
#[ignore] // heavy: 9B x (1 local + 4 agents)
fn test_qwen9b_four_node_tcp_pipeline() {
    let r = root();
    std::env::set_current_dir(&r).unwrap();
    let model = std::env::var("CAP_MODEL")
        .unwrap_or("/home/randozart/Downloads/Qwen3.8-9B-Q6_K.gguf".into());
    if !std::path::Path::new(&model).exists() || !r.join("shards9b/model.json").exists() {
        eprintln!("model or shards9b missing");
        return;
    }
    let card = Card::load("shards9b/model.json").unwrap();
    let paths = ["shards9b/shard_1.bmts", "shards9b/shard_2.bmts", "shards9b/shard_3.bmts", "shards9b/shard_4.bmts"];

    // 1) In-process oracle (then drop before agents load 7.5 GB).
    let oracle = {
        let mut m = Qwen35Model::load(&paths, card.clone()).unwrap();
        let mut ids = Vec::new();
        let mut tok = 323usize; // arbitrary token id for determinism check
        for _ in 0..5 {
            let h = m.step(tok).unwrap();
            let l = m.logits(&h).unwrap();
            tok = Qwen35Model::argmax(&l);
            ids.push(tok as i32);
        }
        ids
    };

    // 2) Four agents, qwen card via OURO_ARCH.
    let card_json = std::fs::read_to_string("shards9b/model.json").unwrap();
    let ports: Vec<u16> = (0..4).map(|_| free_port()).collect();
    let mut kids: Vec<Child> = Vec::new();
    for p in &ports {
        kids.push(
            Command::new(agent_bin())
                .env("OURO_PORT", p.to_string())
                .env("OURO_ARCH", &card_json)
                .env("OURO_MODEL_PATH", &model)
                .env("OURO_N_THREADS", "2")
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

    let plan = ouro_cluster::pipeline::PipelinePlan::load("shards9b/shard_map.json").unwrap();
    let nodes = pipeline::plan_nodes(&plan, &addrs);
    assert_eq!(nodes.len(), 4);

    // Same 5-step stream as oracle: feed identical token sequence.
    // (Orchestrator expects prefill ids; use token 323 + 4 continuation
    //  via the sampler loop equivalence: prefill [323] x1, generate 4.)
    let run = pipeline::run(&nodes, &[323], 4).unwrap();
    eprintln!("tcp    ids={:?}", run.token_ids);
    eprintln!("oracle ids={:?}", oracle);
    assert_eq!(oracle, &run.token_ids[..], "9B TCP pipeline must match in-process");

    for mut k in kids {
        k.kill().ok();
        k.wait().ok();
    }
}
