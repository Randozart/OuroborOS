//! Rung-B orchestrator: drive a sharded pipeline across agent nodes.
//!
//! Stage 0 owns embeddings + tied head; intermediate stages consume/produce
//! ACTS activation frames; the final stage applies output_norm.

use anyhow::{bail, Result};
use ouro_cluster::pipeline::PipelinePlan;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::agent_client::{self, AgentTask};

/// One node in the run: plan stage + reachable agent address + shard path.
#[derive(Debug, Clone)]
pub struct PipelineNode {
    pub node: u16,
    pub addr: String,
    pub shard_path: String,
}

/// Result of a full pipeline generation.
#[derive(Debug, Clone)]
pub struct PipelineRun {
    pub token_ids: Vec<i32>,
    pub text: String,
    pub tok_per_sec: f64,
    /// mean ms per hop kind
    pub hop_ms: Vec<(String, f64)>,
}

const TIMEOUT: Duration = Duration::from_secs(900);

fn task(id: &str, name: &str, payload: String) -> AgentTask {
    AgentTask {
        id: id.to_string(),
        name: name.to_string(),
        payload,
        estimated_watts: 35,
        estimated_seconds: 900,
    }
}

/// Resolve plan + user addresses into ordered nodes.
pub fn plan_nodes(plan: &PipelinePlan, addrs: &[(String, String)]) -> Vec<PipelineNode> {
    let mut nodes = Vec::new();
    for stage in &plan.nodes {
        let want = format!("n{}", stage.node);
        let addr = addrs
            .iter()
            .find(|(id, _)| *id == want)
            .map(|(_, a)| a.clone())
            .or_else(|| addrs.get((stage.node as usize).saturating_sub(1)).map(|(_, a)| a.clone()));
        if let Some(addr) = addr {
            nodes.push(PipelineNode {
                node: stage.node,
                addr,
                shard_path: stage.file.clone(),
            });
        }
    }
    nodes
}

struct Hops {
    t: BTreeMap<String, (u128, u32)>,
}

impl Hops {
    fn new() -> Self {
        Self { t: BTreeMap::new() }
    }
    fn record(&mut self, name: &str, start: Instant) {
        let e = self.t.entry(name.to_string()).or_insert((0, 0));
        e.0 += start.elapsed().as_micros();
        e.1 += 1;
    }
    fn means(&self) -> Vec<(String, f64)> {
        self.t.iter().map(|(k, (us, c))| (k.clone(), *us as f64 / 1000.0 / *c as f64)).collect()
    }
}

fn call(addr: &str, name: &str, id: &str, payload: String, hops: &mut Hops) -> Result<String> {
    let s = Instant::now();
    let r = agent_client::execute_timeout(addr, &task(id, name, payload), TIMEOUT)?;
    hops.record(name, s);
    if r.status != "Success" {
        bail!("{} -> {}: {}", name, addr, r.output);
    }
    Ok(r.output)
}

/// Tokenize text through a node's vocab.
pub fn tokenize(addr: &str, text: &str) -> Result<Vec<i32>> {
    let mut hops = Hops::new();
    let out = call(addr, "tokenize", "tk", format!("|{}", text), &mut hops)?;
    Ok(out.split(',').filter_map(|p| p.trim().parse().ok()).collect())
}

/// Run a generation: `ids` prefill, `n_gen` greedy continuation tokens.
pub fn run(nodes: &[PipelineNode], ids: &[i32], n_gen: u32) -> Result<PipelineRun> {
    if nodes.is_empty() || ids.is_empty() {
        bail!("pipeline needs nodes and prefill ids");
    }
    let mut hops = Hops::new();

    let mut sample_addr = None;
    for n in nodes {
        let summary = call(&n.addr, "stage_setup", "su", n.shard_path.clone(), &mut hops)?;
        if summary.contains("head=tied") || summary.contains("head=untied") {
            sample_addr = Some(n.addr.clone());
        }
        call(&n.addr, "stage_reset", "sr", String::new(), &mut hops)?;
    }

    let head = &nodes[0]; // stage 0 owns token_embd (embed side)
    let sampler = sample_addr.clone().unwrap_or_else(|| nodes[0].addr.clone());

    // Prefill token by token; sample only after the final prompt position.
    let mut pos: usize = 0;
    let mut acts = String::new();
    for &id in ids {
        acts = call(&head.addr, "stage_token", "pf", format!("{}|{}", pos, id), &mut hops)?;
        for n in &nodes[1..] {
            acts = call(&n.addr, "stage_step", "pf", acts, &mut hops)?;
        }
        pos += 1;
    }
    let gen_start = Instant::now();
    let mut cur: i32 = call(&sampler, "stage_sample", "pf", acts, &mut hops)?
        .trim()
        .parse()?;
    let mut out_ids = vec![cur];

    for _ in 0..n_gen {
        acts = call(&head.addr, "stage_token", "gn", format!("{}|{}", pos, cur), &mut hops)?;
        for n in &nodes[1..] {
            acts = call(&n.addr, "stage_step", "gn", acts, &mut hops)?;
        }
        pos += 1;
        cur = call(&sampler, "stage_sample", "gn", acts, &mut hops)?
            .trim()
            .parse()?;
        out_ids.push(cur);
    }

    let tok_per_sec = out_ids.len() as f64 / gen_start.elapsed().as_secs_f64().max(1e-6);
    let csv = out_ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
    let text = call(&sampler, "detok", "dt", csv, &mut hops).unwrap_or_default();

    Ok(PipelineRun {
        token_ids: out_ids,
        text,
        tok_per_sec,
        hop_ms: hops.means(),
    })
}
