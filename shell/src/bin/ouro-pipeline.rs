//! ouro-pipeline: run one model across many machines.
//!
//! Usage:
//!   ouro-pipeline --nodes n1@127.0.0.1:9501,n2@...:9502,..
//!                 [--plan shards/shard_map.json]
//!                 [--prompt "text" | --ids 1,2,3]
//!                 [--tokens 16]

use anyhow::{bail, Result};
use ouro_shell::pipeline;

fn parse_nodes(arg: &str) -> Vec<(String, String)> {
    arg.split(',')
        .filter_map(|p| {
            let (id, addr) = p.split_once('@')?;
            Some((id.trim().to_string(), addr.trim().to_string()))
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
    };

    let addrs = get("--nodes").map(|a| parse_nodes(&a)).unwrap_or_default();
    if addrs.is_empty() {
        bail!("--nodes n1@host:port,n2@... required");
    }
    let plan_path = get("--plan").unwrap_or_else(|| "shards/shard_map.json".into());
    let n_gen: u32 = get("--tokens").unwrap_or("16".into()).parse()?;

    let plan = ouro_cluster::pipeline::PipelinePlan::load(&plan_path)?;
    let nodes = pipeline::plan_nodes(&plan, &addrs);
    if nodes.len() != plan.stage_count() {
        bail!(
            "plan has {} stages, addresses matched {}",
            plan.stage_count(),
            nodes.len()
        );
    }

    println!(
        "OurobourOS pipeline: {} stages across {} nodes ({})",
        nodes.len(),
        nodes.len(),
        plan.model
    );
    for n in &nodes {
        println!("  stage {} -> {} [{}]", n.node, n.addr, n.shard_path);
    }

    let ids: Vec<i32> = if let Some(raw) = get("--ids") {
        raw.split(',').filter_map(|p| p.trim().parse().ok()).collect()
    } else if let Some(prompt) = get("--prompt") {
        println!("Tokenizing via stage 0 vocab...");
        pipeline::tokenize(&nodes[0].addr, &prompt)?
    } else {
        bail!("--prompt or --ids required");
    };
    if ids.is_empty() {
        bail!("empty prefill ids");
    }
    println!("prefill: {} tokens, generate: {} tokens", ids.len(), n_gen);

    let t0 = std::time::Instant::now();
    let run = pipeline::run(&nodes, &ids, n_gen)?;
    println!("\n=== result ({}s) ===", t0.elapsed().as_secs_f64().round());
    println!("token ids: {:?}", run.token_ids);
    println!("text: \"{}\"", run.text.replace('\n', "\\n"));
    println!("throughput: {:.2} tok/s", run.tok_per_sec);
    println!("mean hop times:");
    for (name, ms) in &run.hop_ms {
        println!("  {:12} {:8.1} ms", name, ms);
    }
    Ok(())
}
