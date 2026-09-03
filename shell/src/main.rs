use std::io::{self, BufRead, Write};

use anyhow::Result;
use ouro_cluster::beast::topology::ClusterTopology;
use ouro_cluster::scheduler::Scheduler;

use ouro_hiss::agent_client;
use ouro_hiss::context::Context;
use ouro_hiss::formatter::Formatter;
use ouro_hiss::parser::interpret;
use ouro_hiss::propositions;

/// The HISS wordmark (docs/brand/hiss-ascii.txt), frozen at compile time.
const HISS_WORDMARK: &str = "\
   ▄█    █▄     ▄█     ▄████████    ▄████████
  ███    ███   ███    ███    ███   ███    ███
  ███    ███   ███▌   ███    █▀    ███    █▀
 ▄███▄▄▄▄███▄▄ ███▌   ███          ███
▀▀███▀▀▀▀███▀  ███▌ ▀███████████ ▀███████████
  ███    ███   ███           ███          ███
  ███    ███   ███     ▄█    ███    ▄█    ███
  ███    █▀    █▀    ▄████████▀   ▄████████▀";

/// Print the HISS banner.
fn banner() {
    println!("{HISS_WORDMARK}");
    println!();
    println!("  HISS — Hierarchical Interactive Shell System");
    println!("  OUROBOROS: One Unified Runtime Orchestrating");
    println!("             a Bunch Of Random Old Servers");
    println!("             (the 'a' is silent)");
    println!();
    println!("  The cluster is one machine.");
    println!("  Type ? for cluster summary.");
    println!();
}

/// Load a demo topology for testing.
fn demo_topology() -> ClusterTopology {
    use ouro_cluster::beast::topology::NodeEntry;

    let mut topo = ClusterTopology::new();

    topo.nodes.push(NodeEntry {
        id: "n1".to_string(),
        hostname: "alienware".to_string(),
        ip: "192.168.1.101".to_string(),
        cpu_model: "i7-6700T".to_string(),
        cores: 4,
        threads: 8,
        has_avx: true,
        has_avx2: true,
        has_sse42: true,
        ram_mib: 16384,
        tdp_watts: 35,
        has_gpu: false,
        gpu_model: String::new(),
        gpu_vram_mib: 0,
        gpu_driver: String::new(),
    });

    topo.nodes.push(NodeEntry {
        id: "n2".to_string(),
        hostname: "thinkpad".to_string(),
        ip: "192.168.1.102".to_string(),
        cpu_model: "i5-3320M".to_string(),
        cores: 2,
        threads: 4,
        has_avx: true,
        has_avx2: false,
        has_sse42: true,
        ram_mib: 8192,
        tdp_watts: 35,
        has_gpu: false,
        gpu_model: String::new(),
        gpu_vram_mib: 0,
        gpu_driver: String::new(),
    });

    topo.nodes.push(NodeEntry {
        id: "n3".to_string(),
        hostname: "desktop".to_string(),
        ip: "192.168.1.103".to_string(),
        cpu_model: "i5-4590".to_string(),
        cores: 4,
        threads: 4,
        has_avx: true,
        has_avx2: true,
        has_sse42: true,
        ram_mib: 32768,
        tdp_watts: 84,
        has_gpu: false,
        gpu_model: String::new(),
        gpu_vram_mib: 0,
        gpu_driver: String::new(),
    });

    topo.power_budget_watts = 500;
    // the box we're literally sitting on has an RTX 3060
    if let Some(n1) = topo.nodes.iter_mut().find(|n| n.id == "n1") {
        n1.has_gpu = true;
        n1.gpu_model = "NVIDIA GeForce RTX 3060".to_string();
        n1.gpu_vram_mib = 12288;
        n1.gpu_driver = "610.57.04".to_string();
    }
    topo
}

/// Parse --nodes ip:port,... CLI argument into (node_id, addr) pairs.
/// Convert agent telemetry into live property cache entries.
fn tel_props(tel: &ouro_hiss::agent_client::AgentTelemetry) -> std::collections::HashMap<String, String> {
    let mut props = std::collections::HashMap::new();
    props.insert("power".to_string(), format!("{}W", tel.power_watts));
    props.insert("temp".to_string(), format!("{}C", tel.temp_c));
    props.insert("ram".to_string(), format!("{}MiB used of {}MiB", tel.ram_used_mib, tel.ram_total_mib));
    props.insert("cpu".to_string(), tel.cpu_model.clone());
    props.insert("status".to_string(), "AWAKE".to_string());
    props.insert("load".to_string(), format!("{:.2}", tel.load_avg));
    if !tel.gpus.is_empty() {
        let g = &tel.gpus[0];
        props.insert("gpu".to_string(), format!("{} ({}MiB)", g.model, g.vram_mib));
    }
    props
}

fn parse_nodes_arg(arg: &str) -> Vec<(String, String)> {
    arg.split(',')
        .filter(|s| !s.is_empty())
        .enumerate()
        .map(|(i, addr)| {
            let addr = addr.trim().to_string();
            let id = format!("n{}", i + 1);
            (id, addr)
        })
        .collect()
}

fn main() -> Result<()> {
    banner();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let nodes_arg = args
        .windows(2)
        .find(|w| w[0] == "--nodes")
        .map(|w| w[1].clone());

    let mut topology = demo_topology();
    let mut scheduler = Scheduler::new(topology.clone());
    let mut ctx = Context::new();
    let mut fmt = Formatter::new(false);

    let node_addrs: Vec<(String, String)> = nodes_arg
        .as_deref()
        .map(parse_nodes_arg)
        .unwrap_or_default();

    let mut config = propositions::ShellConfig::new();
    config.node_addrs = node_addrs.clone();

    if !node_addrs.is_empty() {
        println!("Probing {} nodes...", node_addrs.len());
        for (id, addr) in &node_addrs {
            match agent_client::telemetry(addr) {
                Ok(tel) => {
                    ctx.cache_properties(id, tel_props(&tel));
                    println!(
                        "  {}: {} [FOUND] ({}, {}MiB, {}W)",
                        id, addr, tel.cpu_model, tel.ram_total_mib, tel.power_watts
                    );
                }
                Err(e) => {
                    println!("  {}: {} [FAILED] {}", id, addr, e);
                }
            }
        }
        println!();
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut recovery = ouro_cluster::error_recovery::ErrorRecovery::new();

    loop {
        print!("hiss> ");
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            println!();
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        if matches!(input, "quit" | "exit" | "q") {
            println!("Goodbye.");
            break;
        }

        let cmd = interpret(input);

        match &cmd {
            ouro_hiss::parser::Command::Probe if !node_addrs.is_empty() => {
                println!("Probing all nodes...");
                for (id, addr) in &node_addrs {
                    match agent_client::telemetry(addr) {
                        Ok(tel) => {
                            ctx.cache_properties(id, tel_props(&tel));
                            println!(
                                "  {}: {}, {}MiB, {}W [FOUND]",
                                id, tel.cpu_model, tel.ram_total_mib, tel.power_watts
                            );
                        }
                        Err(_) => {
                            println!("  {}: {} [OFFLINE]", id, addr);
                        }
                    }
                }
                continue;
            }
            _ => {}
        }

        match propositions::handle(cmd, &mut topology, &mut scheduler, &mut ctx, &mut fmt, &config, &mut recovery) {
            Ok(output) => println!("{}", output),
            Err(e) => println!("Error: {}", e),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_topology_has_nodes() {
        let topo = demo_topology();
        assert_eq!(topo.node_count(), 3);
    }

    #[test]
    fn test_demo_topology_budget() {
        let topo = demo_topology();
        assert_eq!(topo.power_budget_watts, 500);
    }

    #[test]
    fn test_parse_nodes_arg() {
        let nodes = parse_nodes_arg("127.0.0.1:9501,127.0.0.1:9502");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0], ("n1".into(), "127.0.0.1:9501".into()));
        assert_eq!(nodes[1], ("n2".into(), "127.0.0.1:9502".into()));
    }

    #[test]
    fn test_parse_nodes_arg_empty() {
        let nodes = parse_nodes_arg("");
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_handle_context_set() {
        let topo = demo_topology();
        let mut sched = Scheduler::new(topo.clone());
        let mut ctx = Context::new();
        let mut fmt = Formatter::new(false);
        let mut topo = topo;
        let config = propositions::ShellConfig::new();
        let mut recovery = ouro_cluster::error_recovery::ErrorRecovery::new();
        let cmd = interpret("n1");
        let out = propositions::handle(cmd, &mut topo, &mut sched, &mut ctx, &mut fmt, &config, &mut recovery).unwrap();
        assert_eq!(out, "n1 selected.");
    }
}
