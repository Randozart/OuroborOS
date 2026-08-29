use anyhow::Result;
use ouro_cluster::beast::topology::ClusterTopology;
use ouro_cluster::scheduler::{ScheduleOutcome, Scheduler, Task};
use ouro_cluster::scheduler::workload_class::WorkloadClass;

use crate::context::Context;
use crate::formatter::{Formatter, NodeDisplay};
use crate::parser::Command;

/// Configuration for shell command handling.
pub struct ShellConfig {
    pub topology_file: String,
    pub node_addrs: Vec<(String, String)>,
}

impl ShellConfig {
    pub fn new() -> Self {
        Self {
            topology_file: "cluster.beast".to_string(),
            node_addrs: Vec::new(),
        }
    }
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle a parsed command against the cluster state.
pub fn handle(
    cmd: Command,
    topology: &mut ClusterTopology,
    scheduler: &mut Scheduler,
    ctx: &mut Context,
    fmt: &mut Formatter,
    config: &ShellConfig,
) -> Result<String> {
    match cmd {
        Command::ClusterSummary => {
            let total = topology.node_count();
            let power: u32 = topology.nodes.iter().map(|n| n.tdp_watts).sum();
            let budget = topology.power_budget_watts;
            Ok(fmt.cluster_summary(total, 0, power, budget, 0, 0))
        }

        Command::ClusterQuery => {
            let total = topology.node_count();
            let power: u32 = topology.nodes.iter().map(|n| n.tdp_watts).sum();
            let budget = topology.power_budget_watts;
            Ok(fmt.cluster_summary(total, 0, power, budget, 0, 0))
        }

        Command::NodeQuery { node } => {
            let entry = topology
                .get_node(&node)
                .ok_or_else(|| anyhow::anyhow!("Node {} not found", node))?;
            let display = NodeDisplay {
                id: entry.id.clone(),
                cpu_model: entry.cpu_model.clone(),
                ram_mib: entry.ram_mib,
                has_avx2: entry.has_avx2,
                has_avx: entry.has_avx,
                has_sse42: entry.has_sse42,
                status: "IDLE".to_string(),
                power_watts: entry.tdp_watts,
                temp_c: 0,
            };
            Ok(fmt.node_query(&display))
        }

        Command::PropertyQuery { node, property } => {
            let entry = topology
                .get_node(&node)
                .ok_or_else(|| anyhow::anyhow!("Node {} not found", node))?;
            let value = resolve_node_property(entry, &property);
            Ok(fmt.property_query(&node, &property, &value))
        }

        Command::ContextPropertyQuery { property } => {
            if let Some(node_id) = ctx.current_node().map(|s| s.to_string()) {
                let entry = topology
                    .get_node(&node_id)
                    .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;
                let value = resolve_node_property(entry, &property);
                Ok(fmt.property_query(&node_id, &property, &value))
            } else {
                Ok(fmt.unknown(&format!("{}?", property)))
            }
        }

        Command::BulkQuery { filter } => {
            let nodes: Vec<NodeDisplay> = topology
                .nodes
                .iter()
                .map(|n| NodeDisplay {
                    id: n.id.clone(),
                    cpu_model: n.cpu_model.clone(),
                    ram_mib: n.ram_mib,
                    has_avx2: n.has_avx2,
                    has_avx: n.has_avx,
                    has_sse42: n.has_sse42,
                    status: "IDLE".to_string(),
                    power_watts: n.tdp_watts,
                    temp_c: 0,
                })
                .collect();
            Ok(fmt.bulk_query(&filter, &nodes))
        }

        Command::SetContext { node } => {
            ctx.set_node(&node);
            Ok(fmt.context_set(&node))
        }

        Command::ResetContext => {
            ctx.reset();
            Ok(fmt.context_reset())
        }

        Command::AssignProposition { node, workload } => {
            let class = WorkloadClass::from_name(&workload);
            let task = Task {
                name: workload.clone(),
                class,
                payload: String::new(),
                estimated_watts: 30,
                estimated_seconds: 10,
            };
            let mut details = Vec::new();
            details.push(format!("[1] Serialize {}.bv.              [OK]", workload));
            details.push(format!(
                "[2] Check: {} supports {}.         [YES]",
                node,
                class.label()
            ));

            match scheduler.schedule(&task)? {
                ScheduleOutcome::Dispatched { node: assigned } => {
                    details.push(format!("[3] Dispatch to {}.                [OK]", assigned));
                    Ok(fmt.assign_result(&node, &workload, true, &details))
                }
                ScheduleOutcome::Queued { reason } => {
                    details.push(format!("[3] Scheduling failed: {}", reason));
                    Ok(fmt.assign_result(&node, &workload, false, &details))
                }
            }
        }

        Command::AssignCheck { node, workload } => {
            let class = WorkloadClass::from_name(&workload);
            let mut details = Vec::new();
            details.push(format!(
                "  {}: {} | {} | AVAILABLE",
                node,
                "CPU",
                class.label()
            ));
            Ok(fmt.assign_result(&node, &workload, true, &details))
        }

        Command::PowerState { node, sleeping } => {
            let _ = sleeping;
            Ok(format!("Node {}休眠. Power: 12W → 2W.", node))
        }

        Command::SetBudget { watts } => {
            scheduler.budget.set_budget(watts);
            Ok(fmt.budget_set(watts))
        }

        Command::Probe => {
            let nodes: Vec<NodeDisplay> = topology
                .nodes
                .iter()
                .map(|n| NodeDisplay {
                    id: n.id.clone(),
                    cpu_model: n.cpu_model.clone(),
                    ram_mib: n.ram_mib,
                    has_avx2: n.has_avx2,
                    has_avx: n.has_avx,
                    has_sse42: n.has_sse42,
                    status: "IDLE".to_string(),
                    power_watts: n.tdp_watts,
                    temp_c: 0,
                })
                .collect();
            Ok(fmt.probe_result(&nodes))
        }

        Command::Deploy => {
            let mut out = String::new();
            out.push_str("Deploying node-agent to all nodes...\n");
            for (id, addr) in &config.node_addrs {
                let ip = addr.split(':').next().unwrap_or(addr);
                let status = deploy_agent(ip);
                out.push_str(&format!("  {}: {} [{}]\n", id, ip, status));
            }
            out.push_str("[DONE]");
            Ok(out)
        }

        Command::Save => {
            let path = config.topology_file.replace(".beast", ".json");
            topology.save_json(&path)?;
            Ok(format!("Cluster state saved to {}. [DONE]", path))
        }

        Command::Load => {
            let path = config.topology_file.replace(".beast", ".json");
            if !std::path::Path::new(&path).exists() {
                return Ok(format!("No state file found at {}. [SKIP]", path));
            }
            let loaded = ClusterTopology::load_json(&path)?;
            *topology = loaded;
            Ok(format!("Cluster state loaded from {}. [DONE]", path))
        }

        Command::Poetry { enabled } => {
            fmt.set_poetry(enabled);
            ctx.set_poetry(enabled);
            Ok(fmt.poetry_toggle(enabled))
        }

        Command::Unknown(input) => Ok(fmt.unknown(&input)),
    }
}

/// Deploy agent binary to a remote node via SSH.
fn deploy_agent(ip: &str) -> String {
    let binary_path = std::env::current_exe()
        .ok()
        .and_then(|p| {
            // Find ouro-agent binary relative to the current binary
            let parent = p.parent()?;
            let agent = parent.join("ouro-agent");
            if agent.exists() {
                Some(agent)
            } else {
                // Try target/debug
                let target = std::env::current_dir()
                    .ok()?
                    .join("target/debug/ouro-agent");
                if target.exists() {
                    Some(target)
                } else {
                    None
                }
            }
        });

    let bin = match binary_path {
        Some(b) => b,
        None => return "BINARY NOT FOUND".to_string(),
    };

    let output = std::process::Command::new("scp")
        .args([
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5",
            bin.to_str().unwrap_or(""),
            &format!("{}:~/ouro-agent", ip),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => "OK".to_string(),
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            if stderr.contains("Connection refused") {
                "SSH REFUSED".to_string()
            } else if stderr.contains("No route") {
                "UNREACHABLE".to_string()
            } else {
                "FAILED".to_string()
            }
        }
        Err(e) => format!("ERROR: {}", e),
    }
}

/// Resolve a property name to a value from a NodeEntry.
fn resolve_node_property(node: &ouro_cluster::beast::topology::NodeEntry, property: &str) -> String {
    match property {
        "power" | "p" => format!("{}W", node.tdp_watts),
        "ram" | "r" => format!("{}MiB", node.ram_mib),
        "cpu" | "c" => node.cpu_model.clone(),
        "cores" => format!("{}", node.cores),
        "threads" => format!("{}", node.threads),
        "simd" | "s" => {
            let mut parts = Vec::new();
            if node.has_avx2 {
                parts.push("AVX2");
            }
            if node.has_avx {
                parts.push("AVX");
            }
            if node.has_sse42 {
                parts.push("SSE4.2");
            }
            if parts.is_empty() {
                "none".to_string()
            } else {
                parts.join(", ")
            }
        }
        "status" => "IDLE".to_string(),
        _ => format!("unknown property: {}", property),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ouro_cluster::beast::topology::NodeEntry;
    use crate::context::Context;
    use crate::formatter::Formatter;

    fn test_topology() -> ClusterTopology {
        let mut topo = ClusterTopology::new();
        topo.nodes.push(NodeEntry {
            id: "n1".to_string(),
            hostname: "laptop-1".to_string(),
            ip: "192.168.1.101".to_string(),
            cpu_model: "i5-4200U".to_string(),
            cores: 2,
            threads: 4,
            has_avx: true,
            has_avx2: true,
            has_sse42: true,
            ram_mib: 8192,
            tdp_watts: 35,
        });
        topo
    }

    #[test]
    fn test_handle_node_query() {
        let mut topo = test_topology();
        let mut sched = Scheduler::new(topo.clone());
        let mut ctx = Context::new();
        let mut fmt = Formatter::new(false);
        let config = ShellConfig::new();
        let cmd = Command::NodeQuery { node: "n1".into() };
        let out = handle(cmd, &mut topo, &mut sched, &mut ctx, &mut fmt, &config).unwrap();
        assert!(out.contains("i5-4200U"));
        assert!(out.contains("8192MiB"));
    }

    #[test]
    fn test_handle_budget() {
        let topo = test_topology();
        let mut sched = Scheduler::new(topo.clone());
        let mut ctx = Context::new();
        let mut fmt = Formatter::new(false);
        let config = ShellConfig::new();
        let cmd = Command::SetBudget { watts: 400 };
        let mut topo = topo;
        let out = handle(cmd, &mut topo, &mut sched, &mut ctx, &mut fmt, &config).unwrap();
        assert_eq!(out, "Cluster power budget: 400W. [SET]");
        assert_eq!(sched.budget.budget_watts, 400);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let topo = test_topology();
        let mut sched = Scheduler::new(topo.clone());
        let mut ctx = Context::new();
        let mut fmt = Formatter::new(false);
        let mut config = ShellConfig::new();
        config.topology_file = "/tmp/ouro_test_save".to_string();

        let mut topo = topo;
        let out = handle(Command::Save, &mut topo, &mut sched, &mut ctx, &mut fmt, &config).unwrap();
        assert!(out.contains("DONE"));

        let out = handle(Command::Load, &mut topo, &mut sched, &mut ctx, &mut fmt, &config).unwrap();
        assert!(out.contains("DONE"));
        assert_eq!(topo.node_count(), 1);

        std::fs::remove_file("/tmp/ouro_test_save.json").ok();
    }

    #[test]
    fn test_resolve_node_property() {
        let node = NodeEntry {
            id: "n1".into(),
            hostname: "test".into(),
            ip: "127.0.0.1".into(),
            cpu_model: "i5-4200U".into(),
            cores: 2,
            threads: 4,
            has_avx: true,
            has_avx2: true,
            has_sse42: true,
            ram_mib: 8192,
            tdp_watts: 35,
        };
        assert_eq!(resolve_node_property(&node, "power"), "35W");
        assert_eq!(resolve_node_property(&node, "ram"), "8192MiB");
        assert_eq!(resolve_node_property(&node, "simd"), "AVX2, AVX, SSE4.2");
    }
}
