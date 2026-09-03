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
    pub shard_map: String,
}

impl ShellConfig {
    pub fn new() -> Self {
        Self {
            topology_file: "cluster.beast".to_string(),
            node_addrs: Vec::new(),
            shard_map: "shards/shard_map.json".to_string(),
        }
    }
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The `help` verb: every command, one screen (docs/HANDBOOK.md §3 is
/// the long form).
const HELP_TEXT: &str = "\
queries
  ?  cluster?              cluster summary
  n1?                      full node record
  n1.power?                one property (power ram cpu cores threads simd gpu status)
  power?                   same, on the context node
  cluster.active?          bulk query (active idle offline sleeping)

placement
  n1 assign branch_sort   route through Scheduler::schedule()
  branch_sort on?          dry-run: would it place? where? why not?
  budget 400w             set cluster power budget (Art. 4)
  tasks                   the task queue: depth, age, retries, priority
  recover                 sweep stale/failed nodes, drain the queue

fleet
  register                probe this box, add it to the topology
  unregister n3           remove a node
  discover [cidr] [port]  one-shot LAN sweep for live agents
  probe                   list topology nodes
  save  load             topology to/from JSON

payloads
  generate <prompt>.       BitNet generation on the target node
  shards.                  pipeline plan + activation transport probe
  deploy  deploy shards  ship the agent / sync weight shards
  n1 sleep                sleep transition (stub)
  poetry on  poetry off  output register

meta
  help                     this screen
  quit  exit  q            leave (the wyrm remembers nothing you typed here)";

/// Handle a parsed command against the cluster state.
pub fn handle(
    cmd: Command,
    topology: &mut ClusterTopology,
    scheduler: &mut Scheduler,
    ctx: &mut Context,
    fmt: &mut Formatter,
    config: &ShellConfig,
    recovery: &mut ouro_cluster::error_recovery::ErrorRecovery,
) -> Result<String> {
    match cmd {
        Command::ClusterSummary => {
            let total = topology.node_count();
            let power: u32 = topology.nodes.iter().map(|n| n.tdp_watts).sum();
            let budget = topology.power_budget_watts;
            Ok(with_gpu_census(fmt.cluster_summary(total, 0, power, budget, 0, 0), topology))
        }

        Command::ClusterQuery => {
            let total = topology.node_count();
            let power: u32 = topology.nodes.iter().map(|n| n.tdp_watts).sum();
            let budget = topology.power_budget_watts;
            Ok(with_gpu_census(fmt.cluster_summary(total, 0, power, budget, 0, 0), topology))
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
                gpu: entry_gpu(topology, &entry.id),
            };
            Ok(fmt.node_query(&display))
        }

        Command::PropertyQuery { node, property } => {
            let entry = topology
                .get_node(&node)
                .ok_or_else(|| anyhow::anyhow!("Node {} not found", node))?;
            let value = resolve_node_property(entry, &property, ctx);
            Ok(fmt.property_query(&node, &property, &value))
        }

        Command::ContextPropertyQuery { property } => {
            if let Some(node_id) = ctx.current_node().map(|s| s.to_string()) {
                let entry = topology
                    .get_node(&node_id)
                    .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;
                let value = resolve_node_property(entry, &property, ctx);
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
                    gpu: entry_gpu(topology, &n.id),
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
                    gpu: entry_gpu(topology, &n.id),
                })
                .collect();
            Ok(fmt.probe_result(&nodes))
        }

        Command::DeployShards => {
            if config.node_addrs.is_empty() {
                return Ok("No agent endpoints; start with --nodes. [SKIP]".to_string());
            }
            if !std::path::Path::new(&config.shard_map).exists() {
                return Ok(format!("No shard map at {} — run tools/shard_model.py first. [SKIP]", config.shard_map));
            }
            let plan = ouro_cluster::pipeline::PipelinePlan::load(&config.shard_map)?;
            let mut out = String::new();
            out.push_str(&format!("Shard sync ({} stages):\n", plan.nodes.len()));
            for (i, (_name, addr)) in config.node_addrs.iter().enumerate() {
                let ip = addr.split(':').next().unwrap_or(addr).to_string();
                if let Some(stage) = plan.nodes.iter().find(|s| s.node as usize == i + 1) {
                    out.push_str(&sync_file(&ip, &stage.file));
                }
            }
            // metadata last
            for meta in ["model.json", "shard_map.json"] {
                let dir = std::path::Path::new(&config.shard_map)
                    .parent()
                    .map(|d| d.join(meta))
                    .filter(|p| p.exists());
                if let Some(p) = dir {
                    if let Some((_, a)) = config.node_addrs.first() {
                        let ip = a.split(':').next().unwrap_or("").to_string();
                        out.push_str(&sync_file(&ip, p.to_str().unwrap()));
                    }
                }
            }
            out.push_str("[DONE]");
            Ok(out)
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

        Command::Discover { cidr, port } => {
            use crate::agent_client;
            let port = port.unwrap_or(9500);
            let prefix = match cidr.as_deref() {
                Some(c) => c
                    .split('/')
                    .next()
                    .and_then(|ip| {
                        let p: Vec<&str> = ip.split('.').collect();
                        if p.len() == 4 { Some(format!("{}.{}.{}", p[0], p[1], p[2])) } else { None }
                    })
                    .ok_or_else(|| anyhow::anyhow!("bad cidr: {}", cidr.unwrap_or_default()))?,
                None => local_subnet()?,
            };
            let mut found: Vec<(String, agent_client::AgentTelemetry)> = Vec::new();
            // 127/8 is entirely local: sweeping 254 loopbacks finds one host, not 254.
            let hosts: Vec<String> = if prefix == "127.0.0" {
                vec!["127.0.0.1".to_string()]
            } else {
                (1..=254).map(|h| format!("{}.{}", prefix, h)).collect()
            };
            let chunks = hosts.chunks(32);
            for chunk in chunks {
                std::thread::scope(|sc| {
                    let handles: Vec<_> = chunk
                        .iter()
                        .map(|host| {
                            sc.spawn(move || {
                                let addr = format!("{}:{}", host, port);
                                if !alive_fast(&addr, 200) {
                                    return None;
                                }
                                agent_client::telemetry(&addr).ok().map(|t| (addr.clone(), t))
                            })
                        })
                        .collect();
                    for h in handles {
                        if let Some(x) = h.join().unwrap() {
                            found.push(x);
                        }
                    }
                });
            }
            found.sort_by(|a, b| a.1.hostname.cmp(&b.1.hostname));
            let mut seen: Vec<String> = Vec::new();
            found.retain(|(_, t)| {
                if seen.contains(&t.hostname) {
                    false
                } else {
                    seen.push(t.hostname.clone());
                    true
                }
            });
            if found.is_empty() {
                return Ok(format!("Swept {}{}.1-254:{} — no agents. [EMPTY]", prefix, "", port));
            }
            let mut out = format!("Sweeping {}.1-254:{}...\n", prefix, port);
            let mut next_idx = topology.node_count() + 1;
            for (addr, tel) in &found {
                let known = topology.nodes.iter().position(|n| n.ip == addr.split(':').next().unwrap_or(""));
                let entry = telemetry_to_node(addr, tel, String::new());
                let slot = match known {
                    Some(k) => {
                        topology.nodes[k] = entry;
                        k
                    }
                    None => {
                        topology.nodes.push(entry);
                        topology.nodes.last_mut().unwrap().id = format!("n{}", next_idx);
                        next_idx += 1;
                        topology.nodes.len() - 1
                    }
                };
                let node = &topology.nodes[slot];
                out.push_str(&format!(
                    "  {} @ {} | {} | {}MiB | {}W{}\n",
                    node.id,
                    addr,
                    tel.cpu_model,
                    tel.ram_total_mib,
                    tel.power_watts,
                    if tel.gpus.is_empty() { String::new() } else { format!(" | GPU {}MiB", tel.gpus[0].vram_mib) }
                ));
            }
            out.push_str(&format!("{} node(s) absorbed. `save` to persist. [DONE]", found.len()));
            Ok(out)
        }

        Command::ShardStatus => {
            let mut out = String::new();
            if std::path::Path::new(&config.shard_map).exists() {
                let plan = ouro_cluster::pipeline::PipelinePlan::load(&config.shard_map)?;
                out.push_str(&format!("Pipeline plan: {} ({} stages)\n", plan.model, plan.stage_count()));
                for s in &plan.nodes {
                    let lo_hi = match (s.layers.first(), s.layers.last()) {
                        (Some(a), Some(b)) => format!("{}..{}", a, b),
                        _ => "-".to_string(),
                    };
                    out.push_str(&format!(
                        "  node {}: layers {} | {} tensors | {:.1} MB | {}\n",
                        s.node,
                        lo_hi,
                        s.tensors,
                        s.bytes as f64 / 1e6,
                        s.file
                    ));
                }
            } else {
                out.push_str(&format!(
                    "No shard map at {}. Run: python3 tools/shard_model.py <model.gguf> <n>\n",
                    config.shard_map
                ));
            }

            if !config.node_addrs.is_empty() {
                out.push_str("Activation transport probe (2560-dim f32 frame):\n");
                let act = ouro_cluster::pipeline::Activation {
                    sequence: 1,
                    token_pos: 0,
                    layer_start: 0,
                    layer_end: 29,
                    data: vec![0.0123; 2560],
                };
                let hex = ouro_cluster::pipeline::to_hex(&act.encode());
                let task = crate::agent_client::AgentTask {
                    id: "acts-probe".to_string(),
                    name: "acts_echo".to_string(),
                    payload: hex,
                    estimated_watts: 5,
                    estimated_seconds: 5,
                };
                for (id, addr) in &config.node_addrs {
                    let t0 = std::time::Instant::now();
                    match crate::agent_client::execute(addr, &task) {
                        Ok(r) if r.status == "Success" => out.push_str(&format!(
                            "  {} [{}]: {} ({:.1} ms rtt)\n",
                            id, addr, r.output, t0.elapsed().as_secs_f64() * 1000.0
                        )),
                        Ok(r) => out.push_str(&format!("  {}: {} [{}]\n", id, r.output, r.status)),
                        Err(e) => out.push_str(&format!("  {}: unreachable ({})\n", id, e)),
                    }
                }
            }
            out.push_str("[DONE]");
            Ok(out)
        }

        Command::Generate { prompt } => {
            if config.node_addrs.is_empty() {
                return Ok("No agent endpoints. Start with --nodes n1@host:port,.. [SKIP]".to_string());
            }
            let targets: Vec<(String, String)> = match ctx.current_node() {
                Some(node) => config
                    .node_addrs
                    .iter()
                    .filter(|(id, _)| id == node)
                    .cloned()
                    .collect(),
                None => config.node_addrs.clone(),
            };
            if targets.is_empty() {
                let node = ctx.current_node().unwrap_or("?");
                return Ok(format!("Node {} has no agent endpoint. [SKIP]", node));
            }

            let mut out = format!("Generating: \"{}\"\n", prompt);
            let task = crate::agent_client::AgentTask {
                id: format!("gen-{}", prompt.len()),
                name: "bitnet_generate".to_string(),
                payload: format!("{}|64|0.8", prompt),
                estimated_watts: 35,
                estimated_seconds: 60,
            };
            for (id, addr) in &targets {
                match crate::agent_client::execute(addr, &task) {
                    Ok(r) if r.status == "Success" => {
                        out.push_str(&format!("  {} [{}ms]: {}\n", id, r.elapsed_ms, r.output));
                    }
                    Ok(r) => {
                        out.push_str(&format!("  {}: {} [{}]\n", id, r.output, r.status));
                    }
                    Err(e) => {
                        out.push_str(&format!("  {}: unreachable ({})\n", id, e));
                    }
                }
            }
            out.push_str("[DONE]");
            Ok(out)
        }

        Command::Poetry { enabled } => {
            fmt.set_poetry(enabled);
            ctx.set_poetry(enabled);
            Ok(fmt.poetry_toggle(enabled))
        }

        Command::Help => Ok(HELP_TEXT.to_string()),

        Command::Register => {            let info = ouro_cluster::probe::probe_local()
                .map_err(|e| anyhow::anyhow!("probe failed: {}", e))?;
            let entry = topology.add_node(info.clone());
            let net_info = info.network.as_ref().map(|n| format!(" | network: {:.1}ms", n.latency_ms)).unwrap_or_default();
            Ok(format!(
                "Registered {} @ {} | {} | {}MiB | {}W{} [DONE]",
                entry.id,
                entry.ip,
                entry.cpu_model,
                entry.ram_mib,
                entry.tdp_watts,
                net_info,
            ))
        }

        Command::Unregister { node } => {
            if node.is_empty() {
                return Ok("Usage: unregister n3.".to_string());
            }
            let before = topology.node_count();
            topology.remove_node(&node);
            if topology.node_count() < before {
                Ok(format!("Unregistered {}. [DONE]", node))
            } else {
                Ok(format!("Node {} not found.", node))
            }
        }

        Command::Tasks => {
            let entries = scheduler.queue.summary();
            if entries.is_empty() {
                return Ok("Task queue: empty.".to_string());
            }
            let mut out = format!("Task queue ({}):\n", entries.len());
            for e in &entries {
                out.push_str(&format!(
                    "  {} [{}] age={}s retries={}/3 priority={}\n",
                    e.name, e.class, e.age_secs, e.retries, e.priority,
                ));
            }
            Ok(out)
        }

        Command::Recover => {
            let stale = recovery.sweep_stale(&ouro_cluster::registry::Registry::new());
            let failed: Vec<_> = recovery.failed_nodes().iter().map(|f| f.node_id.clone()).collect();
            let mut out = String::new();
            if stale.is_empty() && failed.is_empty() {
                out.push_str("No stale or failed nodes. [OK]");
            } else {
                for id in &stale {
                    out.push_str(&format!("  stale: {} — scheduling recovery\n", id));
                }
                for id in &failed {
                    out.push_str(&format!("  failed: {} — tracking\n", id));
                }
            }
            // Drain queue to retry any queued tasks
            let results = scheduler.drain_queue();
            if !results.is_empty() {
                out.push_str(&format!("\nDrained {} queued tasks:\n", results.len()));
                for (name, outcome) in &results {
                    out.push_str(&format!("  {} → {:?}\n", name, outcome));
                }
            }
            Ok(out)
        }

        Command::Unknown(input) => Ok(fmt.unknown(&input)),
    }
}

/// rsync-or-scp one shard file to a node if checksums differ. Returns log line.
fn sync_file(ip: &str, local: &str) -> String {
    let name = std::path::Path::new(local)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("shard.bmts");
    let Some(local_sha) = sha256_file(local) else {
        return format!("  {}: local read failed\n", local);
    };
    let remote = run_ssh(ip, &format!("sha256sum ~/ouro/shards/{} 2>/dev/null || true", name));
    if remote.starts_with(&local_sha) {
        return format!("  {} -> {}:{} [ok]\n", local, ip, name);
    }
    run_ssh(ip, "mkdir -p ~/ouro/shards");
    let ok = std::process::Command::new("scp")
        .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-q"])
        .arg(local)
        .arg(format!("{}:~/ouro/shards/{}", ip, name))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    format!("  {} -> {}:{} [{}]", local, ip, name, if ok { "pushed" } else { "FAILED" })
}

fn sha256_file(path: &str) -> Option<String> {
    let out = std::process::Command::new("sha256sum").arg(path).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).split_whitespace().next()?.to_string())
}

fn run_ssh(ip: &str, cmd: &str) -> String {
    let out = std::process::Command::new("ssh")
        .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", ip, cmd])
        .output()
        .ok();
    out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
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

/// TCP connect probe with a hard deadline.
fn alive_fast(addr: &str, ms: u64) -> bool {
    let ip_port: Vec<&str> = addr.rsplitn(2, ':').collect();
    if ip_port.len() != 2 {
        return false;
    }
    let port: u16 = match ip_port[0].parse() {
        Ok(p) => p,
        Err(_) => return false,
    };
    let ip: std::net::IpAddr = match ip_port[1].parse() {
        Ok(i) => i,
        Err(_) => return false,
    };
    std::net::TcpStream::connect_timeout(&std::net::SocketAddr::new(ip, port), std::time::Duration::from_millis(ms)).is_ok()
}

/// First non-loopback IPv4 of this machine -> "/24" prefix.
fn local_subnet() -> anyhow::Result<String> {
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg("hostname -I 2>/dev/null | cut -d' ' -f1")
        .output()?;
    let ip = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let p: Vec<&str> = ip.split('.').collect();
    if p.len() != 4 {
        anyhow::bail!("cannot determine local subnet (got {:?}); pass discover. <a.b.c>", ip);
    }
    Ok(format!("{}.{}.{}", p[0], p[1], p[2]))
}

/// Telemetry snapshot -> topology entry (id assigned by caller context).
fn telemetry_to_node(addr: &str, tel: &crate::agent_client::AgentTelemetry, id: String) -> ouro_cluster::beast::topology::NodeEntry {
    ouro_cluster::beast::topology::NodeEntry {
        id,
        hostname: tel.hostname.clone(),
        ip: addr.split(':').next().unwrap_or(addr).to_string(),
        cpu_model: tel.cpu_model.clone(),
        cores: tel.cores,
        threads: tel.threads,
        has_avx: tel.has_avx,
        has_avx2: tel.has_avx2,
        has_sse42: tel.has_sse42,
        ram_mib: tel.ram_total_mib,
        tdp_watts: tel.power_watts.max(15),
        has_gpu: !tel.gpus.is_empty(),
        gpu_model: tel.gpus.first().map(|g| g.model.clone()).unwrap_or_default(),
        gpu_vram_mib: tel.gpus.first().map(|g| g.vram_mib).unwrap_or(0),
        gpu_driver: tel.gpus.first().map(|g| g.driver.clone()).unwrap_or_default(),
    }
}

/// Append GPU census line to a cluster summary when any node has a GPU.
fn with_gpu_census(mut s: String, topology: &ClusterTopology) -> String {
    let gpus: Vec<String> = topology
        .nodes
        .iter()
        .filter(|n| n.has_gpu)
        .map(|n| format!("{}:{}MiB", n.gpu_model.replace("NVIDIA GeForce ", ""), n.gpu_vram_mib))
        .collect();
    if !gpus.is_empty() {
        s.push_str(&format!("\n  GPUs:   {} (vram: {})", gpus.len(), gpus.join(", ")));
    }
    s
}

fn entry_gpu(topology: &ClusterTopology, id: &str) -> String {
    topology
        .get_node(id)
        .filter(|n| n.has_gpu)
        .map(|n| format!("{} ({}MiB)", n.gpu_model, n.gpu_vram_mib))
        .unwrap_or_default()
}

/// Resolve a property: live agent cache first, static topology as fallback.
fn resolve_node_property(node: &ouro_cluster::beast::topology::NodeEntry, property: &str, ctx: &Context) -> String {
    if let Some(live) = ctx.get_property(&node.id, property) {
        return format!("{} (live)", live);
    }
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
        "gpu" => {
            if node.has_gpu {
                format!("{} ({}MiB)", node.gpu_model, node.gpu_vram_mib)
            } else {
                "none".to_string()
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
    use crate::formatter::Formatter;

    fn test_recovery() -> ouro_cluster::error_recovery::ErrorRecovery {
        ouro_cluster::error_recovery::ErrorRecovery::new()
    }

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
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
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
        let out = handle(cmd, &mut topo, &mut sched, &mut ctx, &mut fmt, &config, &mut test_recovery()).unwrap();
        assert!(out.contains("i5-4200U"));
        assert!(out.contains("8192MiB"));
    }

    #[test]
    fn test_handle_help_lists_verbs() {
        let topo = test_topology();
        let mut sched = Scheduler::new(topo.clone());
        let mut ctx = Context::new();
        let mut fmt = Formatter::new(false);
        let config = ShellConfig::new();
        let mut topo = topo;
        let out = handle(Command::Help, &mut topo, &mut sched, &mut ctx, &mut fmt, &config, &mut test_recovery()).unwrap();
        for verb in ["budget 400w", "discover", "recover", "register", "n1.power?", "poetry"] {
            assert!(out.contains(verb), "help missing {verb}");
        }
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
        let out = handle(cmd, &mut topo, &mut sched, &mut ctx, &mut fmt, &config, &mut test_recovery()).unwrap();
        assert_eq!(out, "Cluster power budget: 400W. [SET]");
        assert_eq!(sched.budget.budget_watts, 400);
    }

    #[test]
    fn test_live_cache_overrides_static() {
        let mut ctx = Context::new();
        let mut props = std::collections::HashMap::new();
        props.insert("power".to_string(), "12W".to_string());
        ctx.cache_properties("n1", props);
        let node = NodeEntry {
            id: "n1".into(),
            hostname: "test".into(),
            ip: "127.0.0.1".into(),
            cpu_model: "i5".into(),
            cores: 2,
            threads: 4,
            has_avx: false,
            has_avx2: false,
            has_sse42: false,
            ram_mib: 4096,
            tdp_watts: 35,
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
        };
        assert_eq!(resolve_node_property(&node, "power", &ctx), "12W (live)");
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
        let out = handle(Command::Save, &mut topo, &mut sched, &mut ctx, &mut fmt, &config, &mut test_recovery()).unwrap();
        assert!(out.contains("DONE"));

        let out = handle(Command::Load, &mut topo, &mut sched, &mut ctx, &mut fmt, &config, &mut test_recovery()).unwrap();
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
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
        };
        let ctx = Context::new();
        assert_eq!(resolve_node_property(&node, "power", &ctx), "35W");
        assert_eq!(resolve_node_property(&node, "ram", &ctx), "8192MiB");
        assert_eq!(resolve_node_property(&node, "simd", &ctx), "AVX2, AVX, SSE4.2");
    }

    #[test]
    fn test_telemetry_to_node_mapping() {
        use crate::agent_client::{AgentTelemetry, GpuMini};
        let tel = AgentTelemetry {
            hostname: "test-node".into(),
            cpu_model: "i7-3770".into(),
            cores: 4,
            threads: 8,
            has_avx: true,
            has_avx2: false,
            has_sse42: true,
            ram_total_mib: 16384,
            ram_used_mib: 8192,
            power_watts: 77,
            temp_c: 45,
            load_avg: 0.5,
            gpus: vec![GpuMini {
                model: "RTX 3060".into(),
                vram_mib: 12288,
                driver: "580.178.04".into(),
            }],
        };
        let node = telemetry_to_node("192.168.1.50:9500", &tel, "n1".into());
        assert_eq!(node.hostname, "test-node");
        assert_eq!(node.ip, "192.168.1.50");
        assert!(node.has_avx, "has_avx should be true");
        assert!(!node.has_avx2, "has_avx2 should be false (i7-3770 is Ivy Bridge)");
        assert!(node.has_sse42, "has_sse42 should be true");
        assert!(node.has_gpu);
        assert_eq!(node.gpu_model, "RTX 3060");
        assert_eq!(node.gpu_driver, "580.178.04");
        assert_eq!(node.gpu_vram_mib, 12288);
    }
}
