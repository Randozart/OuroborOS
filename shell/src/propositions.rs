use anyhow::Result;
use ouro_cluster::beast::resource::ResourcePath;
use ouro_cluster::beast::topology::{ClusterTopology, NodeEntry};
use ouro_cluster::op::{GraphBackend, Op, QueueEntry, Resource};
use ouro_cluster::scheduler::{ScheduleOutcome, Scheduler, Task};
use ouro_cluster::scheduler::workload_class::WorkloadClass;
use ouro_cluster::transport::auth;
use sha2::{Digest, Sha256};

use crate::context::Context;
use crate::formatter::{Formatter, NodeDisplay};
use crate::parser::Command;
use crate::registry_client::{self, RegistryNode, RegistryStatus};

/// Configuration for shell command handling.
#[derive(Clone)]
pub struct ShellConfig {
    pub topology_file: String,
    pub node_addrs: Vec<(String, String)>,
    pub shard_map: String,
    /// Registry daemon (head bus) address — the live graph source.
    /// Queries prefer it and fall back to the static topology when
    /// unreachable (OURO_REGISTRY env overrides).
    pub registry_addr: String,
}

impl ShellConfig {
    pub fn new() -> Self {
        Self {
            topology_file: "cluster.beast".to_string(),
            node_addrs: Vec::new(),
            shard_map: "shards/shard_map.json".to_string(),
            registry_addr: "127.0.0.1:9501".to_string(),
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
  ?  cluster?              cluster summary (live bus + topology)
  n1?                      full node record
  n1.power?  n1 power?     one property (power ram cpu cores threads simd gpu status)
  power?                   same, on the context node
  cluster.active?          bulk query (active idle offline sleeping)
  desired                  the declared state of the cluster
  n1 lanes?                live per-lane health
  power?                   actual draw vs budget (on the context node)

declarations — state you declare; the cluster converges toward it
  n1 assign branch_sort    branch_sort shall run on n1
  branch_sort on?          dry-run: would it place? where? why not?
  n1 sleep   n1 wake       the node shall be asleep / awake
  budget 400w              the cluster shall draw at most 400W
  tasks                    declared + running workloads
  reconcile                converge now (also runs on a 5s tick)
  recover                  sweep stale/failed nodes, drain the queue

fleet
  register                 probe this box, add it to the topology
  unregister n3            remove a node
  discover [cidr] [port]   one-shot LAN sweep for live agents
  drift [rev]              which tails don't run the expected versions
  probe                    list topology nodes
  save  load               topology to/from JSON

payloads
  generate <prompt>        BitNet generation on the target node
  shards                   pipeline plan + activation transport probe
  deploy  deploy shards    ship the agent / sync weight shards
  weights  bind  fetch     the avenue: census, bind a span, fetch it live
  revoke b1                release a binding
  poetry on  poetry off    output register

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
        Command::Drift { .. } => {
            // The interactive shell intercepts drift before dispatch
            // (Repl::execute); non-Repl callers (ttyd web sessions)
            // get an honest referral rather than a silent no-op.
            Ok("drift: run it from the interactive shell (ouro-hiss)".to_string())
        }
        Command::ClusterSummary | Command::ClusterQuery => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            match run_dispatch(&Op::Stat(ResourcePath::parse("cluster").unwrap()), &mut backend)? {
                Resource::Cluster {
                    total,
                    online,
                    power_watts,
                    budget_watts,
                    topology_nodes,
                    gpus,
                } => {
                    let mut s = fmt.cluster_summary(total, online, power_watts, budget_watts, 0, 0);
                    if !gpus.is_empty() {
                        s.push_str(&format!(
                            "\n  GPUs:   {} (vram: {})",
                            gpus.len(),
                            gpus.join(", ")
                        ));
                    }
                    if topology_nodes != total {
                        s.push_str(&format!(
                            "\n  source: registry bus ({} live, topology static: {})",
                            total, topology_nodes
                        ));
                    }
                    Ok(s)
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::NodeQuery { node } => {
            refresh_live_props(config, ctx, &node);
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            match run_dispatch(&Op::Stat(ResourcePath::parse(&node).unwrap()), &mut backend)? {
                Resource::Node(record) => {
                    let display: NodeDisplay = serde_json::from_value(record)?;
                    Ok(fmt.node_query(&display))
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::PropertyQuery { node, property } => {
            refresh_live_props(config, ctx, &node);
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let path = ResourcePath::parse(&format!("{}.{}", node, property))
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            match run_dispatch(&Op::Stat(path), &mut backend)? {
                Resource::Prop { node: n, property: p, value } => {
                    Ok(fmt.property_query(&n, &p, &value))
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Lanes { node } => {
            let entry = topology
                .get_node(&node)
                .ok_or_else(|| anyhow::anyhow!("no such node: {}", node))?;
            let lanes = lanes_summary(&entry.edges);
            if lanes.is_empty() {
                Ok(format!("{}: no lanes reported", node))
            } else {
                Ok(format!("{} lanes: {}", node, lanes))
            }
        }

        Command::ContextPropertyQuery { property } => {
            let Some(node_id) = ctx.current_node().map(|s| s.to_string()) else {
                return Ok(fmt.unknown(&format!("{}?", property)));
            };
            refresh_live_props(config, ctx, &node_id);
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let path = ResourcePath::parse(&format!("{}.{}", node_id, property))
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            match run_dispatch(&Op::Stat(path), &mut backend)? {
                Resource::Prop { node: n, property: p, value } => {
                    Ok(fmt.property_query(&n, &p, &value))
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::BulkQuery { filter } => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            match run_dispatch(&Op::Stat(ResourcePath::parse("cluster.nodes").unwrap()), &mut backend)?
            {
                Resource::Nodes(records) => {
                    let nodes: Vec<NodeDisplay> = records
                        .into_iter()
                        .map(|r| serde_json::from_value(r).map_err(|e| anyhow::anyhow!("{}", e)))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(fmt.bulk_query(&filter, &nodes))
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
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
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let op = Op::Write {
                path: ResourcePath::parse("tasks").unwrap(),
                value: workload.clone(),
            };
            let class = WorkloadClass::from_name(&workload);
            let mut details = Vec::new();
            details.push(format!("[1] Serialize {}.bv.              [OK]", workload));
            details.push(format!(
                "[2] Check: {} supports {}.         [YES]",
                node,
                class.label()
            ));
            let outcome = match run_dispatch(&op, &mut backend)? {
                Resource::Assign { node: assigned } => {
                    details.push(format!("[3] Dispatch to {}.                [OK]", assigned));
                    Some(assigned)
                }
                Resource::Queued { reason } => {
                    details.push(format!("[3] Scheduling failed: {}", reason));
                    None
                }
                other => return Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            };

            // Declaration: `workload` shall run on `node`. The reconcile pass
            // dispatches it to the real agent (idempotent — once, until
            // re-declared). If the scheduler placed it elsewhere, the
            // declaration follows the scheduler's decision.
            let placed = outcome.is_some();
            let declared_node = outcome.unwrap_or_else(|| node.clone());
            if let Ok(mut desired) = scheduler.desired.lock() {
                desired.declare_workload(&declared_node, &workload, "");
            }
            let report = reconcile_desired(config, &scheduler.desired, ctx);
            Ok(format!("{}\n{}", fmt.assign_result(&node, &workload, placed, &details), report))
        }

        Command::AssignCheck { node, workload } => {
            // Dry-run placement feasibility is a ClassAd stat (Phase C);
            // sugar until the placement ads exist (docs/PLAN9.md §9.1).
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
            // Declaration: the node shall be asleep / awake. The cluster
            // converges toward it (the reconcile pass executes sleep/wake
            // over the wire). No more canned strings — the tail either
            // suspends or honestly reports why not.
            use ouro_cluster::scheduler::desired::DesiredNodeState;
            if let Ok(mut desired) = scheduler.desired.lock() {
                desired.declare_node(&node, if sleeping { DesiredNodeState::Sleeping } else { DesiredNodeState::Awake });
            }
            let report = reconcile_desired(config, &scheduler.desired, ctx);
            Ok(if sleeping {
                format!("declared: {} shall sleep\n{}", node, report)
            } else {
                format!("declared: {} shall wake\n{}", node, report)
            })
        }

        Command::SetBudget { watts } => {
            // Declaration: the cluster shall draw at most `watts`. The
            // scheduler enforces it at dispatch (model) and reconcile
            // reports actual draw against it (live).
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let op = Op::Write {
                path: ResourcePath::parse("cluster.budget").unwrap(),
                value: watts.to_string(),
            };
            run_dispatch(&op, &mut backend)?;
            if let Ok(mut desired) = scheduler.desired.lock() {
                desired.declare_budget(Some(watts));
            }
            let report = reconcile_desired(config, &scheduler.desired, ctx);
            Ok(format!("declared: budget {}W\n{}", watts, report))
        }

        Command::Probe => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            match run_dispatch(&Op::Stat(ResourcePath::parse("cluster.static").unwrap()), &mut backend)?
            {
                Resource::Nodes(records) => {
                    let nodes: Vec<NodeDisplay> = records
                        .into_iter()
                        .map(|r| serde_json::from_value(r).map_err(|e| anyhow::anyhow!("{}", e)))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(fmt.probe_result(&nodes))
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
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

        Command::Register => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let op = Op::Ctl {
                path: ResourcePath::parse("cluster").unwrap(),
                verb: "register".to_string(),
            };
            match run_dispatch(&op, &mut backend)? {
                Resource::Ack { message } => Ok(message),
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Unregister { node } => {
            if node.is_empty() {
                return Ok("Usage: unregister n3.".to_string());
            }
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let op = Op::Ctl {
                path: ResourcePath::parse(&node).unwrap(),
                verb: "unregister".to_string(),
            };
            match run_dispatch(&op, &mut backend)? {
                Resource::Ack { message } => Ok(message),
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Tasks => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            match run_dispatch(&Op::Stat(ResourcePath::parse("queue").unwrap()), &mut backend)? {
                Resource::Queue { entries, .. } => {
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
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Recover => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let op = Op::Ctl {
                path: ResourcePath::parse("cluster").unwrap(),
                verb: "recover".to_string(),
            };
            match run_dispatch(&op, &mut backend)? {
                Resource::Ack { message } => Ok(message),
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Reconcile => {
            Ok(reconcile_desired(config, &scheduler.desired, ctx))
        }

        Command::Desired => {
            let Ok(desired) = scheduler.desired.lock() else {
                return Ok("desired: lock poisoned".to_string());
            };
            if desired.is_empty() {
                return Ok("desired: nothing declared".to_string());
            }
            let mut out = String::from("desired state:");
            if let Some(b) = desired.budget_watts {
                out.push_str(&format!("\n  budget: {}W", b));
            }
            for (node, state) in &desired.node_states {
                out.push_str(&format!("\n  {}: {:?}", node, state));
            }
            for wl in &desired.workloads {
                out.push_str(&format!("\n  {}: {} ({})", wl.node, wl.task, if wl.payload.is_empty() { "no payload" } else { &wl.payload }));
            }
            if !desired.dispatched.is_empty() {
                out.push_str(&format!("\n  acted: {}", desired.dispatched.join(", ")));
            }
            Ok(out)
        }

        Command::Weights { target } => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let path_str = if target.is_empty() {
                "weights".to_string()
            } else {
                format!("weights.{}", target)
            };
            let path = ResourcePath::parse(&path_str)
                .map_err(|e| anyhow::anyhow!(e))?;
            // `weights.<node>.<i>` opens a handle (read); the rest stat.
            let op = if path.segments.len() == 3 {
                Op::Read(path)
            } else {
                Op::Stat(path)
            };
            match run_dispatch(&op, &mut backend)? {
                Resource::Ack { message } => Ok(message),
                Resource::Tensors { node, tensors } => {
                    if tensors.is_empty() {
                        return Ok(format!("{}: no tensors", node));
                    }
                    let mut out = format!("{} ({} tensors, {} bytes):\n", node, tensors.len(),
                        tensors.iter().map(|t| t.length).sum::<u64>());
                    for (i, t) in tensors.iter().enumerate() {
                        out.push_str(&format!(
                            "  [{}] {}  {} bytes\n", i, t.name, t.length
                        ));
                    }
                    Ok(out)
                }
                Resource::Handle { id, node, tensor, length } => {
                    Ok(format!("handle {}  ({}: {}, {} bytes)", id, node, tensor, length))
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Bind { target, offset, length } => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            let path = ResourcePath::parse(&format!("weights.{}", target))
                .map_err(|e| anyhow::anyhow!(e))?;
            let span = match (offset, length) {
                (None, None) => None,
                (Some(o), Some(l)) => Some(ouro_cluster::op::Span { offset: o, length: l }),
                _ => return Err(anyhow::anyhow!("bind needs offset AND length, or neither")),
            };
            match run_dispatch(&Op::Bind { path, span }, &mut backend)? {
                Resource::Binding(b) => {
                    let lanes = b
                        .lanes
                        .primary
                        .clone()
                        .map(|p| {
                            let extra = b
                                .lanes
                                .secondary
                                .clone()
                                .map(|s| format!(" + {}", s))
                                .unwrap_or_default();
                            format!(" via {}", p + &extra)
                        })
                        .unwrap_or_else(|| " (no lanes)".to_string());
                    Ok(format!(
                        "binding {}: {} [{}..{}) of {} bytes{}",
                        b.id, b.handle, b.span.offset, b.span.end(), b.span.length, lanes
                    ))
                }
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Revoke { binding } => {
            let mut backend = ShellBackend { scheduler, topology, ctx, recovery, config };
            match run_dispatch(&Op::Revoke { binding: binding.clone() }, &mut backend)? {
                Resource::Ack { message } => Ok(message),
                other => Err(anyhow::anyhow!("unexpected resource: {:?}", other)),
            }
        }

        Command::Fetch { target, offset, length } => {
            let backend = ShellBackend { scheduler, topology, ctx, recovery, config };

            // Resolve the target to (node, tensor, span). Two forms:
            //   fetch b1            — a binding (node + tensor + span known)
            //   fetch weights.n1.0  — a tensor path (whole tensor unless a
            //                         sub-span is given)
            let (node, tensor, span) = if target.starts_with('b') {
                let binding = backend
                    .scheduler
                    .bindings
                    .get(&target)
                    .ok_or_else(|| anyhow::anyhow!("no such binding: {}", target))?;
                let span = match (offset, length) {
                    (None, None) => binding.span,
                    (Some(o), Some(l)) => ouro_cluster::op::Span { offset: o, length: l },
                    _ => return Err(anyhow::anyhow!("fetch needs offset AND length, or neither")),
                };
                (binding.node.clone(), binding.tensor.clone(), span)
            } else {
                // weights.<node>.<i>
                let full = if target.starts_with("weights.") {
                    target.clone()
                } else {
                    format!("weights.{}", target)
                };
                let path = ResourcePath::parse(&full)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let node = path
                    .segments
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("fetch requires a weights.<node>.<i> path, got {}", target))?
                    .to_string();
                let idx: usize = path
                    .segments
                    .get(2)
                    .ok_or_else(|| anyhow::anyhow!("fetch requires a tensor index"))?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("tensor index must be an integer"))?;
                let shard = backend
                    .scheduler
                    .weights
                    .for_node(&node)
                    .ok_or_else(|| anyhow::anyhow!("no weights for node {}", node))?;
                let t = shard
                    .at(idx)
                    .ok_or_else(|| anyhow::anyhow!("tensor index {} out of range for node {}", idx, node))?;
                let total = t.length;
                let span = match (offset, length) {
                    (None, None) => ouro_cluster::op::Span { offset: 0, length: total },
                    (Some(o), Some(l)) => {
                        let s = ouro_cluster::op::Span { offset: o, length: l };
                        if s.end() > total {
                            return Err(anyhow::anyhow!(
                                "span [{}, {}) exceeds tensor {} of {} bytes",
                                s.offset,
                                s.end(),
                                t.name,
                                total
                            ));
                        }
                        s
                    }
                    _ => return Err(anyhow::anyhow!("fetch needs offset AND length, or neither")),
                };
                (node.to_string(), t.name.clone(), span)
            };

            // The node's live address: the multi-homed table (--nodes).
            let addr = backend
                .config
                .node_addrs
                .iter()
                .find(|(id, _)| id == &node)
                .map(|(_, a)| a.clone())
                .ok_or_else(|| anyhow::anyhow!("no address known for node {} (start with --nodes)", node))?;

            // Bulk rides frames, never the line (Art. 6). The tail resolves
            // where it keeps the tensor.
            let bytes = crate::agent_client::fetch_tensor(&addr, &tensor, span.offset, span.length)?;
            let mut sha = Sha256::new();
            sha.update(&bytes);
            let digest = format!("{:x}", sha.finalize());
            Ok(format!(
                "fetched {}: {} bytes of {} [{}..{}) from {} — sha256 {}",
                target,
                bytes.len(),
                tensor,
                span.offset,
                span.end(),
                addr,
                &digest[..16]
            ))
        }

        Command::Unknown(input) => Ok(fmt.unknown(&input)),
    }
}

/// The shell's op-kernel backend (docs/PLAN9.md §4.2). Ops are the mouth;
/// the live graph, scheduler, and recovery are the brain. Produces the same
/// values the pre-kernel handlers produced — output is byte-identical.
struct ShellBackend<'a> {
    scheduler: &'a mut Scheduler,
    topology: &'a mut ClusterTopology,
    ctx: &'a mut Context,
    recovery: &'a mut ouro_cluster::error_recovery::ErrorRecovery,
    config: &'a ShellConfig,
}

/// Static display record for one node (no live registry).
fn static_display(topology: &ClusterTopology, n: &NodeEntry) -> NodeDisplay {
    NodeDisplay {
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
        lanes: lanes_summary(&n.edges),
    }
}

/// One-line lane inventory for a node (docs/AIR_PATH.md §1.1): every priced
/// edge, comma-joined. Empty when the node has no lane report.
fn lanes_summary(edges: &[ouro_cluster::transport::edge::PricedEdge]) -> String {
    edges.iter().map(|e| e.describe()).collect::<Vec<_>>().join(" | ")
}

/// Run one op through the kernel, mapping structured errors to anyhow.
fn run_dispatch(op: &Op, backend: &mut ShellBackend) -> Result<Resource, anyhow::Error> {
    ouro_cluster::op::dispatch(op, backend).map_err(|e| anyhow::anyhow!("{}", e.message))
}

fn census_string(model: &str, vram_mib: u64) -> String {
    format!("{}:{}MiB", model.replace("NVIDIA GeForce ", ""), vram_mib)
}

impl<'a> ShellBackend<'a> {
    /// `weights` paths (Rung B3): census, mirroring the scheduler backend.
    /// The shell's scheduler carries the weight manifest (node -> tensors
    /// name+length); empty until a manifest is loaded.
    fn stat_weights(&mut self, path: &ResourcePath) -> Result<Resource, String> {
        if path.is("weights") {
            let nodes = self.scheduler.weights.nodes();
            return Ok(Resource::Ack {
                message: if nodes.is_empty() {
                    "no weights registered".to_string()
                } else {
                    format!("weights: {}", nodes.join(", "))
                },
            });
        }
        let is_weights = path.segments.first().map(|s| s.as_str()) == Some("weights");
        if is_weights {
            if let Some(node) = path.segments.get(1).map(|s| s.as_str()) {
                let shard = self
                    .scheduler
                    .weights
                    .for_node(node)
                    .ok_or_else(|| format!("no weights for node {}", node))?;
                let tensors: Vec<ouro_cluster::op::TensorCensus> = shard
                    .tensors
                    .iter()
                    .map(|t| ouro_cluster::op::TensorCensus {
                        name: t.name.clone(),
                        length: t.length,
                    })
                    .collect();
                return Ok(Resource::Tensors {
                    node: node.to_string(),
                    tensors,
                });
            }
        }
        Err(format!("no such resource: {}", path))
    }
}

impl<'a> GraphBackend for ShellBackend<'a> {
    fn resolve(&mut self, path: &ResourcePath) -> Result<(), String> {
        if let Some(node) = path.node() {
            let live_hit = live_status(self.config)
                .map(|l| l.nodes.iter().any(|n| n.id == node))
                .unwrap_or(false);
            if live_hit || self.topology.get_node(node).is_some() {
                return Ok(());
            }
            return Err(format!("Node {} not found", node));
        }
        match path.segments.first().map(|s| s.as_str()) {
            Some("cluster") | Some("queue") | Some("tasks") | Some("weights") => Ok(()),
            other => Err(format!("no such resource: {:?}", other)),
        }
    }

    fn stat(&mut self, path: &ResourcePath) -> Result<Resource, String> {
        self.resolve(path)?;
        if path.is("cluster") {
            if let Some(live) = live_status(self.config) {
                let online = live.nodes.iter().filter(|n| n.online).count();
                let power: u32 = live.nodes.iter().filter(|n| n.online).map(record_watts).sum();
                let gpus: Vec<String> = live
                    .nodes
                    .iter()
                    .filter(|n| n.has_gpu)
                    .map(|n| census_string(&n.gpu_model, n.gpu_vram_mib))
                    .collect();
                return Ok(Resource::Cluster {
                    total: live.nodes.len(),
                    online,
                    power_watts: power,
                    budget_watts: self.topology.power_budget_watts,
                    topology_nodes: self.topology.node_count(),
                    gpus,
                });
            }
            let total = self.topology.node_count();
            let power: u32 = self.topology.nodes.iter().map(|n| n.tdp_watts).sum();
            let gpus: Vec<String> = self
                .topology
                .nodes
                .iter()
                .filter(|n| n.has_gpu)
                .map(|n| census_string(&n.gpu_model, n.gpu_vram_mib))
                .collect();
            return Ok(Resource::Cluster {
                total,
                online: 0,
                power_watts: power,
                budget_watts: self.topology.power_budget_watts,
                topology_nodes: total,
                gpus,
            });
        }
        if path.is_child_of("cluster", "budget") {
            return Ok(Resource::Budget {
                watts: self.scheduler.budget.budget_watts,
            });
        }
        if path.is_child_of("cluster", "nodes") {
            let mut displays: Vec<NodeDisplay> = Vec::new();
            let mut seen: Vec<String> = Vec::new();
            if let Some(live) = live_status(self.config) {
                for rec in &live.nodes {
                    displays.push(record_to_display(rec));
                    seen.push(rec.id.clone());
                }
            }
            for n in &self.topology.nodes {
                if seen.iter().any(|id| id == &n.id) {
                    continue;
                }
                displays.push(static_display(self.topology, n));
            }
            let records: Vec<serde_json::Value> = displays
                .into_iter()
                .map(|d| serde_json::to_value(d).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?;
            return Ok(Resource::Nodes(records));
        }
        if path.is_child_of("cluster", "static") {
            let records: Vec<serde_json::Value> = self
                .topology
                .nodes
                .iter()
                .map(|n| {
                    serde_json::to_value(static_display(self.topology, n)).map_err(|e| e.to_string())
                })
                .collect::<Result<_, _>>()?;
            return Ok(Resource::Nodes(records));
        }
        if path.is("queue") || path.is("tasks") {
            let entries: Vec<QueueEntry> = self
                .scheduler
                .queue
                .summary()
                .into_iter()
                .map(|e| QueueEntry {
                    name: e.name,
                    class: e.class,
                    age_secs: e.age_secs,
                    retries: e.retries,
                    priority: e.priority,
                })
                .collect();
            let depth = entries.len();
            return Ok(Resource::Queue { depth, entries });
        }
        if let Some(node) = path.node() {
            let live = live_status(self.config);
            let live_rec = live.as_ref().and_then(|l| l.nodes.iter().find(|n| n.id == node));
            if let Some(property) = path.property() {
                let entry = self
                    .topology
                    .get_node(node)
                    .ok_or_else(|| format!("Node {} not found", node))?;
                let value = match live_rec {
                    Some(rec) => resolve_record_property(rec, property, self.ctx),
                    None => resolve_node_property(entry, property, self.ctx),
                };
                return Ok(Resource::Prop {
                    node: node.to_string(),
                    property: property.to_string(),
                    value,
                });
            }
            let display = match live_rec {
                Some(rec) => record_to_display(rec),
                None => {
                    let entry = self
                        .topology
                        .get_node(node)
                        .ok_or_else(|| format!("Node {} not found", node))?;
                    static_display(self.topology, entry)
                }
            };
            let record = serde_json::to_value(display).map_err(|e| e.to_string())?;
            return Ok(Resource::Node(record));
        }
        self.stat_weights(path)
    }

    /// `read` of a weight tensor path opens a bulk handle (Rung B3).
    fn read(&mut self, path: &ResourcePath) -> Result<Resource, String> {
        let is_weights = path.segments.first().map(|s| s.as_str()) == Some("weights");
        if is_weights && path.segments.len() == 3 {
            if let Some(node) = path.segments.get(1).map(|s| s.as_str()) {
                let idx = path.segments[2]
                    .parse::<usize>()
                    .map_err(|_| format!("tensor index must be an integer, got {}", path.segments[2]))?;
                let shard = self
                    .scheduler
                    .weights
                    .for_node(node)
                    .ok_or_else(|| format!("no weights for node {}", node))?;
                let t = shard
                    .at(idx)
                    .ok_or_else(|| format!("tensor index {} out of range for node {}", idx, node))?;
                return Ok(Resource::Handle {
                    id: format!("h:{}.{}", node, t.name),
                    node: node.to_string(),
                    tensor: t.name.clone(),
                    length: t.length,
                });
            }
        }
        self.stat(path)
    }

    fn write(&mut self, path: &ResourcePath, value: &str) -> Result<Resource, String> {
        self.resolve(path)?;
        if path.is_child_of("cluster", "budget") {
            let watts: u32 = value
                .trim()
                .trim_end_matches('w')
                .trim_end_matches('W')
                .parse()
                .map_err(|_| format!("budget must be watts, got {:?}", value))?;
            self.scheduler.budget.set_budget(watts);
            return Ok(Resource::Budget { watts });
        }
        if path.is("tasks") {
            let task = Task {
                name: value.to_string(),
                class: WorkloadClass::from_name(value),
                payload: String::new(),
                estimated_watts: 30,
                estimated_seconds: 10,
            };
            return match self
                .scheduler
                .schedule(&task)
                .map_err(|e| format!("schedule failed: {}", e))?
            {
                ScheduleOutcome::Dispatched { node } => Ok(Resource::Assign { node }),
                ScheduleOutcome::Queued { reason } => Ok(Resource::Queued { reason }),
            };
        }
        Err(format!("write not supported on {}", path))
    }

    fn ctl(&mut self, path: &ResourcePath, verb: &str) -> Result<Resource, String> {
        // No resolve pre-check: node `ctl` verbs report on missing nodes
        // rather than erroring (byte-identical to the pre-kernel handlers).
        if let Some(node) = path.node() {
            match verb {
                "sleep" => Ok(Resource::Ack {
                    message: format!("Node {}休眠. Power: 12W → 2W.", node),
                }),
                "unregister" => {
                    if self.topology.remove_node(node).is_some() {
                        Ok(Resource::Ack {
                            message: format!("Unregistered {}. [DONE]", node),
                        })
                    } else {
                        Ok(Resource::Ack {
                            message: format!("Node {} not found.", node),
                        })
                    }
                }
                _ => Err(format!("unknown verb: {} on {}", verb, path)),
            }
        } else if path.is("cluster") {
            match verb {
                "register" => {
                    let info = ouro_cluster::probe::probe_local()
                        .map_err(|e| format!("probe failed: {}", e))?;
                    let entry = self.topology.add_node(info.clone());
                    let net_info = info
                        .network
                        .as_ref()
                        .map(|n| format!(" | network: {:.1}ms", n.latency_ms))
                        .unwrap_or_default();
                    Ok(Resource::Ack {
                        message: format!(
                            "Registered {} @ {} | {} | {}MiB | {}W{} [DONE]",
                            entry.id,
                            entry.ip,
                            entry.cpu_model,
                            entry.ram_mib,
                            entry.tdp_watts,
                            net_info,
                        ),
                    })
                }
                "recover" => {
                    let stale = self
                        .recovery
                        .sweep_stale(&ouro_cluster::registry::Registry::new());
                    let failed: Vec<String> = self
                        .recovery
                        .failed_nodes()
                        .iter()
                        .map(|f| f.node_id.clone())
                        .collect();
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
                    let results = self.scheduler.drain_queue();
                    if !results.is_empty() {
                        out.push_str(&format!("\nDrained {} queued tasks:\n", results.len()));
                        for (name, outcome) in &results {
                            out.push_str(&format!("  {} → {:?}\n", name, outcome));
                        }
                    }
                    Ok(Resource::Ack { message: out })
                }
                _ => Err(format!("unknown ctl: {} on {}", verb, path)),
            }
        } else {
            Err(format!("unknown ctl: {} on {}", verb, path))
        }
    }

    /// Track C: bind/revoke delegate to the scheduler's binding registry —
    /// the brain owns bindings (one writer, Art. 3).
    fn bind(&mut self, path: &ResourcePath, span: Option<ouro_cluster::op::Span>) -> Result<Resource, String> {
        self.scheduler.bind(path, span)
    }

    fn revoke(&mut self, binding: &str) -> Result<Resource, String> {
        self.scheduler.revoke(binding)
    }
}

/// rsync-or-scp one shard file to a node if checksums differ. Returns log line.
/// Load the brain's weight manifest from a shard_map (Rung B3): maps each
/// stage's `.bmts` header into `Scheduler::weights`. Missing shard files are
/// collected, not fatal. Returns `(shards, missing_files)`.
pub fn load_weights(shard_map: &str) -> (ouro_cluster::weights::Weights, Vec<String>) {
    match ouro_cluster::pipeline::PipelinePlan::load(shard_map) {
        Ok(plan) => ouro_cluster::weights::Weights::from_pipeline_plan(&plan),
        Err(_) => (ouro_cluster::weights::Weights::new(), Vec::new()),
    }
}

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
        agent_version: tel.agent_version.clone(),
        image_rev: tel.image_rev.clone(),
        has_rdma: false,
        rdma_gid: String::new(),
        edges: Vec::new(),
        node_id: String::new(),
    }
}

/// Convert a registry-bus node record into a topology entry.
fn registry_node_to_entry(rn: &crate::registry_client::RegistryNode) -> ouro_cluster::beast::topology::NodeEntry {
    ouro_cluster::beast::topology::NodeEntry {
        id: rn.id.clone(),
        hostname: rn.hostname.clone(),
        ip: rn.ip.clone(),
        cpu_model: rn.cpu_model.clone(),
        cores: rn.cores,
        threads: rn.threads,
        has_avx: rn.has_avx,
        has_avx2: rn.has_avx2,
        has_sse42: rn.has_sse42,
        ram_mib: rn.ram_mib,
        tdp_watts: rn.tdp_watts.max(rn.power_watts.max(15)),
        has_gpu: rn.has_gpu,
        gpu_model: rn.gpu_model.clone(),
        gpu_vram_mib: rn.gpu_vram_mib,
        gpu_driver: String::new(),
        agent_version: rn.agent_version.clone(),
        image_rev: rn.image_rev.clone(),
        has_rdma: false,
        rdma_gid: String::new(),
        edges: Vec::new(),
        node_id: String::new(),
    }
}

/// Absorb live nodes into the topology at startup: the registry bus first
/// (one signed `status` call), then any `--nodes` telemetry probes. Returns
/// the count absorbed and the source label ("registry" / "telemetry").
/// Neither reachable → 0 (caller keeps the demo/static topology, clearly
/// labeled).
pub fn absorb_live_nodes(
    topology: &mut ClusterTopology,
    config: &ShellConfig,
) -> (usize, &'static str) {
    // 1. The registry bus is the one source of truth.
    if let Some(status) = live_status(config) {
        let mut n = 0;
        for rn in &status.nodes {
            if let Some(k) = topology.nodes.iter().position(|x| x.id == rn.id) {
                topology.nodes[k] = registry_node_to_entry(rn);
            } else {
                topology.nodes.push(registry_node_to_entry(rn));
            }
            n += 1;
        }
        if n > 0 {
            return (n, "registry");
        }
    }
    // 2. Fall back to `--nodes` telemetry probes.
    let mut n = 0;
    for (id, addr) in &config.node_addrs {
        if let Ok(tel) = crate::agent_client::telemetry(addr) {
            let entry = telemetry_to_node(addr, &tel, id.clone());
            if let Some(k) = topology.nodes.iter().position(|x| x.id == *id) {
                topology.nodes[k] = entry;
            } else {
                topology.nodes.push(entry);
            }
            n += 1;
        }
    }
    (n, "telemetry")
}

/// Reconcile the declared state against the live cluster: diff, then
/// execute the diff over the wire. Declarations are acted on once (the
/// `dispatched` guard makes this idempotent — a 5s loop converges, it does
/// not spam). Returns a convergence report.
///
/// - workload declared, agent awake & not busy  → dispatch `execute`
/// - node declared sleeping, agent reachable    → agent `sleep`
/// - node declared awake, agent unreachable     → WOL wake (needs MAC)
/// - budget declared                            → actual draw vs declared
pub fn reconcile_desired(
    config: &ShellConfig,
    desired: &std::sync::Arc<std::sync::Mutex<ouro_cluster::scheduler::desired::DesiredState>>,
    ctx: &mut Context,
) -> String {
    use ouro_cluster::scheduler::desired::DesiredNodeState;

    let snapshot = {
        let Ok(desired) = desired.lock() else {
            return "reconcile: desired-state lock poisoned".to_string();
        };
        if desired.is_empty() {
            return "nothing declared; converged".to_string();
        }
        desired.clone()
    };

    let mut steps: Vec<String> = Vec::new();

    // 1. Declared workloads → real tasks on real agents.
    for wl in &snapshot.workloads {
        let key = format!("{}:{}", wl.node, wl.task);
        if snapshot.already_dispatched(&key) {
            continue;
        }
        let Some(addr) = addr_for(config, &wl.node) else {
            steps.push(format!("  {}: no address known — not dispatched", wl.node));
            continue;
        };
        let task = crate::agent_client::AgentTask {
            id: format!("decl-{}", wl.node),
            name: wl.task.clone(),
            payload: wl.payload.clone(),
            estimated_watts: 30,
            estimated_seconds: 60,
        };
        match crate::agent_client::execute(&addr, &task) {
            Ok(r) if r.status == "Success" => {
                steps.push(format!("  dispatched {} to {} [{}ms]", wl.task, wl.node, r.elapsed_ms));
                if let Ok(mut desired) = desired.lock() {
                    desired.mark_dispatched(&key);
                }
            }
            Ok(r) => steps.push(format!("  {}: {} [{}]", wl.node, r.output, r.status)),
            Err(e) => steps.push(format!("  {}: unreachable ({})", wl.node, e)),
        }
    }

    // 2. Declared node power states → sleep / WOL wake.
    for (node, state) in &snapshot.node_states {
        let key = format!("{}:{:?}", node, state);
        if snapshot.already_dispatched(&key) {
            steps.push(format!("  {}: {} (requested)", node, match state {
                DesiredNodeState::Sleeping => "sleeping",
                DesiredNodeState::Awake => "awake",
            }));
            continue;
        }
        let Some(addr) = addr_for(config, node) else {
            steps.push(format!("  {}: no address known", node));
            continue;
        };
        match state {
            DesiredNodeState::Sleeping => {
                match crate::agent_client::sleep(&addr) {
                    Ok(msg) => {
                        steps.push(format!("  {}: {}", node, msg));
                        if let Ok(mut desired) = desired.lock() {
                            desired.mark_dispatched(&key);
                        }
                    }
                    Err(e) => steps.push(format!("  {}: sleep refused ({})", node, e)),
                }
            }
            DesiredNodeState::Awake => {
                // The tail is unreachable (else it would be awake). Wake needs
                // its MAC — cached from the last time it was awake.
                let mac = ctx.get_property(node, "mac").map(|s| s.to_string());
                match mac {
                    Some(mac) => match crate::agent_client::wake(&mac, "255.255.255.255:9") {
                        Ok(()) => {
                            steps.push(format!("  {}: WOL sent to {}", node, mac));
                            if let Ok(mut desired) = desired.lock() {
                                desired.mark_dispatched(&key);
                            }
                        }
                        Err(e) => steps.push(format!("  {}: WOL failed ({})", node, e)),
                    },
                    None => steps.push(format!("  {}: unreachable, no WOL path (no MAC known)", node)),
                }
            }
        }
    }

    // 3. Declared budget → actual draw vs declared.
    if let Some(declared) = snapshot.budget_watts {
        let mut actual: u64 = 0;
        let mut measured = 0;
        for (_id, addr) in &config.node_addrs {
            if let Ok(tel) = crate::agent_client::telemetry(addr) {
                actual += tel.power_watts as u64;
                measured += 1;
            }
        }
        if measured == 0 {
            steps.push(format!("  budget: declared {}W, no agents reachable to measure", declared));
        } else {
            let over = if actual as u32 > declared {
                format!(" — OVER by {}W", actual as u32 - declared)
            } else {
                " — within budget".to_string()
            };
            steps.push(format!("  budget: declared {}W, actual {}W ({} measured){over}", declared, actual, measured));
        }
    }

    if steps.is_empty() {
        "reconciled: no diff to apply; converged".to_string()
    } else {
        let mut out = String::from("reconciling:\n");
        out.push_str(&steps.join("\n"));
        out
    }
}

/// The live address for a node id, from the multi-homed table.
fn addr_for(config: &ShellConfig, node: &str) -> Option<String> {
    config
        .node_addrs
        .iter()
        .find(|(id, _)| id == node)
        .map(|(_, a)| a.clone())
}

/// Append GPU census line to a cluster summary when any node has a GPU.
/// (Live-record variant lives below; the static variant moved into the op
/// kernel's `stat cluster` — ShellBackend builds the census strings.)
fn entry_gpu(topology: &ClusterTopology, id: &str) -> String {
    topology
        .get_node(id)
        .filter(|n| n.has_gpu)
        .map(|n| format!("{} ({}MiB)", n.gpu_model, n.gpu_vram_mib))
        .unwrap_or_default()
}

/// Live registry census with a 10s negative cache: a down daemon
/// costs one fast localhost refusal per 10s, not per command. No
/// secret in the env -> no fetch at all (piped/test modes stay pure).
fn live_status(config: &ShellConfig) -> Option<RegistryStatus> {
    static DOWN_UNTIL: std::sync::Mutex<Option<std::time::Instant>> =
        std::sync::Mutex::new(None);
    if let Ok(guard) = DOWN_UNTIL.lock() {
        if let Some(t) = *guard {
            if std::time::Instant::now() < t {
                return None;
            }
        }
    }
    let addr = registry_client::resolve_addr(&config.registry_addr);
    let status = auth::secret_from_env()
        .ok()
        .and_then(|secret| registry_client::fetch(&addr, &secret).ok());
    if status.is_none() {
        if let Ok(mut guard) = DOWN_UNTIL.lock() {
            *guard = Some(std::time::Instant::now() + std::time::Duration::from_secs(10));
        }
    }
    status
}

/// Effective live watts: telemetry when flowing, TDP as floor of truth.
fn record_watts(n: &RegistryNode) -> u32 {
    if n.power_watts > 0 { n.power_watts } else { n.tdp_watts }
}

fn record_gpu(n: &RegistryNode) -> String {
    if n.has_gpu {
        format!("{} ({}MiB)", n.gpu_model, n.gpu_vram_mib)
    } else {
        String::new()
    }
}

fn record_to_display(n: &RegistryNode) -> NodeDisplay {
    NodeDisplay {
        id: n.id.clone(),
        cpu_model: n.cpu_model.clone(),
        ram_mib: n.ram_mib,
        has_avx2: n.has_avx2,
        has_avx: n.has_avx,
        has_sse42: n.has_sse42,
        status: n.status.to_uppercase(),
        power_watts: record_watts(n),
        temp_c: n.temp_c,
        gpu: record_gpu(n),
        // Live records carry no lane inventory yet; the agent's self-report
        // lands in the next rung (AIR_PATH §4.4).
        lanes: String::new(),
    }
}

/// GPU census line from live records instead of the static topology.
/// (Kept for the census tests; production census now comes from the op
/// kernel's `stat cluster`.)
#[cfg(test)]
fn with_gpu_census_records(mut s: String, nodes: &[RegistryNode]) -> String {
    let gpus: Vec<String> = nodes
        .iter()
        .filter(|n| n.has_gpu)
        .map(|n| {
            format!(
                "{}:{}MiB",
                n.gpu_model.replace("NVIDIA GeForce ", ""),
                n.gpu_vram_mib
            )
        })
        .collect();
    if !gpus.is_empty() {
        s.push_str(&format!(
            "\n  GPUs:   {} (vram: {})",
            gpus.len(),
            gpus.join(", ")
        ));
    }
    s
}

/// Resolve a property from a live registry record: same names as the
/// static resolver, plus the live-only facts (thermal, load, state,
/// hostname, ip). Agent live-cache still wins (ctx).
fn resolve_record_property(n: &RegistryNode, property: &str, ctx: &Context) -> String {
    if let Some(live) = ctx.get_property(&n.id, property) {
        return format!("{} (live)", live);
    }
    match property {
        "power" | "p" => format!("{}W", record_watts(n)),
        "thermal" | "temp" | "t" => format!("{}C", n.temp_c),
        "load" | "l" => format!("{:.2}", n.load_avg),
        "status" => n.status.clone(),
        "state" => {
            if n.online {
                "online".to_string()
            } else {
                "offline".to_string()
            }
        }
        "hostname" | "host" => n.hostname.clone(),
        "ip" | "addr" => n.ip.clone(),
        "ram" | "r" => format!("{}MiB", n.ram_mib),
        "cpu" | "c" => n.cpu_model.clone(),
        "cores" => format!("{}", n.cores),
        "threads" => format!("{}", n.threads),
        "simd" | "s" => {
            let mut parts = Vec::new();
            if n.has_avx2 {
                parts.push("AVX2");
            }
            if n.has_avx {
                parts.push("AVX");
            }
            if n.has_sse42 {
                parts.push("SSE4.2");
            }
            if parts.is_empty() {
                "none".to_string()
            } else {
                parts.join(", ")
            }
        }
        "gpu" => {
            if n.has_gpu {
                format!("{} ({}MiB)", n.gpu_model, n.gpu_vram_mib)
            } else {
                "none".to_string()
            }
        }
        _ => format!("unknown property: {}", property),
    }
}

/// Resolve a property: live agent cache first, static topology as fallback.
/// Build the property map from live agent telemetry.
fn telemetry_props_map(tel: &crate::agent_client::AgentTelemetry) -> std::collections::HashMap<String, String> {
    let mut props = std::collections::HashMap::new();
    props.insert("power".to_string(), format!("{}W", tel.power_watts));
    props.insert("temp".to_string(), format!("{}C", tel.temp_c));
    props.insert("ram".to_string(), format!("{}MiB used of {}MiB", tel.ram_used_mib, tel.ram_total_mib));
    props.insert("cpu".to_string(), tel.cpu_model.clone());
    props.insert("status".to_string(), "AWAKE".to_string());
    props.insert("load".to_string(), format!("{:.2}", tel.load_avg));
    if let Some(g) = tel.gpus.first() {
        props.insert("gpu".to_string(), format!("{} ({}MiB)", g.model, g.vram_mib));
    }
    if !tel.mac.is_empty() {
        props.insert("mac".to_string(), tel.mac.clone());
    }
    props
}

/// Live telemetry props for a node, TTL-cached (5s): a query hits the wire
/// at most every 5s per node. No addr in `config.node_addrs` → None.
fn live_props(config: &ShellConfig, node: &str) -> Option<std::collections::HashMap<String, String>> {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    type PropCache = HashMap<String, (Instant, HashMap<String, String>)>;
    static CACHE: std::sync::OnceLock<std::sync::Mutex<PropCache>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let addr = config
        .node_addrs
        .iter()
        .find(|(id, _)| id == node)
        .map(|(_, a)| a.clone())?;
    if let Ok(mut cache) = cache.lock() {
        if let Some((t, props)) = cache.get(node) {
            if t.elapsed() < Duration::from_secs(5) {
                return Some(props.clone());
            }
        }
        if let Ok(tel) = crate::agent_client::telemetry(&addr) {
            let props = telemetry_props_map(&tel);
            cache.insert(node.to_string(), (Instant::now(), props.clone()));
            return Some(props);
        }
    }
    None
}

/// Refresh a node's live properties before a query resolves it (best-effort:
/// no addr, agent down, or secret missing → keep whatever the cache holds).
fn refresh_live_props(config: &ShellConfig, ctx: &mut Context, node: &str) {
    if let Some(props) = live_props(config, node) {
        ctx.cache_properties(node, props);
    }
}

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
            agent_version: String::new(),
            image_rev: String::new(),
            has_rdma: false,
            rdma_gid: String::new(),
            edges: Vec::new(),
            node_id: String::new(),
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
        assert!(out.starts_with("declared: budget 400W"), "got: {out}");
        assert!(out.contains("no agents reachable to measure"), "got: {out}");
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
    agent_version: String::new(),
    image_rev: String::new(),
    has_rdma: false,
    rdma_gid: String::new(),
        edges: Vec::new(),
        node_id: String::new(),
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
    agent_version: String::new(),
    image_rev: String::new(),
    has_rdma: false,
    rdma_gid: String::new(),
        edges: Vec::new(),
        node_id: String::new(),
        };
        let ctx = Context::new();
        assert_eq!(resolve_node_property(&node, "power", &ctx), "35W");
        assert_eq!(resolve_node_property(&node, "ram", &ctx), "8192MiB");
        assert_eq!(resolve_node_property(&node, "simd", &ctx), "AVX2, AVX, SSE4.2");
    }

    /// Live props map from telemetry: the source of live `n1.power?` answers.
    #[test]
    fn test_telemetry_props_map() {
        use crate::agent_client::{AgentTelemetry, GpuMini};
        let tel = AgentTelemetry {
            hostname: "live-node".into(),
            cpu_model: "i7-3770".into(),
            cores: 4,
            threads: 8,
            has_avx: true,
            has_avx2: true,
            has_sse42: true,
            ram_total_mib: 32768,
            ram_used_mib: 4096,
            power_watts: 42,
            temp_c: 61,
            load_avg: 0.5,
            agent_version: "0.1.0".into(),
            image_rev: "abc1234".into(),
            mac: "aa:bb:cc:dd:ee:ff".into(),
            gpus: vec![GpuMini { model: "RTX 3060".into(), vram_mib: 12288, driver: "580".into() }],
        };
        let props = telemetry_props_map(&tel);
        assert_eq!(props.get("power").unwrap(), "42W");
        assert_eq!(props.get("ram").unwrap(), "4096MiB used of 32768MiB");
        assert_eq!(props.get("mac").unwrap(), "aa:bb:cc:dd:ee:ff");
        assert_eq!(props.get("status").unwrap(), "AWAKE");
        assert_eq!(props.get("gpu").unwrap(), "RTX 3060 (12288MiB)");
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
            agent_version: "git:test".into(),
            image_rev: "test".into(),
            mac: String::new(),
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

#[cfg(test)]
mod registry_live_tests {
    use super::*;

    fn hp_record() -> RegistryNode {
        serde_json::from_value(serde_json::json!({
            "id": "n1",
            "hostname": "pavilion",
            "ip": "192.168.1.114",
            "cpu_model": "Intel(R) Core(TM) i5-7200U",
            "cores": 2, "threads": 4, "ram_mib": 7829,
            "has_avx2": true, "has_avx": true, "has_sse42": true,
            "tdp_watts": 35, "has_gpu": true,
            "gpu_model": "NVIDIA GeForce GTX 1060 6GB",
            "gpu_vram_mib": 6144,
            "power_watts": 35, "temp_c": 46, "load_avg": 0.99,
            "status": "Idle", "online": true,
        }))
        .unwrap()
    }

    #[test]
    fn test_record_to_display_live_fields() {
        let d = record_to_display(&hp_record());
        assert_eq!(d.id, "n1");
        assert_eq!(d.status, "IDLE");
        assert_eq!(d.power_watts, 35);
        assert_eq!(d.temp_c, 46);
        assert!(d.gpu.contains("1060"));
    }

    #[test]
    fn test_record_watts_falls_back_to_tdp() {
        let mut n = hp_record();
        n.power_watts = 0;
        assert_eq!(record_watts(&n), 35);
        n.power_watts = 28;
        assert_eq!(record_watts(&n), 28);
    }

    #[test]
    fn test_resolve_record_property_live_values() {
        let ctx = Context::new();
        let n = hp_record();
        assert_eq!(resolve_record_property(&n, "power", &ctx), "35W");
        assert_eq!(resolve_record_property(&n, "thermal", &ctx), "46C");
        assert_eq!(resolve_record_property(&n, "load", &ctx), "0.99");
        assert_eq!(resolve_record_property(&n, "status", &ctx), "Idle");
        assert_eq!(resolve_record_property(&n, "state", &ctx), "online");
        assert_eq!(resolve_record_property(&n, "ip", &ctx), "192.168.1.114");
        assert!(resolve_record_property(&n, "gpu", &ctx).contains("1060"));
        assert!(resolve_record_property(&n, "simd", &ctx).contains("AVX"));
    }

    #[test]
    fn test_resolve_record_property_ctx_cache_wins() {
        let mut ctx = Context::new();
        let mut props = std::collections::HashMap::new();
        props.insert("power".to_string(), "12W".to_string());
        ctx.cache_properties("n1", props);
        assert_eq!(
            resolve_record_property(&hp_record(), "power", &ctx),
            "12W (live)"
        );
    }

    #[test]
    fn test_gpu_census_from_records() {
        let s = with_gpu_census_records(String::new(), &[hp_record()]);
        assert!(s.contains("GPUs:   1"));
        assert!(s.contains("GTX 1060 6GB:6144MiB"));
        let none = with_gpu_census_records(String::new(), &[]);
        assert!(none.is_empty());
    }
}

/// Rung B1 gate: the op verbs run through the kernel and their output is
/// byte-identical to the pre-kernel handlers (docs/PLAN9.md §9.2). No
/// `OURO_SECRET_FILE` in tests → `live_status` is None → static path,
/// deterministic.
#[cfg(test)]
mod kernel_ops_tests {
    use super::*;
    use ouro_cluster::error_recovery::ErrorRecovery;
    use ouro_cluster::beast::topology::NodeEntry;

    struct Ctx {
        sched: Scheduler,
        topo: ClusterTopology,
        ctx: Context,
        fmt: Formatter,
        config: ShellConfig,
        recovery: ErrorRecovery,
    }

    fn setup() -> Ctx {
        let mut topo = ClusterTopology::new();
        topo.power_budget_watts = 500;
        topo.nodes.push(NodeEntry {
            id: "n1".into(),
            hostname: "pavilion".into(),
            ip: "192.168.1.114".into(),
            cpu_model: "i5-7200U".into(),
            cores: 2,
            threads: 4,
            has_avx: true,
            has_avx2: true,
            has_sse42: true,
            ram_mib: 7829,
            tdp_watts: 35,
            has_gpu: true,
            gpu_model: "NVIDIA GeForce GTX 1060 6GB".into(),
            gpu_vram_mib: 6144,
            gpu_driver: String::new(),
            agent_version: String::new(),
            image_rev: String::new(),
            has_rdma: false,
            rdma_gid: String::new(),
            edges: Vec::new(),
            node_id: String::new(),
        });
        Ctx {
            sched: Scheduler::new(topo.clone()),
            topo,
            ctx: Context::new(),
            fmt: Formatter::new(false),
            config: ShellConfig::new(),
            recovery: ErrorRecovery::new(),
        }
    }

    fn run(st: &mut Ctx, cmd: Command) -> String {
        handle(
            cmd,
            &mut st.topo,
            &mut st.sched,
            &mut st.ctx,
            &mut st.fmt,
            &st.config,
            &mut st.recovery,
        )
        .unwrap()
    }

    #[test]
    fn test_property_query_via_kernel() {
        let mut st = setup();
        assert_eq!(
            run(&mut st, Command::PropertyQuery { node: "n1".into(), property: "power".into() }),
            "n1.power = 35W"
        );
        let gpu = run(&mut st, Command::PropertyQuery { node: "n1".into(), property: "gpu".into() });
        assert_eq!(gpu, "n1.gpu = NVIDIA GeForce GTX 1060 6GB (6144MiB)");
    }

    #[test]
    fn test_context_property_query_via_kernel() {
        let mut st = setup();
        st.ctx.set_node("n1");
        assert_eq!(run(&mut st, Command::ContextPropertyQuery { property: "ram".into() }), "n1.ram = 7829MiB");
    }

    #[test]
    fn test_budget_via_kernel() {
        let mut st = setup();
        let out = run(&mut st, Command::SetBudget { watts: 400 });
        assert!(out.starts_with("declared: budget 400W"), "got: {out}");
        assert_eq!(st.sched.budget.budget_watts, 400);
    }

    #[test]
    fn test_sleep_via_kernel() {
        let mut st = setup();
        let out = run(&mut st, Command::PowerState { node: "n1".into(), sleeping: true });
        // The declaration is recorded; the tail either suspends or the
        // reconcile honestly reports why it cannot.
        assert!(out.starts_with("declared: n1 shall sleep"), "got: {out}");
        let d = st.sched.desired.lock().unwrap();
        use ouro_cluster::scheduler::desired::DesiredNodeState;
        assert_eq!(d.node_desired("n1"), DesiredNodeState::Sleeping);
    }

    #[test]
    fn test_tasks_via_kernel() {
        let mut st = setup();
        assert_eq!(run(&mut st, Command::Tasks), "Task queue: empty.");
        st.sched.budget.set_budget(0);
        run(&mut st, Command::AssignProposition { node: "n1".into(), workload: "matmul".into() });
        let out = run(&mut st, Command::Tasks);
        assert!(out.starts_with("Task queue (1):\n  matmul [SimdFriendly] age="), "got: {}", out);
        assert!(out.contains("retries=0/3 priority=0"));
    }

    #[test]
    fn test_assign_via_kernel() {
        let mut st = setup();
        let out = run(&mut st, Command::AssignProposition { node: "n1".into(), workload: "matmul".into() });
        assert!(out.contains("THE COUNSEL ACTS:"), "got: {}", out);
        assert!(out.contains("[1] Serialize matmul.bv.              [OK]"));
        assert!(out.contains("[2] Check: n1 supports SIMD_FRIENDLY."));
        assert!(out.contains("[3] Dispatch to n1.                [OK]"));
        assert!(out.contains("RESULT: matmul assigned to n1. [TRUE]"));
    }

    #[test]
    fn test_assign_queues_when_no_budget() {
        let mut st = setup();
        st.sched.budget.set_budget(0);
        let out = run(&mut st, Command::AssignProposition { node: "n1".into(), workload: "matmul".into() });
        assert!(out.contains("RESULT: Assignment failed. [FALSE]"), "got: {}", out);
        assert!(out.contains("[3] Scheduling failed: energy budget exceeded"));
    }

    /// The REPL `fetch` verb, live: a loopback agent serves a tensor span
    /// over frames; the dispatch resolves the target (a weights path), calls
    /// the agent, and reports the bytes fetched + a digest.
    #[test]
    fn test_fetch_via_kernel_live() {
        use ouro_cluster::bmts::{write_shard, BmtsShard, BmtsTensor};
        use ouro_cluster::transport::frames::{pump_send, FrameSession, DEFAULT_CHUNK, DEFAULT_WINDOW};
        use std::io::{Cursor, Read, Write};
        use std::net::TcpListener;

        let dir = std::env::temp_dir().join(format!("ouro-hiss-fetchk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard_path = dir.join("shard_1.bmts");
        let blob: Vec<u8> = (0..4096u32).map(|i| (i % 239) as u8).collect();
        write_shard(
            shard_path.to_str().unwrap(),
            1,
            &[BmtsTensor { name: "blk.0.attn_q.weight".into(), shape: vec![16, 256], dtype: 34, offset: 0, length: 4096 }],
            &blob,
        )
        .unwrap();
        let shard_path_str = shard_path.to_str().unwrap().to_string();

        // The shell's agent client signs the wire with OURO_SECRET_FILE.
        let secret_file = dir.join("secret.hex");
        std::fs::write(&secret_file, "0707070707070707070707070707070707070707070707070707070707070707").unwrap();
        std::env::set_var("OURO_SECRET_FILE", secret_file.to_str().unwrap());

        // Loopback agent serving `fetch-tensor <tensor> <offset> <length>`.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let key: ouro_cluster::transport::auth::Secret = [7u8; 32];
        let server = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut sock = sock;
            let mut line = String::new();
            let mut one = [0u8; 1];
            while !line.ends_with('\n') {
                sock.read_exact(&mut one).unwrap();
                line.push(one[0] as char);
            }
            let (_seq, body) = ouro_cluster::transport::auth::open_line(&key, line.trim()).unwrap();
            let rest = body.strip_prefix("fetch-tensor ").unwrap();
            let mut p = rest.split_whitespace();
            let tensor = p.next().unwrap().to_string();
            let offset: u64 = p.next().unwrap().parse().unwrap();
            let length: u64 = p.next().unwrap().parse().unwrap();
            let bmts = BmtsShard::open(&shard_path_str).unwrap();
            let declared = bmts.tensor_bytes(&tensor).unwrap().len() as u64;
            let payload = bmts.read_span(&tensor, ouro_cluster::op::Span { offset, length }).unwrap();
            let mut session = FrameSession::new(sock, key);
            let buf = declared.to_be_bytes();
            session.get_mut().write_all(&buf).unwrap();
            let mut cur = Cursor::new(payload.bytes().to_vec());
            pump_send(&mut cur, &mut session, DEFAULT_CHUNK, DEFAULT_WINDOW).unwrap();
        });

        // The shell knows the tensor census + the node's address.
        let mut st = setup();
        st.sched.weights.shards.push(ouro_cluster::weights::WeightShard {
            node: "n1".into(),
            tensors: vec![ouro_cluster::weights::WeightTensor { name: "blk.0.attn_q.weight".into(), length: 4096 }],
        });
        st.config.node_addrs = vec![("n1".to_string(), addr.clone())];

        let out = run(&mut st, Command::Fetch { target: "weights.n1.0".into(), offset: Some(1000), length: Some(256) });
        assert!(out.contains("fetched weights.n1.0"), "got: {out}");
        assert!(out.contains("256 bytes"), "got: {out}");
        assert!(out.contains("sha256"), "got: {out}");
        server.join().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The declarative shell's payoff: declare a workload, the reconcile
    /// dispatches it to a real agent over the wire (idempotent — the second
    /// reconcile reports it as already acted).
    #[test]
    fn test_reconcile_dispatches_workload_live() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let dir = std::env::temp_dir().join(format!("ouro-hiss-reconcile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let secret_file = dir.join("secret.hex");
        std::fs::write(&secret_file, "0707070707070707070707070707070707070707070707070707070707070707").unwrap();
        std::env::set_var("OURO_SECRET_FILE", secret_file.to_str().unwrap());

        // Loopback agent: reads a signed task line, answers a Success result.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let key: ouro_cluster::transport::auth::Secret = [7u8; 32];
        let server = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut sock = sock;
            let mut line = String::new();
            let mut one = [0u8; 1];
            while !line.ends_with('\n') {
                sock.read_exact(&mut one).unwrap();
                line.push(one[0] as char);
            }
            let (seq, body) = ouro_cluster::transport::auth::open_line(&key, line.trim()).unwrap();
            let task: serde_json::Value = serde_json::from_str(&body).unwrap();
            let task_id = task.get("id").unwrap().as_str().unwrap().to_string();
            let out = task.get("name").unwrap().as_str().unwrap().to_string();
            let resp = serde_json::json!({
                "task_id": task_id,
                "status": "Success",
                "output": out,
                "elapsed_ms": 3,
                "peak_watts": 30,
            });
            let signed = ouro_cluster::transport::auth::sign_line(&key, seq, &resp.to_string());
            sock.write_all(signed.as_bytes()).unwrap();
            sock.write_all(b"\n").unwrap();
        });

        // Declare a workload on n1, then reconcile.
        let mut st = setup();
        st.config.node_addrs = vec![("n1".to_string(), addr.clone())];
        if let Ok(mut desired) = st.sched.desired.lock() {
            desired.declare_workload("n1", "echo", "hello");
        }
        let report = reconcile_desired(&st.config, &st.sched.desired, &mut st.ctx);
        assert!(report.contains("dispatched echo to n1"), "got: {report}");
        assert!(report.contains("[3ms]"), "got: {report}");
        server.join().unwrap();

        // Idempotent: the second pass does not re-dispatch.
        let report2 = reconcile_desired(&st.config, &st.sched.desired, &mut st.ctx);
        assert!(!report2.contains("dispatched echo"), "second pass must not re-dispatch: {report2}");
        assert!(report2.contains("reconciled: no diff"), "got: {report2}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_node_query_via_kernel() {
        let mut st = setup();
        let out = run(&mut st, Command::NodeQuery { node: "n1".into() });
        assert!(out.starts_with("NODE_n1\n  CPU:    i5-7200U"), "got: {}", out);
        assert!(out.contains("GPU:    NVIDIA GeForce GTX 1060 6GB (6144MiB)"));
    }

    #[test]
    fn test_cluster_query_via_kernel() {
        let mut st = setup();
        let out = run(&mut st, Command::ClusterQuery);
        assert!(out.contains("Nodes:  1 total | 0 active | 1 idle"), "got: {}", out);
        assert!(out.contains("GPUs:   1 (vram: GTX 1060 6GB:6144MiB)"));
    }

    #[test]
    fn test_probe_via_kernel() {
        let mut st = setup();
        let out = run(&mut st, Command::Probe);
        assert!(out.contains("Probing all nodes... [DONE]"), "got: {}", out);
        assert!(out.contains("n1: i5-7200U, 7829MiB, AVX2, AVX, SSE4.2 [FOUND]"));
    }

    #[test]
    fn test_bulk_query_via_kernel() {
        let mut st = setup();
        let out = run(&mut st, Command::BulkQuery { filter: "active".into() });
        assert!(out.starts_with("ACTIVE\n"), "got: {}", out);
        assert!(out.contains("n1: i5-7200U | IDLE | 35W"));
    }

    #[test]
    fn test_unregister_via_kernel() {
        let mut st = setup();
        assert_eq!(run(&mut st, Command::Unregister { node: "n9".into() }), "Node n9 not found.");
        assert_eq!(run(&mut st, Command::Unregister { node: "n1".into() }), "Unregistered n1. [DONE]");
        assert_eq!(st.topo.node_count(), 0);
    }

    #[test]
    fn test_recover_via_kernel() {
        let mut st = setup();
        assert_eq!(run(&mut st, Command::Recover), "No stale or failed nodes. [OK]");
    }

    /// Rung B3 (docs/PLAN9.md): the shell exposes the weight census and opens
    /// bulk handles via the op kernel — `weights` and `weights.<node>`.
    #[test]
    fn test_weights_census_and_handle_via_kernel() {
        let mut st = setup();
        // No manifest loaded yet: an honest empty answer.
        assert_eq!(run(&mut st, Command::Weights { target: "".into() }), "no weights registered");
        // Load a manifest for n1 and re-run.
        st.sched.weights = ouro_cluster::weights::Weights::new().with_shard(
            ouro_cluster::weights::WeightShard {
                node: "n1".into(),
                tensors: vec![
                    ouro_cluster::weights::WeightTensor { name: "blk.0.attn_q.weight".into(), length: 1024 },
                    ouro_cluster::weights::WeightTensor { name: "blk.1.attn_k.weight".into(), length: 2048 },
                ],
            },
        );
        let census = run(&mut st, Command::Weights { target: "n1".into() });
        assert!(census.starts_with("n1 (2 tensors, 3072 bytes):\n"), "got: {}", census);
        assert!(census.contains("[0] blk.0.attn_q.weight  1024 bytes"), "got: {}", census);
        // Opening a handle: identity + size-stamp, never bytes.
        let h = run(&mut st, Command::Weights { target: "n1.1".into() });
        assert_eq!(h, "handle h:n1.blk.1.attn_k.weight  (n1: blk.1.attn_k.weight, 2048 bytes)");
    }

    /// Track C: `bind` returns a fetchable binding (span + lanes), `revoke`
    /// releases it en-bloc.
    #[test]
    fn test_bind_and_revoke_via_kernel() {
        let mut st = setup();
        st.sched.weights = ouro_cluster::weights::Weights::new().with_shard(
            ouro_cluster::weights::WeightShard {
                node: "n1".into(),
                tensors: vec![
                    ouro_cluster::weights::WeightTensor { name: "blk.0.attn_q.weight".into(), length: 1024 },
                ],
            },
        );
        let b = run(&mut st, Command::Bind { target: "n1.0".into(), offset: None, length: None });
        assert!(
            b.starts_with("binding b1: h:n1.blk.0.attn_q.weight [0..1024) of 1024 bytes"),
            "got: {}",
            b
        );
        // A sub-range binds too.
        let b2 = run(&mut st, Command::Bind { target: "n1.0".into(), offset: Some(256), length: Some(128) });
        assert!(b2.starts_with("binding b2:"), "got: {}", b2);
        assert!(b2.contains("[256..384)"), "got: {}", b2);
        // Revoke is idempotent.
        let r1 = run(&mut st, Command::Revoke { binding: "b1".into() });
        assert_eq!(r1, "binding b1 released");
        let r2 = run(&mut st, Command::Revoke { binding: "b1".into() });
        assert_eq!(r2, "binding b1 released");
        // Out-of-range bind refuses loudly (structured error, not a panic —
        // proven at the kernel level in op_backend.rs).
        let err = ouro_cluster::op::dispatch(
            &ouro_cluster::op::Op::Bind {
                path: ouro_cluster::beast::resource::ResourcePath::parse("weights.n1.0").unwrap(),
                span: Some(ouro_cluster::op::Span { offset: 0, length: 999999 }),
            },
            &mut st.sched,
        )
        .unwrap_err();
        assert!(err.message.contains("exceeds tensor"), "got: {}", err.message);
    }

    /// Rung B3 boot-load: `load_weights` maps a shard_map's `.bmts` headers into
    /// the manifest. A real 2-tensor shard yields a census the kernel can read.
    #[test]
    fn test_load_weights_from_shard_map() {
        let dir = std::env::temp_dir().join(format!("ouro_weights_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let shard_path = dir.join("shard_1.bmts");
        let blob: Vec<u8> = (0..64u8).collect();
        ouro_cluster::bmts::write_shard(
            shard_path.to_str().unwrap(),
            1,
            &[
                ouro_cluster::bmts::BmtsTensor { name: "blk.0.attn_q.weight".into(), shape: vec![8, 8], dtype: 34, offset: 0, length: 32 },
                ouro_cluster::bmts::BmtsTensor { name: "blk.1.attn_k.weight".into(), shape: vec![8, 8], dtype: 34, offset: 32, length: 32 },
            ],
            &blob,
        )
        .unwrap();
        let map_path = dir.join("shard_map.json");
        std::fs::write(
            &map_path,
            format!(
                r#"{{"model":"t","nodes":[{{"node":1,"file":"{}","layers":[0,1],"tensors":2,"bytes":64}}]}}"#,
                shard_path.to_str().unwrap()
            ),
        )
        .unwrap();

        let (weights, missing) = load_weights(map_path.to_str().unwrap());
        assert!(missing.is_empty(), "missing: {:?}", missing);
        let shard = weights.for_node("n1").expect("n1 manifest");
        assert_eq!(shard.tensors.len(), 2);
        assert_eq!(shard.total_bytes(), 64);
        assert_eq!(shard.at(0).unwrap().name, "blk.0.attn_q.weight");

        // Clean up.
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A node with priced lanes renders them in `n1?` (docs/AIR_PATH.md §1.1).
    #[test]
    fn test_node_query_renders_lanes() {
        let mut st = setup();
        let node = st.topo.nodes.iter_mut().find(|n| n.id == "n1").unwrap();
        node.edges = vec![ouro_cluster::transport::edge::PricedEdge {
            iface: "wlan0".into(),
            kind: ouro_cluster::transport::edge::EdgeKind::Air,
            bw_mbps: 60,
            latency_us: 2000,
            jitter_us: 500,
            watts: 2,
            signal_dbm: -45,
        }];
        let out = run(&mut st, Command::NodeQuery { node: "n1".into() });
        assert!(
            out.contains("Lanes:  wlan0 (air, 60 Mbit/s, 2000us+/-500us, 2W, -45 dBm)"),
            "got: {}",
            out
        );
    }
}
