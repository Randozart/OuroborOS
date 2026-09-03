use std::io::{self, BufRead, IsTerminal};
use std::path::PathBuf;

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

/// Print the HISS banner. Crimson matches the prompt brand; colour
/// drops for NO_COLOR or non-TTY output.
fn banner(color: bool) {
    let mark = if color {
        format!("\x1b[1;31m{HISS_WORDMARK}\x1b[0m")
    } else {
        HISS_WORDMARK.to_string()
    };
    println!("{mark}");
    println!();
    println!("  HISS — Hierarchical Interactive Shell System");
    println!("  OUROBOROS: One Unified Runtime Orchestrating");
    println!("             a Bunch Of Random Old Servers");
    println!();
    println!("  The cluster is one machine.");
    println!("  Type ? for the cluster summary, help for commands.");
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

/// Tiny xorshift64 — enough entropy for prompt theatrics, zero deps.
struct Rng(u64);

impl Rng {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Rng(nanos | 1)
    }

    fn below(&mut self, n: u64) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x % n
    }
}

/// Random-case a letter per PRNG (docs/PROMPT.md — caps is chaos).
fn wob(c: char, rng: &mut Rng) -> char {
    if rng.below(2) == 0 {
        c.to_ascii_uppercase()
    } else {
        c
    }
}

/// The command shape (docs/PROMPT.md): bold-crimson `hiss` with random
/// caps, one extra dim-grey-red `s` per tail device, red `»`. Length is
/// truth; caps is chaos. Coloured variants wrap escapes in \x01..\x02
/// (zero-width for rustyline's width math).
fn hiss_prompt(rng: &mut Rng, tails: usize, color: bool) -> String {
    let mut base = String::new();
    for c in "hi".chars() {
        base.push(wob(c, rng));
    }
    for _ in 0..2 {
        base.push(wob('s', rng));
    }
    let mut tail = String::new();
    for _ in 0..tails {
        tail.push(wob('s', rng));
    }
    let sgr = |code: &str, s: &str| {
        if color {
            format!("\x01\x1b[{code}m\x02{s}\x01\x1b[0m\x02")
        } else {
            s.to_string()
        }
    };
    let caret = if color {
        "\x01\x1b[31m\x02»\x01\x1b[0m\x02"
    } else {
        ">"
    };
    format!("{}{} {caret} ", sgr("1;31", &base), sgr("2;31", &tail))
}

/// Interactive input history: ~/.ouro/hiss_history.
fn history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".ouro");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("hiss_history"))
}

/// The REPL state bundle: one place for everything a command touches.
struct Repl {
    topology: ClusterTopology,
    scheduler: Scheduler,
    ctx: Context,
    fmt: Formatter,
    config: propositions::ShellConfig,
    recovery: ouro_cluster::error_recovery::ErrorRecovery,
    node_addrs: Vec<(String, String)>,
}

impl Repl {
    /// Execute one input line. Returns false when the shell should exit.
    fn execute(&mut self, input: &str) -> bool {
        let input = input.trim();
        if input.is_empty() {
            return true;
        }
        if matches!(input, "quit" | "exit" | "q") {
            println!("Goodbye.");
            return false;
        }

        let cmd = interpret(input);
        if matches!(&cmd, ouro_hiss::parser::Command::Probe) && !self.node_addrs.is_empty() {
            println!("Probing all nodes...");
            for (id, addr) in &self.node_addrs {
                match agent_client::telemetry(addr) {
                    Ok(tel) => {
                        self.ctx.cache_properties(id, tel_props(&tel));
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
            return true;
        }

        match propositions::handle(
            cmd,
            &mut self.topology,
            &mut self.scheduler,
            &mut self.ctx,
            &mut self.fmt,
            &self.config,
            &mut self.recovery,
        ) {
            Ok(output) => println!("{output}"),
            Err(e) => println!("Error: {e}"),
        }
        true
    }
}

fn main() -> Result<()> {
    let color = std::io::stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty());
    banner(color);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let nodes_arg = args
        .windows(2)
        .find(|w| w[0] == "--nodes")
        .map(|w| w[1].clone());

    let topology = demo_topology();
    let scheduler = Scheduler::new(topology.clone());
    let mut ctx = Context::new();
    let fmt = Formatter::new(false);

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

    let mut repl = Repl {
        topology,
        scheduler,
        ctx,
        fmt,
        config,
        recovery: ouro_cluster::error_recovery::ErrorRecovery::new(),
        node_addrs,
    };

    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
    let mut rng = Rng::new();

    if interactive {
        // TTY: rustyline — hisstory, search, cursor handling. The prompt
        // re-hisses per command; its length tracks the topology.
        use rustyline::error::ReadlineError;
        use rustyline::history::FileHistory;
        use rustyline::Editor;

        let hist = history_path();
        let mut rl: Editor<(), FileHistory> = Editor::new()
            .map_err(|e| anyhow::anyhow!("readline init: {e}"))?;
        if let Some(h) = &hist {
            let _ = rl.load_history(h);
        }

        loop {
            let prompt = hiss_prompt(&mut rng, repl.topology.node_count(), !no_color);
            match rl.readline(&prompt) {
                Ok(line) => {
                    if !repl.execute(&line) {
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!();
                    break;
                }
                Err(e) => {
                    println!("input error: {e}");
                    break;
                }
            }
        }

        if let Some(h) = &hist {
            let _ = rl.save_history(h);
        }
    } else {
        // Piped/scripted: no prompt at all — clean `printf '?\n' | ouro-hiss`.
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            if !repl.execute(&line) {
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hiss_prompt_length_matches_topology() {
        // total_s = 2 + tails — the prompt counts the cluster at you.
        let mut rng = Rng::new();
        for tails in [0usize, 1, 5, 40] {
            let p = hiss_prompt(&mut rng, tails, false);
            let s_count = p.chars().filter(|c| c.to_ascii_lowercase() == 's').count();
            assert_eq!(s_count, 2 + tails, "tails={tails}");
        }
    }

    #[test]
    fn test_hiss_prompt_random_caps() {
        let mut rng = Rng::new();
        let variants: Vec<String> = (0..32).map(|_| hiss_prompt(&mut rng, 0, false)).collect();
        assert!(variants.iter().any(|v| v.contains('H')), "expected some caps");
        assert!(variants.iter().any(|v| v.contains('h')), "expected some lower");
    }

    #[test]
    fn test_hiss_prompt_color_uses_zero_width_markers() {
        let mut rng = Rng::new();
        let p = hiss_prompt(&mut rng, 2, true);
        assert!(p.contains("\x01\x1b[1;31m\x02"), "bold red base");
        assert!(p.contains("\x01\x1b[2;31m\x02"), "dim grey-red tail");
        assert!(p.contains("»"));
        let plain = hiss_prompt(&mut rng, 2, false);
        assert!(!plain.contains('\x1b'), "no-color variant has no escapes");
    }

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
