//! `ouro-ttyd` core: FIFO face for one node (WP1, PLAN §18.3 T1).
//!
//! Runbook: `docs/R2_BRINGUP.md` §3. Lives in the shell crate because the
//! master-side transport client (`agent_client`) lives here; the daemon
//! binary is `src/bin/ouro-ttyd.rs`.
//!
//! Line protocol, one request in flight (lockstep):
//! - in:  `ping` | `echo <text>` | `stage_setup <path>` | `stage_reset` |
//!   `stage_token <pos>|<id>` | `stage_step <hex>` | `stage_sample <hex>` |
//!   other agent tasks (`acts_echo <hex>`, `load_shard <path>`, ...) |
//!   dot-form (`budget 120w.`, `probe.`, `n1?`)
//! - out: `ok <text>` | `queued <reason>` | `err <msg>`
//!
//! Every task line routes through `Scheduler::schedule()` and the energy
//! budget before it touches the wire — no FIFO bypass of Art. 4.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as AnyhowContext, Result};
use ouro_cluster::beast::topology::{ClusterTopology, NodeEntry};
use ouro_cluster::scheduler::workload_class::WorkloadClass;
use ouro_cluster::scheduler::{ScheduleOutcome, Scheduler, Task};
use ouro_cluster::transport::auth::{self, Secret};

use crate::agent_client::{self, AgentTask, AgentTaskResult};
use crate::context::Context;
use crate::formatter::Formatter;
use crate::parser::interpret;
use crate::propositions;

/// Default cluster budget until a `budget <N>w.` line says otherwise
/// (runbook example: `budget 120w.`).
pub const DEFAULT_BUDGET_WATTS: u32 = 120;

/// Default per-task wire timeout (inference stages can take a while).
pub const TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Transport to the agent behind this FIFO face.
#[derive(Debug, Clone)]
pub enum TtyWire {
    /// Master-side TCP client (`agent_client`), one connection per request.
    Tcp(String),
    /// Child process speaking the authed stdio protocol — WP3 getty-shim
    /// path, e.g. `ssh -T user@host -- ouro-agent --stdio-tty` or a raw
    /// serial line. Spawned once per tty connection, lockstep over its
    /// pipes; respawned if it dies.
    Child(String),
}

/// A spawned child wire: pipes held open for the session's lifetime.
struct ChildWire {
    _child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

fn spawn_child(cmd: &str) -> Result<ChildWire> {
    use std::process::{Command, Stdio};
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn tty child: {}", cmd))?;
    let stdin = child.stdin.take().context("child stdin")?;
    let stdout = std::io::BufReader::new(child.stdout.take().context("child stdout")?);
    Ok(ChildWire { _child: child, stdin, stdout })
}

/// One lockstep signed request over a child's stdio.
fn child_request(wire: &mut ChildWire, secret: &Secret, seq: u64, body: &str) -> Result<String> {
    use std::io::Write;
    writeln!(wire.stdin, "{}", auth::sign_line(secret, seq, body))?;
    wire.stdin.flush()?;
    let mut line = String::new();
    let n = wire.stdout.read_line(&mut line)?;
    if n == 0 {
        anyhow::bail!("tty child closed the wire (auth failure or exit)");
    }
    let (resp_seq, resp_body) = auth::open_line(secret, line.trim())?;
    if resp_seq != seq {
        anyhow::bail!("child reply seq {} != request {}", resp_seq, seq);
    }
    Ok(resp_body.to_string())
}

const TASK_NAMES: &[&str] = &[
    "echo",
    "acts_echo",
    "bench_sum",
    "load_shard",
    "tokenize",
    "detok",
    "bitnet_generate",
];

static TASK_SEQ: AtomicU64 = AtomicU64::new(0);

/// One response line, as written to the `.out` FIFO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtyResponse {
    Ok(String),
    Queued(String),
    Err(String),
}

impl fmt::Display for TtyResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtyResponse::Ok(s) => write!(f, "ok {}", s),
            TtyResponse::Queued(r) => write!(f, "queued {}", r),
            TtyResponse::Err(e) => write!(f, "err {}", e),
        }
    }
}

/// Master-side state for one FIFO face. Persisted across tty reconnects so
/// budget decisions survive a writer going away.
pub struct TtySession {
    pub node: String,
    pub wire: TtyWire,
    secret: Secret,
    topology: ClusterTopology,
    scheduler: Scheduler,
    ctx: Context,
    fmt: Formatter,
    config: propositions::ShellConfig,
    child: Option<ChildWire>,
}

impl TtySession {
    /// Single-node topology. Placeholders are scheduling-agnostic (class
    /// `Unknown` dispatches anywhere); measured probe facts replace them
    /// when the node registers in the Beast graph (R2_BRINGUP.md §6.2).
    /// The secret is mandatory — a session without one cannot exist.
    pub fn new(node: &str, wire: TtyWire, secret: Secret) -> Self {
        let mut topology = ClusterTopology::new();
        topology.power_budget_watts = DEFAULT_BUDGET_WATTS;
        topology.nodes.push(NodeEntry {
            id: node.to_string(),
            hostname: node.to_string(),
            ip: match &wire {
                TtyWire::Tcp(addr) => addr.clone(),
                TtyWire::Child(_) => "stdio".to_string(),
            },
            cpu_model: "unprobed".to_string(),
            cores: 1,
            threads: 1,
            has_avx: true,
            has_avx2: true,
            has_sse42: true,
            ram_mib: 0,
            tdp_watts: 35,
            has_gpu: false,
            gpu_model: String::new(),
            gpu_vram_mib: 0,
            gpu_driver: String::new(),
        });
        let scheduler = Scheduler::new(topology.clone());
        let mut config = propositions::ShellConfig::new();
        // Live agent endpoints only exist on the TCP wire; a getty/stdio
        // node is reached through the child, not the node_addrs list.
        if let TtyWire::Tcp(addr) = &wire {
            config.node_addrs = vec![(node.to_string(), addr.clone())];
        }
        Self {
            node: node.to_string(),
            wire,
            secret,
            topology,
            scheduler,
            ctx: Context::new(),
            fmt: Formatter::new(false),
            config,
            child: None,
        }
    }

    /// Handle one request line, produce one response line.
    pub fn handle_line(&mut self, line: &str) -> TtyResponse {
        let line = line.trim();
        if line.is_empty() {
            return TtyResponse::Err("empty line".to_string());
        }
        if line == "ping" {
            return self.dispatch_ping();
        }
        if let Some(rest) = line.strip_prefix("stage_") {
            return self.dispatch_stage(rest);
        }
        if let Some((first, rest)) = line.split_once(' ') {
            if TASK_NAMES.contains(&first) {
                let (class, watts) = task_profile(first);
                return self.dispatch_task(first, rest.trim(), class, watts);
            }
        }
        // Dot-form: the shell surface over the FIFO (`budget 120w.`,
        // `probe.`, `n1?`, ...).
        let cmd = interpret(line);
        match propositions::handle(
            cmd,
            &mut self.topology,
            &mut self.scheduler,
            &mut self.ctx,
            &mut self.fmt,
            &self.config,
        ) {
            Ok(out) => TtyResponse::Ok(out),
            Err(e) => TtyResponse::Err(e.to_string()),
        }
    }

    fn dispatch_stage(&mut self, rest: &str) -> TtyResponse {
        let (kind, payload) = match rest.split_once(' ') {
            Some((k, p)) => (k, p.trim()),
            None => (rest, ""),
        };
        let name = format!("stage_{}", kind);
        let watts = match kind {
            "setup" => 10,
            "reset" => 1,
            "token" | "step" => 8,
            "sample" => 10,
            _ => return TtyResponse::Err(format!("unknown stage task {}", name)),
        };
        self.dispatch_task(&name, payload, WorkloadClass::LlmInference, watts)
    }

    fn dispatch_ping(&mut self) -> TtyResponse {
        let task = Task {
            name: "ping".to_string(),
            class: WorkloadClass::Unknown,
            payload: String::new(),
            estimated_watts: 1,
            estimated_seconds: 1,
        };
        match self.scheduler.schedule(&task) {
            Ok(ScheduleOutcome::Dispatched { .. }) => {
                let pong = match self.wire.clone() {
                    TtyWire::Tcp(addr) => agent_client::ping_with(&self.secret, &addr).map(|_| ()),
                    TtyWire::Child(cmd) => self
                        .child_wire_request(&cmd, "ping")
                        .and_then(|body| assert_pong(&body)),
                };
                self.scheduler.complete(1);
                match pong {
                    Ok(()) => TtyResponse::Ok("pong".to_string()),
                    Err(e) => TtyResponse::Err(e.to_string()),
                }
            }
            Ok(ScheduleOutcome::Queued { reason }) => TtyResponse::Queued(reason),
            Err(e) => TtyResponse::Err(e.to_string()),
        }
    }

    /// This node's boot tagline, over the signed wire (registration
    /// echo: the master prints it in crimson when the node joins).
    /// Empty on the node = no banner.
    pub fn motto(&mut self) -> Result<String> {
        let task = Task {
            name: "tagline".to_string(),
            class: WorkloadClass::Unknown,
            payload: String::new(),
            estimated_watts: 1,
            estimated_seconds: 1,
        };
        match self.scheduler.schedule(&task) {
            Ok(ScheduleOutcome::Dispatched { .. }) => {}
            Ok(ScheduleOutcome::Queued { reason }) => anyhow::bail!("queued: {}", reason),
            Err(e) => return Err(e),
        }
        let resp = match self.wire.clone() {
            TtyWire::Tcp(addr) => agent_client::raw_with(&self.secret, &addr, "tagline"),
            TtyWire::Child(cmd) => self.child_wire_request(&cmd, "tagline"),
        };
        self.scheduler.complete(1);
        let text = resp?.trim().to_string();
        if text.is_empty() {
            anyhow::bail!("no tagline on node");
        }
        Ok(text)
    }

    /// Schedule first, dispatch second, release watts after. Lockstep:
    /// exactly one request in flight (matches `stage.rs` sequential
    /// semantics; costs nothing on TTY paths).
    fn dispatch_task(        &mut self,
        name: &str,
        payload: &str,
        class: WorkloadClass,
        watts: u32,
    ) -> TtyResponse {
        let sched_task = Task {
            name: name.to_string(),
            class,
            payload: payload.to_string(),
            estimated_watts: watts,
            estimated_seconds: 1,
        };
        match self.scheduler.schedule(&sched_task) {
            Ok(ScheduleOutcome::Dispatched { .. }) => {}
            Ok(ScheduleOutcome::Queued { reason }) => return TtyResponse::Queued(reason),
            Err(e) => return TtyResponse::Err(e.to_string()),
        }
        let agent_task = AgentTask {
            id: format!("tty-{}", TASK_SEQ.fetch_add(1, Ordering::Relaxed)),
            name: name.to_string(),
            payload: payload.to_string(),
            estimated_watts: watts,
            estimated_seconds: 1,
        };
        let body = match serde_json::to_string(&agent_task) {
            Ok(b) => b,
            Err(e) => {
                self.scheduler.complete(watts);
                return TtyResponse::Err(e.to_string());
            }
        };
        let resp = match self.wire.clone() {
            TtyWire::Tcp(addr) => agent_client::execute_with_timeout(
                &self.secret,
                &addr,
                &agent_task,
                TASK_TIMEOUT,
            )
            .and_then(|r| serde_json::to_string(&r).context("serialize task result")),
            TtyWire::Child(cmd) => self.child_wire_request(&cmd, &body),
        };
        self.scheduler.complete(watts);
        match resp {
            Ok(body) => parse_task_result(&body),
            Err(e) => TtyResponse::Err(e.to_string()),
        }
    }

    /// Send one body over the child wire, respawning a dead child first.
    fn child_wire_request(&mut self, cmd: &str, body: &str) -> Result<String> {
        if self.child.is_none() {
            self.child = Some(spawn_child(cmd)?);
        }
        let seq = TASK_SEQ.fetch_add(1, Ordering::Relaxed);
        match child_request(self.child.as_mut().unwrap(), &self.secret, seq, body) {
            Ok(resp) => Ok(resp),
            Err(e) => {
                self.child = None;
                Err(e)
            }
        }
    }
}

fn assert_pong(body: &str) -> Result<()> {
    if body != "pong" {
        anyhow::bail!("expected pong, got {:?}", body);
    }
    Ok(())
}

fn parse_task_result(body: &str) -> TtyResponse {
    match serde_json::from_str::<AgentTaskResult>(body) {
        Ok(r) if r.status == "Success" => TtyResponse::Ok(r.output),
        Ok(r) => TtyResponse::Err(r.output),
        Err(e) => TtyResponse::Err(e.to_string()),
    }
}

fn task_profile(name: &str) -> (WorkloadClass, u32) {
    match name {
        "bitnet_generate" | "tokenize" | "detok" => (WorkloadClass::LlmInference, 10),
        "acts_echo" => (WorkloadClass::SimdFriendly, 5),
        "bench_sum" => (WorkloadClass::BranchHeavy, 5),
        "load_shard" => (WorkloadClass::Irregular, 10),
        _ => (WorkloadClass::Unknown, 1),
    }
}

/// FIFO paths for one node's tty face.
pub fn fifo_paths(tty_dir: &Path, node: &str) -> (PathBuf, PathBuf) {
    (
        tty_dir.join(format!("{}.in", node)),
        tty_dir.join(format!("{}.out", node)),
    )
}

/// Create a FIFO if missing. Refuses to clobber non-FIFO files.
pub fn ensure_fifo(path: &Path) -> Result<()> {
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.file_type().is_fifo() {
            return Ok(());
        }
        anyhow::bail!("{} exists and is not a FIFO", path.display());
    }
    let c = std::ffi::CString::new(path.as_os_str().as_bytes())?;
    let rc = unsafe { libc::mkfifo(c.as_ptr(), 0o600) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("mkfifo {}", path.display()));
    }
    Ok(())
}

/// Crimson banner escapes (truecolor; fbcon maps to its remapped palette).
pub const CRIMSON: &str = "\x1b[38;2;220;20;60m";
pub const RESET: &str = "\x1b[0m";

/// Serve one tty connection: open `.in` (waits for a writer), then `.out`
/// (waits for a reader), then lockstep — one line in, one response out.
/// The first line on `.out` is the node's crimson tagline banner
/// (skippable by any consumer); Returns when the writer closes its end
/// (EOF).
pub fn serve_connection(session: &mut TtySession, in_path: &Path, out_path: &Path) -> Result<()> {
    let in_file = File::open(in_path).with_context(|| format!("open {}", in_path.display()))?;
    let mut out_file = OpenOptions::new()
        .write(true)
        .open(out_path)
        .with_context(|| format!("open {}", out_path.display()))?;
    if let Ok(motto) = session.motto() {
        writeln!(out_file, "{CRIMSON}◆ {motto}{RESET}")?;
        out_file.flush()?;
    }
    let reader = BufReader::new(in_file);
    for line in reader.lines() {
        let line = line?;
        let response = session.handle_line(&line);
        writeln!(out_file, "{}", response)?;
        out_file.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_client::AgentTaskResult;
    use ouro_cluster::pipeline::{from_hex, to_hex, Activation};
    use ouro_cluster::transport::auth::{self, Secret};
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    const KEY: Secret = [7u8; 32];

    // ---- fake agent: same authed wire protocol as ouro-agent ----

    fn fake_agent() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        thread::spawn(move || {
            for stream in listener.incoming() {
                handle_conn(stream.unwrap());
            }
        });
        addr
    }

    fn handle_conn(stream: TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut writer = stream;
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            let resp = match auth::open_line(&KEY, line.trim()) {
                Ok((seq, body)) => auth::sign_line(&KEY, seq, &fake_process(body)),
                Err(_) => "err auth".to_string(),
            };
            writer.write_all(resp.as_bytes()).unwrap();
            writer.write_all(b"\n").unwrap();
            writer.flush().unwrap();
            line.clear();
        }
    }

    fn fake_process(msg: &str) -> String {
        if msg == "ping" {
            return "pong".to_string();
        }
        if msg == "tagline" {
            return "the tail feeds the head".to_string();
        }
        let task: AgentTask = match serde_json::from_str(msg) {
            Ok(t) => t,
            Err(_) => return r#"{"error":"invalid task"}"#.to_string(),
        };
        let (status, output) = match task.name.as_str() {
            "echo" => ("Success", task.payload.clone()),
            "acts_echo" => ("Success", acts_echo(&task.payload)),
            "stage_step" => ("Success", format!("stage-ack:{}", task.payload)),
            other => ("Failed", format!("unknown task: {}", other)),
        };
        serde_json::to_string(&AgentTaskResult {
            task_id: task.id,
            status: status.to_string(),
            output,
            elapsed_ms: 0,
            peak_watts: task.estimated_watts,
        })
        .unwrap()
    }

    /// Mirrors agent executor's acts_echo: the transport-parity fixture.
    fn acts_echo(hex_payload: &str) -> String {
        let bytes = from_hex(hex_payload).unwrap();
        let act = Activation::decode(&bytes).unwrap();
        format!(
            "seq={} pos={} layers={}-{} elems={}",
            act.sequence, act.token_pos, act.layer_start, act.layer_end, act.data.len()
        )
    }

    fn sample_activation() -> String {
        let act = Activation {
            sequence: 7,
            token_pos: 3,
            layer_start: 0,
            layer_end: 9,
            data: vec![0.25, -0.5, 1.0, 2.0],
        };
        to_hex(&act.encode())
    }

    fn tcp_execute(addr: &str, name: &str, payload: &str) -> String {
        agent_client::execute_with_timeout(
            &KEY,
            addr,
            &AgentTask {
                id: "tcp-ref".into(),
                name: name.into(),
                payload: payload.into(),
                estimated_watts: 1,
                estimated_seconds: 1,
            },
            Duration::from_secs(10),
        )
        .unwrap()
        .output
    }

    // ---- gate: TTY path == TCP path == in-process ----

    #[test]
    fn test_echo_parity_tty_tcp_inprocess() {
        let addr = fake_agent();
        let payload = "hello tty world";
        let tcp = tcp_execute(&addr, "echo", payload);
        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        let tty = session.handle_line(&format!("echo {}", payload));
        assert_eq!(tty, TtyResponse::Ok(tcp));
        assert_eq!(tty, TtyResponse::Ok(payload.to_string()));
    }

    #[test]
    fn test_acts_hex_parity_tty_tcp() {
        let addr = fake_agent();
        let hex = sample_activation();
        let tcp = tcp_execute(&addr, "acts_echo", &hex);
        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        let tty = session.handle_line(&format!("acts_echo {}", hex));
        assert_eq!(tty, TtyResponse::Ok(tcp));
        assert!(tty.to_string().contains("elems=4"));
    }

    #[test]
    fn test_stage_step_reaches_agent_and_returns_continuation() {
        let addr = fake_agent();
        let tcp = tcp_execute(&addr, "stage_step", "deadbeef");
        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        let tty = session.handle_line("stage_step deadbeef");
        assert_eq!(tty, TtyResponse::Ok(tcp));
        assert_eq!(tty, TtyResponse::Ok("stage-ack:deadbeef".into()));
    }

    #[test]
    fn test_ping_line() {
        let addr = fake_agent();
        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        assert_eq!(session.handle_line("ping"), TtyResponse::Ok("pong".into()));
    }

    // ---- Art. 4: budget gate on the FIFO face ----

    #[test]
    fn test_budget_gate_blocks_dispatch() {
        let addr = fake_agent();
        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        let resp = session.handle_line("budget 0w.");
        assert!(matches!(resp, TtyResponse::Ok(_)), "budget line: {}", resp);
        let resp = session.handle_line("echo too expensive");
        assert!(
            matches!(resp, TtyResponse::Queued(ref r) if r.contains("energy budget")),
            "expected queued, got {}",
            resp
        );
    }

    // ---- dot-form over the FIFO ----

    #[test]
    fn test_dot_form_budget_and_node_query() {
        let addr = fake_agent();
        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        let resp = session.handle_line("n1?");
        assert!(matches!(resp, TtyResponse::Ok(ref s) if s.contains("n1")));
    }

    #[test]
    fn test_empty_line_errors() {
        let addr = fake_agent();
        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        assert_eq!(
            session.handle_line("   "),
            TtyResponse::Err("empty line".into())
        );
    }

    // ---- WP3: child wire (getty-shim path) ----

    #[test]
    fn test_child_bridge_signed_roundtrip() {
        // `cat` echoes the signed request line verbatim: exercises
        // sign → write → read → verify → seq-match through a real child.
        let mut wire = spawn_child("cat").unwrap();
        let resp = child_request(&mut wire, &KEY, 11, "hello child").unwrap();
        assert_eq!(resp, "hello child");
        let resp = child_request(&mut wire, &KEY, 12, "second line").unwrap();
        assert_eq!(resp, "second line");
    }

    #[test]
    fn test_child_bridge_dead_child_reports_and_respawns() {
        // `true` exits immediately: first request reports the dead wire.
        let mut session = TtySession::new("n1", TtyWire::Child("true".into()), KEY);
        let resp = session.handle_line("ping");
        assert!(matches!(resp, TtyResponse::Err(_)), "got {}", resp);
    }

    // ---- full FIFO loopback: the WP1 gate ----

    #[test]
    fn test_fifo_loopback_end_to_end() {
        let addr = fake_agent();
        let dir = std::env::temp_dir().join(format!("ouro_ttyd_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (in_path, out_path) = fifo_paths(&dir, "n1");
        ensure_fifo(&in_path).unwrap();
        ensure_fifo(&out_path).unwrap();

        let mut session = TtySession::new("n1", TtyWire::Tcp(addr.clone()), KEY);
        let in_c = in_path.clone();
        let out_c = out_path.clone();
        let server = thread::spawn(move || {
            serve_connection(&mut session, &in_c, &out_c).unwrap();
        });

        // Master side: open .in (write) + .out (read), keep both across
        // requests, one request in flight.
        let mut req = OpenOptions::new().write(true).open(&in_path).unwrap();
        let resp = BufReader::new(File::open(&out_path).unwrap());

        // first line: the registration echo (crimson tagline banner)
        let mut line = String::new();
        let mut resp = resp;
        resp.read_line(&mut line).unwrap();
        assert!(line.contains("◆ the tail feeds the head"), "banner: {}", line);

        req.write_all(b"echo hello fifo\n").unwrap();
        req.flush().unwrap();
        let mut line = String::new();
        resp.read_line(&mut line).unwrap();
        assert_eq!(line.trim(), "ok hello fifo");

        // hex continuation path: same bytes as the TCP reference
        let hex = sample_activation();
        req.write_all(format!("acts_echo {}\n", hex).as_bytes()).unwrap();
        req.flush().unwrap();
        let mut line = String::new();
        resp.read_line(&mut line).unwrap();
        let tcp = tcp_execute(&addr, "acts_echo", &hex);
        assert_eq!(line.trim(), format!("ok {}", tcp));

        // dot-form rides the same face
        req.write_all(b"budget 60w.\n").unwrap();
        req.flush().unwrap();
        let mut line = String::new();
        resp.read_line(&mut line).unwrap();
        assert!(line.starts_with("ok "), "budget over fifo: {}", line);

        drop(req);
        server.join().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
