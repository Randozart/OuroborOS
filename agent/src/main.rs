mod executor;
#[cfg(feature = "gpu")]
mod gpu;
mod head_link;
mod stage;
mod telemetry;
mod update;

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;
use ouro_cluster::transport::auth::{self, Secret};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::broadcast;
use tokio::time::interval;

const DEFAULT_PORT: u16 = 9500;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// WP-U4 rail: an update never yanks a running task (UPDATE_ROADMAP
/// §rails). Set while a task executes; `update begin` is refused with
/// `err busy` while it holds.
static TASK_BUSY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Agent daemon entry point.
///
/// Default mode: TCP daemon for the head node, HMAC-authenticated
/// newline protocol (`seq tag body`); refuses to start without
/// OURO_SECRET_FILE.
///
/// `ouro-agent --stdio-tty` (getty-shim, WP3): same authed protocol on
/// stdin/stdout instead of TCP. The tail's getty line spawns it; the
/// head's ouro-ttyd connects via `ssh -T` (or raw serial). Zero
/// install: any booted Linux with a login joins the graph.
#[tokio::main]
async fn main() -> Result<()> {
    // WP-U4 handoff: a pushed agent takes over before anything else,
    // guarded by the boot counter (three boots without proving itself
    // on the bus → discarded; the baked-in binary is the permanent
    // fallback slot).
    if let Some(live) = update::boot_check() {
        eprintln!("update: handing off to {}", live.display());
        let args: Vec<String> = std::env::args().skip(1).collect();
        use std::os::unix::process::CommandExt;
        // exec replaces the process on success; the return value is
        // the error itself, not a Result — reaching the eprintln IS
        // the failure path.
        let err = std::process::Command::new(&live).args(&args).exec();
        eprintln!("update: exec failed ({err}) — falling back to baked-in");
    }

    let secret = auth::secret_from_env()?;

    // --head <addr>: push-based registration + telemetry heartbeat to
    // the registry daemon. Spawned FIRST so it coexists with every mode
    // below (the getty shim passes both --stdio-tty and --head).
    if let Some(i) = std::env::args().position(|a| a == "--head") {
        if let Some(addr) = std::env::args().nth(i + 1) {
            tokio::spawn(async move {
                if let Err(e) =
                    head_link::run(secret, addr, HEARTBEAT_INTERVAL).await
                {
                    eprintln!("head-link terminated: {}", e);
                }
            });
        }
    }

    if std::env::args().any(|a| a == "--stdio-tty") {
        // One process, two mouths: the login wire on stdio AND the TCP
        // task channel (9500). Found live: shim-only agents never bound
        // 9500 — the head's execute got connection-refused while the
        // bus link purred. Stdout is the wire here, so serve_tcp logs
        // to stderr only.
        let port: u16 = std::env::var("OURO_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_PORT);
        tokio::spawn(async move {
            if let Err(e) = serve_tcp(secret, port).await {
                eprintln!("task channel terminated: {}", e);
            }
        });
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        serve_stdio(&secret, stdin.lock(), stdout.lock())?;
        return Ok(());
    }

    println!("auth: OURO_SECRET_FILE loaded (32B HMAC-SHA256)");
    let port: u16 = std::env::var("OURO_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let shutdown_rx = shutdown_tx.clone();

    // Task channel (daemon mode): the accept loop owns this task.
    let tcp_handle = tokio::spawn(async move {
        if let Err(e) = serve_tcp(secret, port).await {
            eprintln!("task channel terminated: {}", e);
        }
    });

    // Heartbeat task
    let heartbeat_handle = tokio::spawn(async move {
        let mut ticker = interval(HEARTBEAT_INTERVAL);
        let mut rx = shutdown_rx.subscribe();
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    if let Ok(tel) = telemetry::collect() {
                        println!(
                            "[heartbeat] {} | {}W | {}C | load {:.2}",
                            tel.hostname, tel.power_watts, tel.temp_c, tel.load_avg
                        );
                    }
                }
                _ = rx.recv() => {
                    println!("[heartbeat] shutting down");
                    break;
                }
            }
        }
    });

    // Handle Ctrl+C
    let shutdown_handle = shutdown_tx.clone();
    let ctrl_c_handle = tokio::spawn(async move {
        signal::ctrl_c().await.ok();
        println!("\n[agent] received shutdown signal");
        let _ = shutdown_handle.send(());
    });

    // Wait for background tasks (the accept loop lives in tcp_handle)
    let _ = tokio::join!(tcp_handle, heartbeat_handle, ctrl_c_handle);
    println!("ouro-agent stopped.");
    Ok(())
}

/// The TCP task channel: bind + accept loop, one authed wire per
/// connection. Runs in daemon mode AND alongside the getty shim
/// (--stdio-tty): login keeps its stdio wire, the head gets a task
/// port. All logs go to stderr — in shim mode stdout IS the wire.
async fn serve_tcp(secret: Secret, port: u16) -> Result<()> {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await?;
    eprintln!("ouro-agent task channel on {}", addr);
    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                eprintln!("[connect] {}", peer);
                tokio::spawn(async move {
                    handle_connection(secret, stream, peer).await;
                });
            }
            Err(e) => {
                eprintln!("[error] accept: {}", e);
            }
        }
    }
}

/// Handle a single TCP connection.
///
/// Protocol: HMAC-authenticated newline-delimited messages
/// (`<seq> <hex-tag> <body>`). Every valid request gets a signed
/// response under the same seq. Auth failure → terse unsigned `err auth`,
/// connection closed (no oracle).
async fn handle_connection(
    secret: Secret,
    mut stream: tokio::net::TcpStream,
    peer: SocketAddr,
) {
    // Request-response, one line each way — read EXACTLY one line,
    // never a byte more: the client may coalesce the line with the
    // frame payload that follows an `update begin`, and a buffering
    // read would swallow those frame bytes with the BufReader's buffer
    // (found live: the selftest's frames vanished; a lucky manual run
    // won the race). The stream is consumed (into_std) when an update
    // takes over — the whole socket crosses into the blocking world.
    let mut line_buf: Vec<u8> = Vec::new();
    let mut one = [0u8; 1];
    loop {
        line_buf.clear();
        loop {
            match stream.read_exact(&mut one).await {
                Ok(_) => {}
                Err(_) => return,
            }
            if one[0] == b'\n' {
                break;
            }
            if one[0] != b'\r' {
                line_buf.push(one[0]);
            }
            if line_buf.len() > 64 * 1024 {
                eprintln!("[line] too long — closing");
                return;
            }
        }
        let line = String::from_utf8_lossy(&line_buf).to_string();
        // Update verbs switch the socket to frame mode — the whole
        // stream (WP-U4; TCP face only: the tty line discipline cannot
        // carry frames).
        if let Some((expect, wire)) = update_verb(&secret, &line) {
            if TASK_BUSY.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = stream.write_all(b"err busy\n").await;
                continue;
            }
            let stdio_sock = match stream.into_std() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[update] into_std: {e}");
                    break;
                }
            };
            // tokio lives in nonblocking mode; the frame pumps are
            // blocking reads (spawn_blocking) — flip explicitly or every
            // read returns EAGAIN (found live: os error 11).
            if let Err(e) = stdio_sock.set_nonblocking(false) {
                eprintln!("[update] set_nonblocking: {e}");
                break;
            }
            eprintln!("[update] {} begin", expect.as_str());
            match tokio::task::spawn_blocking(move || {
                update::handle_update(secret, stdio_sock, expect, wire)
            })
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("[update] failed: {e:#}"),
                Err(e) => eprintln!("[update] task panic: {e}"),
            }
            break; // receipt was written on the socket; exec or reboot follows
        }
        // Task JSON: run on the blocking pool, never inline on a tokio
        // worker. A long task (gpu_selftest's OpenCL JIT alone takes ~90s
        // on first run) starves the runtime on small cores and wedges the
        // whole task channel (found live on the 2-core HP: ping/diag/task
        // all hung while the heartbeat kept ticking).
        let task = auth::open_line(&secret, &line)
            .ok()
            .and_then(|(seq, body)| serde_json::from_str::<executor::Task>(&body).ok().map(|t| (seq, t)));
        if let Some((seq, task)) = task {
            let name = task.name.clone();
            eprintln!("[task] {name} begin");
            if TASK_BUSY.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let _ = stream.write_all(b"err busy\n").await;
                continue;
            }
            let result = tokio::task::spawn_blocking(move || executor::execute(&task)).await;
            TASK_BUSY.store(false, std::sync::atomic::Ordering::Relaxed);
            let response = match result {
                Ok(Ok(out)) => serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()),
                Ok(Err(e)) => format!(r#"{{"error":"{}"}}"#, e),
                Err(e) => format!(r#"{{"error":"task panic: {}"}}"#, e),
            };
            let signed = auth::sign_line(&secret, seq, &response);
            if stream.write_all(signed.as_bytes()).await.is_err() {
                break;
            }
            if stream.write_all(b"\n").await.is_err() {
                break;
            }
            continue;
        }
        match authed_process(&secret, &line) {
            Some(response) => {
                if stream.write_all(response.as_bytes()).await.is_err() {
                    break;
                }
                if stream.write_all(b"\n").await.is_err() {
                    break;
                }
            }
            None => {
                let _ = stream.write_all(b"err auth\n").await;
                break;
            }
        }
    }

    eprintln!("[disconnect] {}", peer);
}

/// Detect `update begin <signed-manifest>` on the wire: authenticated
/// (the line must carry a valid tag) and the manifest's artifact kind
/// is parsed loosely here — the strict verification happens in
/// [`update::handle_update`] against the image-baked pubkey.
fn update_verb(secret: &Secret, line: &str) -> Option<(ouro_cluster::update::Artifact, String)> {
    let (_seq, body) = match auth::open_line(secret, line) {
        Ok(x) => x,
        Err(e) => {
            if line.starts_with("update ") {
                eprintln!("[update] line auth failed: {e}");
            }
            return None;
        }
    };
    let rest = body.strip_prefix("update begin ")?;
    let value: serde_json::Value = match serde_json::from_str(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[update] manifest json parse: {e}");
            return None;
        }
    };
    let kind = match value.get("artifact").and_then(|v| v.as_str()) {
        Some(k) => match ouro_cluster::update::Artifact::parse(k) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[update] artifact kind: {e}");
                return None;
            }
        },
        None => {
            eprintln!("[update] manifest missing artifact kind");
            return None;
        }
    };
    Some((kind, rest.to_string()))
}

/// Verify + process one line, produce one signed response line.
/// `None` = auth failure.
fn authed_process(secret: &Secret, line: &str) -> Option<String> {
    let (seq, body) = auth::open_line(secret, line).ok()?;
    let response = process_message(body);
    Some(auth::sign_line(secret, seq, &response))
}

/// Getty-shim loop: signed line in from stdin, signed line out on
/// stdout, flush per line. On a real TTY (raw serial / console), the
/// brand banner prints first — interactive eyes get the cinematic boot,
/// pipes (`ssh -T`, FIFO face) get clean protocol only. Auth failure →
/// `err auth`, stop (the spawning getty respawns = fresh login).
/// EOF → clean exit.
fn serve_stdio<R: std::io::BufRead, W: std::io::Write>(
    secret: &Secret,
    mut input: R,
    mut output: W,
) -> Result<()> {
    if unsafe { libc::isatty(0) } == 1 {
        if let Ok(issue) = std::fs::read_to_string("/run/ouro/issue") {
            let _ = write!(output, "{issue}");
            let _ = output.flush();
        }
    }
    let mut line = String::new();
    while input.read_line(&mut line)? > 0 {
        match authed_process(secret, line.trim_end()) {
            Some(resp) => {
                writeln!(output, "{}", resp)?;
                output.flush()?;
            }
            None => {
                writeln!(output, "err auth")?;
                output.flush()?;
                return Ok(());
            }
        }
        line.clear();
    }
    Ok(())
}

/// Process a single message from the head.
fn process_message(msg: &str) -> String {
    let trimmed = msg.trim();

    // Telemetry request
    if trimmed == "telemetry" {
        match telemetry::collect() {
            Ok(tel) => serde_json::to_string(&tel).unwrap_or_else(|_| "{}".into()),
            Err(e) => format!(r#"{{"error":"{}"}}"#, e),
        }
    }
    // Ping
    else if trimmed == "ping" {
        "pong".into()
    }
    // Diagnostic: measured network facts (head-link drills). Read-only,
    // no claims — the kernel's own tables, not our opinion of them.
    else if trimmed == "diag" {
        let mut out = String::from("diag:");
        for f in ["/proc/net/route"] {
            if let Ok(data) = std::fs::read_to_string(f) {
                out.push_str(&format!("\n--- {f} ---\n{data}"));
            }
        }
        if let Ok(dirs) = std::fs::read_dir("/sys/class/net") {
            for d in dirs.flatten() {
                let name = d.file_name().to_string_lossy().to_string();
                let carrier = std::fs::read_to_string(d.path().join("carrier"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| "?".into());
                let mac = std::fs::read_to_string(d.path().join("address"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                out.push_str(&format!("\nnic {name} carrier={carrier} mac={mac}"));
            }
        }
        // OpenCL census: vendor registrations + render nodes — the
        // compute path's own confession (found live: the loader saw
        // zero ICDs because the agent env had no OCL_ICD_VENDORS and
        // NixOS keeps ICDs under /run/opengl-driver).
        out.push_str("\n--- opencl ---");
        out.push_str(&format!(
            "\nOCL_ICD_VENDORS={}",
            std::env::var("OCL_ICD_VENDORS").unwrap_or_else(|_| "(unset)".into())
        ));
        for dir in ["/etc/OpenCL/vendors", "/run/opengl-driver/etc/OpenCL/vendors"] {
            match std::fs::read_dir(dir) {
                Ok(entries) => {
                    for e in entries.flatten() {
                        out.push_str(&format!("\nicd {}", e.path().display()));
                    }
                }
                Err(_) => out.push_str(&format!("\nicd-dir {dir}: absent")),
            }
        }
        match std::fs::read_dir("/dev/dri") {
            Ok(entries) => {
                for e in entries.flatten() {
                    out.push_str(&format!("\ndri {}", e.file_name().to_string_lossy()));
                }
            }
            Err(_) => out.push_str("\ndri /dev/dri: absent"),
        }
        let out = out.replace('\n', " | ");
        // tty line discipline truncates ~4KB canonical lines; keep the
        // essential facts (route + NIC census) well under that.
        let mut out = out;
        if out.len() > 1400 {
            out.truncate(1400);
            out.push_str(" …");
        }
        out
    }
    // Tagline: this boot's motto, for the head's registration echo
    else if trimmed == "tagline" {
        let from_env = std::env::var("OURO_TAGLINE").unwrap_or_default();
        if !from_env.trim().is_empty() {
            from_env
        } else {
            std::fs::read_to_string("/run/ouro/tagline").unwrap_or_default()
        }
    }
    // RDMA setup: `rdma-setup` (auto-detect wired iface) or
    // `rdma-setup <iface>` (explicit). Loads rdma_rxe via the
    // ouro-rdma-setup@ systemd template service (polkit authorizes
    // ouro to manage ouro-* services).
    else if trimmed == "rdma-setup" || trimmed.starts_with("rdma-setup ") {
        let iface = trimmed.strip_prefix("rdma-setup").unwrap_or("").trim();
        let iface = if iface.is_empty() {
            detect_wired_iface()
        } else {
            iface.to_string()
        };
        if iface.is_empty() {
            return "err no-wired-iface".into();
        }
        match std::process::Command::new("systemctl")
            .args(["start", &format!("ouro-rdma-setup@{iface}.service")])
            .output()
        {
            Ok(out) => {
                if out.status.success() {
                    // Read back the GID from sysfs.
                    let gid = std::fs::read_to_string(
                        "/sys/class/infiniband/rxe0/ports/1/gids/0",
                    )
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                    format!(
                        r#"{{"status":"ok","iface":"{iface}","device":"rxe0","gid":"{gid}"}}"#
                    )
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    format!(
                        r#"{{"status":"error","iface":"{iface}","stderr":"{stderr}"}}"#
                    )
                }
            }
            Err(e) => format!(r#"{{"status":"error","error":"{e}"}}"#),
        }
    }
    // Task execution
    else {
        match serde_json::from_str::<executor::Task>(trimmed) {
            Ok(task) => {
                TASK_BUSY.store(true, std::sync::atomic::Ordering::Relaxed);
                let result = executor::execute(&task);
                TASK_BUSY.store(false, std::sync::atomic::Ordering::Relaxed);
                match result {
                    Ok(result) => serde_json::to_string(&result).unwrap_or_else(|_| "{}".into()),
                    Err(e) => format!(r#"{{"error":"{}"}}"#, e),
                }
            }
            Err(e) => format!(r#"{{"error":"invalid task: {}"}}"#, e),
        }
    }
}

/// Detect the first wired Ethernet interface with a carrier (link up).
/// Skips lo, wlan*, tailscale*, docker*, virbr*, veth*.
fn detect_wired_iface() -> String {
    let Ok(entries) = std::fs::read_dir("/sys/class/net") else {
        return String::new();
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == "lo"
            || name.starts_with("wlan")
            || name.starts_with("tailscale")
            || name.starts_with("docker")
            || name.starts_with("virbr")
            || name.starts_with("veth")
        {
            continue;
        }
        // Check carrier (1 = link up).
        let carrier = std::fs::read_to_string(format!("/sys/class/net/{name}/carrier"))
            .map(|s| s.trim() == "1")
            .unwrap_or(false);
        if carrier {
            return name;
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ouro_cluster::transport::auth;

    const KEY: auth::Secret = [7u8; 32];

    #[test]
    fn test_process_ping() {
        assert_eq!(process_message("ping"), "pong");
    }

    #[test]
    fn test_authed_ping_roundtrip() {
        let line = auth::sign_line(&KEY, 5, "ping");
        let resp = authed_process(&KEY, &line).expect("valid line accepted");
        let (seq, body) = auth::open_line(&KEY, &resp).unwrap();
        assert_eq!(seq, 5);
        assert_eq!(body, "pong");
    }

    #[test]
    fn test_authed_rejects_tamper_wrong_key_garbage() {
        let tampered = auth::sign_line(&KEY, 1, "ping").replace("ping", "pins");
        assert!(authed_process(&KEY, &tampered).is_none());
        let other: auth::Secret = [8u8; 32];
        assert!(authed_process(&other, &auth::sign_line(&KEY, 1, "ping")).is_none());
        assert!(authed_process(&KEY, "1 deadbeef ping").is_none());
        assert!(authed_process(&KEY, "ping").is_none());
    }

    #[test]
    fn test_authed_task_roundtrip() {
        let task = r#"{"id":"t1","name":"echo","payload":"hi","estimated_watts":10,"estimated_seconds":1}"#;
        let line = auth::sign_line(&KEY, 9, task);
        let resp = authed_process(&KEY, &line).expect("valid task accepted");
        let (_, body) = auth::open_line(&KEY, &resp).unwrap();
        assert!(body.contains("hi"));
        assert!(body.contains("Success"));
    }

    #[test]
    fn test_serve_stdio_lockstep_roundtrip() {
        let task = r#"{"id":"x","name":"echo","payload":"body via stdio","estimated_watts":1,"estimated_seconds":1}"#;
        let input = format!(
            "{}\n{}\n",
            auth::sign_line(&KEY, 1, "ping"),
            auth::sign_line(&KEY, 2, task)
        );
        let mut out = Vec::new();
        serve_stdio(&KEY, std::io::Cursor::new(input), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let mut lines = text.lines();
        let (seq1, body1) = auth::open_line(&KEY, lines.next().unwrap()).unwrap();
        assert_eq!((seq1, body1), (1, "pong"));
        let (seq2, body2) = auth::open_line(&KEY, lines.next().unwrap()).unwrap();
        assert_eq!(seq2, 2);
        assert!(body2.contains("body via stdio"));
        assert!(body2.contains("Success"));
    }

    #[test]
    fn test_serve_stdio_auth_failure_terminates() {
        let input = "not a signed line\n".to_string();
        let mut out = Vec::new();
        serve_stdio(&KEY, std::io::Cursor::new(input), &mut out).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "err auth\n");
    }

    #[test]
    fn test_process_tagline() {
        std::env::set_var("OURO_TAGLINE", "devour the default.");
        assert_eq!(process_message("tagline"), "devour the default.");
        std::env::set_var("OURO_TAGLINE", "");
        assert_eq!(process_message("tagline"), "");
        std::env::remove_var("OURO_TAGLINE");
    }

    #[test]
    fn test_serve_stdio_eof_exits_cleanly() {
        let mut out = Vec::new();
        serve_stdio(&KEY, std::io::Cursor::new(""), &mut out).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "");
    }

    #[test]
    fn test_process_telemetry() {
        let resp = process_message("telemetry");
        assert!(resp.contains("hostname"));
    }

    #[test]
    fn test_process_task() {
        let task = r#"{"id":"t1","name":"echo","payload":"hi","estimated_watts":10,"estimated_seconds":1}"#;
        let resp = process_message(task);
        assert!(resp.contains("hi"));
        assert!(resp.contains("Success"));
    }

    #[test]
    fn test_process_invalid() {
        let resp = process_message("not json");
        assert!(resp.contains("error"));
    }
}
