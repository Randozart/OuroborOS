mod executor;
mod stage;
mod telemetry;

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;
use ouro_cluster::transport::auth::{self, Secret};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::broadcast;
use tokio::time::interval;

const DEFAULT_PORT: u16 = 9500;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Agent daemon entry point.
///
/// Default mode: TCP daemon for the master node, HMAC-authenticated
/// newline protocol (`seq tag body`); refuses to start without
/// OURO_SECRET_FILE.
///
/// `ouro-agent --stdio-tty` (getty-shim, WP3): same authed protocol on
/// stdin/stdout instead of TCP. A slave's getty line spawns it; the
/// master's ouro-ttyd connects via `ssh -T` (or raw serial). Zero
/// install: any booted Linux with a login joins the graph.
#[tokio::main]
async fn main() -> Result<()> {
    let secret = auth::secret_from_env()?;

    if std::env::args().any(|a| a == "--stdio-tty") {
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

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await?;

    println!("ouro-agent listening on {}", addr);
    println!("  telemetry: collect on connect");
    println!("  execute:   send JSON task, get JSON result");
    println!("  heartbeat: every {}s", HEARTBEAT_INTERVAL.as_secs());

    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let shutdown_rx = shutdown_tx.clone();

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

    // Accept connections
    let mut shutdown_rx_main = shutdown_tx.subscribe();
    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, peer)) => {
                        println!("[connect] {}", peer);
                        let mut rx = shutdown_tx.subscribe();
                        tokio::spawn(async move {
                            handle_connection(secret, stream, peer, &mut rx).await;
                        });
                    }
                    Err(e) => {
                        eprintln!("[error] accept: {}", e);
                    }
                }
            }
            _ = shutdown_rx_main.recv() => {
                println!("[agent] shutting down listener");
                break;
            }
        }
    }

    // Wait for background tasks
    let _ = tokio::join!(heartbeat_handle, ctrl_c_handle);
    println!("ouro-agent stopped.");
    Ok(())
}

/// Handle a single TCP connection.
///
/// Protocol: HMAC-authenticated newline-delimited messages
/// (`<seq> <hex-tag> <body>`). Every valid request gets a signed
/// response under the same seq. Auth failure → terse unsigned `err auth`,
/// connection closed (no oracle).
async fn handle_connection(
    secret: Secret,
    stream: tokio::net::TcpStream,
    peer: SocketAddr,
    _rx: &mut broadcast::Receiver<()>,
) {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        match authed_process(&secret, &line) {
            Some(response) => {
                if writer.write_all(response.as_bytes()).await.is_err() {
                    break;
                }
                if writer.write_all(b"\n").await.is_err() {
                    break;
                }
            }
            None => {
                let _ = writer.write_all(b"err auth\n").await;
                break;
            }
        }
    }

    println!("[disconnect] {}", peer);
}

/// Verify + process one line, produce one signed response line.
/// `None` = auth failure.
fn authed_process(secret: &Secret, line: &str) -> Option<String> {
    let (seq, body) = auth::open_line(secret, line).ok()?;
    let response = process_message(body);
    Some(auth::sign_line(secret, seq, &response))
}

/// Getty-shim loop: signed line in from stdin, signed line out on
/// stdout, flush per line. Auth failure → `err auth`, stop (the spawning
/// getty respawns = fresh login). EOF → clean exit.
fn serve_stdio<R: std::io::BufRead, W: std::io::Write>(
    secret: &Secret,
    mut input: R,
    mut output: W,
) -> Result<()> {
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

/// Process a single message from the master.
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
    // Tagline: this boot's motto, for the master's registration echo
    else if trimmed == "tagline" {
        let from_env = std::env::var("OURO_TAGLINE").unwrap_or_default();
        if !from_env.trim().is_empty() {
            from_env
        } else {
            std::fs::read_to_string("/run/ouro/tagline").unwrap_or_default()
        }
    }
    // Task execution
    else {
        match serde_json::from_str::<executor::Task>(trimmed) {
            Ok(task) => match executor::execute(&task) {
                Ok(result) => serde_json::to_string(&result).unwrap_or_else(|_| "{}".into()),
                Err(e) => format!(r#"{{"error":"{}"}}"#, e),
            },
            Err(e) => format!(r#"{{"error":"invalid task: {}"}}"#, e),
        }
    }
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
