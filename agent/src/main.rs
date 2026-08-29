mod executor;
mod telemetry;

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::broadcast;
use tokio::time::interval;

const DEFAULT_PORT: u16 = 9500;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Agent daemon entry point.
///
/// Listens for TCP connections from the master node.
/// Protocol: newline-delimited JSON.
#[tokio::main]
async fn main() -> Result<()> {
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
                            handle_connection(stream, peer, &mut rx).await;
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
/// Protocol: newline-delimited JSON messages.
/// - Client sends: `"telemetry"` → agent responds with Telemetry JSON
/// - Client sends: Task JSON → agent responds with TaskResult JSON
/// - Client sends: `"ping"` → agent responds with `"pong"`
async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer: SocketAddr,
    _rx: &mut broadcast::Receiver<()>,
) {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let response = process_message(&line).await;
        if writer.write_all(response.as_bytes()).await.is_err() {
            break;
        }
        if writer.write_all(b"\n").await.is_err() {
            break;
        }
    }

    println!("[disconnect] {}", peer);
}

/// Process a single message from the master.
async fn process_message(msg: &str) -> String {
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

    #[tokio::test]
    async fn test_process_ping() {
        assert_eq!(process_message("ping").await, "pong");
    }

    #[tokio::test]
    async fn test_process_telemetry() {
        let resp = process_message("telemetry").await;
        assert!(resp.contains("hostname"));
    }

    #[tokio::test]
    async fn test_process_task() {
        let task = r#"{"id":"t1","name":"echo","payload":"hi","estimated_watts":10,"estimated_seconds":1}"#;
        let resp = process_message(task).await;
        assert!(resp.contains("hi"));
        assert!(resp.contains("Success"));
    }

    #[tokio::test]
    async fn test_process_invalid() {
        let resp = process_message("not json").await;
        assert!(resp.contains("error"));
    }
}
