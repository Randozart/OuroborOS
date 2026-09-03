//! Head link: the agent-side push client for the registry bus.
//!
//! One signed line per exchange, one connection per exchange — the
//! daemon is stateless per connection. On boot: `register` (full
//! telemetry, daemon assigns/keeps our slot by peer IP). Then
//! `heartbeat` every `period`. `unknown` reply (daemon lost state) or
//! any transport error drops back to re-register.

use std::time::Duration;

use anyhow::{Context, Result};
use ouro_cluster::transport::auth::{self, Secret};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::time::sleep;

use crate::telemetry;

const RETRY_BACKOFF: Duration = Duration::from_secs(5);

/// Build the `register`/`heartbeat` request body for this node.
pub fn request_body(verb: &str) -> Result<String> {
    let tel = telemetry::collect()?;
    let json = serde_json::to_string(&tel)?;
    Ok(format!("{} {}", verb, json))
}

/// One signed line out, one signed line back, connection closed.
async fn exchange(secret: &Secret, head: &str, seq: u64, body: &str) -> Result<String> {
    let stream = TcpStream::connect(head)
        .await
        .with_context(|| format!("connect registry daemon {}", head))?;
    stream.set_nodelay(true).ok();
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    writer
        .write_all(auth::sign_line(secret, seq, body).as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let (rseq, resp) = auth::open_line(secret, line.trim())
        .context("unauthenticated registry reply")?;
    if rseq != seq {
        anyhow::bail!("reply seq {} != request {}", rseq, seq);
    }
    Ok(resp.to_string())
}

/// Top-level link loop: register, then heartbeat forever. Re-registers
/// on daemon state loss; backs off on transport errors. Never returns
/// under normal operation.
pub async fn run(secret: Secret, head: String, period: Duration) -> Result<()> {
    let mut seq: u64 = 1;
    loop {
        // Register pass.
        let body = match request_body("register") {
            Ok(b) => b,
            Err(e) => {
                eprintln!("head-link: telemetry collect failed: {} — retry in {}s", e, RETRY_BACKOFF.as_secs());
                sleep(RETRY_BACKOFF).await;
                continue;
            }
        };
        match exchange(&secret, &head, seq, &body).await {
            Ok(resp) if resp.starts_with("registered") => {
                let id = resp.split_whitespace().nth(1).unwrap_or("?").to_string();
                eprintln!("head-link: registered as {} @ {}", id, head);
            }
            Ok(resp) => {
                eprintln!("head-link: register refused: {} — retry", resp);
                sleep(RETRY_BACKOFF).await;
                continue;
            }
            Err(e) => {
                eprintln!("head-link: {} — retry in {}s", e, RETRY_BACKOFF.as_secs());
                sleep(RETRY_BACKOFF).await;
                continue;
            }
        }
        seq = seq.wrapping_add(1);

        // Heartbeat passes until the daemon forgets us or the wire breaks.
        loop {
            sleep(period).await;
            let body = match request_body("heartbeat") {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("head-link: telemetry collect failed: {}", e);
                    continue;
                }
            };
            match exchange(&secret, &head, seq, &body).await {
                Ok(resp) if resp.starts_with("ok") => {}
                Ok(resp) if resp.starts_with("unknown") => {
                    eprintln!("head-link: daemon lost our registration; re-registering");
                    break;
                }
                Ok(resp) => {
                    eprintln!("head-link: bad heartbeat reply: {} — re-registering", resp);
                    break;
                }
                Err(e) => {
                    eprintln!("head-link: {} — re-registering", e);
                    break;
                }
            }
            seq = seq.wrapping_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_body_shape() {
        let body = request_body("heartbeat").unwrap();
        assert!(body.starts_with("heartbeat {"), "got: {}", &body[..40.min(body.len())]);
        assert!(body.contains("\"hostname\""));
        // Round-trips as the bus telemetry struct.
        let json = body.strip_prefix("heartbeat ").unwrap();
        let tel: ouro_cluster::registry::bus::BusTelemetry = serde_json::from_str(json).unwrap();
        assert!(!tel.hostname.is_empty());
    }
}
