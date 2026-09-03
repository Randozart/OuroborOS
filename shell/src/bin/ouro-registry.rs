//! ouro-registry — the cluster's push-based registry daemon.
//!
//! Agents connect (signed line protocol, same wire as the task channel):
//! `register <telemetry-json>` on boot, `heartbeat <telemetry-json>`
//! every few seconds. State lives in a Registry; optionally persisted
//! to disk. The IP always comes from the socket peer, never from the
//! agent's own report.
//!
//! Env: OURO_SECRET_FILE (mandatory — no secret, no daemon).
//! Args: --addr <ip:port> (default 0.0.0.0:9501)
//!       --state <path>   (persist registry JSON across restarts)

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use ouro_cluster::error_recovery::ErrorRecovery;
use ouro_cluster::registry::{bus, Registry};
use ouro_cluster::transport::auth::{self, Secret};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

const DEFAULT_ADDR: &str = "0.0.0.0:9501";

type Shared = Arc<Mutex<(Registry, ErrorRecovery)>>;

#[tokio::main]
async fn main() -> Result<()> {
    let secret = auth::secret_from_env()?;

    let mut addr = DEFAULT_ADDR.to_string();
    let mut state_path: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    for (i, a) in args.iter().enumerate() {
        if a == "--addr" {
            if let Some(v) = args.get(i + 1) {
                addr = v.clone();
            }
        }
        if a == "--state" {
            if let Some(v) = args.get(i + 1) {
                state_path = Some(v.clone());
            }
        }
    }

    let registry = match &state_path {
        Some(p) => Registry::load(std::path::Path::new(p)),
        None => Registry::new(),
    };
    let shared: Shared = Arc::new(Mutex::new((registry, ErrorRecovery::new())));

    let listener = TcpListener::bind(&addr).await?;
    println!("ouro-registry listening on {}", addr);
    if let Some(p) = &state_path {
        println!("  state: {}", p);
    }

    // Periodic sweep: nodes silent past the heartbeat window go offline;
    // their queued work can be retried elsewhere (Art. 6 — no hoarding).
    {
        let shared = shared.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(10));
            loop {
                ticker.tick().await;
                if let Ok(mut guard) = shared.lock() {
                    let (reg, _rec) = &mut *guard;
                    let stale = reg.offline_nodes().len();
                    if stale > 0 {
                        println!("sweep: {} node(s) past heartbeat window", stale);
                    }
                }
            }
        });
    }

    loop {
        let (stream, peer) = listener.accept().await?;
        let shared = shared.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_connection(stream, peer.to_string(), shared, &secret).await {
                eprintln!("{}: connection error: {}", peer, e);
            }
        });
    }
}

/// One signed line in, one signed line out. The connection closes after
/// the reply — heartbeat cadence reconnects, which keeps the daemon
/// stateless per-connection (no half-open bookkeeping).
async fn serve_connection(
    stream: TcpStream,
    peer: String,
    shared: Shared,
    secret: &Secret,
) -> Result<()> {
    stream.set_nodelay(true).ok();
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }
    let (seq, body) = auth::open_line(secret, line.trim())?;

    let peer_ip = peer
        .rsplit_once(':')
        .map(|(ip, _)| ip.to_string())
        .unwrap_or(peer.clone());

    let resp = {
        let mut guard = shared
            .lock()
            .map_err(|_| anyhow::anyhow!("registry state poisoned"))?;
        let (reg, rec) = &mut *guard;
        bus::handle_bus_message(reg, rec, &peer_ip, body)
    };

    writer
        .write_all(auth::sign_line(secret, seq, &resp).as_bytes())
        .await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}
