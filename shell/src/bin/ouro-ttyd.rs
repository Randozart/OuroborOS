//! `ouro-ttyd`: FIFO face daemon for one node (WP1/WP3, docs/R2_BRINGUP.md §3).
//!
//! Usage:
//!   `ouro-ttyd --node n1 --addr 127.0.0.1:9500 [--tty-dir /srv/ouro/tty]`
//!   `ouro-ttyd --node r2 --pty-cmd 'ssh -T r2@192.168.1.50 -- ouro-agent --stdio-tty'`
//!
//! Writes request lines to `<tty-dir>/<node>.in`, reads one response line
//! per request from `.out`. Line protocol: `ouro_hiss::ttyd` module docs.
//! Reconnects: when the tty writer closes the FIFO, the daemon re-arms and
//! waits for the next writer; session state (budget) persists. A dead
//! child wire (getty respawn, auth failure) is respawned on next use.

use std::path::PathBuf;

use anyhow::{bail, Result};

use ouro_cluster::transport::auth;
use ouro_hiss::ttyd::{ensure_fifo, fifo_paths, serve_connection, TtySession, TtyWire};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut node = String::new();
    let mut addr: Option<String> = None;
    let mut pty_cmd: Option<String> = None;
    let mut tty_dir = PathBuf::from("/srv/ouro/tty");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--node" => {
                i += 1;
                node = args
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("--node needs a value"))?;
            }
            "--addr" => {
                i += 1;
                addr = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--addr needs a value"))?,
                );
            }
            "--pty-cmd" => {
                i += 1;
                pty_cmd = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--pty-cmd needs a value"))?,
                );
            }
            "--tty-dir" => {
                i += 1;
                tty_dir = PathBuf::from(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--tty-dir needs a value"))?,
                );
            }
            other => bail!(
                "unknown arg {} (usage: ouro-ttyd --node <id> (--addr <ip:port> | --pty-cmd '<cmd>') [--tty-dir <dir>])",
                other
            ),
        }
        i += 1;
    }
    if node.is_empty() {
        bail!("usage: ouro-ttyd --node <id> (--addr <ip:port> | --pty-cmd '<cmd>') [--tty-dir <dir>]");
    }
    let wire = match (addr, pty_cmd) {
        (Some(a), None) => TtyWire::Tcp(a),
        (None, Some(c)) => TtyWire::Child(c),
        _ => bail!("give exactly one of --addr or --pty-cmd"),
    };

    let secret = auth::secret_from_env()?;
    println!("auth: OURO_SECRET_FILE loaded (32B HMAC-SHA256)");
    std::fs::create_dir_all(&tty_dir)?;
    let (in_path, out_path) = fifo_paths(&tty_dir, &node);
    ensure_fifo(&in_path)?;
    ensure_fifo(&out_path)?;
    let wire_desc = match &wire {
        TtyWire::Tcp(a) => format!("tcp={}", a),
        TtyWire::Child(c) => format!("pty-cmd={}", c),
    };
    println!(
        "ouro-ttyd node={} wire={} in={} out={}",
        node,
        wire_desc,
        in_path.display(),
        out_path.display()
    );

    let mut session = TtySession::new(&node, wire, secret);
    loop {
        if let Err(e) = serve_connection(&mut session, &in_path, &out_path) {
            eprintln!("[ttyd] connection ended: {}", e);
        }
    }
}
