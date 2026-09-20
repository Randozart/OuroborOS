//! ouro-bmts — BMTS shard inspection and DUET identity tooling.
//!
//!   ouro-bmts info <shard.bmts>            node, tensors, epoch
//!   ouro-bmts verify <shard.bmts>          epoch contract check
//!   ouro-bmts upgrade <shard.bmts> [out]   v1 (or bare v2) → v2 with epoch
//!                                          + advisory chunk index
//!   ouro-bmts plan <local> <remote>        dry-run delta-pull wire bill
//!
//! The sharder (tools/shard_model.py) emits v1; `upgrade` stamps DUET
//! identity without a Python capnp dependency.

use anyhow::Context;
use ouro_cluster::bmts::BmtsShard;
use ouro_cluster::sync::plan_diff;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(cmd) = args.get(1) else {
        eprintln!("usage: ouro-bmts <info|verify|upgrade|plan> <shard> [args]");
        std::process::exit(2);
    };
    let Some(path) = args.get(2) else {
        eprintln!("missing shard path");
        std::process::exit(2);
    };
    let r = match cmd.as_str() {
        "info" => info(path),
        "verify" => verify(path),
        "upgrade" => upgrade(path, args.get(3).map(|s| s.as_str())),
        "plan" => plan(path, args.get(3).map(|s| s.as_str())),
        other => {
            eprintln!("unknown command {other}");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("ouro-bmts: {e:#}");
        std::process::exit(1);
    }
}

fn info(path: &str) -> anyhow::Result<()> {
    let shard = BmtsShard::open(path)?;
    println!("node {}", shard.node);
    println!("tensors {}", shard.tensors.len());
    println!("data {} bytes", shard.data_len());
    println!(
        "epoch {}",
        if shard.epoch.is_empty() { "(none — bare v2/v1)" } else { &shard.epoch }
    );
    Ok(())
}

fn verify(path: &str) -> anyhow::Result<()> {
    let shard = BmtsShard::open(path)?;
    if shard.epoch.is_empty() {
        println!("no epoch declared — nothing to verify");
        return Ok(());
    }
    if shard.verify_epoch() {
        println!("OK epoch {}", shard.epoch);
    } else {
        anyhow::bail!("epoch MISMATCH — shard content does not match its claim");
    }
    Ok(())
}

fn upgrade(path: &str, out: Option<&str>) -> anyhow::Result<()> {
    use std::io::{Seek, Write};
    let shard = BmtsShard::open(path)?;
    if !shard.epoch.is_empty() && !shard.verify_epoch() {
        anyhow::bail!("refusing to upgrade a shard failing its own epoch");
    }
    let out_path = out
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}.v2", path));

    // Stream identity + copy from the SOURCE FILE in bounded buffers —
    // never walks the whole mmap at once (gentle under memory pressure).
    let mut src = std::fs::File::open(path)?;
    src.seek(std::io::SeekFrom::Start(shard.data_start))?;
    let (epoch, hashes) =
        ouro_cluster::bmts::identity_of_stream(&mut src, ouro_cluster::bmts::DUET_CHUNK_SIZE)?;
    src.seek(std::io::SeekFrom::Start(shard.data_start))?;

    let meta = ouro_cluster::bmts::build_meta_v2(shard.node, &shard.tensors, &epoch, &hashes)?;

    let stage = format!("{out_path}.duet-staging");
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage)
        .with_context(|| format!("staging {stage} (exists? another upgrade in flight?)"))?;
    let mut dst = std::io::BufWriter::new(f);
    dst.write_all(&ouro_cluster::bmts::BMTS_MAGIC.to_le_bytes())?;
    dst.write_all(&ouro_cluster::bmts::BMTS_VERSION_V2.to_le_bytes())?;
    dst.write_all(&shard.node.to_le_bytes())?;
    dst.write_all(&(shard.tensors.len() as u32).to_le_bytes())?;
    dst.write_all(&(meta.len() as u32).to_le_bytes())?;
    dst.write_all(&meta)?;
    std::io::copy(&mut src, &mut dst)?;
    dst.flush()?;
    dst.into_inner()?.sync_all()?;
    std::fs::rename(&stage, &out_path)?;
    println!("wrote {} (epoch {}...)", out_path, &epoch[..16]);
    Ok(())
}

/// Dry-run: how many bytes would a delta pull from `remote` cost?
/// Both shards must be same-shape (same tensor table) — the usual
/// stale-replica vs fresh-replica case.
fn plan(path: &str, other: Option<&str>) -> anyhow::Result<()> {
    let Some(other) = other else {
        anyhow::bail!("plan needs a second shard: ouro-bmts plan <local> <remote>");
    };
    let local = BmtsShard::open(path)?;
    let remote = BmtsShard::open(other)?;
    let requests = plan_diff(local.data_bytes(), remote.data_bytes())?;
    let bill = ouro_cluster::sync::wire_bytes(&requests);
    let total = remote.data_bytes().len() as u64;
    println!(
        "chunks requested: {} / {}; wire: {} / {} bytes ({:.2}% of full push)",
        requests.len(),
        remote.data_bytes().len().div_ceil(64 * 1024),
        bill,
        total,
        100.0 * bill as f64 / total as f64
    );
    Ok(())
}
