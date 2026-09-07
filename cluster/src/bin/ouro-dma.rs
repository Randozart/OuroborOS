//! ouro-dma — SoftRoCE proof-of-concept.
//!
//! Tier 4 of the DMA ladder: the head reads/writes tail RAM via
//! registered memory regions and ibverbs.
//!
//! Usage:
//!   ouro-dma server  [--port 9600]     — register buffer, accept RDMA
//!   ouro-dma client  --addr IP [--port 9600] [--size 4096] [--iters 1000]
//!   ouro-dma bench   --addr IP [--port 9600] [--size 4096] [--iters 1000]
//!
//! The QP info exchange happens over TCP (control plane); the data
//! plane is pure RDMA — zero-copy, registered memory only.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

use anyhow::{Context, Result};
use ouro_cluster::transport::dma::{MemoryRegion, QpInfo, RdmaDevice};
use ouro_cluster::transport::rdma_ffi as ib;

const DEFAULT_PORT: u16 = 9600;
const DEFAULT_SIZE: usize = 4096;
const DEFAULT_ITERS: usize = 1000;
const IBV_ACCESS: i32 =
    ib::IBV_ACCESS_LOCAL_WRITE | ib::IBV_ACCESS_REMOTE_READ | ib::IBV_ACCESS_REMOTE_WRITE;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ouro-dma <server|client|bench> [options]");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "server" => cmd_server(&args[2..]),
        "client" => cmd_client(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        other => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: ouro-dma <server|client|bench> [options]");
            std::process::exit(1);
        }
    }
}

/// Parse `--key value` from args. Returns None if missing.
fn opt(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
}

fn opt_or(args: &[String], key: &str, default: &str) -> String {
    opt(args, key).unwrap_or_else(|| default.to_string())
}

fn opt_or_usize(args: &[String], key: &str, default: usize) -> usize {
    opt(args, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn cmd_server(args: &[String]) -> Result<()> {
    let port: u16 = opt_or(args, "--port", &DEFAULT_PORT.to_string())
        .parse()
        .context("invalid port")?;
    let buf_size = opt_or_usize(args, "--size", 64 * 1024 * 1024); // 64MiB default for server

    println!("[dma-server] opening RDMA device...");
    let dev = RdmaDevice::open().context("open RDMA device")?;
    let attr = dev.query_device()?;
    println!(
        "[dma-server] device ok — max_qp={}, max_mr={}",
        attr.max_qp, attr.max_mr
    );

    let cq = dev.create_cq(128)?;
    let qp = cq.create_qp(dev.pd(), dev.ctx(), 1)?;
    let info = qp.local_info(1)?;
    println!(
        "[dma-server] QP ready — qp_num={}, lid={:#x}",
        info.qp_num, info.lid
    );

    // Register the server-side buffer (will be the target of RDMA reads/writes).
    let mut server_buf = vec![0u8; buf_size];
    // Stamp a recognizable pattern so we can verify reads.
    for (i, b) in server_buf.iter_mut().enumerate() {
        *b = (i & 0xFF) as u8;
    }
    let server_mr = MemoryRegion::new(dev.pd(), server_buf, IBV_ACCESS)?;
    println!(
        "[dma-server] MR registered — addr={:#x}, len={}MiB, rkey={}",
        server_mr.local_addr(),
        buf_size / (1024 * 1024),
        server_mr.rkey(),
    );

    // Listen for TCP control connection (QP info exchange).
    let listener = TcpListener::bind(format!("0.0.0.0:{port}"))?;
    println!("[dma-server] listening on TCP {port} for QP exchange...");

    for stream in listener.incoming() {
        let mut stream = stream.context("TCP accept")?;
        println!("[dma-server] peer connected from {}", stream.peer_addr()?);

        // Send our QP info (bincode-like: raw struct bytes).
        let info_bytes = unsafe {
            std::slice::from_raw_parts(
                &info as *const QpInfo as *const u8,
                std::mem::size_of::<QpInfo>(),
            )
        };
        stream.write_all(info_bytes)?;

        // Receive peer's QP info.
        let mut peer_buf = vec![0u8; std::mem::size_of::<QpInfo>()];
        stream.read_exact(&mut peer_buf)?;
        let peer_info: QpInfo = unsafe { std::ptr::read_unaligned(peer_buf.as_ptr() as *const QpInfo) };
        println!(
            "[dma-server] peer QP — qp_num={}, lid={:#x}",
            peer_info.qp_num, peer_info.lid
        );

        // Transition QP through the state machine.
        qp.to_init(1)?;
        qp.to_rtr(&peer_info)?;
        qp.to_rts()?;
        println!("[dma-server] QP → RTS — data plane ready");

        // Send MR info to the client (addr + rkey for RDMA operations).
        let mr_info: [u8; 12] = {
            let addr = server_mr.local_addr().to_le_bytes();
            let rkey = server_mr.rkey().to_le_bytes();
            let mut out = [0u8; 12];
            out[..8].copy_from_slice(&addr);
            out[8..12].copy_from_slice(&rkey);
            out
        };
        stream.write_all(&mr_info)?;

        println!("[dma-server] MR info sent — waiting for RDMA operations...");
        println!("[dma-server] (Ctrl+C to stop)");

        // The server stays alive — the client does RDMA ops against our MR.
        // We just keep the connection open so the TCP stream acts as keepalive.
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
            // Check if client is still alive.
            match stream.write(&[]) {
                Ok(_) => {}
                Err(_) => {
                    println!("[dma-server] peer disconnected — waiting for next...");
                    break;
                }
            }
        }
    }

    Ok(())
}

fn cmd_client(args: &[String]) -> Result<()> {
    let addr = opt(args, "--addr").context("usage: ouro-dma client --addr IP [--port 9600]")?;
    let port: u16 = opt_or(args, "--port", &DEFAULT_PORT.to_string())
        .parse()
        .context("invalid port")?;
    let size = opt_or_usize(args, "--size", DEFAULT_SIZE);
    let iters = opt_or_usize(args, "--iters", DEFAULT_ITERS);

    println!("[dma-client] opening RDMA device...");
    let dev = RdmaDevice::open()?;
    let cq = dev.create_cq(128)?;
    let qp = cq.create_qp(dev.pd(), dev.ctx(), 1)?;
    let info = qp.local_info(1)?;
    println!(
        "[dma-client] QP ready — qp_num={}, lid={:#x}",
        info.qp_num, info.lid
    );

    // Connect via TCP for QP exchange.
    let mut stream = TcpStream::connect(format!("{addr}:{port}"))?;
    println!("[dma-client] TCP connected to {addr}:{port}");

    // Exchange QP info.
    let info_bytes = unsafe {
        std::slice::from_raw_parts(
            &info as *const QpInfo as *const u8,
            std::mem::size_of::<QpInfo>(),
        )
    };
    stream.write_all(info_bytes)?;

    let mut peer_buf = vec![0u8; std::mem::size_of::<QpInfo>()];
    stream.read_exact(&mut peer_buf)?;
    let peer_info: QpInfo = unsafe { std::ptr::read_unaligned(peer_buf.as_ptr() as *const QpInfo) };
    println!(
        "[dma-client] peer QP — qp_num={}, lid={:#x}",
        peer_info.qp_num, peer_info.lid
    );

    // Transition QP.
    qp.to_init(1)?;
    qp.to_rtr(&peer_info)?;
    qp.to_rts()?;
    println!("[dma-client] QP → RTS");

    // Receive MR info from server.
    let mut mr_buf = [0u8; 12];
    stream.read_exact(&mut mr_buf)?;
    let remote_addr = u64::from_le_bytes(mr_buf[..8].try_into().unwrap());
    let remote_rkey = u32::from_le_bytes(mr_buf[8..12].try_into().unwrap());
    println!(
        "[dma-client] remote MR — addr={remote_addr:#x}, rkey={remote_rkey}"
    );

    // Register a local buffer for receiving RDMA reads.
    let local_buf = vec![0xABu8; size];
    let local_mr = MemoryRegion::new(dev.pd(), local_buf, IBV_ACCESS)?;
    println!(
        "[dma-client] local MR — addr={:#x}, len={size}",
        local_mr.local_addr()
    );

    println!("[dma-client] running RDMA_READ ×{iters} ({size} bytes each)...");
    let mut total_bytes: u64 = 0;
    let start = Instant::now();

    for i in 0..iters {
        qp.post_rdma_read(
            &local_mr,
            0,
            remote_addr,
            remote_rkey,
            size as u32,
            i as u64,
        )?;
        let (_wr_id, _byte_len) = qp.poll_cq(&cq)?;
        total_bytes += size as u64;
    }

    let elapsed = start.elapsed();
    let throughput_mbps = (total_bytes as f64 / elapsed.as_secs_f64()) / (1024.0 * 1024.0);
    let latency_us = elapsed.as_micros() as f64 / iters as f64;

    println!("[dma-client] ─── RESULTS ───");
    println!(
        "[dma-client] RDMA_READ: {iters} iters × {size} bytes in {:.3}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "[dma-client] throughput: {throughput_mbps:.1} MiB/s, latency: {latency_us:.1} µs/call"
    );

    // Verify first few bytes.
    let first = local_mr.buf()[0];
    println!("[dma-client] first byte = 0x{first:02X} (expected 0x00)");
    if first == 0x00 {
        println!("[dma-client] ✓ data verified — RDMA_READ succeeded");
    } else {
        println!("[dma-client] ✗ unexpected data — check access flags / rkey");
    }

    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<()> {
    let addr = opt(args, "--addr").context("usage: ouro-dma bench --addr IP")?;
    let port: u16 = opt_or(args, "--port", &DEFAULT_PORT.to_string())
        .parse()
        .context("invalid port")?;
    let max_size = opt_or_usize(args, "--size", 1024 * 1024); // 1MiB default
    let iters = opt_or_usize(args, "--iters", DEFAULT_ITERS);

    println!("[dma-bench] opening RDMA device...");
    let dev = RdmaDevice::open()?;
    let cq = dev.create_cq(256)?;
    let qp = cq.create_qp(dev.pd(), dev.ctx(), 1)?;
    let info = qp.local_info(1)?;

    let mut stream = TcpStream::connect(format!("{addr}:{port}"))?;
    let info_bytes = unsafe {
        std::slice::from_raw_parts(
            &info as *const QpInfo as *const u8,
            std::mem::size_of::<QpInfo>(),
        )
    };
    stream.write_all(info_bytes)?;

    let mut peer_buf = vec![0u8; std::mem::size_of::<QpInfo>()];
    stream.read_exact(&mut peer_buf)?;
    let peer_info: QpInfo = unsafe { std::ptr::read_unaligned(peer_buf.as_ptr() as *const QpInfo) };

    qp.to_init(1)?;
    qp.to_rtr(&peer_info)?;
    qp.to_rts()?;

    let mut mr_buf = [0u8; 12];
    stream.read_exact(&mut mr_buf)?;
    let remote_addr = u64::from_le_bytes(mr_buf[..8].try_into().unwrap());
    let remote_rkey = u32::from_le_bytes(mr_buf[8..12].try_into().unwrap());

    println!("[dma-bench] connected — running sweep 64B → {max_size} bytes, {iters} iters each");
    println!();
    println!("{:>12} {:>12} {:>12} {:>12}", "size", "throughput", "latency", "bandwidth");
    println!("{:>12} {:>12} {:>12} {:>12}", "─", "─", "─", "─");

    // Sweep power-of-two sizes.
    let mut size = 64usize;
    while size <= max_size {
        let local_buf = vec![0xABu8; size];
        let local_mr = MemoryRegion::new(dev.pd(), local_buf, IBV_ACCESS)?;

        // Warm up.
        for i in 0..3.min(iters) {
            qp.post_rdma_read(&local_mr, 0, remote_addr, remote_rkey, size as u32, i as u64)?;
            let _ = qp.poll_cq(&cq)?;
        }

        let start = Instant::now();
        for i in 0..iters {
            qp.post_rdma_read(
                &local_mr,
                0,
                remote_addr,
                remote_rkey,
                size as u32,
                i as u64,
            )?;
            let _ = qp.poll_cq(&cq)?;
        }
        let elapsed = start.elapsed();

        let total = (size * iters) as f64;
        let throughput_mib = (total / elapsed.as_secs_f64()) / (1024.0 * 1024.0);
        let latency_us = elapsed.as_micros() as f64 / iters as f64;
        let bandwidth_gbps = (total * 8.0 / elapsed.as_secs_f64()) / 1_000_000_000.0;

        println!(
            "{:>12} {:>11.1}MiB/s {:>10.1}µs {:>11.2}Gbps",
            fmt_size(size), throughput_mib, latency_us, bandwidth_gbps
        );

        size *= 2;
    }

    println!();
    println!("[dma-bench] sweep complete.");

    Ok(())
}

fn fmt_size(n: usize) -> String {
    if n >= 1024 * 1024 {
        format!("{}MiB", n / (1024 * 1024))
    } else if n >= 1024 {
        format!("{}KiB", n / 1024)
    } else {
        format!("{n}B")
    }
}
