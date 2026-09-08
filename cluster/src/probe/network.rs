use anyhow::Result;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use crate::probe::NetworkInfo;
use crate::transport::edge::{EdgeKind, PricedEdge};

/// Measure ICMP latency to a remote node via ping.
pub fn measure_latency(ip: &str) -> Result<f64> {
    let output = std::process::Command::new("ping")
        .args(["-c", "3", "-q", ip])
        .output()?;

    let stdout = String::from_utf8(output.stdout)?;

    for line in stdout.lines() {
        if line.contains("rtt") || line.contains("round-trip") {
            if let Some(stats) = line.split('=').nth(1) {
                if let Some(avg_str) = stats.split('/').nth(1) {
                    if let Ok(avg) = avg_str.parse::<f64>() {
                        return Ok(avg);
                    }
                }
            }
        }
    }

    Ok(f64::MAX)
}

/// Measure TCP round-trip time to an agent (ping/pong).
pub fn measure_tcp_latency(addr: &str) -> Result<f64> {
    let stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;

    let start = Instant::now();
    let mut writer = stream.try_clone()?;
    writer.write_all(b"ping\n")?;
    writer.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader.read_line(&mut response)?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    if response.trim() == "pong" {
        Ok(elapsed_ms)
    } else {
        Ok(f64::MAX)
    }
}

/// Measure bandwidth to a remote node via dd over SSH.
pub fn measure_bandwidth(ip: &str) -> Result<f64> {
    let output = std::process::Command::new("sh")
        .args([
            "-c",
            &format!(
                "dd if=/dev/zero bs=1M count=100 2>/dev/null | ssh -o BatchMode=yes {} \"cat /dev/null\" 2>&1",
                ip
            ),
        ])
        .output()?;

    let stderr = String::from_utf8(output.stderr)?;

    for line in stderr.lines() {
        if line.contains("MB/s") {
            if let Some(mbps_str) = line.split_whitespace().rev().nth(1) {
                if let Ok(mbps) = mbps_str.parse::<f64>() {
                    return Ok(mbps * 8.0);
                }
            }
        }
    }

    Ok(0.0)
}

/// Probe network info to a remote node.
pub fn probe_remote(ip: &str) -> Result<NetworkInfo> {
    let latency_ms = measure_latency(ip)?;
    let bandwidth_mbps = measure_bandwidth(ip)?;

    Ok(NetworkInfo {
        latency_ms,
        bandwidth_mbps,
    })
}

/// Probe TCP latency to an agent address.
pub fn probe_agent(addr: &str) -> Result<NetworkInfo> {
    let latency_ms = measure_tcp_latency(addr)?;

    Ok(NetworkInfo {
        latency_ms,
        bandwidth_mbps: 0.0,
    })
}

/// Enumerate and price every live lane on this node (docs/AIR_PATH.md §1.1,
/// §4.1). Reads `/sys/class/net`, `iw` for air signal/bitrate, and a
/// per-interface ping to `target` for RTT/jitter. Best-effort: a lane that
/// cannot be measured is still listed with zeros, never dropped silently.
pub fn probe_lanes(target: Option<&str>) -> Vec<PricedEdge> {
    let mut edges = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/net") else {
        return edges;
    };
    let mut ifaces: Vec<String> = entries
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
        .filter(|n| n != "lo")
        .collect();
    ifaces.sort();

    for iface in ifaces {
        let kind = kind_from_name(&iface);
        let link_mbps = read_link_speed(&iface);
        let (signal_dbm, air_mbps) = if kind == EdgeKind::Air {
            iw_link(&iface)
        } else {
            (0, 0)
        };
        let bw_mbps = if air_mbps > 0 { air_mbps } else { link_mbps as u32 };
        let (latency_us, jitter_us) = match target {
            Some(t) => ping_lane(&iface, t),
            None => (0, 0),
        };
        let watts = kind_watts(&kind);
        edges.push(PricedEdge {
            iface,
            kind,
            bw_mbps,
            latency_us,
            jitter_us,
            watts,
            signal_dbm,
        });
    }
    edges
}

/// Classify an interface name into a lane kind.
fn kind_from_name(iface: &str) -> EdgeKind {
    if iface.starts_with("wlan") || iface.starts_with("wifi") || iface.starts_with("wl") {
        EdgeKind::Air
    } else if iface.starts_with("rxe") || iface.starts_with("rdma") || iface.starts_with("ib") {
        EdgeKind::Rdma
    } else {
        EdgeKind::Tcp
    }
}

/// Read the negotiated link speed (Mbit/s) from `/sys/class/net/<iface>/speed`.
fn read_link_speed(iface: &str) -> u64 {
    std::fs::read_to_string(format!("/sys/class/net/{}/speed", iface))
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .filter(|s| *s > 0)
        .map(|s| s as u64)
        .unwrap_or(0)
}

/// Parse `iw dev <iface> link`: `(signal_dbm, tx_bitrate_mbps)`.
fn iw_link(iface: &str) -> (i32, u32) {
    let Ok(output) = std::process::Command::new("iw")
        .args(["dev", iface, "link"])
        .output()
    else {
        return (0, 0);
    };
    let text = String::from_utf8_lossy(&output.stdout);
    (parse_signal_dbm(&text), parse_tx_bitrate_mbps(&text))
}

fn parse_signal_dbm(text: &str) -> i32 {
    text.lines()
        .find_map(|l| {
            let t = l.trim();
            let rest = t.strip_prefix("signal:")?;
            let v: i32 = rest.split_whitespace().next()?.parse().ok()?;
            Some(v)
        })
        .unwrap_or(0)
}

fn parse_tx_bitrate_mbps(text: &str) -> u32 {
    text.lines().find_map(|l| {
        let t = l.trim();
        let rest = t.strip_prefix("tx bitrate:")?;
        let mut parts = rest.split_whitespace();
        let v: f64 = parts.next()?.parse().ok()?;
        let unit = parts.next().unwrap_or("");
        let mbit = if unit.starts_with("MB/s") { v * 8.0 } else { v };
        Some(mbit as u32)
    }).unwrap_or(0)
}

/// Per-interface ping against the head: `(latency_us, jitter_us)`.
fn ping_lane(iface: &str, target: &str) -> (u32, u32) {
    let Ok(output) = std::process::Command::new("ping")
        .args(["-I", iface, "-c", "3", "-q", target])
        .output()
    else {
        return (0, 0);
    };
    let text = String::from_utf8_lossy(&output.stdout);
    parse_ping_stats(&text)
}

/// Parse `rtt min/avg/max/mdev = a/b/c/d ms` → `(avg_us, mdev_us)`.
fn parse_ping_stats(text: &str) -> (u32, u32) {
    for line in text.lines() {
        if line.contains("min/avg/max") {
            if let Some(stats) = line.split('=').nth(1) {
                let parts: Vec<f64> = stats
                    .trim()
                    .split('/')
                    .filter_map(|s| {
                        let s = s.trim().trim_end_matches("ms").trim();
                        s.parse().ok()
                    })
                    .collect();
                if parts.len() == 4 {
                    return ((parts[1] * 1000.0) as u32, (parts[3] * 1000.0) as u32);
                }
            }
        }
    }
    (0, 0)
}

/// Active-draw estimate per lane kind (watts, Art. 4). Conservative card-level
/// constants; per-card powercap measurement is a Phase C refinement.
fn kind_watts(kind: &EdgeKind) -> u32 {
    match kind {
        EdgeKind::Air | EdgeKind::Modem => 2,
        EdgeKind::Rdma => 3,
        EdgeKind::Dma => 5,
        EdgeKind::Blk | EdgeKind::Nvme => 5,
        EdgeKind::Gpu => 8,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_tcp_latency_unreachable() {
        let result = measure_tcp_latency("127.0.0.1:19999");
        assert!(result.is_err());
    }

    #[test]
    fn test_probe_agent_unreachable() {
        let result = probe_agent("127.0.0.1:19999");
        assert!(result.is_err());
    }

    #[test]
    fn test_kind_from_name() {
        assert_eq!(kind_from_name("wlan0"), EdgeKind::Air);
        assert_eq!(kind_from_name("wlx4c5e0c82a4a2"), EdgeKind::Air);
        assert_eq!(kind_from_name("rxe0"), EdgeKind::Rdma);
        assert_eq!(kind_from_name("ib0"), EdgeKind::Rdma);
        assert_eq!(kind_from_name("enp3s0"), EdgeKind::Tcp);
        assert_eq!(kind_from_name("eth0"), EdgeKind::Tcp);
    }

    #[test]
    fn test_parse_signal_dbm() {
        let out = "Connected to 12:34:56:78:9a:bc (on wlan0)\n\tSSID: home\n\tsignal: -52 dBm\n\ttx bitrate: 72.2 MBit/s\n";
        assert_eq!(parse_signal_dbm(out), -52);
        assert_eq!(parse_signal_dbm("not connected"), 0);
    }

    #[test]
    fn test_parse_tx_bitrate() {
        let out = "signal: -40 dBm\n\ttx bitrate: 144.4 MBit/s\n";
        assert_eq!(parse_tx_bitrate_mbps(out), 144);
        let bytes_out = "\ttx bitrate: 5.0 MB/s\n";
        assert_eq!(parse_tx_bitrate_mbps(bytes_out), 40);
    }

    #[test]
    fn test_parse_ping_stats() {
        let out = "rtt min/avg/max/mdev = 0.120/0.132/0.145/0.010 ms\n";
        assert_eq!(parse_ping_stats(out), (132, 10));
        assert_eq!(parse_ping_stats("no stats"), (0, 0));
    }

    #[test]
    fn test_kind_watts() {
        assert_eq!(kind_watts(&EdgeKind::Air), 2);
        assert_eq!(kind_watts(&EdgeKind::Tcp), 1);
        assert_eq!(kind_watts(&EdgeKind::Dma), 5);
    }
}
