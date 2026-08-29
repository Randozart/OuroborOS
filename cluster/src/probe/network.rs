use anyhow::Result;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use crate::probe::NetworkInfo;

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
}
