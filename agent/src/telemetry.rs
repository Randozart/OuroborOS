use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

/// Live telemetry snapshot from this node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Telemetry {
    pub hostname: String,
    pub cpu_model: String,
    pub cores: u32,
    pub threads: u32,
    #[serde(default)]
    pub has_avx: bool,
    pub has_avx2: bool,
    #[serde(default)]
    pub has_sse42: bool,
    pub ram_total_mib: u64,
    pub ram_used_mib: u64,
    pub power_watts: u32,
    pub temp_c: u32,
    pub load_avg: f64,
    #[serde(default, skip_serializing_if="Vec::is_empty")]
    pub gpus: Vec<ouro_cluster::probe::gpu::GpuInfo>,
    #[serde(default, skip_serializing_if="String::is_empty")]
    pub agent_version: String,
    #[serde(default, skip_serializing_if="String::is_empty")]
    pub image_rev: String,
    /// Stable identity (SMBIOS-derived); the anchor across many IPs
    /// (docs/AIR_PATH.md §4.4).
    #[serde(default, skip_serializing_if="String::is_empty")]
    pub node_id: String,
    /// This node's priced lanes, self-reported (docs/AIR_PATH.md §1.1).
    #[serde(default, skip_serializing_if="Vec::is_empty")]
    pub edges: Vec<ouro_cluster::transport::edge::PricedEdge>,
    /// The primary interface's MAC address — the WOL target the head needs
    /// to wake a declared-awake tail that has suspended itself.
    #[serde(default, skip_serializing_if="String::is_empty")]
    pub mac: String,
}

/// Compile-time build stamp. Nix sets OURO_BUILD_REV from the flake
/// rev; plain cargo builds report "dev" honestly (WP-U3).
pub fn agent_version() -> &'static str {
    option_env!("OURO_BUILD_REV").unwrap_or("dev")
}

/// The image revision this tail booted (stamped at build into
/// /etc/ouro/image-rev). Unknown on non-image dev machines.
fn image_rev() -> String {
    fs::read_to_string("/etc/ouro/image-rev")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

/// GPU inventory, cached (nvidia-smi spawns a process).
fn cached_gpus() -> &'static Vec<ouro_cluster::probe::gpu::GpuInfo> {
    static GPUS: std::sync::OnceLock<Vec<ouro_cluster::probe::gpu::GpuInfo>> = std::sync::OnceLock::new();
    GPUS.get_or_init(ouro_cluster::probe::gpu::detect_gpus)
}

/// Collect a telemetry snapshot from the local system.
pub fn collect() -> Result<Telemetry> {
    collect_with_head(None)
}

/// Collect a telemetry snapshot, pricing each lane's RTT/jitter against the
/// head when one is known (docs/AIR_PATH.md §4.2): `probe_lanes(Some(head))`
/// pings through each interface, so every edge carries a real
/// latency/jitter pair the bond policy can consume. Without a head, lanes
/// are priced on link speed/signal/watts only.
pub fn collect_with_head(head: Option<&str>) -> Result<Telemetry> {
    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let cpu = read_cpuinfo()?;
    let mem = read_meminfo()?;
    let power = read_power();
    let temp = read_temp();
    let load = read_loadavg();
    let head_ip = head.and_then(|h| h.split(':').next()).map(|s| s.to_string());

    Ok(Telemetry {
        hostname,
        cpu_model: cpu.0,
        cores: cpu.1,
        threads: cpu.2,
        has_avx: cpu.3,
        has_avx2: cpu.4,
        has_sse42: cpu.5,
        ram_total_mib: mem.0,
        ram_used_mib: mem.1,
        power_watts: power,
        temp_c: temp,
        load_avg: load,
        gpus: cached_gpus().clone(),
        agent_version: agent_version().to_string(),
        image_rev: image_rev(),
        node_id: ouro_cluster::probe::derive_node_id(),
        edges: ouro_cluster::probe::network::probe_lanes(head_ip.as_deref()),
        mac: read_mac(),
    })
}

/// The primary interface's MAC address: the wired iface if one is up, else
/// any non-loopback carrier interface. Empty when no MAC is readable (the
/// head then honestly reports "no WOL path" for wake).
fn read_mac() -> String {
    let mut ifaces: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/class/net") {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name == "lo" {
                continue;
            }
            let carrier = std::fs::read_to_string(format!("/sys/class/net/{name}/carrier"))
                .map(|s| s.trim() == "1")
                .unwrap_or(false);
            if carrier {
                ifaces.push(name);
            }
        }
    }
    // Prefer wired (non-wireless) interfaces.
    ifaces.sort_by_key(|n| n.starts_with("wlan") as usize);
    for name in ifaces {
        if let Ok(mac) = std::fs::read_to_string(format!("/sys/class/net/{name}/address")) {
            let mac = mac.trim().to_string();
            if mac.len() == 17 {
                return mac;
            }
        }
    }
    String::new()
}

/// Read CPU model, cores, threads, and SIMD flags from /proc/cpuinfo.
fn read_cpuinfo() -> Result<(String, u32, u32, bool, bool, bool)> {
    let data = fs::read_to_string("/proc/cpuinfo")?;
    let mut model = String::new();
    let mut cores = 0u32;
    let mut has_avx = false;
    let mut has_avx2 = false;
    let mut has_sse42 = false;

    for line in data.lines() {
        if line.starts_with("model name") {
            if let Some(val) = line.split(':').nth(1) {
                model = val.trim().to_string();
            }
        }
        if line.starts_with("cpu cores") {
            if let Some(val) = line.split(':').nth(1) {
                cores = val.trim().parse().unwrap_or(0);
            }
        }
        if line.starts_with("flags") || line.starts_with("Features") {
            has_avx = line.contains(" avx ");
            has_avx2 = line.contains(" avx2 ");
            has_sse42 = line.contains(" sse4_2 ");
        }
    }

    let threads = data.lines().filter(|l| l.starts_with("processor")).count() as u32;
    if cores == 0 {
        cores = threads;
    }

    Ok((model, cores, threads, has_avx, has_avx2, has_sse42))
}

/// Read total and used RAM from /proc/meminfo.
fn read_meminfo() -> Result<(u64, u64)> {
    let data = fs::read_to_string("/proc/meminfo")?;
    let mut total_kb = 0u64;
    let mut available_kb = 0u64;

    for line in data.lines() {
        if line.starts_with("MemTotal") {
            if let Some(val) = line.split_whitespace().nth(1) {
                total_kb = val.parse().unwrap_or(0);
            }
        }
        if line.starts_with("MemAvailable") {
            if let Some(val) = line.split_whitespace().nth(1) {
                available_kb = val.parse().unwrap_or(0);
            }
        }
    }

    let total_mib = total_kb / 1024;
    let used_mib = (total_kb.saturating_sub(available_kb)) / 1024;
    Ok((total_mib, used_mib))
}

/// Read current power draw from Intel RAPL.
fn read_power() -> u32 {
    let rapl = "/sys/class/powercap/intel-rapl:0/power_limit";
    if let Ok(data) = fs::read_to_string(rapl) {
        if let Ok(val) = data.trim().parse::<u64>() {
            return (val / 1_000_000) as u32;
        }
    }
    35 // conservative default
}

/// Read CPU temperature from thermal zone.
fn read_temp() -> u32 {
    for zone in 0..10 {
        let path = format!("/sys/class/thermal/thermal_zone{}/temp", zone);
        if let Ok(data) = fs::read_to_string(&path) {
            if let Ok(millideg) = data.trim().parse::<u32>() {
                return millideg / 1000;
            }
        }
    }
    0
}

/// Read 1-minute load average.
fn read_loadavg() -> f64 {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|data| {
            data.split_whitespace()
                .next()
                .and_then(|s| s.parse::<f64>().ok())
        })
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_runs() {
        let tel = collect().unwrap();
        assert!(!tel.hostname.is_empty());
        assert!(tel.ram_total_mib > 0);
    }

    #[test]
    fn test_read_meminfo() {
        let (total, used) = read_meminfo().unwrap();
        assert!(total > 0);
        assert!(used <= total);
    }
}
