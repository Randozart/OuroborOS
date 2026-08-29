use anyhow::Result;
use crate::probe::CpuInfo;

/// Probe CPU via /proc/cpuinfo (local or remote via SSH).
pub fn probe_local() -> Result<CpuInfo> {
    let data = std::fs::read_to_string("/proc/cpuinfo")?;
    parse_cpuinfo(&data)
}

pub fn probe_remote(ip: &str) -> Result<CpuInfo> {
    let output = std::process::Command::new("ssh")
        .args([ip, "cat /proc/cpuinfo"])
        .output()?;
    let data = String::from_utf8(output.stdout)?;
    parse_cpuinfo(&data)
}

fn parse_cpuinfo(data: &str) -> Result<CpuInfo> {
    let mut model = String::new();
    let mut cores = 0u32;
    let mut has_avx = false;
    let mut has_avx2 = false;
    let mut has_sse42 = false;
    let mut has_bmi1 = false;
    let mut has_bmi2 = false;

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
            has_bmi1 = line.contains(" bmi1 ");
            has_bmi2 = line.contains(" bmi2 ");
        }
    }

    // Count physical processors (entries with "processor" key)
    let threads = data.lines()
        .filter(|l| l.starts_with("processor"))
        .count() as u32;

    if cores == 0 {
        cores = threads;
    }

    // Estimate TDP from model name (rough heuristic)
    let tdp = estimate_tdp(&model);

    Ok(CpuInfo {
        model,
        cores,
        threads,
        has_avx,
        has_avx2,
        has_sse42,
        has_bmi1,
        has_bmi2,
        tdp_watts: tdp,
    })
}

fn estimate_tdp(model: &str) -> u32 {
    let lower = model.to_lowercase();
    if lower.contains("i7") || lower.contains("i9") {
        45
    } else if lower.contains("i5") {
        35
    } else if lower.contains("i3") || lower.contains("celeron") {
        25
    } else if lower.contains("pentium") {
        15
    } else {
        35 // default
    }
}
