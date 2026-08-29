use anyhow::Result;
use crate::probe::EnergyInfo;

/// Probe Intel RAPL energy telemetry.
///
/// Reads from `/sys/class/powercap/intel-rapl:0/` which provides:
/// - `energy_uj`: microjoules consumed (monotonically increasing)
/// - `power_limit`: constraint in microwatts
///
/// If RAPL is unavailable, falls back to TDP estimate.
pub fn probe_local() -> Result<EnergyInfo> {
    let rapl_base = std::path::Path::new("/sys/class/powercap/intel-rapl:0");

    if rapl_base.exists() {
        let _energy_uj = read_file_u64(&rapl_base.join("energy_uj"))?;
        let power_limit = read_file_u64(&rapl_base.join("power_limit"))
            .ok()
            .map(|v| (v / 1_000_000) as u32);

        // For a single reading, we report the power limit as current draw
        // (conservative estimate). In practice, the agent will do delta readings.
        let current_watts = power_limit.unwrap_or(35);

        Ok(EnergyInfo {
            current_watts,
            rapl_available: true,
            power_limit_watts: power_limit,
        })
    } else {
        // RAPL not available — use conservative TDP estimate
        Ok(EnergyInfo {
            current_watts: 35,
            rapl_available: false,
            power_limit_watts: None,
        })
    }
}

pub fn probe_remote(ip: &str) -> Result<EnergyInfo> {
    let output = std::process::Command::new("ssh")
        .args([ip, "cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null || echo 0"])
        .output()?;

    let data = String::from_utf8(output.stdout)?;
    let energy_uj: u64 = data.trim().parse().unwrap_or(0);

    if energy_uj > 0 {
        // Read power limit
        let limit_output = std::process::Command::new("ssh")
            .args([ip, "cat /sys/class/powercap/intel-rapl:0/power_limit 2>/dev/null || echo 0"])
            .output()?;
        let limit_data = String::from_utf8(limit_output.stdout)?;
        let power_limit: Option<u32> = limit_data.trim().parse::<u64>()
            .ok()
            .map(|v| (v / 1_000_000) as u32);

        let current_watts = power_limit.unwrap_or(35);

        Ok(EnergyInfo {
            current_watts,
            rapl_available: true,
            power_limit_watts: power_limit,
        })
    } else {
        Ok(EnergyInfo {
            current_watts: 35,
            rapl_available: false,
            power_limit_watts: None,
        })
    }
}

fn read_file_u64(path: &std::path::Path) -> Result<u64> {
    let data = std::fs::read_to_string(path)?;
    let value: u64 = data.trim().parse()?;
    Ok(value)
}
