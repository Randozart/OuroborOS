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

/// True draw over an interval: ΔµJ / Δs. RAPL's energy counter is
/// monotonic; a single reading is meaningless (the limit is NOT the draw).
pub fn watts_between(prev_uj: u64, prev: std::time::Instant, now_uj: u64, now: std::time::Instant) -> f32 {
    let uj = now_uj.saturating_sub(prev_uj) as f32;
    let secs = now.duration_since(prev).as_secs_f32();
    if secs <= 0.0 {
        return 0.0;
    }
    uj / 1_000_000.0 / secs
}

/// Interval meter over one RAPL domain: hold the previous reading, sample
/// the current, report true watts. Generic reader keeps it testable.
pub struct EnergyMeter<F: FnMut() -> Result<u64>> {
    read: F,
    last_uj: Option<u64>,
    last_t: Option<std::time::Instant>,
}

impl<F: FnMut() -> Result<u64>> EnergyMeter<F> {
    pub fn new(read: F) -> Self {
        Self { read, last_uj: None, last_t: None }
    }

    /// First call primes the meter (returns None); later calls return the
    /// true average watts over the interval since the previous call.
    pub fn sample(&mut self) -> Result<Option<f32>> {
        let uj = (self.read)()?;
        let t = std::time::Instant::now();
        let out = match (self.last_uj, self.last_t) {
            (Some(pu), Some(pt)) => Some(watts_between(pu, pt, uj, t)),
            _ => None,
        };
        self.last_uj = Some(uj);
        self.last_t = Some(t);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watts_between_arithmetic() {
        let t0 = std::time::Instant::now();
        let t1 = t0 + std::time::Duration::from_secs(2);
        // 10 J over 2 s = 5 W (10_000_000 µJ)
        let w = watts_between(0, t0, 10_000_000, t1);
        assert!((w - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_meter_primes_then_reports() {
        let mut reading = 1_000_000u64;
        let mut meter = EnergyMeter::new(move || {
            reading += 2_000_000; // +2 J per sample
            Ok(reading)
        });
        assert!(meter.sample().unwrap().is_none(), "first sample primes");
        let w = meter.sample().unwrap().expect("second sample reports");
        assert!(w.is_finite() && w > 0.0, "positive finite watts, got {w}");
        // arithmetic itself is covered by test_watts_between_arithmetic;
        // the real-interval magnitude depends on wall-clock elapsed time.
    }
}
