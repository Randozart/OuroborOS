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
            let flags = line.to_string();
            has_avx = flags.contains(" avx ") || flags.ends_with(" avx");
            has_avx2 = flags.contains(" avx2 ") || flags.ends_with(" avx2");
            has_sse42 = flags.contains(" sse4_2 ") || flags.ends_with(" sse4_2");
            has_bmi1 = flags.contains(" bmi1 ") || flags.ends_with(" bmi1");
            has_bmi2 = flags.contains(" bmi2 ") || flags.ends_with(" bmi2");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpuinfo_with_simd_flags() {
        let data = "\
processor\t: 0
model name\t: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
cpu cores\t: 4
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc avx avx2 sse4_1 sse4_2 bmi1 bmi2
processor\t: 1
model name\t: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
cpu cores\t: 4
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc avx avx2 sse4_1 sse4_2 bmi1 bmi2
processor\t: 2
model name\t: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
cpu cores\t: 4
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc avx avx2 sse4_1 sse4_2 bmi1 bmi2
processor\t: 3
model name\t: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
cpu cores\t: 4
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc avx avx2 sse4_1 sse4_2 bmi1 bmi2
";
        let info = parse_cpuinfo(data).unwrap();
        assert_eq!(info.model, "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz");
        assert_eq!(info.cores, 4);
        assert_eq!(info.threads, 4);
        assert!(info.has_avx);
        assert!(info.has_avx2);
        assert!(info.has_sse42);
        assert!(info.has_bmi1);
        assert!(info.has_bmi2);
    }

    #[test]
    fn test_parse_cpuinfo_no_avx() {
        let data = "\
processor\t: 0
model name\t: Intel(R) Core(TM)2 Duo CPU E8400 @ 3.00GHz
cpu cores\t: 2
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx lm
";
        let info = parse_cpuinfo(data).unwrap();
        assert!(!info.has_avx);
        assert!(!info.has_avx2);
        assert!(!info.has_sse42);
    }
}
