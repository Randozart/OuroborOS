use anyhow::Result;
use crate::probe::MemoryInfo;

pub fn probe_local() -> Result<MemoryInfo> {
    let data = std::fs::read_to_string("/proc/meminfo")?;
    parse_meminfo(&data)
}

pub fn probe_remote(ip: &str) -> Result<MemoryInfo> {
    let output = std::process::Command::new("ssh")
        .args([ip, "cat /proc/meminfo"])
        .output()?;
    let data = String::from_utf8(output.stdout)?;
    parse_meminfo(&data)
}

fn parse_meminfo(data: &str) -> Result<MemoryInfo> {
    let mut total_mib = 0u64;

    for line in data.lines() {
        if line.starts_with("MemTotal") {
            if let Some(val) = line.split_whitespace().nth(1) {
                if let Ok(kb) = val.parse::<u64>() {
                    total_mib = kb / 1024;
                }
            }
        }
    }

    Ok(MemoryInfo {
        total_mib,
        speed_mhz: None,
        memory_type: None,
    })
}
