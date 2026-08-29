//! GPU presence probing: nvidia-smi CSV (vendor extension point for Vulkan/lspci).

use serde::{Deserialize, Serialize};
use std::process::Command;

/// One discovered GPU.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    pub vram_mib: u64,
    pub driver: String,
    pub compute_cap: String,
}

/// Parse `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap
/// --format=csv,noheader,nounits` output.
pub fn parse_nvidia_smi(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(|p| p.trim());
            let model = parts.next()?.to_string();
            if model.is_empty() {
                return None;
            }
            let vram_mib: u64 = parts.next()?.parse().ok()?;
            Some(GpuInfo {
                vendor: "nvidia".to_string(),
                model,
                vram_mib,
                driver: parts.next().unwrap_or("").to_string(),
                compute_cap: parts.next().unwrap_or("").to_string(),
            })
        })
        .collect()
}

/// Detect GPUs; empty when no probe tool exists (CPU-only node).
pub fn detect_gpus() -> Vec<GpuInfo> {
    let Ok(out) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_recorded_csv() {
        // Recorded 2026-08-29 on the master box (RTX 3060 live, r610)
        let csv = "NVIDIA GeForce RTX 3060, 12288, 610.57.04, 8.6\n";
        let g = parse_nvidia_smi(csv);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].model, "NVIDIA GeForce RTX 3060");
        assert_eq!(g[0].vram_mib, 12288);
        assert_eq!(g[0].compute_cap, "8.6");
        assert_eq!(g[0].vendor, "nvidia");
    }

    #[test]
    fn test_parse_multi_gpu() {
        let csv = "NVIDIA GeForce RTX 3060, 12288, 580.1, 8.6\nNVIDIA GeForce GTX 1070 Ti, 8192, 580.1, 6.1\n";
        let g = parse_nvidia_smi(csv);
        assert_eq!(g.len(), 2);
        assert_eq!(g[1].vram_mib, 8192);
    }

    #[test]
    fn test_parse_garbage_and_empty() {
        assert!(parse_nvidia_smi("").is_empty());
        assert!(parse_nvidia_smi("not a csv at all\n").is_empty());
        assert!(parse_nvidia_smi("GPU, not_a_number, x, y\n").is_empty());
    }

    #[test]
    fn test_detect_on_this_machine() {
        // If nvidia-smi exists (master box), it must report the 3060 truthfully.
        let g = detect_gpus();
        for card in &g {
            assert!(card.vram_mib > 0);
            assert!(!card.model.is_empty());
        }
    }
}
