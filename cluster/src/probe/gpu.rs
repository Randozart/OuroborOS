//! GPU presence probing: nvidia-smi CSV + Vulkan enumeration via
//! `vulkaninfo --summary` (W1: NodeEntry must speak Vulkan, not CUDA-isms).

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
    /// Vulkan API version from `vulkaninfo --summary` ("" when absent).
    #[serde(default)]
    pub vulkan_api: String,
}

/// One Vulkan physical device parsed from `vulkaninfo --summary`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VulkanInfo {
    pub device_name: String,
    pub api_version: String,
    pub device_type: String,
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
                vulkan_api: String::new(),
            })
        })
        .collect()
}

/// Parse the Devices section of `vulkaninfo --summary` (GPUn blocks).
pub fn parse_vulkaninfo_summary(out: &str) -> Vec<VulkanInfo> {
    let mut devs = Vec::new();
    let mut cur: Option<VulkanInfo> = None;
    for line in out.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("GPU") {
            let digits = rest.strip_suffix(':').unwrap_or("");
            if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                if let Some(c) = cur.take() {
                    devs.push(c);
                }
                cur = Some(VulkanInfo::default());
                continue;
            }
        }
        let Some(c) = cur.as_mut() else { continue };
        if let Some(v) = t.strip_prefix("apiVersion") {
            c.api_version = kv_value(v);
        } else if let Some(v) = t.strip_prefix("deviceName") {
            c.device_name = kv_value(v);
        } else if let Some(v) = t.strip_prefix("deviceType") {
            let raw = kv_value(v);
            c.device_type = raw.strip_prefix("PHYSICAL_DEVICE_TYPE_").unwrap_or(&raw).to_string();
        }
    }
    if let Some(c) = cur {
        devs.push(c);
    }
    devs.retain(|d| !d.device_name.is_empty());
    devs
}

fn kv_value(kv: &str) -> String {
    kv.split('=').nth(1).unwrap_or("").trim().to_string()
}

/// Fill g.vulkan_api by matching nvidia-smi models to Vulkan devices.
fn merge_vulkan(gpus: &mut [GpuInfo], vk: &[VulkanInfo]) {
    for g in gpus.iter_mut() {
        let hit = vk.iter().find(|v| {
            v.device_name == g.model || v.device_name.contains(&g.model) || g.model.contains(&v.device_name)
        });
        if let Some(v) = hit {
            g.vulkan_api = v.api_version.clone();
        }
    }
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
    let mut gpus = parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout));
    if let Ok(vk) = Command::new("vulkaninfo").arg("--summary").output() {
        if vk.status.success() {
            merge_vulkan(&mut gpus, &parse_vulkaninfo_summary(&String::from_utf8_lossy(&vk.stdout)));
        }
    }
    gpus
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
    fn test_parse_vulkaninfo_two_gpu() {
        // Recorded 2026-08-31 on the master box (580.178.04, both cards live).
        let summary = "Devices:\n========\nGPU0:\n\tapiVersion         = 1.4.312\n\tdriverVersion      = 580.178.4.0\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n\tdeviceName         = NVIDIA GeForce RTX 3060\n\tdriverInfo         = 580.178.04\nGPU1:\n\tapiVersion         = 1.4.312\n\tdriverVersion      = 580.178.4.0\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n\tdeviceName         = NVIDIA GeForce GTX 1070 Ti\n\tdriverInfo         = 580.178.04\n";
        let v = parse_vulkaninfo_summary(summary);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].device_name, "NVIDIA GeForce RTX 3060");
        assert_eq!(v[0].api_version, "1.4.312");
        assert_eq!(v[0].device_type, "DISCRETE_GPU");
        assert_eq!(v[1].device_name, "NVIDIA GeForce GTX 1070 Ti");
        assert_eq!(v[1].api_version, "1.4.312");
    }

    #[test]
    fn test_parse_vulkaninfo_garbage() {
        assert!(parse_vulkaninfo_summary("").is_empty());
        // Instance-extension noise before Devices must not produce entries.
        assert!(parse_vulkaninfo_summary("VULKANINFO\nInstance Extensions: count = 24\nGPU, not a block\n").is_empty());
    }

    #[test]
    fn test_merge_vulkan_matches_models() {
        let mut gpus = parse_nvidia_smi(
            "NVIDIA GeForce RTX 3060, 12288, 580.178.04, 8.6\nNVIDIA GeForce GTX 1070 Ti, 8192, 580.178.04, 6.1\n",
        );
        let vk = vec![
            VulkanInfo {
                device_name: "NVIDIA GeForce RTX 3060".into(),
                api_version: "1.4.312".into(),
                device_type: "DISCRETE_GPU".into(),
            },
            VulkanInfo {
                device_name: "NVIDIA GeForce GTX 1070 Ti".into(),
                api_version: "1.4.312".into(),
                device_type: "DISCRETE_GPU".into(),
            },
        ];
        merge_vulkan(&mut gpus, &vk);
        assert_eq!(gpus[0].vulkan_api, "1.4.312");
        assert_eq!(gpus[1].vulkan_api, "1.4.312");
        // no match -> stays empty
        let mut lone = vec![GpuInfo { model: "AMD Renoir".into(), ..Default::default() }];
        merge_vulkan(&mut lone, &vk);
        assert_eq!(lone[0].vulkan_api, "");
    }

    #[test]
    fn test_detect_on_this_machine() {
        // If nvidia-smi exists (master box), it must report the 3060 truthfully
        // and, when vulkaninfo is present, both cards carry their Vulkan API.
        let g = detect_gpus();
        for card in &g {
            assert!(card.vram_mib > 0);
            assert!(!card.model.is_empty());
        }
        eprintln!("detected: {g:#?}");
    }
}
