//! W1 (PLAN §18.2): deterministic multi-adapter selection. Own test binary
//! because OURO_GPU_NAME mutates process env.

use ouro_wgpu::GpuPool;

#[test]
fn test_w1_select_by_name_and_default_discrete() {
    let names = ouro_wgpu::GpuPool::list_vulkan_adapters();
    eprintln!("vulkan adapters: {names:?}");
    assert!(!names.is_empty(), "no Vulkan adapters enumerated");

    // Gate: the 1070 Ti must be selectable by name on the dual-GPU box.
    // Generic fallback keeps the test meaningful on single-GPU machines.
    let needle = names
        .iter()
        .find(|n| n.contains("1070"))
        .map(|_| "1070")
        .unwrap_or_else(|| {
            names[0]
                .split(':')
                .nth(1)
                .unwrap_or_default()
                .split_whitespace()
                .last()
                .unwrap_or("NVIDIA")
        });
    std::env::set_var("OURO_GPU_NAME", needle);
    let pool = GpuPool::new().unwrap_or_else(|e| panic!("{needle} selectable by name: {e}"));
    std::env::remove_var("OURO_GPU_NAME");
    assert!(
        pool.adapter_name.to_lowercase().contains(&needle.to_lowercase()),
        "picked {}",
        pool.adapter_name
    );
    eprintln!("picked by name: {}", pool.adapter_name);

    // Default pick must land on a discrete GPU (no env pinning).
    let default = GpuPool::new().expect("default adapter pick");
    assert!(
        default.adapter_name.contains("DiscreteGpu"),
        "default pick should be discrete: {}",
        default.adapter_name
    );
    eprintln!("default pick: {}", default.adapter_name);
}
