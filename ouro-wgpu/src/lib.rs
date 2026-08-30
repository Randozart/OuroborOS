//! Vulkan compute pool: Q6_K dequant-gemv on wgpu (L1 rung of the ladder).
//!
//! Contract: results must match `ouro_cluster::infer::ops::matvec_q(Q6K)`
//! within fp32 accumulation-order tolerance (cos > 0.9999) before any
//! stage binds this pool (docs/CONTRACTS.md L1).

pub const Q6K_BLOCK: usize = 210; // bytes per 256 elements
pub const Q6K_PAD: usize = 212; // u32-aligned stride used on GPU
pub const ELEMS_PER_BLOCK: usize = 256;

const WGSL: &str = r#"
struct Params {
    out_len: u32,
    in_len: u32,
    blocks_per_row: u32,
    row_stride_u32: u32,
};

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> partial: array<f32, 64>;

fn f16_at(byte_off: u32) -> f32 {
    let u = w[byte_off / 4u];
    let sh = (byte_off % 4u) * 8u;
    // assemble the 16-bit f16 pattern inside a u32 (no u16 type in WGSL)
    let bits = ((u >> sh) & 0xFFu) | (((u >> (sh + 8u)) & 0xFFu) << 8u);
    let sign = f32((bits >> 15u) & 1u) * -2.0 + 1.0;
    let exp = (bits >> 10u) & 0x1Fu;
    let frac = bits & 0x3FFu;
    if (exp == 0u) { return sign * f32(frac) * 5.9604645e-8; }
    return sign * (1.0 + f32(frac) / 1024.0) * exp2(f32(exp) - 15.0);
}

// dequant element j of block; weights bytes start at base (byte offset)
fn q6k_elem(base: u32, j: u32) -> f32 {
    let n = j / 128u;
    let r = j % 128u;
    let l = r % 32u;
    let bucket = r / 32u;

    // byte offsets inside the 210-byte block
    let ql_off = base + 0u;
    let qh_off = base + 128u;
    let sc_off = base + 192u;
    let d_off  = base + 208u;

    let qlo_l = ql_off + n * 64u + l;
    let qlo_l32 = qlo_l + 32u;

    // nibble+2bit assembly per bucket (mirror of ggml dequantize_row_q6_K)
    let qlA = byte_at(qlo_l);
    let qlB = byte_at(qlo_l32);
    let qh = byte_at(qh_off + n * 32u + l);
    let shift = bucket * 2u;
    var q: i32 = 0;
    if (bucket == 0u) { q = i32((qlA & 0xFu) | (((qh >> 0u) & 3u) << 4u)); }
    else if (bucket == 1u) { q = i32((qlB & 0xFu) | (((qh >> 2u) & 3u) << 4u)); }
    else if (bucket == 2u) { q = i32(((qlA >> 4u) & 0xFu) | (((qh >> 4u) & 3u) << 4u)); }
    else { q = i32(((qlB >> 4u) & 0xFu) | (((qh >> 6u) & 3u) << 4u)); }
    let qi = q - 32;

    let sc_idx = sc_off + n * 8u + (l / 16u) + bucket * 2u;
    // int8 sign-extension: WGSL i32(u32) is a value cast, not bit reinterpret
    let sc_u = byte_at(sc_idx);
    let sc = i32(sc_u) - select(0, 256, sc_u > 127u);
    let d = f16_at(d_off);
    return d * f32(sc) * f32(qi);
}

fn byte_at(byte_off: u32) -> u32 {
    let u = w[byte_off / 4u];
    let sh = (byte_off % 4u) * 8u;
    return (u >> sh) & 0xFFu;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.out_len) { return; }

    // CPU padded each block to 212 bytes; row_base in padded bytes:
    let row_base_bytes = row * p.blocks_per_row * 212u;

    var acc = 0.0;
    // each thread strides over blocks
    var b = lid.x;
    loop {
        if (b >= p.blocks_per_row) { break; }
        let blk_base_bytes = row_base_bytes + b * 212u;
        for (var j = 0u; j < 256u; j = j + 1u) {
            let wv = q6k_elem(blk_base_bytes, j);
            acc = acc + wv * x[b * 256u + j];
        }
        b = b + 64u;
    }
    partial[lid.x] = acc;
    workgroupBarrier();

    // tree reduce 64 partials
    var s = 32u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lid.x == 0u) { y[row] = partial[0]; }
}
"#;

use anyhow::{bail, Result};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    out_len: u32,
    in_len: u32,
    blocks_per_row: u32,
    row_stride_u32: u32,
}

/// One resident weight matrix on the GPU (Q6_K payload, padded blocks).
pub struct ResidentMat {
    _buf: wgpu::Buffer,
    out_len: u32,
    in_len: u32,
    blocks_per_row: u32,
}

/// Vulkan compute pool for Q6_K matvecs.
pub struct GpuPool {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    mats: std::collections::HashMap<String, ResidentMat>,
    pub adapter_name: String,
}

impl GpuPool {
    /// Init on the Vulkan backend (works: NVIDIA r610 here; r580 ICD/NVK on slaves).
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
        })
        .map_err(|e| anyhow::anyhow!("no Vulkan adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ouro"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 1 << 30,
                        max_buffer_size: 1 << 31,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
        })?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q6k_gemv"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q6k_gemv"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            mats: Default::default(),
            adapter_name: format!("{} ({:?})", info.name, info.backend),
        })
    }

    /// Repack 210-byte Q6_K blocks into 212-byte (53 u32) rows and upload.
    pub fn upload_q6k(&mut self, name: &str, payload: &[u8], out_len: usize, in_len: usize) -> Result<()> {
        if in_len % 256 != 0 || payload.len() != out_len * (in_len / 256) * Q6K_BLOCK {
            bail!("upload_q6k {}: shape mismatch (payload {} out {} in {})", name, payload.len(), out_len, in_len);
        }
        let blocks_per_row = in_len / 256;
        let mut padded = vec![0u8; out_len * blocks_per_row * Q6K_PAD];
        for r in 0..out_len {
            for b in 0..blocks_per_row {
                let src = (r * blocks_per_row + b) * Q6K_BLOCK;
                let dst = (r * blocks_per_row + b) * Q6K_PAD;
                padded[dst..dst + Q6K_BLOCK].copy_from_slice(&payload[src..src + Q6K_BLOCK]);
            }
        }
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: &padded,
                usage: wgpu::BufferUsages::STORAGE,
        });
        self.mats.insert(
            name.to_string(),
            ResidentMat { _buf: buf, out_len: out_len as u32, in_len: in_len as u32, blocks_per_row: blocks_per_row as u32 },
        );
        Ok(())
    }

    /// y = W * x for a resident Q6_K matrix.
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let m = self
            .mats
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("matrix {} not resident", name))?;
        if x.len() != m.in_len as usize {
            bail!("matvec {}: x len {} != in_len {}", name, x.len(), m.in_len);
        }
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
        });
        let y_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: (m.out_len as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y_readback"),
            size: (m.out_len as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = Params {
            out_len: m.out_len,
            in_len: m.in_len,
            blocks_per_row: m.blocks_per_row,
            row_stride_u32: 0,
        };
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: m._buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
            label: Some("q6k_bg"),
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q6k_enc") });
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(m.out_len, 1, 1);
        drop(cpass);
        enc.copy_buffer_to_buffer(&y_buf, 0, &readback, 0, (m.out_len as u64) * 4);
        self.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let _ = self.device.poll(wgpu::PollType::wait());
        rx.recv().unwrap().map_err(|e| anyhow::anyhow!("map: {e}"))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        Ok(out)
    }
}

/// Repack helper shared with tests: scalar reference must read the same bytes.
pub fn padded_bytes(payload: &[u8], out_len: usize, in_len: usize) -> Vec<u8> {
    let bpr = in_len / 256;
    let mut padded = vec![0u8; out_len * bpr * Q6K_PAD];
    for r in 0..out_len {
        for b in 0..bpr {
            let src = (r * bpr + b) * Q6K_BLOCK;
            let dst = (r * bpr + b) * Q6K_PAD;
            padded[dst..dst + Q6K_BLOCK].copy_from_slice(&payload[src..src + Q6K_BLOCK]);
        }
    }
    padded
}
