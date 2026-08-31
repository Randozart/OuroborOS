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
var<workgroup> sc_sh: array<i32, 16>;
var<workgroup> d_sh: f32;

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

// G2: one lane's 6-bit weight from three u32 loads shared by 4 lanes.
// bucket 0: qlA low nibble | 1: qlB low | 2: qlA high | 3: qlB high;
// qh contributes 2 bits at byte-shift (8k + 2*bucket) inside its u32.
fn lane_q(ql_a: u32, ql_b: u32, qh_u: u32, bucket: u32, k: u32) -> i32 {
    let by = 8u * k;
    let hb = (qh_u >> (by + 2u * bucket)) & 3u;
    var nib: u32;
    if (bucket == 0u || bucket == 2u) {
        nib = (ql_a >> (by + (bucket / 2u) * 4u)) & 0xFu;
    } else {
        nib = (ql_b >> (by + ((bucket - 1u) / 2u) * 4u)) & 0xFu;
    }
    return i32(nib | (hb << 4u)) - 32;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.out_len) { return; }

    // 64 threads x 4 lanes = 256 elements = exactly one Q6_K block
    let row_base = row * p.blocks_per_row * 212u;
    let j0 = lid.x * 4u;
    let n = j0 / 128u;
    let l0 = j0 % 32u;
    let bucket = (j0 % 128u) / 32u;
    let sc_i = n * 8u + (l0 / 16u) + bucket * 2u; // shared by the 4 lanes

    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var b = 0u; b < p.blocks_per_row; b = b + 1u) {
        let base = row_base + b * 212u;

        // stage the 16 int8 scales + the f16 d once per block
        if (lid.x < 16u) {
            let sc_off = base + 192u + lid.x;
            let sc_u = (w[sc_off / 4u] >> ((sc_off % 4u) * 8u)) & 0xFFu;
            sc_sh[lid.x] = i32(sc_u) - select(0, 256, sc_u > 127u);
        }
        if (lid.x == 16u) { d_sh = f16_at(base + 208u); }
        workgroupBarrier();

        let ql_a = w[(base + n * 64u + l0) / 4u];
        let ql_b = w[(base + n * 64u + 32u + l0) / 4u];
        let qh_u = w[(base + 128u + n * 32u + l0) / 4u];

        let ds = d_sh * f32(sc_sh[sc_i]);
        let xb = b * 256u + j0;
        acc = acc + vec4<f32>(
            ds * f32(lane_q(ql_a, ql_b, qh_u, bucket, 0u)),
            ds * f32(lane_q(ql_a, ql_b, qh_u, bucket, 1u)),
            ds * f32(lane_q(ql_a, ql_b, qh_u, bucket, 2u)),
            ds * f32(lane_q(ql_a, ql_b, qh_u, bucket, 3u)))
            * vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        workgroupBarrier();
    }

    partial[lid.x] = acc.x + acc.y + acc.z + acc.w;
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
/// G1: every per-call resource is persistent — staging x, y, params, a
/// ring of two MAP_READ buffers, and one bind group created at upload.
pub struct ResidentMat {
    _w_buf: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    readback: [wgpu::Buffer; 2],
    _p_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    out_len: u32,
    in_len: u32,
    rb_idx: std::cell::Cell<usize>,
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
        if !in_len.is_multiple_of(256) || payload.len() != out_len * (in_len / 256) * Q6K_BLOCK {
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
        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: &padded,
                usage: wgpu::BufferUsages::STORAGE,
        });
        let mk_buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let x4 = (in_len * 4) as u64;
        let y4 = (out_len * 4) as u64;
        let x_buf = mk_buf("x", x4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let y_buf = mk_buf("y", y4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let rb_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let readback = [mk_buf("y_rb0", y4, rb_usage), mk_buf("y_rb1", y4, rb_usage)];
        let params = Params {
            out_len: out_len as u32,
            in_len: in_len as u32,
            blocks_per_row: blocks_per_row as u32,
            row_stride_u32: 0,
        };
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
            label: Some("q6k_bg"),
        });
        self.mats.insert(
            name.to_string(),
            ResidentMat {
                _w_buf: w_buf,
                x_buf,
                y_buf,
                readback,
                _p_buf: p_buf,
                bind_group,
                out_len: out_len as u32,
                in_len: in_len as u32,
                rb_idx: std::cell::Cell::new(0),
            },
        );
        Ok(())
    }

    /// y = W * x for a resident Q6_K matrix. G1: write_buffer staging, one
    /// persistent bind group per mat, ring-of-2 readback, map after submit.
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let m = self
            .mats
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("matrix {} not resident", name))?;
        if x.len() != m.in_len as usize {
            bail!("matvec {}: x len {} != in_len {}", name, x.len(), m.in_len);
        }
        let idx = m.rb_idx.get();
        let readback = &m.readback[idx];
        self.queue.write_buffer(&m.x_buf, 0, bytemuck::cast_slice(x));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q6k_enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &m.bind_group, &[]);
            cpass.dispatch_workgroups(m.out_len, 1, 1);
        }
        enc.copy_buffer_to_buffer(&m.y_buf, 0, readback, 0, (m.out_len as u64) * 4);
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
        m.rb_idx.set(idx ^ 1);
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
