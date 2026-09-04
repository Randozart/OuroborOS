//! OpenCL compute pool: Q6_K dequant-gemv — the vendor-agnostic GPU
//! path (GPU_CLAIM.md WP-G3). One code path, any ICD: NVIDIA on the
//! head, Intel NEO (HD 620) on tails.
//!
//! The kernel is a mechanical port of `ouro-wgpu`'s WGSL (L1 rung,
//! parity-proven on the RTX 3060). Contract: results must match
//! `ouro_cluster::infer::matvec_q(Q6K)` within fp32 accumulation-order
//! tolerance (cos > 0.9999) before anything routes GPU work (Art. 10).

use anyhow::{bail, Context, Result};

pub const Q6K_BLOCK: usize = 210; // bytes per 256 elements
pub const Q6K_PAD: usize = 212; // u32-aligned stride on device
pub const ELEMS_PER_BLOCK: usize = 256;

/// One workgroup per output row: 64 threads x 4 lanes = 256 elements =
/// exactly one Q6_K block. Byte offsets stay u32-aligned because every
/// row stride is 212 = 4 * 53 words (same trick as the WGSL port).
const KERNEL: &str = r#"
float f16_at(__global const uint* w, size_t byte_off) {
    uint u = w[byte_off >> 2];
    uint sh = (uint)(byte_off & 3u) * 8u;
    // assemble the 16-bit f16 pattern inside a u32 (no half type here)
    uint bits = ((u >> sh) & 0xFFu) | (((u >> (sh + 8u)) & 0xFFu) << 8u);
    float sign = ((bits >> 15u) & 1u) != 0u ? -1.0f : 1.0f;
    uint e = (bits >> 10u) & 0x1Fu;
    uint frac = bits & 0x3FFu;
    if (e == 0u) { return sign * (float)frac * 5.9604645e-8f; }
    return sign * (1.0f + (float)frac / 1024.0f) * exp2((float)e - 15.0f);
}

// One lane's 6-bit weight from three u32 loads shared by 4 lanes.
// bucket 0: qlA low nibble | 1: qlB low | 2: qlA high | 3: qlB high;
// qh contributes 2 bits at byte-shift (8k + 2*bucket) inside its u32.
int lane_q(uint ql_a, uint ql_b, uint qh_u, uint bucket, uint k) {
    uint by = 8u * k;
    uint hb = (qh_u >> (by + 2u * bucket)) & 3u;
    uint nib;
    if (bucket == 0u || bucket == 2u) {
        nib = (ql_a >> (by + (bucket / 2u) * 4u)) & 0xFu;
    } else {
        nib = (ql_b >> (by + ((bucket - 1u) / 2u) * 4u)) & 0xFu;
    }
    return (int)(nib | (hb << 4u)) - 32;
}

__kernel void q6k_gemv(
    __global const uint* w,
    __global const float* x,
    __global float* y,
    const uint out_len,
    const uint blocks_per_row)
{
    __local float partial[64];
    __local int sc_sh[16];
    __local float d_sh;

    uint row = (uint)get_group_id(0);
    uint lid = (uint)get_local_id(0);
    if (row >= out_len) { return; }

    size_t row_base = (size_t)row * blocks_per_row * 212u;
    uint j0 = lid * 4u;
    uint n = j0 / 128u;
    uint l0 = j0 % 32u;
    uint bucket = (j0 % 128u) / 32u;
    uint sc_i = n * 8u + (l0 / 16u) + bucket * 2u; // shared by the 4 lanes

    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint b = 0u; b < blocks_per_row; b++) {
        size_t base = row_base + (size_t)b * 212u;

        // stage the 16 int8 scales + the f16 d once per block
        if (lid < 16u) {
            size_t sc_off = base + 192u + lid;
            uint sc_u = (w[sc_off >> 2] >> ((uint)(sc_off & 3u) * 8u)) & 0xFFu;
            sc_sh[lid] = (int)sc_u - (sc_u > 127u ? 256 : 0);
        }
        if (lid == 16u) { d_sh = f16_at(w, base + 208u); }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint ql_a = w[(base + n * 64u + l0) >> 2];
        uint ql_b = w[(base + n * 64u + 32u + l0) >> 2];
        uint qh_u = w[(base + 128u + n * 32u + l0) >> 2];

        float ds = d_sh * (float)sc_sh[sc_i];
        uint xb = b * 256u + j0;
        float4 wq = (float4)(
            (float)lane_q(ql_a, ql_b, qh_u, bucket, 0u),
            (float)lane_q(ql_a, ql_b, qh_u, bucket, 1u),
            (float)lane_q(ql_a, ql_b, qh_u, bucket, 2u),
            (float)lane_q(ql_a, ql_b, qh_u, bucket, 3u));
        float4 xv = (float4)(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        acc += (ds * wq) * xv;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    partial[lid] = acc.x + acc.y + acc.z + acc.w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // tree reduce 64 partials
    for (uint s = 32u; s > 0u; s >>= 1) {
        if (lid < s) { partial[lid] += partial[lid + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0u) { y[row] = partial[0]; }
}
"#;

use ocl::{Buffer, ProQue, SpatialDims};
use std::collections::HashMap;

/// One resident weight matrix on the device (padded Q6_K payload).
struct ResidentMat {
    w_buf: Buffer<u32>,
    x_buf: Buffer<f32>,
    y_buf: Buffer<f32>,
    out_len: u32,
    blocks_per_row: u32,
}

/// OpenCL compute pool for Q6_K matvecs.
pub struct GpuPool {
    pro_que: ProQue,
    mats: HashMap<String, ResidentMat>,
    pub adapter_name: String,
}

/// Every (platform, device) pair OpenCL exposes, labeled
/// "device [platform]" — the candidate list for W1 pinning.
fn all_devices() -> Result<Vec<(ocl::Platform, ocl::Device, String)>> {
    let mut out = Vec::new();
    for p in ocl::Platform::list() {
        let pname = p.name().unwrap_or_else(|_| "unknown".into());
        for d in ocl::Device::list_all(p).unwrap_or_default() {
            let dname = d.name().unwrap_or_else(|_| "unknown".into());
            out.push((p, d, format!("{dname} [{pname}]")));
        }
    }
    Ok(out)
}

/// Pure device selection over labels (testable without a GPU):
/// explicit index wins, then name substring (case-insensitive), then
/// first-found. No match = error listing every candidate, never a
/// silent pick (ouro-wgpu W1 pattern).
fn select_label(labels: &[String], index: Option<usize>, name: Option<&str>) -> Result<usize> {
    if let Some(i) = index {
        return labels.get(i).map(|_| i).ok_or_else(|| {
            anyhow::anyhow!("OURO_GPU_INDEX={i} out of range ({} devices)", labels.len())
        });
    }
    if let Some(want) = name {
        let needle = want.to_lowercase();
        return labels
            .iter()
            .position(|l| l.to_lowercase().contains(&needle))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "OURO_GPU_NAME={want:?} matches none of [{}]",
                    labels.join(", ")
                )
            });
    }
    Ok(0)
}

impl GpuPool {
    /// Init on the picked OpenCL platform/device — OURO_GPU_NAME pins a
    /// device by substring (the tail chooses 1060 vs HD 620), or
    /// OURO_GPU_INDEX as the explicit index; default is first-found.
    /// Errors cleanly when no ICD runtime is installed (CPU-only tails)
    /// — callers degrade to CPU, not panic.
    pub fn new() -> Result<Self> {
        let devices = all_devices()?;
        if devices.is_empty() {
            bail!("no OpenCL devices (ICD missing or no compute driver)");
        }
        let labels: Vec<String> = devices.iter().map(|(_, _, l)| l.clone()).collect();
        let index = select_label(
            &labels,
            std::env::var("OURO_GPU_INDEX").ok().and_then(|s| s.parse().ok()),
            std::env::var("OURO_GPU_NAME").ok().as_deref(),
        )?;
        let (platform, device, label) = devices[index].clone();

        let pro_que = ProQue::builder()
            .src(KERNEL)
            .platform(platform)
            .device(device)
            .dims(1)
            .build()
            .context("OpenCL init failed")?;
        Ok(Self { pro_que, mats: HashMap::new(), adapter_name: label })
    }

    /// Repack 210-byte Q6_K blocks into 212-byte (53 u32) rows and
    /// upload. Same shape rules as ouro-wgpu::upload_q6k.
    pub fn upload_q6k(&mut self, name: &str, payload: &[u8], out_len: usize, in_len: usize) -> Result<()> {
        if !in_len.is_multiple_of(ELEMS_PER_BLOCK)
            || payload.len() != out_len * (in_len / ELEMS_PER_BLOCK) * Q6K_BLOCK
        {
            bail!(
                "upload_q6k {name}: shape mismatch (payload {} out {out_len} in {in_len})",
                payload.len()
            );
        }
        let blocks_per_row = in_len / ELEMS_PER_BLOCK;
        let words = padded_words(payload, out_len, in_len);
        let w_buf = Buffer::<u32>::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
            .len(words.len())
            .copy_host_slice(&words)
            .build()?;
        let x_buf = Buffer::<f32>::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(in_len)
            .build()?;
        let y_buf = Buffer::<f32>::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::flags::MEM_WRITE_ONLY)
            .len(out_len)
            .build()?;
        self.mats.insert(
            name.to_string(),
            ResidentMat { w_buf, x_buf, y_buf, out_len: out_len as u32, blocks_per_row: blocks_per_row as u32 },
        );
        Ok(())
    }

    /// y = W * x for a resident Q6_K matrix.
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let m = self.mats.get(name).ok_or_else(|| anyhow::anyhow!("matrix {name} not resident"))?;
        if x.len() != m.blocks_per_row as usize * ELEMS_PER_BLOCK {
            bail!("matvec {name}: x len {} != in_len {}", x.len(), m.blocks_per_row * 256);
        }
        // SAFETY: x outlives the call; block(true) only makes the
        // enqueue synchronous so the device copy completes before return.
        unsafe { m.x_buf.write(x).block(true) }.enq()?;
        let kernel = self
            .pro_que
            .kernel_builder("q6k_gemv")
            .arg(&m.w_buf)
            .arg(&m.x_buf)
            .arg(&m.y_buf)
            .arg(m.out_len)
            .arg(m.blocks_per_row)
            .global_work_size(SpatialDims::One(m.out_len as usize * 64))
            .local_work_size(SpatialDims::One(64))
            .build()?;
        unsafe { kernel.enq()? };
        let mut out = vec![0.0f32; m.out_len as usize];
        // SAFETY: out is owned and not aliased; block(true) fills it
        // fully before enq returns.
        unsafe { m.y_buf.read(&mut out).block(true) }.enq()?;
        Ok(out)
    }
}

/// Repack 210-byte blocks into the 212-byte (u32-padded) device layout,
/// returned as little-endian u32 words (OpenCL uint).
pub fn padded_words(payload: &[u8], out_len: usize, in_len: usize) -> Vec<u32> {
    let bpr = in_len / ELEMS_PER_BLOCK;
    let mut bytes = vec![0u8; out_len * bpr * Q6K_PAD];
    for r in 0..out_len {
        for b in 0..bpr {
            let src = (r * bpr + b) * Q6K_BLOCK;
            let dst = (r * bpr + b) * Q6K_PAD;
            bytes[dst..dst + Q6K_BLOCK].copy_from_slice(&payload[src..src + Q6K_BLOCK]);
        }
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic test matrix: ql/qh arbitrary, scales kept in a sane
/// int8 range (no inf from the f16 d), d fixed at 1.0. Bytes never
/// need to "mean" anything — the CPU reference reads the same payload.
pub fn deterministic_payload(out_len: usize, in_len: usize) -> Vec<u8> {
    let bpr = in_len / ELEMS_PER_BLOCK;
    let mut payload = vec![0u8; out_len * bpr * Q6K_BLOCK];
    for (i, byte) in payload.iter_mut().enumerate() {
        let b = (i / Q6K_BLOCK) as u32;
        let off = i % Q6K_BLOCK;
        *byte = match off {
            192..=207 => {
                // scales: alternate positive (0..63) and negative
                // (192+s -> int8 -64..-1); no inf/nan from the f16 d
                let s = ((off as u32 * 3 + b * 5) % 64) as u8;
                if (b + off as u32) & 1 == 1 { 192 + s } else { s }
            }
            208 => 0x00,                      // d = f16 1.0 (0x3C00 LE)
            209 => 0x3C,
            _ => ((i * 7 + 13) & 0xFF) as u8, // ql / qh
        };
    }
    payload
}

/// Deterministic activation vector.
pub fn deterministic_x(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
        .collect()
}

/// Cosine similarity — the L1 parity gate.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L1 parity gate: the OpenCL kernel must match the CPU reference
    /// (cos > 0.9999) before anything routes GPU work. Skips cleanly on
    /// machines without an OpenCL runtime (CI, QEMU, GPU-less tails).
    #[test]
    fn test_opencl_q6k_parity_l1() {
        let mut pool = match GpuPool::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skipping OpenCL parity (no runtime): {e:#}");
                return;
            }
        };
        let (out_len, in_len) = (8usize, 512usize);
        let payload = deterministic_payload(out_len, in_len);
        pool.upload_q6k("test", &payload, out_len, in_len).unwrap();
        let x = deterministic_x(in_len);

        let gpu = pool.matvec("test", &x).unwrap();
        let cpu = ouro_cluster::infer::matvec_q(
            &payload,
            ouro_cluster::infer::QuantKind::Q6K,
            out_len,
            in_len,
            &x,
        );
        let c = cosine(&gpu, &cpu);
        eprintln!("adapter: {} | cos: {c:.8}", pool.adapter_name);
        assert!(c > 0.9999, "L1 parity failed: cos = {c}");
    }

    #[test]
    fn test_cosine_identity() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-12);
        assert_eq!(cosine(&v, &[-v[0], -v[1], -v[2]]), -1.0);
    }

    fn device_labels() -> Vec<String> {
        vec![
            "NVIDIA GeForce GTX 1060 6GB [NVIDIA CUDA]".into(),
            "Intel(R) HD Graphics 620 [Intel(R) OpenCL HD Graphics]".into(),
            "NVIDIA GeForce RTX 3060 [NVIDIA CUDA]".into(),
        ]
    }

    #[test]
    fn test_select_default_first() {
        assert_eq!(select_label(&device_labels(), None, None).unwrap(), 0);
    }

    #[test]
    fn test_select_by_name_substring_case_insensitive() {
        // the two-GPU-tail case: pin either silicon deterministically
        assert_eq!(select_label(&device_labels(), None, Some("intel")).unwrap(), 1);
        assert_eq!(select_label(&device_labels(), None, Some("1060")).unwrap(), 0);
        assert_eq!(select_label(&device_labels(), None, Some("rtx 30")).unwrap(), 2);
    }

    #[test]
    fn test_select_by_index_wins() {
        assert_eq!(select_label(&device_labels(), Some(2), Some("intel")).unwrap(), 2);
    }

    #[test]
    fn test_select_errors_list_candidates() {
        let err = select_label(&device_labels(), Some(9), None).unwrap_err().to_string();
        assert!(err.contains("out of range") && err.contains("3 devices"));
        let err = select_label(&device_labels(), None, Some("titan")).unwrap_err().to_string();
        assert!(err.contains("matches none of") && err.contains("GTX 1060"));
    }

    #[test]
    fn test_deterministic_payload_layout() {
        let payload = deterministic_payload(2, 256);
        assert_eq!(payload.len(), 2 * Q6K_BLOCK);
        // d must decode to exactly 1.0: bytes [0x00, 0x3C] little-endian
        assert_eq!(payload[208], 0x00);
        assert_eq!(payload[209], 0x3C);
        // scales stay in the safe int8 window
        for &v in &payload[192..208] {
            assert!((-64..=65).contains(&(v as i8)));
        }
        // padding yields 53 words per block
        assert_eq!(padded_words(&payload, 2, 256).len(), 2 * 53);
    }
}
