//! Core inference kernels: norm, rope, matvec, attention.
//!
//! All matrices are row-major: W[out][in], y = W * x.
//! GGML stores tensors with the input dim contiguous, which matches this.

use super::dequant::{dequant_tq1_row, f16_to_f32, TQ1_0_BLOCK_BYTES};

/// RMSNorm: y = x / sqrt(mean(x^2) + eps) * w
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

/// Apply RoPE in NEOX style (split-half pairing) over `rot` leading dims, in place.
/// `v` is one head vector; `t` is the absolute position.
pub fn rope_neox(v: &mut [f32], t: usize, rot: usize, base: f32) {
    debug_assert_eq!(rot % 2, 0);
    let half = rot / 2;
    for i in 0..half {
        let theta = 1.0 / base.powf((2 * i) as f32 / rot as f32);
        let (cos_t, sin_t) = ((t as f32 * theta).cos(), (t as f32 * theta).sin());
        let (x0, x1) = (v[i], v[i + half]);
        v[i] = x0 * cos_t - x1 * sin_t;
        v[i + half] = x1 * cos_t + x0 * sin_t;
    }
}

/// SiLU activation: x * sigmoid(x).
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place softmax over a slice, returns nothing.
pub fn softmax(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    let inv = 1.0 / s;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Dot product (LLVM auto-vectorizes).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Worker threads for matvec; OURO_N_THREADS overrides, 1 disables MT.
pub fn mt_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("OURO_N_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
            })
    })
}

/// y = W * x with W TQ1_0 packed payload, rows of `in_len` elements.
pub fn matvec_tq(payload: &[u8], out_len: usize, in_len: usize, x: &[f32]) -> Vec<f32> {
    let row_bytes = in_len / QK * TQ1_0_BLOCK_BYTES;
    let mut y = vec![0.0f32; out_len];
    let nt = mt_threads().min(out_len).max(1);
    let chunk = out_len.div_ceil(nt);
    std::thread::scope(|sc| {
        for (ci, blk) in y.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            sc.spawn(move || {
                let mut row = vec![0.0f32; in_len];
                for (j, oy) in blk.iter_mut().enumerate() {
                    dequant_tq1_row(payload, base + j, row_bytes, &mut row);
                    *oy = dot(&row, x);
                }
            });
        }
    });
    y
}

/// y = W * x with W f16 LE payload, rows of `in_len` elements.
pub fn matvec_f16(payload: &[u8], out_len: usize, in_len: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; out_len];
    let nt = mt_threads().min(out_len).max(1);
    let chunk = out_len.div_ceil(nt);
    std::thread::scope(|sc| {
        for (ci, blk) in y.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            sc.spawn(move || {
                for (j, oy) in blk.iter_mut().enumerate() {
                    let row = base + j;
                    let start = row * in_len * 2;
                    let w = &payload[start..start + in_len * 2];
                    let mut s = 0.0f32;
                    for i in 0..in_len {
                        let bits = u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]);
                        s += f16_to_f32(bits) * x[i];
                    }
                    *oy = s;
                }
            });
        }
    });
    y
}

/// Fetch one f16 row (embedding lookup) as f32.
pub fn f16_row(payload: &[u8], row: usize, in_len: usize) -> Vec<f32> {
    let start = row * in_len * 2;
    (0..in_len)
        .map(|i| {
            f16_to_f32(u16::from_le_bytes([
                payload[start + i * 2],
                payload[start + i * 2 + 1],
            ]))
        })
        .collect()
}

/// Elements per quant block (re-exported length scale).
pub const QK: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let y = rmsnorm(&x, &w, 1e-5);
        let ms: f32 = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv = 1.0 / (ms + 1e-5).sqrt();
        assert!((y[0] - 1.0 * inv).abs() < 1e-6);
        assert!((y[3] - 4.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn test_rmsnorm_weight_scale() {
        let x = vec![3.0, 3.0];
        let w = vec![2.0, 0.5];
        let y = rmsnorm(&x, &w, 0.0);
        // mean sq = 9, inv = 1/3
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rope_preserves_norm() {
        let v0: Vec<f32> = (0..8).map(|i| 0.1 * i as f32 + 0.05).collect();
        let mut v = v0.clone();
        rope_neox(&mut v, 5, 8, 500000.0);
        let n0: f32 = v0.iter().map(|x| x * x).sum();
        let n1: f32 = v.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-4, "rope must preserve L2 norm");
        assert_ne!(v, v0, "rope must change values");
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let v0: Vec<f32> = (0..8).map(|i| 0.25 * i as f32).collect();
        let mut v = v0.clone();
        rope_neox(&mut v, 0, 8, 500000.0);
        for i in 0..8 {
            assert!((v[i] - v0[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dot() {
        assert_eq!(dot(&[1.0, 2.0], &[5.0, 6.0]), 17.0);
        assert_eq!(dot(&[0.0; 4], &[1.0; 4]), 0.0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, -10.0];
        softmax(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(v[3] < 1e-4);
        assert!(v[2] > v[0]);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0)).abs() < 1e-9);
        assert!((silu(10.0) - 10.0).abs() < 1e-3);
        assert!((silu(-10.0)).abs() < 1e-3);
    }
}
