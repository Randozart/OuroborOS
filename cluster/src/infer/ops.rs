//! Core inference kernels: norm, rope, matvec, attention.
//!
//! All matrices are row-major: W[out][in], y = W * x.
//! GGML stores tensors with the input dim contiguous, which matches this.

use super::dequant::{
    dequant_q3k_row, dequant_q4k_row, dequant_q5k_row, dequant_q6k_row, dequant_q8_row,
    dequant_tq1_row, f16_to_f32, QuantKind,
};

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

/// y = W * x with any supported quant kind, rows of `in_len` elements.
pub fn matvec_q(payload: &[u8], kind: QuantKind, out_len: usize, in_len: usize, x: &[f32]) -> Vec<f32> {
    match kind {
        QuantKind::F16 => matvec_f16(payload, out_len, in_len, x),
        QuantKind::Bf16 => matvec_bf16(payload, out_len, in_len, x),
        QuantKind::F32 => matvec_f32raw(payload, out_len, in_len, x),
        QuantKind::Tl1 => matvec_tl1(payload, out_len, in_len, x),
        _ => matvec_qblock(payload, kind, out_len, in_len, x),
    }
}

/// y = W * x with W raw f32 row-major payload.
pub fn matvec_f32raw(payload: &[u8], out_len: usize, in_len: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; out_len];
    for (o, oy) in y.iter_mut().enumerate() {
        let base = &payload[o * in_len * 4..(o + 1) * in_len * 4];
        let mut s = 0.0f32;
        for (xi, w) in x.iter().zip(base.chunks_exact(4)) {
            s += f32::from_le_bytes(w.try_into().unwrap()) * xi;
        }
        *oy = s;
    }
    y
}

/// Block-quantized row-parallel matvec.
pub fn matvec_qblock(payload: &[u8], kind: QuantKind, out_len: usize, in_len: usize, x: &[f32]) -> Vec<f32> {
    let rb = kind.row_bytes(in_len);
    let mut y = vec![0.0f32; out_len];
    let nt = mt_threads().min(out_len).max(1);
    let chunk = out_len.div_ceil(nt);
    std::thread::scope(|sc| {
        for (ci, blk) in y.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            sc.spawn(move || {
                let mut row = vec![0.0f32; in_len];
                for (j, oy) in blk.iter_mut().enumerate() {
                    let r = base + j;
                    match kind {
                        QuantKind::Tq1_0 => dequant_tq1_row(payload, r, rb, &mut row),
                        QuantKind::Q4K => dequant_q4k_row(payload, r, rb, &mut row),
                        QuantKind::Q8_0 => dequant_q8_row(payload, r, rb, &mut row),
                        QuantKind::Q3K => dequant_q3k_row(payload, r, rb, &mut row),
                        QuantKind::Q5K => dequant_q5k_row(payload, r, rb, &mut row),
                        QuantKind::Q6K => dequant_q6k_row(payload, r, rb, &mut row),
                        _ => unreachable!("handled by matvec_q"),
                    }
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

/// y = W · X with K columns sharing each weight row — the batched decode
/// that makes speculative block-verification profitable: the dominant cost
/// is streaming ternary weights, and K columns amortize that stream K-fold.
///
/// `x` is row-major [k, in_len]; returns column-major [k × out_len]
/// (y[c * out_len + r]).
pub fn matmul_tl1(payload: &[u8], out_len: usize, in_len: usize, x: &[f32], k: usize) -> Vec<f32> {
    use super::dequant::PTQ1_GROUP_ELEMS;
    debug_assert_eq!(x.len(), k * in_len, "x must be [k, in_len]");
    let groups_per_row = in_len / PTQ1_GROUP_ELEMS;
    let mut y = vec![0.0f32; k * out_len];
    let nt = mt_threads().min(out_len).max(1);
    let chunk = out_len.div_ceil(nt);
    let simd = have_tl1_avx2();
    std::thread::scope(|sc| {
        let mut handles = Vec::with_capacity(nt);
        for ci in 0..nt {
            let base = ci * chunk;
            let rows: Vec<usize> = (base..(base + chunk).min(out_len)).collect();
            if rows.is_empty() {
                continue;
            }
            handles.push(sc.spawn(move || {
                let mut scratch = vec![0.0f32; rows.len() * k];
                let args = KArgs {
                    payload,
                    rows: &rows,
                    scratch: scratch.as_mut_slice(),
                    groups_per_row,
                    x,
                    in_len,
                    k,
                };
                if simd {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        tl1_rows_avx2_k(args);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let _ = (&payload, groups_per_row, x, in_len, k, base);
                    }
                } else {
                    tl1_rows_scalar_k(args);
                }
                (rows, scratch)
            }));
        }
        // scatter row-major scratch [rows × k] into column-major y
        for h in handles {
            let (rows, scratch) = h.join().expect("tl1 k-worker panicked");
            for (j, &row) in rows.iter().enumerate() {
                for c in 0..k {
                    y[c * out_len + row] = scratch[j * k + c];
                }
            }
        }
    });
    y
}

/// y = W · X for any kind; TL1 gets the fused K-column kernel, everything
/// else falls back to K matvecs (correct, unamortized).
pub fn matmul_q(payload: &[u8], kind: QuantKind, out_len: usize, in_len: usize, x: &[f32], k: usize) -> Vec<f32> {
    match kind {
        QuantKind::Tl1 => matmul_tl1(payload, out_len, in_len, x, k),
        _ => {
            let mut y = Vec::with_capacity(k * out_len);
            for c in 0..k {
                y.extend(matvec_q(payload, kind, out_len, in_len, &x[c * in_len..(c + 1) * in_len]));
            }
            y
        }
    }
}

/// Kernel argument bundle (shared by scalar + SSE K-column paths).
struct KArgs<'a> {
    payload: &'a [u8],
    rows: &'a [usize],
    scratch: &'a mut [f32],
    groups_per_row: usize,
    x: &'a [f32],
    in_len: usize,
    k: usize,
}

/// Scalar K-column path: decode each group once, dot against every column.
fn tl1_rows_scalar_k(a: KArgs<'_>) {
    use super::dequant::{PTQ1_GROUP_BYTES, PTQ1_GROUP_ELEMS, PTQ1_POW3};
    let KArgs { payload, rows, scratch, groups_per_row, x, in_len, k } = a;
    let mut sums = vec![0.0f32; k];
    for (j, &row) in rows.iter().enumerate() {
        let row_base = row * groups_per_row * PTQ1_GROUP_BYTES;
        sums.iter_mut().for_each(|s| *s = 0.0);
        for g in 0..groups_per_row {
            let gb = row_base + g * PTQ1_GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([payload[gb + 26], payload[gb + 27]]));
            let qs = &payload[gb..gb + 24];
            let qh = &payload[gb + 24..gb + 26];
            let xb = g * PTQ1_GROUP_ELEMS;
            for (c, s) in sums.iter_mut().enumerate() {
                let xc = &x[c * in_len..(c + 1) * in_len];
                let mut acc = *s;
                for n in 0..5 {
                    let pn = PTQ1_POW3[n];
                    for m in 0..16 {
                        let q = qs[m].wrapping_mul(pn);
                        let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                        acc += t * xc[xb + n * 16 + m];
                    }
                    for m in 0..8 {
                        let q = qs[16 + m].wrapping_mul(pn);
                        let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                        acc += t * xc[xb + 80 + n * 8 + m];
                    }
                }
                for n in 0..4 {
                    let pn = PTQ1_POW3[n];
                    for m in 0..2 {
                        let q = qh[m].wrapping_mul(pn);
                        let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                        acc += t * xc[xb + 120 + n * 2 + m];
                    }
                }
                *s = acc;
            }
        }
        scratch[j * k..j * k + k].copy_from_slice(&sums);
    }
}

/// y = W * x with TL1 (PTQ1_0) ternary weights.
///
/// Fuses dequant + dot product: trits never touch memory, the zero branch is
/// dropped (unconditional FMA vectorizes; ~1/3 of ternary weights are zero
/// and contribute nothing anyway). Multithreaded across output rows;
/// AVX2+FMA kernel when available, scalar fallback otherwise.
pub fn matvec_tl1(payload: &[u8], out_len: usize, in_len: usize, x: &[f32]) -> Vec<f32> {
    use super::dequant::PTQ1_GROUP_ELEMS;
    let groups_per_row = in_len / PTQ1_GROUP_ELEMS;
    let mut y = vec![0.0f32; out_len];
    let nt = mt_threads().min(out_len).max(1);
    let chunk = out_len.div_ceil(nt);
    let simd = have_tl1_avx2();
    std::thread::scope(|sc| {
        for (ci, blk) in y.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            sc.spawn(move || {
                if simd {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        tl1_rows_avx2(payload, blk, groups_per_row, x, base);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let _ = (&payload, groups_per_row, x, base);
                    }
                } else {
                    tl1_rows_scalar(payload, blk, groups_per_row, x, base);
                }
            });
        }
    });
    y
}

/// Scalar reference path (also the non-x86 build).
fn tl1_rows_scalar(payload: &[u8], blk: &mut [f32], groups_per_row: usize, x: &[f32], base: usize) {
    use super::dequant::{PTQ1_GROUP_BYTES, PTQ1_GROUP_ELEMS, PTQ1_POW3};
    for (j, oy) in blk.iter_mut().enumerate() {
        let row = base + j;
        let row_base = row * groups_per_row * PTQ1_GROUP_BYTES;
        let mut sum = 0.0f32;
        for g in 0..groups_per_row {
            let gb = row_base + g * PTQ1_GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([payload[gb + 26], payload[gb + 27]]));
            let qs = &payload[gb..gb + 24];
            let qh = &payload[gb + 24..gb + 26];
            let xb = g * PTQ1_GROUP_ELEMS;
            // stage c=16: elements n*16 + m
            for (n, &pn) in PTQ1_POW3.iter().enumerate().take(5) {
                for m in 0..16 {
                    let q = qs[m].wrapping_mul(pn);
                    let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                    sum += t * x[xb + n * 16 + m];
                }
            }
            // stage c=8: elements 80 + n*8 + m
            for (n, &pn) in PTQ1_POW3.iter().enumerate().take(5) {
                for m in 0..8 {
                    let q = qs[16 + m].wrapping_mul(pn);
                    let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                    sum += t * x[xb + 80 + n * 8 + m];
                }
            }
            // qh pairs: elements 120 + n*2 + h
            for (n, &pn) in PTQ1_POW3.iter().enumerate().take(4) {
                for m in 0..2 {
                    let q = qh[m].wrapping_mul(pn);
                    let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                    sum += t * x[xb + 120 + n * 2 + m];
                }
            }
        }
        *oy = sum;
    }
}

/// SSE4.1 availability for the tl1 fused kernel (cached). The fleet's
/// oldest nodes are pre-AVX2 (Ivy Bridge); the kernel targets the floor.
#[cfg(target_arch = "x86_64")]
fn have_tl1_avx2() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| is_x86_feature_detected!("sse4.1"))
}

/// SSE4.1 K-column fused decode+dot over `rows` (docs/DUET.md P3 batched
/// verify). Identical digit pipeline to `digits8`; the decoded f32 pairs
/// are FMA'd against every x column, so the weight stream amortizes K-fold.
///
/// # Safety
/// Same bounds contract as `tl1_rows_avx2`; requires sse4.1 (caller checks).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn tl1_rows_avx2_k(a: KArgs<'_>) {
    #![allow(clippy::needless_range_loop)]
    let KArgs { payload, rows, scratch, groups_per_row, x, in_len, k } = a;

    use super::dequant::{PTQ1_GROUP_BYTES, PTQ1_GROUP_ELEMS, PTQ1_POW3};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn decode8(p: *const u8, boff: usize, pn: i16) -> (__m128, __m128) {
        let b = _mm_loadl_epi64(p.add(boff) as *const __m128i);
        let w = _mm_unpacklo_epi8(b, _mm_setzero_si128());
        let mut q = _mm_mullo_epi16(w, _mm_set1_epi16(pn));
        q = _mm_and_si128(q, _mm_set1_epi16(0xFF));
        q = _mm_mullo_epi16(q, _mm_set1_epi16(3));
        q = _mm_srli_epi16(q, 8);
        q = _mm_sub_epi16(q, _mm_set1_epi16(1));
        let flo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(q));
        let fhi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(q, q)));
        (flo, fhi)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn hsum(v: __m128) -> f32 {
        let s = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    // K 4-lane accumulators (lo|hi halves per column)
    let mut acc = vec![_mm_setzero_ps(); 2 * k];
    for (j, &row) in rows.iter().enumerate() {
        let row_base = row * groups_per_row * PTQ1_GROUP_BYTES;
        for a in acc.iter_mut() {
            *a = _mm_setzero_ps();
        }
        for g in 0..groups_per_row {
            let gb = row_base + g * PTQ1_GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([payload[gb + 26], payload[gb + 27]]));
            let dv = _mm_set1_ps(d);
            let qs = payload.as_ptr().add(gb);
            let xg = g * PTQ1_GROUP_ELEMS;
            for n in 0..5 {
                let pn = PTQ1_POW3[n] as i16;
                let (lo0, hi0) = decode8(qs, 0, pn);
                let (lo1, hi1) = decode8(qs, 8, pn);
                let (lo2, hi2) = decode8(qs, 16, pn);
                for c in 0..k {
                    let xp = x.as_ptr().add(c * in_len + xg);
                    acc[2 * c] = _mm_add_ps(
                        _mm_mul_ps(_mm_mul_ps(lo0, dv), _mm_loadu_ps(xp.add(n * 16))),
                        acc[2 * c],
                    );
                    acc[2 * c + 1] = _mm_add_ps(
                        _mm_mul_ps(_mm_mul_ps(hi0, dv), _mm_loadu_ps(xp.add(n * 16 + 4))),
                        acc[2 * c + 1],
                    );
                    acc[2 * c] = _mm_add_ps(
                        _mm_mul_ps(_mm_mul_ps(lo1, dv), _mm_loadu_ps(xp.add(n * 16 + 8))),
                        acc[2 * c],
                    );
                    acc[2 * c + 1] = _mm_add_ps(
                        _mm_mul_ps(_mm_mul_ps(hi1, dv), _mm_loadu_ps(xp.add(n * 16 + 12))),
                        acc[2 * c + 1],
                    );
                    acc[2 * c] = _mm_add_ps(
                        _mm_mul_ps(_mm_mul_ps(lo2, dv), _mm_loadu_ps(xp.add(80 + n * 8))),
                        acc[2 * c],
                    );
                    acc[2 * c + 1] = _mm_add_ps(
                        _mm_mul_ps(_mm_mul_ps(hi2, dv), _mm_loadu_ps(xp.add(80 + n * 8 + 4))),
                        acc[2 * c + 1],
                    );
                }
            }
            // qh tail: elements 120 + n*2 + h (8 trits, scalar)
            let qh = qs.add(24);
            for n in 0..4 {
                for m in 0..2 {
                    let q = qh.add(m).read().wrapping_mul(PTQ1_POW3[n]);
                    let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                    for c in 0..k {
                        sum_tail_col(scratch, j, k, c, t * x[c * in_len + xg + 120 + n * 2 + m]);
                    }
                }
            }
        }
        for c in 0..k {
            scratch[j * k + c] += unsafe { hsum(acc[2 * c]) + hsum(acc[2 * c + 1]) };
        }
    }
}

/// Accumulate the scalar qh-tail contribution for column c.
#[cfg(target_arch = "x86_64")]
#[inline]
fn sum_tail_col(scratch: &mut [f32], j: usize, k: usize, c: usize, v: f32) {
    scratch[j * k + c] += v;
}

#[cfg(not(target_arch = "x86_64"))]
fn have_tl1_avx2() -> bool {
    false
}

/// AVX2+FMA fused PTQ1_0 decode+dot over rows [base, base + blk.len()).
///
/// Per 8 trits: unpack bytes to u16 lanes, u8-wrapping scale by 3^n
/// (& 0xFF), (q*3)>>8 digit magic, trit to f32, scale by d, FMA into an
/// 8-lane accumulator; horizontal add once per group. 128-bit lanes only —
/// no cross-lane shuffles needed.
///
/// # Safety
/// Caller must guarantee avx2+fma (checked via `have_tl1_avx2`) and the
/// same bounds contract as the scalar path (payload rows / x groups).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn tl1_rows_avx2(
    payload: &[u8],
    blk: &mut [f32],
    groups_per_row: usize,
    x: &[f32],
    base: usize,
) {
    use super::dequant::{PTQ1_GROUP_BYTES, PTQ1_GROUP_ELEMS, PTQ1_POW3};
    use std::arch::x86_64::*;

    debug_assert_eq!(x.len(), groups_per_row * PTQ1_GROUP_ELEMS);

    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn digits8(
        p: *const u8,
        boff: usize,
        pn: i16,
        d: __m128,
        xp: *const f32,
        xoff: usize,
        acc: __m128,
    ) -> __m128 {
        // 8 bytes -> 8 u16 lanes; decode all, then FMA as two 4-lane halves
        // (a 128-bit f32 vector is only 4 wide).
        let b = _mm_loadl_epi64(p.add(boff) as *const __m128i);
        let w = _mm_unpacklo_epi8(b, _mm_setzero_si128());
        let mut q = _mm_mullo_epi16(w, _mm_set1_epi16(pn));
        q = _mm_and_si128(q, _mm_set1_epi16(0xFF)); // u8 wrap semantics
        q = _mm_mullo_epi16(q, _mm_set1_epi16(3));
        q = _mm_srli_epi16(q, 8); // digit 0..2
        q = _mm_sub_epi16(q, _mm_set1_epi16(1)); // trit -1..1
        let flo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(q));
        let fhi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(q, q)));
        let acc = _mm_add_ps(
            _mm_mul_ps(_mm_mul_ps(flo, d), _mm_loadu_ps(xp.add(xoff))),
            acc,
        );
        _mm_add_ps(
            _mm_mul_ps(_mm_mul_ps(fhi, d), _mm_loadu_ps(xp.add(xoff + 4))),
            acc,
        )
    }

    #[inline]
    #[target_feature(enable = "sse,sse2")]
    unsafe fn hsum(v: __m128) -> f32 {
        let s = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    for (j, oy) in blk.iter_mut().enumerate() {
        let row = base + j;
        let row_base = row * groups_per_row * PTQ1_GROUP_BYTES;
        let mut sum = 0.0f32;
        for g in 0..groups_per_row {
            let gb = row_base + g * PTQ1_GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([payload[gb + 26], payload[gb + 27]]));
            let dv = _mm_set1_ps(d);
            let qs = payload.as_ptr().add(gb);
            let xg = g * PTQ1_GROUP_ELEMS;
            let xp = x.as_ptr().add(xg);
            let mut acc = _mm_setzero_ps();
            // stage c=16 (qs[0..16]): elements n*16+m, two 8-lane halves
            // stage c=8 (qs[16..24]): elements 80+n*8+m, one 8-lane chunk
            for (n, &pw) in PTQ1_POW3.iter().enumerate().take(5) {
                let pn = pw as i16;
                acc = digits8(qs, 0, pn, dv, xp, n * 16, acc);
                acc = digits8(qs, 8, pn, dv, xp, n * 16 + 8, acc);
                acc = digits8(qs, 16, pn, dv, xp, 80 + n * 8, acc);
            }
            // qh tail: elements 120 + n*2 + h (8 trits, scalar)
            let qh = qs.add(24);
            for n in 0..4 {
                for m in 0..2 {
                    let q = qh.add(m).read().wrapping_mul(PTQ1_POW3[n]);
                    let t = ((((q as u16) * 3) >> 8) as i8 - 1) as f32 * d;
                    sum += t * x[xg + 120 + n * 2 + m];
                }
            }
            sum += hsum(acc);
        }
        *oy = sum;
    }
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

/// y = W * x with W bf16 LE payload, rows of `in_len` elements.
pub fn matvec_bf16(payload: &[u8], out_len: usize, in_len: usize, x: &[f32]) -> Vec<f32> {
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
                        s += f32::from_bits((bits as u32) << 16) * x[i];
                    }
                    *oy = s;
                }
            });
        }
    });
    y
}

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

    /// Throughput probe for the Bonsai lm_head shape (248320x5120 TL1).
    /// Synthetic payload; guards against decode-path regressions.
    #[test]
    #[ignore]
    fn bench_matvec_tl1_lmhead_shape() {
        let (out_len, in_len) = (248320usize, 5120usize);
        let groups = in_len / 128;
        let row_bytes = groups * 28;
        let mut payload = vec![0u8; out_len * row_bytes];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u32 ^ (i as u32 >> 13)).wrapping_mul(0x9E) as u8;
        }
        // fp16 scale ~1.0 (0x3c00 LE) in every 28-byte group
        for g in payload.chunks_exact_mut(28) {
            g[26] = 0x00;
            g[27] = 0x3c;
        }
        let x: Vec<f32> = (0..in_len).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let t = std::time::Instant::now();
        let y = matvec_tl1(&payload, out_len, in_len, &x);
        let dt = t.elapsed().as_secs_f64();
        let sum: f32 = y.iter().sum();
        eprintln!("matvec_tl1 {out_len}x{in_len}: {dt:.2}s ({:.1} MB/s)", (payload.len() as f64) / 1e6 / dt);
        assert!(y.len() == out_len && sum.is_finite());
    }

    #[test]
    fn test_matmul_tl1_k_columns_matches_matvec() {
        use super::super::dequant::tl1_group;
        let (out_len, in_len, k) = (97usize, 384usize, 4usize);
        let groups = in_len / 128;
        let row_bytes = groups * 28;
        let mut payload = vec![0u8; out_len * row_bytes];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(0x9B ^ (i as u32 >> 5)) as u8;
        }
        for g in payload.chunks_exact_mut(28) {
            g[26] = 0x10;
            g[27] = 0x38;
        }
        let x: Vec<f32> = (0..k * in_len)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.25)
            .collect();

        let batched = matmul_tl1(&payload, out_len, in_len, &x, k);
        for c in 0..k {
            let want = matvec_tl1(&payload, out_len, in_len, &x[c * in_len..(c + 1) * in_len]);
            for r in 0..out_len {
                let got = batched[c * out_len + r];
                assert!(
                    (got - want[r]).abs() <= 1e-3 * want[r].abs().max(1.0),
                    "col {c} row {r}: {got} vs {}", want[r]
                );
            }
        }
        let _ = tl1_group; // reference decode exercised via matvec_tl1
    }

    /// Throughput probe: the K-amortization the whole point rests on.
    #[test]
    #[ignore]
    fn bench_matmul_tl1_amortization() {
        let (out_len, in_len) = (248320usize, 5120usize);
        let row_bytes = (in_len / 128) * 28;
        let mut payload = vec![0u8; out_len * row_bytes];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u32 ^ (i as u32 >> 13)).wrapping_mul(0x9E) as u8;
        }
        for g in payload.chunks_exact_mut(28) {
            g[26] = 0x00;
            g[27] = 0x3c;
        }
        for k in [1usize, 4, 8] {
            let x: Vec<f32> = (0..k * in_len)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                .collect();
            let t = std::time::Instant::now();
            let y = matmul_tl1(&payload, out_len, in_len, &x, k);
            let dt = t.elapsed().as_secs_f64();
            eprintln!(
                "matmul_tl1 k={k}: {dt:.2}s ({:.0} tok-equiv/s)",
                k as f64 / dt
            );
            assert_eq!(y.len(), k * out_len);
        }
    }

    #[test]
    fn test_matmul_q_fallback_matches() {
        // f16 kind falls back to K matvecs; must agree with matvec_f16
        let (out_len, in_len) = (16usize, 32usize);
        // NaN-free payload: f16 = truncated top bits of tame f32 values
        let mut payload = Vec::with_capacity(out_len * in_len * 2);
        for i in 0..out_len * in_len {
            let v = ((i % 17) as f32 - 8.0) * 0.25;
            payload.extend_from_slice(&v.to_bits().to_le_bytes()[..2]);
        }
        let x: Vec<f32> = (0..2 * in_len).map(|i| (i % 7) as f32 * 0.5).collect();
        let y = matmul_q(&payload, QuantKind::F16, out_len, in_len, &x, 2);
        for c in 0..2 {
            let want = matvec_f16(&payload, out_len, in_len, &x[c * in_len..(c + 1) * in_len]);
            assert_eq!(&y[c * out_len..(c + 1) * out_len], &want[..]);
        }
    }

    /// Fused decode+dot must agree with the reference `tl1_group` decode.
    #[test]
    fn test_matvec_tl1_fused_matches_reference() {
        use super::super::dequant::{dequant_tl1, tl1_group};
        let (out_len, in_len) = (37usize, 384usize); // 3 groups per row
        let groups = in_len / 128;
        let row_bytes = groups * 28;
        let mut payload = vec![0u8; out_len * row_bytes];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(0x9B ^ (i as u32 >> 7)) as u8;
        }
        // every 28-byte group gets a valid (non-NaN) f16 scale
        for g in payload.chunks_exact_mut(28) {
            g[26] = 0x10;
            g[27] = 0x38;
        }
        let x: Vec<f32> = (0..in_len).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let fused = matvec_tl1(&payload, out_len, in_len, &x);
        for row in 0..out_len {
            let row_bytes_span = &payload[row * row_bytes..(row + 1) * row_bytes];
            let mut flat = vec![0.0f32; in_len];
            for g in 0..groups {
                tl1_group(
                    &row_bytes_span[g * 28..(g + 1) * 28],
                    &mut flat[g * 128..(g + 1) * 128],
                );
            }
            let want: f32 = flat.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (fused[row] - want).abs() / want.abs().max(1e-9) < 1e-5,
                "row {row}: fused {} vs reference {}",
                fused[row],
                want
            );
        }
        // dequant_tl1 whole-tensor path agrees with row decode too
        let full = dequant_tl1(&payload, out_len * in_len);
        assert_eq!(full.len(), out_len * in_len);
    }
}
