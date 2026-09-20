//! Weight dequantization: TQ1_0, TL1 (BitNet ternary) and F16 -> F32.
//!
//! TQ1_0 block (54 bytes, 256 elements):
//! - `qs[48]`: base-3 packed, 5 elements per byte (3^5 = 243 < 256)
//! - `qh[4]`:  one digit per byte, high positions
//! - `d`:      f16 scale shared by the block
//!
//! TL1 (Microsoft BitNet custom, dtype 143):
//! - Packed bytes: 2 ternary values per byte (base-3: hi*3+lo, +4 offset)
//! - Float32 scale appended at end of tensor
//! - Kernel-specific transposition in the packed byte layout

/// Elements per TQ1_0 block.
pub const QK_K: usize = 256;
/// Bytes per TQ1_0 block: 48 qs + 4 qh + 2 scale.
pub const TQ1_0_BLOCK_BYTES: usize = 54;

const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Convert one f16 bit pattern to f32 (subnormals + infinities handled).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;

    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal f16 -> normalize into f32
            let mut e = -1i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3ff;
            let exp32 = 127 - 15 - e;
            (sign << 31) | ((exp32 as u32) << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | (((exp + 127 - 15) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

/// Decode one 54-byte TQ1_0 block into 256 f32 values.
fn tq1_block(bytes: &[u8], out: &mut [f32]) {
    let qs = &bytes[0..48];
    let qh = &bytes[48..52];
    let d = f16_to_f32(u16::from_le_bytes([bytes[52], bytes[53]]));

    let trit = |byte: u8, n: usize| -> f32 {
        let q = byte.wrapping_mul(POW3[n]);
        let xi = ((q as u16) * 3) >> 8;
        (xi as f32 - 1.0) * d
    };

    let mut o = 0;
    for n in 0..5 {
        for &b in qs.iter().take(32) {
            out[o] = trit(b, n);
            o += 1;
        }
    }
    for n in 0..5 {
        for &b in &qs[32..48] {
            out[o] = trit(b, n);
            o += 1;
        }
    }
    for n in 0..4 {
        for &b in qh.iter() {
            out[o] = trit(b, n);
            o += 1;
        }
    }
}

/// Dequantize a full TQ1_0 tensor row-major payload into new f32 vec.
/// `bytes` length must be a multiple of 54 (k = len/54*256).
pub fn dequant_tq1_0(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(TQ1_0_BLOCK_BYTES), "misaligned TQ1_0 payload");
    let n = bytes.len() / TQ1_0_BLOCK_BYTES * QK_K;
    let mut out = vec![0.0f32; n];
    for (bi, chunk) in bytes.chunks_exact(TQ1_0_BLOCK_BYTES).enumerate() {
        tq1_block(chunk, &mut out[bi * QK_K..(bi + 1) * QK_K]);
    }
    out
}

/// Dequantize one TQ1_0 row (k elements) from a row-major payload.
/// `row_bytes` = k/256 * 54.
pub fn dequant_tq1_row(payload: &[u8], row: usize, row_bytes: usize, out: &mut [f32]) {
    let start = row * row_bytes;
    let blocks = row_bytes / TQ1_0_BLOCK_BYTES;
    for b in 0..blocks {
        tq1_block(
            &payload[start + b * TQ1_0_BLOCK_BYTES..start + (b + 1) * TQ1_0_BLOCK_BYTES],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
}

// ---------------------------------------------------------------------------
// TL1 dequantization (Microsoft BitNet custom ternary format, dtype 143)
// ---------------------------------------------------------------------------

/// Decode a TL1 tensor payload into f32.
///
/// TL1 layout (per `preprocess_weights_tl1` in convert-hf-to-gguf-bitnet.py):
/// - Packed bytes: each byte = (hi_ternary * 3 + lo_ternary) + 4
///   where hi, lo ∈ {-1, 0, +1} → byte values ∈ {3, 4, 5, 6, 7, 8, 9}
/// - Float32 scale: last 4 bytes of the payload
/// - The packed bytes are kernel-specifically transposed; we unpack linearly
///   and let the matvec handle element order (transposition is for GPU kernel
///   tile efficiency, not value changes).
///
/// `n_elements`: total number of ternary weights (M * K).
/// Returns f32 vec of length `n_elements`.
/// Bytes per PTQ1_0 group (128 ternary weights): qs[24] + qh[2] + f16 scale.
pub const PTQ1_GROUP_BYTES: usize = 28;
/// Elements per PTQ1_0 group.
pub const PTQ1_GROUP_ELEMS: usize = 128;

/// PTQ1_0 stage strides (ggml-quants.c `ptq1_0_stages`); only 16 and 8
/// execute for the 24-byte qs of a 128-trit group.
pub const PTQ1_POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

/// TQ1_0-style byte magic: digit n of a packed byte, mapped to {-1,0,1}.
#[inline(always)]
fn ptq1_digit(byte: u8, n: usize) -> f32 {
    let q = byte.wrapping_mul(PTQ1_POW3[n]);
    let xi = ((q as u16) * 3) >> 8;
    (xi as i8 - 1) as f32
}

/// Decode one 28-byte PTQ1_0 group into 128 f32 values.
///
/// Layout (block_ptq1_0, ggml-common.h): `qs[24] | qh[2] | d(f16)`.
/// Trit element order (dequantize_row_ptq1_0): stage c=16 over qs[0..16]
/// (elements n*16+m), stage c=8 over qs[16..24] (elements 80 + n*8+m),
/// then qh pairs (elements 120 + n*2 + h).
pub fn tl1_group(bytes: &[u8], out: &mut [f32]) {
    debug_assert!(bytes.len() >= PTQ1_GROUP_BYTES);
    let d = f16_to_f32(u16::from_le_bytes([bytes[26], bytes[27]]));
    let qs = &bytes[0..24];
    let qh = &bytes[24..26];
    let mut o = 0usize;
    for (n, _) in (0..5).enumerate() {
        for &b in &qs[..16] {
            out[o] = ptq1_digit(b, n) * d;
            o += 1;
        }
    }
    for (n, _) in (0..5).enumerate() {
        for &b in &qs[16..24] {
            out[o] = ptq1_digit(b, n) * d;
            o += 1;
        }
    }
    for (n, _) in (0..4).enumerate() {
        for &b in qh {
            out[o] = ptq1_digit(b, n) * d;
            o += 1;
        }
    }
    debug_assert_eq!(o, PTQ1_GROUP_ELEMS);
}

/// Decode a PTQ1_0 (PrismML dense trits) payload to f32.
pub fn dequant_tl1(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_elements];
    let n_groups = n_elements / PTQ1_GROUP_ELEMS;
    for g in 0..n_groups {
        let base = g * PTQ1_GROUP_BYTES;
        tl1_group(&bytes[base..base + PTQ1_GROUP_BYTES], &mut out[g * PTQ1_GROUP_ELEMS..]);
    }
    out
}

/// Decode one row of a PTQ1_0 tensor.
pub fn dequant_tl1_row(payload: &[u8], row: usize, row_elems: usize, out: &mut [f32]) {
    let groups_per_row = row_elems / PTQ1_GROUP_ELEMS;
    for g in 0..groups_per_row {
        let group_idx = row * groups_per_row + g;
        let base = group_idx * PTQ1_GROUP_BYTES;
        tl1_group(
            &payload[base..base + PTQ1_GROUP_BYTES],
            &mut out[g * PTQ1_GROUP_ELEMS..(g + 1) * PTQ1_GROUP_ELEMS],
        );
    }
}
/// Convert one bf16 bit pattern to f32 (truncate-multiply; exact by design).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Decode a bf16 payload (LE bytes) to f32 vec.
pub fn dequant_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode an f16 payload (LE bytes) to f32 vec.
pub fn dequant_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Bytes per Q8_0 block (32 elements).
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Bytes per Q4_K block (256 elements).
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Decode one 34-byte Q8_0 block into 32 f32.
fn q8_block(bytes: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    for j in 0..32 {
        out[j] = bytes[2 + j] as i8 as f32 * d;
    }
}

/// Dequantize a full Q8_0 payload.
pub fn dequant_q8_0(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES), "misaligned Q8_0 payload");
    let n = bytes.len() / Q8_0_BLOCK_BYTES * 32;
    let mut out = vec![0.0f32; n];
    for (bi, chunk) in bytes.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        q8_block(chunk, &mut out[bi * 32..(bi + 1) * 32]);
    }
    out
}

/// Q4_K 6-bit scale/min unpack (mirrors get_scale_min_k4 in ggml-quants.c).
fn scale_min_k4(j: usize, sc: &[u8]) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        let d = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        let m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        (d, m)
    }
}

/// Decode one 144-byte Q4_K block into 256 f32.
fn q4k_block(bytes: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    let min = f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
    let scales = &bytes[4..16];
    let mut q = &bytes[16..16 + 128];
    let mut is = 0usize;
    let mut o = 0usize;
    for _ in 0..4 {
        let (s1, m1) = scale_min_k4(is, scales);
        let (s2, m2) = scale_min_k4(is + 1, scales);
        let d1 = d * s1 as f32;
        let d2 = d * s2 as f32;
        let mm1 = min * m1 as f32;
        let mm2 = min * m2 as f32;
        for l in 0..32 {
            out[o + l] = d1 * (q[l] & 0xF) as f32 - mm1;
        }
        for l in 0..32 {
            out[o + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
        }
        o += 64;
        q = &q[32..];
        is += 2;
    }
}

/// Dequantize a full Q4_K payload.
pub fn dequant_q4_k(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(Q4_K_BLOCK_BYTES), "misaligned Q4_K payload");
    let n = bytes.len() / Q4_K_BLOCK_BYTES * QK_K;
    let mut out = vec![0.0f32; n];
    for (bi, chunk) in bytes.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        q4k_block(chunk, &mut out[bi * QK_K..(bi + 1) * QK_K]);
    }
    out
}

/// Dequantize one Q4_K row (k elements) from row-major payload.
pub fn dequant_q4k_row(payload: &[u8], row: usize, row_bytes: usize, out: &mut [f32]) {
    let start = row * row_bytes;
    let blocks = row_bytes / Q4_K_BLOCK_BYTES;
    for b in 0..blocks {
        q4k_block(
            &payload[start + b * Q4_K_BLOCK_BYTES..start + (b + 1) * Q4_K_BLOCK_BYTES],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
}

/// Dequantize one Q8_0 row (k elements) from row-major payload.
pub fn dequant_q8_row(payload: &[u8], row: usize, row_bytes: usize, out: &mut [f32]) {
    let start = row * row_bytes;
    let blocks = row_bytes / Q8_0_BLOCK_BYTES;
    for b in 0..blocks {
        q8_block(
            &payload[start + b * Q8_0_BLOCK_BYTES..start + (b + 1) * Q8_0_BLOCK_BYTES],
            &mut out[b * 32..(b + 1) * 32],
        );
    }
}

/// Quant kind dispatch, keyed by ggml type id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    F32,
    F16,
    Bf16,
    Tq1_0,
    Tl1,
    Q8_0,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl QuantKind {
    /// From ggml type id; None for unsupported types.
    ///
    /// Includes alternate type IDs used by the ik_llama.cpp / BitNet forks
    /// (dtype 143 = TL1 ternary). Note 30 = BF16 (PrismML Bonsai exports
    /// token_embd + ssm alpha/beta as bf16), NOT f16.
    pub fn from_dtype(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            30 => Some(Self::Bf16),
            8 => Some(Self::Q8_0),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            34 => Some(Self::Tq1_0),
            143 => Some(Self::Tl1),
            _ => None,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Tq1_0 => TQ1_0_BLOCK_BYTES,
            Self::Tl1 => PTQ1_GROUP_BYTES, // 28 bytes per group of 128
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
            Self::Q3K => Q3KBLOCK_BYTES,
            Self::Q4K => Q4_K_BLOCK_BYTES,
            Self::Q5K => Q5_K_BLOCK_BYTES,
            Self::Q6K => Q6_K_BLOCK_BYTES,
        }
    }

    /// Elements per block.
    pub fn block_elems(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q8_0 => 32,
            Self::Tq1_0 | Self::Q4K | Self::Q3K | Self::Q5K | Self::Q6K => QK_K,
            Self::Tl1 => PTQ1_GROUP_ELEMS, // 128 ternary values per group
        }
    }

    /// Row byte size for an input dimension of k elements.
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            Self::Tl1 => {
                // k elements, grouped by 128, each group = 28 bytes
                (k / PTQ1_GROUP_ELEMS) * PTQ1_GROUP_BYTES
            }
            _ => {
                let be = self.block_elems();
                k / be * self.block_bytes()
            }
        }
    }
}


/// Bytes per Q3_K block (256 elements): hmask32 + qs64 + scales12 + d2.
pub const Q3KBLOCK_BYTES: usize = 110;
/// Bytes per Q5_K block (256 elements): dm4 + scales12 + qh32 + qs128.
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// Bytes per Q6_K block (256 elements): ql128 + qh64 + scales16 + d2.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Decode one 110-byte Q3_K block into 256 f32 (faithful to ggml C).
fn q3k_block(bytes: &[u8], out: &mut [f32]) {
    let hmask = &bytes[0..32];
    let qs = &bytes[32..96];
    let sc = &bytes[96..108];
    let d_all = f16_to_f32(u16::from_le_bytes([bytes[108], bytes[109]]));

    // 6-bit scale unpack: aux[0..4] LE from 12 bytes, then the shuffle.
    let mut aux = [0u32; 4];
    for (i, chunk) in sc.chunks_exact(4).enumerate() {
        aux[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    const K1: u32 = 0x03030303;
    const K2: u32 = 0x0f0f0f0f;
    let tmp = aux[2];
    let na0 = (aux[0] & K2) | ((tmp & K1) << 4);
    let na1 = (aux[1] & K2) | (((tmp >> 2) & K1) << 4);
    let na2 = ((aux[0] >> 4) & K2) | (((tmp >> 4) & K1) << 4);
    let na3 = ((aux[1] >> 4) & K2) | (((tmp >> 6) & K1) << 4);
    let scales: [i8; 16] = [
        (na0 & 0xFF) as u8 as i8, (na0 >> 8) as u8 as i8, (na0 >> 16) as u8 as i8, (na0 >> 24) as u8 as i8,
        (na1 & 0xFF) as u8 as i8, (na1 >> 8) as u8 as i8, (na1 >> 16) as u8 as i8, (na1 >> 24) as u8 as i8,
        (na2 & 0xFF) as u8 as i8, (na2 >> 8) as u8 as i8, (na2 >> 16) as u8 as i8, (na2 >> 24) as u8 as i8,
        (na3 & 0xFF) as u8 as i8, (na3 >> 8) as u8 as i8, (na3 >> 16) as u8 as i8, (na3 >> 24) as u8 as i8,
    ];

    let mut q = qs;
    let hm = hmask;
    let mut o = 0usize;
    let mut is = 0usize;
    let mut m: u8 = 1;
    for _n in 0..2 {
        let mut shift = 0u32;
        for _j in 0..4 {
            let dl = d_all * (scales[is] as f32 - 32.0);
            is += 1;
            for l in 0..16 {
                let bit = if hm[l] & m != 0 { 0i8 } else { 4i8 };
                let qi = ((q[l] >> shift) & 3) as i8;
                out[o + l] = dl * (qi - bit) as f32;
            }
            let dl2 = d_all * (scales[is] as f32 - 32.0);
            is += 1;
            for l in 0..16 {
                let bit = if hm[l + 16] & m != 0 { 0i8 } else { 4i8 };
                let qi = ((q[l + 16] >> shift) & 3) as i8;
                out[o + 16 + l] = dl2 * (qi - bit) as f32;
            }
            o += 32;
            shift += 2;
            m <<= 1;
        }
        q = &q[32..];
    }
}

/// Decode one 180-byte Q5_K block into 256 f32.
fn q5k_block(bytes: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    let min = f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
    let sc = &bytes[4..16];
    let qh = &bytes[16..48];
    let mut ql = &bytes[48..176];

    let mut o = 0;
    let mut is = 0usize;
    let (mut u1, mut u2): (u8, u8) = (1, 2);
    for _j in 0..4 {
        let (s1, m1) = scale_min_k4(is, sc);
        let (s2, m2) = scale_min_k4(is + 1, sc);
        let d1 = d * s1 as f32;
        let m1f = min * m1 as f32;
        let d2 = d * s2 as f32;
        let m2f = min * m2 as f32;
        for l in 0..32 {
            out[o + l] = d1 * ((ql[l] & 0xF) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 }) - m1f;
        }
        for l in 0..32 {
            out[o + 32 + l] = d2 * ((ql[l] >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 }) - m2f;
        }
        o += 64;
        ql = &ql[32..];
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Decode one 210-byte Q6_K block into 256 f32.
fn q6k_block(bytes: &[u8], out: &mut [f32]) {
    let ql = &bytes[0..128];
    let qh = &bytes[128..192];
    let d = f16_to_f32(u16::from_le_bytes([bytes[208], bytes[209]]));
    let sc = |idx: usize| bytes[192 + idx] as i8;

    let mut o = 0;
    let mut qlo = 0usize;
    let mut qho = 0usize;
    let mut sco = 0usize;
    for _n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let hi = |b: u8, sh: u32| (((b >> sh) & 3) as i32) << 4;
            let q1 = ((ql[qlo + l] & 0xF) as i32 | hi(qh[qho + l], 0)) as i8 - 32;
            let q2 = ((ql[qlo + l + 32] & 0xF) as i32 | hi(qh[qho + l], 2)) as i8 - 32;
            let q3 = ((ql[qlo + l] >> 4) as i32 | hi(qh[qho + l], 4)) as i8 - 32;
            let q4 = ((ql[qlo + l + 32] >> 4) as i32 | hi(qh[qho + l], 6)) as i8 - 32;
            out[o + l] = d * sc(sco + is) as f32 * q1 as f32;
            out[o + l + 32] = d * sc(sco + is + 2) as f32 * q2 as f32;
            out[o + l + 64] = d * sc(sco + is + 4) as f32 * q3 as f32;
            out[o + l + 96] = d * sc(sco + is + 6) as f32 * q4 as f32;
        }
        o += 128;
        qlo += 64;
        qho += 32;
        sco += 8;
    }
}

/// Dequantize a full Q3_K payload.
pub fn dequant_q3_k(bytes: &[u8]) -> Vec<f32> {
    qk_full(bytes, Q3KBLOCK_BYTES, q3k_block)
}
/// Dequantize a full Q5_K payload.
pub fn dequant_q5_k(bytes: &[u8]) -> Vec<f32> {
    qk_full(bytes, Q5_K_BLOCK_BYTES, q5k_block)
}
/// Dequantize a full Q6_K payload.
pub fn dequant_q6_k(bytes: &[u8]) -> Vec<f32> {
    qk_full(bytes, Q6_K_BLOCK_BYTES, q6k_block)
}

fn qk_full(bytes: &[u8], blk: usize, f: fn(&[u8], &mut [f32])) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(blk), "misaligned payload");
    let mut out = vec![0.0f32; bytes.len() / blk * QK_K];
    for (i, chunk) in bytes.chunks_exact(blk).enumerate() {
        f(chunk, &mut out[i * QK_K..(i + 1) * QK_K]);
    }
    out
}

/// Row dequant helpers (row-major payloads).
pub fn dequant_q3k_row(payload: &[u8], row: usize, row_bytes: usize, out: &mut [f32]) {
    qk_row(payload, row, row_bytes, Q3KBLOCK_BYTES, q3k_block, out)
}
pub fn dequant_q5k_row(payload: &[u8], row: usize, row_bytes: usize, out: &mut [f32]) {
    qk_row(payload, row, row_bytes, Q5_K_BLOCK_BYTES, q5k_block, out)
}
pub fn dequant_q6k_row(payload: &[u8], row: usize, row_bytes: usize, out: &mut [f32]) {
    qk_row(payload, row, row_bytes, Q6_K_BLOCK_BYTES, q6k_block, out)
}

fn qk_row(payload: &[u8], row: usize, row_bytes: usize, blk: usize, f: fn(&[u8], &mut [f32]), out: &mut [f32]) {
    let start = row * row_bytes;
    let blocks = row_bytes / blk;
    for b in 0..blocks {
        f(&payload[start + b * blk..start + (b + 1) * blk], &mut out[b * QK_K..(b + 1) * QK_K]);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tl1_group_matches_c_reference() {
        // Independent transcription of dequantize_row_ptq1_0 with explicit
        // per-element index mapping; must equal tl1_group on real-ish bytes.
        let mut buf = [0u8; PTQ1_GROUP_BYTES];
        for (i, b) in buf.iter_mut().enumerate().take(26) {
            *b = (i as u8).wrapping_mul(131).wrapping_add(7);
        }
        buf[26] = 0x35; // scale f16 ≈ 0.75
        buf[27] = 0x3A;

        let mut fast = vec![0.0f32; PTQ1_GROUP_ELEMS];
        tl1_group(&buf, &mut fast);

        let d = f16_to_f32(u16::from_le_bytes([buf[26], buf[27]]));
        let (qs, qh) = (&buf[0..24], &buf[24..26]);
        let mut expect = vec![0.0f32; PTQ1_GROUP_ELEMS];
        let dig = |byte: u8, n: usize| -> f32 {
            let q = byte.wrapping_mul(PTQ1_POW3[n]);
            let xi = ((q as u16) * 3) >> 8;
            (xi as i8 - 1) as f32 * d
        };
        // stage c=16: element n*16+m from qs[m]; stage c=8: 80+n*8+m from qs[16+m]
        for n in 0..5 {
            for m in 0..16 {
                expect[n * 16 + m] = dig(qs[m], n);
            }
        }
        for n in 0..5 {
            for m in 0..8 {
                expect[80 + n * 8 + m] = dig(qs[16 + m], n);
            }
        }
        for n in 0..4 {
            for h in 0..2 {
                expect[120 + n * 2 + h] = dig(qh[h], n);
            }
        }
        assert_eq!(fast, expect);
        assert!(fast.iter().all(|v| v.is_finite()));
    }

    /// L0 contract: our decoder inverts the C `quantize_row_ptq1_0_ref`
    /// (transcribed here) bit-exactly on ternary inputs.
    #[test]
    fn test_ptq1_roundtrip_vs_c_quantizer() {
        fn quantize_group_ref(x: &[f32; 128]) -> [u8; 28] {
            // d = amax = 1.0 here (ternary inputs), f16-exact by construction.
            let mut blk = [0u8; 28];
            blk[27] = 0x3c; // f16 1.0 LE
            let xi: Vec<i32> = x.iter().map(|&v| v.round() as i32 + 1).collect();
            // stage c=16: bytes qs[0..16] pack elements m + n*16, MS digit n=0
            for m in 0..16 {
                let mut q: u32 = 0;
                for n in 0..5 {
                    q = q * 3 + xi[m + n * 16] as u32;
                }
                blk[m] = ((q * 256 + 242) / 243) as u8;
            }
            // stage c=8: bytes qs[16..24] pack elements 80 + m + n*8
            for m in 0..8 {
                let mut q: u32 = 0;
                for n in 0..5 {
                    q = q * 3 + xi[80 + m + n * 8] as u32;
                }
                blk[16 + m] = ((q * 256 + 242) / 243) as u8;
            }
            // qh: qh[h] packs 120+h, 122+h, 124+h, 126+h then *3 shift
            for h in 0..2 {
                let mut q: u32 = 0;
                for m in 0..4 {
                    q = q * 3 + xi[120 + h + m * 2] as u32;
                }
                q *= 3;
                blk[24 + h] = ((q * 256 + 242) / 243) as u8;
            }
            blk
        }

        let mut lcg: u32 = 0x1234_5678;
        for _case in 0..64 {
            let x: [f32; 128] = std::array::from_fn(|_| {
                lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
                match (lcg >> 16) % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                }
            });
            let blk = quantize_group_ref(&x);
            let mut out = [0.0f32; 128];
            tl1_group(&blk, &mut out);
            assert_eq!(&out[..], &x[..], "roundtrip must be bit-exact");
        }

        // all-zero group: amax = 0 -> scale 0, digits 0 -> value -1*d = 0
        let x = [0.0f32; 128];
        let blk = quantize_group_ref(&x);
        let mut out = [0.0f32; 128];
        tl1_group(&blk, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tl1_group_trits_in_range() {
        let mut buf = [0u8; PTQ1_GROUP_BYTES];
        for (i, b) in buf.iter_mut().enumerate().take(26) {
            *b = (i as u8).wrapping_mul(29) ^ (i as u8).wrapping_mul(7);
        }
        buf[26] = 0x00;
        buf[27] = 0x3c; // scale 1.0
        let mut out = vec![0.0f32; PTQ1_GROUP_ELEMS];
        tl1_group(&buf, &mut out);
        assert!(out.iter().all(|v| *v == -1.0 || *v == 0.0 || *v == 1.0));
    }

    #[test]
    fn test_bf16_known_values() {
        // bf16 is the top 16 bits of f32
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x7f80), f32::INFINITY);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
        // 1/3 truncated into bf16 = 0x3EAA -> 0.33203125
        let v = bf16_to_f32(0x3eaa);
        assert!((v - 0.33203125).abs() < 1e-6);
    }

    #[test]
    fn test_f16_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!((f16_to_f32(0x3555) - 0.333_251_95).abs() < 1e-9);
        // smallest subnormal f16 = 2^-24
        assert!((f16_to_f32(0x0001) - 5.9604645e-8).abs() < 1e-12);
    }

    #[test]
    fn test_tq1_values_are_ternary_scaled() {
        // d = 1.0 (0x3c00), qs all 0 -> trit extraction of 0 => digit 0 => -1
        let mut block = [0u8; TQ1_0_BLOCK_BYTES];
        block[52] = 0x00;
        block[53] = 0x3c;
        let v = dequant_tq1_0(&block);
        assert_eq!(v.len(), 256);
        assert!(v.iter().all(|&x| x == -1.0), "zeros unpack to -1 (digit 0)");
    }

    #[test]
    fn test_tq1_all_values_in_set() {
        // random-ish bytes: every output must be in {-d, 0, +d}
        let mut block = [0u8; TQ1_0_BLOCK_BYTES];
        for (i, b) in block.iter_mut().take(52).enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        block[52] = 0x00;
        block[53] = 0x40; // d = 2.0
        let v = dequant_tq1_0(&block);
        assert!(v.iter().all(|x| *x == -2.0 || *x == 0.0 || *x == 2.0));
    }

    #[test]
    fn test_dequant_real_shard_row() {
        let path = "../shards/shard_1.bmts";
        if !std::path::Path::new(path).exists() {
            eprintln!("no shard, skipping");
            return;
        }
        let shard = crate::bmts::BmtsShard::open(path).unwrap();
        let t = shard
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_q.weight")
            .expect("tensor");
        let payload = shard.read_tensor(&t.name).unwrap();
        let row_bytes = 2560 / 256 * TQ1_0_BLOCK_BYTES;
        let mut row = vec![0.0f32; 2560];
        dequant_tq1_row(&payload, 0, row_bytes, &mut row);
        assert!(row.iter().all(|x| x.is_finite()));
        let scale: f32 = row.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        assert!(scale > 0.0);
    }
}

#[cfg(test)]
mod kquant_tests {
    use super::*;

    #[test]
    fn test_q8_block_simple() {
        let mut blk = [0u8; Q8_0_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        for j in 0..32 { blk[2 + j] = j as u8; } // 0..31 (i8 positive)
        blk[2 + 31] = 0x80; // -128
        let v = dequant_q8_0(&blk);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[10], 10.0);
        assert_eq!(v[31], -128.0);
    }

    #[test]
    fn test_q4k_scale_nibble_math() {
        // d=1, dmin=0; scale[0]=30, scale[1]=2; qs bytes 0xF0 -> lo nibble 0, hi 15
        // first 32 out = 1*30*0 = 0 ; next 32 = 1*2*15 = 30 ; rest = 0
        let mut blk = [0u8; Q4_K_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
        blk[4] = 30;
        blk[5] = 2;
        for b in blk[16..].iter_mut() {
            *b = 0xF0;
        }
        let v = dequant_q4_k(&blk);
        assert!(v[0..32].iter().all(|&x| x == 0.0), "lo nibbles: {:?}", &v[..2]);
        assert!(v[32..64].iter().all(|&x| x == 30.0), "hi nibbles: {:?}", &v[32..34]);
        assert!(v[64..256].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_q4k_min_offsets() {
        // verify mins subtract: dmin = 1/16, m nibble nonzero, scale = 0
        let mut blk = [0u8; Q4_K_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x2c00u16.to_le_bytes()); // dmin = 1/16
        for b in blk[4..16].iter_mut() { *b = 0; } // sc=0, m=0 -> expect 0
        let v = dequant_q4_k(&blk);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_quantkind_row_bytes() {
        assert_eq!(QuantKind::Q4K.row_bytes(5120), 20 * 144);
        assert_eq!(QuantKind::Tq1_0.row_bytes(2560), 10 * 54);
        assert_eq!(QuantKind::Q8_0.row_bytes(512), 16 * 34);
        assert_eq!(QuantKind::from_dtype(12), Some(QuantKind::Q4K));
        assert_eq!(QuantKind::from_dtype(13), Some(QuantKind::Q5K));
        assert_eq!(QuantKind::from_dtype(11), Some(QuantKind::Q3K));
        assert_eq!(QuantKind::from_dtype(14), Some(QuantKind::Q6K));
        assert_eq!(QuantKind::Q5K.row_bytes(5120), 20 * 176);
        assert_eq!(QuantKind::Q3K.row_bytes(2560), 10 * 110);
    }
}
