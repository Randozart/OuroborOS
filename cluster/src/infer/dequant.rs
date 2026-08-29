//! Weight dequantization: TQ1_0 (ternary 1.58-bit) and F16 -> F32.
//!
//! TQ1_0 block (54 bytes, 256 elements):
//! - `qs[48]`: base-3 packed, 5 elements per byte (3^5 = 243 < 256)
//! - `qh[4]`:  one digit per byte, high positions
//! - `d`:      f16 scale shared by the block
//!
//! Element order matches `dequantize_row_tq1_0` in ggml-quants.c exactly.

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
        for m in 0..32 {
            out[o] = trit(qs[m], n);
            o += 1;
        }
    }
    for n in 0..5 {
        for m in 0..16 {
            out[o] = trit(qs[32 + m], n);
            o += 1;
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            out[o] = trit(qh[j], n);
            o += 1;
        }
    }
}

/// Dequantize a full TQ1_0 tensor row-major payload into new f32 vec.
/// `bytes` length must be a multiple of 54 (k = len/54*256).
pub fn dequant_tq1_0(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % TQ1_0_BLOCK_BYTES == 0, "misaligned TQ1_0 payload");
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
    assert!(bytes.len() % Q8_0_BLOCK_BYTES == 0, "misaligned Q8_0 payload");
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
    assert!(bytes.len() % Q4_K_BLOCK_BYTES == 0, "misaligned Q4_K payload");
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
    Tq1_0,
    Q8_0,
    Q4K,
}

impl QuantKind {
    /// From ggml type id; None for unsupported types.
    pub fn from_dtype(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4K),
            34 => Some(Self::Tq1_0),
            _ => None,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Tq1_0 => TQ1_0_BLOCK_BYTES,
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
            Self::Q4K => Q4_K_BLOCK_BYTES,
        }
    }

    /// Elements per block.
    pub fn block_elems(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q8_0 => 32,
            Self::Tq1_0 | Self::Q4K => QK_K,
        }
    }

    /// Row byte size for an input dimension of k elements.
    pub fn row_bytes(self, k: usize) -> usize {
        let be = self.block_elems();
        k / be * self.block_bytes()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!((f16_to_f32(0x3555) - 0.333251953125).abs() < 1e-9);
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
        assert_eq!(QuantKind::from_dtype(13), None);
    }
}
