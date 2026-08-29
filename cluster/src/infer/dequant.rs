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
