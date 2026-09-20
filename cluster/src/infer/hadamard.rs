//! Blockwise normalized Sylvester-Walsh-Hadamard rotation (PrismML PTQ1_0).
//!
//! Weights are exported in a Hadamard-rotated basis
//! (`prism.hadamard.transform = normalized-sylvester-walsh-hadamard`). The
//! fork (llama-graph.cpp) uses two conjugate transforms:
//!
//! - pre-matmul activations (build_lora_mm): x' = H·(s ∘ x)   — signs FIRST
//! - embedding lookup recovery:               h  = s ∘ (H·z)  — signs AFTER
//!
//! with H[i][j] = (-1)^popcount(i & j) / sqrt(N), block N (1024 for
//! Bonsai-2). These are H_s^T and H_s for H_s = diag(s)·H; composed they
//! yield the identity, which the round-trip test asserts.

/// In-place unnormalized fast Walsh-Hadamard transform (butterfly), O(N log N).
fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for chunk in x.chunks_exact_mut(h * 2) {
            let (lo, hi) = chunk.split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi) {
                let s = *a + *b;
                let d = *a - *b;
                *a = s;
                *b = d;
            }
        }
        h *= 2;
    }
}

fn check_args(x: &[f32], signs: &[f32], block_size: usize) {
    assert_eq!(x.len(), signs.len(), "sign vector must match activation");
    assert!(block_size.is_power_of_two());
    assert!(x.len().is_multiple_of(block_size), "len {} not multiple of block {}", x.len(), block_size);
}

/// Activation-side transform (build_lora_mm): x' = H·(s ∘ x).
/// Apply BEFORE every Hadamard-folded matmul.
pub fn rotate_fwd(x: &mut [f32], signs: &[f32], block_size: usize) {
    check_args(x, signs, block_size);
    let scale = 1.0 / (block_size as f32).sqrt();
    for (v, &s) in x.iter_mut().zip(signs) {
        *v *= s;
    }
    for block in x.chunks_exact_mut(block_size) {
        wht_inplace(block);
    }
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Embedding-side transform (post-lookup): h = s ∘ (H·z).
/// Apply AFTER the Hadamard-latent embedding row lookup.
pub fn rotate_inv(x: &mut [f32], signs: &[f32], block_size: usize) {
    check_args(x, signs, block_size);
    let scale = 1.0 / (block_size as f32).sqrt();
    for block in x.chunks_exact_mut(block_size) {
        wht_inplace(block);
    }
    for (v, &s) in x.iter_mut().zip(signs) {
        *v *= s * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(N²) reference matching the PrismML construction loop.
    fn naive_matrix(signs: &[f32], block_size: usize) -> Vec<f32> {
        let scale = 1.0 / (block_size as f32).sqrt();
        let n = signs.len();
        let mut m = vec![0.0f32; n * n];
        for start in (0..n).step_by(block_size) {
            for row in 0..block_size {
                for col in 0..block_size {
                    let p = (row & col).count_ones();
                    let entry = if p & 1 == 1 { -scale } else { scale };
                    m[(start + row) * n + start + col] = entry;
                }
            }
        }
        m
    }

    fn mat_vec(m: &[f32], x: &[f32]) -> Vec<f32> {
        let n = x.len();
        (0..n)
            .map(|r| (0..n).map(|c| m[r * n + c] * x[c]).sum())
            .collect()
    }

    #[test]
    fn test_fwd_is_h_times_diag_s() {
        let block = 16;
        let n = block;
        let x: Vec<f32> = (0..n).map(|i| ((i * 11) % 7) as f32 - 3.0).collect();
        let signs: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let mut fast = x.clone();
        rotate_fwd(&mut fast, &signs, block);
        let h = naive_matrix(&signs, block);
        let sd: Vec<f32> = x.iter().zip(&signs).map(|(v, &s)| v * s).collect();
        let reference = mat_vec(&h, &sd);
        for (a, b) in fast.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn test_inv_is_diag_s_times_h() {
        let block = 16;
        let n = block;
        let x: Vec<f32> = (0..n).map(|i| ((i * 5) % 9) as f32 - 4.0).collect();
        let signs: Vec<f32> = (0..n).map(|i| if i % 4 == 0 { -1.0 } else { 1.0 }).collect();
        let mut fast = x.clone();
        rotate_inv(&mut fast, &signs, block);
        let h = naive_matrix(&signs, block);
        let hx = mat_vec(&h, &x);
        let reference: Vec<f32> = hx.iter().zip(&signs).map(|(v, &s)| v * s).collect();
        for (a, b) in fast.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_fwd_inv_roundtrip_identity() {
        // H_s^T ∘ H_s = I: embed-then-matmul restores the primal vector.
        let block = 64;
        let n = block * 2;
        let x: Vec<f32> = (0..n).map(|i| ((i * 91) % 23) as f32 * 0.37 - 4.0).collect();
        let signs: Vec<f32> = (0..n).map(|i| if (i / 5) % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let mut y = x.clone();
        rotate_inv(&mut y, &signs, block);
        rotate_fwd(&mut y, &signs, block);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn test_rotation_preserves_l2_norm() {
        let block = 64;
        let n = block * 3;
        let x: Vec<f32> = (0..n).map(|i| ((i * 91) % 23) as f32 * 0.37 - 4.0).collect();
        let signs: Vec<f32> = (0..n).map(|i| if (i / 5) % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let n0: f32 = x.iter().map(|v| v * v).sum();
        let mut a = x.clone();
        rotate_fwd(&mut a, &signs, block);
        let mut b = x.clone();
        rotate_inv(&mut b, &signs, block);
        for y in [&a, &b] {
            let n1: f32 = y.iter().map(|v| v * v).sum();
            assert!((n0 - n1).abs() / n0 < 1e-4, "norm drift {n0} vs {n1}");
        }
    }

    #[test]
    fn test_blockwise_independence() {
        let block = 32;
        let n = block * 2;
        let x: Vec<f32> = (0..n).map(|i| ((i * 7) % 17) as f32 - 8.0).collect();
        let signs = vec![1.0f32; n];
        let mut a = x.clone();
        rotate_fwd(&mut a, &signs, block);
        let mut b = x.clone();
        b[3] += 5.0;
        rotate_fwd(&mut b, &signs, block);
        assert_eq!(&a[block..], &b[block..], "blocks independent");
    }
}
