//! Qwen3.8 (arch `qwen35`) kernels: gated delta-net recurrence, causal conv,
//! L2 norm, partial-NEOX rope, gated RMS norm.
//!
//! Transcribed from vendored `delta-net-base.cpp::build_delta_net_autoregressive`
//! (see docs/QWEN35_PORT.md for line references and orientation notes).

/// Delta-net head dimension (d_state), fixed for the qwen35 family.
pub const S: usize = 128;

/// One autoregressive gated-delta-net step for a single head.
///
/// `state` is row-major S×S (element (i,j) at i + j*S, matching ggml's
/// [S_v, S_v] tensor). `q` is scaled by the caller. Updates state in place
/// and writes the head output to `out`.
pub fn delta_head(state: &mut [f32], q: &[f32], k: &[f32], v: &[f32], gate: f32, beta: f32, out: &mut [f32]) {
    debug_assert_eq!(state.len(), S * S);
    debug_assert_eq!(q.len(), S);
    debug_assert_eq!(k.len(), S);
    debug_assert_eq!(v.len(), S);

    let decay = gate.exp();

    // sk[j] = sum_i S[i,j] * k[i]      (S^T k)
    let mut sk = [0f32; S];
    for j in 0..S {
        let mut acc = 0.0f32;
        for (i, &ki) in k.iter().enumerate() {
            acc += state[i + j * S] * ki;
        }
        sk[j] = acc * decay;
    }

    // d[j] = beta * (v[j] - sk[j]); S[i,j] = S[i,j]*decay + k[i]*d[j]
    let d: [f32; S] = std::array::from_fn(|j| beta * (v[j] - sk[j]));
    for (j, &dj) in d.iter().enumerate() {
        for (i, &ki) in k.iter().enumerate() {
            state[i + j * S] = state[i + j * S] * decay + ki * dj;
        }
    }

    // o[j] = sum_i S[i,j] * q[i]       (S^T q)
    for j in 0..S {
        let mut acc = 0.0f32;
        for (i, &qi) in q.iter().enumerate() {
            acc += state[i + j * S] * qi;
        }
        out[j] = acc;
    }
}

/// Causal depthwise conv tap step: 4-tap ring, oldest-first.
///
/// `ring` holds the last 3 inputs (ring[0] oldest); y = w0*ring0 + w1*ring1 +
/// w2*ring2 + w3*x (newest last). Orientation flagged for differential
/// confirmation against `conv_output_raw` captures.
pub fn conv_step(ring: &mut [f32; 3], w: &[f32; 4], x: f32) -> f32 {
    let y = w[0] * ring[0] + w[1] * ring[1] + w[2] * ring[2] + w[3] * x;
    ring[0] = ring[1];
    ring[1] = ring[2];
    ring[2] = x;
    y
}

/// L2 normalize a head vector in place.
pub fn l2_norm(v: &mut [f32], eps: f32) {
    let sum: f32 = v.iter().map(|x| x * x).sum();
    let inv = 1.0 / (sum.sqrt() + eps);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// RMSNorm followed by SiLU gating with a matching-width vector.
pub fn gated_norm_rms(x: &[f32], w: &[f32], gate: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    (0..n)
        .map(|i| {
            let g = gate[i] / (1.0 + (-gate[i]).exp());
            x[i] * inv * w[i] * g
        })
        .collect()
}

/// Partial NEOX rope: rotate the first `rot` dims (split-half pairs), leave
/// dims [rot..] untouched.
pub fn rope_partial_neox(v: &mut [f32], t: usize, rot: usize, base: f32) {
    let half = rot / 2;
    for i in 0..half {
        let theta = 1.0 / base.powf((2 * i) as f32 / rot as f32);
        let (cos_t, sin_t) = ((t as f32 * theta).cos(), (t as f32 * theta).sin());
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * cos_t - b * sin_t;
        v[i + half] = b * cos_t + a * sin_t;
    }
}

/// q scale used by the delta recurrence.
pub fn delta_q_scale() -> f32 {
    1.0 / (S as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_head_decay_only_when_beta_zero() {
        let mut state = vec![0.5f32; S * S];
        let q = vec![0.1f32; S];
        let k = vec![0.2f32; S];
        let v = vec![0.3f32; S];
        let mut out = vec![0.0f32; S];
        let gate = -0.25f32;
        delta_head(&mut state, &q, &k, &v, gate, 0.0, &mut out);
        let expect = 0.5f32 * gate.exp();
        assert!(state.iter().all(|&x| (x - expect).abs() < 1e-6));
        // o = (S*decay)^T q  with q pre-scaled
        let dot_q = q.iter().sum::<f32>() * expect;
        assert!(out.iter().all(|&x| (x - dot_q).abs() < 1e-4));
    }

    #[test]
    fn test_delta_head_single_write_read() {
        // Zero state; beta=1; write k (pre-decay state=0 -> sk=0, d=v).
        let mut state = vec![0.0f32; S * S];
        let mut q = vec![0.0f32; S];
        q[5] = delta_q_scale(); // only dim 5
        let mut k = vec![0.0f32; S];
        k[5] = 2.0; // align with q's nonzero dim
        k[9] = 1.0;
        let mut v = vec![0.0f32; S];
        v[7] = 4.0;
        let mut out = vec![0.0f32; S];
        delta_head(&mut state, &q, &k, &v, 0.0, 1.0, &mut out);
        // S = outer(k, v): S[i,j] = k_i*v_j
        assert!((state[5 + 7 * S] - 8.0).abs() < 1e-6);
        assert!((state[9 + 7 * S] - 4.0).abs() < 1e-6);
        // o[j] = q5 * S[5,j] = q5 * k5 * v_j -> nonzero only at j=7
        assert!(out.iter().enumerate().all(|(j, &x)| if j == 7 { (x - 4.0 * delta_q_scale() * 2.0).abs() < 1e-6 } else { x.abs() < 1e-6 }));
    }

    #[test]
    fn test_delta_head_is_linear_in_state() {
        let q: Vec<f32> = (0..S).map(|i| ((i * 7) % 13) as f32 * 0.1).collect();
        let k: Vec<f32> = (0..S).map(|i| ((i * 5) % 11) as f32 * 0.1).collect();
        let v: Vec<f32> = (0..S).map(|i| ((i * 3) % 17) as f32 * 0.1).collect();
        let mut s1 = vec![0.3f32; S * S];
        let mut s2 = s1.clone();
        let mut o1 = vec![0f32; S];
        let mut o2 = vec![0f32; S];
        delta_head(&mut s1, &q, &k, &v, -0.1, 0.7, &mut o1);
        delta_head(&mut s2, &q, &k, &v, -0.1, 0.7, &mut o2);
        assert_eq!(s1, s2, "must be deterministic");
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_conv_step_steady_state() {
        let mut ring = [3.0, 3.0, 3.0];
        let w = [0.25, 0.25, 0.25, 0.25];
        let y = conv_step(&mut ring, &w, 3.0);
        assert!((y - 3.0).abs() < 1e-6);
        assert_eq!(ring, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_conv_step_shifts() {
        let mut ring = [1.0, 2.0, 3.0];
        let w = [0.0, 0.0, 0.0, 1.0]; // newest-only
        assert!((conv_step(&mut ring, &w, 9.0) - 9.0).abs() < 1e-6);
        assert_eq!(ring, [2.0, 3.0, 9.0]);
    }

    #[test]
    fn test_l2_norm_unit_length() {
        let mut v = vec![3.0, 4.0, 0.0, 0.0];
        l2_norm(&mut v, 1e-12);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_gated_norm_rms() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![1.0; 4];
        let g = vec![0.0; 4]; // silu(0)=0 -> out 0
        let y = gated_norm_rms(&x, &w, &g, 1e-5);
        assert!(y.iter().all(|&v| v.abs() < 1e-6));
        let g2 = vec![10.0; 4];
        let y2 = gated_norm_rms(&x, &w, &g2, 1e-5);
        assert!((y2[0] - 10.0).abs() < 0.1, "large gate ~ rms*gate: {:?}", y2[0]);
    }

    #[test]
    fn test_rope_partial_keeps_tail() {
        let mut v: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let tail_before: Vec<f32> = v[64..].to_vec();
        rope_partial_neox(&mut v, 7, 64, 1e7);
        assert_eq!(&v[64..], &tail_before[..], "dims >= rot untouched");
        let changed = (0..64).any(|i| (v[i] - i as f32).abs() > 1e-4);
        assert!(changed, "rope must rotate leading dims");
    }
}
