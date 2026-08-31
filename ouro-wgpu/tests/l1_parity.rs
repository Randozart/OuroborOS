//! L1 rung: wgpu Q6_K gemv must match the scalar reference before any
//! stage binds the GPU pool.

use ouro_cluster::infer::{matvec_q, QuantKind};
use ouro_wgpu::GpuPool;

fn lcg(seed: u64, n: usize) -> Vec<u8> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as u8
        })
        .collect()
}

/// Random-but-sane Q6_K payload: sanitize the two f16 bytes per block.
fn sane_q6k(blocks: usize, seed: u64) -> Vec<u8> {
    let mut b = lcg(seed, blocks * 210);
    for blk in b.chunks_mut(210) {
        // d at 208..210: keep exponent < inf (high byte < 0x78)
        blk[209] %= 0x70;
    }
    b
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        let (x, y) = (a[i] as f64, b[i] as f64);
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-30)
}

#[test]
fn test_l1_q6k_gemv_parity() {
    let mut pool = GpuPool::new().expect("vulkan device");
    eprintln!("adapter: {}", pool.adapter_name);

    let (out_len, in_len) = (512usize, 2048usize); // 8 blocks/row
    let blocks = out_len * in_len / 256;
    let payload = sane_q6k(blocks, 99);
    pool.upload_q6k("t", &payload, out_len, in_len).unwrap();

    let x: Vec<f32> = lcg(7, in_len * 4)
        .chunks(4)
        .map(|c| (i32::from_le_bytes([c[0], c[1], c[2], c[3]]) % 1000) as f32 / 997.0)
        .collect();

    let gpu = pool.matvec("t", &x).unwrap();
    let cpu = matvec_q(&payload, QuantKind::Q6K, out_len, in_len, &x);

    let c = cos(&gpu, &cpu);
    let max_rel = gpu
        .iter()
        .zip(&cpu)
        .fold(0.0f64, |m, (a, b)| m.max(((a - b) as f64).abs() / (b.abs() as f64 + 1e-9)));
    eprintln!("cos={:.8} max_rel={:.6}", c, max_rel);
    assert!(c > 0.9999, "L1 parity cos {}", c);
    assert!(max_rel < 0.01, "L1 max relative deviation {}", max_rel);
}

#[test]
fn test_l1_q6k_throughput() {
    let mut pool = GpuPool::new().unwrap();
    // 9B-shaped: attn_qkv [4096, 8192] = 130 MB payload
    let (out_len, in_len) = (8192usize, 4096usize);
    let blocks = out_len * in_len / 256;
    let payload = sane_q6k(blocks, 5);
    let t0 = std::time::Instant::now();
    pool.upload_q6k("big", &payload, out_len, in_len).unwrap();
    eprintln!("upload+repack 130MB: {:.1}s", t0.elapsed().as_secs_f64());

    let x: Vec<f32> = vec![0.01; in_len];
    // warmup
    let _ = pool.matvec("big", &x).unwrap();
    let n = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let y = pool.matvec("big", &x).unwrap();
        assert!(y.iter().all(|v| v.is_finite()));
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    let bytes = payload.len() as f64;
    eprintln!("per matvec: {:.3} ms -> {:.1} GB/s effective", per * 1e3, bytes / per / 1e9);
    assert!(per < 0.05, "matvec unexpectedly slow: {:.1} ms", per * 1e3);
}

#[test]
fn test_l1_dequant_probe() {
    // one block, one row: x = e_j isolates column j of the dequantized block
    let mut pool = GpuPool::new().unwrap();
    let payload = sane_q6k(1, 1234); // 1 block, 1 row
    pool.upload_q6k("p", &payload, 1, 256).unwrap();
    let cpu_row = matvec_q(&payload, QuantKind::Q6K, 1, 256, &vec![1.0f32; 256])[0];

    // sanity 1: all-ones row-sum
    let ones = pool.matvec("p", &vec![1.0f32; 256]).unwrap()[0];
    eprintln!("row-sum gpu={ones:.5} cpu={cpu_row:.5}");
    let mut bad = Vec::new();
    for j in 0..256usize {
        let mut x = vec![0.0f32; 256];
        x[j] = 1.0;
        let gpu = pool.matvec("p", &x).unwrap()[0];
        let cpu = matvec_q(&payload, QuantKind::Q6K, 1, 256, &x)[0];
        let d = ((gpu - cpu) as f64).abs();
        if d > 1e-3 {
            bad.push((j, gpu, cpu));
        }
    }
    eprintln!("cpu row-sum={cpu_row:.5}; bad count={} first few: {:?}", bad.len(), &bad[..bad.len().min(8)]);
    // dump full mapping for reverse engineering
    let mut dump = String::new();
    for j in 0..256usize {
        let mut x = vec![0.0f32; 256];
        x[j] = 1.0;
        let gpu = pool.matvec("p", &x).unwrap()[0];
        dump.push_str(&format!("{} {:.6}\n", j, gpu));
    }
    std::fs::write("/tmp/wgsl_dump.txt", dump).ok();
    // group by n
    let n0: Vec<_> = bad.iter().filter(|(j, _, _)| *j < 128).map(|t| t.0).collect();
    let n1: Vec<_> = bad.iter().filter(|(j, _, _)| *j >= 128).map(|t| t.0).collect();
    eprintln!("bad in n=0: {:?}
bad in n=1: {:?}", n0, n1);
    assert!(bad.is_empty());
}

#[test]
fn test_l1_row_addressing_probe() {
    let mut pool = GpuPool::new().unwrap();
    let (out_len, in_len) = (2usize, 512usize); // 2 rows, 2 blocks each
    let payload = sane_q6k(out_len * in_len / 256, 31337);
    pool.upload_q6k("r", &payload, out_len, in_len).unwrap();
    let mut bad = Vec::new();
    for j in 0..in_len {
        let mut x = vec![0.0f32; in_len];
        x[j] = 1.0;
        let gpu = pool.matvec("r", &x).unwrap();
        let cpu = matvec_q(&payload, QuantKind::Q6K, out_len, in_len, &x);
        for r in 0..out_len {
            let d = ((gpu[r] - cpu[r]) as f64).abs();
            if d > 1e-3 {
                bad.push((r, j, gpu[r], cpu[r]));
            }
        }
    }
    eprintln!("bad count={} first: {:?}", bad.len(), &bad[..bad.len().min(6)]);
    assert!(bad.is_empty());
}

/// G2 gate (PLAN §18.1): vectorized kernel keeps L1 parity AND delivers
/// matvec throughput >= 8x the scalar CPU reference on this box.
#[test]
fn test_g2_speedup_vs_cpu_scalar() {
    let mut pool = GpuPool::new().unwrap();
    eprintln!("adapter: {}", pool.adapter_name);
    let (out_len, in_len) = (8192usize, 4096usize);
    let payload = sane_q6k(out_len * in_len / 256, 5);
    pool.upload_q6k("g2", &payload, out_len, in_len).unwrap();
    let x: Vec<f32> = vec![0.01; in_len];

    let _ = pool.matvec("g2", &x).unwrap(); // warmup
    let n = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let y = pool.matvec("g2", &x).unwrap();
        assert!(y.iter().all(|v| v.is_finite()));
    }
    let gpu_per = t0.elapsed().as_secs_f64() / n as f64;

    let t0 = std::time::Instant::now();
    let cpu = matvec_q(&payload, QuantKind::Q6K, out_len, in_len, &x);
    let cpu_per = t0.elapsed().as_secs_f64();

    let gpu = pool.matvec("g2", &x).unwrap();
    let c = cos(&gpu, &cpu);
    let speedup = cpu_per / gpu_per;
    let bytes = payload.len() as f64;
    eprintln!(
        "cpu {:.1} ms | gpu {:.3} ms | speedup {:.1}x | cos {:.8} | {:.0} GB/s",
        cpu_per * 1e3, gpu_per * 1e3, speedup, c, bytes / gpu_per / 1e9
    );
    assert!(c > 0.9999, "G2 parity cos {}", c);
    assert!(speedup >= 8.0, "G2 speedup {:.1}x < 8x", speedup);
}
