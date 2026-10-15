//! CPU replicas of two rows of Ferric's fabric_probe (github.com/
//! dcharlot-physicalai-bmi/ferric at 67f7876, crates/ferric-core/examples/
//! fabric_probe.rs): `det`, `fnv`, `matmul_cpu` (lib.rs) and
//! `rmsnorm_tree_cpu` (kernels.rs, single-threaded: rows are independent).
//! Prints the digests that lib/ml/tensor.rce must reproduce.
fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        f32::from_bits(0x3F80_0000 | (s >> 41) as u32) - 1.5 }).collect()
}
fn fnv(bits: impl IntoIterator<Item = f32>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in bits { for b in v.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100_0000_01b3); } }
    h
}
fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m { for col in 0..n { let mut acc = 0.0f32;
        for i in 0..k { acc += a[row * k + i] * b[i * n + col]; } c[row * n + col] = acc; } }
    c
}
fn recip(y: f32) -> f32 { let mut r = f32::from_bits(0x7EF3_11C3u32.wrapping_sub(y.to_bits()));
    for _ in 0..3 { let t = y * r; let ww = 2.0 - t; r = r * ww; } r }
fn rsqrt(y: f32) -> f32 { let hy = 0.5 * y; let mut r = f32::from_bits(0x5F37_59DFu32.wrapping_sub(y.to_bits() >> 1));
    for _ in 0..3 { let t = r * r; let u = hy * t; let ww = 1.5 - u; r = r * ww; } r }
fn rmsnorm_tree_cpu(x: &[f32], w: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * d];
    for row in 0..rows { let base = row * d; let mut red = [0f32; 64];
        for li in 0..64usize { let cls = (li + 1) & 63; let mut p = 0f32; let mut j = cls;
            while j < d { let sq = x[base + j] * x[base + j]; p += sq; j += 64; } red[li] = p; }
        let mut s = 32usize;
        while s >= 1 { for li in 0..s { red[li] += red[li + s]; } if s == 1 { break; } s >>= 1; }
        let ms = red[0] * recip(d as f32); let inv = rsqrt(ms + eps);
        for j in 0..d { let t = x[base + j] * inv; out[base + j] = t * w[j]; } }
    out
}
fn main() {
    const T: usize = 12; const D: usize = 64;
    let x = det(T * D, 1); let w = det(D * D, 2);
    let wn: Vec<f32> = det(D, 3).iter().map(|v| v + 1.0).collect();
    println!("x0 {:08x} x1 {:08x}", x[0].to_bits(), x[1].to_bits());
    println!("mm        {:016x}", fnv(matmul_cpu(&x, &w, T, D, D)));
    println!("rms-tcpu  {:016x}", fnv(rmsnorm_tree_cpu(&x, &wn, T, D, 1e-5)));
}
