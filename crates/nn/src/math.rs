pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = [0.0f32; 8];
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rest: f32 = chunks_a
        .remainder()
        .iter()
        .zip(chunks_b.remainder())
        .map(|(x, y)| x * y)
        .sum();

    for (x, y) in chunks_a.zip(chunks_b) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }

    acc.iter().sum::<f32>() + rest
}

pub(crate) fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

const MR: usize = 4;
const NR: usize = 8;

pub(crate) fn gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    let full_rows = m - m % MR;
    let full_cols = n - n % NR;
    for i in (0..full_rows).step_by(MR) {
        for j in (0..full_cols).step_by(NR) {
            tile(i, j, n, k, a, b, c);
        }
        if full_cols < n {
            edge(i..i + MR, full_cols..n, n, k, a, b, c);
        }
    }
    if full_rows < m {
        edge(full_rows..m, 0..n, n, k, a, b, c);
    }
}

#[inline(always)]
fn tile(i: usize, j: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let rows: [&[f32]; MR] = std::array::from_fn(|r| &a[(i + r) * k..(i + r + 1) * k]);
    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..k {
        let bp: &[f32; NR] = b[p * n + j..p * n + j + NR].try_into().expect("tile width");
        for r in 0..MR {
            let av = rows[r][p];
            for x in 0..NR {
                acc[r][x] += av * bp[x];
            }
        }
    }
    for (r, acc) in acc.iter().enumerate() {
        let row = &mut c[(i + r) * n + j..(i + r) * n + j + NR];
        for (cv, av) in row.iter_mut().zip(acc) {
            *cv += av;
        }
    }
}

fn edge(
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    for i in rows {
        let out = &mut c[i * n + cols.start..i * n + cols.end];
        for p in 0..k {
            let av = a[i * k + p];
            axpy(av, &b[p * n + cols.start..p * n + cols.end], out);
        }
    }
}

fn row(matrix: &[f32], i: usize, k: usize) -> &[f32] {
    &matrix[i * k..(i + 1) * k]
}

pub(crate) fn gemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert!(a.len() >= m * k && b.len() >= n * k && c.len() >= m * n);
    let mut i = 0;
    while i + 2 <= m {
        let (a0, a1) = (row(a, i, k), row(a, i + 1, k));
        let mut j = 0;
        while j + 2 <= n {
            let [d00, d01, d10, d11] = dot2x2(a0, a1, row(b, j, k), row(b, j + 1, k));
            c[i * n + j] += d00;
            c[i * n + j + 1] += d01;
            c[(i + 1) * n + j] += d10;
            c[(i + 1) * n + j + 1] += d11;
            j += 2;
        }
        if j < n {
            c[i * n + j] += dot(a0, row(b, j, k));
            c[(i + 1) * n + j] += dot(a1, row(b, j, k));
        }
        i += 2;
    }
    if i < m {
        for j in 0..n {
            c[i * n + j] += dot(row(a, i, k), row(b, j, k));
        }
    }
}

#[inline(always)]
fn dot2x2(a0: &[f32], a1: &[f32], b0: &[f32], b1: &[f32]) -> [f32; 4] {
    const L: usize = 8;
    let mut acc = [[0.0f32; L]; 4];
    let chunks = a0.len() / L * L;
    for p in (0..chunks).step_by(L) {
        let x0: &[f32; L] = a0[p..p + L].try_into().expect("lane");
        let x1: &[f32; L] = a1[p..p + L].try_into().expect("lane");
        let y0: &[f32; L] = b0[p..p + L].try_into().expect("lane");
        let y1: &[f32; L] = b1[p..p + L].try_into().expect("lane");
        for l in 0..L {
            acc[0][l] += x0[l] * y0[l];
            acc[1][l] += x0[l] * y1[l];
            acc[2][l] += x1[l] * y0[l];
            acc[3][l] += x1[l] * y1[l];
        }
    }
    let mut out = acc.map(|lanes| lanes.iter().sum::<f32>());
    for p in chunks..a0.len() {
        out[0] += a0[p] * b0[p];
        out[1] += a0[p] * b1[p];
        out[2] += a1[p] * b0[p];
        out[3] += a1[p] * b1[p];
    }
    out
}

pub(crate) fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    thread_local! {
        static POOL: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut buf = POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    buf.clear();
    buf.resize(len, 0.0);
    let result = f(&mut buf);
    POOL.with(|pool| pool.borrow_mut().push(buf));
    result
}

pub(crate) fn transpose(rows: usize, cols: usize, src: &[f32], dst: &mut [f32]) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_naive_for_awkward_shapes() {
        for (m, n, k) in [(1, 1, 1), (4, 8, 3), (5, 9, 7), (13, 30, 17), (3, 2, 40)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.5).collect();
            let mut c: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
            let mut expected = c.clone();
            for i in 0..m {
                for j in 0..n {
                    for p in 0..k {
                        expected[i * n + j] += a[i * k + p] * b[p * n + j];
                    }
                }
            }
            gemm(m, n, k, &a, &b, &mut c);
            for (x, y) in c.iter().zip(&expected) {
                assert!((x - y).abs() < 1e-3, "{m}x{n}x{k}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn gemm_nt_matches_naive_for_awkward_shapes() {
        for (m, n, k) in [(1, 1, 1), (2, 2, 8), (5, 3, 19), (16, 27, 64), (3, 7, 2)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
            let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.5).collect();
            let mut c: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
            let mut expected = c.clone();
            for i in 0..m {
                for j in 0..n {
                    for p in 0..k {
                        expected[i * n + j] += a[i * k + p] * b[j * k + p];
                    }
                }
            }
            gemm_nt(m, n, k, &a, &b, &mut c);
            for (x, y) in c.iter().zip(&expected) {
                assert!((x - y).abs() < 1e-3, "{m}x{n}x{k}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn transpose_swaps_axes() {
        let mut out = [0.0; 6];
        transpose(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut out);
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn dot_matches_naive_for_all_lengths() {
        for len in 0..20 {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| 1.0 - i as f32).collect();
            let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - naive).abs() < 1e-3);
        }
    }

    #[test]
    fn axpy_adds_scaled_vector() {
        let mut y = vec![1.0, 2.0, 3.0];
        axpy(2.0, &[1.0, 1.0, -1.0], &mut y);
        assert_eq!(y, vec![3.0, 4.0, 1.0]);
    }
}
