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

#[cfg(test)]
mod tests {
    use super::*;

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
