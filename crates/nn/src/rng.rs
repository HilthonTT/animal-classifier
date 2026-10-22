#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform integer in `[0, n)`. `n` must be non-zero.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Sample from the standard normal distribution (Box-Muller transform)
    pub fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.next_f32(); // (0, 1], avoids ln(0)
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn gamma(&mut self, shape: f32) -> f32 {
        if shape < 1.0 {
            let u = 1.0 - self.next_f32();
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = 1.0 - self.next_f32();
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    pub fn beta(&mut self, a: f32, b: f32) -> f32 {
        let x = self.gamma(a);
        let y = self.gamma(b);
        if x + y > 0.0 { x / (x + y) } else { 0.5 }
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn floats_are_in_unit_interval() {
        let mut rng = Rng::new(1);
        for _ in 0..10_000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn normal_has_roughly_zero_mean_unit_variance() {
        let mut rng = Rng::new(3);
        let n = 50_000;
        let samples: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        let mean = samples.iter().sum::<f32>() / n as f32;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
    }

    #[test]
    fn beta_has_the_expected_mean_and_range() {
        let mut rng = Rng::new(5);
        for (a, b) in [(0.2, 0.2), (1.0, 1.0), (2.0, 5.0)] {
            let n = 20_000;
            let samples: Vec<f32> = (0..n).map(|_| rng.beta(a, b)).collect();
            assert!(samples.iter().all(|x| (0.0..=1.0).contains(x)));
            let mean = samples.iter().sum::<f32>() / n as f32;
            assert!(
                (mean - a / (a + b)).abs() < 0.02,
                "Beta({a}, {b}) mean {mean}"
            );
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(9);
        let mut v: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        assert_ne!(v, sorted);
    }
}
