pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut [f32]], grads: &[Vec<f32>]);
    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, learning_rate: f32);
}

/// Gradient clipping: if the combined L2 norm of all gradients exceeds `max_norm`, scales
/// them down to exactly `max_norm`. Prevents one bad batch from wrecking the weights.
/// Returns the norm before clipping.
pub fn clip_gradients(grads: &mut [Vec<f32>], max_norm: f32) -> f32 {
    let norm = grads.iter().flatten().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        grads.iter_mut().flatten().for_each(|g| *g *= scale);
    }
    norm
}

/// Allocates zeroed per-layer state shaped like `grads`, once.
fn ensure_state(state: &mut Vec<Vec<f32>>, grads: &[Vec<f32>]) {
    if state.len() != grads.len() {
        *state = grads.iter().map(|g| vec![0.0; g.len()]).collect();
    }
}

/// Stochastic gradient descent with momentum.
#[derive(Clone, Debug)]
pub struct Sgd {
    learning_rate: f32,
    momentum: f32,
    velocity: Vec<Vec<f32>>,
}

impl Sgd {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [&mut [f32]], grads: &[Vec<f32>]) {
        ensure_state(&mut self.velocity, grads);
        for ((p, g), v) in params.iter_mut().zip(grads).zip(&mut self.velocity) {
            for ((p, &g), v) in p.iter_mut().zip(g).zip(v.iter_mut()) {
                *v = self.momentum * *v - self.learning_rate * g;
                *p += *v;
            }
        }
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

/// Adam (Kingma & Ba, 2014): per-parameter adaptive learning rates.
#[derive(Clone, Debug)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: i32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut [f32]], grads: &[Vec<f32>]) {
        ensure_state(&mut self.m, grads);
        ensure_state(&mut self.v, grads);
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let bias1 = 1.0 - b1.powi(self.t);
        let bias2 = 1.0 - b2.powi(self.t);
        for (((p, g), m), v) in params
            .iter_mut()
            .zip(grads)
            .zip(&mut self.m)
            .zip(&mut self.v)
        {
            for (((p, &g), m), v) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
                *m = b1 * *m + (1.0 - b1) * g;
                *v = b2 * *v + (1.0 - b2) * g * g;
                *p -= self.learning_rate * (*m / bias1) / ((*v / bias2).sqrt() + self.epsilon);
            }
        }
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimises f(x) = (x - 3)² and checks that the optimizer converges.
    fn converges(mut opt: impl Optimizer) {
        let mut x = vec![0.0f32];
        for _ in 0..2000 {
            let grad = vec![vec![2.0 * (x[0] - 3.0)]];
            opt.step(&mut [x.as_mut_slice()], &grad);
        }
        assert!((x[0] - 3.0).abs() < 1e-2, "ended at {}", x[0]);
    }

    #[test]
    fn sgd_converges() {
        converges(Sgd::new(0.01, 0.9));
    }

    #[test]
    fn adam_converges() {
        converges(Adam::new(0.05));
    }

    #[test]
    fn clipping_limits_the_global_norm() {
        let mut grads = vec![vec![3.0], vec![4.0]];
        assert_eq!(clip_gradients(&mut grads, 1.0), 5.0);
        assert!((grads[0][0] - 0.6).abs() < 1e-6 && (grads[1][0] - 0.8).abs() < 1e-6);
        assert_eq!(clip_gradients(&mut grads, 10.0), 1.0);
        assert!((grads[1][0] - 0.8).abs() < 1e-6);
    }
}
