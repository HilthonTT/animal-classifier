use super::{Layer, LayerKind, parse_config};
use crate::error::{Error, Result};
use crate::rng::Rng;

/// Inverted dropout (Srivastava et al., 2014).
///
/// While training, each value is zeroed with probability `rate` and the survivors are scaled
/// by `1 / (1 − rate)`. The network then cannot rely on any single neuron, which reduces
/// over-fitting. At inference time the layer passes values through unchanged.
#[derive(Clone, Debug)]
pub struct Dropout {
    len: usize,
    rate: f32,
}

impl Dropout {
    /// `rate` is the probability of dropping a value, in `[0, 1)`.
    pub fn new(len: usize, rate: f32) -> Self {
        Self::with_shape(len, rate).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [len, rate_bits] = parse_config(config, "Dropout")?;
        Self::with_shape(len, f32::from_bits(rate_bits as u32))
    }

    fn with_shape(len: usize, rate: f32) -> Result<Self> {
        if len == 0 || !(0.0..1.0).contains(&rate) {
            return Err(Error::format(
                "Dropout needs a non-zero length and a rate in [0, 1)",
            ));
        }
        Ok(Self { len, rate })
    }

    pub fn rate(&self) -> f32 {
        self.rate
    }
}

impl Layer for Dropout {
    fn kind(&self) -> LayerKind {
        LayerKind::Dropout
    }

    fn config(&self) -> Vec<u32> {
        vec![self.len as u32, self.rate.to_bits()]
    }

    fn describe(&self) -> String {
        format!("Dropout {}%", (self.rate * 100.0).round())
    }

    fn input_len(&self) -> usize {
        self.len
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
    }

    fn forward_train(
        &self,
        input: &[f32],
        output: &mut [f32],
        rng: &mut Rng,
        cache: &mut Vec<f32>,
    ) {
        // The cache holds each value's multiplier: 0 (dropped) or 1 / (1 - rate) (kept).
        let keep_scale = 1.0 / (1.0 - self.rate);
        cache.clear();
        cache.extend((0..self.len).map(|_| {
            if rng.next_f32() < self.rate {
                0.0
            } else {
                keep_scale
            }
        }));
        for ((o, &x), &m) in output.iter_mut().zip(input).zip(cache.iter()) {
            *o = x * m;
        }
    }

    fn backward(
        &self,
        _input: &[f32],
        _output: &[f32],
        cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        _grad_params: &mut [f32],
    ) {
        if cache.is_empty() {
            grad_input.copy_from_slice(grad_output); // forward ran in inference mode
        } else {
            for ((gi, &g), &m) in grad_input.iter_mut().zip(grad_output).zip(cache) {
                *gi = g * m;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_about_rate_and_preserves_expectation() {
        let layer = Dropout::new(10_000, 0.3);
        let input = vec![1.0; 10_000];
        let mut output = vec![0.0; 10_000];
        let mut cache = Vec::new();
        layer.forward_train(&input, &mut output, &mut Rng::new(4), &mut cache);
        let dropped = output.iter().filter(|&&v| v == 0.0).count() as f32 / 10_000.0;
        let mean = output.iter().sum::<f32>() / 10_000.0;
        assert!((dropped - 0.3).abs() < 0.02, "dropped {dropped}");
        assert!((mean - 1.0).abs() < 0.03, "mean {mean}");

        let mut grad_input = vec![0.0; 10_000];
        layer.backward(&input, &output, &cache, &input, &mut grad_input, &mut []);
        assert_eq!(grad_input, output);
    }

    #[test]
    fn is_identity_at_inference() {
        let layer = Dropout::new(3, 0.9);
        let mut output = [0.0; 3];
        layer.forward(&[1.0, -2.0, 3.0], &mut output);
        assert_eq!(output, [1.0, -2.0, 3.0]);
    }
}
