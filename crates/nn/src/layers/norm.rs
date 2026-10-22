use super::{BatchContext, Layer, LayerKind, checked_len, parse_config};
use crate::error::{Error, Result};

const EPSILON: f32 = 1e-5;
const MOMENTUM: f32 = 0.1;

#[derive(Clone, Debug)]
pub struct BatchNorm {
    channels: usize,
    plane: usize,
    params: Vec<f32>,
    buffers: Vec<f32>,
}

impl BatchNorm {
    pub fn new(channels: usize, plane: usize) -> Self {
        Self::with_shape(channels, plane).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [channels, plane] = parse_config(config, "BatchNorm")?;
        Self::with_shape(channels, plane)
    }

    fn with_shape(channels: usize, plane: usize) -> Result<Self> {
        if channels == 0 || plane == 0 {
            return Err(Error::format("BatchNorm dimensions must be non-zero"));
        }
        if checked_len(&[channels, plane]).is_none() || checked_len(&[channels, 2]).is_none() {
            return Err(Error::format("BatchNorm layer is too large"));
        }
        let mut params = vec![1.0; channels];
        params.resize(2 * channels, 0.0);
        let mut buffers = vec![0.0; channels];
        buffers.resize(2 * channels, 1.0);
        Ok(Self {
            channels,
            plane,
            params,
            buffers,
        })
    }

    fn gamma(&self) -> &[f32] {
        &self.params[..self.channels]
    }

    fn beta(&self) -> &[f32] {
        &self.params[self.channels..]
    }

    fn running(&self) -> (&[f32], &[f32]) {
        self.buffers.split_at(self.channels)
    }

    fn inference_inv_std(&self) -> Vec<f32> {
        self.running()
            .1
            .iter()
            .map(|v| 1.0 / (v + EPSILON).sqrt())
            .collect()
    }
}

impl Layer for BatchNorm {
    fn kind(&self) -> LayerKind {
        LayerKind::BatchNorm
    }

    fn config(&self) -> Vec<u32> {
        vec![self.channels as u32, self.plane as u32]
    }

    fn describe(&self) -> String {
        format!("BatchNorm ({} channels)", self.channels)
    }

    fn input_len(&self) -> usize {
        self.channels * self.plane
    }

    fn output_len(&self) -> usize {
        self.channels * self.plane
    }

    fn params(&self) -> &[f32] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn buffers(&self) -> &[f32] {
        &self.buffers
    }

    fn buffers_mut(&mut self) -> &mut [f32] {
        &mut self.buffers
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let (mean, _) = self.running();
        let inv_std = self.inference_inv_std();
        for c in 0..self.channels {
            let scale = self.gamma()[c] * inv_std[c];
            let shift = self.beta()[c] - mean[c] * scale;
            let range = c * self.plane..(c + 1) * self.plane;
            for (o, &x) in output[range.clone()].iter_mut().zip(&input[range]) {
                *o = x * scale + shift;
            }
        }
    }

    fn backward(
        &self,
        input: &[f32],
        _output: &[f32],
        _cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        grad_params: &mut [f32],
    ) {
        let (mean, _) = self.running();
        let inv_std = self.inference_inv_std();
        let (grad_gamma, grad_beta) = grad_params.split_at_mut(self.channels);
        for c in 0..self.channels {
            let range = c * self.plane..(c + 1) * self.plane;
            let scale = self.gamma()[c] * inv_std[c];
            for ((gi, &g), &x) in grad_input[range.clone()]
                .iter_mut()
                .zip(&grad_output[range.clone()])
                .zip(&input[range])
            {
                *gi = g * scale;
                grad_gamma[c] += g * (x - mean[c]) * inv_std[c];
                grad_beta[c] += g;
            }
        }
    }

    fn forward_batch(
        &self,
        inputs: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
        caches: &mut [Vec<f32>],
        buffers: &mut [f32],
        ctx: &mut BatchContext,
    ) {
        let (channels, plane) = (self.channels, self.plane);
        let mut sums = vec![0.0f32; channels + 1];
        for input in inputs {
            for (sum, values) in sums.iter_mut().zip(input.chunks_exact(plane)) {
                *sum += values.iter().map(|&v| v as f64).sum::<f64>() as f32;
            }
        }
        sums[channels] = inputs.len() as f32;
        ctx.reducer.sum(&mut sums);
        let count = sums[channels] * plane as f32;
        let mean: Vec<f32> = sums[..channels].iter().map(|s| s / count).collect();

        let mut var = vec![0.0f32; channels];
        for input in inputs {
            for ((v, values), &m) in var.iter_mut().zip(input.chunks_exact(plane)).zip(&mean) {
                *v += values
                    .iter()
                    .map(|&x| ((x - m) as f64).powi(2))
                    .sum::<f64>() as f32;
            }
        }
        ctx.reducer.sum(&mut var);
        var.iter_mut().for_each(|v| *v /= count);
        let inv_std: Vec<f32> = var.iter().map(|v| 1.0 / (v + EPSILON).sqrt()).collect();

        for ((input, output), cache) in inputs.iter().zip(outputs).zip(caches) {
            cache.clear();
            cache.resize(self.input_len(), 0.0);
            for c in 0..channels {
                let range = c * plane..(c + 1) * plane;
                let (gamma, beta) = (self.gamma()[c], self.beta()[c]);
                for ((o, xh), &x) in output[range.clone()]
                    .iter_mut()
                    .zip(&mut cache[range.clone()])
                    .zip(&input[range])
                {
                    *xh = (x - mean[c]) * inv_std[c];
                    *o = gamma * *xh + beta;
                }
            }
            cache.extend_from_slice(&inv_std);
        }

        let unbiased = if count > 1.0 {
            count / (count - 1.0)
        } else {
            1.0
        };
        let (running_mean, running_var) = self.running();
        let (new_mean, new_var) = buffers.split_at_mut(channels);
        for c in 0..channels {
            new_mean[c] = (1.0 - MOMENTUM) * running_mean[c] + MOMENTUM * mean[c];
            new_var[c] = (1.0 - MOMENTUM) * running_var[c] + MOMENTUM * var[c] * unbiased;
        }
    }

    fn backward_batch(
        &self,
        _inputs: &[Vec<f32>],
        _outputs: &[Vec<f32>],
        caches: &[Vec<f32>],
        grad_outputs: &[Vec<f32>],
        grad_inputs: &mut [Vec<f32>],
        grad_params: &mut [f32],
        ctx: &mut BatchContext,
    ) {
        let (channels, plane, len) = (self.channels, self.plane, self.input_len());
        let mut sums = vec![0.0f32; 2 * channels + 1];
        for (cache, grad) in caches.iter().zip(grad_outputs) {
            for c in 0..channels {
                let range = c * plane..(c + 1) * plane;
                let (mut g_sum, mut gx_sum) = (0.0f64, 0.0f64);
                for (&g, &xh) in grad[range.clone()].iter().zip(&cache[range]) {
                    g_sum += g as f64;
                    gx_sum += (g * xh) as f64;
                }
                sums[c] += g_sum as f32;
                sums[channels + c] += gx_sum as f32;
            }
        }
        let (grad_gamma, grad_beta) = grad_params.split_at_mut(channels);
        for c in 0..channels {
            grad_beta[c] += sums[c];
            grad_gamma[c] += sums[channels + c];
        }
        sums[2 * channels] = caches.len() as f32;
        ctx.reducer.sum(&mut sums);
        let count = sums[2 * channels] * plane as f32;

        for ((cache, grad), grad_input) in caches.iter().zip(grad_outputs).zip(grad_inputs) {
            let inv_std = &cache[len..];
            for c in 0..channels {
                let range = c * plane..(c + 1) * plane;
                let scale = self.gamma()[c] * inv_std[c] / count;
                let (g_sum, gx_sum) = (sums[c], sums[channels + c]);
                for ((gi, &g), &xh) in grad_input[range.clone()]
                    .iter_mut()
                    .zip(&grad[range.clone()])
                    .zip(&cache[range])
                {
                    *gi = scale * (count * g - g_sum - xh * gx_sum);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::NoReduce;
    use crate::rng::Rng;

    #[test]
    fn normalises_each_channel_over_the_batch() {
        let layer = BatchNorm::new(2, 2);
        let inputs = vec![vec![1.0, 3.0, 10.0, 10.0], vec![5.0, 7.0, 20.0, 20.0]];
        let mut outputs = vec![vec![0.0; 4]; 2];
        let mut caches = vec![Vec::new(); 2];
        let mut buffers = layer.buffers().to_vec();
        let mut rng = Rng::new(1);
        let mut ctx = BatchContext {
            rng: &mut rng,
            reducer: &NoReduce,
        };
        layer.forward_batch(&inputs, &mut outputs, &mut caches, &mut buffers, &mut ctx);
        let channel = |c: usize| -> Vec<f32> {
            outputs
                .iter()
                .flat_map(|o| o[c * 2..c * 2 + 2].to_vec())
                .collect()
        };
        for c in 0..2 {
            let values = channel(c);
            let mean = values.iter().sum::<f32>() / 4.0;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-3, "{values:?}");
        }
        assert!((buffers[0] - 0.4).abs() < 1e-5);
        assert!((buffers[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn starts_as_the_identity_at_inference() {
        let layer = BatchNorm::new(1, 3);
        let mut output = [0.0; 3];
        layer.forward(&[1.0, -2.0, 0.5], &mut output);
        for (o, x) in output.iter().zip([1.0, -2.0, 0.5]) {
            assert!((o - x).abs() < 1e-4);
        }
    }
}
