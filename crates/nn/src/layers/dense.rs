use super::{Layer, LayerKind, parse_config};
use crate::error::{Error, Result};
use crate::init::Init;
use crate::math::{axpy, dot};
use crate::rng::Rng;

/// Fully connected layer: `y = W·x + b`.
#[derive(Clone, Debug)]
pub struct Dense {
    inputs: usize,
    outputs: usize,
    /// `outputs × inputs` weights (row-major) followed by `outputs` biases.
    params: Vec<f32>,
}

impl Dense {
    /// Creates a layer with He-initialised weights (well suited to ReLU networks).
    pub fn new(inputs: usize, outputs: usize, rng: &mut Rng) -> Self {
        Self::with_init(inputs, outputs, Init::He, rng)
    }

    /// Creates a layer whose weights are filled by `init`.
    pub fn with_init(inputs: usize, outputs: usize, init: Init, rng: &mut Rng) -> Self {
        let mut layer = Self::with_shape(inputs, outputs).unwrap_or_else(|e| panic!("{e}"));
        init.fill(&mut layer.params[..inputs * outputs], inputs, outputs, rng);
        layer
    }
    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [inputs, outputs] = parse_config(config, "Dense")?;
        Self::with_shape(inputs, outputs)
    }

    fn with_shape(inputs: usize, outputs: usize) -> Result<Self> {
        if inputs == 0 || outputs == 0 {
            return Err(Error::format("Dense dimensions must be non-zero"));
        }
        Ok(Self {
            inputs,
            outputs,
            params: vec![0.0; inputs * outputs + outputs],
        })
    }
}

impl Layer for Dense {
    fn kind(&self) -> LayerKind {
        LayerKind::Dense
    }

    fn config(&self) -> Vec<u32> {
        vec![self.inputs as u32, self.outputs as u32]
    }

    fn describe(&self) -> String {
        format!("Dense {} -> {}", self.inputs, self.outputs)
    }

    fn input_len(&self) -> usize {
        self.inputs
    }

    fn output_len(&self) -> usize {
        self.outputs
    }

    fn params(&self) -> &[f32] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn weight_len(&self) -> usize {
        self.inputs * self.outputs
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let (weights, bias) = self.params.split_at(self.inputs * self.outputs);
        for (o, out) in output.iter_mut().enumerate() {
            let row = &weights[o * self.inputs..(o + 1) * self.inputs];
            *out = bias[o] + dot(row, input);
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
        let n = self.inputs * self.outputs;
        let weights = &self.params[..n];
        let (grad_weights, grad_bias) = grad_params.split_at_mut(n);
        grad_input.fill(0.0);
        for (o, &g) in grad_output.iter().enumerate() {
            grad_bias[o] += g;
            if g == 0.0 {
                continue;
            }
            let range = o * self.inputs..(o + 1) * self.inputs;
            axpy(g, input, &mut grad_weights[range.clone()]);
            axpy(g, &weights[range], grad_input);
        }
    }
}
