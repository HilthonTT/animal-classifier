mod activation;
mod conv2d;
mod dense;
mod dropout;
mod norm;
mod pool;
mod residual;

pub use activation::{Activation, ActivationFn};
pub use conv2d::Conv2d;
pub use dense::Dense;
pub use dropout::Dropout;
pub use norm::BatchNorm;
pub use pool::{AvgPool2d, GlobalAvgPool, MaxPool2d};
pub use residual::Residual;

use crate::error::{Error, Result};
use crate::reduce::Reducer;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerKind {
    Dense = 1,
    Conv2d = 2,
    MaxPool2d = 3,
    Activation = 4,
    Dropout = 5,
    BatchNorm = 6,
    AvgPool2d = 7,
    GlobalAvgPool = 8,
    Residual = 9,
}

impl TryFrom<u8> for LayerKind {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(LayerKind::Dense),
            2 => Ok(LayerKind::Conv2d),
            3 => Ok(LayerKind::MaxPool2d),
            4 => Ok(LayerKind::Activation),
            5 => Ok(LayerKind::Dropout),
            6 => Ok(LayerKind::BatchNorm),
            7 => Ok(LayerKind::AvgPool2d),
            8 => Ok(LayerKind::GlobalAvgPool),
            9 => Ok(LayerKind::Residual),
            other => Err(Error::format(format!("unknown layer kind {other}"))),
        }
    }
}

pub(crate) fn from_config(kind: LayerKind, config: &[u32]) -> Result<Box<dyn Layer>> {
    Ok(match kind {
        LayerKind::Dense => Box::new(Dense::from_config(config)?),
        LayerKind::Conv2d => Box::new(Conv2d::from_config(config)?),
        LayerKind::MaxPool2d => Box::new(MaxPool2d::from_config(config)?),
        LayerKind::Activation => Box::new(Activation::from_config(config)?),
        LayerKind::Dropout => Box::new(Dropout::from_config(config)?),
        LayerKind::BatchNorm => Box::new(BatchNorm::from_config(config)?),
        LayerKind::AvgPool2d => Box::new(AvgPool2d::from_config(config)?),
        LayerKind::GlobalAvgPool => Box::new(GlobalAvgPool::from_config(config)?),
        LayerKind::Residual => Box::new(Residual::from_config(config)?),
    })
}

pub struct BatchContext<'a> {
    pub rng: &'a mut Rng,
    pub reducer: &'a dyn Reducer,
}

pub trait Layer: Send + Sync {
    fn kind(&self) -> LayerKind;

    fn config(&self) -> Vec<u32>;

    fn describe(&self) -> String;

    fn input_len(&self) -> usize;

    fn output_len(&self) -> usize;

    fn params(&self) -> &[f32] {
        &[]
    }

    fn params_mut(&mut self) -> &mut [f32] {
        &mut []
    }

    fn weight_len(&self) -> usize {
        0
    }

    fn buffers(&self) -> &[f32] {
        &[]
    }

    fn buffers_mut(&mut self) -> &mut [f32] {
        &mut []
    }

    fn shortcut(&self) -> Option<usize> {
        None
    }

    fn forward(&self, input: &[f32], output: &mut [f32]);

    fn forward_train(
        &self,
        input: &[f32],
        output: &mut [f32],
        _rng: &mut Rng,
        _cache: &mut Vec<f32>,
    ) {
        self.forward(input, output);
    }

    fn backward(
        &self,
        input: &[f32],
        output: &[f32],
        cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        grad_params: &mut [f32],
    );

    fn forward_batch(
        &self,
        inputs: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
        caches: &mut [Vec<f32>],
        _buffers: &mut [f32],
        ctx: &mut BatchContext,
    ) {
        for ((input, output), cache) in inputs.iter().zip(outputs).zip(caches) {
            self.forward_train(input, output, ctx.rng, cache);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_batch(
        &self,
        inputs: &[Vec<f32>],
        outputs: &[Vec<f32>],
        caches: &[Vec<f32>],
        grad_outputs: &[Vec<f32>],
        grad_inputs: &mut [Vec<f32>],
        grad_params: &mut [f32],
        _ctx: &mut BatchContext,
    ) {
        for i in 0..inputs.len() {
            self.backward(
                &inputs[i],
                &outputs[i],
                &caches[i],
                &grad_outputs[i],
                &mut grad_inputs[i],
                grad_params,
            );
        }
    }
}

pub(crate) fn parse_config<const N: usize>(config: &[u32], layer: &str) -> Result<[usize; N]> {
    if config.len() != N {
        return Err(Error::format(format!(
            "{layer} layer expects {N} configuration values, found {}",
            config.len()
        )));
    }
    Ok(std::array::from_fn(|i| config[i] as usize))
}

pub(crate) fn checked_len(factors: &[usize]) -> Option<usize> {
    factors
        .iter()
        .try_fold(1usize, |acc, &f| acc.checked_mul(f))
        .filter(|&n| n <= u32::MAX as usize)
}

pub(crate) fn checked_param_len(weights: &[usize], biases: usize) -> Option<usize> {
    checked_len(weights).and_then(|w| checked_len(&[w.checked_add(biases)?]))
}
