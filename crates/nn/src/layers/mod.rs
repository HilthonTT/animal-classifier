mod pool;

use crate::{
    error::{Error, Result},
    rng::Rng,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerKind {
    Dense = 1,
    Conv2d = 2,
    MaxPool2d = 3,
    Activation = 4,
    Dropout = 5,
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
            other => Err(Error::format(format!("unknown layer kind {other}"))),
        }
    }
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
