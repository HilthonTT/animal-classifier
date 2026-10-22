use super::{Layer, LayerKind, parse_config};
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct Residual {
    len: usize,
    span: usize,
}

impl Residual {
    pub fn new(len: usize, span: usize) -> Self {
        Self::with_shape(len, span).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [len, span] = parse_config(config, "Residual")?;
        Self::with_shape(len, span)
    }

    fn with_shape(len: usize, span: usize) -> Result<Self> {
        if len == 0 || span == 0 {
            return Err(Error::format("Residual needs a non-zero length and span"));
        }
        Ok(Self { len, span })
    }
}

impl Layer for Residual {
    fn kind(&self) -> LayerKind {
        LayerKind::Residual
    }

    fn config(&self) -> Vec<u32> {
        vec![self.len as u32, self.span as u32]
    }

    fn describe(&self) -> String {
        format!("Residual (adds the input of {} layers back)", self.span)
    }

    fn input_len(&self) -> usize {
        self.len
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn shortcut(&self) -> Option<usize> {
        Some(self.span)
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
    }

    fn backward(
        &self,
        _input: &[f32],
        _output: &[f32],
        _cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        _grad_params: &mut [f32],
    ) {
        grad_input.copy_from_slice(grad_output);
    }
}
