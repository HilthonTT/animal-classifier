use std::fmt;
use std::str::FromStr;

use super::{Layer, LayerKind};
use crate::error::{Error, Result};

/// Slope of [`ActivationFn::LeakyRelu`] for negative inputs.
pub const LEAKY_RELU_SLOPE: f32 = 0.01;
/// α of [`ActivationFn::Elu`].
pub const ELU_ALPHA: f32 = 1.0;

/// Element-wise non-linearities. The numeric ids are part of the model file format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivationFn {
    /// `max(0, x)`
    #[default]
    Relu = 0,
    /// `x` if positive, else `0.01·x`: never completely "dies".
    LeakyRelu = 1,
    /// `x` if positive, else `α·(eˣ − 1)`: smooth around zero.
    Elu = 2,
    /// `1 / (1 + e⁻ˣ)`, squashes into (0, 1).
    Sigmoid = 3,
    /// Squashes into (−1, 1).
    Tanh = 4,
    /// `ln(1 + eˣ)`, a smooth ReLU.
    Softplus = 5,
}

impl ActivationFn {
    pub const ALL: [ActivationFn; 6] = [
        ActivationFn::Relu,
        ActivationFn::LeakyRelu,
        ActivationFn::Elu,
        ActivationFn::Sigmoid,
        ActivationFn::Tanh,
        ActivationFn::Softplus,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ActivationFn::Relu => "relu",
            ActivationFn::LeakyRelu => "leaky-relu",
            ActivationFn::Elu => "elu",
            ActivationFn::Sigmoid => "sigmoid",
            ActivationFn::Tanh => "tanh",
            ActivationFn::Softplus => "softplus",
        }
    }

    pub fn apply(self, x: f32) -> f32 {
        match self {
            ActivationFn::Relu => x.max(0.0),
            ActivationFn::LeakyRelu => {
                if x > 0.0 {
                    x
                } else {
                    LEAKY_RELU_SLOPE * x
                }
            }
            ActivationFn::Elu => {
                if x > 0.0 {
                    x
                } else {
                    ELU_ALPHA * x.exp_m1()
                }
            }
            ActivationFn::Sigmoid => sigmoid(x),
            ActivationFn::Tanh => x.tanh(),
            // ln(1 + eˣ), rearranged so large |x| neither overflows nor loses precision.
            ActivationFn::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        }
    }

    pub fn derivative(self, x: f32, y: f32) -> f32 {
        match self {
            ActivationFn::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFn::LeakyRelu => {
                if x > 0.0 {
                    1.0
                } else {
                    LEAKY_RELU_SLOPE
                }
            }
            ActivationFn::Elu => {
                if x > 0.0 {
                    1.0
                } else {
                    y + ELU_ALPHA
                }
            }
            ActivationFn::Sigmoid => y * (1.0 - y),
            ActivationFn::Tanh => 1.0 - y * y,
            ActivationFn::Softplus => sigmoid(x),
        }
    }

    fn from_id(id: u32) -> Result<Self> {
        ActivationFn::ALL
            .into_iter()
            .find(|f| *f as u32 == id)
            .ok_or_else(|| Error::format(format!("unknown activation function id {id}")))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl fmt::Display for ActivationFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for ActivationFn {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        let wanted = s.to_ascii_lowercase().replace('_', "-");
        ActivationFn::ALL
            .into_iter()
            .find(|f| f.name() == wanted || f.name().replace('-', "") == wanted)
            .ok_or_else(|| {
                let names: Vec<_> = ActivationFn::ALL.iter().map(|f| f.name()).collect();
                format!(
                    "unknown activation `{s}` (expected one of: {})",
                    names.join(", ")
                )
            })
    }
}

/// Applies an [`ActivationFn`] to every value.
#[derive(Clone, Debug)]
pub struct Activation {
    func: ActivationFn,
    len: usize,
}

impl Activation {
    /// `len` is the number of values per sample (the previous layer's output length).
    pub fn new(func: ActivationFn, len: usize) -> Self {
        assert!(len > 0, "Activation length must be non-zero");
        Self { func, len }
    }

    /// Shorthand for `Activation::new(ActivationFn::Relu, len)`.
    pub fn relu(len: usize) -> Self {
        Self::new(ActivationFn::Relu, len)
    }

    pub fn function(&self) -> ActivationFn {
        self.func
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        // Version 1 model files only had ReLU and stored just the length.
        let (len, func) = match *config {
            [len] => (len, ActivationFn::Relu),
            [len, id] => (len, ActivationFn::from_id(id)?),
            _ => {
                return Err(Error::format(
                    "Activation layer expects 1 or 2 configuration values",
                ));
            }
        };
        if len == 0 {
            return Err(Error::format("Activation length must be non-zero"));
        }
        Ok(Self {
            func,
            len: len as usize,
        })
    }
}

impl Layer for Activation {
    fn kind(&self) -> LayerKind {
        LayerKind::Activation
    }

    fn config(&self) -> Vec<u32> {
        vec![self.len as u32, self.func as u32]
    }

    fn describe(&self) -> String {
        match self.func {
            ActivationFn::Relu => "ReLU".to_string(),
            ActivationFn::LeakyRelu => "LeakyReLU".to_string(),
            ActivationFn::Elu => "ELU".to_string(),
            ActivationFn::Sigmoid => "Sigmoid".to_string(),
            ActivationFn::Tanh => "Tanh".to_string(),
            ActivationFn::Softplus => "Softplus".to_string(),
        }
    }

    fn input_len(&self) -> usize {
        self.len
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for (o, &x) in output.iter_mut().zip(input) {
            *o = self.func.apply(x);
        }
    }

    fn backward(
        &self,
        input: &[f32],
        output: &[f32],
        _cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        _grad_params: &mut [f32],
    ) {
        for (((gi, &g), &x), &y) in grad_input
            .iter_mut()
            .zip(grad_output)
            .zip(input)
            .zip(output)
        {
            *gi = g * self.func.derivative(x, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_match_finite_differences() {
        for func in ActivationFn::ALL {
            for x in [-2.5f32, -0.7, 0.3, 1.9] {
                let h = 1e-3;
                let numeric = (func.apply(x + h) - func.apply(x - h)) / (2.0 * h);
                let analytic = func.derivative(x, func.apply(x));
                assert!(
                    (numeric - analytic).abs() < 1e-2,
                    "{func} at {x}: {analytic} vs {numeric}"
                );
            }
        }
    }

    #[test]
    fn softplus_is_stable_for_large_inputs() {
        assert_eq!(ActivationFn::Softplus.apply(100.0), 100.0);
        assert!(ActivationFn::Softplus.apply(-100.0) >= 0.0);
    }

    #[test]
    fn parses_names() {
        assert_eq!(
            "LeakyReLU".parse::<ActivationFn>(),
            Ok(ActivationFn::LeakyRelu)
        );
        assert_eq!(
            "leaky_relu".parse::<ActivationFn>(),
            Ok(ActivationFn::LeakyRelu)
        );
        assert!("swish".parse::<ActivationFn>().is_err());
    }

    #[test]
    fn reads_version_1_relu_config() {
        assert_eq!(
            Activation::from_config(&[8]).unwrap().function(),
            ActivationFn::Relu
        );
    }
}
