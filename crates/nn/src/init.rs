use std::fmt;
use std::str::FromStr;

use crate::rng::Rng;

/// How a layer's weights are filled before training. Biases always start at zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Init {
    /// Normal with variance `2 / fan_in` (He et al., 2015). The best fit for ReLU-like activations.
    #[default]
    He,
    /// Normal with variance `2 / (fan_in + fan_out)` (Glorot & Bengio, 2010). Suits tanh / sigmoid.
    Xavier,
    /// Uniform in `±1 / √fan_in`.
    Uniform,
    /// All zeros. Only useful for experiments: every neuron then learns the same thing.
    Zero,
}

impl Init {
    pub const ALL: [Init; 4] = [Init::He, Init::Xavier, Init::Uniform, Init::Zero];

    pub fn fill(self, weights: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut Rng) {
        let (fan_in, fan_out) = (fan_in.max(1) as f32, fan_out.max(1) as f32);
        for w in weights {
            *w = match self {
                Init::He => rng.normal() * (2.0 / fan_in).sqrt(),
                Init::Xavier => rng.normal() * (2.0 / (fan_in + fan_out)).sqrt(),
                Init::Uniform => (rng.next_f32() * 2.0 - 1.0) / fan_in.sqrt(),
                Init::Zero => 0.0,
            };
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Init::He => "he",
            Init::Xavier => "xavier",
            Init::Uniform => "uniform",
            Init::Zero => "zero",
        }
    }
}

impl fmt::Display for Init {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Init {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Init::ALL
            .into_iter()
            .find(|i| i.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                format!("unknown initialiser `{s}` (expected he, xavier, uniform or zero)")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variance(init: Init, fan_in: usize, fan_out: usize) -> f32 {
        let mut w = vec![0.0; 20_000];
        init.fill(&mut w, fan_in, fan_out, &mut Rng::new(1));
        w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32
    }

    #[test]
    fn variances_match_their_formulas() {
        assert!((variance(Init::He, 50, 10) - 2.0 / 50.0).abs() < 0.003);
        assert!((variance(Init::Xavier, 50, 50) - 2.0 / 100.0).abs() < 0.002);
        assert!((variance(Init::Uniform, 48, 1) - 1.0 / (3.0 * 48.0)).abs() < 0.001);
        assert_eq!(variance(Init::Zero, 5, 5), 0.0);
    }

    #[test]
    fn parses_names() {
        assert_eq!("Xavier".parse::<Init>(), Ok(Init::Xavier));
        assert!("nope".parse::<Init>().is_err());
    }
}
