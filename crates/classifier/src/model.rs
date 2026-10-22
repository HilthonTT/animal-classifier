use std::fmt;
use std::str::FromStr;

use crate::data::{CHANNELS, IMAGE_SIZE};
use crate::nn::{
    Activation, ActivationFn, AvgPool2d, BatchNorm, Conv2d, Dense, Dropout, GlobalAvgPool, Init,
    MaxPool2d, Network, Residual, Rng,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pool {
    #[default]
    Max,
    Avg,
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Pool::Max => "max",
            Pool::Avg => "avg",
        })
    }
}

impl FromStr for Pool {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "max" => Ok(Pool::Max),
            "avg" | "average" => Ok(Pool::Avg),
            _ => Err(format!("unknown pooling `{s}` (expected max or avg)")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelOptions {
    pub activation: ActivationFn,
    pub init: Init,
    pub dropout: f32,
    pub width: usize,
    pub batch_norm: bool,
    pub residual: bool,
    pub pool: Pool,
    pub global_pool: bool,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            activation: ActivationFn::Relu,
            init: Init::He,
            dropout: 0.25,
            width: 16,
            batch_norm: false,
            residual: false,
            pool: Pool::Max,
            global_pool: false,
        }
    }
}

pub fn pawnet_cnn(labels: &[String], options: &ModelOptions, rng: &mut Rng) -> Network {
    let ModelOptions {
        activation,
        init,
        dropout,
        width,
        batch_norm,
        residual,
        pool,
        global_pool,
    } = *options;
    let mut net = Network::new(labels.to_vec());
    let mut channels = CHANNELS;
    let mut size = IMAGE_SIZE;
    let conv_unit = |net: &mut Network, rng: &mut Rng, size: usize, shape: (usize, usize)| {
        let (in_channels, out_channels) = shape;
        net.push(Conv2d::with_init(
            in_channels,
            size,
            size,
            out_channels,
            3,
            1,
            init,
            rng,
        ));
        if batch_norm {
            net.push(BatchNorm::new(out_channels, size * size));
        }
    };
    for out_channels in [width, width * 2, width * 4] {
        conv_unit(&mut net, rng, size, (channels, out_channels));
        net.push(Activation::new(activation, net.output_len()));
        if residual {
            let start = net.layers().len();
            conv_unit(&mut net, rng, size, (out_channels, out_channels));
            net.push(Activation::new(activation, net.output_len()));
            conv_unit(&mut net, rng, size, (out_channels, out_channels));
            net.push(Residual::new(net.output_len(), net.layers().len() - start));
            net.push(Activation::new(activation, net.output_len()));
        }
        match pool {
            Pool::Max => net.push(MaxPool2d::new(out_channels, size, size, 2)),
            Pool::Avg => net.push(AvgPool2d::new(out_channels, size, size, 2)),
        };
        channels = out_channels;
        size /= 2;
    }
    let mut features = net.output_len();
    if global_pool {
        net.push(GlobalAvgPool::new(channels, size, size));
        features = channels;
    } else {
        let hidden = 128;
        net.push(Dense::with_init(features, hidden, init, rng));
        if batch_norm {
            net.push(BatchNorm::new(hidden, 1));
        }
        net.push(Activation::new(activation, hidden));
        features = hidden;
    }
    if dropout > 0.0 {
        net.push(Dropout::new(features, dropout));
    }
    net.push(Dense::with_init(features, labels.len(), init, rng));
    net
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::LayerKind;

    fn labels() -> Vec<String> {
        ["a", "b", "c"].iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn builds_with_every_option() {
        let labels = labels();
        let default = pawnet_cnn(&labels, &ModelOptions::default(), &mut Rng::new(1));
        assert_eq!(default.output_len(), 3);
        assert_eq!(default.param_count(), 155_171);

        let options = ModelOptions {
            activation: ActivationFn::Elu,
            init: Init::Xavier,
            dropout: 0.0,
            width: 8,
            ..ModelOptions::default()
        };
        let small = pawnet_cnn(&labels, &options, &mut Rng::new(1));
        assert_eq!(small.layers().len(), default.layers().len() - 1);
        assert!(small.param_count() < default.param_count());
    }

    #[test]
    fn builds_a_residual_batch_norm_network() {
        let options = ModelOptions {
            batch_norm: true,
            residual: true,
            pool: Pool::Avg,
            global_pool: true,
            ..ModelOptions::default()
        };
        let net = pawnet_cnn(&labels(), &options, &mut Rng::new(1));
        let count = |kind| net.layers().iter().filter(|l| l.kind() == kind).count();
        assert_eq!(count(LayerKind::Conv2d), 9);
        assert_eq!(count(LayerKind::BatchNorm), 9);
        assert_eq!(count(LayerKind::Residual), 3);
        assert_eq!(count(LayerKind::AvgPool2d), 3);
        assert_eq!(count(LayerKind::GlobalAvgPool), 1);
        assert_eq!(count(LayerKind::Dense), 1);
        assert_eq!(net.output_len(), 3);
        let probabilities = net.predict(&vec![0.5; net.input_len()]);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn parses_pooling() {
        assert_eq!("avg".parse::<Pool>(), Ok(Pool::Avg));
        assert_eq!(Pool::Max.to_string(), "max");
        assert!("min".parse::<Pool>().is_err());
    }
}
