use crate::data::{CHANNELS, IMAGE_SIZE};
use crate::nn::{Activation, ActivationFn, Conv2d, Dense, Dropout, Init, MaxPool2d, Network, Rng};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelOptions {
    pub activation: ActivationFn,
    pub init: Init,
    pub dropout: f32,
    pub width: usize,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            activation: ActivationFn::Relu,
            init: Init::He,
            dropout: 0.25,
            width: 16,
        }
    }
}

pub fn pawnet_cnn(labels: &[String], options: &ModelOptions, rng: &mut Rng) -> Network {
    let ModelOptions {
        activation,
        init,
        dropout,
        width,
    } = *options;
    let mut net = Network::new(labels.to_vec());
    let mut channels = CHANNELS;
    let mut size = IMAGE_SIZE;
    for out_channels in [width, width * 2, width * 4] {
        net.push(Conv2d::with_init(
            channels,
            size,
            size,
            out_channels,
            3,
            1,
            init,
            rng,
        ));
        net.push(Activation::new(activation, net.output_len()));
        net.push(MaxPool2d::new(out_channels, size, size, 2));
        channels = out_channels;
        size /= 2;
    }
    let hidden = 128;
    net.push(Dense::with_init(net.output_len(), hidden, init, rng));
    net.push(Activation::new(activation, hidden));
    if dropout > 0.0 {
        net.push(Dropout::new(hidden, dropout));
    }
    net.push(Dense::with_init(hidden, labels.len(), init, rng));
    net
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_with_every_option() {
        let labels: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let default = pawnet_cnn(&labels, &ModelOptions::default(), &mut Rng::new(1));
        assert_eq!(default.output_len(), 3);
        assert_eq!(default.param_count(), 155_171);

        let options = ModelOptions {
            activation: ActivationFn::Elu,
            init: Init::Xavier,
            dropout: 0.0,
            width: 8,
        };
        let small = pawnet_cnn(&labels, &options, &mut Rng::new(1));
        assert_eq!(small.layers().len(), default.layers().len() - 1);
        assert!(small.param_count() < default.param_count());
    }
}
