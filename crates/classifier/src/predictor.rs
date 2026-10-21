use crate::nn::Network;

pub trait Predictor: Sync {
    fn labels(&self) -> &[String];

    fn input_len(&self) -> usize;

    fn predict(&self, input: &[f32]) -> Vec<f32>;

    fn logit_gradient(&self, input: &[f32], class: usize) -> Vec<f32>;
}

impl Predictor for Network {
    fn labels(&self) -> &[String] {
        Network::labels(self)
    }

    fn input_len(&self) -> usize {
        Network::input_len(self)
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        Network::predict(self, input)
    }

    fn logit_gradient(&self, input: &[f32], class: usize) -> Vec<f32> {
        let mut seed = vec![0.0; self.output_len()];
        seed[class] = 1.0;
        self.input_gradient(input, &seed)
    }
}
