use crate::error::{Error, Result};
use crate::layers::Layer;
use crate::loss::{Loss, argmax, softmax};
use crate::optimizer::Optimizer;
use crate::rng::Rng;

pub struct Network {
    pub(crate) layers: Vec<Box<dyn Layer>>,
    pub(crate) labels: Vec<String>,
}

impl Network {
    pub fn new(labels: Vec<String>) -> Self {
        Self {
            layers: Vec::new(),
            labels,
        }
    }

    pub fn push(&mut self, layer: impl Layer + 'static) -> &mut Self {
        if let Err(e) = self.try_push(Box::new(layer)) {
            panic!("{e}");
        }
        self
    }

    pub fn try_push(&mut self, layer: Box<dyn Layer>) -> Result<()> {
        if let Some(last) = self.layers.last() {
            if last.output_len() != layer.input_len() {
                return Err(Error::format(format!(
                    "layer `{}` expects {} inputs but `{}` produces {}",
                    layer.describe(),
                    layer.input_len(),
                    last.describe(),
                    last.output_len()
                )));
            }
        }
        self.layers.push(layer);
        Ok(())
    }

    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [Box<dyn Layer>] {
        &mut self.layers
    }

    /// Class names, indexed by class id.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn input_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.input_len())
    }

    pub fn output_len(&self) -> usize {
        self.layers.last().map_or(0, |l| l.output_len())
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.params().len()).sum()
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_len(), "wrong input size");
        let mut current = input.to_vec();
        for layer in &self.layers {
            let mut next = vec![0.0; layer.output_len()];
            layer.forward(&current, &mut next);
            current = next;
        }
        current
    }

    /// Class probabilities for one sample.
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        softmax(&self.forward(input))
    }

    /// Zeroed gradient buffers, one per layer, shaped like the parameters.
    pub fn zero_grads(&self) -> Vec<Vec<f32>> {
        self.layers
            .iter()
            .map(|l| vec![0.0; l.params().len()])
            .collect()
    }

    /// Training-mode forward + backward pass for one labelled sample. Parameter gradients
    /// are **added** to `grads`. `rng` drives stochastic layers such as dropout.
    /// Returns the loss and the predicted class.
    pub fn accumulate_gradients(
        &self,
        input: &[f32],
        target: usize,
        loss: Loss,
        rng: &mut Rng,
        grads: &mut [Vec<f32>],
    ) -> (f32, usize) {
        // Keep every intermediate activation (and layer cache): the backward pass needs them.
        let mut activations: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len() + 1);
        let mut caches: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        activations.push(input.to_vec());
        for layer in &self.layers {
            let mut output = vec![0.0; layer.output_len()];
            let mut cache = Vec::new();
            let input = activations.last().expect("input pushed above");
            layer.forward_train(input, &mut output, rng, &mut cache);
            activations.push(output);
            caches.push(cache);
        }

        let logits = activations.last().expect("input pushed above");
        let predicted = argmax(logits);
        let (loss, grad) = loss.evaluate(logits, target);
        self.backpropagate(&activations, &caches, grad, grads);
        (loss, predicted)
    }

    pub fn input_gradient(&self, input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_len(), "wrong input size");
        assert_eq!(grad_output.len(), self.output_len(), "wrong gradient size");
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());
        for layer in &self.layers {
            let mut output = vec![0.0; layer.output_len()];
            layer.forward(activations.last().expect("input pushed above"), &mut output);
            activations.push(output);
        }
        let caches = vec![Vec::new(); self.layers.len()];
        let mut grads = self.zero_grads();
        self.backpropagate(&activations, &caches, grad_output.to_vec(), &mut grads)
    }

    fn backpropagate(
        &self,
        activations: &[Vec<f32>],
        caches: &[Vec<f32>],
        mut grad: Vec<f32>,
        grads: &mut [Vec<f32>],
    ) -> Vec<f32> {
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let mut grad_input = vec![0.0; layer.input_len()];
            layer.backward(
                &activations[i],
                &activations[i + 1],
                &caches[i],
                &grad,
                &mut grad_input,
                &mut grads[i],
            );
            grad = grad_input;
        }
        grad
    }

    /// L2 regularisation ("weight decay"): adds `lambda · w` to the gradient of every
    /// weight (not bias), which pulls weights towards zero and discourages over-fitting.
    pub fn add_weight_decay(&self, grads: &mut [Vec<f32>], lambda: f32) {
        for (layer, grad) in self.layers.iter().zip(grads) {
            let n = layer.weight_len();
            for (g, &w) in grad[..n].iter_mut().zip(&layer.params()[..n]) {
                *g += lambda * w;
            }
        }
    }

    /// Lets `optimizer` update every layer's parameters from `grads`.
    pub fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, grads: &[Vec<f32>]) {
        let mut params: Vec<&mut [f32]> = self.layers.iter_mut().map(|l| l.params_mut()).collect();
        optimizer.step(&mut params, grads);
    }

    /// A table describing every layer.
    pub fn summary(&self) -> String {
        let mut out = format!("{:<44}{:>10}{:>12}\n", "Layer", "Outputs", "Params");
        out += &format!("{}\n", "-".repeat(66));
        for layer in &self.layers {
            out += &format!(
                "{:<44}{:>10}{:>12}\n",
                layer.describe(),
                layer.output_len(),
                layer.params().len()
            );
        }
        out += &format!("{}\n", "-".repeat(66));
        out += &format!(
            "{:<54}{:>12}\n",
            format!("Total ({} classes)", self.labels.len()),
            self.param_count()
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Activation, Dense, Dropout};

    #[test]
    #[should_panic(expected = "expects")]
    fn push_rejects_shape_mismatch() {
        let mut rng = Rng::new(1);
        let mut net = Network::new(vec![]);
        net.push(Dense::new(4, 3, &mut rng))
            .push(Dense::new(5, 2, &mut rng));
    }

    #[test]
    fn weight_decay_skips_biases() {
        let mut rng = Rng::new(1);
        let mut net = Network::new(vec!["a".into()]);
        net.push(Dense::new(2, 1, &mut rng));
        net.layers_mut()[0]
            .params_mut()
            .copy_from_slice(&[1.0, -2.0, 5.0]);
        let mut grads = net.zero_grads();
        net.add_weight_decay(&mut grads, 0.5);
        assert_eq!(grads[0], vec![0.5, -1.0, 0.0]);
    }

    #[test]
    fn dropout_only_acts_while_training() {
        let mut rng = Rng::new(3);
        let mut net = Network::new(vec!["a".into(), "b".into()]);
        net.push(Dense::new(4, 8, &mut rng))
            .push(Activation::relu(8))
            .push(Dropout::new(8, 0.5))
            .push(Dense::new(8, 2, &mut rng));
        let input = [0.5, -1.0, 2.0, 0.1];
        assert_eq!(net.forward(&input), net.forward(&input));

        let mut grads = net.zero_grads();
        let (a, _) = net.accumulate_gradients(&input, 0, Loss::CrossEntropy, &mut rng, &mut grads);
        let (b, _) = net.accumulate_gradients(&input, 0, Loss::CrossEntropy, &mut rng, &mut grads);
        assert_ne!(a, b, "dropout masks should differ between passes");
    }

    #[test]
    fn input_gradient_matches_finite_differences() {
        let mut rng = Rng::new(5);
        let mut net = Network::new(vec!["a".into(), "b".into(), "c".into()]);
        net.push(Dense::new(4, 6, &mut rng))
            .push(Activation::new(crate::ActivationFn::Tanh, 6))
            .push(Dropout::new(6, 0.5))
            .push(Dense::new(6, 3, &mut rng));
        let input = [0.3, -0.7, 1.1, 0.2];
        let seed = [0.5, -1.0, 2.0];
        let score =
            |x: &[f32]| -> f32 { net.forward(x).iter().zip(&seed).map(|(o, s)| o * s).sum() };
        let analytic = net.input_gradient(&input, &seed);
        assert_eq!(analytic, net.input_gradient(&input, &seed));
        let eps = 1e-3;
        for i in 0..input.len() {
            let (mut plus, mut minus) = (input, input);
            plus[i] += eps;
            minus[i] -= eps;
            let numerical = (score(&plus) - score(&minus)) / (2.0 * eps);
            assert!(
                (numerical - analytic[i]).abs() < 1e-2 * (0.1 + numerical.abs()),
                "input {i}: analytic {} vs numerical {numerical}",
                analytic[i]
            );
        }
    }
}
