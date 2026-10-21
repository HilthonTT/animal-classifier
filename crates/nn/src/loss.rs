use std::fmt;
use std::str::FromStr;

/// Which loss to minimise during training.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Loss {
    /// `−ln p[target]`. The standard choice for classification.
    #[default]
    CrossEntropy,
    /// Mean of `(p − one_hot)²`. Learns more slowly, included for comparison.
    MeanSquaredError,
}

impl Loss {
    /// The loss for one sample and its gradient with respect to the logits.
    pub fn evaluate(self, logits: &[f32], target: usize) -> (f32, Vec<f32>) {
        match self {
            Loss::CrossEntropy => softmax_cross_entropy(logits, target),
            Loss::MeanSquaredError => softmax_mean_squared_error(logits, target),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Loss::CrossEntropy => "cross-entropy",
            Loss::MeanSquaredError => "mse",
        }
    }
}

impl fmt::Display for Loss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Loss {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cross-entropy" | "crossentropy" | "ce" => Ok(Loss::CrossEntropy),
            "mse" | "mean-squared-error" => Ok(Loss::MeanSquaredError),
            _ => Err(format!(
                "unknown loss `{s}` (expected cross-entropy or mse)"
            )),
        }
    }
}

/// Converts raw scores (logits) into probabilities that sum to 1.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Cross-entropy loss of `softmax(logits)` against the class `target`, together with its
/// gradient with respect to the logits (which conveniently is `probabilities − one_hot`).
pub fn softmax_cross_entropy(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let mut grad = softmax(logits);
    let loss = -grad[target].max(1e-12).ln();
    grad[target] -= 1.0;
    (loss, grad)
}

/// Mean-squared error between `softmax(logits)` and the one-hot target, with its gradient
/// with respect to the logits (chained through the softmax Jacobian).
pub fn softmax_mean_squared_error(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let p = softmax(logits);
    let n = p.len() as f32;
    let one_hot = |i: usize| if i == target { 1.0 } else { 0.0 };
    let loss = p
        .iter()
        .enumerate()
        .map(|(i, &pi)| (pi - one_hot(i)).powi(2))
        .sum::<f32>()
        / n;
    // dL/dp, then dL/dz_j = p_j · (dL/dp_j − Σ_k p_k · dL/dp_k)
    let grad_p: Vec<f32> = p
        .iter()
        .enumerate()
        .map(|(i, &pi)| 2.0 * (pi - one_hot(i)) / n)
        .collect();
    let weighted: f32 = p.iter().zip(&grad_p).map(|(a, b)| a * b).sum();
    let grad = p
        .iter()
        .zip(&grad_p)
        .map(|(&pj, &gj)| pj * (gj - weighted))
        .collect();
    (loss, grad)
}

/// Index of the largest value (the predicted class).
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let p = softmax(&[1000.0, 1001.0, 999.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&p), 1);
    }

    #[test]
    fn cross_entropy_gradient_is_prob_minus_one_hot() {
        let (loss, grad) = softmax_cross_entropy(&[0.0, 0.0], 0);
        assert!((loss - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((grad[0] + 0.5).abs() < 1e-6 && (grad[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn loss_gradients_match_finite_differences() {
        let logits = [0.3f32, -1.2, 2.0, 0.5];
        for loss in [Loss::CrossEntropy, Loss::MeanSquaredError] {
            let (_, grad) = loss.evaluate(&logits, 1);
            for i in 0..logits.len() {
                let h = 1e-3;
                let (mut up, mut down) = (logits, logits);
                up[i] += h;
                down[i] -= h;
                let numeric = (loss.evaluate(&up, 1).0 - loss.evaluate(&down, 1).0) / (2.0 * h);
                assert!(
                    (numeric - grad[i]).abs() < 1e-3,
                    "{loss} logit {i}: {} vs {numeric}",
                    grad[i]
                );
            }
        }
    }

    #[test]
    fn parses_names() {
        assert_eq!("MSE".parse::<Loss>(), Ok(Loss::MeanSquaredError));
        assert_eq!("cross-entropy".parse::<Loss>(), Ok(Loss::CrossEntropy));
    }
}
