use pawnet::model::{self, ModelOptions};
use pawnet::nn::{Loss, Rng};
use std::time::Instant;

fn main() {
    let labels: Vec<String> = (0..6).map(|i| i.to_string()).collect();
    let mut rng = Rng::new(1);
    let net = model::pawnet_cnn(&labels, &ModelOptions::default(), &mut rng);
    let inputs: Vec<Vec<f32>> = (0..256)
        .map(|_| (0..3072).map(|_| rng.normal()).collect())
        .collect();
    let mut grads = net.zero_grads();
    for _ in 0..2 {
        for x in &inputs[..32] {
            net.accumulate_gradients(x, 1, Loss::CrossEntropy, &mut rng, &mut grads);
        }
    }
    let t = Instant::now();
    for x in &inputs {
        net.accumulate_gradients(x, 1, Loss::CrossEntropy, &mut rng, &mut grads);
    }
    let train = t.elapsed().as_secs_f64() / inputs.len() as f64;
    let t = Instant::now();
    let mut s = 0.0;
    for x in &inputs {
        s += net.forward(x)[0];
    }
    let fwd = t.elapsed().as_secs_f64() / inputs.len() as f64;
    println!(
        "train step {:.1} us/sample, forward {:.1} us/sample ({s})",
        train * 1e6,
        fwd * 1e6
    );
}
