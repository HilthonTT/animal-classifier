use pawnet::model::{self, ModelOptions};
use pawnet::nn::Rng;
use std::time::Instant;

fn main() {
    let labels: Vec<String> = (0..6).map(|i| i.to_string()).collect();
    let mut rng = Rng::new(1);
    let net = model::pawnet_cnn(&labels, &ModelOptions::default(), &mut rng);
    let reps = 50;
    let mut x: Vec<f32> = (0..3072).map(|_| rng.normal()).collect();
    for layer in net.layers() {
        let mut out = vec![0.0; layer.output_len()];
        let mut cache = Vec::new();
        let t = Instant::now();
        for _ in 0..reps {
            layer.forward_train(&x, &mut out, &mut rng, &mut cache);
        }
        let f = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let go: Vec<f32> = (0..out.len()).map(|_| rng.normal()).collect();
        let mut gi = vec![0.0; x.len()];
        let mut gp = vec![0.0; layer.params().len()];
        let t = Instant::now();
        for _ in 0..reps {
            layer.backward(&x, &out, &cache, &go, &mut gi, &mut gp);
        }
        let b = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        println!("{:<44} fwd {f:8.1}us  bwd {b:8.1}us", layer.describe());
        x = out;
    }
}
