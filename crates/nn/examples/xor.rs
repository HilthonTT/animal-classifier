use pawnet_nn::{Activation, ActivationFn, Adam, Dense, Loss, Network, Rng};

fn main() {
    let samples = [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 0),
    ];

    let mut rng = Rng::new(1);
    let mut net = Network::new(vec!["false".into(), "true".into()]);
    net.push(Dense::new(2, 8, &mut rng))
        .push(Activation::new(ActivationFn::Tanh, 8))
        .push(Dense::new(8, 2, &mut rng));
    let mut optimizer = Adam::new(0.05);

    for epoch in 0..=300 {
        let mut grads = net.zero_grads();
        let mut total = 0.0;
        for (input, label) in &samples {
            total += net
                .accumulate_gradients(input, *label, Loss::CrossEntropy, &mut rng, &mut grads)
                .0;
        }
        net.apply_gradients(&mut optimizer, &grads);
        if epoch % 100 == 0 {
            println!("epoch {epoch:>3}  loss {:.4}", total / samples.len() as f32);
        }
    }

    for (input, label) in &samples {
        let p = net.predict(input);
        println!("{input:?} -> {:.3} (expected {label})", p[1]);
    }
}
