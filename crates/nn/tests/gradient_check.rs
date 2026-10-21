use pawnet_nn::{Activation, ActivationFn, Conv2d, Dense, Init, Loss, MaxPool2d, Network, Rng};

fn build(func: ActivationFn, rng: &mut Rng) -> Network {
    let mut net = Network::new((0..4).map(|i| i.to_string()).collect());
    net.push(Conv2d::with_init(2, 6, 6, 3, 3, 1, Init::Xavier, rng))
        .push(Activation::new(func, 3 * 6 * 6))
        .push(MaxPool2d::new(3, 6, 6, 2))
        .push(Conv2d::with_init(3, 3, 3, 2, 2, 0, Init::Xavier, rng))
        .push(Activation::new(func, 2 * 2 * 2))
        .push(Dense::with_init(8, 4, Init::Xavier, rng));
    net
}

fn param(net: &mut Network, layer: usize, index: usize) -> &mut f32 {
    &mut net.layers_mut()[layer].params_mut()[index]
}

fn check(mut net: Network, loss: Loss, rng: &mut Rng) {
    let input: Vec<f32> = (0..net.input_len()).map(|_| rng.normal()).collect();
    let target = 2;
    let loss_at = |net: &Network| loss.evaluate(&net.forward(&input), target).0;

    let mut grads = net.zero_grads();
    net.accumulate_gradients(&input, target, loss, rng, &mut grads);

    let eps = 1e-3;
    for (l, layer_grads) in grads.iter().enumerate() {
        for (p, &analytic) in layer_grads.iter().enumerate() {
            let original = *param(&mut net, l, p);
            *param(&mut net, l, p) = original + eps;
            let plus = loss_at(&net);
            *param(&mut net, l, p) = original - eps;
            let minus = loss_at(&net);
            *param(&mut net, l, p) = original;

            let numerical = (plus - minus) / (2.0 * eps);
            let tolerance = 1e-2 * (0.1 + numerical.abs().max(analytic.abs()));
            assert!(
                (numerical - analytic).abs() <= tolerance,
                "{} {loss}, layer {l} param {p}: analytic {analytic} vs numerical {numerical}",
                net.layers()[1].describe()
            );
        }
    }
}

#[test]
fn gradients_are_correct_for_every_activation() {
    let mut rng = Rng::new(42);
    for func in ActivationFn::ALL {
        let net = build(func, &mut rng);
        check(net, Loss::CrossEntropy, &mut rng);
    }
}

#[test]
fn gradients_are_correct_for_mean_squared_error() {
    let mut rng = Rng::new(7);
    let net = build(ActivationFn::Tanh, &mut rng);
    check(net, Loss::MeanSquaredError, &mut rng);
}

#[test]
fn layers_report_parameter_counts() {
    let net = build(ActivationFn::Relu, &mut Rng::new(1));
    let counts: Vec<usize> = net.layers().iter().map(|l| l.params().len()).collect();
    assert_eq!(
        counts,
        vec![2 * 3 * 9 + 3, 0, 0, 3 * 2 * 4 + 2, 0, 8 * 4 + 4]
    );
}
