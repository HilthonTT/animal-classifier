use pawnet_nn::{
    Activation, ActivationFn, AvgPool2d, BatchContext, BatchNorm, Conv2d, Dense, GlobalAvgPool,
    Init, Loss, MaxPool2d, Network, NoReduce, Reducer, Residual, Rng, SharedReduce, Target,
};

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
fn gradients_are_correct_for_strided_convolutions() {
    let mut rng = Rng::new(11);
    for (stride, kernel, pad) in [(2, 3, 1), (3, 2, 0)] {
        let mut net = Network::new((0..4).map(|i| i.to_string()).collect());
        net.push(Conv2d::with_init(2, 7, 7, 3, 3, 1, Init::Xavier, &mut rng));
        net.push(Activation::new(ActivationFn::Tanh, net.output_len()));
        net.push(
            Conv2d::with_init(3, 7, 7, 4, kernel, pad, Init::Xavier, &mut rng).with_stride(stride),
        );
        net.push(Activation::new(ActivationFn::Tanh, net.output_len()));
        net.push(Dense::with_init(
            net.output_len(),
            4,
            Init::Xavier,
            &mut rng,
        ));
        check(net, Loss::CrossEntropy, &mut rng);
    }
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

fn modern(rng: &mut Rng) -> Network {
    let mut net = Network::new((0..4).map(|i| i.to_string()).collect());
    net.push(Conv2d::with_init(2, 6, 6, 3, 3, 1, Init::Xavier, rng))
        .push(BatchNorm::new(3, 36))
        .push(Activation::new(ActivationFn::Tanh, 108))
        .push(Conv2d::with_init(3, 6, 6, 3, 3, 1, Init::Xavier, rng))
        .push(BatchNorm::new(3, 36))
        .push(Residual::new(108, 4))
        .push(AvgPool2d::new(3, 6, 6, 2))
        .push(Conv2d::with_init(3, 3, 3, 4, 3, 1, Init::Xavier, rng))
        .push(GlobalAvgPool::new(4, 3, 3))
        .push(BatchNorm::new(4, 1))
        .push(Dense::with_init(4, 4, Init::Xavier, rng));
    for layer in net.layers_mut() {
        let n = layer.params().len();
        if layer.weight_len() == 0 && n > 0 {
            for (i, p) in layer.params_mut().iter_mut().enumerate() {
                *p += 0.1 * (i as f32 - n as f32 / 2.0) / n as f32;
            }
        }
    }
    net
}

fn batch(
    net: &Network,
    inputs: &[Vec<f32>],
    targets: &[Target],
    reducer: &dyn Reducer,
) -> (f32, Vec<Vec<f32>>) {
    let mut grads = net.zero_grads();
    let mut buffers = net.buffers();
    let mut rng = Rng::new(0);
    let mut ctx = BatchContext {
        rng: &mut rng,
        reducer,
    };
    let (loss, _) = net.accumulate_batch(
        inputs,
        targets,
        Loss::CrossEntropy,
        &mut ctx,
        &mut grads,
        &mut buffers,
    );
    (loss, grads)
}

fn sample_batch(net: &Network, rng: &mut Rng) -> (Vec<Vec<f32>>, Vec<Target>) {
    let inputs = (0..4)
        .map(|_| (0..net.input_len()).map(|_| rng.normal()).collect())
        .collect();
    let targets = vec![
        Target::class(0),
        Target::class(2),
        Target::mix(1, 3, 0.7),
        Target::class(3),
    ];
    (inputs, targets)
}

#[test]
fn batch_gradients_are_correct_for_batch_norm_pooling_and_shortcuts() {
    let mut rng = Rng::new(11);
    let mut net = modern(&mut rng);
    let (inputs, targets) = sample_batch(&net, &mut rng);
    let (_, grads) = batch(&net, &inputs, &targets, &NoReduce);

    let eps = 1e-2;
    for (l, layer_grads) in grads.iter().enumerate() {
        for (p, &analytic) in layer_grads.iter().enumerate() {
            let original = *param(&mut net, l, p);
            *param(&mut net, l, p) = original + eps;
            let plus = batch(&net, &inputs, &targets, &NoReduce).0;
            *param(&mut net, l, p) = original - eps;
            let minus = batch(&net, &inputs, &targets, &NoReduce).0;
            *param(&mut net, l, p) = original;

            let numerical = (plus - minus) / (2.0 * eps);
            let tolerance = 2e-2 * (0.1 + numerical.abs().max(analytic.abs()));
            assert!(
                (numerical - analytic).abs() <= tolerance,
                "{}, param {p}: analytic {analytic} vs numerical {numerical}",
                net.layers()[l].describe()
            );
        }
    }
}

#[test]
fn splitting_a_batch_across_threads_gives_the_same_gradients() {
    let mut rng = Rng::new(12);
    let net = modern(&mut rng);
    let (inputs, targets) = sample_batch(&net, &mut rng);
    let (whole_loss, whole) = batch(&net, &inputs, &targets, &NoReduce);

    let shared = SharedReduce::new(2);
    let parts: Vec<(f32, Vec<Vec<f32>>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|t| {
                let reducer = shared.participant(t);
                let (net, inputs, targets) =
                    (&net, &inputs[t * 2..t * 2 + 2], &targets[t * 2..t * 2 + 2]);
                scope.spawn(move || batch(net, inputs, targets, &reducer))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let split_loss = parts[0].0 + parts[1].0;
    assert!((split_loss - whole_loss).abs() < 1e-4 * whole_loss.abs().max(1.0));
    for (l, layer) in whole.iter().enumerate() {
        for (p, &w) in layer.iter().enumerate() {
            let s = parts[0].1[l][p] + parts[1].1[l][p];
            assert!(
                (s - w).abs() < 1e-3 * (0.1 + w.abs()),
                "layer {l} param {p}: {s} vs {w}"
            );
        }
    }
}

#[test]
fn input_gradient_flows_through_shortcuts_at_inference() {
    let mut rng = Rng::new(13);
    let mut net = modern(&mut rng);
    for layer in net.layers_mut() {
        let n = layer.buffers().len() / 2;
        for (i, b) in layer.buffers_mut().iter_mut().enumerate() {
            *b = if i < n { 0.1 } else { 1.5 };
        }
    }
    let input: Vec<f32> = (0..net.input_len()).map(|_| rng.normal()).collect();
    let seed = [0.5, -1.0, 2.0, 0.3];
    let score = |x: &[f32]| -> f32 { net.forward(x).iter().zip(&seed).map(|(o, s)| o * s).sum() };
    let analytic = net.input_gradient(&input, &seed);
    let eps = 1e-2;
    for i in 0..input.len() {
        let (mut plus, mut minus) = (input.clone(), input.clone());
        plus[i] += eps;
        minus[i] -= eps;
        let numerical = (score(&plus) - score(&minus)) / (2.0 * eps);
        assert!(
            (numerical - analytic[i]).abs() < 2e-2 * (0.1 + numerical.abs()),
            "input {i}: analytic {} vs numerical {numerical}",
            analytic[i]
        );
    }
}
