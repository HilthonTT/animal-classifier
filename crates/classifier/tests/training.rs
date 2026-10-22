use std::ops::ControlFlow;

use pawnet::Ensemble;
use pawnet::data::cifar::{self, Split};
use pawnet::data::expand::{ExpandOptions, expand};
use pawnet::data::{Dataset, IMAGE_LEN, IMAGE_SIZE};
use pawnet::model::{self, ModelOptions, Pool};
use pawnet::nn::{
    Activation, Adam, AdamW, Dense, Dropout, Loss, LrSchedule, Network, Optimizer, RmsProp, Rng,
};
use pawnet::train::{self, TrainConfig};

fn colour_dataset(n: usize, rng: &mut Rng) -> Dataset {
    let mut ds = Dataset::new(vec!["red".into(), "green".into(), "blue".into()]);
    let plane = IMAGE_SIZE * IMAGE_SIZE;
    for i in 0..n {
        let label = i % 3;
        let image: Vec<u8> = (0..IMAGE_LEN)
            .map(|j| {
                let base = if j / plane == label { 150.0 } else { 90.0 };
                (base + rng.normal() * 40.0).clamp(0.0, 255.0) as u8
            })
            .collect();
        ds.push(&image, label as u8);
    }
    ds
}

fn small_net(classes: Vec<String>, dropout: f32, rng: &mut Rng) -> Network {
    let mut net = Network::new(classes);
    net.push(Dense::new(IMAGE_LEN, 16, rng))
        .push(Activation::relu(16));
    if dropout > 0.0 {
        net.push(Dropout::new(16, dropout));
    }
    net.push(Dense::new(16, 3, rng));
    net
}

fn train_and_score(net: &mut Network, config: &TrainConfig, seed: u64) -> f32 {
    let mut optimizer = Adam::new(config.learning_rate);
    train_with(net, config, &mut optimizer, seed)
}

fn train_with(
    net: &mut Network,
    config: &TrainConfig,
    optimizer: &mut dyn Optimizer,
    seed: u64,
) -> f32 {
    let mut rng = Rng::new(seed);
    let train_set = colour_dataset(300, &mut rng);
    let test_set = colour_dataset(90, &mut rng);
    let mut epochs_seen = 0;
    train::train(net, &train_set, optimizer, config, &mut rng, |_, stats| {
        epochs_seen = stats.epoch;
        assert!(stats.loss.is_finite());
        Ok(ControlFlow::Continue(()))
    })
    .unwrap();
    assert_eq!(epochs_seen, config.epochs);
    train::evaluate(net, &test_set, 2, false).accuracy()
}

fn quiet() -> TrainConfig {
    TrainConfig::new()
        .epochs(3)
        .batch_size(16)
        .threads(2)
        .verbose(false)
}

#[test]
fn learns_a_simple_task() {
    let mut rng = Rng::new(1);
    let mut net = small_net(colour_dataset(1, &mut rng).class_names, 0.0, &mut rng);
    let accuracy = train_and_score(&mut net, &quiet(), 1);
    assert!(accuracy > 0.9, "accuracy {accuracy}");
}

#[test]
fn learns_with_every_regularisation_option() {
    let mut rng = Rng::new(2);
    let mut net = small_net(colour_dataset(1, &mut rng).class_names, 0.3, &mut rng);
    let config = quiet()
        .epochs(4)
        .loss(Loss::MeanSquaredError)
        .l2(1e-4)
        .gradient_clipping(1.0)
        .schedule(LrSchedule::Step {
            factor: 0.5,
            every: 2,
        })
        .learning_rate(3e-3);
    let accuracy = train_and_score(&mut net, &config, 2);
    assert!(accuracy > 0.9, "accuracy {accuracy}");
}

#[test]
fn learns_with_every_augmentation_and_schedule_option() {
    let mut rng = Rng::new(6);
    let mut net = small_net(colour_dataset(1, &mut rng).class_names, 0.0, &mut rng);
    let config = quiet()
        .epochs(5)
        .warmup(1)
        .schedule(LrSchedule::Cosine { min: 0.1 })
        .color_jitter(0.2)
        .cutout(8)
        .mixup(0.2)
        .learning_rate(3e-3);
    let mut optimizer = AdamW::new(config.learning_rate, 1e-2);
    let accuracy = train_with(&mut net, &config, &mut optimizer, 6);
    assert!(accuracy > 0.9, "accuracy {accuracy}");
}

#[test]
fn trains_a_residual_batch_norm_network() {
    let mut rng = Rng::new(7);
    let options = ModelOptions {
        width: 4,
        batch_norm: true,
        residual: true,
        pool: Pool::Avg,
        global_pool: true,
        dropout: 0.0,
        ..ModelOptions::default()
    };
    let mut net = model::pawnet_cnn(&colour_dataset(1, &mut rng).class_names, &options, &mut rng);
    let before = net.buffers();
    let config = quiet().epochs(2).learning_rate(3e-3);
    let mut optimizer = RmsProp::new(config.learning_rate);
    let accuracy = train_with(&mut net, &config, &mut optimizer, 7);
    assert!(accuracy > 0.9, "accuracy {accuracy}");
    assert_ne!(
        net.buffers(),
        before,
        "running statistics should be updated"
    );
}

#[test]
fn training_stops_when_the_callback_breaks() {
    let mut rng = Rng::new(8);
    let data = colour_dataset(30, &mut rng);
    let mut net = small_net(data.class_names.clone(), 0.0, &mut rng);
    let mut epochs = 0;
    train::train(
        &mut net,
        &data,
        &mut Adam::new(1e-3),
        &quiet().epochs(10),
        &mut rng,
        |_, stats| {
            epochs = stats.epoch;
            Ok(if stats.epoch == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            })
        },
    )
    .unwrap();
    assert_eq!(epochs, 2);
}

#[test]
fn ensemble_evaluates_like_its_members() {
    let mut rng = Rng::new(3);
    let data = colour_dataset(60, &mut rng);
    let members: Vec<Network> = (0..3)
        .map(|_| {
            let mut net = small_net(data.class_names.clone(), 0.0, &mut rng);
            train_and_score(&mut net, &quiet(), rng.next_u64());
            net
        })
        .collect();
    let best_member = members
        .iter()
        .map(|m| train::evaluate(m, &data, 2, false).accuracy())
        .fold(0.0, f32::max);
    let ensemble = Ensemble::new(members).unwrap();
    let accuracy = train::evaluate(&ensemble, &data, 2, true).accuracy();
    assert!(
        accuracy >= best_member - 0.05,
        "ensemble {accuracy} vs best member {best_member}"
    );
}

#[test]
fn expanded_dataset_survives_save_and_load() {
    let mut rng = Rng::new(4);
    let mut original = cifar::empty_animal_dataset();
    for i in 0..12u8 {
        original.push(&[i * 20; IMAGE_LEN], i % 6);
    }
    let expanded = expand(&original, &ExpandOptions::default(), &mut rng);
    assert_eq!(expanded.len(), 48);

    let dir = std::env::temp_dir().join(format!("expand-test-{}", std::process::id()));
    let written = cifar::save_animals(&expanded, &dir).unwrap();
    let loaded = cifar::load_animals(&dir, Split::Train);
    std::fs::remove_dir_all(&dir).unwrap();

    let loaded = loaded.unwrap();
    assert_eq!(written.len(), 1);
    assert_eq!(loaded.labels, expanded.labels);
    assert_eq!(loaded.images, expanded.images);
}

#[test]
fn trained_model_survives_save_and_load() {
    let mut rng = Rng::new(5);
    let data = colour_dataset(30, &mut rng);
    let net = small_net(data.class_names.clone(), 0.5, &mut rng);

    let path = std::env::temp_dir().join(format!("model-test-{}.bin", std::process::id()));
    net.save(&path).unwrap();
    let loaded = Network::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(
        train::evaluate(&net, &data, 1, false),
        train::evaluate(&loaded, &data, 1, false)
    );
}
