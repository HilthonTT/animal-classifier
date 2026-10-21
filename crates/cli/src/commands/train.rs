use pawnet::data::cifar::{self, Split};
use pawnet::model::{self, ModelOptions};
use pawnet::nn::{Adam, Network, Optimizer, Rng, Sgd};
use pawnet::train::{self, TrainConfig};

use super::{AppResult, DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, time_seed};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&[
        "data",
        "model",
        "threads",
        "epochs",
        "batch-size",
        "lr",
        "schedule",
        "optimizer",
        "loss",
        "l2",
        "clip",
        "activation",
        "init",
        "dropout",
        "width",
        "seed",
        "limit",
        "no-augment",
        "resume",
    ])?;
    let data_dir = opts.path("data", DEFAULT_DATA_DIR);
    let model_path = opts.path("model", DEFAULT_MODEL_PATH);
    let defaults = TrainConfig::default();
    let config = TrainConfig {
        epochs: opts.get("epochs", defaults.epochs)?,
        batch_size: opts.get("batch-size", defaults.batch_size)?,
        learning_rate: opts.get("lr", defaults.learning_rate)?,
        schedule: opts.get("schedule", defaults.schedule)?,
        loss: opts.get("loss", defaults.loss)?,
        l2: opts.get("l2", defaults.l2)?,
        gradient_clipping: opts.get_opt("clip")?,
        threads: opts.get("threads", defaults.threads)?,
        augment: !opts.flag("no-augment"),
        verbose: true,
    };
    let model_defaults = ModelOptions::default();
    let model_options = ModelOptions {
        activation: opts.get("activation", model_defaults.activation)?,
        init: opts.get("init", model_defaults.init)?,
        dropout: opts.get("dropout", model_defaults.dropout)?,
        width: opts.get("width", model_defaults.width)?,
    };
    if !(0.0..1.0).contains(&model_options.dropout) || model_options.width == 0 {
        return Err("--dropout must be in [0, 1) and --width at least 1".into());
    }
    if config
        .gradient_clipping
        .is_some_and(|c| !(c.is_finite() && c > 0.0))
    {
        return Err("--clip must be a positive number".into());
    }
    let seed = opts.get("seed", time_seed())?;

    println!("Loading CIFAR-10 animals from {}", data_dir.display());
    let mut train_set = cifar::load_animals(&data_dir, Split::Train)?;
    let test_set = cifar::load_animals(&data_dir, Split::Test)?;
    if let Some(limit) = opts.get_opt::<usize>("limit")? {
        train_set.truncate(limit);
    }
    println!(
        "  {} training images, {} test images",
        train_set.len(),
        test_set.len()
    );
    println!("  classes: {}\n", train_set.class_names.join(", "));

    let mut rng = Rng::new(seed);
    let mut net = if opts.flag("resume") {
        println!("Resuming from {}", model_path.display());
        Network::load(&model_path)?
    } else {
        model::pawnet_cnn(&train_set.class_names, &model_options, &mut rng)
    };
    if net.labels() != train_set.class_names.as_slice() {
        return Err("the model was trained on different classes than this dataset".into());
    }
    print!("{}", net.summary());

    let mut optimizer: Box<dyn Optimizer> = match opts.string("optimizer", "adam").as_str() {
        "adam" => Box::new(Adam::new(config.learning_rate)),
        "sgd" => Box::new(Sgd::new(config.learning_rate, 0.9)),
        other => return Err(format!("unknown optimizer `{other}` (expected adam or sgd)").into()),
    };
    let clip = config
        .gradient_clipping
        .map_or("off".to_string(), |c| c.to_string());
    println!(
        "\nTraining for {} epochs: batch {}, lr {} ({}), loss {}, l2 {}, clip {clip}, {} threads, seed {seed}\n",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.schedule,
        config.loss,
        config.l2,
        config.threads
    );

    let mut best = if opts.flag("resume") {
        let accuracy = train::evaluate(&net, &test_set, config.threads, false).accuracy();
        println!("Resumed model test accuracy {:.1}%\n", accuracy * 100.0);
        accuracy
    } else {
        f32::NEG_INFINITY
    };
    let mut improved = false;
    train::train(
        &mut net,
        &train_set,
        optimizer.as_mut(),
        &config,
        &mut rng,
        |net, stats| {
            let test_accuracy = train::evaluate(net, &test_set, config.threads, false).accuracy();
            let saved = if test_accuracy > best {
                best = test_accuracy;
                improved = true;
                net.save(&model_path)?;
                "  [saved]"
            } else {
                ""
            };
            println!(
                "epoch {:>2}/{}  loss {:.4}  train {:5.1}%  test {:5.1}%  lr {:.1e}  {:>5.1}s{saved}",
                stats.epoch,
                config.epochs,
                stats.loss,
                stats.accuracy * 100.0,
                test_accuracy * 100.0,
                stats.learning_rate,
                stats.duration.as_secs_f32()
            );
            Ok(())
        },
    )?;

    if improved {
        println!(
            "\nBest test accuracy {:.1}% — model saved to {}",
            best * 100.0,
            model_path.display()
        );
    } else if opts.flag("resume") {
        println!(
            "\nNo epoch beat the resumed model; {} left unchanged",
            model_path.display()
        );
    }
    Ok(())
}
