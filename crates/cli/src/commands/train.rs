use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use pawnet::data::AugmentOptions;
use pawnet::data::cifar::{DatasetKind, Split};
use pawnet::model::{self, ModelOptions};
use pawnet::nn::{Adam, AdamW, Network, Optimizer, RmsProp, Rng, Sgd};
use pawnet::train::{self, EpochStats, TrainConfig};

use super::{AppResult, DEFAULT_MODEL_PATH, dataset, time_seed};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&[
        "data",
        "dataset",
        "model",
        "threads",
        "epochs",
        "batch-size",
        "lr",
        "schedule",
        "warmup",
        "optimizer",
        "weight-decay",
        "loss",
        "l2",
        "clip",
        "activation",
        "init",
        "dropout",
        "width",
        "batch-norm",
        "residual",
        "pool",
        "global-pool",
        "jitter",
        "cutout",
        "mixup",
        "patience",
        "seed",
        "limit",
        "no-augment",
        "resume",
        "val",
        "log",
    ])?;
    let (kind, data_dir) = dataset(opts, DatasetKind::Animals)?;
    let model_path = opts.path("model", DEFAULT_MODEL_PATH);
    let defaults = TrainConfig::default();
    let config = TrainConfig {
        epochs: opts.get("epochs", defaults.epochs)?,
        batch_size: opts.get("batch-size", defaults.batch_size)?,
        learning_rate: opts.get("lr", defaults.learning_rate)?,
        schedule: opts.get("schedule", defaults.schedule)?,
        warmup: opts.get("warmup", defaults.warmup)?,
        loss: opts.get("loss", defaults.loss)?,
        l2: opts.get("l2", defaults.l2)?,
        gradient_clipping: opts.get_opt("clip")?,
        threads: opts.get("threads", defaults.threads)?,
        augment: !opts.flag("no-augment"),
        augmentation: AugmentOptions {
            color_jitter: opts.get("jitter", defaults.augmentation.color_jitter)?,
            cutout: opts.get("cutout", defaults.augmentation.cutout)?,
        },
        mixup: opts.get("mixup", defaults.mixup)?,
        verbose: true,
    };
    let model_defaults = ModelOptions::default();
    let model_options = ModelOptions {
        activation: opts.get("activation", model_defaults.activation)?,
        init: opts.get("init", model_defaults.init)?,
        dropout: opts.get("dropout", model_defaults.dropout)?,
        width: opts.get("width", model_defaults.width)?,
        batch_norm: opts.flag("batch-norm"),
        residual: opts.flag("residual"),
        pool: opts.get("pool", model_defaults.pool)?,
        global_pool: opts.flag("global-pool"),
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
    if !(0.0..=1.0).contains(&config.augmentation.color_jitter) {
        return Err("--jitter must be in [0, 1]".into());
    }
    if !(config.mixup >= 0.0 && config.mixup.is_finite()) {
        return Err("--mixup must be zero or a positive number".into());
    }
    let patience: Option<usize> = opts.get_opt("patience")?;
    if patience == Some(0) {
        return Err("--patience must be at least 1".into());
    }
    let optimizer_name = opts.string("optimizer", "adam");
    let weight_decay: f32 = opts.get("weight-decay", 0.01)?;
    if opts.flag("weight-decay") && optimizer_name != "adamw" {
        return Err(
            "--weight-decay only applies to --optimizer adamw (use --l2 with the others)".into(),
        );
    }
    let val_fraction: f32 = opts.get("val", 0.1)?;
    if !(0.0..1.0).contains(&val_fraction) {
        return Err("--val must be in [0, 1)".into());
    }
    let seed = opts.get("seed", time_seed())?;

    println!("Loading {kind} from {}", data_dir.display());
    let mut train_set = kind.load(&data_dir, Split::Train)?;
    let test_set = kind.load(&data_dir, Split::Test)?;
    if let Some(limit) = opts.get_opt::<usize>("limit")? {
        train_set.truncate(limit);
    }
    let mut rng = Rng::new(seed);
    let val_set = if val_fraction > 0.0 {
        let (train, val) = train_set.split_validation(val_fraction, &mut rng);
        if val.is_empty() {
            return Err("--val is too small: the validation set is empty".into());
        }
        train_set = train;
        Some(val)
    } else {
        None
    };
    println!(
        "  {} training images, {} validation images, {} test images",
        train_set.len(),
        val_set.as_ref().map_or(0, |v| v.len()),
        test_set.len()
    );
    if train_set.class_names.len() <= 10 {
        println!("  classes: {}", train_set.class_names.join(", "));
    } else {
        println!("  {} classes", train_set.class_names.len());
    }
    match &val_set {
        Some(_) => println!("  the best epoch is chosen by validation accuracy\n"),
        None => println!("  --val 0: the best epoch is chosen by test accuracy (optimistic)\n"),
    }
    let selection_set = val_set.as_ref().unwrap_or(&test_set);
    let selection_name = if val_set.is_some() {
        "validation"
    } else {
        "test"
    };
    let mut log = match opts.get_opt::<PathBuf>("log")? {
        Some(path) => Some(TrainingLog::create(&path)?),
        None => None,
    };

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

    let mut optimizer: Box<dyn Optimizer> = match optimizer_name.as_str() {
        "adam" => Box::new(Adam::new(config.learning_rate)),
        "adamw" => Box::new(AdamW::new(config.learning_rate, weight_decay)),
        "sgd" => Box::new(Sgd::new(config.learning_rate, 0.9)),
        "rmsprop" => Box::new(RmsProp::new(config.learning_rate)),
        other => {
            return Err(format!(
                "unknown optimizer `{other}` (expected adam, adamw, sgd or rmsprop)"
            )
            .into());
        }
    };
    let clip = config
        .gradient_clipping
        .map_or("off".to_string(), |c| c.to_string());
    let warmup = if config.warmup > 0 {
        format!(", {} warm-up epochs", config.warmup)
    } else {
        String::new()
    };
    println!(
        "\nTraining for {} epochs: {optimizer_name}, batch {}, lr {} ({}{warmup}), loss {}, l2 {}, clip {clip}, {} threads, seed {seed}",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.schedule,
        config.loss,
        config.l2,
        config.threads
    );
    let mut extras = Vec::new();
    if config.augmentation.color_jitter > 0.0 {
        extras.push(format!(
            "colour jitter {}",
            config.augmentation.color_jitter
        ));
    }
    if config.augmentation.cutout > 0 {
        extras.push(format!("cutout {}px", config.augmentation.cutout));
    }
    if config.mixup > 0.0 {
        extras.push(format!("mixup alpha {}", config.mixup));
    }
    if let Some(patience) = patience {
        extras.push(format!(
            "early stopping after {patience} epochs without improvement"
        ));
    }
    if !extras.is_empty() {
        println!("  {}", extras.join(", "));
    }
    println!();

    let mut best = if opts.flag("resume") {
        let accuracy = train::evaluate(&net, selection_set, config.threads, false).accuracy();
        println!(
            "Resumed model {selection_name} accuracy {:.1}%\n",
            accuracy * 100.0
        );
        accuracy
    } else {
        f32::NEG_INFINITY
    };
    let mut improved = false;
    let mut since_best = 0;
    train::train(
        &mut net,
        &train_set,
        optimizer.as_mut(),
        &config,
        &mut rng,
        |net, stats| {
            let test_accuracy = train::evaluate(net, &test_set, config.threads, false).accuracy();
            let val_accuracy = val_set
                .as_ref()
                .map(|val| train::evaluate(net, val, config.threads, false).accuracy());
            let score = val_accuracy.unwrap_or(test_accuracy);
            let saved = if score > best {
                best = score;
                improved = true;
                since_best = 0;
                net.save(&model_path)?;
                "  [saved]"
            } else {
                since_best += 1;
                ""
            };
            let val = val_accuracy.map_or(String::new(), |v| format!("  val {:5.1}%", v * 100.0));
            println!(
                "epoch {:>2}/{}  loss {:.4}  train {:5.1}%{val}  test {:5.1}%  lr {:.1e}  {:>5.1}s{saved}",
                stats.epoch,
                config.epochs,
                stats.loss,
                stats.accuracy * 100.0,
                test_accuracy * 100.0,
                stats.learning_rate,
                stats.duration.as_secs_f32()
            );
            if let Some(log) = log.as_mut() {
                log.record(stats, val_accuracy, test_accuracy, !saved.is_empty())?;
            }
            if patience.is_some_and(|p| since_best >= p) {
                println!(
                    "\nNo {selection_name} improvement for {since_best} epochs; stopping early"
                );
                return Ok(ControlFlow::Break(()));
            }
            Ok(ControlFlow::Continue(()))
        },
    )?;

    if improved {
        println!(
            "\nBest {selection_name} accuracy {:.1}% — model saved to {}",
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

struct TrainingLog {
    out: BufWriter<File>,
}

impl TrainingLog {
    fn create(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let mut out = BufWriter::new(File::create(path)?);
        writeln!(
            out,
            "epoch,loss,train_accuracy,val_accuracy,test_accuracy,learning_rate,seconds,saved"
        )?;
        out.flush()?;
        Ok(Self { out })
    }

    fn record(
        &mut self,
        stats: &EpochStats,
        val_accuracy: Option<f32>,
        test_accuracy: f32,
        saved: bool,
    ) -> std::io::Result<()> {
        writeln!(
            self.out,
            "{},{:.6},{:.6},{},{:.6},{:.4e},{:.3},{}",
            stats.epoch,
            stats.loss,
            stats.accuracy,
            val_accuracy.map_or(String::new(), |v| format!("{v:.6}")),
            test_accuracy,
            stats.learning_rate,
            stats.duration.as_secs_f32(),
            saved
        )?;
        self.out.flush()
    }
}
