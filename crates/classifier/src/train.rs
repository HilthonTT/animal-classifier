use std::io::Write;
use std::thread;
use std::time::{Duration, Instant};

use crate::data::{Augment, Dataset, IMAGE_LEN};
use crate::error::{Error, Result};
use crate::image::preprocess::average;
use crate::metrics::ConfusionMatrix;
use crate::nn::loss::argmax;
use crate::nn::optimizer::clip_gradients;
use crate::nn::{Loss, LrSchedule, Network, Optimizer, Rng};
use crate::predictor::Predictor;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub schedule: LrSchedule,
    pub loss: Loss,
    pub l2: f32,
    pub gradient_clipping: Option<f32>,
    pub threads: usize,
    pub augment: bool,
    pub verbose: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 64,
            learning_rate: 1e-3,
            schedule: LrSchedule::default(),
            loss: Loss::CrossEntropy,
            l2: 0.0,
            gradient_clipping: None,
            threads: default_threads(),
            augment: true,
            verbose: true,
        }
    }
}

impl TrainConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn schedule(mut self, schedule: LrSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    pub fn l2(mut self, lambda: f32) -> Self {
        self.l2 = lambda;
        self
    }

    pub fn gradient_clipping(mut self, max_norm: f32) -> Self {
        self.gradient_clipping = Some(max_norm);
        self
    }

    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn augment(mut self, augment: bool) -> Self {
        self.augment = augment;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

pub fn default_threads() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get())
}

#[derive(Clone, Debug)]
pub struct EpochStats {
    pub epoch: usize,
    pub loss: f32,
    pub accuracy: f32,
    pub learning_rate: f32,
    pub duration: Duration,
}

pub fn train<F>(
    net: &mut Network,
    data: &Dataset,
    optimizer: &mut dyn Optimizer,
    config: &TrainConfig,
    rng: &mut Rng,
    mut on_epoch: F,
) -> Result<()>
where
    F: FnMut(&Network, &EpochStats) -> Result<()>,
{
    if data.is_empty() {
        return Err(Error::format("training set is empty"));
    }
    if net.labels().len() != data.class_names.len() || net.input_len() != IMAGE_LEN {
        return Err(Error::format("the network does not fit this dataset"));
    }
    let batch_size = config.batch_size.max(1);
    let threads = config.threads.max(1);
    let mut order: Vec<usize> = (0..data.len()).collect();

    for epoch in 0..config.epochs {
        let learning_rate = config.schedule.rate(config.learning_rate, epoch);
        optimizer.set_learning_rate(learning_rate);
        rng.shuffle(&mut order);

        let start = Instant::now();
        let (mut total_loss, mut total_correct, mut seen) = (0.0f64, 0usize, 0usize);
        let batches = order.len().div_ceil(batch_size);
        for (b, batch) in order.chunks(batch_size).enumerate() {
            let seeds: Vec<u64> = (0..threads).map(|_| rng.next_u64()).collect();
            let (mut grads, loss, correct) = batch_gradients(net, data, batch, &seeds, config);
            if config.l2 > 0.0 {
                net.add_weight_decay(&mut grads, config.l2);
            }
            if let Some(max_norm) = config.gradient_clipping {
                clip_gradients(&mut grads, max_norm);
            }
            net.apply_gradients(optimizer, &grads);

            total_loss += loss as f64;
            total_correct += correct;
            seen += batch.len();
            if config.verbose && (b % 10 == 0 || b + 1 == batches) {
                let progress = Progress {
                    epoch: epoch + 1,
                    epochs: config.epochs,
                    batch: b + 1,
                    batches,
                };
                progress.print(total_loss / seen as f64, total_correct as f64 / seen as f64);
            }
        }
        if config.verbose {
            eprint!("\r{:80}\r", "");
        }

        let stats = EpochStats {
            epoch: epoch + 1,
            loss: (total_loss / seen as f64) as f32,
            accuracy: total_correct as f32 / seen as f32,
            learning_rate,
            duration: start.elapsed(),
        };
        on_epoch(net, &stats)?;
    }
    Ok(())
}

fn batch_gradients(
    net: &Network,
    data: &Dataset,
    batch: &[usize],
    seeds: &[u64],
    config: &TrainConfig,
) -> (Vec<Vec<f32>>, f32, usize) {
    let chunk = batch.len().div_ceil(seeds.len());
    let partials: Vec<(Vec<Vec<f32>>, f32, usize)> = thread::scope(|scope| {
        let handles: Vec<_> = batch
            .chunks(chunk)
            .zip(seeds)
            .map(|(part, &seed)| {
                scope.spawn(move || {
                    let mut rng = Rng::new(seed);
                    let mut grads = net.zero_grads();
                    let mut input = vec![0.0; IMAGE_LEN];
                    let (mut loss, mut correct) = (0.0, 0);
                    for &i in part {
                        let aug = if config.augment {
                            Augment::random(&mut rng)
                        } else {
                            Augment::NONE
                        };
                        data.sample_into(i, aug, &mut input);
                        let label = data.labels[i] as usize;
                        let (l, predicted) = net.accumulate_gradients(
                            &input,
                            label,
                            config.loss,
                            &mut rng,
                            &mut grads,
                        );
                        loss += l;
                        correct += usize::from(predicted == label);
                    }
                    (grads, loss, correct)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("training thread panicked"))
            .collect()
    });

    let mut partials = partials.into_iter();
    let (mut grads, mut loss, mut correct) = partials.next().expect("batch is not empty");
    for (g, l, c) in partials {
        for (total, part) in grads.iter_mut().zip(&g) {
            for (t, p) in total.iter_mut().zip(part) {
                *t += p;
            }
        }
        loss += l;
        correct += c;
    }
    let scale = 1.0 / batch.len() as f32;
    for g in grads.iter_mut().flat_map(|g| g.iter_mut()) {
        *g *= scale;
    }
    (grads, loss, correct)
}

pub fn evaluate<P: Predictor + ?Sized>(
    model: &P,
    data: &Dataset,
    threads: usize,
    test_time_augmentation: bool,
) -> ConfusionMatrix {
    let classes = model.labels().len();
    let chunk = data.len().div_ceil(threads.max(1)).max(1);
    let partials: Vec<ConfusionMatrix> = thread::scope(|scope| {
        let handles: Vec<_> = (0..data.len())
            .step_by(chunk)
            .map(|start| {
                scope.spawn(move || {
                    let mut local = ConfusionMatrix::new(classes);
                    let mut input = vec![0.0; IMAGE_LEN];
                    let mut mirrored = vec![0.0; IMAGE_LEN];
                    for i in start..(start + chunk).min(data.len()) {
                        data.sample_into(i, Augment::NONE, &mut input);
                        let probabilities = if test_time_augmentation {
                            let flip = Augment {
                                flip: true,
                                ..Augment::NONE
                            };
                            data.sample_into(i, flip, &mut mirrored);
                            average([model.predict(&input), model.predict(&mirrored)].into_iter())
                        } else {
                            model.predict(&input)
                        };
                        local.record(data.labels[i] as usize, argmax(&probabilities));
                    }
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("evaluation thread panicked"))
            .collect()
    });
    let mut matrix = ConfusionMatrix::new(classes);
    for partial in &partials {
        matrix.merge(partial);
    }
    matrix
}

struct Progress {
    epoch: usize,
    epochs: usize,
    batch: usize,
    batches: usize,
}

impl Progress {
    fn print(&self, loss: f64, accuracy: f64) {
        const WIDTH: usize = 30;
        let filled = self.batch * WIDTH / self.batches;
        eprint!(
            "\repoch {:>2}/{} [{}{}] {:>3}%  loss {loss:.4}  acc {:5.1}%",
            self.epoch,
            self.epochs,
            "#".repeat(filled),
            ".".repeat(WIDTH - filled),
            self.batch * 100 / self.batches,
            accuracy * 100.0
        );
        let _ = std::io::stderr().flush();
    }
}
