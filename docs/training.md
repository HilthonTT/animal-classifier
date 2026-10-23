# Training guide

Everything here assumes the dataset has been downloaded with `scripts/download_data.sh`.

## The defaults

```bash
cargo run --release -- train
```

This trains for 20 epochs: Adam, learning rate 0.001 decaying by 10% per epoch, batch size 64,
cross-entropy loss, ReLU, He initialisation, 25% dropout, and random flips and shifts.

10% of the training images are held out as a **validation set**. After every epoch the model
is evaluated on the validation set and on the 6 000 test images. The epoch with the best
*validation* accuracy is saved to `models/pawnet.bin`, so the test accuracy stays an honest
estimate. Saving writes a temporary file and renames it, so an interrupted run never leaves a
corrupt model behind.

## Every option

| Option | What it does | Default |
|--------|--------------|---------|
| `--epochs N` | passes over the training set | 20 |
| `--batch-size N` | images per weight update | 64 |
| `--lr RATE` | learning rate of the first epoch | 0.001 |
| `--schedule SPEC` | `constant`, `exp:<decay>`, `step:<factor>:<every>`, `time:<decay>` or `cosine[:<min>]` (cosine annealing down to `min` × the initial rate) | `exp:0.9` |
| `--warmup N` | ramp the learning rate up linearly over the first N epochs | 0 |
| `--optimizer NAME` | `adam`, `adamw`, `sgd` (with 0.9 momentum) or `rmsprop` | `adam` |
| `--weight-decay W` | AdamW's decoupled weight decay | 0.01 |
| `--loss NAME` | `cross-entropy` or `mse` | `cross-entropy` |
| `--l2 LAMBDA` | L2 regularisation / weight decay (try `1e-4`) | 0 |
| `--clip NORM` | gradient clipping by global norm (try `5`) | off |
| `--activation NAME` | `relu`, `leaky-relu`, `elu`, `sigmoid`, `tanh`, `softplus` | `relu` |
| `--init NAME` | `he`, `xavier`, `uniform`, `zero` | `he` |
| `--dropout RATE` | dropout before the output layer (0 disables it) | 0.25 |
| `--width N` | filters in the first convolution; later blocks use 2× and 4× | 16 |
| `--batch-norm` | batch normalisation after every convolution and the hidden layer | off |
| `--residual` | add a residual block (two convolutions and a shortcut) to each stage | off |
| `--pool NAME` | `max` or `avg` pooling between stages, or `stride` for a learned stride-2 convolution | `max` |
| `--global-pool` | global average pooling instead of the 128-unit hidden layer | off |
| `--jitter S` | random brightness, contrast and saturation, each scaled by a factor in `[1 − S, 1 + S]` | 0 |
| `--cutout N` | blank a random N×N square of every training image | 0 |
| `--mixup A` | blend each image with a random other one, and their labels, with a weight drawn from Beta(A, A) | 0 |
| `--no-augment` | turn off all of the random augmentation above, and the flips and shifts | on |
| `--patience N` | stop after N epochs without a new best model | off |
| `--dataset NAME` | `animals` (6 classes), `cifar10` (all 10 classes) or `cifar100` | `animals` |
| `--limit N` | train on only the first N images | all |
| `--seed N` | make a run reproducible | random |
| `--resume` | continue training the model at `--model` | off |
| `--val FRACTION` | share of the training images held out to choose the best epoch; `0` chooses by test accuracy | 0.1 |
| `--log FILE` | write one CSV row per epoch (see below) | off |
| `--threads N` | CPU threads | all cores |

The same options are available from Rust through the builder:

```rust
use pawnet::nn::{Loss, LrSchedule};
use pawnet::train::TrainConfig;

let config = TrainConfig::new()
    .epochs(30)
    .schedule(LrSchedule::Cosine { min: 0.01 })
    .warmup(2)
    .l2(1e-4)
    .gradient_clipping(5.0)
    .cutout(8)
    .mixup(0.2)
    .loss(Loss::CrossEntropy);
```

`train::train` calls your closure after every epoch; return `ControlFlow::Break(())` from it
to stop early.

## Training logs

`--log runs/baseline.csv` writes one row per epoch, flushed as training goes:

```csv
epoch,loss,train_accuracy,val_accuracy,test_accuracy,learning_rate,seconds,saved
1,1.971198,0.234815,0.313333,0.316667,1.0000e-3,4.344,true
2,1.632769,0.336667,0.436667,0.389667,9.0000e-4,5.810,true
```

Accuracies are fractions between 0 and 1. `val_accuracy` is empty with `--val 0`, and `saved`
says whether that epoch became the new best model. Open the file in any spreadsheet to plot
and compare runs.

## Recipes

### More data: offline expansion

```bash
cargo run --release -- expand                       # 30 000 → 120 000 images
cargo run --release -- train --data data/cifar-10-expanded --epochs 15
```

`expand` adds a mirrored, a shifted (±2 px) and a rotated (±12°) copy of every image, writes
them as CIFAR-10 files and copies the test set alongside. Each epoch takes about four times as
long, but the network sees much more variety. Use `--shift`, `--rotate` and `--no-flip` to
change the mix.

On expanded data the validation split can contain copies of training images, so validation
accuracy reads high. It still ranks the epochs, but trust the test accuracy for the final number.

### An ensemble

Train a few models with different seeds, then point `--model` at the folder:

```bash
for seed in 1 2 3; do
  cargo run --release -- train --seed $seed --model models/ensemble/pawnet-seed$seed.bin
done
cargo run --release -- evaluate --model models/ensemble --tta
```

To fit three runs on one machine at once, give each `--threads 2`.

### A bigger network

```bash
cargo run --release -- train --width 32 --epochs 30 --l2 1e-4
```

Doubling `--width` makes each epoch roughly 3× slower, but usually gains a few points.

### Experiments

```bash
cargo run --release -- train --activation tanh --init xavier --model /tmp/tanh.bin
cargo run --release -- train --loss mse --model /tmp/mse.bin
cargo run --release -- train --optimizer sgd --lr 0.01 --schedule step:0.5:5 --model /tmp/sgd.bin
```

Use `--limit 3000 --epochs 3` for a fast first look before committing to a full run.

## Reference results

All runs on an 8-thread CPU, 6 000-image test set. "TTA" means evaluated with `--tta`.

| Model | Data | Epochs | Accuracy | With `--tta` |
|-------|------|-------:|---------:|-------------:|
| `models/pawnet.bin` (seed 4) | expanded ×4 | 15 | 78.4% | **81.0%** |
| `models/ensemble` (3 models, seeds 1–3) | standard | 20 each | 76.2% | 77.1% |
| `models/ensemble/pawnet-seed1.bin` | standard | 20 | 73.3% | |
| `models/ensemble/pawnet-seed2.bin` | standard | 20 | 72.8% | |
| `models/ensemble/pawnet-seed3.bin` | standard | 20 | 74.1% | |
| All four models together | | | 79.0% | 79.4% |
| v0.1 model (no dropout) | standard | 20 | 73.7% | 75.2% |

Lessons from these runs:

- **More data helps most.** Training on the expanded set gave the biggest single gain (+5 points).
- **Ensembles help when the members are equally strong.** Three similar models beat each of
  their members by 2–3 points. Adding the three weaker models to the stronger one lowered its
  accuracy (81.0% → 79.4%).
- **Test-time augmentation is cheap** (2× slower inference) and gave +1 to +2.6 points every time.

Training time: about 27 s per epoch on the standard set and about 2 minutes per epoch on the
expanded set (8 threads).

> Note: these models were trained before `--val` existed and kept the epoch with the best
> *test* accuracy, so their figures are slightly optimistic. New runs choose the epoch on the
> validation split instead.
