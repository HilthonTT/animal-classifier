# Training guide

Everything here assumes the dataset has been downloaded with `scripts/download_data.sh`.

## The defaults

```bash
cargo run --release -- train
```

This trains for 20 epochs: Adam, learning rate 0.001 decaying by 10% per epoch, batch size 64,
cross-entropy loss, ReLU, He initialisation, 25% dropout, and random flips and shifts. After
every epoch the model is evaluated on the 6 000 test images. The best epoch is saved to
`models/pawnet.bin`.

## Every option

| Option | What it does | Default |
|--------|--------------|---------|
| `--epochs N` | passes over the training set | 20 |
| `--batch-size N` | images per weight update | 64 |
| `--lr RATE` | learning rate of the first epoch | 0.001 |
| `--schedule SPEC` | `constant`, `exp:<decay>`, `step:<factor>:<every>` or `time:<decay>` | `exp:0.9` |
| `--optimizer NAME` | `adam` or `sgd` (with 0.9 momentum) | `adam` |
| `--loss NAME` | `cross-entropy` or `mse` | `cross-entropy` |
| `--l2 LAMBDA` | L2 regularisation / weight decay (try `1e-4`) | 0 |
| `--clip NORM` | gradient clipping by global norm (try `5`) | off |
| `--activation NAME` | `relu`, `leaky-relu`, `elu`, `sigmoid`, `tanh`, `softplus` | `relu` |
| `--init NAME` | `he`, `xavier`, `uniform`, `zero` | `he` |
| `--dropout RATE` | dropout before the output layer (0 disables it) | 0.25 |
| `--width N` | filters in the first convolution; later blocks use 2× and 4× | 16 |
| `--no-augment` | turn off the random flips and shifts | on |
| `--limit N` | train on only the first N images | all |
| `--seed N` | make a run reproducible | random |
| `--resume` | continue training the model at `--model` | off |
| `--threads N` | CPU threads | all cores |

The same options are available from Rust through the builder:

```rust
use pawnet::nn::{Loss, LrSchedule};
use pawnet::train::TrainConfig;

let config = TrainConfig::new()
    .epochs(30)
    .schedule(LrSchedule::Step { factor: 0.5, every: 10 })
    .l2(1e-4)
    .gradient_clipping(5.0)
    .loss(Loss::CrossEntropy);
```

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

> Note: `train` keeps the epoch with the best test accuracy, so these figures are slightly
> optimistic. A strict setup would choose the epoch on a separate validation split.
