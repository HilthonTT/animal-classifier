# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **JPEG support:** a from-scratch JPEG decoder for baseline and progressive files, with any
  chroma subsampling, restart markers, greyscale and CMYK. Photos are turned upright using their
  EXIF orientation.
- **Strided convolutions:** `Conv2d::with_stride`, and `train --pool stride` downsamples with a
  learned 3×3 convolution of stride 2 instead of pooling. The stride is stored only when it is
  not 1, so stride-1 models are written exactly as before.
- `predict --top N` lists the N most likely classes, and `predict --json` prints the results as
  JSON.
- **Batch normalisation:** a `BatchNorm` layer (`train --batch-norm`). Its statistics cover the
  whole mini-batch even though the batch is split across threads: the threads add up their
  partial sums at a barrier, in a fixed order, so training stays reproducible.
- **Skip connections:** a `Residual` layer that adds the input of an earlier layer
  (`train --residual` adds a two-convolution residual block to every stage).
- **Pooling:** `AvgPool2d` (`--pool avg`) and `GlobalAvgPool` (`--global-pool`, which replaces
  the 128-unit hidden layer).
- **Optimizers and schedules:** AdamW (`--optimizer adamw --weight-decay W`), RMSprop
  (`--optimizer rmsprop`), cosine annealing (`--schedule cosine[:min]`) and linear warm-up
  (`--warmup N`).
- **Early stopping:** `train --patience N` stops after N epochs without a new best model.
- **Augmentation:** colour jitter (`--jitter`), cutout (`--cutout`) and mixup (`--mixup`).
- **More datasets:** `--dataset cifar10` trains on all ten CIFAR-10 classes and
  `--dataset cifar100` on CIFAR-100 (`scripts/download_data.sh cifar100`). `evaluate` picks the
  dataset from the model's classes.
- A batch-level training API: `Network::accumulate_batch`, `Layer::forward_batch` and
  `Layer::backward_batch`, `Reducer`/`SharedReduce`, and `Target` for mixed labels.
- **PNG support:** a from-scratch PNG decoder with its own DEFLATE decompressor. Every colour
  type and bit depth, interlaced images, and CRC and Adler-32 checks; transparency becomes white.
- **Heat maps:** `predict --explain OUT` saves the pixels behind each prediction (SmoothGrad on
  gradient × input), also for ensembles and `--tta`. `Network::input_gradient` exposes the
  underlying gradient.
- **Validation split:** `train --val FRACTION` (default 0.1) holds out training images and
  keeps the epoch with the best validation accuracy, so test accuracy stays honest. `--val 0`
  restores the old behaviour.
- **Training logs:** `train --log FILE` writes per-epoch loss and accuracy as CSV.
- Project logo (`assets/logo.svg`).
- `docs/usage.md` and a documentation index (`docs/README.md`).

### Changed
- **Faster convolutions:** `Conv2d` now uses im2col and blocked matrix multiplication, and max
  pooling has a 2×2 fast path. A training step is about 4× faster and inference about 5× faster.
- Model files are now format version 3, which stores non-trainable state (batch-norm running
  statistics). Version 1 and 2 files still load.
- `LrSchedule::rate` takes the total number of epochs, and the `train::train` callback returns
  `ControlFlow` so it can stop training.
- `Augment` gained colour and cutout fields and is no longer `Eq`.
- Training now runs each layer over a thread's whole sub-batch, so the same seed no longer
  reproduces models trained with 0.2.0 bit for bit.
- Renamed the project to **Pawnet**: the crates are now `pawnet-nn`, `pawnet` and `pawnet-cli`,
  and the binary is `pawnet`. The models are now `models/pawnet.bin` and
  `models/ensemble/pawnet-seed<n>.bin`, and model files start with `PAWNET`; files written
  before the rename still load.
- The README is now a short overview; the details moved to `docs/`.
- `Predictor` has a new required method, `logit_gradient`, used by the heat maps.

### Fixed
- Saving a model writes a temporary file and renames it, so an interrupted save can no longer
  corrupt the previous model.
- Model files with oversized layer shapes, or with no layers, are rejected with an error instead of crashing.
- PPM: samples above `maxval` are rejected instead of wrapping around, and a huge header no longer allocates gigabytes.
- BMP: 32-bit `BI_BITFIELDS` colour masks are honoured.
- `predict` and `evaluate` report a model with the wrong input size or classes instead of panicking.
- `train --resume` no longer overwrites the resumed model with a worse epoch.
- `--clip` rejects zero and negative norms.
- `expand` refuses to write into its source folder, even when the path is spelled differently.
- Unknown commands are reported before option errors.
- `cargo doc` no longer overwrites the library docs with the binary's.

## [0.2.0] - 2026-10-21

### Added
- **Regularisation:** `Dropout` layer, L2 weight decay (`--l2`) and gradient clipping (`--clip`).
- **Activation functions:** LeakyReLU, ELU, Sigmoid, Tanh and Softplus alongside ReLU (`--activation`).
- **Mean-squared-error loss** as an alternative to cross-entropy (`--loss mse`).
- **Weight initialisers:** He, Xavier, uniform and zero (`--init`).
- **Learning-rate schedules:** constant, exponential, step and time-based (`--schedule`).
- `TrainConfig` builder API (`TrainConfig::new().epochs(30).l2(1e-4)…`).
- **Ensembles:** `--model` accepts several files or a folder and averages the predictions.
- **Offline dataset expansion:** the `expand` command writes flipped, shifted and rotated copies.
- **Photo preprocessing:** `--fit crop|pad|stretch`, `--auto-contrast`, and test-time augmentation (`--tta`).
- Pre-trained models in `models/`: a single model and a 3-model ensemble.
- Docs (`docs/`), contributing guide, issue and PR templates, `.editorconfig`.
- Examples: `cargo run -p pawnet-nn --example xor` and `cargo run -p pawnet --example classify`.

### Changed
- Split into a Cargo workspace: `pawnet-nn` (framework), `pawnet` (library) and
  `pawnet-cli` (binary).
- Model files are now format version 2. Version 1 files still load.
- The default network adds 25% dropout before the output layer.
- Training accepts any number of `data_batch_<n>.bin` files.

## [0.1.0] - 2026-10-21

### Added
- First release: a from-scratch CNN in pure Rust, trained on the CIFAR-10 animal classes,
  with `train`, `evaluate`, `predict`, `export-samples` and `info` commands.
