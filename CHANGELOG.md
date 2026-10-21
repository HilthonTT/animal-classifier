# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project logo (`assets/logo.svg`).
- `docs/usage.md` and a documentation index (`docs/README.md`).

### Changed
- Renamed the project to **Pawnet**: the crates are now `pawnet-nn`, `pawnet` and `pawnet-cli`,
  and the binary is `pawnet`. The models are now `models/pawnet.bin` and
  `models/ensemble/pawnet-seed<n>.bin`, and model files start with `PAWNET`; files written
  before the rename still load.
- The README is now a short overview; the details moved to `docs/`.

### Fixed
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
