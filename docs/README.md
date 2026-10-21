# Documentation

| Guide | Read it to… |
|-------|-------------|
| [Usage](usage.md) | classify images, learn the commands, use the crates from Rust |
| [Training guide](training.md) | train, tune and reproduce models |
| [Architecture](architecture.md) | understand the crates, the network and back-propagation |
| [Model format](model-format.md) | read or write `.bin` model files |

Also see [the dataset](../data/README.md), [the pre-trained models](../models/README.md)
and [CONTRIBUTING.md](../CONTRIBUTING.md).

## Features at a glance

**The framework** ([`crates/nn`](../crates/nn), crate `pawnet-nn`)
- Layers: `Conv2d`, `MaxPool2d`, `Dense`, `Dropout`, and 6 activations (ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softplus)
- Softmax cross-entropy and mean-squared-error losses; Adam and SGD with momentum
- Dropout, L2 weight decay and gradient clipping
- He, Xavier, uniform and zero initialisation; constant, exponential, step and time-based learning-rate schedules
- A versioned binary model format
- Back-propagation checked against numerical gradients for every layer, activation and loss

**The classifier** ([`crates/classifier`](../crates/classifier), crate `pawnet`)
- Multi-threaded, reproducible training with `std::thread::scope`
- On-the-fly augmentation, and offline dataset expansion (4× the images, including rotations)
- Ensembles and test-time augmentation
- Heat maps of the pixels behind each prediction
- Best-epoch selection on a validation split, and CSV training logs
- PNG (with its own DEFLATE decompressor), PPM and BMP decoders with crop/pad/stretch fitting and auto-contrast
- A builder API: `TrainConfig::new().epochs(30).l2(1e-4).gradient_clipping(5.0)`

**The CLI** ([`crates/cli`](../crates/cli), binary `pawnet`): `train`, `evaluate`,
`predict`, `expand`, `export-samples` and `info`.

## Repository layout

```
crates/nn/            the neural-network framework (layers, losses, optimizers, model format)
crates/classifier/    CIFAR-10 data, images, training, ensembles, metrics
crates/cli/           the command-line tool, one file per command
docs/                 these guides
models/               pre-trained models (committed)
data/                 the dataset, once downloaded (git-ignored)
scripts/              dataset download for Linux/macOS and Windows
assets/               the logo
```
