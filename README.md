<p align="center">
  <img src="assets/logo.svg" width="128" alt="Pawnet logo">
</p>

<h1 align="center">Pawnet</h1>

<p align="center">
  A convolutional neural network that recognises animals in photos,<br>
  written from scratch in pure Rust with <b>zero dependencies</b>.
</p>

<p align="center">
  <a href="https://github.com/HilthonTT/pawnet/actions/workflows/ci.yml"><img src="https://github.com/HilthonTT/pawnet/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
</p>

🐦 bird · 🐱 cat · 🦌 deer · 🐶 dog · 🐸 frog · 🐴 horse — trained on the animal classes of
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). The best included model scores
**81%** on the 6 000 test images (random guessing: 16.7%).

## Quick start

```bash
git clone https://github.com/HilthonTT/pawnet.git
cd pawnet
cargo run --release -- predict my-photo.ppm --tta    # pre-trained model included
```

Photos must be PPM or BMP (`magick photo.jpg photo.ppm`); see [usage](docs/usage.md#using-your-own-photos).

To train your own model:

```bash
./scripts/download_data.sh
cargo run --release -- train
cargo run --release -- evaluate --tta
```

## Documentation

- [Usage](docs/usage.md): commands, your own photos, using it as a library
- [Training guide](docs/training.md): every option, recipes, reference results
- [Architecture](docs/architecture.md): the crates, the network, how training works
- [Model format](docs/model-format.md) · [Dataset](data/README.md) · [Pre-trained models](models/README.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

[MIT](LICENSE). Inspired by [m-jovanovic/digit-recognizer](https://github.com/m-jovanovic/digit-recognizer).
