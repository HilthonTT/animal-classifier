# Usage

You need [Rust](https://rustup.rs) 1.85 or newer. Pre-trained models are committed in
[`models/`](../models), so `predict`, `evaluate` and `info` work right after cloning.

## Classify an image

```bash
cargo run --release -- predict dog.png --tta
```

```
dog.png (128x128)
  dog      ████████████████████░░░░░░░░░░  67.1%
  cat      ███████░░░░░░░░░░░░░░░░░░░░░░░  22.4%
  bird     █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3.7%
  horse    █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3.4%
  deer     █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.0%
  frog     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.4%
  => dog (67.1% confident)
```

| Option | Effect |
|--------|--------|
| `--model PATH` | a model file, a folder, or a comma-separated list. Several models form an **ensemble** that averages its votes |
| `--tta` | test-time augmentation: also classify the mirror image (about 2× slower, usually +1–2 points) |
| `--fit crop\|pad\|stretch` | how a non-square photo is made square before shrinking it to 32×32 |
| `--auto-contrast` | stretch the colour range of dull photos |
| `--explain OUT` | save a heat map of the pixels behind the answer (see below) |

Models with more than ten classes, such as a CIFAR-100 model, list their ten most likely
classes.

### See what the model looks at

```bash
cargo run --release -- predict dog.png --explain dog-heatmap.ppm
cargo run --release -- predict photos/*.png --explain heatmaps/    # one map per image
```

`--explain` writes a 256×256 heat map of the predicted class: the 32×32 image the network
actually sees, with the pixels that pushed it towards its answer glowing orange to yellow.

![Test images (top) and their heat maps (bottom)](../assets/heatmaps.png)

The maps use SmoothGrad on gradient × input: the class score is back-propagated to the input
pixels for 32 slightly noisy copies of the image and the results are averaged. With `--tta`
the mirrored copy contributes too, and an ensemble averages its members. Expect about 0.1 s
per image for one model. Very bright or saturated backgrounds can light up as well, because
the maps measure change relative to the dataset's average colour.

### Using your own photos

The classifier reads **PNG**, **PPM** and **24/32-bit BMP** files. The PNG decoder, including
its DEFLATE decompressor, is written from scratch like the rest of the project. It supports
every colour type and bit depth and interlaced images; transparent areas become white.

JPEG would need a far larger decoder, so convert JPEG photos first:

```bash
magick my-dog.jpg my-dog.png        # ImageMagick
ffmpeg -i my-dog.jpg my-dog.png     # FFmpeg
```

The network sees photos at only 32×32 pixels. You get the best results when the animal
fills most of the frame. If the animal sits off-centre, try `--fit pad`.

No photos at hand? `export-samples` writes test-set images as `.ppm` files (needs the
dataset).

## Train and evaluate

```bash
./scripts/download_data.sh                 # ~163 MB (Windows: .\scripts\download_data.ps1)
cargo run --release -- train               # ~10 minutes on an 8-core CPU
cargo run --release -- evaluate --tta      # accuracy, per-class precision/recall, confusion matrix
```

To train on all ten CIFAR-10 classes or on CIFAR-100 instead of the six animals:

```bash
cargo run --release -- train --dataset cifar10 --model models/cifar10.bin
./scripts/download_data.sh cifar100
cargo run --release -- train --dataset cifar100 --model models/cifar100.bin
cargo run --release -- evaluate --model models/cifar100.bin   # finds the right dataset itself
```

Every training option, recipes and reference results are in the [training guide](training.md).

> For extra speed: `RUSTFLAGS="-C target-cpu=native" cargo run --release -- train`

## Commands

| Command | Description |
|---------|-------------|
| `train` | Train a model; the best epoch (by validation accuracy) is saved |
| `evaluate` | Accuracy, per-class precision and recall, and a confusion matrix |
| `predict <IMAGE>…` | Classify `.png`, `.ppm` or `.bmp` files |
| `expand` | Write a 4× larger training set (flipped, shifted and rotated copies) |
| `export-samples` | Write test-set images as `.ppm` files |
| `info` | Print the network architecture |

Run `cargo run --release -- --help` for every option. Once installed with
`cargo install --path crates/cli`, the binary is called `pawnet`.

## As a library

Classify an image from Rust with the `pawnet` crate:

```rust
use pawnet::image::{Preprocessor, RgbImage};
use pawnet::nn::Network;

let net = Network::load("models/pawnet.bin")?;
let image = RgbImage::load("dog.png")?;
let probabilities = Preprocessor::default().predict(&net, &image);
```

Or build a network of your own with just the `pawnet-nn` framework:

```rust
use pawnet_nn::{Activation, ActivationFn, Dense, Dropout, Network, Rng};

let mut rng = Rng::new(42);
let mut net = Network::new(vec!["no".into(), "yes".into()]);
net.push(Dense::new(2, 16, &mut rng))
    .push(Activation::new(ActivationFn::Tanh, 16))
    .push(Dropout::new(16, 0.1))
    .push(Dense::new(16, 2, &mut rng));
```

Complete, runnable examples:

```bash
cargo run -p pawnet-nn --example xor
cargo run --release -p pawnet --example classify -- models/pawnet.bin dog.png
```

API documentation: `cargo doc --workspace --no-deps --open`.
