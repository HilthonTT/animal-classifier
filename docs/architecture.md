# Architecture

## Crates

The repository is a Cargo workspace of three crates, each with one job:

```
                ┌──────────────────────────┐
                │        pawnet-cli        │  crates/cli: the `pawnet` binary
                │  argument parsing,       │  (train, evaluate, predict, expand, …)
                │  one module per command  │
                └────────────┬─────────────┘
                             │ uses
                ┌────────────▼─────────────┐
                │          pawnet          │  crates/classifier: the domain library
                │  CIFAR-10 data, images,  │  (data, image, model, train, ensemble,
                │  training, ensembles     │   metrics, predictor)
                └────────────┬─────────────┘
                             │ uses (re-exported as `pawnet::nn`)
                ┌────────────▼─────────────┐
                │        pawnet-nn         │  crates/nn: a generic neural-network
                │  layers, losses,         │  framework that knows nothing about
                │  optimizers, save/load   │  animals or images
                └──────────────────────────┘
```

This mirrors the split in the [reference project](https://github.com/m-jovanovic/digit-recognizer)
(`MachineLearning`, `Core`, `Engine` / `DatasetExpansion`). The CLI is the reference's `Engine`
and `DatasetExpansion` combined.

| Path | Contents |
|------|----------|
| `crates/nn/src/layers/` | `Conv2d`, `MaxPool2d`, `Dense`, `Activation` (6 functions), `Dropout`, and the `Layer` trait |
| `crates/nn/src/loss.rs` | softmax, cross-entropy, mean-squared error |
| `crates/nn/src/optimizer.rs` | Adam, SGD with momentum, gradient clipping |
| `crates/nn/src/schedule.rs` | learning-rate schedules |
| `crates/nn/src/init.rs` | He / Xavier / uniform / zero initialisation |
| `crates/nn/src/network.rs` | the sequential `Network`, back-propagation, L2 regularisation |
| `crates/nn/src/format.rs` | the [model file format](model-format.md) |
| `crates/classifier/src/data/` | CIFAR-10 reader/writer, normalisation, augmentation, dataset expansion |
| `crates/classifier/src/image/` | PNG (with DEFLATE), PPM and BMP decoders, a PPM encoder, resize/crop/pad/flip/contrast, the photo `Preprocessor` |
| `crates/classifier/src/model.rs` | the default CNN and its options |
| `crates/classifier/src/train.rs` | `TrainConfig` builder, multi-threaded training and evaluation |
| `crates/classifier/src/ensemble.rs` | averaging several models |
| `crates/classifier/src/saliency.rs` | heat maps of the pixels behind a prediction (SmoothGrad) |
| `crates/cli/src/commands/` | one file per sub-command |

## The network

```
 32×32 RGB image
       │
 ┌─────▼────────────────────────────┐
 │ Conv 3×3 (16) → ReLU → MaxPool   │  16×16×16
 │ Conv 3×3 (32) → ReLU → MaxPool   │  32×8×8
 │ Conv 3×3 (64) → ReLU → MaxPool   │  64×4×4
 │ Dense 1024 → 128 → ReLU          │
 │ Dropout 25%                      │
 │ Dense 128 → 6                    │
 └─────┬────────────────────────────┘
       ▼
 softmax → P(bird), P(cat), P(deer), P(dog), P(frog), P(horse)
```

The activation, initialisation, dropout rate and width are all configurable (see
[training.md](training.md)).

## How a prediction is made

1. **Load:** PNG, PPM or BMP is decoded into an `RgbImage`.
2. **Fit:** the image is made square by cropping, padding or stretching, then shrunk to
   32×32 with a box filter. Auto-contrast is optional.
3. **Normalise:** pixels are scaled to `[0, 1]`, then centred with the CIFAR-10 per-channel
   mean and standard deviation, in planar `[R][G][B]` order.
4. **Forward:** each layer turns its input into an output, ending in six scores (logits).
5. **Softmax:** the scores become probabilities.
6. **Optional averaging:** over the mirror image (test-time augmentation) and/or over
   several models (an ensemble).

## How training works

For each mini-batch of 64 images:

1. The batch is split across all CPU cores with `std::thread::scope`.
2. Each thread takes its images and, for each one:
   - applies a random flip and a shift of up to ±2 px;
   - runs the **training-mode** forward pass, where dropout is active and its mask is cached;
   - computes the loss gradient and **back-propagates** it through every layer, adding the
     parameter gradients into the thread's own buffers.
3. The buffers are summed and averaged. L2 regularisation and gradient clipping are applied
   if enabled.
4. The optimizer (Adam by default) updates every weight.

Threads never share mutable state, and their gradients are always summed in the same order.
Training is therefore fully reproducible for a given `--seed` and `--threads`. Each thread
has its own random stream for augmentation and dropout, so changing the thread count changes
the random choices, but not the quality you can expect.
After each epoch the learning rate follows the schedule. The model is then evaluated on the
test set, and saved if it beats the best epoch so far.

## Correctness

`crates/nn/tests/gradient_check.rs` compares every analytic gradient with a numerical one
(central finite differences), for every layer type, every activation function and both
losses. If back-propagation were wrong anywhere, this test would fail.
