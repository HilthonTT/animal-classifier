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
| `crates/nn/src/layers/` | `Conv2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool`, `Dense`, `BatchNorm`, `Residual`, `Activation` (6 functions), `Dropout`, and the `Layer` trait |
| `crates/nn/src/loss.rs` | softmax, cross-entropy, mean-squared error, mixed (mixup) targets |
| `crates/nn/src/optimizer.rs` | Adam, AdamW, SGD with momentum, RMSprop, gradient clipping |
| `crates/nn/src/schedule.rs` | learning-rate schedules, including cosine annealing |
| `crates/nn/src/math.rs` | the dot product and the blocked matrix multiplications behind `Conv2d` |
| `crates/nn/src/reduce.rs` | sums values across the threads of one batch (for batch normalisation) |
| `crates/nn/src/init.rs` | He / Xavier / uniform / zero initialisation |
| `crates/nn/src/network.rs` | the sequential `Network`, back-propagation, L2 regularisation |
| `crates/nn/src/format.rs` | the [model file format](model-format.md) |
| `crates/classifier/src/data/` | CIFAR-10 and CIFAR-100 readers/writers, normalisation, augmentation, dataset expansion |
| `crates/classifier/src/image/` | JPEG, PNG (with DEFLATE), PPM and BMP decoders, a PPM encoder, resize/crop/pad/flip/contrast, the photo `Preprocessor` |
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
[training.md](training.md)). Options can also make it a more modern network:

```
 32×32 RGB image
       │
 ┌─────▼──────────────────────────────────────────────────────┐
 │ Conv 3×3 → BatchNorm → ReLU                                 │  ┐
 │   ┌─ Conv 3×3 → BatchNorm → ReLU → Conv 3×3 → BatchNorm ─┐  │  │ --residual adds the
 │   └──────────────── + shortcut ◄─────────────────────────┘  │  │ shortcut block
 │ ReLU → MaxPool or AvgPool                                   │  ┘ × 3 stages (16, 32, 64)
 │ GlobalAvgPool 64×4×4 → 64                                   │  --global-pool
 │ Dropout → Dense 64 → classes                                │
 └─────┬──────────────────────────────────────────────────────┘
       ▼
 softmax
```

`--batch-norm` adds the BatchNorm layers, `--residual` the shortcut blocks, `--pool avg` swaps
max pooling for average pooling, and `--global-pool` replaces the 128-unit hidden layer with
global average pooling. A shortcut is a `Residual` layer: it adds the input of the layer
`span` places earlier to its own input, and back-propagation sends the gradient down both
paths.

## How a prediction is made

1. **Load:** JPEG, PNG, PPM or BMP is decoded into an `RgbImage`.
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
2. Each thread prepares its images: a random flip and a shift of up to ±2 px, plus colour
   jitter, cutout and mixup when enabled.
3. Each thread runs the **training-mode** forward pass one layer at a time over all of its
   images, so a layer sees the whole sub-batch. Dropout is active and caches its mask.
   `BatchNorm` needs the mean and variance of the *whole* batch, so the threads meet at a
   barrier and add up their partial sums (`SharedReduce`); every thread then normalises with
   the same statistics.
4. Each thread computes the loss gradients and **back-propagates** them layer by layer,
   adding the parameter gradients into its own buffers. `BatchNorm` meets the other threads
   once more to sum its gradient terms.
5. The buffers are summed and averaged. L2 regularisation and gradient clipping are applied
   if enabled, and batch norm's running statistics are updated.
6. The optimizer (Adam by default) updates every weight. AdamW also shrinks every weight
   (not the biases or batch-norm parameters) by `lr × weight decay`.

Threads only share the reduction slots, which are summed in a fixed order, and their
gradients are always summed in the same order. Training is therefore fully reproducible for
a given `--seed` and `--threads`. Each thread has its own random stream for augmentation
and dropout, so changing the thread count changes the random choices, but not the quality
you can expect.

The learning rate follows the schedule epoch by epoch; with `--warmup N` it also ramps up
linearly over the first N epochs, batch by batch. After each epoch the model is evaluated on
the validation and test sets, and saved if it beats the best epoch so far. `--patience`
stops training early when no epoch has improved for a while.

## How a convolution is computed

`Conv2d` uses **im2col**: it copies every 3×3 patch of the input into a column of a matrix,
so the whole convolution becomes one matrix multiplication of the weights
(`out_channels × in_channels·9`) by that matrix (`in_channels·9 × height·width`). The
multiplication works on 4×8 tiles of the result kept in local arrays, so the compiler keeps
them in vector registers. The backward pass is two more multiplications: the output gradient
times the patch matrix gives the weight gradients, and the transposed weights times the
output gradient give the patch gradients, which are added back onto the input (col2im).
The patch matrix from the forward pass is kept for the backward pass, and scratch buffers are
reused per thread instead of being allocated for every image.

This made a training step about 4× faster and inference about 5× faster than the earlier
loop over kernel taps. The code is plain safe Rust (the workspace forbids `unsafe`, which
rules out `std::arch` intrinsics). It relies on auto-vectorisation, so
`RUSTFLAGS="-C target-cpu=native"` helps further by allowing AVX2 on CPUs that have it.

## Correctness

`crates/nn/tests/gradient_check.rs` compares every analytic gradient with a numerical one
(central finite differences), for every layer type, every activation function and both
losses. Batch normalisation, average pooling and shortcuts are checked in training mode
over a whole batch, including a mixup target. A second test splits that batch across two
threads and checks that the gradients still match. If back-propagation were wrong anywhere,
these tests would fail.
