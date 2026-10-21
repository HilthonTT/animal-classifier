# Model file format

Trained networks are saved as small binary files (usually `.bin`). A file holds the
architecture, the class names and every weight, so loading one needs no other information.
The code lives in [`crates/nn/src/format.rs`](../crates/nn/src/format.rs).

All integers are **little-endian**. Floats are IEEE-754 `f32`.

| Field | Type | Notes |
|-------|------|-------|
| magic | 8 bytes | `PAWNET\0\0` (files written before the rename start with `ANIMLNN\0`; both are accepted) |
| version | u32 | currently `2`; version `1` files are still readable |
| label count | u32 | |
| *each label* | u32 length + UTF-8 bytes | class names, in class-id order |
| layer count | u32 | |
| *each layer* | see below | in forward order |

Each layer:

| Field | Type | Notes |
|-------|------|-------|
| kind | u8 | see table below |
| config length | u32 | |
| config | u32 × length | layer-specific shape parameters |
| parameter count | u32 | |
| parameters | f32 × count | weights first, then biases |

| Kind | Layer | Config | Parameters |
|-----:|-------|--------|------------|
| 1 | Dense | `inputs, outputs` | `outputs×inputs` weights (row-major), `outputs` biases |
| 2 | Conv2d | `in_channels, height, width, out_channels, kernel, padding` | `out×in×k×k` weights, `out` biases |
| 3 | MaxPool2d | `channels, height, width, size` | none |
| 4 | Activation | `length, function` (version 1: just `length`, meaning ReLU) | none |
| 5 | Dropout | `length, rate` (rate stored as `f32` bits) | none |

Activation function ids: `0` ReLU, `1` LeakyReLU, `2` ELU, `3` Sigmoid, `4` Tanh, `5` Softplus.

## Compatibility

- New layer kinds or activation ids may be added; existing numbers never change.
- The version only increases when existing files would be read differently.
- Files are validated when loaded: truncated data, unknown kinds, parameter counts that do
  not match the configuration, and layers that do not fit together are all rejected with an
  error rather than a crash.
