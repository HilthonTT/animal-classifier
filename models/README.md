# Pre-trained models

These models are committed so the classifier works right after cloning, with no
dataset download and no training:

```bash
cargo run --release -- predict path/to/animal.ppm                    # single model
cargo run --release -- predict path/to/animal.ppm --model models/ensemble --tta
```

| File | Trained on | Test accuracy | With `--tta` |
|------|-----------|--------------:|-------------:|
| `pawnet.bin` (default) | expanded dataset (120 000 images), 15 epochs | 78.4% | **81.0%** |
| `ensemble/` (3 files) | standard dataset (30 000 images), 20 epochs each | 76.2% | 77.1% |

The single model with `--tta` is the most accurate option. The ensemble shows how averaging
3 weaker models beats each of them (72.8–74.1% alone).

Every file is a self-contained [model file](../docs/model-format.md) of about 600 KB. It holds
the architecture, the class names (bird, cat, deer, dog, frog, horse) and the weights.

To reproduce them, see [docs/training.md](../docs/training.md):

```bash
cargo run --release -- expand --seed 7
cargo run --release -- train --data data/cifar-10-expanded --epochs 15 --seed 4 --threads 6 --val 0 --model models/pawnet.bin
for seed in 1 2 3; do
  cargo run --release -- train --seed $seed --threads 2 --val 0 --model models/ensemble/pawnet-seed$seed.bin
done
```

These models were trained with version 0.2.0. Later versions process each layer over a
whole batch and multiply matrices in a different order, so the commands now give models of
the same quality but not bit-identical ones. Within one version, the same seed and the same
`--threads` value reproduce the same model on the same platform. `--val 0` reproduces how these models were trained: the best epoch was chosen by
test accuracy, which makes their figures slightly optimistic.
