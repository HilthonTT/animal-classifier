# Dataset

This project trains on the **animal classes of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**:
a public research dataset of 60 000 labelled 32×32 colour photos.

| Class | Train | Test |
|-------|------:|-----:|
| bird  | 5 000 | 1 000 |
| cat   | 5 000 | 1 000 |
| deer  | 5 000 | 1 000 |
| dog   | 5 000 | 1 000 |
| frog  | 5 000 | 1 000 |
| horse | 5 000 | 1 000 |

The other four CIFAR-10 classes (airplane, automobile, ship, truck) are skipped when loading,
unless you train with `--dataset cifar10`, which uses all ten.

`--dataset cifar100` trains on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
instead: the same kind of 32×32 photos in 100 classes, with 500 training and 100 test images
per class. The class names come from the dataset's `fine_label_names.txt`. `evaluate` works
out which dataset a model needs from its class names.

## Download

```bash
./scripts/download_data.sh          # Linux / macOS / WSL
./scripts/download_data.sh cifar100 # CIFAR-100 (~161 MB)
```
```powershell
.\scripts\download_data.ps1         # Windows PowerShell
.\scripts\download_data.ps1 cifar100
```

This puts the files here:

```
data/
├── cifar-10-batches-bin/
│   ├── batches.meta.txt
│   ├── data_batch_1.bin … data_batch_5.bin   # training set
│   └── test_batch.bin                         # test set
└── cifar-100-binary/                          # only with `cifar100`
    ├── fine_label_names.txt
    ├── coarse_label_names.txt
    ├── train.bin
    └── test.bin
```

`cargo run --release -- expand` writes a 4× larger training set to
`data/cifar-10-expanded/` (about 370 MB) in the same format. With `--dataset cifar100` it
writes `data/cifar-100-expanded/` instead.

The dataset files are **not** committed to git (see `.gitignore`). The pre-trained models in
`models/` are, so you only need the data to train or evaluate.

## File format

We use the *binary* version because it needs no image decoder, so it can be read with
the standard library alone. Each `.bin` file holds 10 000 records of 3 073 bytes:

```
<1 byte label 0-9><1024 bytes red><1024 bytes green><1024 bytes blue>
```

Each colour plane is stored row by row, top to bottom. CIFAR-100 records have two label
bytes, the coarse (superclass) label and then the fine label, so they are 3 074 bytes long.
Pawnet uses the fine labels; `expand` writes 0 as the coarse label.

## Citation

> Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
