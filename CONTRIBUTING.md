# Contributing

Thanks for your interest! Bug reports, ideas and pull requests are all welcome.

## Ground rules

- **No external dependencies.** Everything is built on the Rust standard library. That is
  the point of the project, so please don't add crates to any `Cargo.toml`.
- Keep the code readable. This is a learning project, so clarity beats cleverness.
- New layers, losses and optimizers belong in `crates/nn`, together with a gradient check
  in `crates/nn/tests/gradient_check.rs`.

## Development

```bash
cargo test --workspace                                   # all tests, no dataset needed
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo run -p pawnet-nn --example xor                     # quick framework sanity check
```

To try changes on real data, download the dataset (`scripts/download_data.sh`) and do a
short run:

```bash
cargo run --release -- train --limit 3000 --epochs 2 --model /tmp/test.bin
```

## Pull requests

1. Fork and create a branch from `main`.
2. Make your change, with tests.
3. Update `CHANGELOG.md` and the docs if user-visible behaviour changed.
4. Open the PR. CI runs fmt, clippy and the tests on Linux, Windows and macOS.

## Model files

The models in `models/` are committed so the project works right after cloning. If you
change the network or the model format, retrain them (see [docs/training.md](docs/training.md))
and keep reading old files working where possible.
