use std::fs;

use pawnet::data::cifar::{DatasetKind, Split};
use pawnet::data::expand::{ExpandOptions, expand};
use pawnet::nn::Rng;

use super::{AppResult, dataset, time_seed};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&[
        "data", "dataset", "out", "shift", "rotate", "no-flip", "seed",
    ])?;
    let (kind, data_dir) = dataset(opts, DatasetKind::Animals)?;
    let out_dir = opts.path("out", kind.default_expanded_dir());
    let same_dir = match (fs::canonicalize(&out_dir), fs::canonicalize(&data_dir)) {
        (Ok(out), Ok(data)) => out == data,
        _ => out_dir == data_dir,
    };
    if same_dir {
        return Err("--out must differ from --data".into());
    }
    let defaults = ExpandOptions::default();
    let options = ExpandOptions {
        flip: !opts.flag("no-flip"),
        shift: opts.get("shift", defaults.shift)?,
        rotate: opts.get("rotate", defaults.rotate)?,
    };
    let mut rng = Rng::new(opts.get("seed", time_seed())?);

    let train_set = kind.load(&data_dir, Split::Train)?;
    println!(
        "Expanding {} images ×{} (flip: {}, shift: {}px, rotate: ±{}°)…",
        train_set.len(),
        options.factor(),
        options.flip,
        options.shift,
        options.rotate
    );
    let expanded = expand(&train_set, &options, &mut rng);
    let files = kind.save(&expanded, &out_dir)?;

    for file in kind.files(&data_dir, Split::Test)? {
        fs::copy(&file, out_dir.join(file.file_name().expect("file path")))?;
    }
    for name in kind.metadata_files() {
        let file = data_dir.join(name);
        if file.exists() {
            fs::copy(&file, out_dir.join(name))?;
        }
    }
    println!(
        "Wrote {} training images in {} files (+ test set) to {}",
        expanded.len(),
        files.len(),
        out_dir.display()
    );
    let dataset_flag = match kind {
        DatasetKind::Animals => String::new(),
        other => format!(" --dataset {other}"),
    };
    println!(
        "Train on it with: pawnet train{dataset_flag} --data {}",
        out_dir.display()
    );
    Ok(())
}
