use std::fs;

use pawnet::data::cifar::{self, Split};
use pawnet::data::expand::{ExpandOptions, expand};
use pawnet::nn::Rng;

use super::{AppResult, DEFAULT_DATA_DIR, time_seed};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&["data", "out", "shift", "rotate", "no-flip", "seed"])?;
    let data_dir = opts.path("data", DEFAULT_DATA_DIR);
    let out_dir = opts.path("out", "data/cifar-10-expanded");
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

    let train_set = cifar::load_animals(&data_dir, Split::Train)?;
    println!(
        "Expanding {} images ×{} (flip: {}, shift: {}px, rotate: ±{}°)…",
        train_set.len(),
        options.factor(),
        options.flip,
        options.shift,
        options.rotate
    );
    let expanded = expand(&train_set, &options, &mut rng);
    let files = cifar::save_animals(&expanded, &out_dir)?;

    let test_files = Split::Test.files(&data_dir)?;
    for file in &test_files {
        fs::copy(file, out_dir.join(file.file_name().expect("file path")))?;
    }
    println!(
        "Wrote {} training images in {} files (+ test set) to {}",
        expanded.len(),
        files.len(),
        out_dir.display()
    );
    println!(
        "Train on it with: pawnet train --data {}",
        out_dir.display()
    );
    Ok(())
}
