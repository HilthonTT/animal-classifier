use std::path::PathBuf;

use pawnet::data::cifar;
use pawnet::ensemble::model_files;
use pawnet::model::{self, ModelOptions};
use pawnet::nn::{Network, Rng};

use super::{AppResult, DEFAULT_MODEL_PATH};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&["model"])?;
    let spec = opts.string("model", DEFAULT_MODEL_PATH);
    let paths: Vec<PathBuf> = spec.split(',').map(|p| PathBuf::from(p.trim())).collect();
    if paths.iter().all(|p| !p.exists()) {
        println!("No model at {spec} — showing the untrained default architecture.\n");
        let labels: Vec<String> = cifar::ANIMAL_CLASSES
            .iter()
            .map(|s| s.to_string())
            .collect();
        let net = model::pawnet_cnn(&labels, &ModelOptions::default(), &mut Rng::new(0));
        print!("{}", net.summary());
        return Ok(());
    }
    for path in &paths {
        for file in model_files(path)? {
            let net = Network::load(&file)?;
            println!("Model: {}\n", file.display());
            print!("{}", net.summary());
            println!("Classes: {}\n", net.labels().join(", "));
        }
    }
    Ok(())
}
