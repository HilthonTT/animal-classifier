pub mod evaluate;
pub mod expand;
pub mod export_samples;
pub mod info;
pub mod predict;
pub mod train;

use std::error::Error;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use pawnet::data::IMAGE_LEN;
use pawnet::data::cifar::DatasetKind;
use pawnet::{Ensemble, Predictor};

use crate::args::Options;

pub type AppResult = Result<(), Box<dyn Error>>;

pub const DEFAULT_MODEL_PATH: &str = "models/pawnet.bin";

pub fn dataset(opts: &Options, default: DatasetKind) -> Result<(DatasetKind, PathBuf), String> {
    let kind = opts.get("dataset", default)?;
    Ok((kind, opts.path("data", kind.default_dir())))
}

pub fn load_predictor(opts: &Options) -> Result<Box<dyn Predictor>, Box<dyn Error>> {
    let spec = opts.string("model", DEFAULT_MODEL_PATH);
    let paths: Vec<PathBuf> = spec.split(',').map(|p| PathBuf::from(p.trim())).collect();
    let ensemble = Ensemble::load(&paths)?;
    if ensemble.input_len() != IMAGE_LEN {
        return Err(format!(
            "the model expects {} inputs, not a {IMAGE_LEN}-value 32x32 RGB image",
            ensemble.input_len()
        )
        .into());
    }
    Ok(match ensemble.into_single() {
        Ok(network) => Box::new(network),
        Err(ensemble) => {
            eprintln!("Using an ensemble of {} models\n", ensemble.len());
            Box::new(ensemble)
        }
    })
}

pub fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pawnet::nn::{Dense, Network, Rng};

    #[test]
    fn rejects_models_with_the_wrong_input_size() {
        let path = std::env::temp_dir().join(format!("cli-input-{}.bin", std::process::id()));
        let labels = vec!["a".to_string(), "b".to_string()];
        let mut net = Network::new(labels);
        net.push(Dense::new(2, 2, &mut Rng::new(1)));
        net.save(&path).unwrap();
        let opts = Options::parse(&["--model".to_string(), path.display().to_string()]).unwrap();
        let result = load_predictor(&opts);
        std::fs::remove_file(&path).unwrap();
        let err = result.err().unwrap().to_string();
        assert!(err.contains("expects 2 inputs"), "{err}");
    }
}
