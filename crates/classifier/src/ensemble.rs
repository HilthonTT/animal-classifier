use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::image::preprocess::average;
use crate::nn::Network;
use crate::predictor::Predictor;

pub const MODEL_EXTENSION: &str = "bin";

pub struct Ensemble {
    members: Vec<Network>,
}

impl Ensemble {
    pub fn new(members: Vec<Network>) -> Result<Self> {
        let first = members
            .first()
            .ok_or_else(|| Error::format("an ensemble needs at least one model"))?;
        for other in &members[1..] {
            if other.labels() != first.labels() || other.input_len() != first.input_len() {
                return Err(Error::format(
                    "all ensemble members must have the same classes and input size",
                ));
            }
        }
        Ok(Self { members })
    }

    pub fn load(paths: &[PathBuf]) -> Result<Self> {
        let mut members = Vec::new();
        for path in paths {
            for file in model_files(path)? {
                members.push(Network::load(&file)?);
            }
        }
        Self::new(members)
    }

    pub fn members(&self) -> &[Network] {
        &self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn into_single(mut self) -> std::result::Result<Network, Self> {
        if self.members.len() == 1 {
            Ok(self.members.remove(0))
        } else {
            Err(self)
        }
    }
}

impl Predictor for Ensemble {
    fn labels(&self) -> &[String] {
        self.members[0].labels()
    }

    fn input_len(&self) -> usize {
        self.members[0].input_len()
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        average(self.members.iter().map(|m| m.predict(input)))
    }
}

pub fn model_files(path: &Path) -> Result<Vec<PathBuf>> {
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = fs::read_dir(path)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == MODEL_EXTENSION))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::format(format!(
            "no .{MODEL_EXTENSION} model files in {}",
            path.display()
        )));
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Dense, Rng};

    fn net(labels: &[&str], seed: u64) -> Network {
        let mut net = Network::new(labels.iter().map(|s| s.to_string()).collect());
        net.push(Dense::new(3, labels.len(), &mut Rng::new(seed)));
        net
    }

    #[test]
    fn averages_member_predictions() {
        let (a, b) = (net(&["x", "y"], 1), net(&["x", "y"], 2));
        let input = [0.3, -0.2, 0.9];
        let expected: Vec<f32> = a
            .predict(&input)
            .iter()
            .zip(b.predict(&input))
            .map(|(p, q)| (p + q) / 2.0)
            .collect();
        let ensemble = Ensemble::new(vec![a, b]).unwrap();
        assert_eq!(Predictor::predict(&ensemble, &input), expected);
    }

    #[test]
    fn rejects_mismatched_members() {
        assert!(Ensemble::new(vec![net(&["x", "y"], 1), net(&["x", "z"], 2)]).is_err());
        assert!(Ensemble::new(vec![]).is_err());
    }

    #[test]
    fn loads_every_model_in_a_directory() {
        let dir = std::env::temp_dir().join(format!("ensemble-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        net(&["x", "y"], 1).save(dir.join("a.bin")).unwrap();
        net(&["x", "y"], 2).save(dir.join("b.bin")).unwrap();
        fs::write(dir.join("README.md"), "not a model").unwrap();
        let ensemble = Ensemble::load(std::slice::from_ref(&dir));
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(ensemble.unwrap().len(), 2);
    }
}
