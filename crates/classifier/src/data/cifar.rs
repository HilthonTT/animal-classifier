use std::fs;
use std::path::{Path, PathBuf};

use super::{Dataset, IMAGE_LEN};
use crate::error::{Error, Result};

pub const CIFAR10_CLASSES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

pub const ANIMAL_CLASSES: [&str; 6] = ["bird", "cat", "deer", "dog", "frog", "horse"];

const FIRST_ANIMAL_LABEL: u8 = 2;

const RECORD_LEN: usize = 1 + IMAGE_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

pub const RECORDS_PER_FILE: usize = 10_000;

const HINT: &str = "hint: download the dataset first with `scripts/download_data.sh` \
                    (or `scripts\\download_data.ps1` on Windows)";

impl Split {
    pub fn files(self, dir: &Path) -> Result<Vec<PathBuf>> {
        if self == Split::Test {
            return Ok(vec![dir.join("test_batch.bin")]);
        }
        let entries = fs::read_dir(dir)
            .map_err(|e| Error::format(format!("cannot open {}: {e}\n{HINT}", dir.display())))?;
        let mut numbered: Vec<(u32, PathBuf)> = entries
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter_map(|path| {
                let name = path.file_name()?.to_str()?;
                let n = name
                    .strip_prefix("data_batch_")?
                    .strip_suffix(".bin")?
                    .parse()
                    .ok()?;
                Some((n, path))
            })
            .collect();
        if numbered.is_empty() {
            return Err(Error::format(format!(
                "no data_batch_*.bin files in {}\n{HINT}",
                dir.display()
            )));
        }
        numbered.sort();
        Ok(numbered.into_iter().map(|(_, path)| path).collect())
    }
}

pub fn empty_animal_dataset() -> Dataset {
    Dataset::new(ANIMAL_CLASSES.iter().map(|s| s.to_string()).collect())
}

pub fn load_animals(dir: &Path, split: Split) -> Result<Dataset> {
    let mut dataset = empty_animal_dataset();
    for path in split.files(dir)? {
        let bytes = fs::read(&path)
            .map_err(|e| Error::format(format!("cannot read {}: {e}\n{HINT}", path.display())))?;
        parse_animals(&bytes, &mut dataset)
            .map_err(|e| Error::format(format!("{}: {e}", path.display())))?;
    }
    Ok(dataset)
}

pub fn encode_animals(dataset: &Dataset) -> Vec<u8> {
    let mut out = Vec::with_capacity(dataset.len() * RECORD_LEN);
    for i in 0..dataset.len() {
        out.push(dataset.labels[i] + FIRST_ANIMAL_LABEL);
        out.extend_from_slice(dataset.image(i));
    }
    out
}

pub fn save_animals(dataset: &Dataset, dir: &Path) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(dir)?;
    let mut written = Vec::new();
    for (n, start) in (0..dataset.len()).step_by(RECORDS_PER_FILE).enumerate() {
        let mut chunk = Dataset::new(dataset.class_names.clone());
        for i in start..(start + RECORDS_PER_FILE).min(dataset.len()) {
            chunk.push(dataset.image(i), dataset.labels[i]);
        }
        let path = dir.join(format!("data_batch_{}.bin", n + 1));
        fs::write(&path, encode_animals(&chunk))?;
        written.push(path);
    }
    Ok(written)
}

pub fn parse_animals(bytes: &[u8], dataset: &mut Dataset) -> Result<()> {
    if bytes.is_empty() || bytes.len() % RECORD_LEN != 0 {
        return Err(Error::format(format!(
            "file size {} is not a multiple of the CIFAR-10 record size {RECORD_LEN}",
            bytes.len()
        )));
    }
    for record in bytes.chunks_exact(RECORD_LEN) {
        let label = record[0];
        if label as usize >= CIFAR10_CLASSES.len() {
            return Err(Error::format(format!("invalid CIFAR-10 label {label}")));
        }
        let animal = label.wrapping_sub(FIRST_ANIMAL_LABEL);
        if (animal as usize) < ANIMAL_CLASSES.len() {
            dataset.push(&record[1..], animal);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animal_classes_match_cifar_labels() {
        for (i, name) in ANIMAL_CLASSES.iter().enumerate() {
            assert_eq!(CIFAR10_CLASSES[FIRST_ANIMAL_LABEL as usize + i], *name);
        }
    }

    #[test]
    fn keeps_only_animals() {
        let mut bytes = Vec::new();
        for label in [0u8, 3, 9, 7] {
            bytes.push(label);
            bytes.extend(std::iter::repeat_n(label, IMAGE_LEN));
        }
        let mut ds = empty_animal_dataset();
        parse_animals(&bytes, &mut ds).unwrap();
        assert_eq!(ds.labels, vec![1, 5]);
        assert_eq!(ds.image(1)[0], 7);
    }

    #[test]
    fn encode_round_trips_through_parse() {
        let mut ds = empty_animal_dataset();
        ds.push(&[1; IMAGE_LEN], 0);
        ds.push(&[2; IMAGE_LEN], 5);
        let bytes = encode_animals(&ds);
        assert_eq!(bytes[0], 2);
        let mut back = empty_animal_dataset();
        parse_animals(&bytes, &mut back).unwrap();
        assert_eq!(back.labels, ds.labels);
        assert_eq!(back.images, ds.images);
    }

    #[test]
    fn finds_training_files_in_numeric_order() {
        let dir = std::env::temp_dir().join(format!("cifar-files-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for name in [
            "data_batch_10.bin",
            "data_batch_2.bin",
            "test_batch.bin",
            "notes.txt",
        ] {
            fs::write(dir.join(name), b"").unwrap();
        }
        let files = Split::Train.files(&dir).unwrap();
        fs::remove_dir_all(&dir).unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["data_batch_2.bin", "data_batch_10.bin"]);
    }

    #[test]
    fn rejects_bad_sizes() {
        let mut ds = empty_animal_dataset();
        assert!(parse_animals(&[1, 2, 3], &mut ds).is_err());
    }
}
