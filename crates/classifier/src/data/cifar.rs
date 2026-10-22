use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

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
const CIFAR100_RECORD_LEN: usize = 2 + IMAGE_LEN;
const CIFAR100_CLASS_COUNT: usize = 100;
const CIFAR100_NAMES: &str = "fine_label_names.txt";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

pub const RECORDS_PER_FILE: usize = 10_000;

const HINT: &str = "hint: download the dataset first with `scripts/download_data.sh` \
                    (or `scripts\\download_data.ps1` on Windows)";
const CIFAR100_HINT: &str = "hint: download CIFAR-100 first with \
                             `scripts/download_data.sh cifar100` \
                             (or `scripts\\download_data.ps1 cifar100` on Windows)";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DatasetKind {
    #[default]
    Animals,
    Cifar10,
    Cifar100,
}

impl DatasetKind {
    pub fn default_dir(self) -> &'static str {
        match self {
            DatasetKind::Animals | DatasetKind::Cifar10 => "data/cifar-10-batches-bin",
            DatasetKind::Cifar100 => "data/cifar-100-binary",
        }
    }

    pub fn default_expanded_dir(self) -> &'static str {
        match self {
            DatasetKind::Animals | DatasetKind::Cifar10 => "data/cifar-10-expanded",
            DatasetKind::Cifar100 => "data/cifar-100-expanded",
        }
    }

    pub fn detect(labels: &[String]) -> Self {
        if labels.iter().eq(ANIMAL_CLASSES.iter()) {
            DatasetKind::Animals
        } else if labels.iter().eq(CIFAR10_CLASSES.iter()) {
            DatasetKind::Cifar10
        } else {
            DatasetKind::Cifar100
        }
    }

    pub fn class_names(self, dir: &Path) -> Result<Vec<String>> {
        let names = |list: &[&str]| list.iter().map(|s| s.to_string()).collect();
        match self {
            DatasetKind::Animals => Ok(names(&ANIMAL_CLASSES)),
            DatasetKind::Cifar10 => Ok(names(&CIFAR10_CLASSES)),
            DatasetKind::Cifar100 => {
                let path = dir.join(CIFAR100_NAMES);
                let text = fs::read_to_string(&path).map_err(|e| {
                    Error::format(format!(
                        "cannot read {}: {e}\n{CIFAR100_HINT}",
                        path.display()
                    ))
                })?;
                let names: Vec<String> = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect();
                if names.len() != CIFAR100_CLASS_COUNT {
                    return Err(Error::format(format!(
                        "{} lists {} classes, expected {CIFAR100_CLASS_COUNT}",
                        path.display(),
                        names.len()
                    )));
                }
                Ok(names)
            }
        }
    }

    pub fn files(self, dir: &Path, split: Split) -> Result<Vec<PathBuf>> {
        match (self, split) {
            (DatasetKind::Cifar100, Split::Train) => Ok(vec![dir.join("train.bin")]),
            (DatasetKind::Cifar100, Split::Test) => Ok(vec![dir.join("test.bin")]),
            _ => split.files(dir),
        }
    }

    pub fn metadata_files(self) -> &'static [&'static str] {
        match self {
            DatasetKind::Animals | DatasetKind::Cifar10 => &["batches.meta.txt"],
            DatasetKind::Cifar100 => &[CIFAR100_NAMES, "coarse_label_names.txt"],
        }
    }

    pub fn load(self, dir: &Path, split: Split) -> Result<Dataset> {
        let mut dataset = Dataset::new(self.class_names(dir)?);
        let hint = match self {
            DatasetKind::Cifar100 => CIFAR100_HINT,
            _ => HINT,
        };
        for path in self.files(dir, split)? {
            let bytes = fs::read(&path).map_err(|e| {
                Error::format(format!("cannot read {}: {e}\n{hint}", path.display()))
            })?;
            self.parse(&bytes, &mut dataset)
                .map_err(|e| Error::format(format!("{}: {e}", path.display())))?;
        }
        Ok(dataset)
    }

    pub fn parse(self, bytes: &[u8], dataset: &mut Dataset) -> Result<()> {
        if self == DatasetKind::Animals {
            return parse_animals(bytes, dataset);
        }
        let record_len = self.record_len();
        if bytes.is_empty() || bytes.len() % record_len != 0 {
            return Err(Error::format(format!(
                "file size {} is not a multiple of the record size {record_len}",
                bytes.len()
            )));
        }
        let label_at = record_len - IMAGE_LEN - 1;
        for record in bytes.chunks_exact(record_len) {
            let label = record[label_at];
            if label as usize >= dataset.class_names.len() {
                return Err(Error::format(format!("invalid label {label}")));
            }
            dataset.push(&record[record_len - IMAGE_LEN..], label);
        }
        Ok(())
    }

    pub fn encode(self, dataset: &Dataset) -> Vec<u8> {
        let mut out = Vec::with_capacity(dataset.len() * self.record_len());
        for i in 0..dataset.len() {
            let label = dataset.labels[i];
            match self {
                DatasetKind::Animals => out.push(label + FIRST_ANIMAL_LABEL),
                DatasetKind::Cifar10 => out.push(label),
                DatasetKind::Cifar100 => out.extend_from_slice(&[0, label]),
            }
            out.extend_from_slice(dataset.image(i));
        }
        out
    }

    pub fn save(self, dataset: &Dataset, dir: &Path) -> Result<Vec<PathBuf>> {
        fs::create_dir_all(dir)?;
        if self == DatasetKind::Cifar100 {
            let path = dir.join("train.bin");
            fs::write(&path, self.encode(dataset))?;
            return Ok(vec![path]);
        }
        let mut written = Vec::new();
        for (n, start) in (0..dataset.len()).step_by(RECORDS_PER_FILE).enumerate() {
            let mut chunk = Dataset::new(dataset.class_names.clone());
            for i in start..(start + RECORDS_PER_FILE).min(dataset.len()) {
                chunk.push(dataset.image(i), dataset.labels[i]);
            }
            let path = dir.join(format!("data_batch_{}.bin", n + 1));
            fs::write(&path, self.encode(&chunk))?;
            written.push(path);
        }
        Ok(written)
    }

    fn record_len(self) -> usize {
        match self {
            DatasetKind::Animals | DatasetKind::Cifar10 => RECORD_LEN,
            DatasetKind::Cifar100 => CIFAR100_RECORD_LEN,
        }
    }
}

impl fmt::Display for DatasetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DatasetKind::Animals => "animals",
            DatasetKind::Cifar10 => "cifar10",
            DatasetKind::Cifar100 => "cifar100",
        })
    }
}

impl FromStr for DatasetKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_ascii_lowercase().replace('-', "").as_str() {
            "animals" => Ok(DatasetKind::Animals),
            "cifar10" => Ok(DatasetKind::Cifar10),
            "cifar100" => Ok(DatasetKind::Cifar100),
            _ => Err(format!(
                "unknown dataset `{s}` (expected animals, cifar10 or cifar100)"
            )),
        }
    }
}

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
    DatasetKind::Animals.load(dir, split)
}

pub fn encode_animals(dataset: &Dataset) -> Vec<u8> {
    DatasetKind::Animals.encode(dataset)
}

pub fn save_animals(dataset: &Dataset, dir: &Path) -> Result<Vec<PathBuf>> {
    DatasetKind::Animals.save(dataset, dir)
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
    fn every_kind_round_trips_through_encode_and_parse() {
        let dir = std::env::temp_dir().join(format!("cifar-kinds-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let names: String = (0..100).map(|i| format!("class{i}\n")).collect();
        fs::write(dir.join(CIFAR100_NAMES), names).unwrap();
        for kind in [
            DatasetKind::Animals,
            DatasetKind::Cifar10,
            DatasetKind::Cifar100,
        ] {
            let classes = kind.class_names(&dir).unwrap();
            let mut ds = Dataset::new(classes.clone());
            let last = (classes.len() - 1) as u8;
            ds.push(&[1; IMAGE_LEN], 0);
            ds.push(&[2; IMAGE_LEN], last);
            let bytes = kind.encode(&ds);
            let mut back = Dataset::new(classes);
            kind.parse(&bytes, &mut back).unwrap();
            assert_eq!(back.labels, vec![0, last], "{kind}");
            assert_eq!(back.images, ds.images, "{kind}");
            assert_eq!(kind.to_string().parse::<DatasetKind>(), Ok(kind));
        }
        let saved = DatasetKind::Cifar100
            .save(
                &{
                    let mut ds = Dataset::new(DatasetKind::Cifar100.class_names(&dir).unwrap());
                    ds.push(&[7; IMAGE_LEN], 42);
                    ds
                },
                &dir,
            )
            .unwrap();
        let loaded = DatasetKind::Cifar100.load(&dir, Split::Train);
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(saved.len(), 1);
        let loaded = loaded.unwrap();
        assert_eq!(loaded.labels, vec![42]);
        assert_eq!(loaded.class_names[42], "class42");
    }

    #[test]
    fn detects_the_dataset_from_class_names() {
        let names = |list: &[&str]| -> Vec<String> { list.iter().map(|s| s.to_string()).collect() };
        assert_eq!(
            DatasetKind::detect(&names(&ANIMAL_CLASSES)),
            DatasetKind::Animals
        );
        assert_eq!(
            DatasetKind::detect(&names(&CIFAR10_CLASSES)),
            DatasetKind::Cifar10
        );
        assert_eq!(
            DatasetKind::detect(&names(&["apple", "aquarium_fish"])),
            DatasetKind::Cifar100
        );
    }

    #[test]
    fn rejects_bad_sizes() {
        let mut ds = empty_animal_dataset();
        assert!(parse_animals(&[1, 2, 3], &mut ds).is_err());
    }
}
