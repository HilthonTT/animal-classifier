use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::layers::{self, LayerKind};
use crate::network::Network;

/// Magic bytes at the start of every model file.
const MAGIC: &[u8; 8] = b"PAWNET\0\0";
const LEGACY_MAGIC: &[u8; 8] = b"ANIMLNN\0";
/// Model file format version written by this crate.
const VERSION: u32 = 2;
/// Oldest version this crate can still read.
const MIN_VERSION: u32 = 1;

impl Network {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.param_count() * 4 + 256);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.labels.len() as u32).to_le_bytes());
        for label in &self.labels {
            out.extend_from_slice(&(label.len() as u32).to_le_bytes());
            out.extend_from_slice(label.as_bytes());
        }
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            out.push(layer.kind() as u8);
            let config = layer.config();
            out.extend_from_slice(&(config.len() as u32).to_le_bytes());
            for value in config {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&(layer.params().len() as u32).to_le_bytes());
            for p in layer.params() {
                out.extend_from_slice(&p.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { bytes, pos: 0 };
        let magic = r.take(MAGIC.len())?;
        if magic != MAGIC && magic != LEGACY_MAGIC {
            return Err(Error::format("not a pawnet model file"));
        }
        let version = r.u32()?;
        if !(MIN_VERSION..=VERSION).contains(&version) {
            return Err(Error::format(format!(
                "unsupported model version {version} (this build reads {MIN_VERSION} to {VERSION})"
            )));
        }

        let label_count = r.u32()? as usize;
        let mut labels = Vec::new();
        for _ in 0..label_count {
            let len = r.u32()? as usize;
            let label = std::str::from_utf8(r.take(len)?)
                .map_err(|_| Error::format("label is not valid UTF-8"))?;
            labels.push(label.to_string());
        }

        let mut net = Network::new(labels);
        let layer_count = r.u32()?;
        for _ in 0..layer_count {
            let kind = LayerKind::try_from(r.u8()?)?;
            let config_len = r.u32()? as usize;
            let config = (0..config_len)
                .map(|_| r.u32())
                .collect::<Result<Vec<_>>>()?;
            let mut layer = layers::from_config(kind, &config)?;
            let param_count = r.u32()? as usize;
            if param_count != layer.params().len() {
                return Err(Error::format(format!(
                    "layer `{}` should have {} parameters, file has {param_count}",
                    layer.describe(),
                    layer.params().len()
                )));
            }
            let data = r.take(param_count * 4)?;
            for (p, chunk) in layer.params_mut().iter_mut().zip(data.chunks_exact(4)) {
                *p = f32::from_le_bytes(chunk.try_into().expect("chunk of 4 bytes"));
            }
            net.try_push(layer)?;
        }

        if net.layers.is_empty() {
            return Err(Error::format("model has no layers"));
        }
        if net.output_len() != net.labels.len() {
            return Err(Error::format(
                "model output size does not match its number of classes",
            ));
        }
        Ok(net)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.to_bytes())?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|e| {
            Error::format(format!(
                "cannot read model {}: {e} (train one first with `pawnet train`)",
                path.display()
            ))
        })?;
        Self::from_bytes(&bytes).map_err(|e| Error::format(format!("{}: {e}", path.display())))
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len());
        let end = end.ok_or_else(|| Error::format("model file is truncated"))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Activation, ActivationFn, Conv2d, Dense, Dropout, MaxPool2d};
    use crate::rng::Rng;

    fn tiny_net() -> Network {
        let mut rng = Rng::new(5);
        let mut net = Network::new(vec!["a".into(), "b".into()]);
        net.push(Conv2d::new(1, 4, 4, 2, 3, 1, &mut rng))
            .push(Activation::new(ActivationFn::Elu, 32))
            .push(MaxPool2d::new(2, 4, 4, 2))
            .push(Dropout::new(8, 0.25))
            .push(Dense::new(8, 2, &mut rng));
        net
    }

    #[test]
    fn save_and_load_round_trip() {
        let net = tiny_net();
        let restored = Network::from_bytes(&net.to_bytes()).unwrap();
        assert_eq!(restored.labels(), net.labels());
        assert_eq!(restored.layers().len(), net.layers().len());
        assert_eq!(restored.to_bytes(), net.to_bytes());
        let input: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        assert_eq!(restored.forward(&input), net.forward(&input));
    }

    #[test]
    fn rejects_corrupt_files() {
        let bytes = tiny_net().to_bytes();
        assert!(Network::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(Network::from_bytes(b"garbage").is_err());
    }

    #[test]
    fn rejects_huge_layer_shapes_without_panicking() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(b'x');
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert!(Network::from_bytes(&bytes).is_err());
    }

    #[test]
    fn reports_models_without_layers() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = Network::from_bytes(&bytes).err().unwrap();
        assert_eq!(err.to_string(), "model has no layers");
    }

    #[test]
    fn reads_files_with_the_legacy_magic() {
        let mut bytes = tiny_net().to_bytes();
        bytes[..8].copy_from_slice(LEGACY_MAGIC);
        assert!(Network::from_bytes(&bytes).is_ok());
    }

    #[test]
    fn reads_version_1_files() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(b'x');
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(-3.0f32).to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.push(4);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let net = Network::from_bytes(&bytes).unwrap();
        assert_eq!(net.forward(&[1.0]), vec![0.0]);
    }
}
