use super::{Layer, LayerKind, parse_config};
use crate::error::{Error, Result};

/// Max pooling over non-overlapping `size × size` windows (stride = size).
#[derive(Clone, Debug)]
pub struct MaxPool2d {
    channels: usize,
    in_h: usize,
    in_w: usize,
    size: usize,
}

impl MaxPool2d {
    pub fn new(channels: usize, in_h: usize, in_w: usize, size: usize) -> Self {
        Self::with_shape(channels, in_h, in_w, size).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [channels, in_h, in_w, size] = parse_config(config, "MaxPool2d")?;
        Self::with_shape(channels, in_h, in_w, size)
    }

    fn with_shape(channels: usize, in_h: usize, in_w: usize, size: usize) -> Result<Self> {
        if channels == 0 || size == 0 || in_h < size || in_w < size {
            return Err(Error::format(
                "MaxPool2d window must fit inside a non-empty input",
            ));
        }
        Ok(Self {
            channels,
            in_h,
            in_w,
            size,
        })
    }

    fn out_h(&self) -> usize {
        self.in_h / self.size
    }

    fn out_w(&self) -> usize {
        self.in_w / self.size
    }

    fn argmax(&self, input: &[f32], c: usize, oy: usize, ox: usize) -> usize {
        let plane = c * self.in_h * self.in_w;
        let mut best = plane + oy * self.size * self.in_w + ox * self.size;
        for dy in 0..self.size {
            for dx in 0..self.size {
                let i = plane + (oy * self.size + dy) * self.in_w + ox * self.size + dx;
                if input[i] > input[best] {
                    best = i;
                }
            }
        }
        best
    }
}
