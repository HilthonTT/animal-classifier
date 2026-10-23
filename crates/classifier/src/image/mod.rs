use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

pub mod bmp;
mod inflate;
pub mod jpeg;
pub mod png;
pub mod ppm;
pub mod preprocess;

pub use preprocess::{Fit, Preprocessor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            width * height * 3,
            "pixel buffer has the wrong size"
        );
        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .map_err(|e| Error::format(format!("cannot read {}: {e}", path.display())))?;
        Self::decode(&bytes).map_err(|e| Error::format(format!("{}: {e}", path.display())))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.starts_with(&png::SIGNATURE[..4]) {
            return Self::decode_png(bytes);
        }
        if bytes.starts_with(&jpeg::SIGNATURE) {
            return Self::decode_jpeg(bytes);
        }
        match bytes.get(..2) {
            Some(b"P6") | Some(b"P3") => Self::decode_ppm(bytes),
            Some(b"BM") => Self::decode_bmp(bytes),
            _ => Err(Error::format(
                "unsupported image format: only JPEG, PNG, PPM and BMP can be read without external libraries. \
                 Convert with e.g. `magick photo.webp photo.png`",
            )),
        }
    }

    pub fn from_planar(width: usize, height: usize, planar: &[u8]) -> Self {
        let plane = width * height;
        assert_eq!(planar.len(), plane * 3, "planar buffer has the wrong size");
        let pixels = (0..plane)
            .flat_map(|i| [planar[i], planar[plane + i], planar[2 * plane + i]])
            .collect();
        Self::new(width, height, pixels)
    }

    pub fn to_planar(&self) -> Vec<u8> {
        (0..3)
            .flat_map(|c| self.pixels.iter().skip(c).step_by(3).copied())
            .collect()
    }

    pub fn center_crop_square(&self) -> Self {
        let side = self.width.min(self.height);
        let (x0, y0) = ((self.width - side) / 2, (self.height - side) / 2);
        let mut pixels = Vec::with_capacity(side * side * 3);
        for y in y0..y0 + side {
            let start = (y * self.width + x0) * 3;
            pixels.extend_from_slice(&self.pixels[start..start + side * 3]);
        }
        Self::new(side, side, pixels)
    }

    pub fn pad_square(&self, colour: [u8; 3]) -> Self {
        let side = self.width.max(self.height);
        let (x0, y0) = ((side - self.width) / 2, (side - self.height) / 2);
        let mut pixels: Vec<u8> = colour
            .iter()
            .copied()
            .cycle()
            .take(side * side * 3)
            .collect();
        for y in 0..self.height {
            let src = y * self.width * 3;
            let dst = ((y0 + y) * side + x0) * 3;
            pixels[dst..dst + self.width * 3]
                .copy_from_slice(&self.pixels[src..src + self.width * 3]);
        }
        Self::new(side, side, pixels)
    }

    pub fn flip_horizontal(&self) -> Self {
        let mut pixels = Vec::with_capacity(self.pixels.len());
        for row in self.pixels.chunks_exact(self.width * 3) {
            for px in row.chunks_exact(3).rev() {
                pixels.extend_from_slice(px);
            }
        }
        Self::new(self.width, self.height, pixels)
    }

    pub fn auto_contrast(&self) -> Self {
        let mut histogram = [0usize; 256];
        for &v in &self.pixels {
            histogram[v as usize] += 1;
        }
        let clip = self.pixels.len() / 100;
        let percentile = |order: &[usize]| {
            let mut seen = 0;
            order
                .iter()
                .copied()
                .find(|&v| {
                    seen += histogram[v];
                    seen > clip
                })
                .unwrap_or(0)
        };
        let levels: Vec<usize> = (0..256).collect();
        let low = percentile(&levels);
        let reversed: Vec<usize> = levels.iter().rev().copied().collect();
        let high = percentile(&reversed);
        if high <= low + 10 {
            return self.clone();
        }
        let scale = 255.0 / (high - low) as f32;
        let pixels = self
            .pixels
            .iter()
            .map(|&v| ((v as f32 - low as f32) * scale).round().clamp(0.0, 255.0) as u8)
            .collect();
        Self::new(self.width, self.height, pixels)
    }

    pub fn resize(&self, width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0, "target size must be non-zero");
        let span = |o: usize, src: usize, dst: usize| {
            let start = o * src / dst;
            let end = ((o + 1) * src).div_ceil(dst).clamp(start + 1, src);
            start..end
        };
        let mut pixels = Vec::with_capacity(width * height * 3);
        for oy in 0..height {
            let ys = span(oy, self.height, height);
            for ox in 0..width {
                let xs = span(ox, self.width, width);
                let mut sum = [0u64; 3];
                for y in ys.clone() {
                    for x in xs.clone() {
                        let p = (y * self.width + x) * 3;
                        for (s, &v) in sum.iter_mut().zip(&self.pixels[p..p + 3]) {
                            *s += v as u64;
                        }
                    }
                }
                let n = (ys.len() * xs.len()) as u64;
                pixels.extend(sum.iter().map(|&s| ((s + n / 2) / n) as u8));
            }
        }
        Self::new(width, height, pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RgbImage {
        RgbImage::new(2, 2, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30])
    }

    #[test]
    fn planar_round_trip() {
        let img = sample();
        let planar = img.to_planar();
        assert_eq!(planar, vec![255, 0, 0, 10, 0, 255, 0, 20, 0, 0, 255, 30]);
        assert_eq!(RgbImage::from_planar(2, 2, &planar), img);
    }

    #[test]
    fn resize_averages_blocks() {
        let img = RgbImage::new(2, 1, vec![0, 0, 0, 100, 200, 50]);
        assert_eq!(img.resize(1, 1).pixels, vec![50, 100, 25]);
        let up = img.resize(4, 2);
        assert_eq!(up.width, 4);
        assert_eq!(&up.pixels[..6], &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn crops_to_centre_square() {
        let img = RgbImage::new(3, 1, vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(img.center_crop_square().pixels, vec![2, 2, 2]);
    }

    #[test]
    fn pads_to_square_and_flips() {
        let img = RgbImage::new(2, 1, vec![1, 1, 1, 2, 2, 2]);
        let padded = img.pad_square([9, 9, 9]);
        assert_eq!((padded.width, padded.height), (2, 2));
        assert_eq!(padded.pixels, vec![1, 1, 1, 2, 2, 2, 9, 9, 9, 9, 9, 9]);
        assert_eq!(img.flip_horizontal().pixels, vec![2, 2, 2, 1, 1, 1]);
    }

    #[test]
    fn auto_contrast_stretches_range() {
        let pixels: Vec<u8> = (0..300).map(|i| 100 + (i % 51) as u8).collect();
        let stretched = RgbImage::new(10, 10, pixels).auto_contrast();
        assert_eq!(*stretched.pixels.iter().min().unwrap(), 0);
        assert_eq!(*stretched.pixels.iter().max().unwrap(), 255);
    }

    #[test]
    fn rejects_unknown_formats() {
        assert!(RgbImage::decode(b"\x89PNG....").is_err());
        assert!(RgbImage::decode(b"GIF89a").is_err());
        assert!(RgbImage::decode(b"P6\n2 2\n255\n\x00").is_err());
    }
}
