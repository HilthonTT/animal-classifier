pub mod augment;
pub mod cifar;
pub mod expand;

pub use augment::Augment;

use crate::nn::Rng;

pub const IMAGE_SIZE: usize = 32;
pub const CHANNELS: usize = 3;
pub const IMAGE_LEN: usize = CHANNELS * IMAGE_SIZE * IMAGE_SIZE;

pub const CHANNEL_MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];
pub const CHANNEL_STD: [f32; 3] = [0.2470, 0.2435, 0.2616];

#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub class_names: Vec<String>,
}

impl Dataset {
    pub fn new(class_names: Vec<String>) -> Self {
        Self {
            images: Vec::new(),
            labels: Vec::new(),
            class_names,
        }
    }

    pub fn push(&mut self, image: &[u8], label: u8) {
        assert_eq!(
            image.len(),
            IMAGE_LEN,
            "image must be {IMAGE_SIZE}x{IMAGE_SIZE} RGB"
        );
        assert!(
            (label as usize) < self.class_names.len(),
            "label out of range"
        );
        self.images.extend_from_slice(image);
        self.labels.push(label);
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn truncate(&mut self, n: usize) {
        self.labels.truncate(n);
        self.images.truncate(self.labels.len() * IMAGE_LEN);
    }

    pub fn image(&self, index: usize) -> &[u8] {
        &self.images[index * IMAGE_LEN..(index + 1) * IMAGE_LEN]
    }

    pub fn class_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.class_names.len()];
        for &label in &self.labels {
            counts[label as usize] += 1;
        }
        counts
    }

    pub fn shuffled(&self, rng: &mut Rng) -> Dataset {
        let mut order: Vec<usize> = (0..self.len()).collect();
        rng.shuffle(&mut order);
        let mut out = Dataset::new(self.class_names.clone());
        for i in order {
            out.push(self.image(i), self.labels[i]);
        }
        out
    }

    pub fn split_validation(&self, fraction: f32, rng: &mut Rng) -> (Dataset, Dataset) {
        let mut train = Dataset::new(self.class_names.clone());
        let mut validation = Dataset::new(self.class_names.clone());
        for class in 0..self.class_names.len() {
            let mut indices: Vec<usize> = (0..self.len())
                .filter(|&i| self.labels[i] as usize == class)
                .collect();
            rng.shuffle(&mut indices);
            let held_out = (indices.len() as f32 * fraction.clamp(0.0, 1.0)).round() as usize;
            for (n, &i) in indices.iter().enumerate() {
                let target = if n < held_out {
                    &mut validation
                } else {
                    &mut train
                };
                target.push(self.image(i), self.labels[i]);
            }
        }
        (train.shuffled(rng), validation)
    }

    pub fn sample_into(&self, index: usize, augment: Augment, out: &mut [f32]) {
        normalize_into(self.image(index), augment, out);
    }
}

pub fn normalize_into(raw: &[u8], augment: Augment, out: &mut [f32]) {
    let n = IMAGE_SIZE as i32;
    let plane_len = IMAGE_SIZE * IMAGE_SIZE;
    for c in 0..CHANNELS {
        let plane = &raw[c * plane_len..(c + 1) * plane_len];
        let (mean, std) = (CHANNEL_MEAN[c], CHANNEL_STD[c]);
        for y in 0..n {
            for x in 0..n {
                let sx = if augment.flip { n - 1 - x } else { x } + augment.dx;
                let sy = y + augment.dy;
                let value = if (0..n).contains(&sx) && (0..n).contains(&sy) {
                    (plane[(sy * n + sx) as usize] as f32 / 255.0 - mean) / std
                } else {
                    0.0
                };
                out[c * plane_len + (y * n + x) as usize] = value;
            }
        }
    }
}

pub fn preprocess(raw: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0; IMAGE_LEN];
    normalize_into(raw, Augment::NONE, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_mirrors_rows() {
        let mut raw = vec![0u8; IMAGE_LEN];
        raw[0] = 255;
        let mut out = vec![0.0; IMAGE_LEN];
        normalize_into(
            &raw,
            Augment {
                flip: true,
                dx: 0,
                dy: 0,
            },
            &mut out,
        );
        let bright = (1.0 - CHANNEL_MEAN[0]) / CHANNEL_STD[0];
        assert!((out[IMAGE_SIZE - 1] - bright).abs() < 1e-5);
        assert!(out[0] < 0.0);
    }

    #[test]
    fn shift_fills_with_zero() {
        let raw = vec![255u8; IMAGE_LEN];
        let mut out = vec![1.0; IMAGE_LEN];
        normalize_into(
            &raw,
            Augment {
                flip: false,
                dx: 2,
                dy: 0,
            },
            &mut out,
        );
        assert_eq!(out[IMAGE_SIZE - 1], 0.0);
        assert!(out[0] > 0.0);
    }

    #[test]
    fn shuffle_keeps_image_label_pairs() {
        let mut ds = Dataset::new(vec!["a".into(), "b".into()]);
        for label in [0u8, 1, 1, 0, 1] {
            ds.push(&[label * 100; IMAGE_LEN], label);
        }
        let shuffled = ds.shuffled(&mut Rng::new(3));
        assert_eq!(shuffled.class_counts(), vec![2, 3]);
        for i in 0..shuffled.len() {
            assert_eq!(shuffled.image(i)[0], shuffled.labels[i] * 100);
        }
    }

    #[test]
    fn validation_split_is_stratified_and_disjoint() {
        let mut ds = Dataset::new(vec!["a".into(), "b".into()]);
        for i in 0..40u8 {
            let label = u8::from(i >= 30);
            let mut image = [label; IMAGE_LEN];
            image[1] = i;
            ds.push(&image, label);
        }
        let (train, validation) = ds.split_validation(0.2, &mut Rng::new(1));
        assert_eq!(validation.class_counts(), vec![6, 2]);
        assert_eq!(train.class_counts(), vec![24, 8]);
        let mut ids: Vec<u8> = (0..train.len())
            .map(|i| train.image(i)[1])
            .chain((0..validation.len()).map(|i| validation.image(i)[1]))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..40).collect::<Vec<u8>>());
    }
}
