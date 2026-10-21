use crate::data::{self, CHANNELS, IMAGE_SIZE};
use crate::image::{Preprocessor, RgbImage};
use crate::nn::Rng;
use crate::predictor::Predictor;

const INFERNO: [[f32; 3]; 5] = [
    [0.0, 0.0, 4.0],
    [87.0, 16.0, 110.0],
    [188.0, 55.0, 84.0],
    [249.0, 142.0, 9.0],
    [252.0, 255.0, 164.0],
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Saliency {
    pub samples: usize,
    pub noise: f32,
    pub seed: u64,
}

impl Default for Saliency {
    fn default() -> Self {
        Self {
            samples: 32,
            noise: 0.15,
            seed: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeatMap {
    pub size: usize,
    pub values: Vec<f32>,
}

impl Saliency {
    pub fn heat_map<P: Predictor + ?Sized>(
        &self,
        model: &P,
        preprocessor: &Preprocessor,
        image: &RgbImage,
        class: usize,
    ) -> HeatMap {
        let thumb = preprocessor.thumbnail(image);
        let mut views = vec![(thumb.to_planar(), false)];
        if preprocessor.test_time_augmentation {
            views.push((thumb.flip_horizontal().to_planar(), true));
        }
        let plane = IMAGE_SIZE * IMAGE_SIZE;
        let mut importance = vec![0.0; plane];
        let mut rng = Rng::new(self.seed);
        for (raw, flipped) in views {
            let input = data::preprocess(&raw);
            let (low, high) = input
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            let sigma = self.noise * (high - low).max(0.0);
            for _ in 0..self.samples.max(1) {
                let noisy: Vec<f32> = input.iter().map(|&x| x + sigma * rng.normal()).collect();
                let grad = model.logit_gradient(&noisy, class);
                for (p, value) in importance.iter_mut().enumerate() {
                    let (y, x) = (p / IMAGE_SIZE, p % IMAGE_SIZE);
                    let source = if flipped {
                        y * IMAGE_SIZE + IMAGE_SIZE - 1 - x
                    } else {
                        p
                    };
                    let contribution: f32 = (0..CHANNELS)
                        .map(|c| grad[c * plane + source] * input[c * plane + source])
                        .sum();
                    *value += contribution.abs();
                }
            }
        }
        HeatMap::normalised(IMAGE_SIZE, blur(IMAGE_SIZE, &importance))
    }
}

impl HeatMap {
    pub fn normalised(size: usize, raw: Vec<f32>) -> Self {
        assert_eq!(raw.len(), size * size, "heat map has the wrong size");
        let mut sorted: Vec<f32> = raw.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(f32::total_cmp);
        let ceiling = sorted
            .get(sorted.len() * 99 / 100)
            .or(sorted.last())
            .copied()
            .unwrap_or(0.0);
        let values = raw
            .iter()
            .map(|&v| {
                if ceiling > 0.0 && v.is_finite() {
                    (v / ceiling).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect();
        Self { size, values }
    }

    pub fn render(&self, base: &RgbImage, scale: usize) -> RgbImage {
        assert!(scale > 0, "scale must be at least 1");
        assert_eq!(
            (base.width, base.height),
            (self.size, self.size),
            "the heat map and image sizes differ"
        );
        let side = self.size * scale;
        let to_map = |o: usize| (o as f32 + 0.5) / scale as f32 - 0.5;
        let mut pixels = Vec::with_capacity(side * side * 3);
        for oy in 0..side {
            for ox in 0..side {
                let p = ((oy / scale) * base.width + ox / scale) * 3;
                let rgb = &base.pixels[p..p + 3];
                let grey = 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32;
                let heat = self.sample(to_map(ox), to_map(oy));
                let colour = ramp(heat);
                let alpha = 0.9 * heat.powf(1.5);
                for c in 0..3 {
                    let muted = 0.75 * (0.3 * rgb[c] as f32 + 0.7 * grey);
                    let value = muted * (1.0 - alpha) + colour[c] * alpha;
                    pixels.push(value.round().clamp(0.0, 255.0) as u8);
                }
            }
        }
        RgbImage::new(side, side, pixels)
    }

    fn sample(&self, x: f32, y: f32) -> f32 {
        let max = (self.size - 1) as f32;
        let (x, y) = (x.clamp(0.0, max), y.clamp(0.0, max));
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.size - 1), (y0 + 1).min(self.size - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |x: usize, y: usize| self.values[y * self.size + x];
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

pub fn ramp(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0) * (INFERNO.len() - 1) as f32;
    let i = (t.floor() as usize).min(INFERNO.len() - 2);
    let f = t - i as f32;
    std::array::from_fn(|c| INFERNO[i][c] * (1.0 - f) + INFERNO[i + 1][c] * f)
}

fn blur(size: usize, values: &[f32]) -> Vec<f32> {
    const KERNEL: [f32; 3] = [0.25, 0.5, 0.25];
    let pass = |values: &[f32], horizontal: bool| -> Vec<f32> {
        (0..size * size)
            .map(|p| {
                let (y, x) = ((p / size) as isize, (p % size) as isize);
                let (mut sum, mut weight) = (0.0, 0.0);
                for (k, w) in KERNEL.iter().enumerate() {
                    let d = k as isize - 1;
                    let (sy, sx) = if horizontal { (y, x + d) } else { (y + d, x) };
                    if (0..size as isize).contains(&sx) && (0..size as isize).contains(&sy) {
                        sum += w * values[sy as usize * size + sx as usize];
                        weight += w;
                    }
                }
                sum / weight
            })
            .collect()
    };
    pass(&pass(values, true), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::IMAGE_LEN;
    use crate::nn::{Dense, Network};

    fn left_edge_model() -> Network {
        let mut net = Network::new(vec!["a".into(), "b".into()]);
        net.push(Dense::new(IMAGE_LEN, 2, &mut Rng::new(1)));
        let params = net.layers_mut()[0].params_mut();
        for (i, w) in params[..IMAGE_LEN].iter_mut().enumerate() {
            *w = if i % IMAGE_SIZE < IMAGE_SIZE / 4 {
                1.0
            } else {
                0.0
            };
        }
        net
    }

    fn noise_image() -> RgbImage {
        let mut rng = Rng::new(9);
        let pixels = (0..IMAGE_LEN).map(|_| rng.below(256) as u8).collect();
        RgbImage::new(IMAGE_SIZE, IMAGE_SIZE, pixels)
    }

    fn column_sum(map: &HeatMap, x: usize) -> f32 {
        (0..map.size).map(|y| map.values[y * map.size + x]).sum()
    }

    #[test]
    fn heat_map_follows_the_weights_and_mirrors_tta() {
        let net = left_edge_model();
        let image = noise_image();
        let saliency = Saliency {
            samples: 2,
            ..Default::default()
        };
        let plain = saliency.heat_map(&net, &Preprocessor::default(), &image, 0);
        assert_eq!(plain.values.len(), IMAGE_SIZE * IMAGE_SIZE);
        assert!(plain.values.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(column_sum(&plain, 2) > 5.0);
        assert_eq!(column_sum(&plain, IMAGE_SIZE - 3), 0.0);

        let tta = Preprocessor {
            test_time_augmentation: true,
            ..Default::default()
        };
        let both = saliency.heat_map(&net, &tta, &image, 0);
        assert!(column_sum(&both, 2) > 5.0);
        assert!(column_sum(&both, IMAGE_SIZE - 3) > 5.0);
        assert_eq!(column_sum(&both, IMAGE_SIZE / 2), 0.0);
    }

    #[test]
    fn normalises_into_unit_range() {
        let map = HeatMap::normalised(2, vec![0.0, 1.0, 4.0, f32::NAN]);
        assert!(map.values.iter().all(|v| (0.0..=1.0).contains(v)));
        assert_eq!(map.values[2], 1.0);
        assert_eq!(map.values[3], 0.0);
        assert_eq!(HeatMap::normalised(1, vec![0.0]).values, vec![0.0]);
    }

    #[test]
    fn renders_an_upscaled_blend() {
        let base = noise_image();
        let cold = HeatMap::normalised(IMAGE_SIZE, vec![0.0; IMAGE_SIZE * IMAGE_SIZE]);
        let cold_image = cold.render(&base, 8);
        assert_eq!((cold_image.width, cold_image.height), (256, 256));
        assert_eq!(cold_image.pixels.len(), 256 * 256 * 3);
        assert!(cold_image.pixels.iter().all(|&v| v <= 192));

        let hot = HeatMap {
            size: IMAGE_SIZE,
            values: vec![1.0; IMAGE_SIZE * IMAGE_SIZE],
        };
        let hot_image = hot.render(&base, 3);
        assert_eq!((hot_image.width, hot_image.height), (96, 96));
        let mean = |c: usize| {
            hot_image
                .pixels
                .iter()
                .skip(c)
                .step_by(3)
                .map(|&v| v as f32)
                .sum::<f32>()
                / (96.0 * 96.0)
        };
        assert!(
            mean(0) > 200.0 && mean(0) > mean(2),
            "{} {}",
            mean(0),
            mean(2)
        );
    }

    #[test]
    fn ramp_runs_from_dark_to_bright() {
        let brightness = |t: f32| ramp(t).iter().sum::<f32>();
        assert!(brightness(0.0) < 10.0);
        assert!(brightness(1.0) > 600.0);
        assert!((0..10).all(|i| brightness(i as f32 / 10.0) < brightness((i + 1) as f32 / 10.0)));
    }
}
