use super::{CHANNELS, IMAGE_LEN, IMAGE_SIZE};
use crate::nn::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cutout {
    pub x: i32,
    pub y: i32,
    pub size: i32,
}

impl Cutout {
    pub fn covers(&self, x: i32, y: i32) -> bool {
        let (x0, y0) = (self.x - self.size / 2, self.y - self.size / 2);
        (x0..x0 + self.size).contains(&x) && (y0..y0 + self.size).contains(&y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Augment {
    pub flip: bool,
    pub dx: i32,
    pub dy: i32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub cutout: Option<Cutout>,
}

impl Default for Augment {
    fn default() -> Self {
        Self::NONE
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AugmentOptions {
    pub color_jitter: f32,
    pub cutout: usize,
}

impl Augment {
    pub const NONE: Augment = Augment {
        flip: false,
        dx: 0,
        dy: 0,
        brightness: 1.0,
        contrast: 1.0,
        saturation: 1.0,
        cutout: None,
    };

    pub fn random(rng: &mut Rng) -> Self {
        Self::random_with(rng, &AugmentOptions::default())
    }

    pub fn random_with(rng: &mut Rng, options: &AugmentOptions) -> Self {
        let mut augment = Self {
            flip: rng.next_u64() & 1 == 1,
            dx: rng.below(5) as i32 - 2,
            dy: rng.below(5) as i32 - 2,
            ..Self::NONE
        };
        let jitter = options.color_jitter;
        if jitter > 0.0 {
            let mut factor = || 1.0 + jitter * (2.0 * rng.next_f32() - 1.0);
            augment.brightness = factor();
            augment.contrast = factor();
            augment.saturation = factor();
        }
        if options.cutout > 0 {
            augment.cutout = Some(Cutout {
                x: rng.below(IMAGE_SIZE) as i32,
                y: rng.below(IMAGE_SIZE) as i32,
                size: options.cutout as i32,
            });
        }
        augment
    }

    pub fn adjusts_colour(&self) -> bool {
        self.brightness != 1.0 || self.contrast != 1.0 || self.saturation != 1.0
    }

    pub fn adjust_colour(&self, rgb: [f32; 3]) -> [f32; 3] {
        let [r, g, b] = rgb.map(|v| (v * self.brightness - 0.5) * self.contrast + 0.5);
        let grey = 0.299 * r + 0.587 * g + 0.114 * b;
        [r, g, b].map(|v| (grey + self.saturation * (v - grey)).clamp(0.0, 1.0))
    }
}

pub fn flip_horizontal(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(IMAGE_LEN);
    for row in raw.chunks_exact(IMAGE_SIZE) {
        out.extend(row.iter().rev());
    }
    out
}

pub fn translate(raw: &[u8], dx: i32, dy: i32) -> Vec<u8> {
    sample(raw, |x, y| ((x - dx as f32), (y - dy as f32)))
}

pub fn rotate(raw: &[u8], degrees: f32) -> Vec<u8> {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let c = (IMAGE_SIZE as f32 - 1.0) / 2.0;
    sample(raw, |x, y| {
        let (u, v) = (x - c, y - c);
        (cos * u - sin * v + c, sin * u + cos * v + c)
    })
}

fn sample(raw: &[u8], source: impl Fn(f32, f32) -> (f32, f32)) -> Vec<u8> {
    let n = IMAGE_SIZE;
    let max = (n - 1) as f32;
    let mut out = vec![0u8; IMAGE_LEN];
    for y in 0..n {
        for x in 0..n {
            let (sx, sy) = source(x as f32, y as f32);
            let (sx, sy) = (sx.clamp(0.0, max), sy.clamp(0.0, max));
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(n - 1), (y0 + 1).min(n - 1));
            let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
            for c in 0..CHANNELS {
                let p = |xx: usize, yy: usize| raw[c * n * n + yy * n + xx] as f32;
                let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
                let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
                out[c * n * n + y * n + x] = (top * (1.0 - fy) + bottom * fy).round() as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_image() -> Vec<u8> {
        (0..IMAGE_LEN).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn identity_transforms_change_nothing() {
        let img = gradient_image();
        assert_eq!(translate(&img, 0, 0), img);
        assert_eq!(rotate(&img, 0.0), img);
        assert_eq!(flip_horizontal(&flip_horizontal(&img)), img);
    }

    #[test]
    fn translate_moves_pixels() {
        let mut img = vec![0u8; IMAGE_LEN];
        img[5 * IMAGE_SIZE + 5] = 200;
        let moved = translate(&img, 2, 1);
        assert_eq!(moved[6 * IMAGE_SIZE + 7], 200);
        assert_eq!(moved[5 * IMAGE_SIZE + 5], 0);
    }

    #[test]
    fn random_augment_respects_its_options() {
        let mut rng = Rng::new(2);
        let plain = Augment::random(&mut rng);
        assert!(!plain.adjusts_colour() && plain.cutout.is_none());
        let options = AugmentOptions {
            color_jitter: 0.4,
            cutout: 8,
        };
        for _ in 0..100 {
            let a = Augment::random_with(&mut rng, &options);
            for f in [a.brightness, a.contrast, a.saturation] {
                assert!((0.6..=1.4).contains(&f), "{f}");
            }
            assert_eq!(a.cutout.map(|c| c.size), Some(8));
        }
    }

    #[test]
    fn colour_adjustments_behave() {
        let grey = Augment {
            saturation: 0.0,
            ..Augment::NONE
        };
        let [r, g, b] = grey.adjust_colour([1.0, 0.0, 0.0]);
        assert!((r - g).abs() < 1e-6 && (g - b).abs() < 1e-6);
        let bright = Augment {
            brightness: 2.0,
            ..Augment::NONE
        };
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(close(
            bright.adjust_colour([0.25, 0.75, 0.5]),
            [0.5, 1.0, 1.0]
        ));
        assert!(close(
            Augment::NONE.adjust_colour([0.2, 0.4, 0.6]),
            [0.2, 0.4, 0.6]
        ));
    }

    #[test]
    fn cutout_covers_a_square() {
        let cut = Cutout {
            x: 5,
            y: 5,
            size: 4,
        };
        assert!(cut.covers(3, 3) && cut.covers(6, 6));
        assert!(!cut.covers(7, 5) && !cut.covers(5, 2));
    }

    #[test]
    fn quarter_turn_rotates_corners() {
        let mut img = vec![0u8; IMAGE_LEN];
        img[0] = 255;
        let rotated = rotate(&img, 90.0);
        assert_eq!(rotated[(IMAGE_SIZE - 1) * IMAGE_SIZE], 255);
    }
}
