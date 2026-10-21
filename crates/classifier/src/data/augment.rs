use super::{CHANNELS, IMAGE_LEN, IMAGE_SIZE};
use crate::nn::Rng;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Augment {
    pub flip: bool,
    pub dx: i32,
    pub dy: i32,
}

impl Augment {
    pub const NONE: Augment = Augment {
        flip: false,
        dx: 0,
        dy: 0,
    };

    pub fn random(rng: &mut Rng) -> Self {
        Self {
            flip: rng.next_u64() & 1 == 1,
            dx: rng.below(5) as i32 - 2,
            dy: rng.below(5) as i32 - 2,
        }
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
    fn quarter_turn_rotates_corners() {
        let mut img = vec![0u8; IMAGE_LEN];
        img[0] = 255;
        let rotated = rotate(&img, 90.0);
        assert_eq!(rotated[(IMAGE_SIZE - 1) * IMAGE_SIZE], 255);
    }
}
