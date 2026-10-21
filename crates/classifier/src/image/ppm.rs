use super::RgbImage;
use crate::error::{Error, Result};

impl RgbImage {
    pub fn decode_ppm(bytes: &[u8]) -> Result<Self> {
        let binary = bytes.starts_with(b"P6");
        let mut pos = 2;
        let width = read_ppm_number(bytes, &mut pos)? as usize;
        let height = read_ppm_number(bytes, &mut pos)? as usize;
        let max = read_ppm_number(bytes, &mut pos)?;
        if width == 0 || height == 0 || !(1..=65535).contains(&max) {
            return Err(Error::format("invalid PPM header"));
        }
        let samples = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| Error::format("PPM image is too large"))?;
        let scale = |v: u32| {
            if v > max {
                return Err(Error::format("PPM sample exceeds the maximum value"));
            }
            Ok(((v * 255 + max / 2) / max) as u8)
        };

        let pixels = if binary {
            pos += 1;
            let bytes_per_sample = if max < 256 { 1 } else { 2 };
            let data = samples
                .checked_mul(bytes_per_sample)
                .and_then(|len| pos.checked_add(len))
                .and_then(|end| bytes.get(pos..end))
                .ok_or_else(|| Error::format("PPM pixel data is truncated"))?;
            if bytes_per_sample == 1 {
                data.iter()
                    .map(|&v| scale(v as u32))
                    .collect::<Result<Vec<_>>>()?
            } else {
                data.chunks_exact(2)
                    .map(|c| scale(u16::from_be_bytes([c[0], c[1]]) as u32))
                    .collect::<Result<Vec<_>>>()?
            }
        } else {
            let mut pixels = Vec::with_capacity(samples.min(bytes.len()));
            for _ in 0..samples {
                pixels.push(scale(read_ppm_number(bytes, &mut pos)?)?);
            }
            pixels
        };
        Ok(Self::new(width, height, pixels))
    }

    pub fn encode_ppm(&self) -> Vec<u8> {
        let mut out = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
        out.extend_from_slice(&self.pixels);
        out
    }
}

fn read_ppm_number(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    while let Some(&b) = bytes.get(*pos) {
        if b == b'#' {
            while bytes.get(*pos).is_some_and(|&b| b != b'\n') {
                *pos += 1;
            }
        } else if b.is_ascii_whitespace() {
            *pos += 1;
        } else {
            break;
        }
    }
    let start = *pos;
    let mut value: u32 = 0;
    while let Some(&b) = bytes.get(*pos).filter(|b| b.is_ascii_digit()) {
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add((b - b'0') as u32))
            .ok_or_else(|| Error::format("number in PPM file is too large"))?;
        *pos += 1;
    }
    if *pos == start {
        return Err(Error::format("malformed PPM file"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RgbImage {
        RgbImage::new(2, 2, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30])
    }

    #[test]
    fn ppm_round_trip() {
        let img = sample();
        assert_eq!(RgbImage::decode(&img.encode_ppm()).unwrap(), img);
    }

    #[test]
    fn ascii_ppm_with_comments_and_maxval() {
        let text = b"P3\n# a comment\n2 1\n# another\n15\n15 0 0  0 15 0\n";
        let img = RgbImage::decode(text).unwrap();
        assert_eq!(img.pixels, vec![255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn binary_samples_above_maxval_are_rejected() {
        assert!(RgbImage::decode(b"P6\n1 1\n15\n\xff\x00\x00").is_err());
        let mut wide = b"P6\n1 1\n1000\n".to_vec();
        wide.extend_from_slice(&[0xff, 0xff, 0, 0, 0, 0]);
        assert!(RgbImage::decode(&wide).is_err());
    }

    #[test]
    fn decodes_16_bit_binary_ppm() {
        let mut bytes = b"P6\n1 1\n1000\n".to_vec();
        bytes.extend_from_slice(&[0x03, 0xe8, 0x01, 0xf4, 0, 0]);
        assert_eq!(RgbImage::decode(&bytes).unwrap().pixels, vec![255, 128, 0]);
    }

    #[test]
    fn huge_dimensions_are_rejected_without_allocating() {
        assert!(RgbImage::decode(b"P6\n60000 60000\n255\n\x00\x00\x00").is_err());
        assert!(RgbImage::decode(b"P3\n60000 60000\n255\n0 0 0").is_err());
    }
}
