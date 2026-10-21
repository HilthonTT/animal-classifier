use super::RgbImage;
use crate::error::{Error, Result};

impl RgbImage {
    pub fn decode_bmp(bytes: &[u8]) -> Result<Self> {
        let truncated = || Error::format("BMP file is truncated");
        let u16_at = |o: usize| {
            bytes
                .get(o..o + 2)
                .map(|s| u16::from_le_bytes([s[0], s[1]]))
                .ok_or_else(truncated)
        };
        let u32_at = |o: usize| {
            bytes
                .get(o..o + 4)
                .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
                .ok_or_else(truncated)
        };

        let data_offset = u32_at(10)? as usize;
        let header_size = u32_at(14)?;
        let width = u32_at(18)? as i32;
        let height = u32_at(22)? as i32;
        let bits = u16_at(28)?;
        let compression = u32_at(30)?;
        if header_size < 40 {
            return Err(Error::format(
                "unsupported BMP header (OS/2 bitmaps are not supported)",
            ));
        }
        let supported = (bits == 24 && compression == 0)
            || (bits == 32 && (compression == 0 || compression == 3));
        if !supported {
            return Err(Error::format(format!(
                "unsupported BMP ({bits}-bit, compression {compression}); save as 24-bit uncompressed"
            )));
        }
        if width <= 0 || height == 0 {
            return Err(Error::format("invalid BMP dimensions"));
        }

        let channels = if compression == 3 {
            let mut offsets = [0; 3];
            for (c, offset) in offsets.iter_mut().enumerate() {
                let mask = u32_at(54 + 4 * c)?;
                *offset = match mask {
                    0x0000_00FF => 0,
                    0x0000_FF00 => 1,
                    0x00FF_0000 => 2,
                    0xFF00_0000 => 3,
                    _ => {
                        return Err(Error::format(format!(
                            "unsupported BMP colour mask {mask:#010x}; save as 24-bit uncompressed"
                        )));
                    }
                };
            }
            offsets
        } else {
            [2, 1, 0]
        };

        let (width, top_down) = (width as usize, height < 0);
        let height = height.unsigned_abs() as usize;
        let bytes_per_pixel = bits as usize / 8;
        let stride = (width * bytes_per_pixel).div_ceil(4) * 4;
        let end = data_offset
            .checked_add(stride * height)
            .ok_or_else(truncated)?;
        let data = bytes.get(data_offset..end).ok_or_else(truncated)?;

        let mut pixels = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            let row = if top_down { y } else { height - 1 - y };
            for px in data[row * stride..]
                .chunks_exact(bytes_per_pixel)
                .take(width)
            {
                pixels.extend_from_slice(&[px[channels[0]], px[channels[1]], px[channels[2]]]);
            }
        }
        Ok(Self::new(width, height, pixels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(width: i32, height: i32, bits: u16, compression: u32, data_offset: u32) -> Vec<u8> {
        let mut bmp = vec![0u8; 54];
        bmp[..2].copy_from_slice(b"BM");
        bmp[10..14].copy_from_slice(&data_offset.to_le_bytes());
        bmp[14..18].copy_from_slice(&40u32.to_le_bytes());
        bmp[18..22].copy_from_slice(&width.to_le_bytes());
        bmp[22..26].copy_from_slice(&height.to_le_bytes());
        bmp[28..30].copy_from_slice(&bits.to_le_bytes());
        bmp[30..34].copy_from_slice(&compression.to_le_bytes());
        bmp
    }

    #[test]
    fn decodes_bottom_up_24_bit_bmp() {
        let mut bmp = header(2, 2, 24, 0, 54);
        bmp.extend_from_slice(&[255, 0, 0, 10, 20, 30, 0, 0]);
        bmp.extend_from_slice(&[0, 0, 255, 0, 255, 0, 0, 0]);
        let img = RgbImage::decode(&bmp).unwrap();
        assert_eq!(
            img.pixels,
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 30, 20, 10]
        );
    }

    #[test]
    fn decodes_top_down_32_bit_bmp() {
        let mut bmp = header(2, -1, 32, 0, 54);
        bmp.extend_from_slice(&[1, 2, 3, 255, 4, 5, 6, 255]);
        assert_eq!(
            RgbImage::decode(&bmp).unwrap().pixels,
            vec![3, 2, 1, 6, 5, 4]
        );
    }

    #[test]
    fn honours_bitfield_masks() {
        let mut bmp = header(1, 1, 32, 3, 66);
        for mask in [0x0000_00FFu32, 0x0000_FF00, 0x00FF_0000] {
            bmp.extend_from_slice(&mask.to_le_bytes());
        }
        bmp.extend_from_slice(&[10, 20, 30, 255]);
        assert_eq!(RgbImage::decode(&bmp).unwrap().pixels, vec![10, 20, 30]);

        let mut odd = header(1, 1, 32, 3, 66);
        for mask in [0x0000_0F00u32, 0x0000_00F0, 0x0000_000F] {
            odd.extend_from_slice(&mask.to_le_bytes());
        }
        odd.extend_from_slice(&[0; 4]);
        assert!(RgbImage::decode(&odd).is_err());
    }
}
