use super::RgbImage;
use super::inflate;
use crate::error::{Error, Result};

pub(crate) const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
const MAX_PIXELS: usize = 1 << 26;
const MAX_DEFLATE_RATIO: usize = 1032;
const ADAM7: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

pub(crate) fn crc32(data: &[u8]) -> u32 {
    !data.iter().fold(!0u32, |c, &b| {
        CRC_TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8)
    })
}

fn truncated() -> Error {
    Error::format("PNG file is truncated")
}

struct Header {
    width: usize,
    height: usize,
    depth: u8,
    colour: u8,
    interlaced: bool,
}

impl Header {
    fn parse(data: &[u8]) -> Result<Self> {
        let &[
            w0,
            w1,
            w2,
            w3,
            h0,
            h1,
            h2,
            h3,
            depth,
            colour,
            compression,
            filter,
            interlace,
        ] = data
        else {
            return Err(Error::format("PNG IHDR chunk has the wrong length"));
        };
        let width = u32::from_be_bytes([w0, w1, w2, w3]);
        let height = u32::from_be_bytes([h0, h1, h2, h3]);
        if width == 0 || height == 0 || width > i32::MAX as u32 || height > i32::MAX as u32 {
            return Err(Error::format(format!(
                "invalid PNG dimensions {width}x{height}"
            )));
        }
        let allowed: &[u8] = match colour {
            0 => &[1, 2, 4, 8, 16],
            3 => &[1, 2, 4, 8],
            2 | 4 | 6 => &[8, 16],
            _ => return Err(Error::format(format!("invalid PNG colour type {colour}"))),
        };
        if !allowed.contains(&depth) {
            return Err(Error::format(format!(
                "invalid PNG bit depth {depth} for colour type {colour}"
            )));
        }
        if compression != 0 || filter != 0 || interlace > 1 {
            return Err(Error::format(
                "unsupported PNG compression, filter or interlace method",
            ));
        }
        Ok(Self {
            width: width as usize,
            height: height as usize,
            depth,
            colour,
            interlaced: interlace == 1,
        })
    }

    fn channels(&self) -> usize {
        match self.colour {
            2 => 3,
            4 => 2,
            6 => 4,
            _ => 1,
        }
    }

    fn bits_per_pixel(&self) -> usize {
        self.channels() * self.depth as usize
    }

    fn stride(&self, width: usize) -> Option<usize> {
        width
            .checked_mul(self.bits_per_pixel())?
            .checked_add(7)
            .map(|bits| bits / 8 + 1)
    }

    fn passes(&self) -> Vec<(usize, usize, usize, usize, usize, usize)> {
        if !self.interlaced {
            return vec![(0, 0, 1, 1, self.width, self.height)];
        }
        ADAM7
            .iter()
            .map(|&(x0, y0, dx, dy)| {
                let size = |total: usize, start: usize, step: usize| {
                    total.saturating_sub(start).div_ceil(step)
                };
                (
                    x0,
                    y0,
                    dx,
                    dy,
                    size(self.width, x0, dx),
                    size(self.height, y0, dy),
                )
            })
            .filter(|&(.., w, h)| w > 0 && h > 0)
            .collect()
    }

    fn data_len(&self) -> Option<usize> {
        self.passes().iter().try_fold(0usize, |total, &(.., w, h)| {
            total.checked_add(self.stride(w)?.checked_mul(h)?)
        })
    }
}

fn next_chunk<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<([u8; 4], &'a [u8])> {
    let head = bytes.get(*pos..*pos + 8).ok_or_else(truncated)?;
    let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    let kind = [head[4], head[5], head[6], head[7]];
    let name = String::from_utf8_lossy(&kind).into_owned();
    if len > i32::MAX as u32 || !kind.iter().all(u8::is_ascii_alphabetic) {
        return Err(Error::format(format!(
            "corrupt PNG chunk header ({name:?})"
        )));
    }
    let start = *pos + 4;
    let end = start.checked_add(4 + len as usize).ok_or_else(truncated)?;
    let body = bytes.get(start..end).ok_or_else(truncated)?;
    let crc = bytes.get(end..end + 4).ok_or_else(truncated)?;
    if crc32(body) != u32::from_be_bytes([crc[0], crc[1], crc[2], crc[3]]) {
        return Err(Error::format(format!(
            "PNG chunk {name} is corrupt (CRC mismatch)"
        )));
    }
    *pos = end + 4;
    Ok((kind, &body[4..]))
}

fn unfilter(data: &mut [u8], stride: usize, bpp: usize) -> Result<()> {
    let zeros = vec![0u8; stride - 1];
    let mut previous: Option<usize> = None;
    for start in (0..data.len()).step_by(stride) {
        let (before, rest) = data.split_at_mut(start);
        let prior = previous.map_or(&zeros[..], |p| &before[p + 1..p + stride]);
        let (filter, row) = rest[..stride].split_at_mut(1);
        match filter[0] {
            0 => {}
            1 => {
                for i in bpp..row.len() {
                    row[i] = row[i].wrapping_add(row[i - bpp]);
                }
            }
            2 => {
                for (x, &up) in row.iter_mut().zip(prior) {
                    *x = x.wrapping_add(up);
                }
            }
            3 => {
                for i in 0..row.len() {
                    let left = if i >= bpp { row[i - bpp] } else { 0 };
                    row[i] = row[i].wrapping_add(((left as u16 + prior[i] as u16) / 2) as u8);
                }
            }
            4 => {
                for i in 0..row.len() {
                    let (left, up_left) = if i >= bpp {
                        (row[i - bpp], prior[i - bpp])
                    } else {
                        (0, 0)
                    };
                    row[i] = row[i].wrapping_add(paeth(left, prior[i], up_left));
                }
            }
            other => {
                return Err(Error::format(format!(
                    "corrupt PNG image data (unknown filter type {other})"
                )));
            }
        }
        previous = Some(start);
    }
    Ok(())
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = (
        (p - a as i16).abs(),
        (p - b as i16).abs(),
        (p - c as i16).abs(),
    );
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

struct Converter<'a> {
    depth: usize,
    channels: usize,
    colour: u8,
    max: u32,
    palette: &'a [[u8; 3]],
}

impl Converter<'_> {
    fn sample(&self, row: &[u8], index: usize) -> u32 {
        match self.depth {
            8 => row[index] as u32,
            16 => u16::from_be_bytes([row[2 * index], row[2 * index + 1]]) as u32,
            d => {
                let bit = index * d;
                (row[bit / 8] as u32 >> (8 - d - bit % 8)) & self.max
            }
        }
    }

    fn to_u8(&self, v: u32) -> u8 {
        match self.depth {
            8 => v as u8,
            16 => ((v * 2 + 257) / 514) as u8,
            _ => (v * 255 / self.max) as u8,
        }
    }

    fn over_white(&self, v: u32, alpha: u32) -> u8 {
        let max = self.max as u64;
        let blended = (v as u64 * alpha as u64 + max * (max - alpha as u64) + max / 2) / max;
        self.to_u8(blended as u32)
    }

    fn pixel(&self, row: &[u8], x: usize) -> [u8; 3] {
        let base = x * self.channels;
        let s = |c: usize| self.sample(row, base + c);
        match self.colour {
            0 => [self.to_u8(s(0)); 3],
            2 => [self.to_u8(s(0)), self.to_u8(s(1)), self.to_u8(s(2))],
            3 => self
                .palette
                .get(s(0) as usize)
                .copied()
                .unwrap_or([0, 0, 0]),
            4 => [self.over_white(s(0), s(1)); 3],
            _ => {
                let alpha = s(3);
                [
                    self.over_white(s(0), alpha),
                    self.over_white(s(1), alpha),
                    self.over_white(s(2), alpha),
                ]
            }
        }
    }
}

impl RgbImage {
    pub fn decode_png(bytes: &[u8]) -> Result<Self> {
        if !bytes.starts_with(&SIGNATURE) {
            return Err(Error::format(
                "corrupt PNG signature (was the file transferred in text mode?)",
            ));
        }
        let mut pos = SIGNATURE.len();
        let mut header: Option<Header> = None;
        let mut palette: Option<Vec<[u8; 3]>> = None;
        let mut idat = Vec::new();
        let mut seen_idat = false;
        loop {
            let (kind, data) = next_chunk(bytes, &mut pos)?;
            if header.is_none() && &kind != b"IHDR" {
                return Err(Error::format("PNG file does not start with an IHDR chunk"));
            }
            match &kind {
                b"IHDR" => {
                    if header.is_some() {
                        return Err(Error::format("PNG file has more than one IHDR chunk"));
                    }
                    header = Some(Header::parse(data)?);
                }
                b"PLTE" => {
                    if seen_idat || palette.is_some() {
                        return Err(Error::format("PNG PLTE chunk is out of place"));
                    }
                    if data.is_empty() || data.len() % 3 != 0 || data.len() > 768 {
                        return Err(Error::format("PNG PLTE chunk has an invalid length"));
                    }
                    palette = Some(data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect());
                }
                b"IDAT" => {
                    seen_idat = true;
                    idat.extend_from_slice(data);
                }
                b"IEND" => break,
                _ if kind[0] & 0x20 == 0 => {
                    return Err(Error::format(format!(
                        "unsupported critical PNG chunk {}",
                        String::from_utf8_lossy(&kind)
                    )));
                }
                _ => {}
            }
        }

        let header = header.ok_or_else(truncated)?;
        let (width, height) = (header.width, header.height);
        let palette = match (header.colour, palette) {
            (3, None) => return Err(Error::format("palette PNG has no PLTE chunk")),
            (3, Some(p)) => p,
            _ => Vec::new(),
        };
        if idat.is_empty() {
            return Err(Error::format("PNG file has no image data"));
        }
        let too_large = || {
            Error::format(format!(
                "PNG image is too large ({width}x{height}; at most {MAX_PIXELS} pixels are supported)"
            ))
        };
        let pixel_count = width
            .checked_mul(height)
            .filter(|&n| n <= MAX_PIXELS)
            .ok_or_else(too_large)?;
        let expected = header.data_len().ok_or_else(too_large)?;
        if expected / MAX_DEFLATE_RATIO > idat.len() {
            return Err(Error::format(format!(
                "PNG image data is too short for a {width}x{height} image"
            )));
        }

        let mut raw = inflate::zlib_decompress(&idat, expected)
            .map_err(|e| Error::format(format!("corrupt PNG image data: {e}")))?;
        if raw.len() != expected {
            return Err(Error::format("PNG image data is truncated"));
        }

        let converter = Converter {
            depth: header.depth as usize,
            channels: header.channels(),
            colour: header.colour,
            max: (1u32 << header.depth) - 1,
            palette: &palette,
        };
        let bpp = header.bits_per_pixel().div_ceil(8);
        let mut pixels = vec![0u8; pixel_count * 3];
        let mut offset = 0;
        for (x0, y0, dx, dy, w, h) in header.passes() {
            let stride = header.stride(w).ok_or_else(too_large)?;
            let pass = &mut raw[offset..offset + stride * h];
            offset += stride * h;
            unfilter(pass, stride, bpp)?;
            for (j, row) in pass.chunks_exact(stride).enumerate() {
                let row = &row[1..];
                let y = y0 + j * dy;
                for i in 0..w {
                    let dst = (y * width + x0 + i * dx) * 3;
                    pixels[dst..dst + 3].copy_from_slice(&converter.pixel(row, i));
                }
            }
        }
        Ok(Self::new(width, height, pixels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    }

    fn stored_zlib(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        let blocks: Vec<&[u8]> = data.chunks(1000).collect();
        for (i, block) in blocks.iter().enumerate() {
            out.push(u8::from(i + 1 == blocks.len()));
            let len = block.len() as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(block);
        }
        out.extend_from_slice(&inflate::adler32(data).to_be_bytes());
        out
    }

    fn filter_rows(raw_rows: &[Vec<u8>], bpp: usize, filter: impl Fn(usize) -> u8) -> Vec<u8> {
        let mut out = Vec::new();
        for (y, row) in raw_rows.iter().enumerate() {
            let zeros = vec![0u8; row.len()];
            let prior = if y == 0 { &zeros } else { &raw_rows[y - 1] };
            let kind = filter(y);
            out.push(kind);
            for i in 0..row.len() {
                let a = if i >= bpp { row[i - bpp] } else { 0 };
                let c = if i >= bpp { prior[i - bpp] } else { 0 };
                let b = prior[i];
                let predicted = match kind {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => ((a as u16 + b as u16) / 2) as u8,
                    _ => paeth(a, b, c),
                };
                out.push(row[i].wrapping_sub(predicted));
            }
        }
        out
    }

    struct Spec {
        width: u32,
        height: u32,
        depth: u8,
        colour: u8,
        interlaced: bool,
    }

    fn pack_row(samples: &[u16], depth: u8) -> Vec<u8> {
        match depth {
            8 => samples.iter().map(|&s| s as u8).collect(),
            16 => samples.iter().flat_map(|&s| s.to_be_bytes()).collect(),
            d => {
                let per_byte = 8 / d as usize;
                samples
                    .chunks(per_byte)
                    .map(|group| {
                        group.iter().enumerate().fold(0u8, |byte, (k, &s)| {
                            byte | ((s as u8) << (8 - d as usize * (k + 1)))
                        })
                    })
                    .collect()
            }
        }
    }

    fn encode(spec: &Spec, samples: &[Vec<u16>], palette: Option<&[u8]>) -> Vec<u8> {
        let header = Header {
            width: spec.width as usize,
            height: spec.height as usize,
            depth: spec.depth,
            colour: spec.colour,
            interlaced: spec.interlaced,
        };
        let channels = header.channels();
        let bpp = header.bits_per_pixel().div_ceil(8);
        let mut raw = Vec::new();
        for (filter_seed, (x0, y0, dx, dy, w, h)) in header.passes().into_iter().enumerate() {
            let rows: Vec<Vec<u8>> = (0..h)
                .map(|j| {
                    let y = y0 + j * dy;
                    let row: Vec<u16> = (0..w)
                        .flat_map(|i| {
                            let x = x0 + i * dx;
                            samples[y][x * channels..(x + 1) * channels].to_vec()
                        })
                        .collect();
                    pack_row(&row, spec.depth)
                })
                .collect();
            raw.extend(filter_rows(&rows, bpp, |y| ((y + filter_seed) % 5) as u8));
        }
        let mut png = SIGNATURE.to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&spec.width.to_be_bytes());
        ihdr.extend_from_slice(&spec.height.to_be_bytes());
        ihdr.extend_from_slice(&[spec.depth, spec.colour, 0, 0, u8::from(spec.interlaced)]);
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"tEXt", b"Comment\0test");
        if let Some(palette) = palette {
            chunk(&mut png, b"PLTE", palette);
            chunk(&mut png, b"tRNS", &[0, 128]);
        }
        let zlib = stored_zlib(&raw);
        let mid = zlib.len() / 2;
        chunk(&mut png, b"IDAT", &zlib[..mid]);
        chunk(&mut png, b"IDAT", &zlib[mid..]);
        chunk(&mut png, b"IEND", &[]);
        png
    }

    fn pattern(spec: &Spec, channels: usize) -> Vec<Vec<u16>> {
        let max = ((1u32 << spec.depth) - 1) as usize;
        (0..spec.height as usize)
            .map(|y| {
                (0..spec.width as usize * channels)
                    .map(|i| ((i * 37 + y * 101 + i * y * 13) % (max + 1)) as u16)
                    .collect()
            })
            .collect()
    }

    fn scale(v: u16, depth: u8) -> u8 {
        let max = (1u32 << depth) - 1;
        ((v as f64) * 255.0 / max as f64).round() as u8
    }

    fn blend(c: u16, a: u16, depth: u8) -> u8 {
        let max = ((1u32 << depth) - 1) as f64;
        let v = (c as f64 * a as f64 + max * (max - a as f64)) / max;
        ((v.round()) * 255.0 / max).round() as u8
    }

    fn check(colour: u8, depth: u8, interlaced: bool) {
        let spec = Spec {
            width: 13,
            height: 11,
            depth,
            colour,
            interlaced,
        };
        let channels = [1, 0, 3, 1, 2, 0, 4][colour as usize];
        let samples = pattern(&spec, channels);
        let palette: Vec<u8> = (0..256u32 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let png = encode(&spec, &samples, (colour == 3).then_some(&palette[..]));
        let img = RgbImage::decode(&png).unwrap_or_else(|e| {
            panic!("colour {colour} depth {depth} interlaced {interlaced}: {e}")
        });
        assert_eq!((img.width, img.height), (13, 11));
        for (y, row) in samples.iter().enumerate() {
            for x in 0..13 {
                let s = &row[x * channels..(x + 1) * channels];
                let expected = match colour {
                    0 => [scale(s[0], depth); 3],
                    2 => [scale(s[0], depth), scale(s[1], depth), scale(s[2], depth)],
                    3 => {
                        let p = s[0] as usize * 3;
                        [palette[p], palette[p + 1], palette[p + 2]]
                    }
                    4 => [blend(s[0], s[1], depth); 3],
                    _ => [
                        blend(s[0], s[3], depth),
                        blend(s[1], s[3], depth),
                        blend(s[2], s[3], depth),
                    ],
                };
                let p = (y * 13 + x) * 3;
                assert_eq!(
                    img.pixels[p..p + 3],
                    expected,
                    "colour {colour} depth {depth} interlaced {interlaced} at ({x}, {y})"
                );
            }
        }
    }

    const COMBINATIONS: [(u8, &[u8]); 5] = [
        (0, &[1, 2, 4, 8, 16]),
        (2, &[8, 16]),
        (3, &[1, 2, 4, 8]),
        (4, &[8, 16]),
        (6, &[8, 16]),
    ];

    #[test]
    fn decodes_every_colour_type_and_bit_depth() {
        for (colour, depths) in COMBINATIONS {
            for &depth in depths {
                check(colour, depth, false);
            }
        }
    }

    #[test]
    fn decodes_adam7_interlaced_images() {
        for (colour, depths) in COMBINATIONS {
            for &depth in depths {
                check(colour, depth, true);
            }
        }
    }

    #[test]
    fn decodes_tiny_interlaced_images_with_empty_passes() {
        for (w, h) in [(1, 1), (2, 1), (1, 3), (5, 2)] {
            let spec = Spec {
                width: w,
                height: h,
                depth: 8,
                colour: 2,
                interlaced: true,
            };
            let samples = pattern(&spec, 3);
            let img = RgbImage::decode(&encode(&spec, &samples, None)).unwrap();
            let expected: Vec<u8> = samples.iter().flatten().map(|&v| v as u8).collect();
            assert_eq!(img.pixels, expected);
        }
    }

    #[test]
    fn alpha_is_composited_over_white() {
        let spec = Spec {
            width: 3,
            height: 1,
            depth: 8,
            colour: 6,
            interlaced: false,
        };
        let samples = vec![vec![10, 20, 30, 0, 10, 20, 30, 255, 0, 0, 0, 128]];
        let img = RgbImage::decode(&encode(&spec, &samples, None)).unwrap();
        assert_eq!(img.pixels, vec![255, 255, 255, 10, 20, 30, 127, 127, 127]);
    }

    #[test]
    fn every_filter_type_round_trips() {
        let rows: Vec<Vec<u8>> = (0..6)
            .map(|y| {
                (0..12)
                    .map(|x| (x * 31 + y * 57 + x * y * 3) as u8)
                    .collect()
            })
            .collect();
        for kind in 0..5u8 {
            let mut data = filter_rows(&rows, 3, |_| kind);
            unfilter(&mut data, 13, 3).unwrap();
            let decoded: Vec<u8> = data
                .chunks_exact(13)
                .flat_map(|r| r[1..].to_vec())
                .collect();
            assert_eq!(decoded, rows.concat(), "filter {kind}");
        }
        let mut bad = vec![5, 0, 0, 0];
        assert!(unfilter(&mut bad, 4, 3).is_err());
    }

    fn sample_png() -> Vec<u8> {
        let spec = Spec {
            width: 4,
            height: 3,
            depth: 8,
            colour: 2,
            interlaced: false,
        };
        encode(&spec, &pattern(&spec, 3), None)
    }

    #[test]
    fn crc32_matches_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn rejects_bad_crc() {
        let mut png = sample_png();
        png[29] ^= 0xff;
        let err = RgbImage::decode(&png).unwrap_err().to_string();
        assert!(err.contains("CRC"), "{err}");
    }

    #[test]
    fn rejects_truncated_files() {
        let png = sample_png();
        for len in 0..png.len() {
            assert!(RgbImage::decode(&png[..len]).is_err(), "length {len}");
        }
        assert!(RgbImage::decode(&png).is_ok());
    }

    #[test]
    fn rejects_unknown_critical_chunks_and_ignores_ancillary_ones() {
        let png = sample_png();
        let iend = png.len() - 12;
        let mut with_ancillary = png[..iend].to_vec();
        chunk(&mut with_ancillary, b"zzZz", b"whatever");
        with_ancillary.extend_from_slice(&png[iend..]);
        assert!(RgbImage::decode(&with_ancillary).is_ok());

        let mut with_critical = png[..iend].to_vec();
        chunk(&mut with_critical, b"ZZZZ", b"whatever");
        with_critical.extend_from_slice(&png[iend..]);
        let err = RgbImage::decode(&with_critical).unwrap_err().to_string();
        assert!(err.contains("ZZZZ"), "{err}");
    }

    #[test]
    fn rejects_bad_headers() {
        for (depth, colour) in [(3, 0), (16, 3), (4, 2), (8, 5), (1, 6)] {
            let mut png = SIGNATURE.to_vec();
            let mut ihdr = vec![0, 0, 0, 1, 0, 0, 0, 1, depth, colour, 0, 0, 0];
            chunk(&mut png, b"IHDR", &ihdr);
            assert!(RgbImage::decode(&png).is_err());
            ihdr[12] = 2;
            assert!(Header::parse(&ihdr).is_err());
        }
        let mut png = SIGNATURE.to_vec();
        chunk(&mut png, b"IDAT", &[]);
        assert!(RgbImage::decode(&png).is_err());
    }

    #[test]
    fn missing_palette_is_an_error() {
        let mut png = SIGNATURE.to_vec();
        chunk(&mut png, b"IHDR", &[0, 0, 0, 1, 0, 0, 0, 1, 8, 3, 0, 0, 0]);
        chunk(&mut png, b"IDAT", &stored_zlib(&[0, 0]));
        chunk(&mut png, b"IEND", &[]);
        assert!(RgbImage::decode(&png).is_err());
    }

    #[test]
    fn huge_dimensions_are_rejected_without_allocating() {
        for (w, h) in [
            (0x7fff_ffffu32, 0x7fff_ffffu32),
            (60000, 60000),
            (1 << 20, 1),
        ] {
            let mut png = SIGNATURE.to_vec();
            let mut ihdr = Vec::new();
            ihdr.extend_from_slice(&w.to_be_bytes());
            ihdr.extend_from_slice(&h.to_be_bytes());
            ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
            chunk(&mut png, b"IHDR", &ihdr);
            chunk(&mut png, b"IDAT", &stored_zlib(&[0, 1, 2, 3]));
            chunk(&mut png, b"IEND", &[]);
            assert!(RgbImage::decode(&png).is_err());
        }
    }

    #[test]
    fn short_and_long_image_data_are_errors() {
        let mut png = SIGNATURE.to_vec();
        chunk(&mut png, b"IHDR", &[0, 0, 0, 2, 0, 0, 0, 1, 8, 0, 0, 0, 0]);
        chunk(&mut png, b"IDAT", &stored_zlib(&[0, 1]));
        chunk(&mut png, b"IEND", &[]);
        assert!(RgbImage::decode(&png).is_err());

        let mut png = SIGNATURE.to_vec();
        chunk(&mut png, b"IHDR", &[0, 0, 0, 2, 0, 0, 0, 1, 8, 0, 0, 0, 0]);
        chunk(&mut png, b"IDAT", &stored_zlib(&[0, 1, 2, 3]));
        chunk(&mut png, b"IEND", &[]);
        assert!(RgbImage::decode(&png).is_err());
    }

    #[test]
    fn corrupted_bytes_never_panic() {
        let png = sample_png();
        let mut state = 0x9e37_79b9u32;
        for _ in 0..3000 {
            let mut copy = png.clone();
            for _ in 0..3 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let i = (state as usize >> 8) % copy.len();
                copy[i] = state as u8;
            }
            let _ = RgbImage::decode(&copy);
        }
    }
}
