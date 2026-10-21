use crate::error::{Error, Result};

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn corrupt(message: &str) -> Error {
    Error::format(format!("corrupt compressed data: {message}"))
}

fn truncated() -> Error {
    corrupt("unexpected end of stream")
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    count: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            buf: 0,
            count: 0,
        }
    }

    fn refill(&mut self) {
        while self.count <= 56 {
            let Some(&b) = self.data.get(self.pos) else {
                break;
            };
            self.buf |= (b as u64) << self.count;
            self.count += 8;
            self.pos += 1;
        }
    }

    fn bits(&mut self, n: u32) -> Result<u32> {
        if self.count < n {
            self.refill();
            if self.count < n {
                return Err(truncated());
            }
        }
        let value = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.count -= n;
        Ok(value)
    }

    fn align_and_take(&mut self, len: usize) -> Result<&'a [u8]> {
        self.pos -= (self.count / 8) as usize;
        self.buf = 0;
        self.count = 0;
        let end = self.pos.checked_add(len).ok_or_else(truncated)?;
        let bytes = self.data.get(self.pos..end).ok_or_else(truncated)?;
        self.pos = end;
        Ok(bytes)
    }
}

struct Huffman {
    table: Vec<u16>,
    bits: u32,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut count = [0u32; 16];
        for &len in lengths {
            count[len as usize] += 1;
        }
        count[0] = 0;
        let mut left: i64 = 1;
        for &n in &count[1..] {
            left = left * 2 - n as i64;
            if left < 0 {
                return Err(corrupt("over-subscribed Huffman code"));
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0;
        for len in 1..16 {
            code = (code + count[len - 1]) << 1;
            next[len] = code;
        }
        let bits = (1..16).rev().find(|&l| count[l] > 0).unwrap_or(1) as u32;
        let mut table = vec![0u16; 1 << bits];
        for (symbol, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let code = next[len as usize];
            next[len as usize] += 1;
            let reversed = (code.reverse_bits() >> (32 - len as u32)) as usize;
            let entry = ((symbol as u16) << 4) | len as u16;
            for slot in table.iter_mut().skip(reversed).step_by(1 << len) {
                *slot = entry;
            }
        }
        Ok(Self { table, bits })
    }

    fn decode(&self, reader: &mut BitReader) -> Result<u16> {
        if reader.count < self.bits {
            reader.refill();
        }
        let entry = self.table[(reader.buf & ((1u64 << self.bits) - 1)) as usize];
        let len = (entry & 15) as u32;
        if len == 0 {
            return Err(corrupt("invalid Huffman code"));
        }
        if len > reader.count {
            return Err(truncated());
        }
        reader.buf >>= len;
        reader.count -= len;
        Ok(entry >> 4)
    }
}

fn fixed_tables() -> Result<(Huffman, Huffman)> {
    let mut lengths = [0u8; 288];
    lengths[..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    Ok((Huffman::new(&lengths)?, Huffman::new(&[5; 30])?))
}

fn dynamic_tables(reader: &mut BitReader) -> Result<(Huffman, Huffman)> {
    let literals = reader.bits(5)? as usize + 257;
    let distances = reader.bits(5)? as usize + 1;
    let code_lengths = reader.bits(4)? as usize + 4;
    if literals > 286 || distances > 30 {
        return Err(corrupt("too many length or distance codes"));
    }
    let mut cl = [0u8; 19];
    for &index in &CODE_LENGTH_ORDER[..code_lengths] {
        cl[index] = reader.bits(3)? as u8;
    }
    let cl = Huffman::new(&cl)?;

    let total = literals + distances;
    let mut lengths = Vec::with_capacity(total);
    while lengths.len() < total {
        let (value, repeat) = match cl.decode(reader)? {
            symbol @ 0..=15 => (symbol as u8, 1),
            16 => {
                let &previous = lengths
                    .last()
                    .ok_or_else(|| corrupt("repeated code length with no previous length"))?;
                (previous, 3 + reader.bits(2)? as usize)
            }
            17 => (0, 3 + reader.bits(3)? as usize),
            _ => (0, 11 + reader.bits(7)? as usize),
        };
        if lengths.len() + repeat > total {
            return Err(corrupt("code lengths overflow the alphabet"));
        }
        lengths.resize(lengths.len() + repeat, value);
    }
    if lengths[256] == 0 {
        return Err(corrupt("missing end-of-block code"));
    }
    Ok((
        Huffman::new(&lengths[..literals])?,
        Huffman::new(&lengths[literals..])?,
    ))
}

fn too_much() -> Error {
    corrupt("more data than expected")
}

fn inflate_block(
    reader: &mut BitReader,
    out: &mut Vec<u8>,
    limit: usize,
    literals: &Huffman,
    distances: &Huffman,
) -> Result<()> {
    loop {
        let symbol = literals.decode(reader)?;
        if symbol < 256 {
            if out.len() >= limit {
                return Err(too_much());
            }
            out.push(symbol as u8);
            continue;
        }
        if symbol == 256 {
            return Ok(());
        }
        let index = symbol as usize - 257;
        if index >= LENGTH_BASE.len() {
            return Err(corrupt("invalid length code"));
        }
        let len = LENGTH_BASE[index] as usize + reader.bits(LENGTH_EXTRA[index] as u32)? as usize;
        let index = distances.decode(reader)? as usize;
        if index >= DIST_BASE.len() {
            return Err(corrupt("invalid distance code"));
        }
        let distance = DIST_BASE[index] as usize + reader.bits(DIST_EXTRA[index] as u32)? as usize;
        if distance > out.len() {
            return Err(corrupt("distance points before the start of the data"));
        }
        if len > limit - out.len() {
            return Err(too_much());
        }
        let start = out.len() - distance;
        if distance >= len {
            out.extend_from_within(start..start + len);
        } else {
            for i in start..start + len {
                let b = out[i];
                out.push(b);
            }
        }
    }
}

fn inflate(reader: &mut BitReader, limit: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(limit.min(reader.data.len().saturating_mul(4)));
    let mut fixed = None;
    loop {
        let last = reader.bits(1)? == 1;
        match reader.bits(2)? {
            0 => {
                let header = reader.align_and_take(4)?;
                let len = u16::from_le_bytes([header[0], header[1]]);
                let nlen = u16::from_le_bytes([header[2], header[3]]);
                if len != !nlen {
                    return Err(corrupt("stored block length check failed"));
                }
                let bytes = reader.align_and_take(len as usize)?;
                if bytes.len() > limit - out.len() {
                    return Err(too_much());
                }
                out.extend_from_slice(bytes);
            }
            1 => {
                if fixed.is_none() {
                    fixed = Some(fixed_tables()?);
                }
                if let Some((literals, distances)) = &fixed {
                    inflate_block(reader, &mut out, limit, literals, distances)?;
                }
            }
            2 => {
                let (literals, distances) = dynamic_tables(reader)?;
                inflate_block(reader, &mut out, limit, &literals, &distances)?;
            }
            _ => return Err(corrupt("reserved block type")),
        }
        if last {
            return Ok(out);
        }
    }
}

pub(crate) fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

pub(crate) fn zlib_decompress(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    let &[cmf, flg, ..] = data else {
        return Err(truncated());
    };
    if cmf & 0x0f != 8 || cmf >> 4 > 7 || (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(corrupt("invalid zlib header"));
    }
    if flg & 0x20 != 0 {
        return Err(corrupt("zlib preset dictionaries are not supported"));
    }
    let mut reader = BitReader::new(&data[2..]);
    let out = inflate(&mut reader, limit)?;
    let checksum = reader.align_and_take(4)?;
    let expected = u32::from_be_bytes([checksum[0], checksum[1], checksum[2], checksum[3]]);
    if adler32(&out) != expected {
        return Err(corrupt("Adler-32 checksum mismatch"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO: &[u8] = b"hello hello hello hello, deflate!";

    fn stored_zlib(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        let mut chunks = data.chunks(7).peekable();
        if data.is_empty() {
            out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
        }
        while let Some(chunk) = chunks.next() {
            out.push(u8::from(chunks.peek().is_none()));
            let len = chunk.len() as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(chunk);
        }
        out.extend_from_slice(&adler32(data).to_be_bytes());
        out
    }

    #[test]
    fn adler32_matches_reference() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn stored_blocks() {
        let packed = stored_zlib(HELLO);
        assert_eq!(zlib_decompress(&packed, 1000).unwrap(), HELLO);
        assert_eq!(zlib_decompress(&stored_zlib(b""), 0).unwrap(), b"");
    }

    #[test]
    fn fixed_huffman_block() {
        let packed = [
            0x78, 0xda, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x75, 0x14, 0x52,
            0x52, 0xd3, 0x72, 0x12, 0x4b, 0x52, 0x15, 0x01, 0xcf, 0xf3, 0x0b, 0xf3,
        ];
        assert_eq!((packed[2] >> 1) & 3, 1);
        assert_eq!(zlib_decompress(&packed, 1000).unwrap(), HELLO);
    }

    #[test]
    fn fixed_huffman_overlapping_copy() {
        let packed = [
            0x78, 0x9c, 0x4b, 0x4c, 0x4a, 0x24, 0x1b, 0x02, 0x00, 0xb9, 0x44, 0x16, 0xdb,
        ];
        let expected: Vec<u8> = b"ab".repeat(30);
        assert_eq!(zlib_decompress(&packed, 1000).unwrap(), expected);
    }

    #[test]
    fn dynamic_huffman_block() {
        let packed = [
            0x78, 0xda, 0x3d, 0x8b, 0x81, 0x0d, 0x00, 0x30, 0x08, 0xc2, 0x6e, 0x95, 0xb2, 0xff,
            0x5f, 0x18, 0xb8, 0x64, 0x18, 0x09, 0xc4, 0x8a, 0xa7, 0x8a, 0x13, 0x63, 0xb4, 0x3e,
            0xca, 0x26, 0xa7, 0xa8, 0xa1, 0x09, 0xec, 0xad, 0x87, 0x42, 0x11, 0xff, 0x5a, 0x9a,
            0x52, 0xaf, 0xee, 0x78, 0xdf, 0x24, 0xeb, 0x02, 0x8c, 0x42, 0x26, 0x3d,
        ];
        assert_eq!((packed[2] >> 1) & 3, 2);
        let expected = dynamic_expected();
        assert_eq!(zlib_decompress(&packed, 1000).unwrap(), expected);
    }

    fn dynamic_expected() -> Vec<u8> {
        let mut state = 7u32;
        (0..100)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
                b"aaaaaaaabbbbccde"[((state >> 16) % 16) as usize]
            })
            .collect()
    }

    #[test]
    fn rejects_bad_header_and_checksum() {
        let mut packed = stored_zlib(HELLO);
        packed[0] = 0x79;
        assert!(zlib_decompress(&packed, 1000).is_err());
        let mut packed = stored_zlib(HELLO);
        let last = packed.len() - 1;
        packed[last] ^= 1;
        let err = zlib_decompress(&packed, 1000).unwrap_err().to_string();
        assert!(err.contains("Adler-32"), "{err}");
    }

    #[test]
    fn rejects_truncated_and_oversized_output() {
        let packed = stored_zlib(HELLO);
        for len in 0..packed.len() {
            assert!(zlib_decompress(&packed[..len], 1000).is_err());
        }
        assert!(zlib_decompress(&packed, HELLO.len() - 1).is_err());
    }

    #[test]
    fn rejects_reserved_block_type_and_bad_stored_length() {
        assert!(zlib_decompress(&[0x78, 0x01, 0x07, 0, 0, 0, 0], 100).is_err());
        assert!(zlib_decompress(&[0x78, 0x01, 0x01, 1, 0, 0, 0, 7, 0, 7, 0, 7], 100).is_err());
    }

    #[test]
    fn rejects_distance_before_start() {
        let packed = [0x78, 0x01, 0x03, 0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
        assert!(zlib_decompress(&packed, 100).is_err());
    }

    #[test]
    fn garbage_never_panics() {
        let mut state = 0x1234_5678u32;
        for _ in 0..2000 {
            let len = (state % 64) as usize;
            let mut data = vec![0x78, 0x9c];
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                data.push(state as u8);
            }
            let _ = zlib_decompress(&data, 4096);
        }
    }
}
