use super::RgbImage;
use crate::error::{Error, Result};

pub(crate) const SIGNATURE: [u8; 3] = [0xFF, 0xD8, 0xFF];
const MAX_PIXELS: usize = 1 << 26;

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

fn corrupt(message: &str) -> Error {
    Error::format(format!("corrupt JPEG: {message}"))
}

fn truncated() -> Error {
    Error::format("JPEG file is truncated")
}

fn unsupported(what: &str) -> Error {
    Error::format(format!("unsupported JPEG: {what}"))
}

const LOOKUP_BITS: u32 = 9;

struct Huffman {
    lookup: Vec<u16>,
    max_code: [i32; 17],
    offset: [i32; 17],
    symbols: Vec<u8>,
}

impl Huffman {
    fn parse(counts: &[u8], symbols: &[u8]) -> Result<Self> {
        let mut lookup = vec![0u16; 1 << LOOKUP_BITS];
        let mut max_code = [-1i32; 17];
        let mut offset = [0i32; 17];
        let mut code = 0i32;
        let mut index = 0usize;
        for len in 1..=16 {
            let count = counts[len - 1] as usize;
            offset[len] = index as i32 - code;
            for _ in 0..count {
                if code >= 1 << len {
                    return Err(corrupt("invalid Huffman table"));
                }
                if len as u32 <= LOOKUP_BITS {
                    let shift = LOOKUP_BITS - len as u32;
                    let first = (code as usize) << shift;
                    let entry = ((len as u16) << 8) | symbols[index] as u16;
                    lookup[first..first + (1 << shift)].fill(entry);
                }
                code += 1;
                index += 1;
            }
            if count > 0 {
                max_code[len] = code - 1;
            }
            code <<= 1;
        }
        Ok(Self {
            lookup,
            max_code,
            offset,
            symbols: symbols.to_vec(),
        })
    }

    fn decode(&self, reader: &mut BitReader) -> Result<u8> {
        let entry = self.lookup[reader.peek(LOOKUP_BITS) as usize];
        if entry != 0 {
            reader.consume(entry as u32 >> 8);
            return Ok(entry as u8);
        }
        let mut code = 0i32;
        for len in 1..=16 {
            code = (code << 1) | reader.bit() as i32;
            if code <= self.max_code[len] {
                return Ok(self.symbols[(self.offset[len] + code) as usize]);
            }
        }
        Err(corrupt("invalid Huffman code"))
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    bits: u32,
    at_marker: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            acc: 0,
            bits: 0,
            at_marker: false,
        }
    }

    fn fill(&mut self) {
        while self.bits <= 24 {
            let mut byte = 0u32;
            if !self.at_marker {
                match self.data.get(self.pos) {
                    Some(&0xFF) if self.data.get(self.pos + 1) == Some(&0) => {
                        byte = 0xFF;
                        self.pos += 2;
                    }
                    Some(&0xFF) | None => self.at_marker = true,
                    Some(&b) => {
                        byte = b as u32;
                        self.pos += 1;
                    }
                }
            }
            self.acc |= byte << (24 - self.bits);
            self.bits += 8;
        }
    }

    fn peek(&mut self, n: u32) -> u32 {
        self.fill();
        self.acc >> (32 - n)
    }

    fn consume(&mut self, n: u32) {
        self.acc <<= n;
        self.bits -= n;
    }

    fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let value = self.peek(n);
        self.consume(n);
        value
    }

    fn bit(&mut self) -> u32 {
        self.bits(1)
    }

    fn receive_extend(&mut self, size: u8) -> i32 {
        if size == 0 {
            return 0;
        }
        let value = self.bits(size as u32) as i32;
        if value < 1 << (size - 1) {
            value - (1 << size) + 1
        } else {
            value
        }
    }

    fn restart(&mut self) -> Result<()> {
        self.acc = 0;
        self.bits = 0;
        self.at_marker = false;
        while self.data.get(self.pos) == Some(&0xFF) && self.data.get(self.pos + 1) == Some(&0xFF) {
            self.pos += 1;
        }
        match self.data.get(self.pos..self.pos + 2) {
            Some(&[0xFF, m]) if (0xD0..=0xD7).contains(&m) => {
                self.pos += 2;
                Ok(())
            }
            Some(_) => Err(corrupt("missing restart marker")),
            None => Err(truncated()),
        }
    }
}

struct Component {
    id: u8,
    h: usize,
    v: usize,
    quant: usize,
    quant_table: Option<[u16; 64]>,
    blocks_w: usize,
    blocks_h: usize,
    width: usize,
    height: usize,
    coeffs: Vec<i16>,
    dc_pred: i32,
}

impl Component {
    fn block(&mut self, bx: usize, by: usize) -> &mut [i16] {
        let start = (by * self.blocks_w + bx) * 64;
        &mut self.coeffs[start..start + 64]
    }
}

struct Frame {
    progressive: bool,
    width: usize,
    height: usize,
    h_max: usize,
    v_max: usize,
    mcus_x: usize,
    mcus_y: usize,
    components: Vec<Component>,
}

struct Scan {
    components: Vec<(usize, usize, usize)>,
    start: usize,
    end: usize,
    high: u32,
    low: u32,
}

struct Decoder {
    frame: Option<Frame>,
    quant: [Option<[u16; 64]>; 4],
    dc_tables: [Option<Huffman>; 4],
    ac_tables: [Option<Huffman>; 4],
    restart_interval: usize,
    adobe_transform: Option<u8>,
    orientation: u8,
}

fn segment(bytes: &[u8], pos: usize) -> Result<&[u8]> {
    let len = bytes
        .get(pos..pos + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]) as usize)
        .ok_or_else(truncated)?;
    if len < 2 {
        return Err(corrupt("segment length is too small"));
    }
    bytes.get(pos + 2..pos + len).ok_or_else(truncated)
}

impl Decoder {
    fn new() -> Self {
        Self {
            frame: None,
            quant: [None; 4],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            restart_interval: 0,
            adobe_transform: None,
            orientation: 1,
        }
    }

    fn read_quant(&mut self, mut data: &[u8]) -> Result<()> {
        while let Some((&info, rest)) = data.split_first() {
            let (precision, id) = (info >> 4, (info & 15) as usize);
            if id > 3 || precision > 1 {
                return Err(corrupt("invalid quantisation table"));
            }
            let size = 64 << precision;
            let values = rest.get(..size).ok_or_else(truncated)?;
            let mut table = [0u16; 64];
            for (k, &z) in ZIGZAG.iter().enumerate() {
                table[z] = if precision == 0 {
                    values[k] as u16
                } else {
                    u16::from_be_bytes([values[2 * k], values[2 * k + 1]])
                };
            }
            self.quant[id] = Some(table);
            data = &rest[size..];
        }
        Ok(())
    }

    fn read_huffman(&mut self, mut data: &[u8]) -> Result<()> {
        while data.len() >= 17 {
            let (class, id) = (data[0] >> 4, (data[0] & 15) as usize);
            if class > 1 || id > 3 {
                return Err(corrupt("invalid Huffman table"));
            }
            let counts = &data[1..17];
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            let symbols = data.get(17..17 + total).ok_or_else(truncated)?;
            let table = Huffman::parse(counts, symbols)?;
            if class == 0 {
                self.dc_tables[id] = Some(table);
            } else {
                self.ac_tables[id] = Some(table);
            }
            data = &data[17 + total..];
        }
        if data.is_empty() {
            Ok(())
        } else {
            Err(corrupt("invalid Huffman table"))
        }
    }

    fn read_frame(&mut self, data: &[u8], progressive: bool, remaining: usize) -> Result<()> {
        if self.frame.is_some() {
            return Err(unsupported("more than one frame"));
        }
        let &[precision, h0, h1, w0, w1, count, ref rest @ ..] = data else {
            return Err(truncated());
        };
        if precision != 8 {
            return Err(unsupported(&format!("{precision}-bit samples")));
        }
        let height = u16::from_be_bytes([h0, h1]) as usize;
        let width = u16::from_be_bytes([w0, w1]) as usize;
        if width == 0 || height == 0 {
            return Err(unsupported("images without a height in the header"));
        }
        if width * height > MAX_PIXELS {
            return Err(Error::format(format!(
                "JPEG image is too large ({width}x{height}; at most {MAX_PIXELS} pixels are supported)"
            )));
        }
        let count = count as usize;
        if !matches!(count, 1 | 3 | 4) || rest.len() < count * 3 {
            return Err(corrupt("invalid frame header"));
        }
        let mut components = Vec::with_capacity(count);
        for c in rest[..count * 3].chunks_exact(3) {
            let (h, v, quant) = ((c[1] >> 4) as usize, (c[1] & 15) as usize, c[2] as usize);
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) || quant > 3 {
                return Err(corrupt("invalid frame header"));
            }
            components.push(Component {
                id: c[0],
                h,
                v,
                quant,
                quant_table: None,
                blocks_w: 0,
                blocks_h: 0,
                width: 0,
                height: 0,
                coeffs: Vec::new(),
                dc_pred: 0,
            });
        }
        let h_max = components.iter().map(|c| c.h).max().unwrap_or(1);
        let v_max = components.iter().map(|c| c.v).max().unwrap_or(1);
        if components
            .iter()
            .any(|c| h_max % c.h != 0 || v_max % c.v != 0)
        {
            return Err(unsupported("non-integer chroma subsampling"));
        }
        let mcus_x = width.div_ceil(8 * h_max);
        let mcus_y = height.div_ceil(8 * v_max);
        if mcus_x * mcus_y > remaining.saturating_mul(8) {
            return Err(truncated());
        }
        for c in &mut components {
            c.blocks_w = mcus_x * c.h;
            c.blocks_h = mcus_y * c.v;
            c.width = (width * c.h).div_ceil(h_max);
            c.height = (height * c.v).div_ceil(v_max);
            c.coeffs = vec![0; c.blocks_w * c.blocks_h * 64];
        }
        self.frame = Some(Frame {
            progressive,
            width,
            height,
            h_max,
            v_max,
            mcus_x,
            mcus_y,
            components,
        });
        Ok(())
    }

    fn read_scan(&self, data: &[u8]) -> Result<Scan> {
        let frame = self
            .frame
            .as_ref()
            .ok_or_else(|| corrupt("scan before frame"))?;
        let (&count, rest) = data.split_first().ok_or_else(truncated)?;
        let count = count as usize;
        if count == 0 || count > 4 || rest.len() != count * 2 + 3 {
            return Err(corrupt("invalid scan header"));
        }
        let mut components = Vec::with_capacity(count);
        for c in rest[..count * 2].chunks_exact(2) {
            let index = frame
                .components
                .iter()
                .position(|comp| comp.id == c[0])
                .ok_or_else(|| corrupt("scan names an unknown component"))?;
            let (dc, ac) = ((c[1] >> 4) as usize, (c[1] & 15) as usize);
            if dc > 3 || ac > 3 || components.iter().any(|&(i, _, _)| i == index) {
                return Err(corrupt("invalid scan header"));
            }
            components.push((index, dc, ac));
        }
        let tail = &rest[count * 2..];
        let scan = Scan {
            components,
            start: tail[0] as usize,
            end: tail[1] as usize,
            high: (tail[2] >> 4) as u32,
            low: (tail[2] & 15) as u32,
        };
        let valid = if frame.progressive {
            scan.end <= 63
                && scan.start <= scan.end
                && (scan.start == 0) == (scan.end == 0)
                && (scan.start == 0 || count == 1)
                && scan.low <= 13
                && scan.high <= 13
        } else {
            scan.start == 0 && scan.end == 63 && scan.high == 0 && scan.low == 0
        };
        if !valid {
            return Err(corrupt("invalid scan parameters"));
        }
        if count > 1 {
            let blocks: usize = scan
                .components
                .iter()
                .map(|&(i, _, _)| frame.components[i].h * frame.components[i].v)
                .sum();
            if blocks > 10 {
                return Err(corrupt("too many blocks per MCU"));
            }
        }
        Ok(scan)
    }

    fn table(tables: &[Option<Huffman>; 4], id: usize) -> Result<&Huffman> {
        tables[id]
            .as_ref()
            .ok_or_else(|| corrupt("scan uses an undefined Huffman table"))
    }

    fn decode_scan(&mut self, bytes: &[u8], pos: usize, scan: &Scan) -> Result<usize> {
        let mut frame = self
            .frame
            .take()
            .ok_or_else(|| corrupt("scan before frame"))?;
        let result = self.decode_scan_into(&mut frame, bytes, pos, scan);
        self.frame = Some(frame);
        result
    }

    fn decode_scan_into(
        &mut self,
        frame: &mut Frame,
        bytes: &[u8],
        pos: usize,
        scan: &Scan,
    ) -> Result<usize> {
        let mut tables = Vec::with_capacity(scan.components.len());
        for &(_, dc, ac) in &scan.components {
            let needs_dc = scan.start == 0 && scan.high == 0;
            let needs_ac = scan.end > 0;
            let dc = if needs_dc {
                Some(Self::table(&self.dc_tables, dc)?)
            } else {
                None
            };
            let ac = if needs_ac {
                Some(Self::table(&self.ac_tables, ac)?)
            } else {
                None
            };
            tables.push((dc, ac));
        }
        for &(i, _, _) in &scan.components {
            let comp = &mut frame.components[i];
            if comp.quant_table.is_none() {
                comp.quant_table = Some(
                    self.quant[comp.quant]
                        .ok_or_else(|| corrupt("undefined quantisation table"))?,
                );
            }
            comp.dc_pred = 0;
        }
        let mut reader = BitReader::new(bytes, pos);
        let mut eob_run = 0u32;
        let single = scan.components.len() == 1;
        let (units_x, units_y) = if single {
            let c = &frame.components[scan.components[0].0];
            (c.width.div_ceil(8), c.height.div_ceil(8))
        } else {
            (frame.mcus_x, frame.mcus_y)
        };
        let total = units_x * units_y;
        for unit in 0..total {
            if self.restart_interval > 0 && unit > 0 && unit % self.restart_interval == 0 {
                reader.restart()?;
                eob_run = 0;
                for &(i, _, _) in &scan.components {
                    frame.components[i].dc_pred = 0;
                }
            }
            let (ux, uy) = (unit % units_x, unit / units_x);
            for (&(i, _, _), &(dc, ac)) in scan.components.iter().zip(&tables) {
                let comp = &mut frame.components[i];
                let (bw, bh) = if single { (1, 1) } else { (comp.h, comp.v) };
                for y in 0..bh {
                    for x in 0..bw {
                        let (bx, by) = (ux * bw + x, uy * bh + y);
                        decode_block(comp, bx, by, &mut reader, scan, dc, ac, &mut eob_run)?;
                    }
                }
            }
        }
        Ok(next_marker(bytes, reader.pos))
    }

    fn read_exif(&mut self, data: &[u8]) {
        if let Some(tiff) = data.strip_prefix(b"Exif\0\0") {
            if let Some(o) = exif_orientation(tiff) {
                self.orientation = o;
            }
        }
    }

    fn decode(mut self, bytes: &[u8]) -> Result<RgbImage> {
        if !bytes.starts_with(&SIGNATURE[..2]) {
            return Err(corrupt("missing start-of-image marker"));
        }
        let mut pos = 2;
        let mut scans = 0;
        loop {
            while bytes.get(pos).is_some_and(|&b| b != 0xFF) {
                pos += 1;
            }
            while bytes.get(pos) == Some(&0xFF) {
                pos += 1;
            }
            let Some(&marker) = bytes.get(pos) else {
                if scans > 0 {
                    break;
                }
                return Err(truncated());
            };
            pos += 1;
            match marker {
                0xD9 => break,
                0xD0..=0xD8 | 0x01 => continue,
                _ => {}
            }
            let data = segment(bytes, pos)?;
            let next = pos + 2 + data.len();
            match marker {
                0xC0..=0xC2 => self.read_frame(data, marker == 0xC2, bytes.len() - next)?,
                0xC3 | 0xC5..=0xC7 | 0xCB | 0xCD..=0xCF => {
                    return Err(unsupported("lossless or hierarchical coding"));
                }
                0xC9 | 0xCA => return Err(unsupported("arithmetic coding")),
                0xC4 => self.read_huffman(data)?,
                0xDB => self.read_quant(data)?,
                0xDD => {
                    let &[a, b] = data else {
                        return Err(corrupt("invalid restart interval"));
                    };
                    self.restart_interval = u16::from_be_bytes([a, b]) as usize;
                }
                0xE1 => self.read_exif(data),
                0xEE if data.starts_with(b"Adobe") && data.len() >= 12 => {
                    self.adobe_transform = Some(data[11]);
                }
                0xDA => {
                    let scan = self.read_scan(data)?;
                    pos = self.decode_scan(bytes, next, &scan)?;
                    scans += 1;
                    continue;
                }
                _ => {}
            }
            pos = next;
        }
        if scans == 0 {
            return Err(corrupt("no image data"));
        }
        let frame = self.frame.take().ok_or_else(|| corrupt("missing frame"))?;
        let image = self.render(frame)?;
        Ok(orient(&image, self.orientation))
    }

    fn render(&self, frame: Frame) -> Result<RgbImage> {
        let idct = IdctTable::new();
        let mut planes = Vec::with_capacity(frame.components.len());
        for comp in &frame.components {
            let quant = comp
                .quant_table
                .as_ref()
                .or(self.quant[comp.quant].as_ref())
                .ok_or_else(|| corrupt("undefined quantisation table"))?;
            let plane = idct.plane(comp, quant);
            planes.push(upsample(
                &plane,
                comp,
                frame.h_max / comp.h,
                frame.v_max / comp.v,
                frame.width,
                frame.height,
            ));
        }
        let ids: Vec<u8> = frame.components.iter().map(|c| c.id).collect();
        let count = frame.width * frame.height;
        let mut pixels = Vec::with_capacity(count * 3);
        match planes.len() {
            1 => {
                for &v in &planes[0] {
                    pixels.extend_from_slice(&[v, v, v]);
                }
            }
            3 => {
                let rgb = self.adobe_transform == Some(0)
                    || (self.adobe_transform.is_none() && ids == *b"RGB");
                for ((&a, &b), &c) in planes[0].iter().zip(&planes[1]).zip(&planes[2]) {
                    pixels.extend_from_slice(&if rgb { [a, b, c] } else { ycbcr(a, b, c) });
                }
            }
            _ => {
                let ycck = self.adobe_transform == Some(2);
                let samples = planes[0]
                    .iter()
                    .zip(&planes[1])
                    .zip(&planes[2])
                    .zip(&planes[3]);
                for (((&a, &b), &c), &k) in samples {
                    let cmy = if ycck {
                        ycbcr(a, b, c).map(|v| 255 - v)
                    } else {
                        [a, b, c]
                    };
                    pixels.extend(cmy.map(|v| ((v as u32 * k as u32 + 127) / 255) as u8));
                }
            }
        }
        Ok(RgbImage::new(frame.width, frame.height, pixels))
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_block(
    comp: &mut Component,
    bx: usize,
    by: usize,
    reader: &mut BitReader,
    scan: &Scan,
    dc: Option<&Huffman>,
    ac: Option<&Huffman>,
    eob_run: &mut u32,
) -> Result<()> {
    if scan.start == 0 {
        if scan.high == 0 {
            let size = dc
                .ok_or_else(|| corrupt("missing DC table"))?
                .decode(reader)?;
            if size > 15 {
                return Err(corrupt("invalid DC coefficient"));
            }
            comp.dc_pred = comp.dc_pred.wrapping_add(reader.receive_extend(size));
            let value = comp.dc_pred.wrapping_shl(scan.low) as i16;
            comp.block(bx, by)[0] = value;
        } else if reader.bit() == 1 {
            comp.block(bx, by)[0] |= 1 << scan.low;
        }
        if scan.end == 0 {
            return Ok(());
        }
    }
    let ac = ac.ok_or_else(|| corrupt("missing AC table"))?;
    let block = comp.block(bx, by);
    let first = scan.start.max(1);
    if scan.high == 0 {
        decode_ac_first(block, reader, ac, first, scan.end, scan.low, eob_run)
    } else {
        decode_ac_refine(block, reader, ac, first, scan.end, scan.low, eob_run)
    }
}

fn decode_ac_first(
    block: &mut [i16],
    reader: &mut BitReader,
    ac: &Huffman,
    start: usize,
    end: usize,
    low: u32,
    eob_run: &mut u32,
) -> Result<()> {
    if *eob_run > 0 {
        *eob_run -= 1;
        return Ok(());
    }
    let mut k = start;
    while k <= end {
        let rs = ac.decode(reader)?;
        let (run, size) = ((rs >> 4) as usize, rs & 15);
        if size == 0 {
            if run < 15 {
                *eob_run = (1u32 << run) - 1 + reader.bits(run as u32);
                break;
            }
            k += 16;
            continue;
        }
        k += run;
        if k > end {
            return Err(corrupt("coefficient index out of range"));
        }
        block[ZIGZAG[k]] = reader.receive_extend(size).wrapping_shl(low) as i16;
        k += 1;
    }
    Ok(())
}

fn refine(coeff: &mut i16, reader: &mut BitReader, bit: i16) {
    if reader.bit() == 1 && *coeff & bit == 0 {
        *coeff = if *coeff >= 0 {
            coeff.wrapping_add(bit)
        } else {
            coeff.wrapping_sub(bit)
        };
    }
}

fn decode_ac_refine(
    block: &mut [i16],
    reader: &mut BitReader,
    ac: &Huffman,
    start: usize,
    end: usize,
    low: u32,
    eob_run: &mut u32,
) -> Result<()> {
    let bit = (1i32 << low) as i16;
    let mut k = start;
    if *eob_run == 0 {
        while k <= end {
            let rs = ac.decode(reader)?;
            let (mut run, size) = ((rs >> 4) as i32, rs & 15);
            let mut value = 0i16;
            match size {
                0 if run < 15 => {
                    *eob_run = (1u32 << run) + reader.bits(run as u32);
                    break;
                }
                0 => {}
                1 => {
                    value = if reader.bit() == 1 {
                        bit
                    } else {
                        bit.wrapping_neg()
                    }
                }
                _ => return Err(corrupt("invalid refinement coefficient")),
            }
            while k <= end {
                let coeff = &mut block[ZIGZAG[k]];
                if *coeff != 0 {
                    refine(coeff, reader, bit);
                } else {
                    run -= 1;
                    if run < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if value != 0 && k <= end {
                block[ZIGZAG[k]] = value;
            }
            k += 1;
        }
    }
    if *eob_run > 0 {
        while k <= end {
            let coeff = &mut block[ZIGZAG[k]];
            if *coeff != 0 {
                refine(coeff, reader, bit);
            }
            k += 1;
        }
        *eob_run -= 1;
    }
    Ok(())
}

fn next_marker(bytes: &[u8], mut pos: usize) -> usize {
    while pos + 1 < bytes.len() {
        if bytes[pos] == 0xFF && !matches!(bytes[pos + 1], 0x00 | 0xD0..=0xD7 | 0xFF) {
            return pos;
        }
        pos += 1;
    }
    bytes.len()
}

struct IdctTable([[f32; 8]; 8]);

impl IdctTable {
    fn new() -> Self {
        let mut table = [[0.0; 8]; 8];
        for (x, row) in table.iter_mut().enumerate() {
            for (u, value) in row.iter_mut().enumerate() {
                let scale = if u == 0 { 0.5 / 2f32.sqrt() } else { 0.5 };
                let angle = (2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0;
                *value = scale * angle.cos();
            }
        }
        Self(table)
    }

    fn block(&self, coeffs: &[i16], quant: &[u16; 64], out: &mut [u8], stride: usize) {
        let mut f = [0f32; 64];
        let mut ac = false;
        for i in 0..64 {
            f[i] = coeffs[i] as f32 * quant[i] as f32;
            ac |= i > 0 && coeffs[i] != 0;
        }
        if !ac {
            let v = (f[0] / 8.0 + 128.0).round().clamp(0.0, 255.0) as u8;
            for y in 0..8 {
                out[y * stride..y * stride + 8].fill(v);
            }
            return;
        }
        let mut tmp = [0f32; 64];
        for v in 0..8 {
            for x in 0..8 {
                tmp[v * 8 + x] = (0..8).map(|u| self.0[x][u] * f[v * 8 + u]).sum();
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let s: f32 = (0..8).map(|v| self.0[y][v] * tmp[v * 8 + x]).sum();
                out[y * stride + x] = (s + 128.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    fn plane(&self, comp: &Component, quant: &[u16; 64]) -> Vec<u8> {
        let stride = comp.blocks_w * 8;
        let mut plane = vec![0u8; stride * comp.blocks_h * 8];
        for by in 0..comp.blocks_h {
            for bx in 0..comp.blocks_w {
                let start = (by * comp.blocks_w + bx) * 64;
                let offset = by * 8 * stride + bx * 8;
                self.block(
                    &comp.coeffs[start..start + 64],
                    quant,
                    &mut plane[offset..],
                    stride,
                );
            }
        }
        plane
    }
}

fn upsample_line(src: &[f32], factor: usize, out_len: usize) -> Vec<f32> {
    let last = src.len() - 1;
    (0..out_len)
        .map(|o| {
            let i = (o / factor).min(last);
            if factor != 2 {
                return src[i];
            }
            let neighbour = if o % 2 == 0 {
                i.saturating_sub(1)
            } else {
                (i + 1).min(last)
            };
            0.75 * src[i] + 0.25 * src[neighbour]
        })
        .collect()
}

fn upsample(
    plane: &[u8],
    comp: &Component,
    fx: usize,
    fy: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let stride = comp.blocks_w * 8;
    if fx == 1 && fy == 1 {
        return (0..height)
            .flat_map(|y| plane[y * stride..y * stride + width].iter().copied())
            .collect();
    }
    let rows: Vec<Vec<f32>> = (0..comp.height)
        .map(|y| {
            let row: Vec<f32> = plane[y * stride..y * stride + comp.width]
                .iter()
                .map(|&v| v as f32)
                .collect();
            upsample_line(&row, fx, width)
        })
        .collect();
    let mut out = vec![0u8; width * height];
    let mut column = vec![0f32; comp.height];
    for x in 0..width {
        for (c, row) in column.iter_mut().zip(&rows) {
            *c = row[x];
        }
        for (y, v) in upsample_line(&column, fy, height).into_iter().enumerate() {
            out[y * width + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn ycbcr(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let (y, cb, cr) = (y as f32, cb as f32 - 128.0, cr as f32 - 128.0);
    [
        y + 1.402 * cr,
        y - 0.344_136 * cb - 0.714_136 * cr,
        y + 1.772 * cb,
    ]
    .map(|v| v.round().clamp(0.0, 255.0) as u8)
}

fn exif_orientation(tiff: &[u8]) -> Option<u8> {
    let big_endian = match tiff.get(..4)? {
        b"MM\0*" => true,
        b"II*\0" => false,
        _ => return None,
    };
    let u16_at = |p: usize| {
        let b = tiff.get(p..p + 2)?;
        Some(if big_endian {
            u16::from_be_bytes([b[0], b[1]])
        } else {
            u16::from_le_bytes([b[0], b[1]])
        })
    };
    let u32_at = |p: usize| {
        let b = tiff.get(p..p + 4)?;
        Some(if big_endian {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
    };
    let ifd = u32_at(4)? as usize;
    let entries = u16_at(ifd)? as usize;
    (0..entries)
        .map(|i| ifd + 2 + i * 12)
        .find(|&entry| u16_at(entry) == Some(0x0112) && u16_at(entry + 2) == Some(3))
        .and_then(|entry| u16_at(entry + 8))
        .filter(|o| (1..=8).contains(o))
        .map(|o| o as u8)
}

pub(crate) fn orient(image: &RgbImage, orientation: u8) -> RgbImage {
    if !(2..=8).contains(&orientation) {
        return image.clone();
    }
    let (w, h) = (image.width, image.height);
    let swap = orientation >= 5;
    let (out_w, out_h) = if swap { (h, w) } else { (w, h) };
    let mut pixels = Vec::with_capacity(image.pixels.len());
    for y in 0..out_h {
        for x in 0..out_w {
            let (sx, sy) = match orientation {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                _ => (w - 1 - y, x),
            };
            let p = (sy * w + sx) * 3;
            pixels.extend_from_slice(&image.pixels[p..p + 3]);
        }
    }
    RgbImage::new(out_w, out_h, pixels)
}

impl RgbImage {
    pub fn decode_jpeg(bytes: &[u8]) -> Result<Self> {
        Decoder::new().decode(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BitWriter {
        out: Vec<u8>,
        acc: u32,
        bits: u32,
    }

    impl BitWriter {
        fn put(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.acc = (self.acc << 1) | ((value >> i) & 1);
                self.bits += 1;
                if self.bits == 8 {
                    self.out.push(self.acc as u8);
                    if self.acc == 0xFF {
                        self.out.push(0);
                    }
                    self.acc = 0;
                    self.bits = 0;
                }
            }
        }

        fn flush(&mut self) {
            while self.bits != 0 {
                self.put(1, 1);
            }
        }
    }

    fn marker(out: &mut Vec<u8>, code: u8, data: &[u8]) {
        out.extend_from_slice(&[0xFF, code]);
        out.extend_from_slice(&((data.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(data);
    }

    fn dc_only_grey(dcs: &[i32], restart: Option<u16>) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8];
        let mut dqt = vec![0];
        dqt.extend_from_slice(&[1; 64]);
        marker(&mut out, 0xDB, &dqt);
        let width = (dcs.len() * 8) as u16;
        let mut sof = vec![8, 0, 8];
        sof.extend_from_slice(&width.to_be_bytes());
        sof.extend_from_slice(&[1, 1, 0x11, 0]);
        marker(&mut out, 0xC0, &sof);
        let mut dht = vec![0x00, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        dht.extend(0..12u8);
        dht.extend_from_slice(&[0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        marker(&mut out, 0xC4, &dht);
        if let Some(interval) = restart {
            marker(&mut out, 0xDD, &interval.to_be_bytes());
        }
        marker(&mut out, 0xDA, &[1, 1, 0x00, 0, 63, 0]);
        let mut writer = BitWriter {
            out: Vec::new(),
            acc: 0,
            bits: 0,
        };
        let mut pred = 0;
        for (i, &dc) in dcs.iter().enumerate() {
            if let Some(interval) = restart.filter(|&n| i > 0 && i % n as usize == 0) {
                writer.flush();
                writer
                    .out
                    .extend_from_slice(&[0xFF, 0xD0 + ((i / interval as usize - 1) % 8) as u8]);
                pred = 0;
            }
            let diff = dc - pred;
            pred = dc;
            let size = 32 - diff.unsigned_abs().leading_zeros();
            writer.put(size, 4);
            let bits = if diff < 0 { diff - 1 } else { diff };
            writer.put(bits as u32 & ((1 << size) - 1), size);
            writer.put(0, 1);
        }
        writer.flush();
        out.extend_from_slice(&writer.out);
        out.extend_from_slice(&[0xFF, 0xD9]);
        out
    }

    #[test]
    fn decodes_flat_blocks_with_and_without_restarts() {
        let dcs = [80, 160, -80, 0, 8, -1024];
        let expected: Vec<u8> = [138, 148, 118, 128, 129, 0]
            .iter()
            .flat_map(|&v| [v; 24])
            .collect();
        for restart in [None, Some(1), Some(2), Some(4)] {
            let image = RgbImage::decode(&dc_only_grey(&dcs, restart)).unwrap();
            assert_eq!((image.width, image.height), (48, 8));
            let first_row: Vec<u8> = image.pixels[..48 * 3]
                .chunks_exact(24)
                .flat_map(|c| c.to_vec())
                .collect();
            assert_eq!(first_row, expected, "restart {restart:?}");
        }
    }

    #[test]
    fn rejects_missing_restart_markers() {
        let mut jpeg = dc_only_grey(&[80, 160], None);
        let sos = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        jpeg.splice(sos..sos, [0xFF, 0xDD, 0, 4, 0, 1]);
        assert!(RgbImage::decode(&jpeg).is_err());
    }

    #[test]
    fn reads_exif_orientation_in_both_byte_orders() {
        let little = [
            b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0,
        ];
        let big = [
            b'M', b'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 8, 0, 0,
        ];
        assert_eq!(exif_orientation(&little), Some(6));
        assert_eq!(exif_orientation(&big), Some(8));
        assert_eq!(exif_orientation(b"II*\0\xff\0\0\0"), None);
    }

    #[test]
    fn orients_images() {
        let image = RgbImage::new(2, 1, vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(orient(&image, 1), image);
        assert_eq!(orient(&image, 2).pixels, vec![2, 2, 2, 1, 1, 1]);
        let cw = orient(&image, 6);
        assert_eq!((cw.width, cw.height), (1, 2));
        assert_eq!(cw.pixels, vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(orient(&image, 8).pixels, vec![2, 2, 2, 1, 1, 1]);
        assert_eq!(orient(&orient(&image, 6), 8), image);
    }

    #[test]
    fn rejects_unsupported_and_broken_files() {
        assert!(RgbImage::decode(&[0xFF, 0xD8, 0xFF]).is_err());
        let mut arithmetic = vec![0xFF, 0xD8];
        marker(&mut arithmetic, 0xC9, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
        assert!(RgbImage::decode(&arithmetic).is_err());
        let mut huge = vec![0xFF, 0xD8];
        marker(&mut huge, 0xC0, &[8, 0xFF, 0xFF, 0xFF, 0xFF, 1, 1, 0x11, 0]);
        assert!(RgbImage::decode(&huge).is_err());
    }

    #[test]
    fn corrupted_bytes_never_panic() {
        let files: [&[u8]; 3] = [
            include_bytes!("../../tests/fixtures/jpeg-420.jpg"),
            include_bytes!("../../tests/fixtures/jpeg-progressive.jpg"),
            &dc_only_grey(&[80, 160, -80, 0], Some(1)),
        ];
        let mut state = 0x9e37_79b9u32;
        for file in files {
            for _ in 0..1500 {
                let mut copy = file.to_vec();
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
}
