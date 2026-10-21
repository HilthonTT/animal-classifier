use std::path::PathBuf;

use pawnet::image::RgbImage;

const FIXTURES: [&str; 14] = [
    "rgb8",
    "rgb8-interlaced",
    "rgb16",
    "grey1",
    "grey2-interlaced",
    "grey4",
    "grey8",
    "grey16-interlaced",
    "palette",
    "palette-interlaced",
    "rgba8",
    "rgba16-interlaced",
    "grey-alpha8",
    "grey-alpha16",
];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn imagemagick_pngs_match_their_reference_pixels() {
    for name in FIXTURES {
        let png = RgbImage::load(fixture(&format!("{name}.png")))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let reference = RgbImage::load(fixture(&format!("{name}.ppm"))).unwrap();
        assert_eq!(png, reference, "{name}");
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn fix_crcs(png: &mut [u8]) {
    let mut pos = 8;
    while pos + 12 <= png.len() {
        let len = u32::from_be_bytes([png[pos], png[pos + 1], png[pos + 2], png[pos + 3]]);
        let end = pos + 8 + len as usize;
        if end + 4 > png.len() {
            return;
        }
        let crc = crc32(&png[pos + 4..end]);
        png[end..end + 4].copy_from_slice(&crc.to_be_bytes());
        pos = end + 4;
    }
}

#[test]
fn mutated_pngs_with_valid_crcs_never_panic() {
    let mut state = 0x2545_f491u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for name in FIXTURES {
        let original = std::fs::read(fixture(&format!("{name}.png"))).unwrap();
        for _ in 0..300 {
            let mut png = original.clone();
            for _ in 0..1 + next() % 4 {
                let i = 8 + next() as usize % (png.len() - 8);
                png[i] = next() as u8;
            }
            fix_crcs(&mut png);
            if let Ok(img) = RgbImage::decode(&png) {
                assert_eq!(img.pixels.len(), img.width * img.height * 3);
            }
        }
    }
}
