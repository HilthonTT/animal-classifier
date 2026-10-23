use std::path::PathBuf;

use pawnet::image::RgbImage;

const FIXTURES: [&str; 10] = [
    "jpeg-444",
    "jpeg-420",
    "jpeg-422",
    "jpeg-440",
    "jpeg-411",
    "jpeg-progressive",
    "jpeg-grey",
    "jpeg-grey-progressive",
    "jpeg-cmyk",
    "jpeg-requant",
];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn differences(a: &RgbImage, b: &RgbImage) -> (u8, f32) {
    let diffs: Vec<u8> = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| x.abs_diff(y))
        .collect();
    let max = diffs.iter().copied().max().unwrap_or(0);
    let mean = diffs.iter().map(|&d| d as f32).sum::<f32>() / diffs.len() as f32;
    (max, mean)
}

#[test]
fn imagemagick_jpegs_match_their_reference_pixels() {
    for name in FIXTURES {
        let jpeg = RgbImage::load(fixture(&format!("{name}.jpg")))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let reference = RgbImage::load(fixture(&format!("{name}.ppm"))).unwrap();
        assert_eq!(
            (jpeg.width, jpeg.height),
            (reference.width, reference.height),
            "{name}"
        );
        let (max, mean) = differences(&jpeg, &reference);
        assert!(max <= 4 && mean < 0.5, "{name}: max {max}, mean {mean}");
    }
}

fn with_orientation(jpeg: &[u8], orientation: u8) -> Vec<u8> {
    let mut exif = b"Exif\0\0MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0".to_vec();
    exif.extend_from_slice(&[orientation, 0, 0, 0, 0, 0, 0]);
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&exif);
    out.extend_from_slice(&jpeg[2..]);
    out
}

#[test]
fn applies_exif_orientation() {
    let jpeg = std::fs::read(fixture("jpeg-444.jpg")).unwrap();
    let upright = RgbImage::decode(&jpeg).unwrap();
    let (w, h) = (upright.width, upright.height);
    let pixel = |img: &RgbImage, x: usize, y: usize| {
        let p = (y * img.width + x) * 3;
        img.pixels[p..p + 3].to_vec()
    };

    let rotated = RgbImage::decode(&with_orientation(&jpeg, 6)).unwrap();
    assert_eq!((rotated.width, rotated.height), (h, w));
    assert_eq!(pixel(&rotated, 0, 0), pixel(&upright, 0, h - 1));
    assert_eq!(pixel(&rotated, h - 1, 0), pixel(&upright, 0, 0));

    let upside_down = RgbImage::decode(&with_orientation(&jpeg, 3)).unwrap();
    assert_eq!(pixel(&upside_down, 0, 0), pixel(&upright, w - 1, h - 1));

    assert_eq!(
        RgbImage::decode(&with_orientation(&jpeg, 1)).unwrap(),
        upright
    );
}
