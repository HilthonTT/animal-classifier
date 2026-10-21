use std::fs;

use pawnet::data::IMAGE_SIZE;
use pawnet::data::cifar::{self, Split};
use pawnet::image::RgbImage;

use super::{AppResult, DEFAULT_DATA_DIR};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&["data", "out", "count", "scale"])?;
    let test_set = cifar::load_animals(&opts.path("data", DEFAULT_DATA_DIR), Split::Test)?;
    let out_dir = opts.path("out", "samples");
    let count: usize = opts.get("count", 12)?;
    let scale: usize = opts.get("scale", 4)?;
    if scale == 0 {
        return Err("--scale must be at least 1".into());
    }
    fs::create_dir_all(&out_dir)?;
    let count = count.min(test_set.len());
    for i in 0..count {
        let image = RgbImage::from_planar(IMAGE_SIZE, IMAGE_SIZE, test_set.image(i))
            .resize(IMAGE_SIZE * scale, IMAGE_SIZE * scale);
        let label = &test_set.class_names[test_set.labels[i] as usize];
        fs::write(
            out_dir.join(format!("{i:04}_{label}.ppm")),
            image.encode_ppm(),
        )?;
    }
    println!("Wrote {count} images to {}", out_dir.display());
    Ok(())
}
