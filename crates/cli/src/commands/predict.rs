use pawnet::image::{Fit, Preprocessor, RgbImage};

use super::{AppResult, load_predictor};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&["model", "fit", "auto-contrast", "tta"])?;
    if opts.positional.is_empty() {
        return Err("usage: pawnet predict <image.ppm|image.bmp>…".into());
    }
    let model = load_predictor(opts)?;
    let preprocessor = Preprocessor {
        fit: opts.get("fit", Fit::Crop)?,
        auto_contrast: opts.flag("auto-contrast"),
        test_time_augmentation: opts.flag("tta"),
    };
    for path in &opts.positional {
        let image = RgbImage::load(path)?;
        let probabilities = preprocessor.predict(model.as_ref(), &image);

        let mut ranked: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("{path} ({}x{})", image.width, image.height);
        for &(class, p) in &ranked {
            println!(
                "  {:<8} {} {:5.1}%",
                model.labels()[class],
                bar(p, 30),
                p * 100.0
            );
        }
        let (best, p) = ranked[0];
        println!(
            "  => {} ({:.1}% confident)\n",
            model.labels()[best],
            p * 100.0
        );
    }
    Ok(())
}

fn bar(fraction: f32, width: usize) -> String {
    let filled = ((fraction * width as f32).round() as usize).min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}
