use std::fs;
use std::path::{Path, PathBuf};

use pawnet::Saliency;
use pawnet::image::{Fit, Preprocessor, RgbImage};

use super::{AppResult, load_predictor};
use crate::args::Options;

const HEAT_MAP_SCALE: usize = 8;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&["model", "fit", "auto-contrast", "tta", "explain"])?;
    if opts.positional.is_empty() {
        return Err("usage: pawnet predict <image.jpg|image.png|image.ppm|image.bmp>…".into());
    }
    let explain = opts.get_opt::<PathBuf>("explain")?;
    let several = opts.positional.len() > 1;
    if let Some(out) = explain.as_ref().filter(|_| several) {
        fs::create_dir_all(out)?;
    }
    let model = load_predictor(opts)?;
    let preprocessor = Preprocessor {
        fit: opts.get("fit", Fit::Crop)?,
        auto_contrast: opts.flag("auto-contrast"),
        test_time_augmentation: opts.flag("tta"),
    };
    let saliency = Saliency::default();
    for path in &opts.positional {
        let image = RgbImage::load(path)?;
        let probabilities = preprocessor.predict(model.as_ref(), &image);

        let mut ranked: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("{path} ({}x{})", image.width, image.height);
        let shown = &ranked[..ranked.len().min(10)];
        let width = shown
            .iter()
            .map(|&(class, _)| model.labels()[class].chars().count())
            .max()
            .unwrap_or(0)
            .max(8);
        for &(class, p) in shown {
            println!(
                "  {:<width$} {} {:5.1}%",
                model.labels()[class],
                bar(p, 30),
                p * 100.0
            );
        }
        let (best, p) = ranked[0];
        println!(
            "  => {} ({:.1}% confident)",
            model.labels()[best],
            p * 100.0
        );
        if let Some(out) = &explain {
            let target = heat_map_path(out, path, several || out.is_dir());
            let map = saliency.heat_map(model.as_ref(), &preprocessor, &image, best);
            let rendered = map.render(&preprocessor.thumbnail(&image), HEAT_MAP_SCALE);
            fs::write(&target, rendered.encode_ppm())?;
            println!(
                "  heat map for `{}` written to {}",
                model.labels()[best],
                target.display()
            );
        }
        println!();
    }
    Ok(())
}

fn heat_map_path(out: &Path, image: &str, in_directory: bool) -> PathBuf {
    if !in_directory {
        return out.to_path_buf();
    }
    let stem = Path::new(image)
        .file_stem()
        .map_or_else(|| "image".into(), |s| s.to_string_lossy());
    out.join(format!("{stem}-heatmap.ppm"))
}

fn bar(fraction: f32, width: usize) -> String {
    let filled = ((fraction * width as f32).round() as usize).min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_heat_maps_after_the_image() {
        assert_eq!(
            heat_map_path(Path::new("maps"), "photos/cat.ppm", true),
            Path::new("maps").join("cat-heatmap.ppm")
        );
        assert_eq!(
            heat_map_path(Path::new("maps"), "dog.large.bmp", true),
            Path::new("maps").join("dog.large-heatmap.ppm")
        );
        assert_eq!(
            heat_map_path(Path::new("out/cat-map.ppm"), "photos/cat.ppm", false),
            Path::new("out/cat-map.ppm")
        );
    }
}
