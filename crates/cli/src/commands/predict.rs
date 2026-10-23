use std::fs;
use std::path::{Path, PathBuf};

use pawnet::Saliency;
use pawnet::image::{Fit, Preprocessor, RgbImage};

use super::{AppResult, load_predictor};
use crate::args::Options;

const HEAT_MAP_SCALE: usize = 8;
const DEFAULT_TOP: usize = 10;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&[
        "model",
        "fit",
        "auto-contrast",
        "tta",
        "explain",
        "top",
        "json",
    ])?;
    if opts.positional.is_empty() {
        return Err("usage: pawnet predict <image.jpg|image.png|image.ppm|image.bmp>…".into());
    }
    let top = opts.get("top", DEFAULT_TOP)?;
    if top == 0 {
        return Err("--top must be at least 1".into());
    }
    let json = opts.flag("json");
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
    if json {
        println!("[");
    }
    for (i, path) in opts.positional.iter().enumerate() {
        let image = RgbImage::load(path)?;
        let probabilities = preprocessor.predict(model.as_ref(), &image);

        let mut ranked: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let shown: Vec<(&str, f32)> = ranked
            .iter()
            .take(top)
            .map(|&(class, p)| (model.labels()[class].as_str(), p))
            .collect();
        let heat_map = match &explain {
            Some(out) => {
                let target = heat_map_path(out, path, several || out.is_dir());
                let map = saliency.heat_map(model.as_ref(), &preprocessor, &image, ranked[0].0);
                let rendered = map.render(&preprocessor.thumbnail(&image), HEAT_MAP_SCALE);
                fs::write(&target, rendered.encode_ppm())?;
                Some(target)
            }
            None => None,
        };
        let prediction = Prediction {
            path,
            image: &image,
            ranked: &shown,
            heat_map: heat_map.as_deref(),
        };
        if json {
            let separator = if i + 1 < opts.positional.len() {
                ","
            } else {
                ""
            };
            println!("  {}{separator}", prediction.to_json());
        } else {
            prediction.print();
        }
    }
    if json {
        println!("]");
    }
    Ok(())
}

struct Prediction<'a> {
    path: &'a str,
    image: &'a RgbImage,
    ranked: &'a [(&'a str, f32)],
    heat_map: Option<&'a Path>,
}

impl Prediction<'_> {
    fn print(&self) {
        println!("{} ({}x{})", self.path, self.image.width, self.image.height);
        let width = self
            .ranked
            .iter()
            .map(|(label, _)| label.chars().count())
            .max()
            .unwrap_or(0)
            .max(8);
        for &(label, p) in self.ranked {
            println!("  {label:<width$} {} {:5.1}%", bar(p, 30), p * 100.0);
        }
        let (best, p) = self.ranked[0];
        println!("  => {best} ({:.1}% confident)", p * 100.0);
        if let Some(target) = self.heat_map {
            println!("  heat map for `{best}` written to {}", target.display());
        }
        println!();
    }

    fn to_json(&self) -> String {
        let (best, confidence) = self.ranked[0];
        let top: Vec<String> = self
            .ranked
            .iter()
            .map(|&(label, p)| {
                format!(
                    "{{\"label\": {}, \"probability\": {p}}}",
                    json_string(label)
                )
            })
            .collect();
        let heat_map = self
            .heat_map
            .map_or_else(|| "null".to_string(), |p| json_string(&p.to_string_lossy()));
        format!(
            "{{\"image\": {}, \"width\": {}, \"height\": {}, \"prediction\": {}, \"confidence\": {confidence}, \"top\": [{}], \"heat_map\": {heat_map}}}",
            json_string(self.path),
            self.image.width,
            self.image.height,
            json_string(best),
            top.join(", ")
        )
    }
}

fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
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
    fn escapes_json_strings() {
        assert_eq!(json_string("cat"), "\"cat\"");
        assert_eq!(json_string("a\"b\\c\nd\u{1}"), "\"a\\\"b\\\\c\\nd\\u0001\"");
    }

    #[test]
    fn writes_predictions_as_json() {
        let image = RgbImage::new(1, 1, vec![0, 0, 0]);
        let ranked = [("dog", 0.75), ("cat", 0.25)];
        let prediction = Prediction {
            path: "photos/dog.jpg",
            image: &image,
            ranked: &ranked,
            heat_map: None,
        };
        assert_eq!(
            prediction.to_json(),
            "{\"image\": \"photos/dog.jpg\", \"width\": 1, \"height\": 1, \"prediction\": \"dog\", \
             \"confidence\": 0.75, \"top\": [{\"label\": \"dog\", \"probability\": 0.75}, \
             {\"label\": \"cat\", \"probability\": 0.25}], \"heat_map\": null}"
        );
    }

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
