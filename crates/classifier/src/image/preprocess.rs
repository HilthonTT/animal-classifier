use std::fmt;
use std::str::FromStr;

use super::RgbImage;
use crate::data::{self, IMAGE_SIZE};
use crate::predictor::Predictor;

const PAD_COLOUR: [u8; 3] = [125, 123, 114];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fit {
    #[default]
    Crop,
    Pad,
    Stretch,
}

impl FromStr for Fit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "crop" => Ok(Fit::Crop),
            "pad" => Ok(Fit::Pad),
            "stretch" => Ok(Fit::Stretch),
            _ => Err(format!("unknown fit `{s}` (expected crop, pad or stretch)")),
        }
    }
}

impl fmt::Display for Fit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Fit::Crop => "crop",
            Fit::Pad => "pad",
            Fit::Stretch => "stretch",
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Preprocessor {
    pub fit: Fit,
    pub auto_contrast: bool,
    pub test_time_augmentation: bool,
}

impl Preprocessor {
    pub fn thumbnail(&self, image: &RgbImage) -> RgbImage {
        let square = match self.fit {
            Fit::Crop => image.center_crop_square(),
            Fit::Pad => image.pad_square(PAD_COLOUR),
            Fit::Stretch => image.clone(),
        };
        let small = square.resize(IMAGE_SIZE, IMAGE_SIZE);
        if self.auto_contrast {
            small.auto_contrast()
        } else {
            small
        }
    }

    pub fn inputs(&self, image: &RgbImage) -> Vec<Vec<f32>> {
        let thumb = self.thumbnail(image);
        let mut views = vec![thumb.to_planar()];
        if self.test_time_augmentation {
            views.push(thumb.flip_horizontal().to_planar());
        }
        views.iter().map(|raw| data::preprocess(raw)).collect()
    }

    pub fn predict<P: Predictor + ?Sized>(&self, model: &P, image: &RgbImage) -> Vec<f32> {
        average(self.inputs(image).iter().map(|input| model.predict(input)))
    }
}

pub(crate) fn average(predictions: impl Iterator<Item = Vec<f32>>) -> Vec<f32> {
    let mut sum: Vec<f32> = Vec::new();
    let mut count = 0;
    for p in predictions {
        if sum.is_empty() {
            sum = vec![0.0; p.len()];
        }
        for (s, v) in sum.iter_mut().zip(&p) {
            *s += v;
        }
        count += 1;
    }
    sum.iter_mut().for_each(|s| *s /= count.max(1) as f32);
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_32x32_thumbnails_for_every_fit() {
        let image = RgbImage::new(64, 48, vec![200; 64 * 48 * 3]);
        for fit in [Fit::Crop, Fit::Pad, Fit::Stretch] {
            let thumb = Preprocessor {
                fit,
                ..Default::default()
            }
            .thumbnail(&image);
            assert_eq!((thumb.width, thumb.height), (IMAGE_SIZE, IMAGE_SIZE));
        }
    }

    #[test]
    fn tta_adds_a_mirrored_view() {
        let image = RgbImage::new(8, 8, (0..192).map(|v| v as u8).collect());
        let plain = Preprocessor::default();
        let tta = Preprocessor {
            test_time_augmentation: true,
            ..plain
        };
        assert_eq!(plain.inputs(&image).len(), 1);
        assert_eq!(tta.inputs(&image).len(), 2);
    }

    #[test]
    fn averages_predictions() {
        let avg = average([vec![1.0, 0.0], vec![0.0, 1.0]].into_iter());
        assert_eq!(avg, vec![0.5, 0.5]);
    }
}
