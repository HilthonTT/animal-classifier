use super::Dataset;
use super::augment::{flip_horizontal, rotate, translate};
use crate::nn::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpandOptions {
    pub flip: bool,
    pub shift: i32,
    pub rotate: f32,
}

impl Default for ExpandOptions {
    fn default() -> Self {
        Self {
            flip: true,
            shift: 2,
            rotate: 12.0,
        }
    }
}

impl ExpandOptions {
    pub fn factor(&self) -> usize {
        1 + usize::from(self.flip) + usize::from(self.shift != 0) + usize::from(self.rotate != 0.0)
    }
}

pub fn expand(dataset: &Dataset, options: &ExpandOptions, rng: &mut Rng) -> Dataset {
    let mut out = Dataset::new(dataset.class_names.clone());
    for i in 0..dataset.len() {
        let (image, label) = (dataset.image(i), dataset.labels[i]);
        out.push(image, label);
        if options.flip {
            out.push(&flip_horizontal(image), label);
        }
        if options.shift != 0 {
            let s = options.shift;
            let (dx, dy) = [(s, 0), (-s, 0), (0, s), (0, -s)][rng.below(4)];
            out.push(&translate(image, dx, dy), label);
        }
        if options.rotate != 0.0 {
            let angle = (rng.next_f32() * 2.0 - 1.0) * options.rotate;
            out.push(&rotate(image, angle), label);
        }
    }
    out.shuffled(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::IMAGE_LEN;

    #[test]
    fn multiplies_every_class_by_the_factor() {
        let mut ds = Dataset::new(vec!["a".into(), "b".into()]);
        ds.push(&[10; IMAGE_LEN], 0);
        ds.push(&[20; IMAGE_LEN], 1);
        ds.push(&[30; IMAGE_LEN], 1);
        let options = ExpandOptions::default();
        let expanded = expand(&ds, &options, &mut Rng::new(1));
        assert_eq!(options.factor(), 4);
        assert_eq!(expanded.class_counts(), vec![4, 8]);

        let none = ExpandOptions {
            flip: false,
            shift: 0,
            rotate: 0.0,
        };
        assert_eq!(expand(&ds, &none, &mut Rng::new(1)).len(), 3);
    }
}
