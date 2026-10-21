#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfusionMatrix {
    classes: usize,
    counts: Vec<usize>,
}

impl ConfusionMatrix {
    pub fn new(classes: usize) -> Self {
        Self {
            classes,
            counts: vec![0; classes * classes],
        }
    }

    pub fn record(&mut self, actual: usize, predicted: usize) {
        self.counts[actual * self.classes + predicted] += 1;
    }

    pub fn merge(&mut self, other: &ConfusionMatrix) {
        assert_eq!(self.classes, other.classes);
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
    }

    pub fn count(&self, actual: usize, predicted: usize) -> usize {
        self.counts[actual * self.classes + predicted]
    }

    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    pub fn correct(&self) -> usize {
        (0..self.classes).map(|c| self.count(c, c)).sum()
    }

    pub fn accuracy(&self) -> f32 {
        ratio(self.correct(), self.total())
    }

    pub fn recall(&self, class: usize) -> f32 {
        ratio(
            self.count(class, class),
            (0..self.classes).map(|p| self.count(class, p)).sum(),
        )
    }

    pub fn precision(&self, class: usize) -> f32 {
        ratio(
            self.count(class, class),
            (0..self.classes).map(|a| self.count(a, class)).sum(),
        )
    }

    pub fn render(&self, labels: &[String]) -> String {
        let name = |i: usize| labels.get(i).map_or_else(|| i.to_string(), |s| s.clone());
        let mut out = format!("{:<20}", "actual \\ predicted");
        for p in 0..self.classes {
            out += &format!("{:>8}", name(p));
        }
        out += &format!("{:>10}\n", "recall");
        for a in 0..self.classes {
            out += &format!("{:<20}", name(a));
            for p in 0..self.classes {
                out += &format!("{:>8}", self.count(a, p));
            }
            out += &format!("{:>9.1}%\n", self.recall(a) * 100.0);
        }
        out += &format!("{:<20}", "precision");
        for p in 0..self.classes {
            out += &format!("{:>7.1}%", self.precision(p) * 100.0);
        }
        out += &format!(
            "\n\naccuracy: {:.2}% ({} / {})\n",
            self.accuracy() * 100.0,
            self.correct(),
            self.total()
        );
        out
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_accuracy_precision_recall() {
        let mut cm = ConfusionMatrix::new(2);
        cm.record(0, 0);
        cm.record(0, 0);
        cm.record(0, 1);
        cm.record(1, 1);
        assert_eq!(cm.accuracy(), 0.75);
        assert!((cm.recall(0) - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(cm.precision(1), 0.5);

        let mut other = ConfusionMatrix::new(2);
        other.record(1, 0);
        cm.merge(&other);
        assert_eq!(cm.total(), 5);
        assert!(
            cm.render(&["a".into(), "b".into()])
                .contains("accuracy: 60.00%")
        );
    }
}
