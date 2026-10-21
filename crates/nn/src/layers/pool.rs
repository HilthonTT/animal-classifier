use super::{Layer, LayerKind, parse_config};
use crate::error::{Error, Result};

/// Max pooling over non-overlapping `size × size` windows (stride = size).
#[derive(Clone, Debug)]
pub struct MaxPool2d {
    channels: usize,
    in_h: usize,
    in_w: usize,
    size: usize,
}

impl MaxPool2d {
    pub fn new(channels: usize, in_h: usize, in_w: usize, size: usize) -> Self {
        Self::with_shape(channels, in_h, in_w, size).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [channels, in_h, in_w, size] = parse_config(config, "MaxPool2d")?;
        Self::with_shape(channels, in_h, in_w, size)
    }

    fn with_shape(channels: usize, in_h: usize, in_w: usize, size: usize) -> Result<Self> {
        if channels == 0 || size == 0 || in_h < size || in_w < size {
            return Err(Error::format(
                "MaxPool2d window must fit inside a non-empty input",
            ));
        }
        Ok(Self {
            channels,
            in_h,
            in_w,
            size,
        })
    }

    fn out_h(&self) -> usize {
        self.in_h / self.size
    }

    fn out_w(&self) -> usize {
        self.in_w / self.size
    }

    /// Index (into `input`) of the maximum of the window feeding output `(c, oy, ox)`.
    fn argmax(&self, input: &[f32], c: usize, oy: usize, ox: usize) -> usize {
        let plane = c * self.in_h * self.in_w;
        let mut best = plane + oy * self.size * self.in_w + ox * self.size;
        for dy in 0..self.size {
            for dx in 0..self.size {
                let i = plane + (oy * self.size + dy) * self.in_w + ox * self.size + dx;
                if input[i] > input[best] {
                    best = i;
                }
            }
        }
        best
    }
}

impl Layer for MaxPool2d {
    fn kind(&self) -> LayerKind {
        LayerKind::MaxPool2d
    }

    fn config(&self) -> Vec<u32> {
        [self.channels, self.in_h, self.in_w, self.size]
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    fn describe(&self) -> String {
        format!(
            "MaxPool2d {}x{}x{} -> {}x{}x{} ({s}x{s})",
            self.channels,
            self.in_h,
            self.in_w,
            self.channels,
            self.out_h(),
            self.out_w(),
            s = self.size
        )
    }

    fn input_len(&self) -> usize {
        self.channels * self.in_h * self.in_w
    }

    fn output_len(&self) -> usize {
        self.channels * self.out_h() * self.out_w()
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let (oh, ow) = (self.out_h(), self.out_w());
        for c in 0..self.channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    output[(c * oh + oy) * ow + ox] = input[self.argmax(input, c, oy, ox)];
                }
            }
        }
    }

    fn backward(
        &self,
        input: &[f32],
        _output: &[f32],
        _cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        _grad_params: &mut [f32],
    ) {
        // Only the winning input of each window receives the gradient.
        let (oh, ow) = (self.out_h(), self.out_w());
        grad_input.fill(0.0);
        for c in 0..self.channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    grad_input[self.argmax(input, c, oy, ox)] +=
                        grad_output[(c * oh + oy) * ow + ox];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_window_maximum_and_routes_gradient() {
        let pool = MaxPool2d::new(1, 2, 4, 2);
        let input = [1.0, 5.0, 2.0, 0.0, 3.0, 4.0, 8.0, 1.0];
        let mut output = [0.0; 2];
        pool.forward(&input, &mut output);
        assert_eq!(output, [5.0, 8.0]);

        let mut grad_input = [9.0; 8];
        pool.backward(&input, &output, &[], &[1.0, 2.0], &mut grad_input, &mut []);
        assert_eq!(grad_input, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }
}
