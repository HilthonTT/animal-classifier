use super::{Layer, LayerKind, checked_len, parse_config};
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
        if checked_len(&[channels, in_h, in_w]).is_none() {
            return Err(Error::format("MaxPool2d layer is too large"));
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

    fn for_each_window(&self, input: &[f32], mut f: impl FnMut(usize, usize)) {
        let (oh, ow, s) = (self.out_h(), self.out_w(), self.size);
        if s == 2 {
            for c in 0..self.channels {
                let plane = c * self.in_h * self.in_w;
                for oy in 0..oh {
                    let top = plane + 2 * oy * self.in_w;
                    let bottom = top + self.in_w;
                    let (upper, lower) =
                        (&input[top..top + 2 * ow], &input[bottom..bottom + 2 * ow]);
                    for (ox, (u, l)) in upper.chunks_exact(2).zip(lower.chunks_exact(2)).enumerate()
                    {
                        let mut best = (u[0], top + 2 * ox);
                        for (v, i) in [
                            (u[1], top + 2 * ox + 1),
                            (l[0], bottom + 2 * ox),
                            (l[1], bottom + 2 * ox + 1),
                        ] {
                            if v > best.0 {
                                best = (v, i);
                            }
                        }
                        f((c * oh + oy) * ow + ox, best.1);
                    }
                }
            }
            return;
        }
        let mut best = vec![0usize; ow];
        for c in 0..self.channels {
            let plane = c * self.in_h * self.in_w;
            for oy in 0..oh {
                for dy in 0..s {
                    let row = plane + (oy * s + dy) * self.in_w;
                    for (ox, best) in best.iter_mut().enumerate() {
                        let start = row + ox * s;
                        if dy == 0 {
                            *best = start;
                        }
                        for (i, &v) in input[start..start + s].iter().enumerate() {
                            if v > input[*best] {
                                *best = start + i;
                            }
                        }
                    }
                }
                for (ox, &best) in best.iter().enumerate() {
                    f((c * oh + oy) * ow + ox, best);
                }
            }
        }
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
        self.for_each_window(input, |o, i| output[o] = input[i]);
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
        grad_input.fill(0.0);
        self.for_each_window(input, |o, i| grad_input[i] += grad_output[o]);
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

    #[test]
    fn three_by_three_windows_pick_the_maximum() {
        let pool = MaxPool2d::new(1, 3, 6, 3);
        let input = [
            1.0, 2.0, 3.0, 9.0, 0.0, 0.0, 4.0, 8.0, 5.0, 0.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0, 0.0,
            1.0,
        ];
        let mut output = [0.0; 2];
        pool.forward(&input, &mut output);
        assert_eq!(output, [8.0, 9.0]);
    }

    #[test]
    fn rejects_oversized_shapes() {
        assert!(MaxPool2d::from_config(&[u32::MAX, u32::MAX, u32::MAX, 2]).is_err());
    }
}
