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

#[derive(Clone, Debug)]
pub struct AvgPool2d {
    channels: usize,
    in_h: usize,
    in_w: usize,
    size: usize,
}

impl AvgPool2d {
    pub fn new(channels: usize, in_h: usize, in_w: usize, size: usize) -> Self {
        Self::with_shape(channels, in_h, in_w, size).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [channels, in_h, in_w, size] = parse_config(config, "AvgPool2d")?;
        Self::with_shape(channels, in_h, in_w, size)
    }

    fn with_shape(channels: usize, in_h: usize, in_w: usize, size: usize) -> Result<Self> {
        if channels == 0 || size == 0 || in_h < size || in_w < size {
            return Err(Error::format(
                "AvgPool2d window must fit inside a non-empty input",
            ));
        }
        if checked_len(&[channels, in_h, in_w]).is_none() {
            return Err(Error::format("AvgPool2d layer is too large"));
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

    fn for_each_tap(&self, mut f: impl FnMut(usize, usize)) {
        let (oh, ow, s) = (self.out_h(), self.out_w(), self.size);
        for c in 0..self.channels {
            for oy in 0..oh {
                for dy in 0..s {
                    let row = c * self.in_h * self.in_w + (oy * s + dy) * self.in_w;
                    for ox in 0..ow {
                        for dx in 0..s {
                            f((c * oh + oy) * ow + ox, row + ox * s + dx);
                        }
                    }
                }
            }
        }
    }
}

impl Layer for AvgPool2d {
    fn kind(&self) -> LayerKind {
        LayerKind::AvgPool2d
    }

    fn config(&self) -> Vec<u32> {
        [self.channels, self.in_h, self.in_w, self.size]
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    fn describe(&self) -> String {
        format!(
            "AvgPool2d {}x{}x{} -> {}x{}x{} ({s}x{s})",
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
        output.fill(0.0);
        self.for_each_tap(|o, i| output[o] += input[i]);
        let scale = 1.0 / (self.size * self.size) as f32;
        output.iter_mut().for_each(|o| *o *= scale);
    }

    fn backward(
        &self,
        _input: &[f32],
        _output: &[f32],
        _cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        _grad_params: &mut [f32],
    ) {
        let scale = 1.0 / (self.size * self.size) as f32;
        grad_input.fill(0.0);
        self.for_each_tap(|o, i| grad_input[i] = grad_output[o] * scale);
    }
}

#[derive(Clone, Debug)]
pub struct GlobalAvgPool {
    channels: usize,
    in_h: usize,
    in_w: usize,
}

impl GlobalAvgPool {
    pub fn new(channels: usize, in_h: usize, in_w: usize) -> Self {
        Self::with_shape(channels, in_h, in_w).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [channels, in_h, in_w] = parse_config(config, "GlobalAvgPool")?;
        Self::with_shape(channels, in_h, in_w)
    }

    fn with_shape(channels: usize, in_h: usize, in_w: usize) -> Result<Self> {
        if channels == 0 || in_h == 0 || in_w == 0 {
            return Err(Error::format("GlobalAvgPool dimensions must be non-zero"));
        }
        if checked_len(&[channels, in_h, in_w]).is_none() {
            return Err(Error::format("GlobalAvgPool layer is too large"));
        }
        Ok(Self {
            channels,
            in_h,
            in_w,
        })
    }

    fn plane(&self) -> usize {
        self.in_h * self.in_w
    }
}

impl Layer for GlobalAvgPool {
    fn kind(&self) -> LayerKind {
        LayerKind::GlobalAvgPool
    }

    fn config(&self) -> Vec<u32> {
        [self.channels, self.in_h, self.in_w]
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    fn describe(&self) -> String {
        format!(
            "GlobalAvgPool {}x{}x{} -> {}",
            self.channels, self.in_h, self.in_w, self.channels
        )
    }

    fn input_len(&self) -> usize {
        self.channels * self.plane()
    }

    fn output_len(&self) -> usize {
        self.channels
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let scale = 1.0 / self.plane() as f32;
        for (o, values) in output.iter_mut().zip(input.chunks_exact(self.plane())) {
            *o = values.iter().sum::<f32>() * scale;
        }
    }

    fn backward(
        &self,
        _input: &[f32],
        _output: &[f32],
        _cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        _grad_params: &mut [f32],
    ) {
        let scale = 1.0 / self.plane() as f32;
        for (&g, values) in grad_output
            .iter()
            .zip(grad_input.chunks_exact_mut(self.plane()))
        {
            values.fill(g * scale);
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
        assert!(AvgPool2d::from_config(&[u32::MAX, u32::MAX, u32::MAX, 2]).is_err());
        assert!(GlobalAvgPool::from_config(&[u32::MAX, u32::MAX, u32::MAX]).is_err());
    }

    #[test]
    fn average_pools_windows_and_spreads_gradient() {
        let pool = AvgPool2d::new(1, 2, 4, 2);
        let input = [1.0, 5.0, 2.0, 0.0, 3.0, 3.0, 8.0, 2.0];
        let mut output = [0.0; 2];
        pool.forward(&input, &mut output);
        assert_eq!(output, [3.0, 3.0]);

        let mut grad_input = [9.0; 8];
        pool.backward(&input, &output, &[], &[4.0, 8.0], &mut grad_input, &mut []);
        assert_eq!(grad_input, [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn global_average_pools_each_channel() {
        let pool = GlobalAvgPool::new(2, 1, 2);
        let mut output = [0.0; 2];
        pool.forward(&[1.0, 3.0, -2.0, 6.0], &mut output);
        assert_eq!(output, [2.0, 2.0]);

        let mut grad_input = [0.0; 4];
        pool.backward(&[], &output, &[], &[2.0, -4.0], &mut grad_input, &mut []);
        assert_eq!(grad_input, [1.0, 1.0, -2.0, -2.0]);
    }
}
