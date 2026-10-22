use super::{Layer, LayerKind, checked_len, checked_param_len, parse_config};
use crate::error::{Error, Result};
use crate::init::Init;
use crate::math::{axpy, gemm, gemm_nt, transpose, with_scratch};
use crate::rng::Rng;

/// 2-D convolution with a square kernel, stride 1 and zero padding.
///
/// Input shape `[in_channels][height][width]`, output shape
/// `[out_channels][height + 2·pad − k + 1][width + 2·pad − k + 1]`.
#[derive(Clone, Debug)]
pub struct Conv2d {
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    kernel: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
    /// Weights `[out_channels][in_channels][kernel][kernel]` followed by `out_channels` biases.
    params: Vec<f32>,
}

impl Conv2d {
    /// Creates a layer with He-initialised weights.
    pub fn new(
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        kernel: usize,
        pad: usize,
        rng: &mut Rng,
    ) -> Self {
        Self::with_init(
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel,
            pad,
            Init::He,
            rng,
        )
    }

    /// Creates a layer whose weights are filled by `init`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_init(
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        kernel: usize,
        pad: usize,
        init: Init,
        rng: &mut Rng,
    ) -> Self {
        let mut layer = Self::with_shape(in_channels, in_h, in_w, out_channels, kernel, pad)
            .unwrap_or_else(|e| panic!("{e}"));
        let n_weights = layer.weight_count();
        let area = kernel * kernel;
        init.fill(
            &mut layer.params[..n_weights],
            in_channels * area,
            out_channels * area,
            rng,
        );
        layer
    }

    pub(crate) fn from_config(config: &[u32]) -> Result<Self> {
        let [in_channels, in_h, in_w, out_channels, kernel, pad] = parse_config(config, "Conv2d")?;
        Self::with_shape(in_channels, in_h, in_w, out_channels, kernel, pad)
    }

    fn with_shape(
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        kernel: usize,
        pad: usize,
    ) -> Result<Self> {
        if in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 || kernel == 0 {
            return Err(Error::format("Conv2d dimensions must be non-zero"));
        }
        if in_h + 2 * pad < kernel || in_w + 2 * pad < kernel {
            return Err(Error::format(
                "Conv2d kernel is larger than the padded input",
            ));
        }
        let (out_h, out_w) = (in_h + 2 * pad - kernel + 1, in_w + 2 * pad - kernel + 1);
        let fits = checked_len(&[in_channels, in_h, in_w]).is_some()
            && checked_len(&[out_channels, out_h, out_w]).is_some()
            && checked_param_len(&[out_channels, in_channels, kernel, kernel], out_channels)
                .is_some();
        if !fits {
            return Err(Error::format("Conv2d layer is too large"));
        }
        let mut layer = Self {
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel,
            pad,
            out_h,
            out_w,
            params: Vec::new(),
        };
        layer.params = vec![0.0; layer.weight_count() + out_channels];
        Ok(layer)
    }

    fn weight_count(&self) -> usize {
        self.out_channels * self.in_channels * self.kernel * self.kernel
    }

    /// Output positions `o` for which the input position `o + k − pad` lies in `[0, in_len)`.
    fn valid_range(k: usize, pad: usize, in_len: usize, out_len: usize) -> (usize, usize) {
        let start = pad.saturating_sub(k);
        let end = (in_len + pad).saturating_sub(k).min(out_len);
        (start, end.max(start))
    }

    fn patch_len(&self) -> usize {
        self.in_channels * self.kernel * self.kernel
    }

    fn for_each_row(&self, mut f: impl FnMut(usize, usize, usize, usize)) {
        let k = self.kernel;
        for ic in 0..self.in_channels {
            for ky in 0..k {
                for kx in 0..k {
                    f((ic * k + ky) * k + kx, ic, ky, kx);
                }
            }
        }
    }

    fn im2col(&self, input: &[f32], cols: &mut [f32]) {
        let (in_plane, out_plane) = (self.in_h * self.in_w, self.out_h * self.out_w);
        cols.fill(0.0);
        self.for_each_row(|row, ic, ky, kx| {
            let (y0, y1) = Self::valid_range(ky, self.pad, self.in_h, self.out_h);
            let (x0, x1) = Self::valid_range(kx, self.pad, self.in_w, self.out_w);
            if y0 >= y1 || x0 >= x1 {
                return;
            }
            let src_plane = &input[ic * in_plane..(ic + 1) * in_plane];
            let dst_plane = &mut cols[row * out_plane..(row + 1) * out_plane];
            for oy in y0..y1 {
                let src = (oy + ky - self.pad) * self.in_w + (x0 + kx - self.pad);
                let dst = oy * self.out_w + x0;
                dst_plane[dst..dst + x1 - x0].copy_from_slice(&src_plane[src..src + x1 - x0]);
            }
        });
    }

    fn col2im(&self, cols: &[f32], grad_input: &mut [f32]) {
        let (in_plane, out_plane) = (self.in_h * self.in_w, self.out_h * self.out_w);
        grad_input.fill(0.0);
        self.for_each_row(|row, ic, ky, kx| {
            let (y0, y1) = Self::valid_range(ky, self.pad, self.in_h, self.out_h);
            let (x0, x1) = Self::valid_range(kx, self.pad, self.in_w, self.out_w);
            if y0 >= y1 || x0 >= x1 {
                return;
            }
            let src_plane = &cols[row * out_plane..(row + 1) * out_plane];
            let dst_plane = &mut grad_input[ic * in_plane..(ic + 1) * in_plane];
            for oy in y0..y1 {
                let dst = (oy + ky - self.pad) * self.in_w + (x0 + kx - self.pad);
                let src = oy * self.out_w + x0;
                axpy(
                    1.0,
                    &src_plane[src..src + x1 - x0],
                    &mut dst_plane[dst..dst + x1 - x0],
                );
            }
        });
    }

    fn backward_cols(
        &self,
        cols: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        grad_params: &mut [f32],
    ) {
        let n_weights = self.weight_count();
        let (patch, out_plane) = (self.patch_len(), self.out_h * self.out_w);
        let (grad_weights, grad_bias) = grad_params.split_at_mut(n_weights);
        for (go, gb) in grad_output.chunks_exact(out_plane).zip(grad_bias) {
            *gb += go.iter().sum::<f32>();
        }
        if patch < 64 {
            gemm_nt(
                self.out_channels,
                patch,
                out_plane,
                grad_output,
                cols,
                grad_weights,
            );
        } else {
            with_scratch(patch * out_plane, |cols_t| {
                transpose(patch, out_plane, cols, cols_t);
                gemm(
                    self.out_channels,
                    patch,
                    out_plane,
                    grad_output,
                    cols_t,
                    grad_weights,
                );
            });
        }
        with_scratch(n_weights, |weights_t| {
            transpose(
                self.out_channels,
                patch,
                &self.params[..n_weights],
                weights_t,
            );
            with_scratch(patch * out_plane, |grad_cols| {
                gemm(
                    patch,
                    out_plane,
                    self.out_channels,
                    weights_t,
                    grad_output,
                    grad_cols,
                );
                self.col2im(grad_cols, grad_input);
            });
        });
    }

    fn forward_cols(&self, cols: &[f32], output: &mut [f32]) {
        let (weights, bias) = self.params.split_at(self.weight_count());
        let out_plane = self.out_h * self.out_w;
        for (out, &b) in output.chunks_exact_mut(out_plane).zip(bias) {
            out.fill(b);
        }
        gemm(
            self.out_channels,
            out_plane,
            self.patch_len(),
            weights,
            cols,
            output,
        );
    }
}

impl Layer for Conv2d {
    fn kind(&self) -> LayerKind {
        LayerKind::Conv2d
    }

    fn config(&self) -> Vec<u32> {
        [
            self.in_channels,
            self.in_h,
            self.in_w,
            self.out_channels,
            self.kernel,
            self.pad,
        ]
        .iter()
        .map(|&v| v as u32)
        .collect()
    }

    fn describe(&self) -> String {
        format!(
            "Conv2d {}x{}x{} -> {}x{}x{} ({k}x{k}, pad {})",
            self.in_channels,
            self.in_h,
            self.in_w,
            self.out_channels,
            self.out_h,
            self.out_w,
            self.pad,
            k = self.kernel
        )
    }

    fn input_len(&self) -> usize {
        self.in_channels * self.in_h * self.in_w
    }

    fn output_len(&self) -> usize {
        self.out_channels * self.out_h * self.out_w
    }

    fn params(&self) -> &[f32] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn weight_len(&self) -> usize {
        self.weight_count()
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        with_scratch(self.patch_len() * self.out_h * self.out_w, |cols| {
            self.im2col(input, cols);
            self.forward_cols(cols, output);
        });
    }

    fn forward_train(
        &self,
        input: &[f32],
        output: &mut [f32],
        _rng: &mut Rng,
        cache: &mut Vec<f32>,
    ) {
        cache.clear();
        cache.resize(self.patch_len() * self.out_h * self.out_w, 0.0);
        self.im2col(input, cache);
        self.forward_cols(cache, output);
    }

    fn backward(
        &self,
        input: &[f32],
        _output: &[f32],
        cache: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        grad_params: &mut [f32],
    ) {
        let (patch, out_plane) = (self.patch_len(), self.out_h * self.out_w);
        if cache.len() == patch * out_plane {
            self.backward_cols(cache, grad_output, grad_input, grad_params);
        } else {
            with_scratch(patch * out_plane, |cols| {
                self.im2col(input, cols);
                self.backward_cols(cols, grad_output, grad_input, grad_params);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_kernel_copies_input() {
        let mut conv = Conv2d::from_config(&[1, 3, 3, 1, 3, 1]).unwrap();
        conv.params_mut()[4] = 1.0; // centre tap
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let mut output = vec![0.0; 9];
        conv.forward(&input, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn valid_convolution_sums_window() {
        let mut conv = Conv2d::from_config(&[1, 3, 3, 1, 2, 0]).unwrap();
        conv.params_mut()[..4].fill(1.0);
        conv.params_mut()[4] = 0.5; // bias
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let mut output = vec![0.0; 4];
        conv.forward(&input, &mut output);
        assert_eq!(output, vec![12.5, 16.5, 24.5, 28.5]);
    }

    #[test]
    fn rejects_oversized_shapes() {
        assert!(Conv2d::from_config(&[u32::MAX, u32::MAX, u32::MAX, 1, 1, 0]).is_err());
        assert!(Conv2d::from_config(&[1, 4, 4, u32::MAX, u32::MAX, 0]).is_err());
    }
}
