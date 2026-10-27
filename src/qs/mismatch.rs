//! FFT-based mismatch field for Quick Sampling.
//!
//! Three performance optimisations applied (cumulative):
//!
//! 1. **Real-input FFT** (R2C / C2R via `realfft` on top of `rustfft`). The
//!    X-axis transform is half-size complex; Y and Z stay complex.
//! 2. **Partial FFT** for the per-target kernel. Rows / z-slabs with no
//!    non-zero data are skipped on the forward sweep. With a small template
//!    in a large padded grid, the kernel's support is <1% of the buffer.
//! 3. **Fusion**: the multiple cross-correlations needed per target (two for
//!    continuous, `n_classes` for categorical) are summed in the frequency
//!    domain and inverted in **one** C2R, instead of computing each
//!    spatial-domain term separately. Saves one IFFT per target on
//!    continuous; saves `n_classes - 1` IFFTs on categorical.
//!
//! All buffers are `f32` for ~2× speed and memory bandwidth over `f64`. The
//! public API still hands back `Grid3<f64>` (converted at the boundary), so
//! downstream code is unaffected. Numerical precision is sufficient for QS:
//! the algorithm is rank-based (top-K), not a precision-sensitive numerical
//! integration.

use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use super::grid::{next_pow2, Grid3};
use super::template::QsTemplate;

/// Precomputed FFT context for a fixed TI shape. All buffers owned and
/// reused across every target call.
pub struct FftCtx {
    pub nx: usize, // TI dims
    pub ny: usize,
    pub nz: usize,
    pub ex: usize, // template extents
    pub ey: usize,
    pub ez: usize,
    pub px: usize, // padded real size
    pub py: usize,
    pub pz: usize,
    pub px_c: usize, // = px/2 + 1

    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    fft_y_fwd: Arc<dyn Fft<f32>>,
    fft_y_inv: Arc<dyn Fft<f32>>,
    fft_z_fwd: Arc<dyn Fft<f32>>,
    fft_z_inv: Arc<dyn Fft<f32>>,

    /// Padded real working buffer, `px * py * pz`.
    pub real_scratch: Vec<f32>,
    /// Frequency-domain working buffer, `px_c * py * pz`.
    pub freq_scratch: Vec<Complex<f32>>,
    /// Frequency-domain *accumulator*, same shape as `freq_scratch`. Used
    /// by the fusion path: multiple kernels are MACC'd into here before a
    /// single inverse FFT.
    pub freq_accum: Vec<Complex<f32>>,

    /// Per-call 1D FFT scratch.
    scratch_x_complex: Vec<Complex<f32>>,
    scratch_y: Vec<Complex<f32>>,
    scratch_z: Vec<Complex<f32>>,
    /// realfft-required scratch.
    r2c_scratch: Vec<Complex<f32>>,
    c2r_scratch: Vec<Complex<f32>>,
}

impl FftCtx {
    pub fn new(ti_dims: (usize, usize, usize), template_extents: (i32, i32, i32)) -> Self {
        let (nx, ny, nz) = ti_dims;
        let (ex, ey, ez) = (
            template_extents.0 as usize,
            template_extents.1 as usize,
            template_extents.2 as usize,
        );
        let px = next_pow2(nx + 2 * ex);
        let py = next_pow2(ny + 2 * ey);
        let pz = next_pow2(nz + 2 * ez);
        let px_c = px / 2 + 1;

        let mut real_planner = RealFftPlanner::<f32>::new();
        let mut complex_planner = FftPlanner::<f32>::new();
        let r2c = real_planner.plan_fft_forward(px);
        let c2r = real_planner.plan_fft_inverse(px);
        let fft_y_fwd = complex_planner.plan_fft_forward(py);
        let fft_y_inv = complex_planner.plan_fft_inverse(py);
        let fft_z_fwd = complex_planner.plan_fft_forward(pz);
        let fft_z_inv = complex_planner.plan_fft_inverse(pz);

        let r2c_scratch_len = r2c.get_scratch_len();
        let c2r_scratch_len = c2r.get_scratch_len();

        Self {
            nx,
            ny,
            nz,
            ex,
            ey,
            ez,
            px,
            py,
            pz,
            px_c,
            r2c,
            c2r,
            fft_y_fwd,
            fft_y_inv,
            fft_z_fwd,
            fft_z_inv,
            real_scratch: vec![0.0; px * py * pz],
            freq_scratch: vec![Complex::default(); px_c * py * pz],
            freq_accum: vec![Complex::default(); px_c * py * pz],
            scratch_x_complex: vec![Complex::default(); px_c],
            scratch_y: vec![Complex::default(); py],
            scratch_z: vec![Complex::default(); pz],
            r2c_scratch: vec![Complex::default(); r2c_scratch_len],
            c2r_scratch: vec![Complex::default(); c2r_scratch_len],
        }
    }

    pub fn padded_real_len(&self) -> usize {
        self.px * self.py * self.pz
    }
    pub fn padded_complex_len(&self) -> usize {
        self.px_c * self.py * self.pz
    }

    /// Place a real grid `g` into the padded real buffer; rest is zero.
    pub fn embed_real(&self, g: &Grid3<f64>, out: &mut [f32]) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for z in 0..self.nz {
            for y in 0..self.ny {
                let dst_base = (z * self.py + y) * self.px;
                let src_base = (z * self.ny + y) * self.nx;
                for x in 0..self.nx {
                    out[dst_base + x] = g.data[src_base + x] as f32;
                }
            }
        }
    }

    /// Place an indicator grid (`{0, 1}` ⊂ `u32`) into the padded real buffer
    /// as `0.0` / `1.0`.
    pub fn embed_indicator(&self, src: &[f64], out: &mut [f32]) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for z in 0..self.nz {
            for y in 0..self.ny {
                let dst_base = (z * self.py + y) * self.px;
                let src_base = (z * self.ny + y) * self.nx;
                for x in 0..self.nx {
                    out[dst_base + x] = src[src_base + x] as f32;
                }
            }
        }
    }

    /// Forward 3D R2C FFT (full). Input consumed.
    pub fn fft3_r2c(&mut self, real_in: &mut [f32], complex_out: &mut [Complex<f32>]) {
        self.fft3_r2c_inner(real_in, complex_out, None);
    }

    /// Partial forward 3D R2C FFT. Rows/slabs flagged `false` are skipped.
    pub fn fft3_r2c_partial(
        &mut self,
        real_in: &mut [f32],
        complex_out: &mut [Complex<f32>],
        nonempty_rows: &[bool],
        nonempty_slabs: &[bool],
    ) {
        self.fft3_r2c_inner(real_in, complex_out, Some((nonempty_rows, nonempty_slabs)));
    }

    fn fft3_r2c_inner(
        &mut self,
        real_in: &mut [f32],
        complex_out: &mut [Complex<f32>],
        partial: Option<(&[bool], &[bool])>,
    ) {
        let (px, py, pz, px_c) = (self.px, self.py, self.pz, self.px_c);
        assert_eq!(real_in.len(), px * py * pz);
        assert_eq!(complex_out.len(), px_c * py * pz);

        for c in complex_out.iter_mut() {
            *c = Complex::default();
        }

        // X-axis R2C (skip empty rows when partial).
        for z in 0..pz {
            for y in 0..py {
                if let Some((nonempty_rows, _)) = partial {
                    if !nonempty_rows[z * py + y] {
                        continue;
                    }
                }
                let src_base = (z * py + y) * px;
                let dst_base = (z * py + y) * px_c;
                self.r2c
                    .process_with_scratch(
                        &mut real_in[src_base..src_base + px],
                        &mut complex_out[dst_base..dst_base + px_c],
                        &mut self.r2c_scratch,
                    )
                    .expect("R2C FFT failed");
            }
        }

        // Y-axis C2C (skip slabs that ended up all zero when partial).
        for z in 0..pz {
            if let Some((_, nonempty_slabs)) = partial {
                if !nonempty_slabs[z] {
                    continue;
                }
            }
            for xc in 0..px_c {
                for y in 0..py {
                    self.scratch_y[y] = complex_out[(z * py + y) * px_c + xc];
                }
                self.fft_y_fwd.process(&mut self.scratch_y);
                for y in 0..py {
                    complex_out[(z * py + y) * px_c + xc] = self.scratch_y[y];
                }
            }
        }

        // Z-axis C2C (must run on all (xc, y); kernel data spreads through z).
        for y in 0..py {
            for xc in 0..px_c {
                for z in 0..pz {
                    self.scratch_z[z] = complex_out[(z * py + y) * px_c + xc];
                }
                self.fft_z_fwd.process(&mut self.scratch_z);
                for z in 0..pz {
                    complex_out[(z * py + y) * px_c + xc] = self.scratch_z[z];
                }
            }
        }
    }

    /// Inverse 3D C2R FFT. Input consumed; output normalised by the padded
    /// size (so `IFFT(FFT(x)) == x`).
    pub fn fft3_c2r(&mut self, complex_in: &mut [Complex<f32>], real_out: &mut [f32]) {
        let (px, py, pz, px_c) = (self.px, self.py, self.pz, self.px_c);
        assert_eq!(complex_in.len(), px_c * py * pz);
        assert_eq!(real_out.len(), px * py * pz);

        // Z inverse.
        for y in 0..py {
            for xc in 0..px_c {
                for z in 0..pz {
                    self.scratch_z[z] = complex_in[(z * py + y) * px_c + xc];
                }
                self.fft_z_inv.process(&mut self.scratch_z);
                for z in 0..pz {
                    complex_in[(z * py + y) * px_c + xc] = self.scratch_z[z];
                }
            }
        }
        // Y inverse.
        for z in 0..pz {
            for xc in 0..px_c {
                for y in 0..py {
                    self.scratch_y[y] = complex_in[(z * py + y) * px_c + xc];
                }
                self.fft_y_inv.process(&mut self.scratch_y);
                for y in 0..py {
                    complex_in[(z * py + y) * px_c + xc] = self.scratch_y[y];
                }
            }
        }
        // X axis: realfft requires DC and Nyquist to be real. Floating
        // point noise from the multiply + y/z inverse leaves residual
        // imaginary parts; we zero them.
        for z in 0..pz {
            for y in 0..py {
                let base = (z * py + y) * px_c;
                complex_in[base].im = 0.0;
                complex_in[base + px_c - 1].im = 0.0;
            }
        }
        for z in 0..pz {
            for y in 0..py {
                let src_base = (z * py + y) * px_c;
                let dst_base = (z * py + y) * px;
                self.scratch_x_complex
                    .copy_from_slice(&complex_in[src_base..src_base + px_c]);
                self.c2r
                    .process_with_scratch(
                        &mut self.scratch_x_complex,
                        &mut real_out[dst_base..dst_base + px],
                        &mut self.c2r_scratch,
                    )
                    .expect("C2R FFT failed");
            }
        }
        let scale = 1.0 / (px * py * pz) as f32;
        for v in real_out.iter_mut() {
            *v *= scale;
        }
    }

    /// Reset the frequency-domain accumulator.
    #[inline]
    pub fn reset_accum(&mut self) {
        for c in self.freq_accum.iter_mut() {
            *c = Complex::default();
        }
    }

    /// `freq_accum += alpha * ti_freq * conj(kernel_freq)`.
    ///
    /// Hot loop: this is the inner product over the freq-domain buffer that
    /// dominates per-target cost. Written so that LLVM with `target-cpu=
    /// native` auto-vectorises into AVX2 fused-multiply-adds on f32x8.
    #[inline]
    pub fn macc_accum(
        &mut self,
        ti_freq: &[Complex<f32>],
        kernel_freq: &[Complex<f32>],
        alpha: f32,
    ) {
        let n = self.freq_accum.len();
        assert_eq!(ti_freq.len(), n);
        assert_eq!(kernel_freq.len(), n);
        let acc = &mut self.freq_accum[..n];
        let t = &ti_freq[..n];
        let k = &kernel_freq[..n];
        for i in 0..n {
            // t[i] * conj(k[i]) = (t.re + i t.im) * (k.re - i k.im)
            //   = (t.re * k.re + t.im * k.im) + i (t.im * k.re - t.re * k.im)
            let re = t[i].re * k[i].re + t[i].im * k[i].im;
            let im = t[i].im * k[i].re - t[i].re * k[i].im;
            acc[i].re += alpha * re;
            acc[i].im += alpha * im;
        }
    }

    /// Inverse FFT of `freq_accum` -> `real_scratch`. After this the
    /// padded-real buffer holds the spatial correlation field.
    pub fn invert_accum(&mut self) {
        let mut real = std::mem::take(&mut self.real_scratch);
        let mut freq = std::mem::take(&mut self.freq_accum);
        self.fft3_c2r(&mut freq, &mut real);
        self.freq_accum = freq;
        self.real_scratch = real;
    }

    /// Copy the valid (TI-shape) region of `self.real_scratch` into
    /// `out_grid`, converting back to f64.
    pub fn extract_into_f64(&self, out_grid: &mut Grid3<f64>) {
        assert_eq!(out_grid.nx, self.nx);
        assert_eq!(out_grid.ny, self.ny);
        assert_eq!(out_grid.nz, self.nz);
        for z in 0..self.nz {
            for y in 0..self.ny {
                let dst_base = (z * self.ny + y) * self.nx;
                let src_base = (z * self.py + y) * self.px;
                for x in 0..self.nx {
                    out_grid.data[dst_base + x] = self.real_scratch[src_base + x] as f64;
                }
            }
        }
    }
}

#[inline]
fn wrap(d: i32, size: usize) -> usize {
    let s = size as i32;
    let mut v = d % s;
    if v < 0 {
        v += s;
    }
    v as usize
}

/// Frequency-domain precomputations for a TI. Built once per realization.
pub struct TiFreq {
    pub t: Vec<Complex<f32>>,
    pub t2: Vec<Complex<f32>>,
    pub indicators: Vec<Vec<Complex<f32>>>,
}

impl TiFreq {
    pub fn for_continuous(ti: &Grid3<f64>, ctx: &mut FftCtx) -> Self {
        let n_freq = ctx.padded_complex_len();
        let n_real = ctx.padded_real_len();
        let mut real_buf = vec![0.0f32; n_real];
        let mut t = vec![Complex::default(); n_freq];

        ctx.embed_real(ti, &mut real_buf);
        ctx.fft3_r2c(&mut real_buf, &mut t);

        ctx.embed_real(ti, &mut real_buf);
        for v in real_buf.iter_mut() {
            *v *= *v;
        }
        let mut t2 = vec![Complex::default(); n_freq];
        ctx.fft3_r2c(&mut real_buf, &mut t2);

        Self {
            t,
            t2,
            indicators: Vec::new(),
        }
    }

    pub fn for_categorical(ti: &Grid3<u32>, n_classes: u32, ctx: &mut FftCtx) -> Self {
        let n_freq = ctx.padded_complex_len();
        let n_real = ctx.padded_real_len();
        let mut indicators = Vec::with_capacity(n_classes as usize);
        let mut real_buf = vec![0.0f32; n_real];
        for k in 0..n_classes {
            // Build indicator directly to avoid intermediate Grid3<f64>.
            for v in real_buf.iter_mut() {
                *v = 0.0;
            }
            for z in 0..ti.nz {
                for y in 0..ti.ny {
                    let dst_base = (z * ctx.py + y) * ctx.px;
                    let src_base = (z * ti.ny + y) * ti.nx;
                    for x in 0..ti.nx {
                        real_buf[dst_base + x] = if ti.data[src_base + x] == k { 1.0 } else { 0.0 };
                    }
                }
            }
            let mut freq = vec![Complex::default(); n_freq];
            ctx.fft3_r2c(&mut real_buf, &mut freq);
            indicators.push(freq);
        }
        Self {
            t: Vec::new(),
            t2: Vec::new(),
            indicators,
        }
    }
}

/// Build a kernel into `real_scratch` from a closure returning the per-lag
/// real value. Returns `(nonempty_rows, nonempty_slabs)` for partial FFT.
fn build_kernel<F: Fn(usize, &super::template::LagWeight) -> f32>(
    ctx: &mut FftCtx,
    template: &QsTemplate,
    value_for: F,
) -> (Vec<bool>, Vec<bool>) {
    let (px, py, pz) = (ctx.px, ctx.py, ctx.pz);
    for o in ctx.real_scratch.iter_mut() {
        *o = 0.0;
    }
    let mut nonempty_rows = vec![false; py * pz];
    let mut nonempty_slabs = vec![false; pz];
    for (i, lag) in template.lags.iter().enumerate() {
        let v = value_for(i, lag);
        if v == 0.0 {
            continue;
        }
        let kx = wrap(lag.dx, px);
        let ky = wrap(lag.dy, py);
        let kz = wrap(lag.dz, pz);
        let idx = (kz * py + ky) * px + kx;
        ctx.real_scratch[idx] += v;
        nonempty_rows[kz * py + ky] = true;
        nonempty_slabs[kz] = true;
    }
    (nonempty_rows, nonempty_slabs)
}

/// Fused continuous mismatch — single IFFT for both `T²⊛W` and `T⊛Wo` terms.
///
/// In freq-space:
///   accum = T²_freq * conj(W_freq) - 2 * T_freq * conj(WO_freq)
/// Then `IFFT(accum)` is exactly `term1 - 2*term2` in spatial space.
/// Adding `const_term` and dividing by `wsum` gives the final mismatch.
pub fn mismatch_continuous(
    ctx: &mut FftCtx,
    ti_freq: &TiFreq,
    template: &QsTemplate,
    obs: &[f64],
    mask: &[f64],
    out: &mut Grid3<f64>,
    _scratch_unused: &mut Vec<Complex<f32>>,
) {
    assert_eq!(obs.len(), template.lags.len());
    assert_eq!(mask.len(), template.lags.len());

    let mut const_term: f32 = 0.0;
    let mut wsum: f32 = 0.0;
    for (i, lag) in template.lags.iter().enumerate() {
        let mw = (mask[i] * lag.weight) as f32;
        let o = obs[i] as f32;
        const_term += mw * o * o;
        wsum += mw;
    }
    let inv_w = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };

    ctx.reset_accum();

    // --- Term 1: T²_freq * conj(W) ---
    let (rows1, slabs1) = build_kernel(ctx, template, |i, lag| (mask[i] * lag.weight) as f32);
    let mut real = std::mem::take(&mut ctx.real_scratch);
    let mut freq = std::mem::take(&mut ctx.freq_scratch);
    ctx.fft3_r2c_partial(&mut real, &mut freq, &rows1, &slabs1);
    ctx.real_scratch = real;
    ctx.macc_accum(&ti_freq.t2, &freq, 1.0);
    ctx.freq_scratch = freq;

    // --- Term 2: -2 * T_freq * conj(WO) ---
    let (rows2, slabs2) = build_kernel(ctx, template, |i, lag| {
        (mask[i] * lag.weight * obs[i]) as f32
    });
    let mut real = std::mem::take(&mut ctx.real_scratch);
    let mut freq = std::mem::take(&mut ctx.freq_scratch);
    ctx.fft3_r2c_partial(&mut real, &mut freq, &rows2, &slabs2);
    ctx.real_scratch = real;
    ctx.macc_accum(&ti_freq.t, &freq, -2.0);
    ctx.freq_scratch = freq;

    // --- Single inverse FFT of the accumulated freq buffer. ---
    ctx.invert_accum();

    // Copy + finalise.
    let (nx, ny, nz, px, py) = (ctx.nx, ctx.ny, ctx.nz, ctx.px, ctx.py);
    for z in 0..nz {
        for y in 0..ny {
            let dst_base = (z * ny + y) * nx;
            let src_base = (z * py + y) * px;
            for x in 0..nx {
                let m = (ctx.real_scratch[src_base + x] + const_term) * inv_w;
                out.data[dst_base + x] = m.max(0.0) as f64;
            }
        }
    }
}

/// Fused categorical mismatch — single IFFT for all classes.
///
/// In freq-space:
///   accum = sum_k I_k_freq * conj(W_k_freq)
/// where `W_k` carries the lag weights at lags where `obs == k`. Then
/// `IFFT(accum)` is the *total match count* in spatial space, and
/// `mismatch = (wsum - match) / wsum`.
pub fn mismatch_categorical(
    ctx: &mut FftCtx,
    ti_freq: &TiFreq,
    template: &QsTemplate,
    obs: &[u32],
    mask: &[f64],
    out: &mut Grid3<f64>,
) {
    assert_eq!(obs.len(), template.lags.len());
    assert_eq!(mask.len(), template.lags.len());
    let n_classes = ti_freq.indicators.len();

    ctx.reset_accum();

    for k in 0..n_classes {
        let (rows, slabs) = build_kernel(ctx, template, |i, lag| {
            if obs[i] as usize == k && mask[i] > 0.0 {
                (mask[i] * lag.weight) as f32
            } else {
                0.0
            }
        });
        // Skip if all template lags missed class k.
        let any_row = rows.iter().any(|&r| r);
        if !any_row {
            continue;
        }
        let mut real = std::mem::take(&mut ctx.real_scratch);
        let mut freq = std::mem::take(&mut ctx.freq_scratch);
        ctx.fft3_r2c_partial(&mut real, &mut freq, &rows, &slabs);
        ctx.real_scratch = real;
        ctx.macc_accum(&ti_freq.indicators[k], &freq, 1.0);
        ctx.freq_scratch = freq;
    }

    ctx.invert_accum();

    let mut wsum: f32 = 0.0;
    for (i, lag) in template.lags.iter().enumerate() {
        wsum += (mask[i] * lag.weight) as f32;
    }
    let inv_w = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };

    let (nx, ny, nz, px, py) = (ctx.nx, ctx.ny, ctx.nz, ctx.px, ctx.py);
    for z in 0..nz {
        for y in 0..ny {
            let dst_base = (z * ny + y) * nx;
            let src_base = (z * py + y) * px;
            for x in 0..nx {
                let match_count = ctx.real_scratch[src_base + x];
                let mismatch = (wsum - match_count) * inv_w;
                out.data[dst_base + x] = mismatch.clamp(0.0, 1.0) as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qs::template::{LagWeight, QsTemplate};

    fn naive_continuous_mismatch(
        ti: &Grid3<f64>,
        template: &QsTemplate,
        obs: &[f64],
        mask: &[f64],
    ) -> Grid3<f64> {
        let mut out = Grid3::filled(ti.nx, ti.ny, ti.nz, 0.0);
        let mut wsum = 0.0;
        for (i, lag) in template.lags.iter().enumerate() {
            wsum += mask[i] * lag.weight;
        }
        let inv_w = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
        for z in 0..ti.nz {
            for y in 0..ti.ny {
                for x in 0..ti.nx {
                    let mut s = 0.0;
                    for (i, lag) in template.lags.iter().enumerate() {
                        let xx = x as i64 + lag.dx as i64;
                        let yy = y as i64 + lag.dy as i64;
                        let zz = z as i64 + lag.dz as i64;
                        if !ti.in_bounds(xx, yy, zz) {
                            continue;
                        }
                        let t = *ti.get(xx as usize, yy as usize, zz as usize);
                        let d = t - obs[i];
                        s += mask[i] * lag.weight * d * d;
                    }
                    *out.get_mut(x, y, z) = s * inv_w;
                }
            }
        }
        out
    }

    // f32 round-trip tolerance — looser than the previous f64 path.
    const F32_TOL: f64 = 1e-4;

    #[test]
    fn f32_fft_continuous_matches_naive_in_interior() {
        let (nx, ny, nz) = (8, 8, 4);
        let mut ti = Grid3::<f64>::new(nx, ny, nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    *ti.get_mut(x, y, z) = (x as f64) * 0.1 + (y as f64) * 0.07 + (z as f64) * 0.13;
                }
            }
        }
        let template = QsTemplate::new(vec![
            LagWeight {
                dx: 1,
                dy: 0,
                dz: 0,
                weight: 1.0,
            },
            LagWeight {
                dx: -1,
                dy: 0,
                dz: 0,
                weight: 1.0,
            },
            LagWeight {
                dx: 0,
                dy: 1,
                dz: 0,
                weight: 1.0,
            },
            LagWeight {
                dx: 0,
                dy: 0,
                dz: 1,
                weight: 0.5,
            },
        ]);
        let obs = vec![0.5, 0.3, 0.4, 0.6];
        let mask = vec![1.0, 1.0, 1.0, 1.0];

        let mut ctx = FftCtx::new((nx, ny, nz), template.extents());
        let ti_freq = TiFreq::for_continuous(&ti, &mut ctx);
        let mut fft_out = Grid3::filled(nx, ny, nz, 0.0);
        let mut scratch = Vec::new();
        mismatch_continuous(
            &mut ctx,
            &ti_freq,
            &template,
            &obs,
            &mask,
            &mut fft_out,
            &mut scratch,
        );

        let naive_out = naive_continuous_mismatch(&ti, &template, &obs, &mask);

        let (ex, ey, ez) = (1usize, 1, 1);
        for z in ez..(nz - ez) {
            for y in ey..(ny - ey) {
                for x in ex..(nx - ex) {
                    let a = *fft_out.get(x, y, z);
                    let b = *naive_out.get(x, y, z);
                    assert!(
                        (a - b).abs() < F32_TOL,
                        "mismatch at ({x},{y},{z}): r2c-fft={a} naive={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn f32_fft_categorical_matches_naive() {
        let (nx, ny, nz) = (6, 6, 3);
        let mut ti = Grid3::<u32>::filled(nx, ny, nz, 0);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    *ti.get_mut(x, y, z) = ((x + y + z) % 3) as u32;
                }
            }
        }
        let template = QsTemplate::new(vec![
            LagWeight {
                dx: 1,
                dy: 0,
                dz: 0,
                weight: 1.0,
            },
            LagWeight {
                dx: 0,
                dy: 1,
                dz: 0,
                weight: 1.0,
            },
            LagWeight {
                dx: 0,
                dy: 0,
                dz: 1,
                weight: 1.0,
            },
        ]);
        let obs = vec![1u32, 2, 0];
        let mask = vec![1.0, 1.0, 1.0];

        let mut ctx = FftCtx::new((nx, ny, nz), template.extents());
        let ti_freq = TiFreq::for_categorical(&ti, 3, &mut ctx);
        let mut out_fft = Grid3::filled(nx, ny, nz, 0.0);
        mismatch_categorical(&mut ctx, &ti_freq, &template, &obs, &mask, &mut out_fft);

        let mut wsum = 0.0;
        for (i, lag) in template.lags.iter().enumerate() {
            wsum += mask[i] * lag.weight;
        }
        for z in 1..(nz - 1) {
            for y in 1..(ny - 1) {
                for x in 1..(nx - 1) {
                    let mut miss = 0.0;
                    for (i, lag) in template.lags.iter().enumerate() {
                        let xx = (x as i32 + lag.dx) as usize;
                        let yy = (y as i32 + lag.dy) as usize;
                        let zz = (z as i32 + lag.dz) as usize;
                        if *ti.get(xx, yy, zz) != obs[i] {
                            miss += mask[i] * lag.weight;
                        }
                    }
                    let expected = miss / wsum;
                    let got = *out_fft.get(x, y, z);
                    assert!(
                        (expected - got).abs() < F32_TOL,
                        "({x},{y},{z}): expected {expected} got {got}"
                    );
                }
            }
        }
    }
}
