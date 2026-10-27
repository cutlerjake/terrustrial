//! Triangular products with independent output rows in SIMD lanes.
use faer::MatRef;
use fearless_simd::{prelude::*, Level};
use fearless_simd_macros::simd;

#[inline]
pub(super) fn apply(factor: MatRef<'_, f64>, mean: &[f64], noise: &[f64], output: &mut [f64]) {
    // Tiny groups spend most of their time generating normals. Keep the short
    // scalar product in the caller; SIMD dispatch only pays off on wider draws.
    if output.len() < 8 {
        for (i, value) in output.iter_mut().enumerate() {
            *value = mean[i] + (0..=i).map(|j| factor[(i, j)] * noise[j]).sum::<f64>();
        }
        return;
    }
    fearless_simd::dispatch!(Level::new(), simd => apply_kernel(simd, factor, mean, noise, output))
}

#[simd]
fn apply_kernel<S: Simd>(
    simd: S,
    factor: MatRef<'_, f64>,
    mean: &[f64],
    noise: &[f64],
    output: &mut [f64],
) {
    let lanes = S::f64s::LEN;
    // Two independent native vectors hide reduction latency. This is an
    // unroll factor, not a hardware lane count.
    let rows = 2 * lanes;
    let tail_start = output.len() / rows * rows;
    let mut blocks = output.chunks_exact_mut(rows);
    for (block, out) in blocks.by_ref().enumerate() {
        let start = block * rows;
        // Keep each row's scalar reduction order, including Iterator::sum's
        // negative-zero identity. No horizontal reduction or fused operations.
        let mut totals = [S::f64s::splat(simd, -0.0); 2];
        for (j, &z) in noise[..=start].iter().enumerate() {
            for (vector, total) in totals.iter_mut().enumerate() {
                *total +=
                    S::f64s::from_fn(simd, |lane| factor[(start + vector * lanes + lane, j)]) * z;
            }
        }
        for (offset, out) in out.iter_mut().enumerate() {
            let row = start + offset;
            let mut value = totals[offset / lanes][offset % lanes];
            // The final diagonal staircase differs by lane. Never touch the
            // unused upper triangle or add artificial zero terms to a row.
            for j in start + 1..=row {
                value += factor[(row, j)] * noise[j];
            }
            *out = mean[row] + value;
        }
    }
    for (i, value) in blocks.into_remainder().iter_mut().enumerate() {
        let row = tail_start + i;
        *value = mean[row] + (0..=row).map(|j| factor[(row, j)] * noise[j]).sum::<f64>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn triangular_draw_matches_scalar_bits_with_tails_cancellation_and_unused_nan() {
        for n in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 64, 65, 128, 129, 256,
        ] {
            for zeros in [false, true] {
                let matrix = Mat::from_fn(n + 2, n + 2, |i, j| {
                    if j > i {
                        f64::NAN
                    } else if zeros {
                        -0.
                    } else {
                        match j % 5 {
                            0 => 1e100,
                            1 => -1e100,
                            2 => 1. + 2_f64.powi(-27),
                            3 => -1.,
                            _ => -0.,
                        }
                    }
                });
                let factor = matrix.as_ref().submatrix(1, 1, n, n);
                let noise: Vec<_> = (0..n)
                    .map(|j| match (j + 1) % 5 {
                        0 | 1 => 1e-100,
                        2 => 1. - 2_f64.powi(-27),
                        _ => 1.,
                    })
                    .collect();
                let mean: Vec<_> = (0..n).map(|i| if i % 2 == 0 { -0. } else { 1. }).collect();
                let expected: Vec<_> = (0..n)
                    .map(|i| mean[i] + (0..=i).map(|j| factor[(i, j)] * noise[j]).sum::<f64>())
                    .collect();
                let mut output = vec![19.; n + 2];
                let allocations = allocation_counter::measure(|| {
                    apply(factor, &mean, &noise, &mut output[1..=n])
                });
                assert_eq!(allocations.count_total, 0);
                assert_eq!(output[0], 19.);
                assert_eq!(output[n + 1], 19.);
                for i in 0..n {
                    assert_eq!(
                        output[i + 1].to_bits(),
                        expected[i].to_bits(),
                        "n={n}, zeros={zeros}, row={i}"
                    );
                }
            }
        }
    }
}
