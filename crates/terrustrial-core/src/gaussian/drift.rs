//! Column-major drift updates. Independent observation rows occupy SIMD lanes,
//! while each lane retains the scalar order of the drift-column reduction.
use faer::{MatMut, MatRef};
use fearless_simd::{prelude::*, Level};
use fearless_simd_macros::simd;

/// Ft^T - Fd^T W. Targets occupy independent lanes; the observation reduction
/// retains its original order, including the scalar sum's negative-zero identity.
pub(super) fn residual(
    level: Level,
    fd: MatRef<'_, f64>,
    weights: MatRef<'_, f64>,
    ft: MatRef<'_, f64>,
    correction: MatMut<'_, f64>,
) {
    fearless_simd::dispatch!(level, simd => residual_kernel(simd, fd, weights, ft, correction))
}

#[simd]
fn residual_kernel<S: Simd>(
    simd: S,
    fd: MatRef<'_, f64>,
    weights: MatRef<'_, f64>,
    ft: MatRef<'_, f64>,
    mut correction: MatMut<'_, f64>,
) {
    let lanes = S::f64s::LEN;
    let m = weights.ncols();
    for i in 0..fd.ncols() {
        for block in 0..m / lanes {
            let mut total = S::f64s::splat(simd, -0.0);
            for r in 0..fd.nrows() {
                let values = S::f64s::from_fn(simd, |lane| weights[(r, lanes * block + lane)]);
                total += values * fd[(r, i)];
            }
            let result = S::f64s::from_fn(simd, |lane| ft[(lanes * block + lane, i)]) - total;
            for lane in 0..lanes {
                correction[(i, lanes * block + lane)] = result[lane];
            }
        }
        for j in m / lanes * lanes..m {
            correction[(i, j)] = ft[(j, i)]
                - (0..fd.nrows())
                    .map(|r| fd[(r, i)] * weights[(r, j)])
                    .sum::<f64>();
        }
    }
}

pub(super) fn add_weights(
    level: Level,
    weights: MatMut<'_, f64>,
    cinv_f: MatRef<'_, f64>,
    correction: MatRef<'_, f64>,
) {
    fearless_simd::dispatch!(level, simd => add_kernel(simd, weights, cinv_f, correction))
}

#[simd]
fn add_kernel<S: Simd>(
    simd: S,
    mut weights: MatMut<'_, f64>,
    cinv_f: MatRef<'_, f64>,
    correction: MatRef<'_, f64>,
) {
    let lanes = S::f64s::LEN;
    for j in 0..weights.ncols() {
        let column = weights
            .as_mut()
            .col_mut(j)
            .try_as_col_major_mut()
            .expect("owned weights")
            .as_slice_mut();
        let mut chunks = column.chunks_exact_mut(lanes);
        for (block, out) in chunks.by_ref().enumerate() {
            // Match f64 Iterator::sum's identity, including an all-negative-zero
            // reduction. Starting at positive zero would change those bits.
            let mut total = S::f64s::splat(simd, -0.0);
            for r in 0..cinv_f.ncols() {
                let values = S::f64s::from_fn(simd, |lane| cinv_f[(lanes * block + lane, r)]);
                total += values * correction[(r, j)];
            }
            let updated = S::f64s::from_slice(simd, out) + total;
            updated.store_slice(out);
        }
        let start = cinv_f.nrows() / lanes * lanes;
        for (tail, out) in chunks.into_remainder().iter_mut().enumerate() {
            *out += (0..cinv_f.ncols())
                .map(|r| cinv_f[(start + tail, r)] * correction[(r, j)])
                .sum::<f64>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn residual_lanes_match_scalar_bits_including_cancellation_tails_and_zero() {
        for n in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129,
        ] {
            for p in [0, 1, 2, 8, 17] {
                for m in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17] {
                    let fd = Mat::from_fn(n + 2, p + 2, |r, i| {
                        if i % 3 == 0 {
                            -0.
                        } else {
                            match r % 5 {
                                0 => 1e100,
                                1 => -1e100,
                                2 => 1. + 2_f64.powi(-27),
                                3 => -1.,
                                _ => -0.,
                            }
                        }
                    });
                    let weights = Mat::from_fn(n + 2, m + 2, |r, j| match r % 5 {
                        0 | 1 => (j + 1) as f64 * 1e-100,
                        2 => 1. - 2_f64.powi(-27),
                        _ => 1.,
                    });
                    let ft =
                        Mat::from_fn(m + 2, p + 2, |j, i| if (i + j) % 2 == 0 { -0. } else { 1. });
                    let mut actual = Mat::full(p + 2, m + 2, 19.);
                    let mut expected = actual.clone();
                    for i in 0..p {
                        for j in 0..m {
                            let mut sum = -0.;
                            for r in 0..n {
                                sum += fd[(r + 1, i + 1)] * weights[(r + 1, j + 1)];
                            }
                            expected[(i + 1, j + 1)] = ft[(j + 1, i + 1)] - sum;
                        }
                    }
                    let allocations = allocation_counter::measure(|| {
                        residual(
                            Level::new(),
                            fd.as_ref().submatrix(1, 1, n, p),
                            weights.as_ref().submatrix(1, 1, n, m),
                            ft.as_ref().submatrix(1, 1, m, p),
                            actual.as_mut().submatrix_mut(1, 1, p, m),
                        )
                    });
                    assert_eq!(allocations.count_total, 0);
                    for i in 0..p + 2 {
                        for j in 0..m + 2 {
                            assert_eq!(
                                actual[(i, j)].to_bits(),
                                expected[(i, j)].to_bits(),
                                "n={n},p={p},m={m},i={i},j={j}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lane_updates_match_scalar_order_with_tails_strides_and_cancellation() {
        let level = Level::new();
        let epsilon = 2.0_f64.powi(-27);
        for n in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 129,
        ] {
            for p in [0, 1, 2, 3, 8, 17] {
                for m in [0, 1, 4, 17] {
                    // Offsets and padded leading dimensions exercise nontrivial
                    // views without changing the production column-major contract.
                    let left = Mat::from_fn(n + 2, p + 2, |i, r| {
                        if i % 11 == 0 {
                            -0.
                        } else {
                            match r % 5 {
                                0 => 1e100 * (1. + i as f64 * 0.01),
                                1 => -1.,
                                2 => 1. + epsilon,
                                3 => -0.,
                                _ => -1e100 * (1. + i as f64 * 0.01),
                            }
                        }
                    });
                    let right = Mat::from_fn(p + 2, m + 2, |r, j| match r % 5 {
                        0 | 4 => 1e-100 * (1. + j as f64 * 0.01),
                        2 => 1. - epsilon,
                        _ => 1.,
                    });
                    let mut actual = Mat::from_fn(n + 2, m + 2, |i, j| {
                        if (i + j) % 3 == 0 {
                            -0.
                        } else {
                            (i + j) as f64 * 0.01
                        }
                    });
                    let mut expected = actual.clone();
                    for i in 0..n {
                        for j in 0..m {
                            let mut sum = -0.0;
                            for r in 0..p {
                                sum += left[(i + 1, r + 1)] * right[(r + 1, j + 1)];
                            }
                            expected[(i + 1, j + 1)] += sum;
                        }
                    }
                    let allocations = allocation_counter::measure(|| {
                        add_weights(
                            level,
                            actual.as_mut().submatrix_mut(1, 1, n, m),
                            left.as_ref().submatrix(1, 1, n, p),
                            right.as_ref().submatrix(1, 1, p, m),
                        );
                    });
                    assert_eq!(allocations.count_total, 0, "n={n} p={p} m={m}");
                    for j in 0..m + 2 {
                        for i in 0..n + 2 {
                            assert_eq!(
                                actual[(i, j)].to_bits(),
                                expected[(i, j)].to_bits(),
                                "n={n} p={p} m={m} row={i} col={j}"
                            );
                        }
                    }
                }
            }
        }
    }
}
