//! Fixed-order semidefinite Cholesky, preserving deterministic zero variances.
use super::{finite, Error, Result};
use faer::{MatMut, MatRef};
use fearless_simd::{prelude::*, Level};
use fearless_simd_macros::simd;

pub(super) fn factor(
    matrix: MatRef<'_, f64>,
    factor: MatMut<'_, f64>,
    threshold: impl Fn(usize, usize) -> f64,
) -> Result<usize> {
    fearless_simd::dispatch!(Level::new(), simd => factor_kernel(simd, matrix, factor, threshold))
}

#[simd]
fn factor_kernel<S: Simd>(
    simd: S,
    matrix: MatRef<'_, f64>,
    factor: MatMut<'_, f64>,
    threshold: impl Fn(usize, usize) -> f64,
) -> Result<usize> {
    if S::f64s::LEN < 4 {
        factor_vectors::<S, 2>(simd, matrix, factor, threshold)
    } else {
        factor_vectors::<S, 1>(simd, matrix, factor, threshold)
    }
}

#[inline(always)]
fn factor_vectors<S: Simd, const VECTORS: usize>(
    simd: S,
    matrix: MatRef<'_, f64>,
    mut factor: MatMut<'_, f64>,
    threshold: impl Fn(usize, usize) -> f64,
) -> Result<usize> {
    let lanes = S::f64s::LEN;
    let rows = VECTORS * lanes;
    let n = matrix.nrows();
    factor.fill(0.0);
    let mut rounded = 0;
    for j in 0..n {
        let diagonal = matrix[(j, j)] - (0..j).map(|k| factor[(j, k)].powi(2)).sum::<f64>();
        let diagonal_threshold = threshold(j, j);
        if diagonal < -diagonal_threshold {
            return Err(Error::Numerical {
                context: "conditional covariance is not positive semidefinite",
            });
        }
        if diagonal <= diagonal_threshold {
            rounded += usize::from(diagonal != 0.0);
        } else {
            factor[(j, j)] = diagonal.sqrt();
        }
        let pivot = factor[(j, j)];
        let end = j + 1 + (n - j - 1) / rows * rows;
        for start in (j + 1..end).step_by(rows) {
            // Independent rows occupy lanes. Retain each dot product's scalar
            // order and negative-zero identity; do not use a horizontal sum.
            let mut sums = [S::f64s::splat(simd, -0.0); VECTORS];
            for k in 0..j {
                for (vector, sum) in sums.iter_mut().enumerate() {
                    *sum +=
                        S::f64s::from_fn(simd, |lane| factor[(start + vector * lanes + lane, k)])
                            * factor[(j, k)];
                }
            }
            for (vector, sum) in sums.into_iter().enumerate() {
                let first = start + vector * lanes;
                let residual = S::f64s::from_fn(simd, |lane| matrix[(first + lane, j)]) - sum;
                if pivot == 0.0 {
                    for lane in 0..lanes {
                        check_zero(residual[lane], threshold(first + lane, j), &mut rounded)?;
                    }
                } else {
                    let values = residual / pivot;
                    for lane in 0..lanes {
                        factor[(first + lane, j)] = values[lane];
                    }
                }
            }
        }
        for i in end..n {
            let value =
                matrix[(i, j)] - (0..j).map(|k| factor[(i, k)] * factor[(j, k)]).sum::<f64>();
            if pivot == 0.0 {
                check_zero(value, threshold(i, j), &mut rounded)?;
            } else {
                factor[(i, j)] = value / pivot;
            }
        }
    }
    finite(factor.as_ref(), "conditional factor")?;
    Ok(rounded)
}

fn check_zero(value: f64, threshold: f64, rounded: &mut usize) -> Result<()> {
    if value.abs() > threshold {
        return Err(Error::Numerical {
            context: "nonzero covariance at a zero conditional-variance pivot",
        });
    }
    *rounded += usize::from(value != 0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn dyadic_positive_and_rank_deficient_covariances_recover_known_factors_without_allocation() {
        for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 64, 65, 129] {
            for rank in 0..3 {
                // Dyadic entries keep construction and elimination exact, so
                // the original lower factor is an independent bitwise oracle.
                let known = Mat::from_fn(n, n, |i, j| {
                    if j > i || rank == 0 || (rank == 1 && j % 3 == 0) {
                        0.
                    } else if i == j {
                        2_f64.powi(j as i32 % 3 - 1)
                    } else {
                        ((i * 7 + j * 11) % 9) as f64 * 0.125 - 0.5
                    }
                });
                let mut matrix = Mat::full(n + 2, n + 2, 19.);
                for j in 0..n {
                    for i in 0..n {
                        matrix[(i + 1, j + 1)] =
                            (0..n).map(|k| known[(i, k)] * known[(j, k)]).sum();
                    }
                }
                let mut actual = Mat::full(n + 2, n + 2, 19.);
                let allocations = allocation_counter::measure(|| {
                    assert_eq!(
                        factor(
                            matrix.as_ref().submatrix(1, 1, n, n),
                            actual.as_mut().submatrix_mut(1, 1, n, n),
                            |_, _| 0.,
                        )
                        .unwrap(),
                        0
                    );
                });
                assert_eq!(allocations.count_total, 0);
                for j in 0..n + 2 {
                    for i in 0..n + 2 {
                        let expected = if (1..=n).contains(&i) && (1..=n).contains(&j) {
                            known[(i - 1, j - 1)]
                        } else {
                            19.
                        };
                        assert_eq!(
                            actual[(i, j)].to_bits(),
                            expected.to_bits(),
                            "n={n}, rank={rank}, i={i}, j={j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn per_entry_thresholds_preserve_rounding_counts_errors_and_recovery() {
        for n in [5, 8, 9, 17] {
            let threshold = |i: usize, j: usize| (i + j + 1) as f64 * 1e-12;
            let mut matrix = Mat::<f64>::identity(n, n);
            matrix[(0, 0)] = 0.;
            matrix[(n - 1, n - 1)] = -0.5 * threshold(n - 1, n - 1);
            for i in 1..n {
                let value = if i % 2 == 0 { 0.5 } else { -0.5 } * threshold(i, 0);
                matrix[(i, 0)] = value;
                matrix[(0, i)] = value;
            }
            let mut out = Mat::full(n, n, 19.);
            assert_eq!(factor(matrix.as_ref(), out.as_mut(), threshold).unwrap(), n);
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(out[(i, j)], f64::from(i == j && i != 0 && i != n - 1));
                }
            }
            // Exercise zero-pivot rejections in both vector lanes and tails.
            for i in [1, 4, n - 1] {
                let saved = matrix[(i, 0)];
                matrix[(i, 0)] = 2. * threshold(i, 0);
                assert!(matches!(
                    factor(matrix.as_ref(), out.as_mut(), threshold),
                    Err(Error::Numerical {
                        context: "nonzero covariance at a zero conditional-variance pivot"
                    })
                ));
                matrix[(i, 0)] = saved;
            }
            matrix[(1, 1)] = -2. * threshold(1, 1);
            assert!(matches!(
                factor(matrix.as_ref(), out.as_mut(), threshold),
                Err(Error::Numerical {
                    context: "conditional covariance is not positive semidefinite"
                })
            ));
            matrix[(1, 1)] = 1.;
            assert_eq!(factor(matrix.as_ref(), out.as_mut(), threshold).unwrap(), n);
            assert!(factor(matrix.as_ref(), out.as_mut(), |_, _| 0.).is_err());
        }
    }
}
