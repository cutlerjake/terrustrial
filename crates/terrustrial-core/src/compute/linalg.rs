//! Shared finite-matrix checks and sequential faer SPD operations.
use crate::{error::invalid, Error, Result};
use dyn_stack::{MemBuffer, MemStack};
use faer::{linalg::cholesky::llt, MatMut, MatRef, Par};
use fearless_simd::{prelude::*, Level};
use fearless_simd_macros::simd;

pub(crate) fn finite(matrix: MatRef<'_, f64>, context: &'static str) -> Result<()> {
    if fearless_simd::dispatch!(Level::new(), simd => finite_kernel(simd, matrix)) {
        Ok(())
    } else {
        Err(Error::Numerical { context })
    }
}

#[simd]
fn finite_kernel<S: Simd>(simd: S, matrix: MatRef<'_, f64>) -> bool {
    // Binary64 NaNs and infinities have an all-ones exponent. Checking their
    // bits avoids floating-point operations, including on signaling NaNs.
    let exponent = S::u64s::splat(simd, 0x7ff0_0000_0000_0000);
    for col in matrix.col_iter() {
        if let Some(col) = col.try_as_col_major() {
            let mut chunks = col.as_slice().chunks_exact(S::f64s::LEN);
            for chunk in chunks.by_ref() {
                let bits: S::u64s = S::f64s::from_slice(simd, chunk).bitcast();
                if (bits & exponent).simd_eq(exponent).any_true() {
                    return false;
                }
            }
            if chunks.remainder().iter().any(|x| !x.is_finite()) {
                return false;
            }
        } else if col.iter().any(|x| !x.is_finite()) {
            // Transposed and reversed views need not have contiguous rows.
            return false;
        }
    }
    true
}

/// Local systems use an explicit sequential backend. Global faer parallelism
/// must not make results depend on the application's worker configuration.
pub(crate) fn factor_spd(matrix: MatMut<'_, f64>, context: &'static str) -> Result<()> {
    let request =
        llt::factor::cholesky_in_place_scratch::<f64>(matrix.nrows(), Par::Seq, Default::default());
    let mut scratch = MemBuffer::try_new(request)
        .map_err(|_| invalid("unable to allocate factorization workspace"))?;
    factor_spd_with_parameters(matrix, context, &mut scratch, faer::auto!(f64))
}
pub(crate) fn factor_spd_with_parameters(
    matrix: MatMut<'_, f64>,
    context: &'static str,
    scratch: &mut MemBuffer,
    parameters: llt::factor::LltParams,
) -> Result<()> {
    let request =
        llt::factor::cholesky_in_place_scratch::<f64>(matrix.nrows(), Par::Seq, parameters.into());
    if !MemStack::new(scratch).can_hold(request) {
        return Err(invalid("Gaussian factor workspace capacity exceeded"));
    }
    llt::factor::cholesky_in_place(
        matrix,
        Default::default(),
        Par::Seq,
        MemStack::new(scratch),
        parameters.into(),
    )
    .map_err(|_| Error::Singular { context })?;
    Ok(())
}

pub(crate) fn solve_spd(factor: MatRef<'_, f64>, rhs: MatMut<'_, f64>) {
    llt::solve::solve_in_place(factor, rhs, Par::Seq, MemStack::new(&mut []));
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn check(matrix: MatRef<'_, f64>, expected: bool) {
        for level in [Level::baseline(), Level::new()] {
            let actual = fearless_simd::dispatch!(level, simd => finite_kernel(simd, matrix));
            assert_eq!(actual, expected);
        }
        match finite(matrix, "finite validation test") {
            Ok(()) => assert!(expected),
            Err(Error::Numerical { context }) => {
                assert!(!expected);
                assert_eq!(context, "finite validation test");
            }
            Err(error) => panic!("unexpected error: {error}"),
        }
    }

    #[test]
    fn finite_validation_classifies_every_exponent_and_preserves_bits() {
        let mut matrix = Mat::full(17, 3, -0.0);
        let mut position = 0;
        for sign in [0, 1_u64 << 63] {
            for exponent in 0..=0x7ff_u64 {
                for fraction in [0, 1, 0x0008_0000_0000_0000, 0x000f_ffff_ffff_ffff] {
                    let bits = sign | (exponent << 52) | fraction;
                    let value = f64::from_bits(bits);
                    let row = position % matrix.nrows();
                    let column = (position / matrix.nrows()) % matrix.ncols();
                    matrix[(row, column)] = value;
                    check(matrix.as_ref(), value.is_finite());
                    assert_eq!(matrix[(row, column)].to_bits(), bits);
                    matrix[(row, column)] = -0.0;
                    position += 1;
                }
            }
        }
    }

    #[test]
    fn finite_validation_handles_lanes_tails_padding_and_strides() {
        let finite_values = [
            0.0,
            -0.0,
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::from_bits((1_u64 << 63) | 1),
        ];
        let invalid_values = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff0_0000_0000_0001),
            f64::from_bits(0xfff0_0000_0000_0001),
            f64::from_bits(0xffff_ffff_ffff_ffff),
        ];
        for rows in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for columns in [0, 1, 3] {
                // NaNs surround the logical view, including its column padding.
                let mut matrix = Mat::full(rows + 2, columns + 2, f64::NAN);
                for j in 0..columns {
                    for i in 0..rows {
                        matrix[(i + 1, j + 1)] = finite_values[(i + j) % finite_values.len()];
                    }
                }
                let check_views = |matrix: &Mat<f64>, expected| {
                    let view = matrix.as_ref().submatrix(1, 1, rows, columns);
                    check(view, expected);
                    check(view.transpose(), expected);
                    check(view.reverse_rows(), expected);
                    check(view.reverse_rows_and_cols(), expected);
                };
                check_views(&matrix, true);
                for j in 0..columns {
                    for i in 0..rows {
                        let original = matrix[(i + 1, j + 1)];
                        for value in invalid_values {
                            matrix[(i + 1, j + 1)] = value;
                            check_views(&matrix, false);
                        }
                        matrix[(i + 1, j + 1)] = original;
                    }
                }
            }
        }
    }

    #[test]
    fn finite_validation_does_not_allocate() {
        let matrix = Mat::full(337, 57, 1.0);
        // Resolve runtime CPU detection before measuring the validation itself.
        finite(matrix.as_ref(), "finite validation test").unwrap();
        let allocations = allocation_counter::measure(|| {
            finite(matrix.as_ref(), "finite validation test").unwrap();
            finite(matrix.as_ref().transpose(), "finite validation test").unwrap();
        });
        assert_eq!(allocations.count_total, 0);
    }
}
