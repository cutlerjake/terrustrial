//! Optional spectral diagnostics of the systems represented by prepared factors.
use super::*;
use dyn_stack::StackReq;
use faer::{linalg::svd, ColMut};

/// Spectral (2-norm) condition estimate. It describes the represented floating-point
/// matrix, not a bound on forward error or a certificate of exact rank.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralCondition {
    pub rows: usize,
    pub columns: usize,
    /// In [0, 1]. Can underflow to zero even when log10_condition is finite.
    pub reciprocal_condition: f64,
    /// Logarithm avoids overflow for very ill-conditioned systems. Infinity means
    /// that faer's computed smallest singular value was zero.
    pub log10_condition: f64,
    pub computed_zero_singular_values: usize,
}

/// Conditions in the internal solve coordinates, after observation equilibration
/// and drift column scaling. These are deliberately not physical-unit condition
/// numbers. Physical-unit condition numbers depend on the chosen units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditioningDiagnostics {
    /// Observation covariance including explicit measurement noise.
    pub covariance: Option<SpectralCondition>,
    /// Scaled observation drift design BEFORE any QR change of basis.
    pub drift_design: Option<SpectralCondition>,
    /// Drift normal equations AFTER any QR change of basis.
    pub drift_normal: Option<SpectralCondition>,
    pub observation_equilibration: bool,
    pub common_covariance_scale: f64,
    pub drift_qr_applied: bool,
}

/// Separate, budgeted faer SVD workspace. Diagnostics are opt-in and do not change
/// conditioning factors, predictions, model/noise, or numerical acceptance policy.
/// Work is cubic in the largest diagnosed dimension. Cancellation is checked
/// around each backend call; a running faer SVD cannot be interrupted internally.
/// Buffer payloads and faer's requested scratch are charged, excluding allocator
/// metadata and faer's internal matrix-product packing. Owned buffers are reused;
/// faer can still allocate temporary packing for larger dimensions. These optional
/// diagnostics are separate from the allocation-sensitive preparation path.
pub struct DiagnosticWorkspace {
    maximum_dimension: usize,
    matrix: Vec<f64>,
    singular: Vec<f64>,
    scratch: MemBuffer,
    _reservation: Reservation,
}
fn request(rows: usize, columns: usize) -> StackReq {
    svd::svd_scratch::<f64>(
        rows,
        columns,
        svd::ComputeSvdVectors::No,
        svd::ComputeSvdVectors::No,
        Par::Seq,
        Default::default(),
    )
}
fn capacity_request(maximum: usize) -> StackReq {
    // A tall matrix's QR-first path can need MORE scratch than a larger square
    // matrix. Bound both faer branches at the maximum dimensions. This changes
    // only the reservation, not faer's default algorithm selection at execution.
    let qr = svd::SvdParams {
        qr_ratio_threshold: 0.,
        ..faer::auto!(f64)
    };
    request(maximum, maximum).or(svd::svd_scratch::<f64>(
        maximum,
        maximum,
        svd::ComputeSvdVectors::No,
        svd::ComputeSvdVectors::No,
        Par::Seq,
        qr.into(),
    ))
}
impl DiagnosticWorkspace {
    pub fn new(maximum_dimension: usize, budget: &MemoryBudget) -> Result<Self> {
        // Guard backend intermediate dimension arithmetic before requesting its
        // layout, as well as Rust's allocation-addressability limit.
        let edge = maximum_dimension
            .checked_add(1)
            .ok_or(Error::SizeOverflow)?;
        let bound = checked_len(&[edge, edge, 8, 64])?;
        if bound > isize::MAX as usize {
            return Err(Error::SizeOverflow);
        }
        let elements = checked_len(&[maximum_dimension, maximum_dimension])?;
        let request = capacity_request(maximum_dimension);
        let scratch_bytes = request.layout().map_err(|_| Error::SizeOverflow)?.size();
        let bytes = elements
            .checked_add(maximum_dimension)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(scratch_bytes))
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        fn buffer(count: usize) -> Result<Vec<f64>> {
            let mut out = Vec::new();
            out.try_reserve_exact(count)
                .map_err(|_| invalid("condition diagnostic buffer allocation"))?;
            out.resize(count, 0.);
            Ok(out)
        }
        Ok(Self {
            maximum_dimension,
            matrix: buffer(elements)?,
            singular: buffer(maximum_dimension)?,
            scratch: MemBuffer::try_new(request)
                .map_err(|_| invalid("condition diagnostic SVD scratch allocation"))?,
            _reservation: reservation,
        })
    }
    pub fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
    }
    pub fn maximum_dimension(&self) -> usize {
        self.maximum_dimension
    }

    pub(super) fn evaluate(
        &mut self,
        workspace: &GaussianWorkspace,
        cancel: &Cancellation,
    ) -> Result<ConditioningDiagnostics> {
        cancel.check()?;
        let covariance = self.measure(workspace.c.as_ref(), Triangle::Lower, true, cancel)?;
        let drift_design = if let Some(r) = workspace.drift_basis.diagnostic_r() {
            let mut condition = self.measure(r, Triangle::Upper, false, cancel)?;
            // R has the same singular values as the original tall, scaled F.
            if let Some(condition) = &mut condition {
                condition.rows = workspace.fd.nrows();
            }
            condition
        } else {
            self.measure(workspace.fd.as_ref(), Triangle::Full, false, cancel)?
        };
        let drift_normal =
            self.measure(workspace.normal.as_ref(), Triangle::Lower, true, cancel)?;
        Ok(ConditioningDiagnostics {
            covariance,
            drift_design,
            drift_normal,
            observation_equilibration: workspace.equilibration.data_scaling_active(),
            common_covariance_scale: workspace.covariance_units,
            drift_qr_applied: workspace.drift_basis.diagnostic_r().is_some(),
        })
    }
    fn measure(
        &mut self,
        input: MatRef<'_, f64>,
        triangle: Triangle,
        squared: bool,
        cancel: &Cancellation,
    ) -> Result<Option<SpectralCondition>> {
        cancel.check()?;
        let (rows, columns) = (input.nrows(), input.ncols());
        let size = rows.min(columns);
        if rows.max(columns) > self.maximum_dimension {
            return Err(invalid("condition diagnostic capacity exceeded"));
        }
        if size == 0 {
            return Ok(None);
        }
        let scratch = request(rows, columns);
        if !MemStack::new(&mut self.scratch).can_hold(scratch) {
            return Err(invalid(
                "condition diagnostic backend scratch capacity exceeded",
            ));
        }
        let values = &mut self.matrix[..rows * columns];
        let mut scale = 0_f64;
        for j in 0..columns {
            cancel.check()?;
            for i in 0..rows {
                let value = if triangle.includes(i, j) {
                    input[(i, j)]
                } else {
                    0.
                };
                if !value.is_finite() {
                    return Err(Error::Numerical {
                        context: "nonfinite condition diagnostic input",
                    });
                }
                values[i + j * rows] = value;
                scale = scale.max(value.abs());
            }
        }
        if scale != 0. && scale != 1. {
            for value in values.iter_mut() {
                let normalized = *value / scale;
                if normalized == 0. && *value != 0. {
                    return Err(Error::Numerical {
                        context: "condition diagnostic normalization underflow",
                    });
                }
                *value = normalized;
            }
        }
        svd::svd(
            MatRef::from_column_major_slice(values, rows, columns),
            ColMut::from_slice_mut(&mut self.singular[..size]).as_diagonal_mut(),
            None,
            None,
            Par::Seq,
            MemStack::new(&mut self.scratch),
            Default::default(),
        )
        .map_err(|_| Error::Numerical {
            context: "condition diagnostic SVD did not converge",
        })?;
        cancel.check()?;
        let singular = &self.singular[..size];
        if singular.iter().any(|s| !s.is_finite() || *s < 0.)
            || singular.windows(2).any(|s| s[0] < s[1])
        {
            return Err(Error::Numerical {
                context: "invalid condition diagnostic singular values",
            });
        }
        let (largest, smallest) = (singular[0], singular[size - 1]);
        let ratio = if largest == 0. {
            0.
        } else {
            smallest / largest
        };
        let power = if squared { 2. } else { 1. };
        Ok(Some(SpectralCondition {
            rows,
            columns,
            reciprocal_condition: if squared { ratio * ratio } else { ratio },
            log10_condition: if smallest == 0. {
                f64::INFINITY
            } else {
                power * (largest.log10() - smallest.log10()).max(0.)
            },
            computed_zero_singular_values: singular.iter().filter(|&&s| s == 0.).count(),
        }))
    }
}
#[derive(Clone, Copy)]
enum Triangle {
    Full,
    Lower,
    Upper,
}
impl Triangle {
    fn includes(self, row: usize, column: usize) -> bool {
        match self {
            Self::Full => true,
            Self::Lower => row >= column,
            Self::Upper => row <= column,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn svd_scratch_admission_covers_smaller_rectangular_shapes() {
        for maximum in [
            0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 95, 96, 127, 128, 129, 255, 256, 257,
            511,
        ] {
            let capacity = capacity_request(maximum).layout().unwrap();
            let mut dims = if maximum <= 32 {
                (0..=maximum).collect::<Vec<_>>()
            } else {
                vec![
                    0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 95, 96, 127, 128, 129, 255, 256,
                    257, 511, maximum,
                ]
            };
            dims.retain(|&n| n <= maximum);
            for &rows in &dims {
                for &columns in &dims {
                    let need = request(rows, columns).layout().unwrap();
                    assert!(need.size()<=capacity.size() && need.align()<=capacity.align(), "maximum={maximum}, shape={rows}x{columns}, capacity={capacity:?}, need={need:?}");
                }
            }
        }
    }
    #[test]
    fn logarithms_distinguish_reciprocal_underflow_from_computed_zero_and_failures_are_reusable() {
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let mut workspace = DiagnosticWorkspace::new(3, &budget).unwrap();
        for tiny in [1e-100, 1e-180, 0.] {
            let matrix = faer::mat![[1., f64::NAN], [0., tiny]];
            let found = workspace
                .measure(matrix.as_ref(), Triangle::Lower, true, &cancel)
                .unwrap()
                .unwrap();
            if tiny == 0. {
                assert_eq!(found.computed_zero_singular_values, 1);
                assert_eq!(found.log10_condition, f64::INFINITY);
            } else {
                assert_eq!(found.computed_zero_singular_values, 0);
                assert!((found.log10_condition + 2. * tiny.log10()).abs() < 1e-12);
            }
            if tiny < 1e-162 {
                assert_eq!(found.reciprocal_condition, 0.);
            }
        }
        let nonfinite = faer::mat![[1., 0.], [f64::INFINITY, 1.]];
        assert!(matches!(
            workspace.measure(nonfinite.as_ref(), Triangle::Lower, true, &cancel),
            Err(Error::Numerical { .. })
        ));
        let underflow = faer::mat![[1e300, 0.], [0., 1e-300]];
        assert!(matches!(
            workspace.measure(underflow.as_ref(), Triangle::Full, false, &cancel),
            Err(Error::Numerical {
                context: "condition diagnostic normalization underflow"
            })
        ));
        let valid = faer::mat![[1., 0.], [0., 0.5]];
        assert_eq!(
            workspace
                .measure(valid.as_ref(), Triangle::Full, false, &cancel)
                .unwrap()
                .unwrap()
                .reciprocal_condition,
            0.5
        );
    }
}
