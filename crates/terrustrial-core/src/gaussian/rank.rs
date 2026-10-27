//! Adaptive change of drift basis, without changing the constraints or covariance.
use super::*;
use dyn_stack::StackReq;
use faer::linalg::{qr::col_pivoting::factor as qr, triangular_solve};

/// The normal-equation path is retained when its Cholesky pivots are healthy.
/// Otherwise QR supplies a right preconditioner: both observations and targets
/// use the same triangular transformation F P R^-1.
/// All storage is reserved with the Gaussian workspace, including QR scratch.
pub(super) struct DriftBasis {
    active: bool,
    r: Mat<f64>,
    coefficients: Mat<f64>,
    permutations: Vec<usize>,
}

impl DriftBasis {
    pub(super) fn new(p: usize) -> Result<Self> {
        let mut r = Mat::new();
        r.try_reserve(p, p)
            .map_err(|_| invalid("Gaussian drift QR allocation"))?;
        let mut coefficients = Mat::new();
        coefficients
            .try_reserve(1, p)
            .map_err(|_| invalid("Gaussian drift reflectors allocation"))?;
        let mut permutations = Vec::new();
        permutations
            .try_reserve_exact(checked_len(&[2, p])?)
            .map_err(|_| invalid("Gaussian drift permutation allocation"))?;
        Ok(Self {
            active: false,
            r,
            coefficients,
            permutations,
        })
    }

    pub(super) fn diagnostic_r(&self) -> Option<MatRef<'_, f64>> {
        self.active.then(|| self.r.as_ref())
    }
    pub(super) fn reset(&mut self) {
        self.active = false;
    }

    pub(super) fn scratch(n: usize, p: usize) -> StackReq {
        qr::qr_in_place_scratch::<usize, f64>(n, p, 1, Par::Seq, Default::default())
    }

    /// This is a conservative fallback trigger, not a condition-number estimate.
    /// A pivot ratio below eps^(1/4) risks losing half the available digits in
    /// normal equations. QR then tests rank in the original normalized basis.
    pub(super) fn suspect(factor: MatRef<'_, f64>) -> bool {
        let mut smallest = f64::INFINITY;
        let mut largest = 0.0_f64;
        for i in 0..factor.nrows() {
            let diagonal = factor[(i, i)].abs();
            smallest = smallest.min(diagonal);
            largest = largest.max(diagonal);
        }
        smallest <= largest * f64::EPSILON.sqrt().sqrt()
    }

    pub(super) fn orthogonalize(
        &mut self,
        mut basis: MatMut<'_, f64>,
        mut q: MatMut<'_, f64>,
        scratch: &mut MemBuffer,
    ) -> Result<()> {
        let (n, p) = (basis.nrows(), basis.ncols());
        if !MemStack::new(scratch).can_hold(Self::scratch(n, p)) {
            return Err(invalid("Gaussian drift QR workspace capacity exceeded"));
        }
        self.r.resize_with(p, p, |_, _| 0.0);
        self.coefficients.resize_with(1, p, |_, _| 0.0);
        self.permutations.resize(2 * p, 0);
        q.copy_from(basis.as_ref());
        let (forward, inverse) = self.permutations.split_at_mut(p);
        qr::qr_in_place(
            basis.as_mut(),
            self.coefficients.as_mut(),
            forward,
            inverse,
            Par::Seq,
            MemStack::new(scratch),
            Default::default(),
        );
        // Refuse numerically unidentified columns. Do not drop a constraint or
        // silently regularize it. This threshold is relative to pivoted R.
        let threshold = f64::EPSILON * n.max(p) as f64 * basis[(0, 0)].abs();
        for j in 0..p {
            if !basis[(j, j)].is_finite() || basis[(j, j)].abs() <= threshold {
                return Err(Error::Singular {
                    context: "drift columns are numerically rank deficient",
                });
            }
            for i in 0..=j {
                self.r[(i, j)] = basis[(i, j)];
            }
        }
        // Form the observation basis by the same solve used for target rows.
        // Explicit Householder Q differs from F P R^-1 by amplified QR
        // backward error for nearly dependent columns. Mixing those two forms
        // breaks exact interpolation, including zero conditional covariances.
        for j in 0..p {
            for i in 0..n {
                basis[(i, j)] = q[(i, forward[j])];
            }
        }
        triangular_solve::solve_lower_triangular_in_place(
            self.r.as_ref().transpose(),
            basis.as_mut().transpose_mut(),
            Par::Seq,
        );
        finite(basis.as_ref(), "preconditioned drift basis")?;
        q.copy_from(basis.as_ref());
        self.active = true;
        Ok(())
    }

    pub(super) fn transform_targets(
        &self,
        mut targets: MatMut<'_, f64>,
        buffer: &mut [f64],
    ) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let p = targets.ncols();
        for i in 0..targets.nrows() {
            for j in 0..p {
                buffer[j] = targets[(i, self.permutations[j])];
            }
            for j in 0..p {
                targets[(i, j)] = buffer[j];
            }
        }
        triangular_solve::solve_lower_triangular_in_place(
            self.r.as_ref().transpose(),
            targets.as_mut().transpose_mut(),
            Par::Seq,
        );
        finite(targets.as_ref(), "transformed target drift")
    }
}
