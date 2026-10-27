//! Reusable joint disjunctive prediction in a marginal standard-normal basis.
//! Distinct Hermite orders are orthogonal under a jointly Gaussian latent model.
//! Error covariance is for the represented finite expansions, not an unreported
//! infinite expansion or a general non-Gaussian copula.
use super::*;
use crate::compute::scaling::{exponent, shift};
use crate::data::Point3;

#[derive(Clone, Debug)]
pub struct Expansion {
    coefficients: Vec<f64>,
}
impl Expansion {
    /// Coefficients of orthonormal probabilists' Hermite polynomials, starting
    /// with the constant. They may represent different functions for each field.
    pub fn new(coefficients: Vec<f64>) -> Result<Self> {
        if coefficients.is_empty()
            || coefficients.len() > i32::MAX as usize
            || coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(invalid("finite nonempty Hermite expansion required"));
        }
        Ok(Self { coefficients })
    }
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }
    fn coefficient(&self, order: usize) -> f64 {
        self.coefficients.get(order).copied().unwrap_or(0.)
    }
    pub fn evaluate(&self, score: f64) -> Result<f64> {
        if !score.is_finite() {
            return Err(invalid("finite latent score required"));
        }
        let (mut previous, mut current) = (1., score);
        let mut result = self.coefficients[0];
        for order in 1..self.coefficients.len() {
            if order > 1 {
                let next = (score * current - ((order - 1) as f64).sqrt() * previous)
                    / (order as f64).sqrt();
                previous = current;
                current = next;
            }
            result += self.coefficients[order] * current;
            if !current.is_finite() || !result.is_finite() {
                return Err(Error::Numerical {
                    context: "Hermite expansion overflow",
                });
            }
        }
        Ok(result)
    }
}

/// Reused local-neighborhood solver. Covariance callbacks are evaluated once for
/// conditioning/cross matrices; subsequent Hermite orders reuse their powers.
/// Hot buffers are preallocated. Large faer factorizations may still allocate
/// backend packing; callbacks retain their own allocation contract.
pub struct Workspace {
    maximum_observations: usize,
    maximum_targets: usize,
    base: Vec<f64>,
    powers: Vec<f64>,
    cross: Vec<f64>,
    cross_powers: Vec<f64>,
    deviations: Vec<f64>,
    previous: Vec<f64>,
    current: Vec<f64>,
    prediction: Vec<f64>,
    means: Vec<f64>,
    covariance: Mat<f64>,
    solver: GaussianWorkspace,
    _reservation: Reservation,
}
pub struct Estimate<'a> {
    pub means: &'a [f64],
    pub covariance: Option<MatRef<'a, f64>>,
    pub fitted_orders: usize,
}
impl Workspace {
    pub fn new(n: usize, m: usize, budget: &MemoryBudget) -> Result<Self> {
        if m == 0 {
            return Err(invalid("disjunctive target capacity required"));
        }
        let nn = checked_len(&[n, n])?;
        let nm = checked_len(&[n, m])?;
        let count = nn
            .checked_mul(2)
            .and_then(|x| x.checked_add(nm.checked_mul(2)?))
            .and_then(|x| x.checked_add(n.checked_mul(3)?))
            .and_then(|x| x.checked_add(m.checked_mul(3)?))
            .and_then(|x| x.checked_add(m.checked_mul(m)?))
            .ok_or(Error::SizeOverflow)?;
        let bytes = checked_len(&[count, 8])?;
        if bytes > isize::MAX as usize {
            return Err(Error::SizeOverflow);
        }
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        fn buffer(n: usize) -> Result<Vec<f64>> {
            let mut v = Vec::new();
            v.try_reserve_exact(n)
                .map_err(|_| invalid("disjunctive workspace allocation"))?;
            v.resize(n, 0.);
            Ok(v)
        }
        let mut covariance = Mat::new();
        covariance
            .try_reserve(m, m)
            .map_err(|_| invalid("disjunctive covariance allocation"))?;
        Ok(Self {
            maximum_observations: n,
            maximum_targets: m,
            base: buffer(nn)?,
            powers: buffer(nn)?,
            cross: buffer(nm)?,
            cross_powers: buffer(nm)?,
            deviations: buffer(n.checked_add(m).ok_or(Error::SizeOverflow)?)?,
            previous: buffer(n)?,
            current: buffer(n)?,
            prediction: buffer(m)?,
            means: buffer(m)?,
            covariance,
            solver: GaussianWorkspace::new(n, m, 0, budget)?,
            _reservation: reservation,
        })
    }
    /// Observations must already be marginal standard-normal scores of jointly
    /// Gaussian latent variables. Physical-unit means are encoded in expansions.
    /// Point support and exact latent observations only: nonlinear transformation
    /// does not commute with block averaging or additive measurement noise.
    pub fn estimate(
        &mut self,
        model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        expansions: &[Expansion],
        uncertainty: Uncertainty,
        cancel: &Cancellation,
    ) -> Result<Estimate<'_>> {
        cancel.check()?;
        let (n, m) = (observations.len(), targets.len());
        if n > self.maximum_observations
            || m == 0
            || m > self.maximum_targets
            || expansions.len() != model.variables()
        {
            return Err(invalid("disjunctive workspace or expansion dimensions"));
        }
        if observations.iter().any(|o| {
            !o.site().support.is_point()
                || o.noise_variance() != 0.
                || o.site().variable >= expansions.len()
        }) || targets
            .iter()
            .any(|s| !s.support.is_point() || s.variable >= expansions.len())
        {
            return Err(invalid(
                "disjunctive prediction needs exact point scores with valid variables",
            ));
        }
        for (i, s) in observations
            .iter()
            .map(Observation::site)
            .chain(targets)
            .enumerate()
        {
            cancel.check()?;
            let v = model.covariance(s, s)?;
            if !v.is_finite() || v <= 0. {
                return Err(invalid("positive finite latent marginal variance required"));
            }
            self.deviations[i] = v.sqrt();
        }
        for i in 0..n {
            cancel.check()?;
            model.covariance_row(
                observations[i].site(),
                &observations[..=i],
                &mut self.base[i * n..i * n + i + 1],
            )?;
            for j in 0..=i {
                let rho =
                    correlation(self.base[i * n + j], self.deviations[i], self.deviations[j])?;
                let rho = if observations[i].site() == observations[j].site() {
                    1.
                } else {
                    rho
                };
                self.base[i * n + j] = rho;
                self.base[j * n + i] = rho;
            }
        }
        for (j, t) in targets.iter().enumerate() {
            cancel.check()?;
            model.covariance_column(observations, t, &mut self.cross[j * n..(j + 1) * n])?;
            for (i, o) in observations.iter().enumerate() {
                let rho = correlation(
                    self.cross[j * n + i],
                    self.deviations[i],
                    self.deviations[n + j],
                )?;
                self.cross[j * n + i] = if o.site() == t { 1. } else { rho };
            }
            self.means[j] = expansions[t.variable].coefficient(0);
        }
        self.powers[..n * n].copy_from_slice(&self.base[..n * n]);
        self.cross_powers[..n * m].copy_from_slice(&self.cross[..n * m]);
        self.previous[..n].fill(1.);
        for (v, o) in self.current.iter_mut().zip(observations) {
            *v = o.value();
        }
        let joint = uncertainty == Uncertainty::Joint;
        self.covariance.resize_with(
            if joint { m } else { 0 },
            if joint { m } else { 0 },
            |_, _| 0.,
        );
        self.covariance.fill(0.);
        let order = targets
            .iter()
            .map(|t| expansions[t.variable].order())
            .max()
            .unwrap();
        let mut fitted_orders = 0;
        for r in 1..=order {
            cancel.check()?;
            if r > 1 {
                for (power, base) in self.powers[..n * n].iter_mut().zip(&self.base) {
                    *power *= base;
                }
                for (power, base) in self.cross_powers[..n * m].iter_mut().zip(&self.cross) {
                    *power *= base;
                }
                for (i, o) in observations.iter().enumerate() {
                    let next = (o.value() * self.current[i]
                        - ((r - 1) as f64).sqrt() * self.previous[i])
                        / (r as f64).sqrt();
                    if !next.is_finite() {
                        return Err(Error::Numerical {
                            context: "Hermite observation recurrence overflow",
                        });
                    }
                    self.previous[i] = self.current[i];
                    self.current[i] = next;
                }
            }
            if targets
                .iter()
                .all(|t| expansions[t.variable].coefficient(r) == 0.)
            {
                continue;
            }
            let kernel = Powers {
                model,
                observations,
                targets,
                data: &self.powers[..n * n],
                cross: &self.cross_powers[..n * m],
                order: r as i32,
            };
            let plan = self.solver.prepare(
                &kernel,
                observations,
                targets,
                &Simple,
                WorkspaceOptions {
                    uncertainty,
                    tolerance: NumericalTolerance::default(),
                    cancellation: cancel,
                },
            )?;
            plan.estimate_into(&self.current[..n], &mut self.prediction[..m])?;
            for (j, t) in targets.iter().enumerate() {
                self.means[j] += expansions[t.variable].coefficient(r) * self.prediction[j];
                if !self.means[j].is_finite() {
                    return Err(Error::Numerical {
                        context: "disjunctive estimate overflow",
                    });
                }
            }
            if let Some(cov) = plan.covariance() {
                for j in 0..m {
                    cancel.check()?;
                    for i in 0..m {
                        let value = product3(
                            expansions[targets[i].variable].coefficient(r),
                            expansions[targets[j].variable].coefficient(r),
                            cov[(i, j)],
                        )?;
                        self.covariance[(i, j)] += value;
                        if !self.covariance[(i, j)].is_finite() {
                            return Err(Error::Numerical {
                                context: "disjunctive covariance overflow",
                            });
                        }
                    }
                }
            }
            fitted_orders += 1;
        }
        cancel.check()?;
        Ok(Estimate {
            means: &self.means[..m],
            covariance: joint.then(|| self.covariance.as_ref()),
            fitted_orders,
        })
    }
}
fn correlation(value: f64, a: f64, b: f64) -> Result<f64> {
    // Divide by the smaller deviation first to preserve small correlations
    // between fields expressed in very different units.
    let rho = value / a.min(b) / a.max(b);
    if !rho.is_finite() || rho.abs() > 1. + 1e-10 {
        return Err(invalid("invalid latent correlation"));
    }
    Ok(rho.clamp(-1., 1.))
}
fn product3(a: f64, b: f64, c: f64) -> Result<f64> {
    if a == 0. || b == 0. || c == 0. {
        return Ok(0.);
    }
    let (ea, eb, ec) = (exponent(a.abs()), exponent(b.abs()), exponent(c.abs()));
    let value = shift(shift(a, -ea) * shift(b, -eb) * shift(c, -ec), ea + eb + ec);
    if !value.is_finite() {
        return Err(Error::Numerical {
            context: "disjunctive covariance term overflow",
        });
    }
    Ok(value)
}
struct Powers<'a, C> {
    model: &'a C,
    observations: &'a [Observation],
    targets: &'a [Site],
    data: &'a [f64],
    cross: &'a [f64],
    order: i32,
}
impl<C: Covariance> Covariance for Powers<'_, C> {
    fn variables(&self) -> usize {
        self.model.variables()
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        let aa = self.model.point_covariance(a, va, a, va)?;
        let bb = self.model.point_covariance(b, vb, b, vb)?;
        if !aa.is_finite() || !bb.is_finite() || aa <= 0. || bb <= 0. {
            return Err(invalid("positive latent marginal variance required"));
        }
        if a == b && va == vb {
            return Ok(1.);
        }
        Ok(correlation(
            self.model.point_covariance(a, va, b, vb)?,
            aa.sqrt(),
            bb.sqrt(),
        )?
        .powi(self.order))
    }
    fn covariance_row(&self, a: &Site, data: &[Observation], out: &mut [f64]) -> Result<()> {
        if data.len() != out.len() {
            return Err(invalid("Hermite covariance row length"));
        }
        if !data.is_empty()
            && data.len() <= self.observations.len()
            && std::ptr::eq(data.as_ptr(), self.observations.as_ptr())
            && data.last().unwrap().site() == a
        {
            let start = (data.len() - 1) * self.observations.len();
            out.copy_from_slice(&self.data[start..start + data.len()]);
            return Ok(());
        }
        for (v, o) in out.iter_mut().zip(data) {
            *v = self.covariance(a, o.site())?;
        }
        Ok(())
    }
    fn covariance_column(&self, data: &[Observation], b: &Site, out: &mut [f64]) -> Result<()> {
        if data.len() != out.len() {
            return Err(invalid("Hermite covariance column length"));
        }
        if data.len() == self.observations.len()
            && std::ptr::eq(data.as_ptr(), self.observations.as_ptr())
        {
            if let Some(j) = self.targets.iter().position(|t| t == b) {
                let start = j * data.len();
                out.copy_from_slice(&self.cross[start..start + data.len()]);
                return Ok(());
            }
        }
        for (v, o) in out.iter_mut().zip(data) {
            *v = self.covariance(o.site(), b)?;
        }
        Ok(())
    }
}
