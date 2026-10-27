//! A single-entry conditioning cache scoped to one deterministic estimation call.
use super::*;

// Own the workspace so no other operation can overwrite its factors. Borrow the
// model and drift for the entire call; no pointer identity or global state. The
// parallel driver requires stable outputs, including interior mutable callbacks.
pub(crate) struct NeighborhoodConditioning<'a, C, D> {
    model: &'a C,
    drift: &'a D,
    workspace: Option<GaussianWorkspace>,
    observations: Vec<Observation>,
    prepared: Option<(usize, usize)>,
    reservation: Reservation,
    budget: MemoryBudget,
}
impl<'a, C: Covariance, D: Drift> NeighborhoodConditioning<'a, C, D> {
    pub(crate) fn new(model: &'a C, drift: &'a D, budget: &MemoryBudget) -> Result<Self> {
        Ok(Self {
            model,
            drift,
            workspace: None,
            observations: Vec::new(),
            prepared: None,
            reservation: budget.reserve(0)?,
            budget: budget.clone(),
        })
    }
    // Release insufficient solver capacity before growing other worker arenas.
    pub(crate) fn release_if_insufficient(&mut self, n: usize, m: usize) {
        if self
            .workspace
            .as_ref()
            .is_some_and(|w| !w.supports_shape(n, m, self.drift.columns()))
        {
            self.workspace = None;
            self.prepared = None;
        }
    }
    #[cfg(test)]
    pub(crate) fn prepare(
        &mut self,
        observations: &[Observation],
        targets: &[Site],
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        self.prepare_component(self.model, self.drift, observations, targets, options)
    }
    pub(crate) fn prepare_component(
        &mut self,
        target_model: &impl Covariance,
        target_drift: &impl Drift,
        observations: &[Observation],
        targets: &[Site],
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        options.cancellation.check()?;
        options.tolerance.threshold(1.)?;
        let n = observations.len();
        let p = self.drift.columns();
        let dimensions = (self.model.variables(), p);
        if targets.is_empty() || p > n || targets.iter().any(|t| t.variable >= dimensions.0) {
            return Err(invalid(
                "estimation conditioning dimensions or target variables",
            ));
        }
        self.release_if_insufficient(n, targets.len());
        let matches = self.prepared == Some(dimensions)
            && self.observations.len() == n
            && self
                .observations
                .iter()
                .zip(observations)
                .all(|(a, b)| a.same_bits(b));
        if !matches {
            // Invalidate before any fallible growth, callback or factorization.
            self.prepared = None;
            if n > self.observations.capacity() {
                let previous = self.reservation.bytes();
                let bytes = checked_len(&[n, std::mem::size_of::<Observation>()])?;
                self.reservation
                    .resize(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
                if self
                    .observations
                    .try_reserve_exact(n - self.observations.len())
                    .is_err()
                {
                    self.reservation.resize(previous)?;
                    return Err(invalid("conditioning cache allocation"));
                }
            }
            GaussianWorkspace::cached(&mut self.workspace, n, targets.len(), p, &self.budget)?
                .prepare_data(self.model, observations, self.drift, options.cancellation)?;
            self.observations.clear();
            self.observations.extend_from_slice(observations);
            self.prepared = Some(dimensions);
        }
        self.workspace
            .as_mut()
            .expect("prepared conditioning workspace")
            .prepare_targets(
                target_model,
                observations,
                targets,
                target_drift,
                dimensions.0,
                p,
                options,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_component_with_residuals(
        &mut self,
        target_model: &impl Covariance,
        target_drift: &impl Drift,
        observations: &[Observation],
        targets: &[Site],
        diagnostics: &mut ResidualWorkspace,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&GaussianPlan, SolveResiduals)> {
        let cancel = options.cancellation;
        self.prepare_component(target_model, target_drift, observations, targets, options)?;
        let workspace = self
            .workspace
            .as_ref()
            .expect("prepared conditioning workspace");
        let mut report = diagnostics.evaluate(
            workspace,
            self.model,
            target_model,
            observations,
            targets,
            cancel,
        )?;
        report.original_drift_constraints = diagnostics.original_drift_with_target(
            workspace,
            observations,
            targets,
            self.drift,
            target_drift,
            cancel,
        )?;
        Ok((&workspace.plan, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{Point3, Support},
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counted {
        model: Lmc,
        rows: AtomicUsize,
    }
    impl Covariance for Counted {
        fn variables(&self) -> usize {
            self.model.variables()
        }
        fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
            self.model.point_covariance(a, va, b, vb)
        }
        fn covariance_row(&self, a: &Site, data: &[Observation], out: &mut [f64]) -> Result<()> {
            self.rows.fetch_add(1, Ordering::Relaxed);
            self.model.covariance_row(a, data, out)
        }
        fn covariance_column(&self, data: &[Observation], b: &Site, out: &mut [f64]) -> Result<()> {
            self.model.covariance_column(data, b, out)
        }
    }
    fn fixture() -> (Counted, Vec<Observation>, Vec<Site>) {
        let model = Counted {
            model: Lmc::new(vec![Structure {
                correlation: Correlation::Exponential,
                anisotropy: Anisotropy::isotropic(3.).unwrap(),
                coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.3, 0.8])
                    .unwrap(),
            }])
            .unwrap(),
            rows: AtomicUsize::new(0),
        };
        let data = (0..8)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new([i as f64 * 0.3, 0., 0.]).unwrap(),
                        i as usize % 2,
                    ),
                    i as f64,
                    0.01,
                )
                .unwrap()
            })
            .collect();
        let targets = (0..3)
            .map(|i| {
                Site::point(
                    Point3::new([0.13 + i as f64 * 0.1, 0.1, 0.]).unwrap(),
                    i % 2,
                )
            })
            .collect();
        (model, data, targets)
    }
    fn bits(plan: &GaussianPlan) -> Vec<u64> {
        let mut out = Vec::new();
        for mat in [plan.weights(), plan.covariance().unwrap()] {
            for j in 0..mat.ncols() {
                for i in 0..mat.nrows() {
                    out.push(mat[(i, j)].to_bits());
                }
            }
        }
        out.push(plan.roundoff_corrections() as u64);
        out
    }
    #[test]
    fn every_observable_record_field_and_order_invalidates_the_cache() {
        let (model, data, targets) = fixture();
        let budget = MemoryBudget::new(1 << 22);
        let cancel = Cancellation::default();
        let options = || WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: &cancel,
        };
        let drift = Ordinary { variables: 2 };
        let mut cache = NeighborhoodConditioning::new(&model, &drift, &budget).unwrap();
        let mut fresh = GaussianWorkspace::new(8, 3, 2, &budget).unwrap();
        for field in 0..9 {
            let mut changed = data.clone();
            let mut site = *data[0].site();
            let mut id = data[0].id();
            let mut value = data[0].value();
            let mut noise = data[0].noise_variance();
            match field {
                0 => id = 99,
                1 => value = -0.,
                2 => noise = 0.02,
                3 => site.variable = 1,
                4 => site.support = Support::point(Point3::new([-0., 0., 0.]).unwrap()),
                5 => {
                    site.support =
                        Support::block(Point3::new([0.; 3]).unwrap(), [-0., 0., 0.], [1; 3])
                            .unwrap()
                }
                6 => {
                    site.support =
                        Support::block(Point3::new([0.; 3]).unwrap(), [0.1, 0., 0.], [2, 1, 1])
                            .unwrap()
                }
                7 => {
                    site.support =
                        Support::block(Point3::new([0.; 3]).unwrap(), [0.1, 0., 0.], [3, 1, 1])
                            .unwrap()
                }
                _ => {}
            }
            changed[0] = Observation::new(id, site, value, noise).unwrap();
            if field == 8 {
                changed.swap(0, 1);
            }
            cache.prepare(&data, &targets, options()).unwrap();
            let before = model.rows.load(Ordering::Relaxed);
            let actual = bits(cache.prepare(&changed, &targets[..2], options()).unwrap());
            assert_eq!(
                model.rows.load(Ordering::Relaxed) - before,
                data.len(),
                "field {field}"
            );
            assert_eq!(
                actual,
                bits(
                    fresh
                        .prepare(&model.model, &changed, &targets[..2], &drift, options())
                        .unwrap()
                )
            );
            let before = model.rows.load(Ordering::Relaxed);
            let actual = bits(cache.prepare(&changed, &targets, options()).unwrap());
            assert_eq!(model.rows.load(Ordering::Relaxed), before);
            assert_eq!(
                actual,
                bits(
                    fresh
                        .prepare(&model.model, &changed, &targets, &drift, options())
                        .unwrap()
                )
            );
        }
        drop(cache);
        drop(fresh);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn target_failures_capacity_changes_and_failed_conditioning_recover() {
        let (model, data, targets) = fixture();
        let budget = MemoryBudget::new(1 << 22);
        let cancel = Cancellation::default();
        let options = || WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: &cancel,
        };
        let mut cache = NeighborhoodConditioning::new(&model, &Simple, &budget).unwrap();
        let mut fresh = GaussianWorkspace::new(8, 3, 0, &budget).unwrap();
        cache.prepare(&data, &targets[..1], options()).unwrap();
        assert!(cache.prepare(&data, &[], options()).is_err());
        let invalid = [Site::point(Point3::new([0.; 3]).unwrap(), 2)];
        assert!(cache.prepare(&data, &invalid, options()).is_err());
        let before = model.rows.load(Ordering::Relaxed);
        cache.prepare(&data, &targets[..1], options()).unwrap();
        assert_eq!(model.rows.load(Ordering::Relaxed), before);
        cache.prepare(&data, &targets, options()).unwrap();
        assert_eq!(
            model.rows.load(Ordering::Relaxed) - before,
            data.len(),
            "target capacity growth rebuilds factors"
        );
        for slice in [&data[..4], &data[..], &data[..0]] {
            assert_eq!(
                bits(cache.prepare(slice, &targets, options()).unwrap()),
                bits(
                    fresh
                        .prepare(&model.model, slice, &targets, &Simple, options())
                        .unwrap()
                )
            );
        }
        let repeated = vec![data[0], data[0]];
        assert!(cache.prepare(&repeated, &targets, options()).is_err());
        assert_eq!(
            bits(cache.prepare(&data, &targets, options()).unwrap()),
            bits(
                fresh
                    .prepare(&model.model, &data, &targets, &Simple, options())
                    .unwrap()
            )
        );
        let singular: Vec<_> = (0..2)
            .map(|i| Observation::new(i, *data[0].site(), 0., 0.).unwrap())
            .collect();
        assert!(cache.prepare(&singular, &targets, options()).is_err());
        assert_eq!(
            bits(cache.prepare(&data, &targets, options()).unwrap()),
            bits(
                fresh
                    .prepare(&model.model, &data, &targets, &Simple, options())
                    .unwrap()
            )
        );
        // Deny cache growth after a successful smaller preparation, then release
        // the competing reservation and require a fully rebuilt, correct result.
        drop(cache);
        let mut cache = NeighborhoodConditioning::new(&model, &Simple, &budget).unwrap();
        cache.prepare(&data[..1], &targets, options()).unwrap();
        let competing = budget.reserve(budget.limit() - budget.used()).unwrap();
        assert!(cache.prepare(&data, &targets, options()).is_err());
        drop(competing);
        assert_eq!(
            bits(cache.prepare(&data, &targets, options()).unwrap()),
            bits(
                fresh
                    .prepare(&model.model, &data, &targets, &Simple, options())
                    .unwrap()
            )
        );
        drop(cache);
        drop(fresh);
        assert_eq!(budget.used(), 0);
    }
}
