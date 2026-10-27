//! Capacity-reusing storage for a plan retained beyond its solver preparation.
use super::*;

pub(crate) struct RetainedPlan {
    plan: GaussianPlan,
    observations: usize,
    targets: usize,
    joint_targets: usize,
    valid: bool,
}
impl RetainedPlan {
    pub(crate) fn new(budget: &MemoryBudget) -> Result<Self> {
        Ok(Self {
            plan: GaussianPlan {
                weights: Mat::new(),
                covariance: Mat::new(),
                factor: Mat::new(),
                joint: false,
                roundoff_corrections: 0,
                reservation: budget.reserve(0)?,
            },
            observations: 0,
            targets: 0,
            joint_targets: 0,
            valid: false,
        })
    }
    pub(crate) fn invalidate(&mut self) {
        self.valid = false;
    }
    pub(crate) fn get(&self) -> Result<&GaussianPlan> {
        if self.valid {
            Ok(&self.plan)
        } else {
            Err(invalid("no successfully retained Gaussian plan"))
        }
    }
    /// Exchange equally charged buffers after preparation and its diagnostics
    /// have succeeded. Neither reservation moves: the workspace also charges
    /// its other numerical arenas, whereas this slot charges only its plan.
    /// Unequal capacities use the copy path so a larger retained slot can never
    /// move uncharged capacity into a smaller worker (or vice versa).
    pub(crate) fn update_from_workspace(&mut self, source: &mut GaussianWorkspace) -> Result<()> {
        if self.observations != source.maximum_observations
            || self.targets != source.maximum_targets
            || self.joint_targets != source.maximum_targets
        {
            let _span = crate::execution::profiling::Span::new(
                crate::execution::profiling::Metric::RetainCopy,
            );
            return self.update(&source.plan);
        }
        let _span =
            crate::execution::profiling::Span::new(crate::execution::profiling::Metric::RetainSwap);
        self.invalidate();
        std::mem::swap(&mut self.plan.weights, &mut source.plan.weights);
        std::mem::swap(&mut self.plan.covariance, &mut source.plan.covariance);
        std::mem::swap(&mut self.plan.factor, &mut source.plan.factor);
        std::mem::swap(&mut self.plan.joint, &mut source.plan.joint);
        std::mem::swap(
            &mut self.plan.roundoff_corrections,
            &mut source.plan.roundoff_corrections,
        );
        self.valid = true;
        Ok(())
    }
    pub(crate) fn update(&mut self, source: &GaussianPlan) -> Result<()> {
        self.invalidate();
        let n = source.weights.nrows();
        let m = source.weights.ncols();
        let maximum_n = n.max(self.observations);
        let maximum_m = m.max(self.targets);
        let joint_m = self.joint_targets.max(if source.joint { m } else { 0 });
        if maximum_n != self.observations
            || maximum_m != self.targets
            || joint_m != self.joint_targets
        {
            let entries = checked_len(&[maximum_n, maximum_m])?
                .checked_add(checked_len(&[joint_m, joint_m, 2])?)
                .ok_or(Error::SizeOverflow)?;
            let bytes =
                u64::try_from(checked_len(&[entries, 8])?).map_err(|_| Error::SizeOverflow)?;
            self.plan.reservation.resize(bytes)?;
            if self.plan.weights.try_reserve(maximum_n, maximum_m).is_err()
                || self.plan.covariance.try_reserve(joint_m, joint_m).is_err()
                || self.plan.factor.try_reserve(joint_m, joint_m).is_err()
            {
                // Discard partial growth before releasing its charge. The old
                // plan must never be exposed as the result of this update.
                self.plan.weights = Mat::new();
                self.plan.covariance = Mat::new();
                self.plan.factor = Mat::new();
                self.observations = 0;
                self.targets = 0;
                self.joint_targets = 0;
                self.plan.reservation.resize(0)?;
                return Err(invalid("retained Gaussian plan allocation"));
            }
            self.observations = maximum_n;
            self.targets = maximum_m;
            self.joint_targets = joint_m;
        }
        self.plan.weights.resize_with(n, m, |_, _| 0.);
        self.plan
            .weights
            .as_mut()
            .copy_from(source.weights.as_ref());
        let joint = if source.joint { m } else { 0 };
        self.plan.covariance.resize_with(joint, joint, |_, _| 0.);
        self.plan.factor.resize_with(joint, joint, |_, _| 0.);
        if source.joint {
            self.plan
                .covariance
                .as_mut()
                .copy_from(source.covariance.as_ref());
            self.plan.factor.as_mut().copy_from(source.factor.as_ref());
        }
        self.plan.joint = source.joint;
        self.plan.roundoff_corrections = source.roundoff_corrections;
        self.valid = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::Point3,
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    };

    #[test]
    fn exchanged_plans_preserve_bits_capacities_and_independent_budget_lifetimes() {
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(3.).unwrap(),
            coregionalization: Coregionalization::scalar(1.).unwrap(),
        }])
        .unwrap();
        let observations: Vec<_> = (0..32)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(Point3::new([i as f64 * 0.2, 0., 0.]).unwrap(), 0),
                    i as f64,
                    0.01,
                )
                .unwrap()
            })
            .collect();
        let targets: Vec<_> = (0..8)
            .map(|i| Site::point(Point3::new([0.13 + i as f64 * 0.11, 0.1, 0.]).unwrap(), 0))
            .collect();
        let budget = MemoryBudget::new(1 << 20);
        let retained_bytes = (32 * 8 + 2 * 8 * 8) * 8;
        let retained_budget = MemoryBudget::new(retained_bytes);
        let mut workspace = GaussianWorkspace::new(32, 8, 0, &budget).unwrap();
        let worker_charge = budget.used();
        let mut retained = RetainedPlan::new(&retained_budget).unwrap();
        let cancel = Cancellation::default();
        for (iteration, (n, m, joint)) in [
            (32, 8, true),
            (4, 2, true),
            (0, 1, true),
            (16, 7, false),
            (32, 8, true),
        ]
        .into_iter()
        .enumerate()
        {
            let source = workspace
                .prepare(
                    &model,
                    &observations[..n],
                    &targets[..m],
                    &Simple,
                    WorkspaceOptions {
                        uncertainty: if joint {
                            Uncertainty::Joint
                        } else {
                            Uncertainty::None
                        },
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )
                .unwrap();
            let weights: Vec<_> = (0..m)
                .flat_map(|j| (0..n).map(move |i| source.weights[(i, j)].to_bits()))
                .collect();
            let covariance: Vec<_> = if joint {
                (0..m)
                    .flat_map(|j| (0..m).map(move |i| source.covariance[(i, j)].to_bits()))
                    .collect()
            } else {
                Vec::new()
            };
            let values: Vec<_> = observations[..n].iter().map(Observation::value).collect();
            let mean = source.estimate(&values).unwrap();
            let draw = joint.then(|| source.sample(&mean, 72, 3, 10).unwrap());
            let pointers = (
                source.weights.as_ptr(),
                source.covariance.as_ptr(),
                source.factor.as_ptr(),
            );
            let old_pointers = (
                retained.plan.weights.as_ptr(),
                retained.plan.covariance.as_ptr(),
                retained.plan.factor.as_ptr(),
            );
            let measured = allocation_counter::measure(|| {
                retained.update_from_workspace(&mut workspace).unwrap()
            });
            let plan = retained.get().unwrap();
            if iteration != 0 {
                assert_eq!(measured.count_total, 0);
                assert_eq!(
                    pointers,
                    (
                        plan.weights.as_ptr(),
                        plan.covariance.as_ptr(),
                        plan.factor.as_ptr()
                    )
                );
                assert_eq!(
                    old_pointers,
                    (
                        workspace.plan.weights.as_ptr(),
                        workspace.plan.covariance.as_ptr(),
                        workspace.plan.factor.as_ptr()
                    )
                );
            }
            assert_eq!(
                weights,
                (0..m)
                    .flat_map(|j| (0..n).map(move |i| plan.weights[(i, j)].to_bits()))
                    .collect::<Vec<_>>()
            );
            assert_eq!(mean, plan.estimate(&values).unwrap());
            if let Some(draw) = draw {
                assert_eq!(draw, plan.sample(&mean, 72, 3, 10).unwrap());
                assert_eq!(
                    covariance,
                    (0..m)
                        .flat_map(|j| (0..m).map(move |i| plan.covariance[(i, j)].to_bits()))
                        .collect::<Vec<_>>()
                );
            } else {
                assert!(plan.covariance().is_none());
            }
            assert_eq!(budget.used(), worker_charge);
            assert_eq!(retained_budget.used(), retained_bytes);
        }
        // A larger slot reaching a smaller worker must copy, not exchange its
        // larger buffers into the smaller worker's reservation.
        let mut small = GaussianWorkspace::new(4, 2, 0, &budget).unwrap();
        small
            .prepare(
                &model,
                &observations[..4],
                &targets[..2],
                &Simple,
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        let small_pointer = small.plan.weights.as_ptr();
        let used = budget.used();
        let measured =
            allocation_counter::measure(|| retained.update_from_workspace(&mut small).unwrap());
        assert_eq!(measured.count_total, 0);
        assert_eq!(small_pointer, small.plan.weights.as_ptr());
        assert_ne!(small_pointer, retained.get().unwrap().weights.as_ptr());
        assert_eq!(budget.used(), used);
        drop(small);
        drop(workspace);
        assert_eq!(budget.used(), 0);
        assert!(retained.get().unwrap().sample(&[0., 0.], 72, 3, 10).is_ok());
        assert_eq!(retained_budget.used(), retained_bytes);
        drop(retained);
        assert_eq!(retained_budget.used(), 0);
    }

    #[test]
    fn retained_shapes_reuse_storage_and_preserve_predictions_and_draws() {
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(3.).unwrap(),
            coregionalization: Coregionalization::scalar(1.).unwrap(),
        }])
        .unwrap();
        let observations: Vec<_> = (0..32)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(Point3::new([i as f64 * 0.2, 0., 0.]).unwrap(), 0),
                    i as f64,
                    0.01,
                )
                .unwrap()
            })
            .collect();
        let targets: Vec<_> = (0..8)
            .map(|i| Site::point(Point3::new([0.13 + i as f64 * 0.11, 0.1, 0.]).unwrap(), 0))
            .collect();
        let budget = MemoryBudget::new(1 << 20);
        let retained_budget = MemoryBudget::new((32 * 8 + 2 * 8 * 8) * 8);
        let cancel = Cancellation::default();
        let mut workspace = GaussianWorkspace::new(32, 8, 0, &budget).unwrap();
        let mut retained = RetainedPlan::new(&retained_budget).unwrap();
        assert!(retained.get().is_err());
        for (iteration, (n, m, joint)) in [
            (32, 8, true),
            (4, 2, true),
            (0, 1, true),
            (16, 7, false),
            (32, 8, true),
        ]
        .into_iter()
        .enumerate()
        {
            let source = workspace
                .prepare(
                    &model,
                    &observations[..n],
                    &targets[..m],
                    &Simple,
                    WorkspaceOptions {
                        uncertainty: if joint {
                            Uncertainty::Joint
                        } else {
                            Uncertainty::None
                        },
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )
                .unwrap();
            // The first update grows. All later shapes fit the charged capacity.
            let measured = allocation_counter::measure(|| retained.update(source).unwrap());
            if iteration != 0 {
                assert_eq!(measured.count_total, 0);
            }
            let copy = retained.get().unwrap();
            assert_eq!(retained_budget.used(), (32 * 8 + 2 * 8 * 8) * 8);
            assert_eq!(copy.weights().shape(), source.weights().shape());
            for i in 0..n {
                for j in 0..m {
                    assert_eq!(
                        copy.weights()[(i, j)].to_bits(),
                        source.weights()[(i, j)].to_bits()
                    );
                }
            }
            let values: Vec<_> = observations[..n].iter().map(Observation::value).collect();
            let prediction = source.estimate(&values).unwrap();
            assert_eq!(prediction, copy.estimate(&values).unwrap());
            assert_eq!(copy.roundoff_corrections(), source.roundoff_corrections());
            if joint {
                for i in 0..m {
                    for j in 0..m {
                        assert_eq!(
                            copy.covariance().unwrap()[(i, j)].to_bits(),
                            source.covariance().unwrap()[(i, j)].to_bits()
                        );
                    }
                }
                assert_eq!(
                    copy.sample(&prediction, 72, 3, 10).unwrap(),
                    source.sample(&prediction, 72, 3, 10).unwrap()
                );
            } else {
                assert!(copy.covariance().is_none());
                assert!(copy.sample(&prediction, 72, 3, 10).is_err());
            }
        }
        drop(workspace);
        assert_eq!(budget.used(), 0);
        assert!(retained.get().is_ok());
        drop(retained);
        assert_eq!(retained_budget.used(), 0);
    }

    #[test]
    fn failed_growth_hides_previous_plan_and_allows_recovery() {
        // Copying is independent of the source's mathematical construction.
        let source = |n, m| GaussianPlan {
            weights: Mat::zeros(n, m),
            covariance: Mat::identity(m, m),
            factor: Mat::identity(m, m),
            joint: true,
            roundoff_corrections: 0,
            reservation: MemoryBudget::new(0).reserve(0).unwrap(),
        };
        let small = source(4, 2);
        let large = source(5, 3);
        let budget = MemoryBudget::new((4 * 2 + 2 * 2 * 2) * 8);
        let mut retained = RetainedPlan::new(&budget).unwrap();
        retained.update(&small).unwrap();
        let used = budget.used();
        assert!(retained.update(&large).is_err());
        assert!(retained.get().is_err());
        assert_eq!(used, budget.used());
        let measured = allocation_counter::measure(|| retained.update(&small).unwrap());
        assert_eq!(measured.count_total, 0);
        assert!(retained.get().is_ok());
        retained.invalidate();
        assert!(retained.get().is_err());
        drop(retained);
        assert_eq!(budget.used(), 0);
    }
}
