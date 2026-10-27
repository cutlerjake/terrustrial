//! Reusable, budgeted storage for path-ordered Gaussian drawing.
use super::*;
use crate::execution::Reservation;

pub(super) struct DrawingWorkspace {
    observations: usize,
    targets: usize,
    with_mean: bool,
    values: Vec<f64>,
    target_means: Vec<f64>,
    prediction: Vec<f64>,
    noise: Vec<f64>,
    draw: Vec<f64>,
    output: Vec<Observation>,
    reservation: Reservation,
}
impl DrawingWorkspace {
    pub(super) fn new(budget: &MemoryBudget) -> Result<Self> {
        Ok(Self {
            observations: 0,
            targets: 0,
            with_mean: false,
            values: Vec::new(),
            target_means: Vec::new(),
            prediction: Vec::new(),
            noise: Vec::new(),
            draw: Vec::new(),
            output: Vec::new(),
            reservation: budget.reserve(0)?,
        })
    }
    fn ensure(&mut self, n: usize, m: usize, with_mean: bool) -> Result<()> {
        if n <= self.observations && m <= self.targets && (!with_mean || self.with_mean) {
            return Ok(());
        }
        let n = n.max(self.observations);
        let m = m.max(self.targets);
        let with_mean = with_mean || self.with_mean;
        let bytes = checked_len(&[n, 8])?
            .checked_add(checked_len(&[
                m,
                std::mem::size_of::<Observation>() + if with_mean { 32 } else { 24 },
            ])?)
            .ok_or(Error::SizeOverflow)?;
        self.reservation
            .resize(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        fn reserve<T>(buffer: &mut Vec<T>, capacity: usize) -> bool {
            capacity <= buffer.capacity()
                || buffer.try_reserve_exact(capacity - buffer.len()).is_ok()
        }
        if !reserve(&mut self.values, n)
            || !reserve(&mut self.prediction, m)
            || !reserve(&mut self.noise, m)
            || !reserve(&mut self.draw, m)
            || !reserve(&mut self.output, m)
            || (with_mean && !reserve(&mut self.target_means, m))
        {
            // A vector may already have grown. Release all scratch so a failed
            // multi-buffer growth cannot leave an understated memory charge.
            self.values = Vec::new();
            self.prediction = Vec::new();
            self.noise = Vec::new();
            self.draw = Vec::new();
            self.output = Vec::new();
            self.target_means = Vec::new();
            self.observations = 0;
            self.targets = 0;
            self.with_mean = false;
            self.reservation.resize(0)?;
            return Err(invalid("simulation drawing buffer allocation"));
        }
        self.observations = n;
        self.targets = m;
        self.with_mean = with_mean;
        Ok(())
    }
    pub(super) fn draw<'a>(
        &'a mut self,
        plan: &GaussianPlan,
        observations: &[Observation],
        group: &Group,
        options: &SimulationOptions<'_>,
    ) -> Result<&'a [Observation]> {
        options.cancellation.check()?;
        self.output.clear();
        let m = group.targets.len();
        if plan.weights().nrows() != observations.len() || plan.weights().ncols() != m {
            return Err(invalid("simulation drawing plan dimensions"));
        }
        self.ensure(observations.len(), m, options.mean.is_some())?;
        self.values.clear();
        self.values
            .extend(observations.iter().map(Observation::value));
        self.target_means.clear();
        if let Some(mean) = options.mean {
            let evaluate = |site: &Site| -> Result<f64> {
                options.cancellation.check()?;
                let value = mean.evaluate(site)?;
                options.cancellation.check()?;
                if !value.is_finite() {
                    return Err(invalid(
                        "known mean must be finite on every selected support",
                    ));
                }
                Ok(value)
            };
            for (value, observation) in self.values.iter_mut().zip(observations) {
                *value -= evaluate(observation.site())?;
                if !value.is_finite() {
                    return Err(Error::Numerical {
                        context: "known-mean observation residual",
                    });
                }
            }
            for target in &group.targets {
                self.target_means.push(evaluate(&target.site)?);
            }
        }
        self.prediction.resize(m, 0.);
        plan.estimate_into(&self.values, &mut self.prediction)?;
        if options.mean.is_some() {
            for (prediction, target_mean) in self.prediction.iter_mut().zip(&self.target_means) {
                *prediction += target_mean;
                if !prediction.is_finite() {
                    return Err(Error::Numerical {
                        context: "known-mean conditional prediction",
                    });
                }
            }
        }
        options.cancellation.check()?;
        self.noise.resize(m, 0.);
        self.draw.resize(m, 0.);
        plan.sample_into(
            &self.prediction,
            options.seed,
            options.realization,
            group.id,
            &mut self.noise,
            &mut self.draw,
        )?;
        for (target, &draw) in group.targets.iter().zip(&self.draw) {
            let value = observations
                .iter()
                .find(|o| o.noise_variance() == 0. && o.site() == &target.site)
                .map_or(draw, Observation::value);
            self.output
                .push(Observation::new(target.id, target.site, value, 0.)?);
        }
        Ok(&self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::Support,
        gaussian::{PrepareOptions, Simple},
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
        spatial::Neighborhood,
    };
    fn options<'a>(
        budget: &'a MemoryBudget,
        cancel: &'a Cancellation,
        mean: Option<&'a dyn KnownMean>,
    ) -> SimulationOptions<'a> {
        SimulationOptions {
            seed: 901,
            realization: 7,
            mean,
            hard_neighborhood: Neighborhood {
                maximum: 32,
                radius: 100.,
            }
            .into(),
            simulated_neighborhood: Neighborhood {
                maximum: 32,
                radius: 100.,
            }
            .into(),
            budget,
            cancellation: cancel,
        }
    }
    fn model() -> Lmc {
        Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(3.).unwrap(),
            coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.3, 0.8]).unwrap(),
        }])
        .unwrap()
    }
    fn data() -> Vec<Observation> {
        (0..32)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new([i as f64 * 0.2, 0., 0.]).unwrap(),
                        i as usize % 2,
                    ),
                    if i == 0 { -0. } else { (i as f64).sin() },
                    if i == 0 { 0. } else { 0.01 },
                )
                .unwrap()
            })
            .collect()
    }
    fn group(m: usize) -> Group {
        Group {
            id: 19 + m as u64,
            targets: (0..m)
                .map(|i| Target {
                    id: 100 + i as u64,
                    site: Site {
                        variable: i % 2,
                        support: if i == 0 {
                            Support::point(Point3::new([0.; 3]).unwrap())
                        } else {
                            Support::block(
                                Point3::new([0.13 + i as f64 * 0.2, 0.1, 0.]).unwrap(),
                                [0.1, 0., 0.],
                                [3, 1, 1],
                            )
                            .unwrap()
                        },
                    },
                })
                .collect(),
        }
    }
    fn trend(site: &Site) -> Result<f64> {
        Ok(site
            .support
            .nodes()
            .map(|p| p.coordinates()[0].powi(2) + site.variable as f64)
            .sum::<f64>()
            / site.support.node_count() as f64)
    }
    // Allocating public-API reference preserves the original residual, prediction,
    // target-mean addition, seeded draw and exact-pin arithmetic order.
    fn reference(
        plan: &GaussianPlan,
        observations: &[Observation],
        group: &Group,
        mean: Option<&dyn KnownMean>,
    ) -> Vec<Observation> {
        let values: Vec<_> = observations
            .iter()
            .map(|o| o.value() - mean.map(|m| m.evaluate(o.site()).unwrap()).unwrap_or(0.))
            .collect();
        let mut prediction = plan.estimate(&values).unwrap();
        if let Some(mean) = mean {
            for (p, t) in prediction.iter_mut().zip(&group.targets) {
                *p += mean.evaluate(&t.site).unwrap();
            }
        }
        let draw = plan.sample(&prediction, 901, 7, group.id).unwrap();
        group
            .targets
            .iter()
            .zip(draw)
            .map(|(t, v)| {
                Observation::new(
                    t.id,
                    t.site,
                    observations
                        .iter()
                        .find(|o| o.noise_variance() == 0. && o.site() == &t.site)
                        .map_or(v, Observation::value),
                    0.,
                )
                .unwrap()
            })
            .collect()
    }
    #[test]
    fn changing_draw_shapes_means_and_exact_pins_match_reference_without_allocating() {
        let model = model();
        let data = data();
        let budget = MemoryBudget::new(1 << 24);
        let cancel = Cancellation::default();
        let mut cases = Vec::new();
        for n in [0, 8, 16, 32] {
            for m in 1..=4 {
                let group = group(m);
                let sites: Vec<_> = group.targets.iter().map(|t| t.site).collect();
                let plan = GaussianPlan::prepare(
                    &model,
                    &data[..n],
                    &sites,
                    &Simple,
                    PrepareOptions {
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance::default(),
                        budget: &budget,
                        cancellation: &cancel,
                    },
                )
                .unwrap();
                let expected = [
                    reference(&plan, &data[..n], &group, None),
                    reference(&plan, &data[..n], &group, Some(&trend)),
                ];
                cases.push((n, group, plan, expected));
            }
        }
        let mut workspace = DrawingWorkspace::new(&budget).unwrap();
        workspace.ensure(32, 4, true).unwrap();
        let allocations = allocation_counter::measure(|| {
            for _ in 0..100 {
                for (n, group, plan, expected) in &cases {
                    for (index, mean) in [None, Some(&trend as &dyn KnownMean)]
                        .into_iter()
                        .enumerate()
                    {
                        let actual = workspace
                            .draw(plan, &data[..*n], group, &options(&budget, &cancel, mean))
                            .unwrap();
                        assert_eq!(actual.len(), expected[index].len());
                        assert!(actual
                            .iter()
                            .zip(&expected[index])
                            .all(|(a, b)| a.same_bits(b)));
                        if *n > 0 {
                            assert_eq!(actual[0].value().to_bits(), (-0_f64).to_bits());
                        }
                    }
                }
            }
        });
        assert_eq!(allocations.count_total, 0);
        assert_eq!(allocations.bytes_total, 0);
        drop(cases);
        drop(workspace);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn drawing_failures_release_or_preserve_budget_and_recover() {
        let model = model();
        let data = data();
        let group = group(2);
        let sites: Vec<_> = group.targets.iter().map(|t| t.site).collect();
        let plans = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let plan = GaussianPlan::prepare(
            &model,
            &data,
            &sites,
            &Simple,
            PrepareOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                budget: &plans,
                cancellation: &cancel,
            },
        )
        .unwrap();
        let bytes = 32 * 8 + 4 * (std::mem::size_of::<Observation>() + 32);
        let budget = MemoryBudget::new(bytes as u64);
        let mut workspace = DrawingWorkspace::new(&budget).unwrap();
        workspace.ensure(1, 1, false).unwrap();
        let used = budget.used();
        let hold = budget.reserve(budget.limit() - used).unwrap();
        assert!(workspace
            .draw(
                &plan,
                &data,
                &group,
                &options(&budget, &cancel, Some(&trend))
            )
            .is_err());
        assert_eq!(budget.used(), budget.limit());
        drop(hold);
        assert_eq!(budget.used(), used);
        workspace.ensure(32, 4, true).unwrap();
        assert_eq!(budget.used(), budget.limit());
        assert!(workspace.ensure(usize::MAX, 4, true).is_err());
        assert_eq!(budget.used(), budget.limit());
        let bad = |_: &Site| Ok(f64::NAN);
        assert!(workspace
            .draw(&plan, &data, &group, &options(&budget, &cancel, Some(&bad)))
            .is_err());
        let bad = |_: &Site| Err(invalid("known mean error"));
        assert!(workspace
            .draw(&plan, &data, &group, &options(&budget, &cancel, Some(&bad)))
            .is_err());
        let stopped = Cancellation::default();
        let cancel_mean = |_: &Site| {
            stopped.cancel();
            Ok(0.)
        };
        assert!(matches!(
            workspace.draw(
                &plan,
                &data,
                &group,
                &options(&budget, &stopped, Some(&cancel_mean))
            ),
            Err(Error::Cancelled)
        ));
        assert!(workspace
            .draw(&plan, &data[..1], &group, &options(&budget, &cancel, None))
            .is_err());
        let expected = reference(&plan, &data, &group, Some(&trend));
        let actual = workspace
            .draw(
                &plan,
                &data,
                &group,
                &options(&budget, &cancel, Some(&trend)),
            )
            .unwrap();
        assert!(actual.iter().zip(&expected).all(|(a, b)| a.same_bits(b)));
        drop(workspace);
        assert_eq!(budget.used(), 0);
        drop(plan);
        assert_eq!(plans.used(), 0);
    }
}
