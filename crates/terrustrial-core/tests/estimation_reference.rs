//! Independent augmented-system reference for complete estimation drivers.
//! Scalar elimination here deliberately does not call the production faer solver.
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    estimation::{estimate_groups, EstimationOptions},
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{ExternalDrift, NumericalTolerance, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    spatial::{Neighborhood, ObservationSource, SpatialIndex},
};

fn site(x: f64, variable: usize, block: bool) -> Site {
    let center = Point3::new([x, 0., 0.]).unwrap();
    Site {
        variable,
        support: if block {
            Support::block(center, [0.12, 0., 0.], [3, 1, 1]).unwrap()
        } else {
            Support::point(center)
        },
    }
}
// Construct midpoint coordinates independently of Support::nodes and covariance assembly.
fn nodes(s: &Site) -> Vec<f64> {
    let n = s.support.divisions()[0];
    let x = s.support.center().coordinates()[0];
    let h = s.support.half_extents()[0];
    (0..n)
        .map(|i| x - h + (i as f64 + 0.5) * 2. * h / n as f64)
        .collect()
}
fn mean(s: &Site) -> f64 {
    let nodes = nodes(s);
    nodes
        .iter()
        .map(|x| 0.13 * x * x + s.variable as f64 * 0.07)
        .sum::<f64>()
        / nodes.len() as f64
}
fn covariance(a: &Site, b: &Site) -> f64 {
    let aa = nodes(a);
    let bb = nodes(b);
    let sill = if a.variable == b.variable { 1. } else { 0.2 };
    sill * aa
        .iter()
        .flat_map(|a| bb.iter().map(move |b| (-(a - b).abs() / 2.).exp()))
        .sum::<f64>()
        / (aa.len() * bb.len()) as f64
}
fn basis(s: &Site, variables: usize, degree: usize) -> Vec<f64> {
    let mut out = vec![0.; variables * degree];
    if degree > 0 {
        out[s.variable * degree] = 1.;
    }
    if degree > 1 {
        out[s.variable * degree + 1] = s.support.center().coordinates()[0];
    }
    out
}
// Indexed loops keep the independent elimination formula directly auditable.
#[allow(clippy::needless_range_loop)]
fn solve(mut a: Vec<Vec<f64>>, mut rhs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = rhs[0].len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        assert!(a[pivot][k].abs() > 1e-12, "singular independent fixture");
        a.swap(k, pivot);
        rhs.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k + 1..n {
                a[i][j] -= factor * a[k][j];
            }
            for j in 0..m {
                rhs[i][j] -= factor * rhs[k][j];
            }
        }
    }
    for i in (0..n).rev() {
        for j in 0..m {
            for k in i + 1..n {
                rhs[i][j] -= a[i][k] * rhs[k][j];
            }
            rhs[i][j] /= a[i][i];
        }
    }
    rhs
}
fn reference(
    hard: &[Observation],
    group: &Group,
    variables: usize,
    degree: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = hard.len();
    let m = group.targets.len();
    let p = variables * degree;
    let c: Vec<Vec<_>> = hard
        .iter()
        .enumerate()
        .map(|(i, a)| {
            hard.iter()
                .enumerate()
                .map(|(j, b)| {
                    covariance(a.site(), b.site()) + if i == j { a.noise_variance() } else { 0. }
                })
                .collect()
        })
        .collect();
    let cross: Vec<Vec<_>> = hard
        .iter()
        .map(|a| {
            group
                .targets
                .iter()
                .map(|b| covariance(a.site(), &b.site))
                .collect()
        })
        .collect();
    let mut a = vec![vec![0.; n + p]; n + p];
    let mut rhs = vec![vec![0.; m]; n + p];
    for i in 0..n {
        a[i][..n].copy_from_slice(&c[i]);
        rhs[i].copy_from_slice(&cross[i]);
        for (j, f) in basis(hard[i].site(), variables, degree)
            .into_iter()
            .enumerate()
        {
            a[i][n + j] = f;
            a[n + j][i] = f;
        }
    }
    for (j, t) in group.targets.iter().enumerate() {
        for (i, f) in basis(&t.site, variables, degree).into_iter().enumerate() {
            rhs[n + i][j] = f;
        }
    }
    let weights = solve(a, rhs);
    let values = (0..m)
        .map(|j| {
            mean(&group.targets[j].site)
                + (0..n)
                    .map(|i| weights[i][j] * (hard[i].value() - mean(hard[i].site())))
                    .sum::<f64>()
        })
        .collect();
    // Error covariance Ctt - W'K - K'W + W'C W, independent of production Schur form.
    let cw: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..m)
                .map(|j| (0..n).map(|k| c[i][k] * weights[k][j]).sum())
                .collect()
        })
        .collect();
    let covariance = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| {
                    covariance(&group.targets[i].site, &group.targets[j].site)
                        + (0..n)
                            .map(|k| {
                                weights[k][i] * cw[k][j]
                                    - weights[k][i] * cross[k][j]
                                    - cross[k][i] * weights[k][j]
                            })
                            .sum::<f64>()
                })
                .collect()
        })
        .collect();
    (values, covariance)
}

fn check<D: terrustrial_core::gaussian::Drift>(
    model: &Lmc,
    records: &[Observation],
    groups: &[Group],
    sources: &[&(dyn ObservationSource + Sync)],
    variables: usize,
    degree: usize,
    drift: &D,
) {
    let reference: Vec<_> = groups
        .iter()
        .map(|g| reference(records, g, variables, degree))
        .collect();
    let budget = MemoryBudget::new(256 << 20);
    let cancel = Cancellation::default();
    let mut bits = None;
    let known_mean = |s: &Site| Ok(mean(s));
    for workers in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for source in sources {
            let mut actual = Vec::new();
            let mut emitted = 0;
            pool.install(|| estimate_groups(model,*source,groups.iter().cloned(),drift,EstimationOptions {
                neighborhood: Neighborhood { maximum: records.len(), radius: 100. }.into(),
                mean: Some(&known_mean),
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                budget: &budget, cancellation: &cancel,
            },ExecutionOptions {batch_groups:8,maximum_targets_per_group:variables,bytes_per_group:4<<20},|out| {
                let (values,covariance) = &reference[emitted];
                assert_eq!(out.group.id,emitted as u64);
                assert_eq!(out.values.len(),variables);
                for (a,e) in out.values.iter().zip(values) {
                    assert!((a-e).abs()<=1e-10*(1.+e.abs()),"prediction: variables={variables} degree={degree} actual={a} expected={e}");
                    actual.push(a.to_bits());
                }
                let c=out.covariance().unwrap();
                for j in 0..variables { for i in 0..variables {
                    let e=covariance[i][j]; let a=c[(i,j)];
                    assert!((a-e).abs()<=1e-10*(1.+e.abs()),"covariance: variables={variables} degree={degree} actual={a} expected={e}");
                    actual.push(a.to_bits());
                }}
                emitted+=1; Ok(())
            })).unwrap();
            assert_eq!(emitted, groups.len());
            assert_eq!(budget.used(), 0);
            if let Some(bits) = &bits {
                assert_eq!(&actual, bits, "source/worker reproducibility");
            } else {
                bits = Some(actual);
            }
        }
    }
}

#[test]
fn joint_estimation_matches_independent_augmented_system_across_sources_and_workers() {
    use terrustrial_core::gaussian::{Ordinary, Simple};
    for variables in [1, 2, 8, 32] {
        let factor: Vec<_> = (0..variables)
            .flat_map(|v| {
                (0..=variables).map(move |j| {
                    if j == 0 {
                        0.2_f64.sqrt()
                    } else if j == v + 1 {
                        0.8_f64.sqrt()
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(2.).unwrap(),
            coregionalization: Coregionalization::from_factor(variables, variables + 1, &factor)
                .unwrap(),
        }])
        .unwrap();
        for block in [false, true] {
            let records: Vec<_> = (0..variables * 3)
                .map(|i| {
                    let s = site(
                        (i / variables) as f64 * 1.7 + (i % variables) as f64 * 0.017,
                        i % variables,
                        block && i % 2 == 0,
                    );
                    Observation::new(i as u64, s, mean(&s) + (i as f64 * 0.7).sin(), 0.03).unwrap()
                })
                .collect();
            let groups: Vec<_> = (0..9)
                .map(|g| Group {
                    id: g,
                    targets: (0..variables)
                        .map(|v| Target {
                            id: u64::MAX - g * variables as u64 - v as u64,
                            site: site(0.21 + g as f64 * 0.23 + v as f64 * 0.003, v, block),
                        })
                        .collect(),
                })
                .collect();
            let resident = SpatialIndex::new(records.clone()).unwrap();
            #[cfg(feature = "sqlite")]
            let storage_budget = MemoryBudget::new(32 << 20);
            #[cfg(feature = "sqlite")]
            let directory = tempfile::tempdir().unwrap();
            #[cfg(feature = "sqlite")]
            let store = {
                use terrustrial_core::storage::packed::{Options, PackedBuilder};
                let mut builder = PackedBuilder::create(
                    directory.path().join("reference.sqlite"),
                    Options {
                        leaf_records: 7,
                        cached_pages: 2,
                        readers: 8,
                        directory_pages: 2,
                        sqlite_cache_bytes: 4096,
                        ..Options::default()
                    },
                    &storage_budget,
                )
                .unwrap();
                for o in records.iter().rev() {
                    builder.push(*o, 0, &Cancellation::default()).unwrap();
                }
                builder.finish(&Cancellation::default()).unwrap()
            };
            #[cfg(feature = "sqlite")]
            let snapshot = store.snapshot().unwrap();
            let sources: Vec<&(dyn ObservationSource + Sync)> = vec![
                &resident,
                #[cfg(feature = "sqlite")]
                &snapshot,
            ];
            check(&model, &records, &groups, &sources, variables, 0, &Simple);
            check(
                &model,
                &records,
                &groups,
                &sources,
                variables,
                1,
                &Ordinary { variables },
            );
            let external = ExternalDrift::new(2 * variables, |s: &Site, values: &mut [f64]| {
                values.fill(0.);
                values[2 * s.variable] = 1.;
                values[2 * s.variable + 1] = s.support.center().coordinates()[0];
                Ok(())
            })
            .unwrap();
            check(&model, &records, &groups, &sources, variables, 2, &external);
        }
    }
}
