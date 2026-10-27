use std::io::{self, Read};
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    execution::{Cancellation, MemoryBudget},
    gaussian::{LinearDrift, Ordinary, Simple},
    identity::{self, CanonicalFingerprint},
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    simulation::{identity::RunProvenance, SimulationOptions},
    spatial::{Neighborhood, ObservationSource, SearchNeighborhood, VariableLimit},
    Error,
};
fn point(x: f64) -> Site {
    Site::point(Point3::new([x, 0., 0.]).unwrap(), 0)
}
fn obs(id: u64) -> Observation {
    Observation::new(id, point(id as f64), -0., 0.125).unwrap()
}
fn model(range: f64, correlation: Correlation) -> Lmc {
    Lmc::new(vec![Structure {
        correlation,
        anisotropy: Anisotropy::isotropic(range).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.5, 1.]).unwrap(),
    }])
    .unwrap()
}

#[test]
fn canonical_observation_bytes_match_an_independent_layout_and_preserve_every_field() {
    let site = Site {
        support: Support::block(
            Point3::new([1., -0., 3.]).unwrap(),
            [0.25, 0.5, 0.],
            [2, 3, 1],
        )
        .unwrap(),
        variable: 7,
    };
    let observation = Observation::new(u64::MAX, site, -2., 0.125).unwrap();
    // Explicit wire record: domain=2, record marker, ID, support, variable,
    // value/noise, end marker, count. No production encoder is used here.
    let mut bytes = vec![2, 1];
    for value in [
        u64::MAX,
        1_f64.to_bits(),
        (-0.0_f64).to_bits(),
        3_f64.to_bits(),
        0.25_f64.to_bits(),
        0.5_f64.to_bits(),
        0,
        2,
        3,
        1,
        7,
        (-2_f64).to_bits(),
        0.125_f64.to_bits(),
    ] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.push(0);
    bytes.extend(1_u64.to_le_bytes());
    let mut hasher = blake3::Hasher::new_derive_key("terrustrial canonical identity v1");
    hasher.update(&bytes);
    let cancel = Cancellation::default();
    let actual = identity::observations(&[observation][..], &cancel).unwrap();
    assert_eq!(actual.to_bytes(), *hasher.finalize().as_bytes());
    for variant in [
        Observation::new(1, site, -2., 0.125).unwrap(),
        Observation::new(
            u64::MAX,
            Site {
                variable: 8,
                ..site
            },
            -2.,
            0.125,
        )
        .unwrap(),
        Observation::new(
            u64::MAX,
            Site {
                support: Support::block(
                    Point3::new([1., 0., 3.]).unwrap(),
                    [0.25, 0.5, 0.],
                    [2, 3, 1],
                )
                .unwrap(),
                ..site
            },
            -2.,
            0.125,
        )
        .unwrap(),
        Observation::new(
            u64::MAX,
            Site {
                support: Support::block(
                    Point3::new([1., -0., 3.]).unwrap(),
                    [0.5, 0.5, 0.],
                    [2, 3, 1],
                )
                .unwrap(),
                ..site
            },
            -2.,
            0.125,
        )
        .unwrap(),
        Observation::new(
            u64::MAX,
            Site {
                support: Support::block(
                    Point3::new([1., -0., 3.]).unwrap(),
                    [0.25, 0.5, 0.],
                    [3, 3, 1],
                )
                .unwrap(),
                ..site
            },
            -2.,
            0.125,
        )
        .unwrap(),
        Observation::new(u64::MAX, site, -3., 0.125).unwrap(),
        Observation::new(u64::MAX, site, -2., 0.25).unwrap(),
    ] {
        assert_ne!(
            actual,
            identity::observations(&[variant][..], &cancel).unwrap()
        );
    }
    assert_ne!(
        identity::observations(&[obs(1), obs(2)][..], &cancel).unwrap(),
        identity::observations(&[obs(2), obs(1)][..], &cancel).unwrap()
    );
    assert_ne!(
        identity::observations(&[obs(1)][..], &cancel).unwrap(),
        identity::observations(&[obs(1), obs(1)][..], &cancel).unwrap()
    );
    assert_ne!(
        identity::opaque("a", b"bc").unwrap(),
        identity::opaque("ab", b"c").unwrap()
    );
    assert!(identity::opaque("", b"definition").is_err());
    assert!(identity::opaque("namespace", b"").is_err());
}

struct Chunks<'a> {
    bytes: &'a [u8],
    chunk: usize,
    interrupt: bool,
}
impl Read for Chunks<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.interrupt {
            self.interrupt = false;
            return Err(io::ErrorKind::Interrupted.into());
        }
        let n = out.len().min(self.chunk).min(self.bytes.len());
        out[..n].copy_from_slice(&self.bytes[..n]);
        self.bytes = &self.bytes[n..];
        Ok(n)
    }
}
#[test]
fn byte_stream_chunking_buffer_boundaries_and_failures_are_explicit() {
    let cancel = Cancellation::default();
    for n in [0, 1, 4095, 4096, 4097, 16384, 65537] {
        let data: Vec<_> = (0..n).map(|i| (i % 251) as u8).collect();
        let mut reference = blake3::Hasher::new_derive_key("terrustrial canonical identity v1");
        reference.update(&[1]);
        reference.update(&data);
        reference.update(&(n as u64).to_le_bytes());
        for chunk in [1, 7, 4096, 32768] {
            let actual = identity::reader(
                Chunks {
                    bytes: &data,
                    chunk,
                    interrupt: true,
                },
                &cancel,
            )
            .unwrap();
            assert_eq!(actual.to_bytes(), *reference.finalize().as_bytes());
        }
    }
    struct Fails;
    impl Read for Fails {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::ErrorKind::Other.into())
        }
    }
    assert!(matches!(
        identity::reader(Fails, &cancel),
        Err(Error::Io(_))
    ));
    cancel.cancel();
    assert!(matches!(
        identity::reader(&b"data"[..], &cancel),
        Err(Error::Cancelled)
    ));
}

struct Generated {
    count: u64,
    fail: bool,
}
impl ObservationSource for Generated {
    fn scan(
        &self,
        visitor: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
    ) -> terrustrial_core::Result<()> {
        for id in 0..self.count {
            visitor(&obs(id))?;
        }
        if self.fail {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}
#[test]
fn streaming_observation_hashing_allocates_nothing_and_propagates_late_failure() {
    let cancel = Cancellation::default();
    identity::observations(
        &Generated {
            count: 1,
            fail: false,
        },
        &cancel,
    )
    .unwrap();
    let measure = allocation_counter::measure(|| {
        identity::observations(
            &Generated {
                count: 100_000,
                fail: false,
            },
            &cancel,
        )
        .unwrap();
    });
    assert_eq!(measure.count_total, 0);
    assert!(matches!(
        identity::observations(
            &Generated {
                count: 32,
                fail: true
            },
            &cancel
        ),
        Err(Error::Cancelled)
    ));
    cancel.cancel();
    assert!(matches!(
        identity::observations(
            &Generated {
                count: 1,
                fail: false
            },
            &cancel
        ),
        Err(Error::Cancelled)
    ));
}

#[test]
fn paths_include_group_boundaries_target_order_ids_and_support() {
    let cancel = Cancellation::default();
    let groups = vec![Group {
        id: 0,
        targets: vec![
            Target {
                id: 10,
                site: point(0.),
            },
            Target {
                id: 11,
                site: point(1.),
            },
        ],
    }];
    let original = identity::path(groups.clone(), &cancel).unwrap();
    let mut variants = Vec::new();
    let mut v = groups.clone();
    v[0].id = 1;
    variants.push(v);
    let mut v = groups.clone();
    v[0].targets.reverse();
    variants.push(v);
    let mut v = groups.clone();
    v[0].targets[0].id = 12;
    variants.push(v);
    let mut v = groups.clone();
    v[0].targets[0].site = point(-0.);
    variants.push(v);
    variants.push(vec![
        Group {
            id: 0,
            targets: vec![groups[0].targets[0].clone()],
        },
        Group {
            id: 1,
            targets: vec![groups[0].targets[1].clone()],
        },
    ]);
    for variant in variants {
        assert_ne!(original, identity::path(variant, &cancel).unwrap());
    }
    assert!(identity::path(
        vec![Group {
            id: 0,
            targets: vec![]
        }],
        &cancel
    )
    .is_err());
    assert!(identity::path(vec![groups[0].clone(), groups[0].clone()], &cancel).is_err());
    assert_ne!(
        identity::path(Vec::new(), &cancel).unwrap(),
        identity::observations(&[][..], &cancel).unwrap()
    );
}

#[test]
fn model_drift_search_and_run_contract_changes_change_identity() {
    let base = model(1., Correlation::Exponential);
    assert_ne!(
        base.fingerprint().unwrap(),
        model(2., Correlation::Exponential).fingerprint().unwrap()
    );
    assert_ne!(
        base.fingerprint().unwrap(),
        model(1., Correlation::Gaussian).fingerprint().unwrap()
    );
    let mut structures = base.structures().to_vec();
    structures[0].coregionalization =
        Coregionalization::from_factor(2, 2, &[1., 0., 0.75, 1.]).unwrap();
    assert_ne!(
        base.fingerprint().unwrap(),
        Lmc::new(structures).unwrap().fingerprint().unwrap()
    );
    assert_ne!(
        Simple.fingerprint().unwrap(),
        Ordinary { variables: 2 }.fingerprint().unwrap()
    );
    let linear = LinearDrift::new(2, [0.; 3], 1.).unwrap();
    for d in [
        LinearDrift::new(1, [0.; 3], 1.).unwrap(),
        LinearDrift::new(2, [1., 0., 0.], 1.).unwrap(),
        LinearDrift::new(2, [0.; 3], 2.).unwrap(),
    ] {
        assert_ne!(linear.fingerprint().unwrap(), d.fingerprint().unwrap());
    }
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut options = SimulationOptions {
        seed: 1,
        realization: 2,
        mean: None,
        hard_neighborhood: Neighborhood {
            maximum: 8,
            radius: 3.,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 16,
            radius: 4.,
        }
        .into(),
        budget: &budget,
        cancellation: &cancel,
    };
    let input = identity::observations(&[][..], &cancel).unwrap();
    let path = identity::path(Vec::new(), &cancel).unwrap();
    let provenance = RunProvenance {
        implementation: identity::opaque("impl", b"v1").unwrap(),
        backend: identity::opaque("backend", b"cpu-f64").unwrap(),
        mean: None,
    };
    let hash = |options: &SimulationOptions<'_>, provenance| {
        terrustrial_core::simulation::identity::gaussian_run(
            &base, &Simple, input, path, options, provenance,
        )
        .unwrap()
    };
    let original = hash(&options, provenance);
    options.seed = 2;
    assert_ne!(original, hash(&options, provenance));
    options.seed = 1;
    options.realization = 3;
    assert_ne!(original, hash(&options, provenance));
    options.realization = 2;
    for query in [
        Neighborhood {
            maximum: 7,
            radius: 3.,
        },
        Neighborhood {
            maximum: 8,
            radius: 3.5,
        },
    ] {
        options.hard_neighborhood = query.into();
        assert_ne!(original, hash(&options, provenance));
    }
    options.hard_neighborhood = Neighborhood {
        maximum: 8,
        radius: 3.,
    }
    .into();
    options.simulated_neighborhood = Neighborhood {
        maximum: 15,
        radius: 4.,
    }
    .into();
    assert_ne!(original, hash(&options, provenance));
    options.simulated_neighborhood = Neighborhood {
        maximum: 16,
        radius: 4.,
    }
    .into();
    for p in [
        RunProvenance {
            implementation: identity::opaque("impl", b"v2").unwrap(),
            ..provenance
        },
        RunProvenance {
            backend: identity::opaque("backend", b"gpu-f32").unwrap(),
            ..provenance
        },
    ] {
        assert_ne!(original, hash(&options, p));
    }
    let mean = |_: &Site| Ok(0.);
    options.mean = Some(&mean);
    assert!(terrustrial_core::simulation::identity::gaussian_run(
        &base, &Simple, input, path, &options, provenance
    )
    .is_err());
    let mean_provenance = RunProvenance {
        mean: Some(identity::opaque("mean", b"zero").unwrap()),
        ..provenance
    };
    assert_ne!(original, hash(&options, mean_provenance));
    options.mean = None;
    assert!(terrustrial_core::simulation::identity::gaussian_run(
        &base,
        &Simple,
        input,
        path,
        &options,
        mean_provenance
    )
    .is_err());
    let constrained = SearchNeighborhood::new(
        8,
        Anisotropy::isotropic(3.).unwrap(),
        Some(2),
        vec![VariableLimit {
            variable: 0,
            maximum: 4,
        }],
    )
    .unwrap();
    options.hard_neighborhood =
        terrustrial_core::spatial::NeighborhoodQuery::Constrained(&constrained);
    let c = hash(&options, provenance);
    assert_ne!(original, c);
    for n in [
        SearchNeighborhood::new(
            8,
            Anisotropy::isotropic(3.).unwrap(),
            Some(3),
            vec![VariableLimit {
                variable: 0,
                maximum: 4,
            }],
        )
        .unwrap(),
        SearchNeighborhood::new(
            8,
            Anisotropy::isotropic(3.).unwrap(),
            Some(2),
            vec![VariableLimit {
                variable: 1,
                maximum: 4,
            }],
        )
        .unwrap(),
        SearchNeighborhood::new(
            8,
            Anisotropy::isotropic(3.).unwrap(),
            Some(2),
            vec![VariableLimit {
                variable: 0,
                maximum: 3,
            }],
        )
        .unwrap(),
    ] {
        let changed = SimulationOptions {
            hard_neighborhood: terrustrial_core::spatial::NeighborhoodQuery::Constrained(&n),
            ..options
        };
        assert_ne!(c, hash(&changed, provenance));
    }
}

#[test]
fn opaque_adapter_preserves_all_covariance_overrides() {
    struct Custom;
    impl Covariance for Custom {
        fn variables(&self) -> usize {
            1
        }
        fn point_covariance(
            &self,
            _: Point3,
            _: usize,
            _: Point3,
            _: usize,
        ) -> terrustrial_core::Result<f64> {
            Ok(1.)
        }
        fn covariance(&self, _: &Site, _: &Site) -> terrustrial_core::Result<f64> {
            Ok(2.)
        }
        fn covariance_row(
            &self,
            _: &Site,
            _: &[Observation],
            out: &mut [f64],
        ) -> terrustrial_core::Result<()> {
            out.fill(3.);
            Ok(())
        }
        fn covariance_column(
            &self,
            _: &[Observation],
            _: &Site,
            out: &mut [f64],
        ) -> terrustrial_core::Result<()> {
            out.fill(4.);
            Ok(())
        }
    }
    let model = identity::Identified::new(Custom, "custom model", b"all definitions v1").unwrap();
    assert_eq!(model.variables(), 1);
    assert_eq!(
        model
            .point_covariance(
                Point3::new([0.; 3]).unwrap(),
                0,
                Point3::new([0.; 3]).unwrap(),
                0
            )
            .unwrap(),
        1.
    );
    assert_eq!(model.covariance(&point(0.), &point(1.)).unwrap(), 2.);
    let mut output = [0.; 1];
    model
        .covariance_row(&point(0.), &[obs(1)], &mut output)
        .unwrap();
    assert_eq!(output, [3.]);
    model
        .covariance_column(&[obs(1)], &point(0.), &mut output)
        .unwrap();
    assert_eq!(output, [4.]);
}

#[test]
fn quota_run_identity_preserves_old_queries_and_canonicalizes_variable_order() {
    use terrustrial_core::spatial::{MinimumQuotas, NeighborhoodQuery, VariableMinimum};
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let model = model(3., Correlation::Exponential);
    let input = identity::observations(&[][..], &cancel).unwrap();
    let path = identity::path(Vec::new(), &cancel).unwrap();
    let provenance = RunProvenance {
        implementation: identity::opaque("impl", b"quota-test").unwrap(),
        backend: identity::opaque("backend", b"cpu").unwrap(),
        mean: None,
    };
    let base = SearchNeighborhood::new(
        8,
        Anisotropy::isotropic(3.).unwrap(),
        None,
        vec![
            VariableLimit {
                variable: 0,
                maximum: 4,
            },
            VariableLimit {
                variable: usize::MAX,
                maximum: 4,
            },
        ],
    )
    .unwrap();
    let hash = |query: &SearchNeighborhood| {
        let options = SimulationOptions {
            seed: 1,
            realization: 0,
            mean: None,
            hard_neighborhood: NeighborhoodQuery::Constrained(query),
            simulated_neighborhood: Neighborhood {
                maximum: 8,
                radius: 3.,
            }
            .into(),
            budget: &budget,
            cancellation: &cancel,
        };
        terrustrial_core::simulation::identity::gaussian_run(
            &model, &Simple, input, path, &options, provenance,
        )
        .unwrap()
    };
    let original = hash(&base);
    assert_eq!(
        original,
        hash(
            &base
                .clone()
                .with_minimums(MinimumQuotas::default())
                .unwrap()
        )
    );
    let minimums = MinimumQuotas {
        total: 2,
        variables: vec![
            VariableMinimum {
                variable: 0,
                minimum: 1,
            },
            VariableMinimum {
                variable: usize::MAX,
                minimum: 1,
            },
        ],
        octants: [1, 1, 0, 0, 0, 0, 0, 0],
    };
    let configured = hash(&base.clone().with_minimums(minimums.clone()).unwrap());
    assert_ne!(configured, original);
    let mut reversed = minimums.clone();
    reversed.variables.reverse();
    assert_eq!(
        configured,
        hash(&base.clone().with_minimums(reversed).unwrap())
    );
    for field in 0..11 {
        let mut changed = minimums.clone();
        match field {
            0 => changed.total += 1,
            1 | 2 => changed.variables[field - 1].minimum += 1,
            _ => changed.octants[field - 3] += 1,
        }
        assert_ne!(
            configured,
            hash(&base.clone().with_minimums(changed).unwrap())
        );
    }
    let cleared = base
        .clone()
        .with_minimums(minimums)
        .unwrap()
        .with_minimums(MinimumQuotas::default())
        .unwrap();
    assert_eq!(original, hash(&cleared));
    let fixed = base
        .clone()
        .with_minimums(MinimumQuotas {
            total: 2,
            ..Default::default()
        })
        .unwrap();
    let expanded = fixed
        .clone()
        .with_expansion(vec![[4., 3., 3.], [4., 5., 3.]])
        .unwrap();
    assert_ne!(hash(&fixed), hash(&expanded));
    for stages in [
        vec![[4., 5., 3.]],
        vec![[4., 4., 3.], [4., 5., 3.]],
        vec![[4., 3., 3.], [4., 6., 3.]],
    ] {
        assert_ne!(
            hash(&expanded),
            hash(&fixed.clone().with_expansion(stages).unwrap())
        );
    }
    assert_eq!(
        hash(&fixed),
        hash(&expanded.with_expansion(vec![]).unwrap())
    );
    let zero = MinimumQuotas {
        variables: vec![VariableMinimum {
            variable: 0,
            minimum: 0,
        }],
        ..Default::default()
    };
    assert_eq!(original, hash(&base.with_minimums(zero).unwrap()));
}
