#![cfg(feature = "fft")]
use std::collections::BTreeMap;
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    variography::grid::{
        self as variography,
        spectral::{Accuracy, Method, Workspace},
    },
    Error, Result,
};

struct Data {
    layout: Layout,
    values: Vec<Vec<Option<f64>>>,
    reads: u128,
    fail: bool,
    cancel: Option<Cancellation>,
}
impl Source for Data {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        self.values.len()
    }
    fn read(&mut self, variable: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
        if self.fail {
            return Err(Error::InvalidInput("injected read failure".into()));
        }
        out.copy_from_slice(&self.values[variable][first as usize..first as usize + out.len()]);
        self.reads += out.len() as u128;
        if let Some(cancel) = &self.cancel {
            cancel.cancel();
        }
        Ok(())
    }
}
fn data(shape: [u64; 3], mask: bool) -> Data {
    let layout = Layout::new(shape).unwrap();
    Data {
        layout,
        values: (0..32)
            .map(|v| {
                (0..layout.cells())
                    .map(|i| {
                        (!mask || !(i + v).is_multiple_of(7 + v % 4))
                            .then_some(((i * 123 + v * 321 + i * i * 17) % 97) as f64 - 48.0)
                    })
                    .collect()
            })
            .collect(),
        reads: 0,
        fail: false,
        cancel: None,
    }
}
fn coordinates(layout: Layout, i: usize) -> [i64; 3] {
    let [nx, ny, _] = layout.dimensions();
    [
        (i as u64 % nx) as i64,
        (i as u64 / nx % ny) as i64,
        (i as u64 / nx / ny) as i64,
    ]
}
fn oracle(data: &Data, variables: [usize; 2]) -> BTreeMap<[i64; 3], (u64, f64)> {
    let mut output = BTreeMap::new();
    for i in 0..data.layout.cells() as usize {
        for j in 0..data.layout.cells() as usize {
            let (Some(a), Some(b), Some(c), Some(d)) = (
                data.values[variables[0]][i],
                data.values[variables[0]][j],
                data.values[variables[1]][i],
                data.values[variables[1]][j],
            ) else {
                continue;
            };
            let x = coordinates(data.layout, i);
            let y = coordinates(data.layout, j);
            let offset = std::array::from_fn(|a| y[a] - x[a]);
            let entry = output.entry(offset).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += 0.5 * (a - b) * (c - d);
        }
    }
    output
}

#[test]
fn all_lags_match_independent_pairs_across_masks_variables_tiles_and_workers() -> Result<()> {
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let mut serial = BTreeMap::new();
    let mut fourier = 0;
    for workers in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| -> Result<()> {
            for tile in [1, 17, 128] {
                let mut workspace = Workspace::new(tile, &budget)?;
                for shape in [[5, 4, 3], [1, 7, 1], [17, 1, 1], [2, 2, 2], [1, 1, 1]] {
                    for masked in [false, true] {
                        let mut source = data(shape, masked);
                        for variables in [[0, 0], [0, 1], [1, 0], [31, 7]] {
                            let expected = oracle(&source, variables);
                            let mut map = workspace.compute(
                                &mut source,
                                variables,
                                [2., 3., 4.],
                                Accuracy::default(),
                                &cancel,
                            )?;
                            for (&offset, &(pairs, sum)) in &expected {
                                let answer = map.lag(offset)?;
                                fourier += usize::from(answer.method == Method::Fourier);
                                assert_eq!(answer.statistics.pairs, pairs);
                                let actual = answer.statistics.semivariance.unwrap();
                                let reference = sum / pairs as f64;
                                assert!(
                                    (actual - reference).abs() <= 1e-10 * (1. + reference.abs()),
                                    "{offset:?}: {actual} vs {reference}"
                                );
                                let key = (shape, masked, variables, offset);
                                if workers == 1 && tile == 1 {
                                    serial.insert(key, actual.to_bits());
                                } else {
                                    assert_eq!(serial[&key], actual.to_bits());
                                }
                                assert_eq!(map.pairs(offset.map(|v| -v)), pairs);
                                if offset != [0; 3] {
                                    let distance = ((offset[0] * 2) as f64)
                                        .hypot((offset[1] * 3) as f64)
                                        .hypot((offset[2] * 4) as f64);
                                    assert!(
                                        (answer.statistics.mean_distance.unwrap() - distance).abs()
                                            < 1e-12
                                    );
                                }
                            }
                            for offset in [[i64::MIN, 0, 0], [i64::MAX; 3], [shape[0] as i64, 0, 0]]
                            {
                                let answer = map.lag(offset)?;
                                assert_eq!(answer.statistics.pairs, 0);
                                assert!(answer.statistics.semivariance.is_none());
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    assert!(fourier > 10000, "{fourier}");
    Ok(())
}

fn compare_direct(
    source: &mut Data,
    variables: [usize; 2],
    accuracy: Accuracy,
) -> Result<Vec<(Method, f64)>> {
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let offsets: Vec<_> = (1..source.layout.dimensions()[0] as i64)
        .map(|i| [i, 0, 0])
        .collect();
    let expected = variography::experimental(
        source,
        variography::Options {
            variables,
            offsets: &offsets,
            spacing: [1.; 3],
            tile_cells: 7,
            budget: &budget,
            cancellation: &cancel,
        },
    )?;
    let mut workspace = Workspace::new(3, &budget)?;
    let mut map = workspace.compute(source, variables, [1.; 3], accuracy, &cancel)?;
    let mut methods = vec![];
    for lag in expected.lags {
        let actual = map.lag(lag.offset)?;
        assert_eq!(actual.statistics.pairs, lag.statistics.pairs);
        if lag.statistics.pairs == 0 {
            assert_eq!(actual.statistics.semivariance, None);
            continue;
        }
        let (actual_value, expected_value) = (
            actual.statistics.semivariance.unwrap(),
            lag.statistics.semivariance.unwrap(),
        );
        if actual.method != Method::Fourier {
            assert_eq!(actual_value, expected_value);
        } else {
            assert!(
                (actual_value - expected_value).abs()
                    <= 1e-10 * expected_value.abs().max(f64::MIN_POSITIVE),
                "{actual_value} vs {expected_value}"
            );
        }
        methods.push((actual.method, actual_value));
    }
    Ok(methods)
}
#[test]
fn physical_units_large_offsets_cancellation_and_tiny_values() -> Result<()> {
    for scales in [
        [1., -3.],
        [1e150, -1e-150],
        [1e-150, 1e-150],
        [1e150, 1e150],
    ] {
        let mut source = data([17, 1, 1], false);
        for i in 0..17 {
            source.values[0][i] = Some((i * i) as f64 * scales[0]);
            source.values[1][i] = Some((i * i) as f64 * scales[1]);
        }
        let methods = compare_direct(
            &mut source,
            [0, 1],
            Accuracy {
                absolute: 0.,
                relative: 1e-10,
            },
        )?;
        assert!(methods.iter().any(|(m, _)| *m == Method::Fourier));
    }
    let mut source = data([17, 1, 1], false);
    for i in 0..17 {
        source.values[0][i] = Some(2_f64.powi(52) + i as f64);
        source.values[1][i] = Some(-2_f64.powi(53) + (2 * i) as f64);
    }
    compare_direct(&mut source, [0, 1], Accuracy::default())?;
    // Cancellation at h=1 must trigger refinement with a relative-only screen.
    let mut source = data([3, 1, 1], false);
    source.values[0] = vec![Some(0.), Some(1.), Some(2.)];
    source.values[1] = vec![Some(0.), Some(1.), Some(0.)];
    let result = compare_direct(
        &mut source,
        [0, 1],
        Accuracy {
            absolute: 0.,
            relative: 1e-10,
        },
    )?;
    assert_eq!(result[0], (Method::Direct, 0.));
    // Representable half-product with an unrepresentable raw difference.
    source.layout = Layout::new([2, 1, 1])?;
    source.values[0] = vec![Some(f64::MAX), Some(-f64::MAX)];
    source.values[1] = vec![Some(0.), Some(-0.5)];
    assert_eq!(
        compare_direct(&mut source, [0, 1], Accuracy::default())?[0],
        (Method::Direct, f64::MAX / 2.)
    );
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let mut workspace = Workspace::new(7, &budget)?;
    assert!(workspace
        .compute(&mut source, [0, 0], [1.; 3], Accuracy::default(), &cancel)?
        .lag([1, 0, 0])
        .is_err());
    source.values[0] = vec![Some(0.), Some(f64::from_bits(1))];
    source.values[1] = vec![Some(0.), Some(2.)];
    assert_eq!(
        compare_direct(&mut source, [0, 1], Accuracy::default())?[0],
        (Method::Direct, f64::from_bits(1))
    );
    Ok(())
}

#[test]
fn analytic_masks_constants_and_forced_direct_queries() -> Result<()> {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut workspace = Workspace::new(7, &budget)?;
    let mut source = data([17, 1, 1], false);
    for mask in [Some(f64::MAX), Some(-0.0), None] {
        source.values[0].fill(mask);
        let mut map =
            workspace.compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
        for offset in [[0; 3], [1, 0, 0], [-16, 0, 0]] {
            let answer = map.lag(offset)?;
            assert_eq!(answer.method, Method::Analytic);
            assert_eq!(answer.statistics.semivariance, mask.map(|_| 0.));
        }
    }
    let mut source = data([17, 1, 1], true);
    assert!(compare_direct(
        &mut source,
        [0, 1],
        Accuracy {
            absolute: 0.,
            relative: 0.
        }
    )?
    .iter()
    .all(|(m, _)| *m == Method::Direct));
    Ok(())
}

#[test]
fn source_errors_cancellation_budgets_and_reuse_release_reservations() -> Result<()> {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    assert!(Workspace::new(0, &budget).is_err());
    assert!(Workspace::new(usize::MAX, &budget).is_err());
    {
        let mut workspace = Workspace::new(7, &budget)?;
        let mut source = data([5, 4, 3], true);
        for spacing in [[0.; 3], [f64::INFINITY; 3], [f64::NAN; 3]] {
            assert!(workspace
                .compute(&mut source, [0, 1], spacing, Accuracy::default(), &cancel)
                .is_err());
        }
        for accuracy in [
            Accuracy {
                absolute: -1.,
                relative: 0.,
            },
            Accuracy {
                absolute: 0.,
                relative: f64::NAN,
            },
        ] {
            assert!(workspace
                .compute(&mut source, [0, 1], [1.; 3], accuracy, &cancel)
                .is_err());
        }
        assert!(workspace
            .compute(&mut source, [0, 32], [1.; 3], Accuracy::default(), &cancel)
            .is_err());
        source.fail = true;
        assert!(workspace
            .compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)
            .is_err());
        source.fail = false;
        source.values[0][0] = Some(f64::NAN);
        assert!(workspace
            .compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)
            .is_err());
        source.values[0][0] = Some(2.);
        let cancelled = Cancellation::default();
        source.cancel = Some(cancelled.clone());
        assert!(matches!(
            workspace.compute(
                &mut source,
                [0, 1],
                [1.; 3],
                Accuracy::default(),
                &cancelled
            ),
            Err(Error::Cancelled)
        ));
        source.cancel = None;
        let mut map =
            workspace.compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
        assert!(map.lag([1, 0, 0])?.statistics.pairs > 0);
        cancel.cancel();
        assert!(matches!(map.lag([1, 0, 0]), Err(Error::Cancelled)));
    }
    assert_eq!(budget.used(), 0);
    let budget = MemoryBudget::new(7 * 6 * 16);
    let mut workspace = Workspace::new(7, &budget)?;
    let mut source = data([5, 4, 3], false);
    assert!(matches!(
        workspace.compute(
            &mut source,
            [0, 1],
            [1.; 3],
            Accuracy::default(),
            &Cancellation::default()
        ),
        Err(Error::MemoryBudget { .. })
    ));
    source.values[0].fill(Some(1.));
    assert_eq!(
        workspace
            .compute(
                &mut source,
                [0, 1],
                [1.; 3],
                Accuracy::default(),
                &Cancellation::default()
            )?
            .lag([1, 0, 0])?
            .method,
        Method::Analytic
    );
    drop(workspace);
    assert_eq!(budget.used(), 0);
    Ok(())
}

#[test]
fn file_source_matches_resident_and_warm_serial_execution_allocates_nothing() -> Result<()> {
    let budget = MemoryBudget::new(8 << 20);
    let cancel = Cancellation::default();
    let mut source = data([5, 4, 3], true);
    let temporary = tempfile::tempfile()?;
    terrustrial_core::storage::grid::write(&mut source, &temporary, 7, &budget, &cancel)?;
    let mut disk = terrustrial_core::storage::grid::FileSource::open(temporary, &budget)?;
    let mut a = Workspace::new(7, &budget)?;
    let mut b = Workspace::new(13, &budget)?;
    let mut resident = a.compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    let mut file = b.compute(&mut disk, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    for offset in [[0; 3], [1, 0, 0], [-4, -3, -2], [2, 1, -1]] {
        assert_eq!(
            resident.lag(offset)?.statistics.semivariance,
            file.lag(offset)?.statistics.semivariance
        );
        assert_eq!(
            resident.lag_direct(offset)?.statistics.semivariance,
            file.lag_direct(offset)?.statistics.semivariance
        );
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        a.compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)
            .unwrap();
        let allocation = allocation_counter::measure(|| {
            let mut map = a
                .compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)
                .unwrap();
            for z in -2..=2 {
                for y in -3..=3 {
                    for x in -4..=4 {
                        map.lag([x, y, z]).unwrap();
                        map.lag_direct([x, y, z]).unwrap();
                    }
                }
            }
        });
        assert_eq!(allocation.count_total, 0, "{allocation:?}");
    });
    Ok(())
}
