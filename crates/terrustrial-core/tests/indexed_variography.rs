use terrustrial_core::{
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    variography::{
        self,
        indexed::{SampleIndex, Workspace},
        Bin, Direction, Sample,
    },
    Error,
};

fn sample(point: [f64; 3], a: Option<f64>, b: Option<f64>) -> Sample {
    Sample {
        point: Point3::new(point).unwrap(),
        values: vec![a, b],
    }
}
fn fixture(n: usize) -> Vec<Sample> {
    let mut state = 77_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    (0..n)
        .map(|i| {
            let xyz = std::array::from_fn(|_| (next() % 41) as f64 * 0.25);
            let value = (next() % 17) as f64 - 8.;
            sample(
                xyz,
                (i % 11 != 0).then_some(value),
                (i % 13 != 0).then_some(-2. * value),
            )
        })
        .collect()
}
fn close(actual: &[Bin], expected: &[Bin]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.pairs, b.pairs);
        for (a, b) in [
            (a.mean_distance, b.mean_distance),
            (a.semivariance, b.semivariance),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert!((a - b).abs() <= 2e-12 * (1. + b.abs()), "{a} != {b}")
                }
                (None, None) => (),
                _ => panic!("empty-bin disagreement"),
            }
        }
    }
}
fn bits(bins: &[Bin]) -> Vec<(u64, Option<u64>, Option<u64>)> {
    bins.iter()
        .map(|b| {
            (
                b.pairs,
                b.mean_distance.map(f64::to_bits),
                b.semivariance.map(f64::to_bits),
            )
        })
        .collect()
}

#[test]
fn indexed_pairs_match_enumeration_with_missingness_directions_and_boundaries() {
    let mut data = fixture(1200);
    // Duplicates, exact lag boundaries, and immediately neighboring floats.
    for x in [0., 0., 0.5_f64.next_down(), 0.5, 0.5_f64.next_up(), 1., 2.] {
        data.push(sample([x, 0., 0.], Some(x), Some(-x)));
    }
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 24);
    let index = SampleIndex::new(&data, &budget, &cancel).unwrap();
    let mut workspace = Workspace::new(5, 7, &budget).unwrap();
    let axis = Direction::new([1., 0., 0.], 0., 0.5).unwrap();
    let oblique = Direction::new([1., -2., 3.], 0.4, 1.5).unwrap();
    let full = Direction::new([1., 0., 0.], std::f64::consts::FRAC_PI_2, 100.).unwrap();
    let edges = [0.5, 1., 2., 4., 7.];
    for direction in [None, Some(&axis), Some(&oblique), Some(&full)] {
        for variables in [[0, 0], [0, 1], [1, 1]] {
            let reference = variography::experimental(
                &data,
                variables[0],
                variables[1],
                &edges,
                direction,
                &cancel,
            )
            .unwrap();
            let result = index
                .experimental(variables, &edges, direction, &mut workspace, &cancel)
                .unwrap();
            close(result.bins, &reference);
            assert_eq!(
                result.population_pairs,
                data.len() as u128 * (data.len() - 1) as u128 / 2
            );
            assert!(result.candidate_pairs <= result.population_pairs);
            assert!(result.candidate_pairs >= result.bins.iter().map(|b| b.pairs as u128).sum());
        }
    }
}

#[test]
fn workers_windows_and_warm_reuse_preserve_bits_without_serial_allocations() {
    let data = fixture(2100);
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 24);
    let index = SampleIndex::new(&data, &budget, &cancel).unwrap();
    let edges = [0.5, 1., 2., 3.];
    let mut serial = Workspace::new(4, 1, &budget).unwrap();
    let expected = bits(
        index
            .experimental([0, 1], &edges, None, &mut serial, &cancel)
            .unwrap()
            .bins,
    );
    let allocations = allocation_counter::measure(|| {
        for _ in 0..3 {
            let result = index
                .experimental([0, 1], &edges, None, &mut serial, &cancel)
                .unwrap();
            assert_eq!(result.bins[0].pairs, expected[0].0);
        }
    });
    assert_eq!(allocations.count_total, 0, "{allocations:?}");
    for workers in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for window in [1, 2, 7, 32] {
            let mut workspace = Workspace::new(8, window, &budget).unwrap();
            pool.install(|| {
                for _ in 0..2 {
                    let actual = index
                        .experimental([0, 1], &edges, None, &mut workspace, &cancel)
                        .unwrap();
                    assert_eq!(bits(actual.bins), expected);
                    // Change active shape/attributes, then reuse the larger workspace.
                    index
                        .experimental([1, 1], &[2.], None, &mut workspace, &cancel)
                        .unwrap();
                }
            });
        }
    }
}

#[test]
fn finite_extremes_coincidences_and_signed_cross_products() {
    let max = f64::MAX;
    let tiny = f64::from_bits(1);
    for coordinates in [
        vec![-max, -max, 0., max, max],
        vec![0., tiny, 2. * tiny, 3. * tiny],
        vec![1e150, 1e150_f64.next_up(), 1e150_f64.next_down()],
    ] {
        let data: Vec<_> = coordinates
            .into_iter()
            .enumerate()
            .map(|(i, x)| sample([x, 0., 0.], Some(i as f64), Some(-(i as f64))))
            .collect();
        let cancel = Cancellation::default();
        let budget = MemoryBudget::new(1 << 20);
        let index = SampleIndex::new(&data, &budget, &cancel).unwrap();
        let mut workspace = Workspace::new(3, 2, &budget).unwrap();
        let edges = [tiny, 1e135, max];
        let reference = variography::experimental(&data, 0, 1, &edges, None, &cancel).unwrap();
        let result = index
            .experimental([0, 1], &edges, None, &mut workspace, &cancel)
            .unwrap();
        close(result.bins, &reference);
    }
    let data = vec![
        sample([0.; 3], Some(max), Some(0.25)),
        sample([0.; 3], Some(-max), Some(-0.25)),
    ];
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 20);
    let index = SampleIndex::new(&data, &budget, &cancel).unwrap();
    let mut workspace = Workspace::new(1, 1, &budget).unwrap();
    let result = index
        .experimental([0, 1], &[1.], None, &mut workspace, &cancel)
        .unwrap();
    assert_eq!(result.bins[0].pairs, 1);
    assert_eq!(result.bins[0].mean_distance, Some(0.));
    assert_eq!(result.bins[0].semivariance, Some(max * 0.5));
    assert!(matches!(
        index.experimental([0, 0], &[1.], None, &mut workspace, &cancel),
        Err(Error::Numerical { .. })
    ));
    assert_eq!(
        index
            .experimental([1, 1], &[1.], None, &mut workspace, &cancel)
            .unwrap()
            .bins[0]
            .semivariance,
        Some(0.125)
    );
}

#[test]
fn invalid_inputs_budget_cancellation_and_empty_results() {
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 20);
    let data = fixture(100);
    assert!(matches!(
        SampleIndex::new(&data, &MemoryBudget::new(1), &cancel),
        Err(Error::MemoryBudget { .. })
    ));
    assert!(Workspace::new(usize::MAX, 2, &budget).is_err());
    assert!(Workspace::new(0, 2, &budget).is_err());
    assert!(Workspace::new(2, 0, &budget).is_err());
    assert!(Workspace::new(20, 20, &MemoryBudget::new(1)).is_err());
    assert_eq!(budget.used(), 0);
    {
        let index = SampleIndex::new(&data, &budget, &cancel).unwrap();
        let mut workspace = Workspace::new(2, 2, &budget).unwrap();
        for edges in [
            &[][..],
            &[0.][..],
            &[1., 1.][..],
            &[f64::NAN][..],
            &[f64::INFINITY][..],
            &[1., 2., 3.][..],
        ] {
            assert!(index
                .experimental([0, 1], edges, None, &mut workspace, &cancel)
                .is_err());
        }
        assert!(index
            .experimental([usize::MAX, 0], &[1.], None, &mut workspace, &cancel)
            .is_err());
        cancel.cancel();
        assert!(matches!(
            index.experimental([0, 1], &[1.], None, &mut workspace, &cancel),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            SampleIndex::new(&[], &budget, &cancel),
            Err(Error::Cancelled)
        ));
        index
            .experimental(
                [0, 1],
                &[1.],
                None,
                &mut workspace,
                &Cancellation::default(),
            )
            .unwrap();
    }
    assert_eq!(budget.used(), 0);
    let cancel = Cancellation::default();
    let invalid = vec![sample([0.; 3], Some(f64::NAN), None)];
    assert!(SampleIndex::new(&invalid, &budget, &cancel).is_err());
    assert_eq!(budget.used(), 0);
    for data in [vec![], vec![sample([0.; 3], None, None)]] {
        let index = SampleIndex::new(&data, &budget, &cancel).unwrap();
        let mut workspace = Workspace::new(1, 1, &budget).unwrap();
        let result = index
            .experimental([0, 1], &[1.], None, &mut workspace, &cancel)
            .unwrap();
        assert_eq!(result.candidate_pairs, 0);
        assert_eq!(result.bins[0].pairs, 0);
        assert!(result.bins[0].semivariance.is_none());
    }
}

#[test]
fn extreme_internal_tree_nodes_and_inflight_cancellation() {
    let data: Vec<_> = (0..1024)
        .map(|i| {
            let c = match i % 4 {
                0 => -f64::MAX,
                1 => -1.,
                2 => 1.,
                _ => f64::MAX,
            };
            sample([c, c, c], Some((i % 7) as f64), Some(-(i as f64)))
        })
        .collect();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let index = SampleIndex::new(&data, &budget, &cancel).unwrap();
    let mut workspace = Workspace::new(2, 8, &budget).unwrap();
    let reference = variography::experimental(&data, 0, 1, &[4., f64::MAX], None, &cancel).unwrap();
    close(
        index
            .experimental([0, 1], &[4., f64::MAX], None, &mut workspace, &cancel)
            .unwrap()
            .bins,
        &reference,
    );
    let dense = vec![sample([0.; 3], Some(1.), Some(2.)); 10000];
    let index = SampleIndex::new(&dense, &budget, &cancel).unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            barrier.wait();
            std::thread::sleep(std::time::Duration::from_millis(10));
            cancel.cancel();
        });
        barrier.wait();
        assert!(matches!(
            index.experimental([0, 1], &[1.], None, &mut workspace, &cancel),
            Err(Error::Cancelled)
        ));
    });
    let sparse = SampleIndex::new(&data, &budget, &Cancellation::default()).unwrap();
    close(
        sparse
            .experimental(
                [0, 1],
                &[4., f64::MAX],
                None,
                &mut workspace,
                &Cancellation::default(),
            )
            .unwrap()
            .bins,
        &reference,
    );
}
