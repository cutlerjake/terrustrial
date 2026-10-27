#![cfg(feature = "sqlite")]
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::{NearestWorkspace, Neighborhood, ObservationSource},
    storage::{SqliteReadPool, SqliteStore},
};

fn bits(o: &Observation) -> (u64, usize, [u64; 5]) {
    let [x, y, z] = o.site().support.center().coordinates();
    (
        o.id(),
        o.site().variable,
        [x, y, z, o.value(), o.noise_variance()].map(f64::to_bits),
    )
}
fn check(source: &impl ObservationSource, expected: &[Observation]) {
    let mut actual = Vec::new();
    source
        .scan(&mut |o| {
            actual.push(bits(o));
            Ok(())
        })
        .unwrap();
    actual.sort_unstable();
    let mut expected: Vec<_> = expected.iter().map(bits).collect();
    expected.sort_unstable();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual, expected);
    }
}

#[test]
fn all_zero_signs_survive_reopen_and_point_queries() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bits.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    let data: Vec<_> = (0..32)
        .map(|mask| {
            let fields: [f64; 5] =
                std::array::from_fn(|i| if mask & (1 << i) == 0 { 0. } else { -0. });
            Observation::new(
                mask,
                Site::point(Point3::new(fields[..3].try_into().unwrap()).unwrap(), 0),
                fields[3],
                fields[4],
            )
            .unwrap()
        })
        .collect();
    let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
    store
        .import(data.as_slice(), &Cancellation::default())
        .unwrap();
    check(&store, &data);
    drop(store);
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    check(&store, &data);
    let query = Neighborhood {
        maximum: 32,
        radius: 1.,
    };
    let point = Point3::new([0.; 3]).unwrap();
    let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
    for actual in [
        store.at_site(&Site::point(point, 0)).unwrap(),
        store
            .nearest(point, query, &Cancellation::default())
            .unwrap(),
        store
            .nearest_into(point, &mut workspace, &Cancellation::default())
            .unwrap()
            .to_vec(),
    ] {
        assert_eq!(
            actual.iter().map(bits).collect::<Vec<_>>(),
            data.iter().map(bits).collect::<Vec<_>>()
        );
    }
    store.integrity_check().unwrap();
    let pool = SqliteReadPool::open(&path, 2, 4096, &budget).unwrap();
    check(&pool, &data);
    drop((store, pool, workspace));
    assert_eq!(budget.used(), 0);
}

#[test]
fn finite_binary64_payloads_round_trip_without_changing_bits() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("finite.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    let mut state = 0x53ac_680b_84c6_2a1d_u64;
    let mut next = || loop {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state);
        if value.is_finite() {
            break value;
        }
    };
    let mut data: Vec<_> = (0..2048)
        .map(|id| {
            Observation::new(
                u64::MAX - id,
                Site::point(
                    Point3::new([next(), next(), next()]).unwrap(),
                    id as usize % 7,
                ),
                next(),
                next().abs(),
            )
            .unwrap()
        })
        .collect();
    for (i, value) in [
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        -f64::MAX,
        0.,
        -0.,
    ]
    .into_iter()
    .enumerate()
    {
        data.push(
            Observation::new(
                i as u64,
                Site::point(Point3::new([value; 3]).unwrap(), i),
                value,
                value.abs(),
            )
            .unwrap(),
        );
    }
    let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
    store
        .import(data.as_slice(), &Cancellation::default())
        .unwrap();
    drop(store);
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    check(&store, &data);
    store.integrity_check().unwrap();
}
