use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    storage::PointFile,
    variography::{
        experimental,
        sampling::{experimental_sampled, FileSamples, SampleSource, SamplingOptions},
        Direction, Sample,
    },
    Result,
};
fn options<'a>(
    budget: &'a MemoryBudget,
    cancel: &'a Cancellation,
    edges: &'a [f64],
    count: u64,
) -> SamplingOptions<'a> {
    SamplingOptions {
        variables: [0, 1],
        maximum_pairs: count,
        seed: 4,
        edges,
        direction: None,
        budget,
        cancellation: cancel,
    }
}
fn samples() -> Vec<Sample> {
    (0..20)
        .map(|i| Sample {
            point: Point3::new([i as f64, 0.0, 0.0]).unwrap(),
            values: vec![Some(i as f64), Some(-2.0 * i as f64)],
        })
        .collect()
}
#[test]
fn full_sample_matches_independent_oracle_and_subset_repeats_without_duplicates() {
    let data = samples();
    let mut source = data.as_slice();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let edges = [2.0, 5.0, 20.0];
    let full =
        experimental_sampled(&mut source, options(&budget, &cancel, &edges, u64::MAX)).unwrap();
    let reference = experimental(&data, 0, 1, &edges, None, &cancel).unwrap();
    assert_eq!(full.population_pairs, 190);
    assert_eq!(full.sampled_pairs, 190);
    assert_eq!(full.eligible_pairs, 190);
    for (a, b) in full.bins.iter().zip(reference) {
        assert_eq!(a.pairs, b.pairs);
        assert_eq!(a.mean_distance, b.mean_distance);
        assert_eq!(a.semivariance, b.semivariance);
    }
    let a = experimental_sampled(&mut source, options(&budget, &cancel, &edges, 70)).unwrap();
    let b = experimental_sampled(&mut source, options(&budget, &cancel, &edges, 70)).unwrap();
    assert_eq!(a.eligible_pairs, 70);
    assert_eq!(format!("{:?}", a.bins), format!("{:?}", b.bins));
    assert_eq!(budget.used(), 0);
}
#[test]
fn virtual_source_with_u64_max_records_uses_only_requested_samples() {
    struct Virtual {
        reads: usize,
    }
    impl SampleSource for Virtual {
        fn len(&self) -> u64 {
            u64::MAX
        }
        fn read(&mut self, index: u64, out: &mut Sample) -> Result<()> {
            self.reads += 1;
            out.point = Point3::new([index as f64, 0.0, 0.0])?;
            out.values = vec![Some(index as f64), Some(index as f64)];
            Ok(())
        }
    }
    let budget = MemoryBudget::new(8192);
    let cancel = Cancellation::default();
    let mut source = Virtual { reads: 0 };
    let out = experimental_sampled(&mut source, options(&budget, &cancel, &[1e21], 20)).unwrap();
    assert_eq!(out.sampled_pairs, 20);
    assert_eq!(out.eligible_pairs, 20);
    assert!(source.reads <= 40);
    assert!(out.population_pairs > u64::MAX as u128);
}
#[test]
fn point_file_random_reads_agree_with_memory_and_direction_boundaries_are_inclusive() {
    let path =
        std::env::temp_dir().join(format!("terrustrial-sampled-{}.points", std::process::id()));
    let input: Vec<_> = (0..20)
        .map(|i| {
            Observation::new(
                i,
                Site::point(Point3::new([i as f64, 0.0, 0.0]).unwrap(), 0),
                i as f64,
                0.0,
            )
            .unwrap()
        })
        .collect();
    let cancel = Cancellation::default();
    let file = PointFile::create(&path, input.as_slice(), &cancel).unwrap();
    let mut source = FileSamples::new(file.reader().unwrap(), 1).unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let mut configuration = options(&budget, &cancel, &[20.0], 50);
    configuration.variables = [0, 0];
    let disk = experimental_sampled(&mut source, configuration).unwrap();
    let data = samples();
    let mut memory = data.as_slice();
    let mut configuration = options(&budget, &cancel, &[20.0], 50);
    configuration.variables = [0, 0];
    let expected = experimental_sampled(&mut memory, configuration).unwrap();
    assert_eq!(disk.bins[0].semivariance, expected.bins[0].semivariance);
    drop(source);
    std::fs::remove_file(path).unwrap();
    let direction = Direction::new([0.0, 1.0, 0.0], std::f64::consts::FRAC_PI_2, 100.0).unwrap();
    assert_eq!(
        experimental(&data, 0, 0, &[20.0], Some(&direction), &cancel).unwrap()[0].pairs,
        190
    );
}
