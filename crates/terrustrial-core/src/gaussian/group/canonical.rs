//! Stable observation validation and compaction using bounded reusable indices.
use crate::execution::Reservation;
use crate::{
    data::Observation,
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    Error, Result,
};

pub(crate) struct CanonicalWorkspace {
    indices: Vec<usize>,
    keep: Vec<u8>,
    id_order_valid: bool,
    reservation: Reservation,
}
impl CanonicalWorkspace {
    pub(crate) fn new(budget: &MemoryBudget) -> Result<Self> {
        Ok(Self {
            indices: Vec::new(),
            keep: Vec::new(),
            id_order_valid: false,
            reservation: budget.reserve(0)?,
        })
    }
    pub(crate) fn compact(
        &mut self,
        observations: &mut Vec<Observation>,
        check_values: bool,
        cancel: &Cancellation,
    ) -> Result<()> {
        cancel.check()?;
        // The scan wins for small neighborhoods; release comparisons place the
        // crossover between 48 and 64 records on the measured host.
        if observations.len() <= 48 {
            return scalar(observations, check_values);
        }
        self.sorted(observations, check_values, cancel)
    }
    fn sorted(
        &mut self,
        observations: &mut Vec<Observation>,
        check_values: bool,
        cancel: &Cancellation,
    ) -> Result<()> {
        let n = observations.len();
        self.id_order_valid = false;
        self.reserve(n)?;
        self.indices.clear();
        self.indices.extend(0..n);
        self.keep.clear();
        self.keep.resize(n, 1);
        self.indices
            .sort_unstable_by_key(|&i| (observations[i].id(), i));
        cancel.check()?;
        self.validate_sorted(observations, check_values, cancel)
    }

    fn reserve(&mut self, n: usize) -> Result<()> {
        if n > self.indices.capacity() || n > self.keep.capacity() {
            // Grow together. Drop prior scratch on failure so its charge cannot
            // undercount an individual vector that grew before the other failed.
            let capacity = n.max(self.indices.capacity()).max(self.keep.capacity());
            let bytes = checked_len(&[capacity, std::mem::size_of::<usize>() + 1])?;
            self.reservation
                .resize(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
            if self
                .indices
                .try_reserve_exact(capacity - self.indices.len())
                .is_err()
                || self
                    .keep
                    .try_reserve_exact(capacity - self.keep.len())
                    .is_err()
            {
                self.indices = Vec::new();
                self.keep = Vec::new();
                self.reservation.resize(0)?;
                return Err(invalid("canonical observation scratch allocation"));
            }
        }
        Ok(())
    }

    fn validate_sorted(
        &mut self,
        observations: &mut Vec<Observation>,
        check_values: bool,
        cancel: &Cancellation,
    ) -> Result<()> {
        let n = observations.len();
        // Validate every ID, including aliases that will later merge into a
        // different exact record. Never erase evidence of an ID conflict.
        let mut start = 0;
        while start < n {
            let first = self.indices[start];
            let mut end = start + 1;
            while end < n && observations[self.indices[end]].id() == observations[first].id() {
                let next = self.indices[end];
                same_id(&observations[first], &observations[next], check_values)?;
                self.keep[next] = 0;
                end += 1;
            }
            start = end;
        }
        self.indices
            .retain(|&i| self.keep[i] != 0 && observations[i].noise_variance() == 0.);
        self.indices.sort_unstable_by(|&a, &b| {
            observations[a]
                .site()
                .geometry_cmp(observations[b].site())
                .then(a.cmp(&b))
        });
        cancel.check()?;
        let mut start = 0;
        while start < self.indices.len() {
            let first = self.indices[start];
            let mut end = start + 1;
            while end < self.indices.len()
                && observations[self.indices[end]].site() == observations[first].site()
            {
                let next = self.indices[end];
                same_exact(&observations[first], &observations[next], check_values)?;
                self.keep[next] = 0;
                end += 1;
            }
            start = end;
        }
        cancel.check()?;
        let mut index = 0;
        observations.retain(|_| {
            let keep = self.keep[index] != 0;
            index += 1;
            keep
        });
        Ok(())
    }

    /// Order validated, compacted records without sorting their large payloads.
    /// The retained index scratch gives destination-to-source permutation cycles.
    pub(crate) fn sort_by_id(
        &mut self,
        observations: &mut [Observation],
        cancel: &Cancellation,
    ) -> Result<()> {
        cancel.check()?;
        if observations
            .windows(2)
            .all(|pair| pair[0].id() <= pair[1].id())
        {
            return Ok(());
        }
        if observations.len() <= 8 {
            observations.sort_unstable_by_key(Observation::id);
            return Ok(());
        }
        // Small-neighborhood validation preserves this scratch. Reuse the last
        // permutation only after checking that it still orders every current ID.
        // Payload changes are retained and checked later by conditioning reuse.
        if !self.id_order_valid
            || self.indices.len() != observations.len()
            || !self
                .indices
                .windows(2)
                .all(|pair| observations[pair[0]].id() <= observations[pair[1]].id())
        {
            self.id_order_valid = false;
            self.reserve(observations.len())?;
            self.indices.clear();
            self.indices.extend(0..observations.len());
            self.indices.sort_unstable_by_key(|&i| observations[i].id());
            self.id_order_valid = true;
        }
        cancel.check()?;
        if observations.len() <= 48 {
            self.keep.resize(observations.len(), 0);
            self.keep.fill(0);
            self.apply_order::<true>(observations);
        } else {
            // Large-neighborhood validation overwrites the indices next time,
            // so consuming this permutation avoids a separate visited pass.
            self.id_order_valid = false;
            self.apply_order::<false>(observations);
        }
        Ok(())
    }

    fn apply_order<const PRESERVE: bool>(&mut self, observations: &mut [Observation]) {
        for start in 0..observations.len() {
            if self.indices[start] == start || (PRESERVE && self.keep[start] != 0) {
                continue;
            }
            let saved = observations[start];
            let mut destination = start;
            loop {
                let source = self.indices[destination];
                if PRESERVE {
                    self.keep[destination] = 1;
                } else {
                    self.indices[destination] = destination;
                }
                if source == start {
                    observations[destination] = saved;
                    break;
                }
                observations[destination] = observations[source];
                destination = source;
            }
        }
    }
}
fn same_id(a: &Observation, b: &Observation, check_values: bool) -> Result<()> {
    if a.site() != b.site()
        || (check_values && a.value() != b.value())
        || a.noise_variance() != b.noise_variance()
    {
        return Err(invalid("observation ID denotes different data"));
    }
    Ok(())
}
fn same_exact(a: &Observation, b: &Observation, check_values: bool) -> Result<()> {
    if check_values && a.value() != b.value() {
        return Err(invalid(
            "conflicting exact observations on the same support",
        ));
    }
    Ok(())
}
// Small neighborhoods avoid scratch setup and sorting. Swaps permute only the
// already processed prefix, so it still contains every earlier record, including
// discarded aliases. The kept prefix alone is sufficient only for support lookup.
pub(crate) fn scalar(selected: &mut Vec<Observation>, check_values: bool) -> Result<()> {
    let mut kept = 0;
    for index in 0..selected.len() {
        let observation = &selected[index];
        if let Some(existing) = selected[..index]
            .iter()
            .find(|o| o.id() == observation.id())
        {
            same_id(existing, observation, check_values)?;
            continue;
        }
        if observation.noise_variance() == 0. {
            if let Some(existing) = selected[..kept]
                .iter()
                .find(|o| o.noise_variance() == 0. && o.site() == observation.site())
            {
                same_exact(existing, observation, check_values)?;
                continue;
            }
        }
        if kept != index {
            selected.swap(kept, index);
        }
        kept += 1;
    }
    selected.truncate(kept);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Support;
    use crate::data::{Point3, Site};
    use std::{collections::HashMap, hint::black_box, time::Instant};

    #[test]
    fn id_ordering_matches_reference_without_hot_allocations_or_payload_changes() {
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let mut workspace = CanonicalWorkspace::new(&budget).unwrap();
        workspace.reserve(256).unwrap();
        for n in [0, 1, 2, 3, 8, 16, 32, 48, 64, 65, 129, 256] {
            for reverse in [false, true] {
                let mut records: Vec<_> = (0..n)
                    .map(|i| {
                        Observation::new(
                            u64::MAX - ((i * 37) % n) as u64,
                            Site::point(Point3::new([i as f64, -0., 0.]).unwrap(), i % 3),
                            if i % 2 == 0 { -0. } else { i as f64 },
                            0.01,
                        )
                        .unwrap()
                    })
                    .collect();
                if reverse {
                    records.reverse();
                }
                let mut expected = records.clone();
                expected.sort_unstable_by_key(Observation::id);
                let input = records.clone();
                let allocations = allocation_counter::measure(|| {
                    for _ in 0..3 {
                        records.copy_from_slice(&input);
                        workspace.sort_by_id(&mut records, &cancel).unwrap();
                        assert!(records.iter().zip(&expected).all(|(a, b)| a.same_bits(b)));
                        workspace.sort_by_id(&mut records, &cancel).unwrap();
                    }
                });
                assert_eq!(allocations.count_total, 0);
                assert!(records.iter().zip(&expected).all(|(a, b)| a.same_bits(b)));
            }
        }
        let records: [_; 10] = std::array::from_fn(|i| {
            Observation::new(
                10 - i as u64,
                Site::point(Point3::new([i as f64; 3]).unwrap(), 0),
                -0.,
                0.,
            )
            .unwrap()
        });
        let mut actual = records;
        let limited = MemoryBudget::new(1);
        let mut small = CanonicalWorkspace::new(&limited).unwrap();
        assert!(matches!(
            small.sort_by_id(&mut actual, &cancel),
            Err(Error::MemoryBudget { .. })
        ));
        assert!(actual.iter().zip(&records).all(|(a, b)| a.same_bits(b)));
        cancel.cancel();
        assert!(matches!(
            workspace.sort_by_id(&mut actual, &cancel),
            Err(Error::Cancelled)
        ));
        assert!(actual.iter().zip(&records).all(|(a, b)| a.same_bits(b)));
        drop(workspace);
        drop(small);
        assert_eq!(budget.used(), 0);
        assert_eq!(limited.used(), 0);
    }

    fn data(n: usize, exact: bool) -> Vec<Observation> {
        (0..n)
            .map(|i| {
                Observation::new(
                    (i * 11939 % n.max(1)) as u64,
                    Site {
                        variable: i % 3,
                        support: if i % 7 == 0 {
                            Support::block(
                                Point3::new([i as f64, 0., 0.]).unwrap(),
                                [0.2, 0.1, 0.],
                                [2, 3, 1],
                            )
                            .unwrap()
                        } else {
                            Support::point(Point3::new([i as f64, -0., 0.]).unwrap())
                        },
                    },
                    i as f64,
                    if exact { 0. } else { 0.01 },
                )
                .unwrap()
            })
            .collect()
    }
    // Independent allocating oracle: validate IDs across the original complete
    // input before coalescing exact supports in original order.
    fn oracle(input: &[Observation], check_values: bool) -> Result<Vec<Observation>> {
        let mut ids = HashMap::new();
        let mut output: Vec<Observation> = Vec::new();
        for o in input {
            if let Some(previous) = ids.insert(o.id(), o) {
                if previous.site() != o.site()
                    || previous.noise_variance() != o.noise_variance()
                    || (check_values && previous.value() != o.value())
                {
                    return Err(invalid("oracle ID conflict"));
                }
                continue;
            }
            if o.noise_variance() == 0. {
                if let Some(previous) = output
                    .iter()
                    .find(|p| p.noise_variance() == 0. && p.site() == o.site())
                {
                    if check_values && previous.value() != o.value() {
                        return Err(invalid("oracle exact conflict"));
                    }
                    continue;
                }
            }
            output.push(*o);
        }
        Ok(output)
    }
    fn compare(expected: &Result<Vec<Observation>>, result: Result<()>, actual: &[Observation]) {
        match expected {
            Err(_) => assert!(result.is_err()),
            Ok(expected) => {
                result.unwrap();
                assert_eq!(expected.len(), actual.len());
                assert!(expected.iter().zip(actual).all(|(a, b)| a.same_bits(b)));
            }
        }
    }
    #[test]
    fn scalar_and_sorted_match_independent_oracle_with_aliases_supports_and_conflicts() {
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let mut workspace = CanonicalWorkspace::new(&budget).unwrap();
        for n in [0, 1, 2, 3, 16, 64, 127, 128, 129, 257, 1024] {
            for exact in [false, true] {
                let base = data(n, exact);
                for variant in 0..8 {
                    let mut input = base.clone();
                    if n > 0 {
                        let first = &base[0];
                        // Include an exact alias followed by a repeated alias ID.
                        let alias =
                            Observation::new(n as u64 + 7, *first.site(), first.value(), 0.)
                                .unwrap();
                        input.push(alias);
                        input.push(
                            Observation::new(
                                alias.id(),
                                if variant == 1 {
                                    Site::point(Point3::new([99., 1., 0.]).unwrap(), 0)
                                } else {
                                    *alias.site()
                                },
                                if variant == 2 { 999. } else { alias.value() },
                                if variant == 3 {
                                    0.2
                                } else {
                                    alias.noise_variance()
                                },
                            )
                            .unwrap(),
                        );
                        input.push(
                            Observation::new(
                                first.id(),
                                *first.site(),
                                first.value(),
                                first.noise_variance(),
                            )
                            .unwrap(),
                        );
                        if variant >= 4 {
                            let s = Site {
                                variable: 0,
                                support: Support::block(
                                    Point3::new([-0., 0., -0.]).unwrap(),
                                    [-0., 0., 0.],
                                    [1; 3],
                                )
                                .unwrap(),
                            };
                            input.push(
                                Observation::new(
                                    n as u64 + 8,
                                    s,
                                    if variant == 5 { 7. } else { -0. },
                                    -0.,
                                )
                                .unwrap(),
                            );
                        }
                        if variant >= 6 {
                            // Deterministic shuffle changes which duplicate is retained.
                            let mut state = 19823_u64;
                            for i in (1..input.len()).rev() {
                                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                input.swap(i, (state as usize) % (i + 1));
                            }
                        }
                    }
                    for check_values in [false, true] {
                        let expected = oracle(&input, check_values);
                        let mut small = input.clone();
                        let result = scalar(&mut small, check_values);
                        compare(&expected, result, &small);
                        let mut sorted = input.clone();
                        let result = workspace.sorted(&mut sorted, check_values, &cancel);
                        compare(&expected, result, &sorted);
                        let mut adaptive = input.clone();
                        let result = workspace.compact(&mut adaptive, check_values, &cancel);
                        compare(&expected, result, &adaptive);
                    }
                }
            }
        }
        drop(workspace);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn sorted_scratch_is_bounded_reusable_and_failure_recovery_preserves_input() {
        let n = 4096;
        let input = data(n, true);
        let budget = MemoryBudget::new((n * (std::mem::size_of::<usize>() + 1)) as u64);
        let cancel = Cancellation::default();
        let mut workspace = CanonicalWorkspace::new(&budget).unwrap();
        let mut working = input.clone();
        workspace.compact(&mut working, true, &cancel).unwrap();
        assert_eq!(budget.used(), budget.limit());
        let allocations = allocation_counter::measure(|| {
            for _ in 0..20 {
                for size in [129, 257, 513, 4096] {
                    working.clear();
                    working.extend_from_slice(&input[..size]);
                    workspace.compact(&mut working, true, &cancel).unwrap();
                    assert_eq!(working.len(), size);
                }
            }
        });
        assert_eq!(allocations.count_total, 0);
        let mut larger = data(n + 1, false);
        let original = larger.clone();
        assert!(workspace.compact(&mut larger, true, &cancel).is_err());
        assert!(larger.iter().zip(&original).all(|(a, b)| a.same_bits(b)));
        assert_eq!(budget.used(), budget.limit());
        working.clear();
        working.extend_from_slice(&input);
        workspace.compact(&mut working, true, &cancel).unwrap();
        let stopped = Cancellation::default();
        stopped.cancel();
        assert!(workspace.compact(&mut working, true, &stopped).is_err());
        assert_eq!(working.len(), n);
        drop(workspace);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    #[ignore = "manual release timing comparison, not a correctness gate"]
    fn canonicalization_timing_comparison() {
        let cancel = Cancellation::default();
        let budget = MemoryBudget::new(1 << 24);
        let mut workspace = CanonicalWorkspace::new(&budget).unwrap();
        println!("CANONICAL_BENCH metric,value");
        for n in [16, 32, 40, 48, 64, 96, 128, 256, 512, 1024, 4096] {
            for exact in [false, true] {
                let input = data(n, exact);
                let iterations = (32768 / n).max(8);
                let mut working = input.clone();
                workspace.sorted(&mut working, true, &cancel).unwrap();
                let mut seconds = [0.; 2];
                for pair in 0..4 {
                    for backend in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let start = Instant::now();
                        for _ in 0..iterations {
                            working.clear();
                            working.extend_from_slice(black_box(&input));
                            if backend == 0 {
                                scalar(&mut working, true).unwrap();
                            } else {
                                workspace.sorted(&mut working, true, &cancel).unwrap();
                            }
                            black_box(&working);
                        }
                        seconds[backend] += start.elapsed().as_secs_f64();
                    }
                }
                assert!(working.iter().zip(&input).all(|(a, b)| a.same_bits(b)));
                println!(
                    "CANONICAL_BENCH n{n}_exact_{exact}_scalar_seconds,{}",
                    seconds[0]
                );
                println!(
                    "CANONICAL_BENCH n{n}_exact_{exact}_sorted_seconds,{}",
                    seconds[1]
                );
                println!("CANONICAL_BENCH n{n}_exact_{exact}_iterations,{iterations}");
            }
        }
    }
}
