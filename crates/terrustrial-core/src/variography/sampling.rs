//! Reproducible uniform unordered-pair sampling without loading the source in RAM.
//! This estimates variograms from a subset. Report retained pair counts as sampled
//! counts, not as exact full-population counts. Short-lag bins may need stratification.
use super::{Bin, Direction, Sample};
use crate::{
    data::Point3,
    error::{checked_len, invalid},
    execution::{random_stream, Cancellation, MemoryBudget},
    storage::PointReader,
    Error, Result,
};
use rand::RngExt;
use std::collections::HashSet;

pub trait SampleSource {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Fill reusable storage. Implementations must represent the same immutable
    /// input throughout the operation and validate missing/nonfinite data.
    fn read(&mut self, index: u64, output: &mut Sample) -> Result<()>;
}
impl SampleSource for &[Sample] {
    fn len(&self) -> u64 {
        <[Sample]>::len(self) as u64
    }
    fn read(&mut self, index: u64, output: &mut Sample) -> Result<()> {
        let index = usize::try_from(index).map_err(|_| Error::SizeOverflow)?;
        let sample = self
            .get(index)
            .ok_or_else(|| invalid("sample index out of bounds"))?;
        output.point = sample.point;
        output.values.clone_from(&sample.values);
        Ok(())
    }
}
/// A point file has one attribute per record. Other attributes remain missing.
/// Supply a wide SampleSource when computing co-located cross-variograms.
pub struct FileSamples {
    reader: PointReader,
    variables: usize,
}
impl FileSamples {
    pub fn new(reader: PointReader, variables: usize) -> Result<Self> {
        if variables == 0 {
            return Err(invalid("positive sample variable count required"));
        }
        Ok(Self { reader, variables })
    }
}
impl SampleSource for FileSamples {
    fn len(&self) -> u64 {
        self.reader.len()
    }
    fn read(&mut self, index: u64, output: &mut Sample) -> Result<()> {
        let record = self.reader.get(index)?;
        if record.site().variable >= self.variables {
            return Err(invalid("point file contains an unexpected variable"));
        }
        output.point = record.site().support.center();
        output.values.resize(self.variables, None);
        output.values.fill(None);
        output.values[record.site().variable] = Some(record.value());
        Ok(())
    }
}
pub struct SamplingOptions<'a> {
    pub variables: [usize; 2],
    pub maximum_pairs: u64,
    pub seed: u64,
    pub edges: &'a [f64],
    pub direction: Option<&'a Direction>,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Clone, Debug)]
pub struct SampledVariogram {
    pub bins: Vec<Bin>,
    pub population_pairs: u128,
    pub sampled_pairs: u64,
    pub eligible_pairs: u64,
    pub seed: u64,
}
/// Floyd's uniform sample without replacement over exact u128 triangular ordinals.
/// Sorted ordinals fix reduction order independently of hash randomization. Source
/// reads are O(sampled pairs); source storage and I/O complexity remain explicit.
pub fn experimental_sampled(
    source: &mut impl SampleSource,
    options: SamplingOptions<'_>,
) -> Result<SampledVariogram> {
    let edges = options.edges;
    if options.maximum_pairs == 0
        || edges.is_empty()
        || edges.iter().any(|e| !e.is_finite() || *e <= 0.0)
        || edges.windows(2).any(|e| e[0] >= e[1])
    {
        return Err(invalid("sampling count or lag edges"));
    }
    options.cancellation.check()?;
    let n = source.len();
    let population = (n as u128) * (n.saturating_sub(1) as u128) / 2;
    let count = population.min(options.maximum_pairs as u128) as u64;
    let partial = (count as u128) < population;
    let sample_count = if partial {
        usize::try_from(count).map_err(|_| Error::SizeOverflow)?
    } else {
        0
    };
    let bytes = checked_len(&[sample_count, 80])?
        .checked_add(checked_len(&[edges.len(), 64])?)
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let mut selected = HashSet::new();
    selected
        .try_reserve(sample_count)
        .map_err(|_| invalid("unable to allocate pair sample"))?;
    let mut ordinals = Vec::new();
    ordinals
        .try_reserve_exact(sample_count)
        .map_err(|_| invalid("unable to allocate pair ordinals"))?;
    if partial {
        let mut rng = random_stream(options.seed, 0, 0, 1);
        for j in population - count as u128..population {
            options.cancellation.check()?;
            let candidate = rng.random_range(0..=j);
            let value = if selected.contains(&candidate) {
                j
            } else {
                candidate
            };
            selected.insert(value);
            ordinals.push(value);
        }
        ordinals.sort_unstable();
    }
    drop(selected);
    let pairs: Box<dyn Iterator<Item = u128>> = if partial {
        Box::new(ordinals.into_iter())
    } else {
        Box::new(0..population)
    };
    let mut a = Sample {
        point: Point3::new([0.0; 3])?,
        values: Vec::new(),
    };
    let mut b = a.clone();
    let mut previous = None;
    let mut bins = vec![(0_u64, 0.0, 0.0); edges.len()];
    let mut eligible = 0_u64;
    for ordinal in pairs {
        options.cancellation.check()?;
        let (i, j) = decode_pair(n, ordinal);
        if previous != Some(i) {
            source.read(i, &mut a)?;
            validate(&a, options.variables)?;
            previous = Some(i);
        }
        source.read(j, &mut b)?;
        validate(&b, options.variables)?;
        let [va, vb] = options.variables;
        let (Some(aa), Some(ab), Some(ba), Some(bb)) =
            (a.values[va], a.values[vb], b.values[va], b.values[vb])
        else {
            continue;
        };
        let distance = a.point.distance(b.point);
        if distance > *edges.last().unwrap()
            || options
                .direction
                .is_some_and(|d| !d.accepts(a.point, b.point, distance))
        {
            continue;
        }
        let gamma = 0.5 * (aa - ba) * (ab - bb);
        let index = edges.partition_point(|edge| distance > *edge);
        let bin = &mut bins[index];
        bin.0 += 1;
        eligible += 1;
        bin.1 += (distance - bin.1) / bin.0 as f64;
        bin.2 += (gamma - bin.2) / bin.0 as f64;
        if !bin.1.is_finite() || !bin.2.is_finite() {
            return Err(Error::Numerical {
                context: "sampled variogram accumulation",
            });
        }
    }
    Ok(SampledVariogram {
        bins: bins
            .into_iter()
            .map(|(pairs, h, g)| Bin {
                pairs,
                mean_distance: (pairs > 0).then_some(h),
                semivariance: (pairs > 0).then_some(g),
            })
            .collect(),
        population_pairs: population,
        sampled_pairs: count,
        eligible_pairs: eligible,
        seed: options.seed,
    })
}
fn validate(sample: &Sample, variables: [usize; 2]) -> Result<()> {
    if sample.values.len() <= variables[0].max(variables[1])
        || sample.values.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(invalid("sample variables or nonfinite values"));
    }
    Ok(())
}
fn prefix(n: u64, i: u64) -> u128 {
    (i as u128) * (2 * n as u128 - i as u128 - 1) / 2
}
fn decode_pair(n: u64, ordinal: u128) -> (u64, u64) {
    let mut lo = 0;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if prefix(n, mid) <= ordinal {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, lo + 1 + (ordinal - prefix(n, lo)) as u64)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn triangular_mapping_handles_every_small_pair_and_u64_boundaries() {
        for n in 2..30 {
            let mut ordinal = 0;
            for i in 0..n {
                for j in i + 1..n {
                    assert_eq!(decode_pair(n, ordinal), (i, j));
                    ordinal += 1;
                }
            }
        }
        let n = u64::MAX;
        let total = n as u128 * (n - 1) as u128 / 2;
        assert_eq!(decode_pair(n, 0), (0, 1));
        assert_eq!(decode_pair(n, total - 1), (n - 2, n - 1));
    }
}
