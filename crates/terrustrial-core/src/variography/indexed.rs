//! Exact resident irregular variography with spatial pruning and parallel rows.
//!
//! An immutable borrowed input prevents stale attribute/index pairs. The packed hierarchy
//! prunes closed Cartesian boxes; the final unsquared distance and directional
//! predicates define membership. Dense searches can still require O(N²) work.
//! No accepted-pair list is materialized. Fixed 256-row blocks and ordered merging
//! make successful results invariant to Rayon worker count and workspace window.
use super::{
    accumulation::{half_increment_product, ScaledMean},
    Bin, Direction, Sample,
};
use crate::spatial::hierarchy::{Bounded, Bounds, Hierarchy};
use crate::{
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    spatial::visit_envelope,
    Error, Result,
};
use rayon::prelude::*;

pub(super) const ROWS_PER_BLOCK: usize = 256;

struct Entry {
    point: [f64; 3],
    row: usize,
}
impl Bounded for Entry {
    fn bounds(&self) -> Bounds {
        Bounds::point(self.point)
    }
}

/// Reusable geometry index. Row positions identify unordered pairs, so coincident
/// coordinates remain distinct samples and contribute zero-distance pairs.
pub struct SampleIndex<'a> {
    samples: &'a [Sample],
    tree: Hierarchy<Entry>,
    variables: usize,
    _reservation: Reservation,
}
impl<'a> SampleIndex<'a> {
    /// Charges entry and node capacities in addition to caller-owned input.
    /// Allocations are fallible. Allocator metadata is outside MemoryBudget.
    /// This resident index is not an out-of-core source or a hard RSS limit.
    pub fn new(
        samples: &'a [Sample],
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        let bytes = checked_len(&[samples.len(), std::mem::size_of::<Entry>()])?
            .checked_add(Hierarchy::<Entry>::node_bytes(samples.len())?)
            .and_then(|n| n.checked_add(Hierarchy::<Entry>::scratch_bytes(samples.len()).ok()?))
            .ok_or(Error::SizeOverflow)?;
        let mut reservation =
            budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(samples.len())
            .map_err(|_| invalid("unable to allocate variogram index entries"))?;
        let mut variables = usize::MAX;
        for (row, sample) in samples.iter().enumerate() {
            cancel.check()?;
            if sample.values.iter().flatten().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite indexed variogram attribute"));
            }
            variables = variables.min(sample.values.len());
            entries.push(Entry {
                point: sample.point.coordinates(),
                row,
            });
        }
        cancel.check()?;
        let tree = Hierarchy::new(entries)?;
        reservation.resize(tree.allocated_bytes() as u64)?;
        cancel.check()?;
        Ok(Self {
            samples,
            tree,
            variables,
            _reservation: reservation,
        })
    }

    /// Exact unordered pairs with bins `[0,e0], (e0,e1], ...` . Cross pairs need
    /// both attributes at both endpoints. Attributes outside this pair may be
    /// missing. Negative cross-semivariances are retained. Uses the current Rayon
    /// pool; install in an explicit pool to choose the worker count.
    ///
    /// Workspace storage is reused, including output. Errors return no partial
    /// result; the same workspace remains reusable after failure/cancellation.
    pub fn experimental<'w>(
        &self,
        variables: [usize; 2],
        edges: &[f64],
        direction: Option<&Direction>,
        workspace: &'w mut Workspace,
        cancel: &Cancellation,
    ) -> Result<Variogram<'w>> {
        cancel.check()?;
        validate_request(variables, self.variables, edges, workspace.bin_capacity)?;
        let capacity = workspace.bin_capacity;
        workspace.reduce(
            self.samples.len() as u64,
            usize::MAX,
            edges.len(),
            cancel,
            |start, active, partial, status| {
                let evaluate =
                    |(slot, (bins, status)): (usize, (&mut [Accumulator], &mut Result<u128>))| {
                        let first = (start as usize + slot) * ROWS_PER_BLOCK;
                        let last = first.saturating_add(ROWS_PER_BLOCK).min(self.samples.len());
                        *status = self.rows(first..last, variables, edges, direction, bins, cancel);
                    };
                if active == 1 {
                    evaluate((0, (&mut partial[..capacity], &mut status[0])));
                } else {
                    partial
                        .par_chunks_mut(capacity)
                        .zip(status.par_iter_mut())
                        .enumerate()
                        .for_each(evaluate);
                }
            },
        )
    }

    fn rows(
        &self,
        rows: std::ops::Range<usize>,
        [va, vb]: [usize; 2],
        edges: &[f64],
        direction: Option<&Direction>,
        bins: &mut [Accumulator],
        cancel: &Cancellation,
    ) -> Result<u128> {
        let radius = edges[edges.len() - 1];
        // Distance and subtraction round. Expand before computing the bounds,
        // then round each bound outwards again. Overflow disables that bound.
        let outward = radius.next_up();
        let mut candidates = 0_u128;
        for i in rows {
            cancel.check()?;
            let a = &self.samples[i];
            let (Some(aa), Some(ab)) = (a.values[va], a.values[vb]) else {
                continue;
            };
            let center = a.point.coordinates();
            let bounds = Bounds::from_corners(
                center.map(|x| (x - outward).next_down()),
                center.map(|x| (x + outward).next_up()),
            );
            let mut offer = |row: usize| -> Result<()> {
                candidates += 1;
                let b = &self.samples[row];
                let (Some(ba), Some(bb)) = (b.values[va], b.values[vb]) else {
                    return Ok(());
                };
                let distance = a.point.distance(b.point);
                if distance > radius
                    || direction.is_some_and(|d| !d.accepts(a.point, b.point, distance))
                {
                    return Ok(());
                }
                let bin = &mut bins[edges.partition_point(|edge| distance > *edge)];
                bin.distance.add(distance)?;
                bin.gamma.add(half_increment_product(aa, ba, ab, bb)?)?;
                Ok(())
            };
            if self.tree.bounds().is_some_and(|b| bounds.contains(b)) {
                // A box covering the whole dataset cannot prune. Visit the
                // remaining rows directly without traversing/indexing old rows.
                for row in i + 1..self.samples.len() {
                    cancel.check()?;
                    offer(row)?;
                }
            } else {
                visit_envelope(&self.tree, bounds, |entry| {
                    cancel.check()?;
                    if entry.row > i {
                        offer(entry.row)?;
                    }
                    Ok(())
                })?;
            }
        }
        Ok(candidates)
    }
}

#[derive(Clone, Default)]
pub(super) struct Accumulator {
    pub(super) distance: ScaledMean,
    pub(super) gamma: ScaledMean,
}

/// Owned buffers are O(bin_capacity * window_blocks), independent of input and
/// accepted pair counts. Set window_blocks >= workers to expose parallelism.
/// Changing this window does not change the fixed numerical reduction order.
pub struct Workspace {
    pub(super) bin_capacity: usize,
    partial: Vec<Accumulator>,
    total: Vec<Accumulator>,
    output: Vec<Bin>,
    status: Vec<Result<u128>>,
    _reservation: Reservation,
}
impl Workspace {
    pub fn new(bin_capacity: usize, window_blocks: usize, budget: &MemoryBudget) -> Result<Self> {
        if bin_capacity == 0 || window_blocks == 0 {
            return Err(invalid(
                "positive indexed variogram workspace capacities required",
            ));
        }
        let partial = checked_len(&[bin_capacity, window_blocks])?;
        let bytes = checked_len(&[partial, std::mem::size_of::<Accumulator>()])?
            .checked_add(checked_len(&[
                bin_capacity,
                std::mem::size_of::<Accumulator>() + std::mem::size_of::<Bin>(),
            ])?)
            .and_then(|b| {
                b.checked_add(window_blocks.checked_mul(std::mem::size_of::<Result<u128>>())?)
            })
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        Ok(Self {
            bin_capacity,
            partial: buffer(partial, Accumulator::default)?,
            total: buffer(bin_capacity, Accumulator::default)?,
            output: buffer(bin_capacity, || Bin {
                pairs: 0,
                mean_distance: None,
                semivariance: None,
            })?,
            status: buffer(window_blocks, || Ok(0))?,
            _reservation: reservation,
        })
    }
}
pub(super) fn validate_request(
    variables: [usize; 2],
    available: usize,
    edges: &[f64],
    capacity: usize,
) -> Result<()> {
    if variables.iter().any(|v| *v >= available)
        || edges.is_empty()
        || edges.len() > capacity
        || edges.iter().any(|e| !e.is_finite() || *e <= 0.)
        || edges.windows(2).any(|e| e[0] >= e[1])
    {
        return Err(invalid(
            "variogram variables, lag edges or workspace capacity",
        ));
    }
    Ok(())
}
impl Workspace {
    /// Shared fixed-block reduction for resident and disk-backed sources.
    pub(super) fn reduce(
        &mut self,
        samples: u64,
        maximum_window: usize,
        bins: usize,
        cancel: &Cancellation,
        mut evaluate: impl FnMut(u64, usize, &mut [Accumulator], &mut [Result<u128>]),
    ) -> Result<Variogram<'_>> {
        cancel.check()?;
        self.total.fill(Accumulator::default());
        let blocks = samples.div_ceil(ROWS_PER_BLOCK as u64);
        let window = self.status.len().min(maximum_window);
        let capacity = self.bin_capacity;
        let mut candidates = 0_u128;
        for start in (0..blocks).step_by(window) {
            cancel.check()?;
            let active = window.min(usize::try_from(blocks - start).unwrap_or(usize::MAX));
            self.partial[..active * capacity].fill(Accumulator::default());
            evaluate(
                start,
                active,
                &mut self.partial[..active * capacity],
                &mut self.status[..active],
            );
            for slot in 0..active {
                candidates = candidates
                    .checked_add(std::mem::replace(&mut self.status[slot], Ok(0))?)
                    .ok_or(Error::SizeOverflow)?;
                for (total, part) in self.total[..bins]
                    .iter_mut()
                    .zip(&self.partial[slot * capacity..slot * capacity + bins])
                {
                    total.distance.merge(&part.distance)?;
                    total.gamma.merge(&part.gamma)?;
                }
            }
        }
        cancel.check()?;
        for (bin, total) in self.output.iter_mut().zip(&self.total).take(bins) {
            *bin = Bin {
                pairs: total.distance.count,
                mean_distance: total.distance.value(),
                semivariance: total.gamma.value(),
            };
        }
        Ok(Variogram {
            bins: &self.output[..bins],
            candidate_pairs: candidates,
            population_pairs: samples as u128 * samples.saturating_sub(1) as u128 / 2,
        })
    }
}

pub(super) fn buffer<T>(len: usize, init: impl FnMut() -> T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| invalid("unable to allocate indexed variogram workspace"))?;
    values.resize_with(len, init);
    Ok(values)
}

pub struct Variogram<'a> {
    pub bins: &'a [Bin],
    /// Unordered pairs visited inside outward boxes, after excluding missing
    /// first endpoints. Includes missing second endpoints and rejected geometry.
    pub candidate_pairs: u128,
    /// All unordered input pairs, before missingness and geometric filters.
    pub population_pairs: u128,
}
