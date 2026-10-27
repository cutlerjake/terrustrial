//! Lower-bound bipartite flow and lexicographic completion with reusable scratch.
use super::*;
const NONE: usize = usize::MAX;
#[derive(Clone, Copy, Default)]
struct Edge {
    to: usize,
    next: usize,
    residual: usize,
}
pub(super) struct QuotaWorkspace {
    edges: Vec<Edge>,
    heads: Vec<usize>,
    previous: Vec<usize>,
    queue: Vec<usize>,
    demands: Vec<i128>,
    remaining: Vec<usize>,
    bucket_of: Vec<usize>,
    _reservation: Reservation,
}
fn buffer<T: Clone + Default>(count: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| invalid("minimum-quota workspace allocation"))?;
    out.resize(count, T::default());
    Ok(out)
}
impl QuotaWorkspace {
    pub(super) fn new(
        classes: usize,
        sectors: usize,
        retained: usize,
        budget: &MemoryBudget,
    ) -> Result<Self> {
        let ordinary_nodes = classes
            .checked_add(sectors)
            .and_then(|n| n.checked_add(2))
            .ok_or(Error::SizeOverflow)?;
        let nodes = ordinary_nodes.checked_add(2).ok_or(Error::SizeOverflow)?;
        let buckets = checked_len(&[classes, sectors])?;
        let edges = buckets
            .checked_add(classes)
            .and_then(|n| n.checked_add(sectors))
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_add(ordinary_nodes))
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::SizeOverflow)?;
        let bytes = checked_len(&[edges, size_of::<Edge>()])?
            .checked_add(checked_len(&[
                nodes,
                3 * size_of::<usize>() + size_of::<i128>(),
            ])?)
            .and_then(|n| n.checked_add(buckets.checked_mul(size_of::<usize>())?))
            .and_then(|n| n.checked_add(retained.checked_mul(size_of::<usize>())?))
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let mut edges = buffer::<Edge>(edges)?;
        edges.clear();
        let mut queue = buffer::<usize>(nodes)?;
        queue.clear();
        let mut bucket_of = buffer::<usize>(retained)?;
        bucket_of.clear();
        Ok(Self {
            edges,
            heads: buffer(nodes)?,
            previous: buffer(nodes)?,
            queue,
            demands: buffer(nodes)?,
            remaining: buffer(buckets)?,
            bucket_of,
            _reservation: reservation,
        })
    }
    pub(super) fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
    }
    fn add(&mut self, from: usize, to: usize, residual: usize) -> usize {
        let index = self.edges.len();
        self.edges.push(Edge {
            to,
            next: self.heads[from],
            residual,
        });
        self.heads[from] = index;
        self.edges.push(Edge {
            to: from,
            next: self.heads[to],
            residual: 0,
        });
        self.heads[to] = index + 1;
        index
    }
    fn bound(&mut self, from: usize, to: usize, lower: usize, upper: usize) -> Option<usize> {
        let residual = upper.checked_sub(lower)?;
        self.demands[from] -= lower as i128;
        self.demands[to] += lower as i128;
        Some(self.add(from, to, residual))
    }
    fn flow(
        &mut self,
        source: usize,
        sink: usize,
        limit: usize,
        cancel: &Cancellation,
    ) -> Result<usize> {
        let mut flow = 0;
        while flow < limit {
            cancel.check()?;
            self.previous.fill(NONE);
            self.previous[source] = 0;
            self.queue.clear();
            self.queue.push(source);
            let mut cursor = 0;
            'bfs: while cursor < self.queue.len() {
                cancel.check()?;
                let node = self.queue[cursor];
                cursor += 1;
                let mut index = self.heads[node];
                while index != NONE {
                    let edge = self.edges[index];
                    if edge.residual != 0 && self.previous[edge.to] == NONE {
                        self.previous[edge.to] = index;
                        if edge.to == sink {
                            break 'bfs;
                        }
                        self.queue.push(edge.to);
                    }
                    index = edge.next;
                }
            }
            if self.previous[sink] == NONE {
                break;
            }
            let mut amount = limit - flow;
            let mut node = sink;
            while node != source {
                let edge = self.previous[node];
                amount = amount.min(self.edges[edge].residual);
                node = self.edges[edge ^ 1].to;
            }
            node = sink;
            while node != source {
                let edge = self.previous[node];
                self.edges[edge].residual -= amount;
                self.edges[edge ^ 1].residual += amount;
                node = self.edges[edge ^ 1].to;
            }
            flow += amount;
        }
        Ok(flow)
    }
    fn maximum(
        &mut self,
        query: &SearchNeighborhood,
        variables: &[usize],
        sectors: &[usize; 8],
        selected: usize,
        limit: usize,
        cancel: &Cancellation,
    ) -> Result<Option<usize>> {
        if selected > limit {
            return Ok(None);
        }
        self.edges.clear();
        self.heads.fill(NONE);
        self.demands.fill(0);
        let classes = query.classes();
        let count = query.sectors();
        let super_source = 2 + classes + count;
        let super_sink = super_source + 1;
        for (class, &used) in variables.iter().enumerate() {
            cancel.check()?;
            let Some(cap) = query.variable_cap(class).checked_sub(used) else {
                return Ok(None);
            };
            if self
                .bound(
                    0,
                    2 + class,
                    query.variable_minimum(class).saturating_sub(used),
                    cap,
                )
                .is_none()
            {
                return Ok(None);
            }
            for sector in 0..count {
                self.add(
                    2 + class,
                    2 + classes + sector,
                    self.remaining[class * count + sector],
                );
            }
        }
        let minimums = query.minimums.as_ref().unwrap();
        for (sector, &used) in sectors.iter().enumerate().take(count) {
            let Some(cap) = query
                .octant_maximum
                .unwrap_or(query.maximum)
                .min(query.maximum)
                .checked_sub(used)
            else {
                return Ok(None);
            };
            let minimum = if count == 8 {
                minimums.octants[sector]
            } else {
                0
            };
            if self
                .bound(2 + classes + sector, 1, minimum.saturating_sub(used), cap)
                .is_none()
            {
                return Ok(None);
            }
        }
        let lower = minimums.total.saturating_sub(selected);
        let Some(closing) = self.bound(1, 0, lower, limit - selected) else {
            return Ok(None);
        };
        let auxiliary = self.edges.len();
        for node in 0..super_source {
            cancel.check()?;
            let demand = self.demands[node];
            if demand > 0 {
                self.add(
                    super_source,
                    node,
                    usize::try_from(demand).map_err(|_| Error::SizeOverflow)?,
                );
            } else if demand < 0 {
                self.add(
                    node,
                    super_sink,
                    usize::try_from(-demand).map_err(|_| Error::SizeOverflow)?,
                );
            }
        }
        self.flow(super_source, super_sink, usize::MAX, cancel)?;
        let mut edge = self.heads[super_source];
        while edge != NONE {
            if self.edges[edge].residual != 0 {
                return Ok(None);
            }
            edge = self.edges[edge].next;
        }
        let initial = lower + self.edges[closing ^ 1].residual;
        for edge in &mut self.edges[auxiliary..] {
            edge.residual = 0;
        }
        self.edges[closing].residual = 0;
        self.edges[closing ^ 1].residual = 0;
        let additional = self.flow(0, 1, limit - selected - initial, cancel)?;
        Ok(Some(selected + initial + additional))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn select(
        &mut self,
        query: &SearchNeighborhood,
        point: Point3,
        geometry: &Anisotropy,
        candidates: &[Candidate],
        selected: &mut Vec<Observation>,
        variables: &mut [usize],
        sectors: &mut [usize; 8],
        cancel: &Cancellation,
    ) -> Result<()> {
        self.remaining.fill(0);
        self.bucket_of.clear();
        for candidate in candidates {
            cancel.check()?;
            let class = query.class(candidate.observation.site().variable).unwrap();
            let (_, sector) = query.metric(
                point,
                candidate.observation.site().support.center(),
                geometry,
            );
            let bucket = class * query.sectors() + sector;
            self.bucket_of.push(bucket);
            self.remaining[bucket] += 1;
            if selected.len() < query.maximum
                && variables[class] < query.variable_cap(class)
                && sectors[sector] < query.octant_maximum.unwrap_or(query.maximum)
            {
                selected.push(candidate.observation);
                variables[class] += 1;
                sectors[sector] += 1;
            }
        }
        // A full greedy result satisfying every minimum is already optimal:
        // it reaches the global cardinality bound, and every skipped earlier
        // candidate conflicts with an accepted prefix. Avoid constructing any
        // flow graph in this common case. Underfilled or infeasible greedy
        // selections still require the exact completion algorithm below.
        let minimums = query.minimums.as_ref().unwrap();
        if selected.len() == query.maximum
            && variables
                .iter()
                .enumerate()
                .all(|(class, &n)| n >= query.variable_minimum(class))
            && sectors
                .iter()
                .zip(minimums.octants)
                .all(|(&n, minimum)| n >= minimum)
        {
            return Ok(());
        }
        selected.clear();
        variables.fill(0);
        sectors.fill(0);
        let Some(target) = self.maximum(query, variables, sectors, 0, query.maximum, cancel)?
        else {
            return Err(Error::InsufficientNeighborhood);
        };
        for (i, candidate) in candidates.iter().enumerate() {
            if selected.len() == target {
                break;
            }
            cancel.check()?;
            let bucket = self.bucket_of[i];
            let class = bucket / query.sectors();
            let sector = bucket % query.sectors();
            self.remaining[bucket] -= 1;
            if variables[class] >= query.variable_cap(class)
                || sectors[sector] >= query.octant_maximum.unwrap_or(query.maximum)
            {
                continue;
            }
            variables[class] += 1;
            sectors[sector] += 1;
            if self.maximum(
                query,
                variables,
                sectors,
                selected.len() + 1,
                target,
                cancel,
            )? == Some(target)
            {
                selected.push(candidate.observation);
            } else {
                variables[class] -= 1;
                sectors[sector] -= 1;
            }
        }
        if selected.len() != target {
            return Err(Error::Numerical {
                context: "minimum-quota lexicographic completion",
            });
        }
        Ok(())
    }
}
