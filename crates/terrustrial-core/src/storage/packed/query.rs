use super::*;
use crate::spatial::{
    nearest::Location,
    paged::{Frontier, Pending},
    SearchWorkspace,
};
use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

enum Selection<'a, 'q> {
    Nearest(Point3, &'a mut NearestWorkspace),
    Bounds(Bounds, &'a mut dyn FnMut(&Observation) -> Result<()>),
    Search(Bounds, &'a mut SearchWorkspace<'q>),
}
impl Selection<'_, '_> {
    fn distance(&self, bounds: Bounds) -> Option<f64> {
        match self {
            Self::Nearest(p, w) => {
                let d = bounds.distance_squared(p.coordinates());
                (d <= w.cutoff_squared()).then_some(d)
            }
            Self::Bounds(b, _) | Self::Search(b, _) => bounds.intersects(*b).then_some(0.),
        }
    }
    fn offer(&mut self, o: &Observation, distance: f64) -> Result<()> {
        match self {
            Self::Nearest(_, w) => {
                w.offer_distance(o, distance);
                Ok(())
            }
            Self::Bounds(_, visitor) => visitor(o),
            Self::Search(_, workspace) => workspace.offer(o),
        }
    }
}

struct Visit<'a> {
    squared: [f64; MAX_LEAF],
    counts: [u64; 2],
    pages: PageView<'a>,
}
impl Snapshot {
    fn walk(
        &self,
        node: Node,
        depth: usize,
        selection: &mut Selection<'_, '_>,
        cancel: &Cancellation,
        visit: &mut Visit<'_>,
        frontier: &mut Option<&mut Frontier>,
    ) -> Result<()> {
        cancel.check()?;
        if node.first_rank > self.manifest.visible || selection.distance(node.bounds).is_none() {
            return Ok(());
        }
        if depth >= 64 {
            return Err(invalid("packed hierarchy exceeds depth bound"));
        }
        let loaded;
        let geometry = if let Some(page) = self.manifest.geometry.directory.get(&node.id) {
            page
        } else {
            loaded = visit
                .pages
                .page(node.id, self.manifest.generation, false, cancel)?;
            &loaded.page
        };
        match geometry {
            Page::Branch(children) => {
                let mut ordered = [(f64::INFINITY, 0usize); FANOUT];
                let mut count = 0;
                for (i, child) in children.iter().enumerate() {
                    if child.id >= node.id {
                        return Err(invalid("packed hierarchy is not acyclic"));
                    }
                    if child.first_rank <= self.manifest.visible {
                        if let Some(d) = selection.distance(child.bounds) {
                            if frontier.as_mut().is_none_or(|f| {
                                !f.push(Pending {
                                    distance: d,
                                    node: *child,
                                    depth: depth + 1,
                                })
                            }) {
                                ordered[count] = (d, i);
                                count += 1;
                            }
                        }
                    }
                }
                ordered[..count].sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                for (_, i) in &ordered[..count] {
                    self.walk(children[*i], depth + 1, selection, cancel, visit, &mut None)?;
                }
            }
            Page::Leaf(leaf) => {
                if let Selection::Nearest(p, _) = selection {
                    let level = fearless_simd::Level::new();
                    fearless_simd::dispatch!(level, simd => distances(simd, leaf, p.coordinates(), &mut visit.squared));
                }
                let mut payload: Option<Arc<Cached>> = None;
                visit.counts[0] += leaf.ranges.len() as u64;
                for (group, &(first, last)) in leaf.ranges.iter().enumerate() {
                    cancel.check()?;
                    let center = std::array::from_fn(|axis| leaf.coordinates[axis][group]);
                    let inside = match selection {
                        Selection::Nearest(_, w) => visit.squared[group] <= w.cutoff_squared(),
                        Selection::Bounds(b, _) | Selection::Search(b, _) => {
                            b.intersects(Bounds::point(center))
                        }
                    };
                    if !inside
                        || !leaf.ranks[first..last]
                            .iter()
                            .any(|rank| *rank <= self.manifest.visible)
                    {
                        continue;
                    }
                    if let Selection::Nearest(p, workspace) = selection {
                        let distance = if visit.squared[group].is_normal() {
                            visit.squared[group].sqrt()
                        } else {
                            crate::data::norm3(std::array::from_fn(|axis| {
                                center[axis] - p.coordinates()[axis]
                            }))
                        };
                        workspace.offer_colocated(
                            distance,
                            (first..last)
                                .filter(|&slot| leaf.ranks[slot] <= self.manifest.visible)
                                .map(|slot| {
                                    visit.counts[1] += 1;
                                    (
                                        leaf.ids[slot],
                                        Location {
                                            page: node.id,
                                            slot,
                                            count: leaf.ranks.len(),
                                            center,
                                        },
                                    )
                                }),
                        );
                        continue;
                    }
                    if payload.is_none() {
                        payload = Some(visit.pages.page(
                            node.id,
                            self.manifest.generation,
                            true,
                            cancel,
                        )?);
                    }
                    let Page::Payload(observations) = &payload.as_ref().unwrap().page else {
                        return Err(invalid("packed payload kind"));
                    };
                    if observations.len() != leaf.ranks.len() {
                        return Err(invalid("packed geometry/payload count mismatch"));
                    }
                    // All attributes at one support center share this exact norm.
                    let distance = match selection {
                        Selection::Nearest(p, _) => {
                            if visit.squared[group].is_normal() {
                                visit.squared[group].sqrt()
                            } else {
                                p.distance(observations.observation(first)?.site().support.center())
                            }
                        }
                        Selection::Bounds(_, _) | Selection::Search(_, _) => 0.,
                    };
                    for i in first..last {
                        let observation = observations.observation(i)?;
                        if observation.site().support.center().coordinates() != center {
                            return Err(invalid("packed geometry/payload mismatch"));
                        }
                        if leaf.ranks[i] <= self.manifest.visible {
                            visit.counts[1] += 1;
                            selection.offer(&observation, distance)?;
                        }
                    }
                }
            }
            Page::Payload(_) => return Err(invalid("packed geometry kind")),
        }
        Ok(())
    }
    fn select<'a>(
        &'a self,
        selection: &mut Selection<'_, '_>,
        cancel: &Cancellation,
        frontier: Option<&mut Frontier>,
    ) -> Result<PageView<'a>> {
        cancel.check()?;
        let mut ordered = [(f64::INFINITY, 0usize); MAX_ROOTS];
        let mut count = 0;
        for (i, node) in self.manifest.geometry.roots.iter().enumerate() {
            if node.first_rank <= self.manifest.visible {
                if let Some(d) = selection.distance(node.bounds) {
                    ordered[count] = (d, i);
                    count += 1;
                }
            }
        }
        ordered[..count].sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut visit = Visit {
            squared: [0.; MAX_LEAF],
            counts: [0; 2],
            pages: PageView::new(&self.inner),
        };
        if let Some(frontier) = frontier {
            frontier.clear();
            for &(distance, i) in &ordered[..count] {
                assert!(frontier.push(Pending {
                    distance,
                    node: self.manifest.geometry.roots[i],
                    depth: 0
                }));
            }
            while let Some(pending) = frontier.pop() {
                if selection.distance(pending.node.bounds).is_none() {
                    continue;
                }
                self.walk(
                    pending.node,
                    pending.depth,
                    selection,
                    cancel,
                    &mut visit,
                    &mut Some(frontier),
                )?;
            }
        } else {
            for (_, i) in &ordered[..count] {
                self.walk(
                    self.manifest.geometry.roots[*i],
                    0,
                    selection,
                    cancel,
                    &mut visit,
                    &mut None,
                )?;
            }
        }
        self.inner
            .counters
            .groups
            .fetch_add(visit.counts[0], Ordering::Relaxed);
        self.inner
            .counters
            .observations
            .fetch_add(visit.counts[1], Ordering::Relaxed);
        Ok(visit.pages)
    }
}
impl ObservationSource for Snapshot {
    fn deferred_page_reads(&self) -> bool {
        true
    }
    fn search<'a>(
        &self,
        point: Point3,
        workspace: &'a mut SearchWorkspace<'_>,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.search_indexed(point, cancel, |lower, upper, workspace| {
            self.select(
                &mut Selection::Search(Bounds { lower, upper }, workspace),
                cancel,
                None,
            )
            .map(|_| ())
        })
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.visit_bounds(
            [f64::NEG_INFINITY; 3],
            [f64::INFINITY; 3],
            &self.inner.uncancelled,
            visitor,
        )
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        if (0..3).any(|i| lower[i].is_nan() || upper[i].is_nan() || lower[i] > upper[i]) {
            return Err(invalid("packed query bounds"));
        }
        self.select(
            &mut Selection::Bounds(Bounds { lower, upper }, visitor),
            cancel,
            None,
        )
        .map(|_| ())
    }
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.reset(point);
        let mut frontier = workspace.packed_frontier.take().unwrap();
        let result = self.select(
            &mut Selection::Nearest(point, workspace),
            cancel,
            Some(&mut frontier),
        );
        workspace.packed_frontier = Some(frontier);
        let mut pages = result?;
        let mut loaded: Option<(u64, Arc<Cached>)> = None;
        workspace.resolve_locations(|location| {
            cancel.check()?;
            if loaded.as_ref().is_none_or(|(id, _)| *id != location.page) {
                loaded = Some((
                    location.page,
                    pages.page(location.page, self.manifest.generation, true, cancel)?,
                ));
            }
            let Page::Payload(observations) = &loaded.as_ref().unwrap().1.page else {
                return Err(invalid("packed payload kind"));
            };
            if observations.len() != location.count {
                return Err(invalid("packed geometry/payload count mismatch"));
            }
            observations.observation(location.slot)
        })?;
        workspace.finish(cancel)
    }
    fn nearest(
        &self,
        point: Point3,
        query: Neighborhood,
        cancel: &Cancellation,
    ) -> Result<Vec<Observation>> {
        let mut workspace = self.nearest_workspace(query)?;
        self.nearest_into(point, &mut workspace, cancel)?;
        Ok(workspace.into_selected())
    }
    fn at_site(&self, site: &crate::data::Site) -> Result<Vec<Observation>> {
        self.at_site_with_center_status(site).map(|(found, _)| found)
    }
    fn at_site_with_center_status(
        &self,
        site: &crate::data::Site,
    ) -> Result<(Vec<Observation>, bool)> {
        let p = site.support.center().coordinates();
        let mut found = Vec::new();
        let mut center_empty = true;
        // The returned Vec becomes caller-owned. Account for its construction,
        // including old/new buffers overlapping during growth.
        // Empty exact-site queries need no allocation or shared-budget update.
        let mut reservation: Option<Reservation> = None;
        self.visit_bounds(p, p, &self.inner.uncancelled, &mut |o| {
            center_empty &= o.site().support.center().coordinates() != p;
            if o.site() == site {
                if found.len() == found.capacity() {
                    let capacity = found
                        .capacity()
                        .max(4)
                        .checked_mul(2)
                        .ok_or(Error::SizeOverflow)?;
                    let peak = capacity
                        .checked_add(found.capacity())
                        .and_then(|n| n.checked_mul(std::mem::size_of::<Observation>()))
                        .ok_or(Error::SizeOverflow)?;
                    let peak = u64::try_from(peak).map_err(|_| Error::SizeOverflow)?;
                    if let Some(reservation) = &mut reservation {
                        self.inner.resize_work(reservation, peak)?;
                    } else {
                        reservation = Some(self.inner.reserve_work(peak)?);
                    }
                    found
                        .try_reserve_exact(capacity - found.len())
                        .map_err(|_| invalid("packed exact-site output allocation"))?;
                    reservation
                        .as_mut()
                        .expect("admitted exact-site output")
                        .resize((found.capacity() * std::mem::size_of::<Observation>()) as u64)?;
                }
                found.push(*o);
            }
            Ok(())
        })?;
        found.sort_unstable_by_key(Observation::id);
        Ok((found, center_empty))
    }
}

#[simd]
fn distances<S: Simd>(simd: S, leaf: &Leaf, p: [f64; 3], out: &mut [f64; MAX_LEAF]) {
    let lanes = S::f64s::LEN;
    let n = leaf.ranges.len();
    let full = n / lanes * lanes;
    for first in (0..full).step_by(lanes) {
        let d: [S::f64s; 3] = std::array::from_fn(|axis| {
            S::f64s::from_fn(simd, |i| leaf.coordinates[axis][first + i])
                - S::f64s::splat(simd, p[axis])
        });
        let squared = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        squared.store_slice(&mut out[first..first + lanes]);
    }
    for (i, out) in out.iter_mut().enumerate().take(n).skip(full) {
        let d: [f64; 3] = std::array::from_fn(|axis| leaf.coordinates[axis][i] - p[axis]);
        *out = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frontier_full_rejection_and_repeated_refill_preserve_order() {
        let mut f = Frontier::new().unwrap();
        let pending = |i: usize| Pending {
            distance: i as f64,
            depth: i % 64,
            node: Node {
                id: i as u64 + 1,
                bounds: Bounds::point([i as f64; 3]),
                first_rank: 0,
                records: 1,
            },
        };
        for _ in 0..8 {
            for i in (0..128).rev() {
                assert!(f.push(pending(i)));
            }
            assert!(!f.push(pending(129)));
            for i in 0..64 {
                assert_eq!(f.pop().unwrap().node.id, i as u64 + 1);
            }
            for i in (0..64).rev() {
                assert!(f.push(pending(i)));
            }
            for i in 0..128 {
                let p = f.pop().unwrap();
                assert_eq!(p.distance, i as f64);
                assert_eq!(p.node.id, i as u64 + 1);
                assert_eq!(p.depth, i % 64);
            }
            assert!(f.pop().is_none());
        }
    }
}
