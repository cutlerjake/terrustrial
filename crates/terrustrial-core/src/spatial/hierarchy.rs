//! Compact immutable resident hierarchy, with bounds also used by disk pages.
//! Median splits on the widest centroid axis bound depth even for coincident input.
//! Query traversal uses only the call stack; records and nodes are contiguous.
use crate::{error::invalid, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub lower: [f64; 3],
    pub upper: [f64; 3],
}
impl Bounds {
    pub fn point(p: [f64; 3]) -> Self {
        Self { lower: p, upper: p }
    }
    pub fn include(&mut self, rhs: Self) {
        for a in 0..3 {
            self.lower[a] = self.lower[a].min(rhs.lower[a]);
            self.upper[a] = self.upper[a].max(rhs.upper[a]);
        }
    }
    pub fn intersects(self, rhs: Self) -> bool {
        (0..3).all(|a| self.lower[a] <= rhs.upper[a] && self.upper[a] >= rhs.lower[a])
    }
    pub fn contains(self, rhs: Self) -> bool {
        (0..3).all(|a| self.lower[a] <= rhs.lower[a] && self.upper[a] >= rhs.upper[a])
    }
    pub fn distance_squared(self, p: [f64; 3]) -> f64 {
        let d = self.delta(p);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }
    pub fn delta(self, p: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| (self.lower[a] - p[a]).max(p[a] - self.upper[a]).max(0.))
    }

    fn center(self, axis: usize) -> f64 {
        self.lower[axis] * 0.5 + self.upper[axis] * 0.5
    }
}
pub trait Bounded {
    fn bounds(&self) -> Bounds;
}
impl Bounded for [f64; 3] {
    fn bounds(&self) -> Bounds {
        Bounds::point(*self)
    }
}

const LEAF_CAPACITY: usize = 8;

#[derive(Clone)]
struct Node {
    bounds: Bounds,
    start: usize,
    end: usize,
    right: usize,
}
#[derive(Clone)]
pub struct Hierarchy<T> {
    items: Vec<T>,
    nodes: Vec<Node>,
}
#[derive(Clone, Copy)]
struct BuildKey {
    center: [f64; 3],
    source: usize,
}
impl<T: Bounded> Hierarchy<T> {
    pub fn new(mut items: Vec<T>) -> Result<Self> {
        let mut keys = Vec::new();
        keys.try_reserve_exact(items.len())
            .map_err(|_| invalid("hierarchy construction scratch allocation"))?;
        for (source, item) in items.iter().enumerate() {
            let b = item.bounds();
            if (0..3).any(|a| {
                !b.lower[a].is_finite() || !b.upper[a].is_finite() || b.lower[a] > b.upper[a]
            }) {
                return Err(invalid("nonfinite or inverted hierarchy bounds"));
            }
            keys.push(BuildKey {
                center: std::array::from_fn(|a| b.center(a)),
                source,
            });
        }
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(Self::node_capacity(items.len())?)
            .map_err(|_| invalid("hierarchy allocation"))?;
        if !items.is_empty() {
            Self::build(&mut keys, &items, 0, &mut nodes);
        }
        // Apply destination-to-source cycles once. Sorting never repeatedly
        // moves full observation payloads, and queries retain contiguous leaves.
        for first in 0..keys.len() {
            let mut current = first;
            loop {
                let next = keys[current].source;
                keys[current].source = current;
                if next == first {
                    break;
                }
                items.swap(current, next);
                current = next;
            }
        }
        Ok(Self { items, nodes })
    }
    fn build(keys: &mut [BuildKey], items: &[T], start: usize, nodes: &mut Vec<Node>) -> usize {
        let index = nodes.len();
        nodes.push(Node {
            bounds: Bounds::point([0.; 3]),
            start,
            end: start + keys.len(),
            right: 0,
        });
        let bounds = if keys.len() <= LEAF_CAPACITY {
            let mut bounds = items[keys[0].source].bounds();
            for key in &keys[1..] {
                bounds.include(items[key.source].bounds());
            }
            bounds
        } else {
            let mut centers = Bounds::point(keys[0].center);
            for key in &keys[1..] {
                centers.include(Bounds::point(key.center));
            }
            let axis = (0..3)
                .max_by(|&a, &b| {
                    (centers.upper[a] - centers.lower[a])
                        .total_cmp(&(centers.upper[b] - centers.lower[b]))
                })
                .unwrap();
            let mid = keys.len() / 2;
            keys.select_nth_unstable_by(mid, |a, b| a.center[axis].total_cmp(&b.center[axis]));
            let (left, right) = keys.split_at_mut(mid);
            let left = Self::build(left, items, start, nodes);
            let right = Self::build(right, items, start + mid, nodes);
            nodes[index].right = right;
            let mut bounds = nodes[left].bounds;
            bounds.include(nodes[right].bounds);
            bounds
        };
        nodes[index].bounds = bounds;
        index
    }
    /// Temporary construction storage in addition to retained nodes and input.
    pub fn scratch_bytes(count: usize) -> Result<usize> {
        count
            .checked_mul(std::mem::size_of::<BuildKey>())
            .ok_or(Error::SizeOverflow)
    }
    pub fn allocated_bytes(&self) -> usize {
        self.items.capacity() * std::mem::size_of::<T>()
            + self.nodes.capacity() * std::mem::size_of::<Node>()
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }
    pub fn into_items(self) -> Vec<T> {
        self.items
    }
    pub fn bounds(&self) -> Option<Bounds> {
        self.nodes.first().map(|n| n.bounds)
    }
    /// Retained node capacity upper bound, excluding the caller's input vector.
    fn node_capacity(count: usize) -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        count
            .div_ceil(LEAF_CAPACITY)
            .checked_next_power_of_two()
            .and_then(|n| n.checked_mul(2))
            .map(|n| n - 1)
            .ok_or(Error::SizeOverflow)
    }
    pub fn node_bytes(count: usize) -> Result<usize> {
        Self::node_capacity(count)?
            .checked_mul(std::mem::size_of::<Node>())
            .ok_or(Error::SizeOverflow)
    }
    pub fn visit_bounds(
        &self,
        bounds: Bounds,
        mut visitor: impl FnMut(&T) -> Result<()>,
    ) -> Result<()> {
        fn walk<T: Bounded>(
            tree: &Hierarchy<T>,
            index: usize,
            bounds: Bounds,
            visitor: &mut impl FnMut(&T) -> Result<()>,
        ) -> Result<()> {
            let node = &tree.nodes[index];
            if !node.bounds.intersects(bounds) {
                return Ok(());
            }
            if node.right == 0 || bounds.contains(node.bounds) {
                for item in &tree.items[node.start..node.end] {
                    if item.bounds().intersects(bounds) {
                        visitor(item)?;
                    }
                }
            } else {
                walk(tree, index + 1, bounds, visitor)?;
                walk(tree, node.right, bounds, visitor)?;
            }
            Ok(())
        }
        if !self.is_empty() {
            walk(self, 0, bounds, &mut visitor)?;
        }
        Ok(())
    }
    /// Near-first exact traversal. `bound` returns None only when the whole
    /// envelope can be excluded; it may tighten as `visitor` updates state.
    pub fn traverse<S>(
        &self,
        state: &mut S,
        bound: impl Fn(&S, Bounds) -> Option<f64>,
        mut visitor: impl FnMut(&mut S, &T, f64) -> Result<()>,
    ) -> Result<()> {
        self.traverse_leaves(state, &bound, |state, items| {
            for item in items {
                if let Some(distance) = bound(state, item.bounds()) {
                    visitor(state, item, distance)?;
                }
            }
            Ok(())
        })
    }
    /// Near-first traversal of contiguous leaves. Callers apply their exact item
    /// predicate, allowing point kernels to avoid generic box-distance arithmetic.
    pub(crate) fn traverse_leaves<S>(
        &self,
        state: &mut S,
        bound: impl Fn(&S, Bounds) -> Option<f64>,
        mut visitor: impl FnMut(&mut S, &[T]) -> Result<()>,
    ) -> Result<()> {
        if self.bounds().is_some_and(|b| bound(state, b).is_some()) {
            walk(self, 0, state, &bound, &mut visitor)?;
        }
        Ok(())
    }
}

/// Allocation-free closed-box iterator with a stack bounded by machine word size.
pub struct Range<'a, T> {
    tree: &'a Hierarchy<T>,
    bounds: Bounds,
    stack: [usize; usize::BITS as usize],
    depth: usize,
    current: std::slice::Iter<'a, T>,
}
impl<T: Bounded> Hierarchy<T> {
    pub fn query(&self, bounds: Bounds) -> Range<'_, T> {
        let stack = [0; usize::BITS as usize];
        Range {
            tree: self,
            bounds,
            stack,
            depth: usize::from(!self.is_empty()),
            current: self.items[..0].iter(),
        }
    }
}
impl<'a, T: Bounded> Iterator for Range<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            for item in self.current.by_ref() {
                if item.bounds().intersects(self.bounds) {
                    return Some(item);
                }
            }
            if self.depth == 0 {
                return None;
            }
            self.depth -= 1;
            let i = self.stack[self.depth];
            let n = &self.tree.nodes[i];
            if !n.bounds.intersects(self.bounds) {
                continue;
            }
            if n.right == 0 {
                self.current = self.tree.items[n.start..n.end].iter();
            } else {
                self.stack[self.depth] = n.right;
                self.stack[self.depth + 1] = i + 1;
                self.depth += 2;
            }
        }
    }
}
impl Bounds {
    pub fn from_corners(lower: [f64; 3], upper: [f64; 3]) -> Self {
        Self { lower, upper }
    }
    pub fn contains_point(self, p: &[f64; 3]) -> bool {
        self.contains(Self::point(*p))
    }
}

/// Exact squared distance from an item to a point. It must be no smaller than
/// the distance to its bounds. Used by the legacy streaming nearest iterator.
pub trait PointDistance: Bounded {
    fn distance_squared(&self, point: [f64; 3]) -> f64;
}
#[derive(Clone, Copy)]
struct Pending {
    distance: f64,
    index: usize,
    item: bool,
}
impl PartialEq for Pending {
    fn eq(&self, rhs: &Self) -> bool {
        self.cmp(rhs).is_eq()
    }
}
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Pending {
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        rhs.distance
            .total_cmp(&self.distance)
            .then(rhs.item.cmp(&self.item))
            .then(rhs.index.cmp(&self.index))
    }
}
/// Lazy nearest iterator. This convenience API owns a growing frontier; bounded
/// production neighborhood queries use `traverse` and caller-owned workspaces.
pub struct Nearest<'a, T> {
    tree: &'a Hierarchy<T>,
    point: [f64; 3],
    queue: std::collections::BinaryHeap<Pending>,
}
impl<T: PointDistance> Hierarchy<T> {
    pub fn nearest_iter(&self, point: [f64; 3]) -> Nearest<'_, T> {
        let mut queue = std::collections::BinaryHeap::new();
        if let Some(b) = self.bounds() {
            queue.push(Pending {
                distance: b.distance_squared(point),
                index: 0,
                item: false,
            });
        }
        Nearest {
            tree: self,
            point,
            queue,
        }
    }
}
impl<'a, T: PointDistance> Iterator for Nearest<'a, T> {
    type Item = (&'a T, f64);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(p) = self.queue.pop() {
            if p.item {
                return Some((&self.tree.items[p.index], p.distance));
            }
            let n = &self.tree.nodes[p.index];
            if n.right == 0 {
                for index in n.start..n.end {
                    self.queue.push(Pending {
                        distance: self.tree.items[index].distance_squared(self.point),
                        index,
                        item: true,
                    });
                }
            } else {
                for index in [p.index + 1, n.right] {
                    self.queue.push(Pending {
                        distance: self.tree.nodes[index].bounds.distance_squared(self.point),
                        index,
                        item: false,
                    });
                }
            }
        }
        None
    }
}

fn walk<T: Bounded, S, F: Fn(&S, Bounds) -> Option<f64>, V: FnMut(&mut S, &[T]) -> Result<()>>(
    tree: &Hierarchy<T>,
    index: usize,
    state: &mut S,
    bound: &F,
    visitor: &mut V,
) -> Result<()> {
    let node = &tree.nodes[index];
    if node.right == 0 {
        visitor(state, &tree.items[node.start..node.end])?;
    } else {
        let left = index + 1;
        let right = node.right;
        match (
            bound(state, tree.nodes[left].bounds),
            bound(state, tree.nodes[right].bounds),
        ) {
            (Some(a), Some(b)) => {
                let (first, second) = if a <= b { (left, right) } else { (right, left) };
                walk(tree, first, state, bound, visitor)?;
                if bound(state, tree.nodes[second].bounds).is_some() {
                    walk(tree, second, state, bound, visitor)?;
                }
            }
            (Some(_), None) => walk(tree, left, state, bound, visitor)?,
            (None, Some(_)) => walk(tree, right, state, bound, visitor)?,
            _ => {}
        }
    }
    Ok(())
}
