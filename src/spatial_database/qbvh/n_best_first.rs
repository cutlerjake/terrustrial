use std::collections::BinaryHeap;

use nalgebra::SimdBool;
use parry3d::{
    bounding_volume::SimdAabb,
    math::{Real, SIMD_WIDTH},
    partitioning::{NodeIndex, Qbvh, SimdBestFirstVisitStatus},
};

use crate::spatial_database::qbvh::weighted_value::WeightedValue;
use nalgebra::RealField;

use super::simd_n_best_first_visitor::SimdNBestFirstVisitor;

macro_rules! array(
    ($callback: expr; SIMD_WIDTH) => {
        {
            #[inline(always)]
            #[allow(dead_code)]
            fn create_arr<T>(mut callback: impl FnMut(usize) -> T) -> [T; SIMD_WIDTH] {
                [callback(0usize), callback(1usize), callback(2usize), callback(3usize)]
            }

            create_arr($callback)
        }
    }
);

pub trait NBestFirst<LeafData> {
    fn traverse_n_best_first<BFS>(&self, visitor: &mut BFS) -> Option<(NodeIndex, BFS::Result)>
    where
        BFS: SimdNBestFirstVisitor<u32, SimdAabb>,
        BFS::Result: Clone; // Because we cannot move out of an array…

    fn traverse_n_best_first_node<BFS>(
        &self,
        visitor: &mut BFS,
        start_node: u32,
        init_cost: Real,
    ) -> Option<(NodeIndex, BFS::Result)>
    where
        BFS: SimdNBestFirstVisitor<LeafData, SimdAabb>,
        BFS::Result: Clone; // B
}

impl NBestFirst<u32> for Qbvh<u32> {
    /// Performs a N-best-first-search on the BVH.
    ///
    /// Returns the content of the leaf with the smallest associated cost, and a result of
    /// user-defined type.
    fn traverse_n_best_first<BFS>(&self, visitor: &mut BFS) -> Option<(NodeIndex, BFS::Result)>
    where
        BFS: SimdNBestFirstVisitor<u32, SimdAabb>,
        BFS::Result: Clone, // Because we cannot move out of an array…
    {
        self.traverse_n_best_first_node(visitor, 0, Real::max_value().unwrap())
    }

    /// Performs a N-best-first-search on the BVH, starting at the given node.
    ///
    /// Returns the content of the leaf with the smallest associated cost, and a result of
    /// user-defined type.
    fn traverse_n_best_first_node<BFS>(
        &self,
        visitor: &mut BFS,
        start_node: u32,
        init_cost: Real,
    ) -> Option<(NodeIndex, BFS::Result)>
    where
        BFS: SimdNBestFirstVisitor<u32, SimdAabb>,
        BFS::Result: Clone, // Because we cannot move out of an array…
    {
        if self.raw_nodes().is_empty() {
            return None;
        }

        let mut queue: BinaryHeap<WeightedValue<u32>> = BinaryHeap::new();

        let mut best_cost = init_cost;
        let mut best_result = None;
        queue.push(WeightedValue::new(start_node, -best_cost / 2.0));

        while let Some(entry) = queue.pop() {
            if -entry.cost >= best_cost {
                // No BV left in the tree that has a lower cost than best_result
                break; // Solution found.
            }

            let node = &self.raw_nodes()[entry.value as usize];
            let leaf_data = if node.is_leaf() {
                Some(
                    array![|ii| Some(&self.raw_proxies().get(node.children[ii] as usize)?.data); SIMD_WIDTH],
                )
            } else {
                None
            };

            match visitor.visit(&mut best_cost, &node.simd_aabb, leaf_data) {
                SimdBestFirstVisitStatus::ExitEarly(result) => {
                    return result.map(|r| (node.parent, r)).or(best_result);
                }
                SimdBestFirstVisitStatus::MaybeContinue {
                    weights,
                    mask,
                    results,
                } => {
                    let bitmask = mask.bitmask();
                    let weights: [Real; SIMD_WIDTH] = weights.into();

                    for ii in 0..SIMD_WIDTH {
                        if (bitmask & (1 << ii)) != 0 {
                            if node.is_leaf() {
                                if weights[ii] < best_cost && results[ii].is_some() {
                                    // We found a leaf!
                                    if let Some(proxy) =
                                        self.raw_proxies().get(node.children[ii] as usize)
                                    {
                                        //best_cost = weights[ii];
                                        best_result =
                                            Some((proxy.node, results[ii].clone().unwrap()))
                                    }
                                }
                            } else {
                                // Internal node, visit the child.
                                // Un fortunately, we have this check because invalid Aabbs
                                // return a hit as well.
                                if (node.children[ii] as usize) < self.raw_nodes().len() {
                                    queue.push(WeightedValue::new(node.children[ii], -weights[ii]));
                                }
                            }
                        }
                    }
                }
            }
        }

        best_result
    }
}
