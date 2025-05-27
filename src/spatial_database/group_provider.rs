use std::ops::Range;

use rstar::{RTree, RTreeObject, AABB};

use crate::geometry::support::Support;

use super::SpatialData;

pub struct GroupProvider {
    supports: Vec<Support>,
    original_idxs: Vec<usize>,
    group_idxs: Vec<usize>,
}

impl GroupProvider {
    pub fn optimized_groups(
        supports: &[Support],
        dx: f64,
        dy: f64,
        dz: f64,
        gx: usize,
        gy: usize,
        gz: usize,
    ) -> Self {
        let mut target_point_tree = RTree::bulk_load(
            supports
                .iter()
                .copied()
                .enumerate()
                .map(|(i, support)| SpatialData {
                    support,
                    data_idx: i as u32,
                })
                .collect(),
        );

        let bounds = target_point_tree.root().envelope();

        let group_size = [gx, gy, gz];
        let env_size = [
            group_size[0] as f64 * dx,
            group_size[1] as f64 * dy,
            group_size[2] as f64 * dz,
        ];
        let mut groups = Vec::new();
        let mut original_idxs = Vec::new();
        let mut x = bounds.lower()[0];
        while x <= bounds.upper()[0] {
            let mut y = bounds.lower()[1];
            while y <= bounds.upper()[1] {
                let mut z = bounds.lower()[2];
                while z <= bounds.upper()[2] {
                    let envelope = AABB::from_corners(
                        [x, y, z],
                        [x + env_size[0], y + env_size[1], z + env_size[2]],
                    );

                    let mut points = target_point_tree
                        .drain_in_envelope_intersecting(envelope)
                        .collect::<Vec<_>>();

                    //sort points by x, y, z
                    points.sort_by(|a, b| {
                        a.envelope()
                            .lower()
                            .partial_cmp(&b.envelope().lower())
                            .unwrap()
                    });

                    points
                        .chunks(group_size.iter().product())
                        .for_each(|chunk| {
                            groups.push(chunk.iter().map(|geom| geom.support).collect::<Vec<_>>());
                            original_idxs.push(
                                chunk
                                    .iter()
                                    .map(|geo| geo.data_idx as usize)
                                    .collect::<Vec<_>>(),
                            );
                        });

                    z += env_size[2];
                }
                y += env_size[1]
            }
            x += env_size[0]
        }

        let mut idxs = vec![0];
        let mut cnt = 0;
        groups.iter().for_each(|group| {
            cnt += group.len();
            idxs.push(cnt);
        });
        let flat_groups = groups.into_iter().flatten().collect::<Vec<_>>();
        let flat_idxs = original_idxs.into_iter().flatten().collect::<Vec<_>>();

        Self {
            supports: flat_groups,
            original_idxs: flat_idxs,
            group_idxs: idxs,
        }
    }

    #[inline(always)]
    pub fn get_supports(&self) -> &[Support] {
        &self.supports
    }

    #[inline(always)]
    pub fn n_nodes(&self) -> usize {
        self.supports.len()
    }

    #[inline(always)]
    pub fn n_groups(&self) -> usize {
        self.group_idxs.len().checked_sub(1).unwrap_or(0)
    }

    #[inline(always)]
    #[track_caller]
    pub fn get_group_range(&self, group: usize) -> Range<usize> {
        let low = self.group_idxs[group];
        let high = self.group_idxs[group + 1];
        low..high
    }

    #[inline(always)]
    #[track_caller]
    pub fn get_group(&self, group: usize) -> &[Support] {
        &self.supports[self.get_group_range(group)]
    }

    #[inline(always)]
    #[track_caller]
    pub fn get_original_idxs(&self, group: usize) -> &[usize] {
        &self.original_idxs[self.get_group_range(group)]
    }

    #[inline(always)]
    pub fn max_group_size(&self) -> usize {
        self.group_idxs
            .windows(2)
            .map(|window| window[1] - window[0])
            .max()
            .unwrap_or(0)
    }
}
