use nalgebra::{Point3, UnitQuaternion};
use rand::seq::SliceRandom;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::{GroupRange, NodeProvider};

/// A node provider for a group of volumes.
/// Suitable for block-point, block-block, and point-block kriging.
pub struct VolumeGroupProvider {
    volumes: Vec<Vec<Point3<f32>>>,
    group_inds: Vec<GroupRange>,
    orientations: Vec<UnitQuaternion<f32>>,
}

impl VolumeGroupProvider {
    pub fn get_group(&self, group: usize) -> &[Vec<Point3<f32>>] {
        let GroupRange { start, end } = self.group_inds[group];
        &self.volumes[start as usize..end as usize]
    }

    pub fn from_groups(
        volumes: Vec<Vec<Vec<Point3<f32>>>>,
        orientations: Vec<UnitQuaternion<f32>>,
    ) -> Self {
        let mut group_inds = Vec::new();
        let mut volumes_flat = Vec::new();

        for volume in volumes {
            let start = volumes_flat.len();
            let end = start + volume.len();
            group_inds.push(GroupRange {
                start: start as u32,
                end: end as u32,
            });
            volumes_flat.extend(volume);
        }

        Self {
            volumes: volumes_flat,
            group_inds,
            orientations,
        }
    }
}

impl NodeProvider for VolumeGroupProvider {
    type Support = Vec<Point3<f32>>;

    #[inline(always)]
    fn n_nodes(&self) -> usize {
        self.volumes.len()
    }

    #[inline(always)]
    fn n_groups(&self) -> usize {
        self.group_inds.len()
    }

    #[inline(always)]
    fn get_group(&self, group: usize) -> &[Self::Support] {
        self.get_group(group)
    }

    #[inline(always)]
    fn get_orientation(&self, group: usize) -> &UnitQuaternion<f32> {
        &self.orientations[group]
    }

    fn groups_and_orientations(
        &self,
    ) -> impl ParallelIterator<Item = (&[Self::Support], UnitQuaternion<f32>)> {
        (0..self.group_inds.len())
            .into_par_iter()
            .map(|group| (self.get_group(group), self.orientations[group]))
    }

    fn indexed_groups_and_orientations(
        &self,
    ) -> impl ParallelIterator<Item = (usize, &[Self::Support], UnitQuaternion<f32>)> {
        (0..self.group_inds.len())
            .into_par_iter()
            .enumerate()
            .map(|(idx, group)| (idx, self.get_group(group), self.orientations[group]))
    }

    fn randomize(&mut self, rng: &mut impl rand::Rng) {
        self.volumes.shuffle(rng);
    }

    #[inline(always)]
    fn get_group_range(&self, group: usize) -> GroupRange {
        self.group_inds[group]
    }
}
