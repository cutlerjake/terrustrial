use nalgebra::{Point3, UnitQuaternion};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::NodeProvider;

pub struct PointGroupProvider {
    points: Vec<Point3<f32>>,
    orientations: Vec<UnitQuaternion<f32>>,
    group_inds: Vec<usize>,
}

impl PointGroupProvider {
    pub fn get_group(&self, group: usize) -> &[Point3<f32>] {
        let start = self.group_inds[group];
        let end = if group == self.group_inds.len() - 1 {
            self.points.len()
        } else {
            self.group_inds[group + 1]
        };

        &self.points[start..end]
    }

    pub fn from_groups(
        groups: Vec<Vec<Point3<f32>>>,
        orientations: Vec<UnitQuaternion<f32>>,
    ) -> Self {
        let mut points = Vec::new();
        let mut group_inds = Vec::new();

        for group in groups {
            let start = points.len();
            points.extend(group);
            group_inds.push(start);
        }

        Self {
            points,
            orientations,
            group_inds,
        }
    }
}

impl NodeProvider for PointGroupProvider {
    type Support = Point3<f32>;

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

    // fn groups_and_orientations(
    //     &self,
    // ) -> impl ParallelIterator<Item = (&[Self::Support], UnitQuaternion<f32>)> {
    //     (0..self.group_inds.len())
    //         .into_par_iter()
    //         .map(|group| (self.get_group(group), self.orientations[group]))
    // }
}

#[cfg(test)]
mod test {
    use nalgebra::UnitQuaternion;

    use super::*;

    #[test]
    fn point_group_provider() {
        let groups = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
            vec![Point3::new(2.0, 2.0, 2.0), Point3::new(3.0, 3.0, 3.0)],
        ];

        let orientations = vec![
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
        ];

        let provider = PointGroupProvider::from_groups(groups, orientations);

        let group = provider.get_group(0);
        assert_eq!(group.len(), 2);
        assert_eq!(group[0], Point3::new(0.0, 0.0, 0.0));
        assert_eq!(group[1], Point3::new(1.0, 1.0, 1.0));

        let group = provider.get_group(1);
        assert_eq!(group.len(), 2);
        assert_eq!(group[0], Point3::new(2.0, 2.0, 2.0));
        assert_eq!(group[1], Point3::new(3.0, 3.0, 3.0));
    }
}
