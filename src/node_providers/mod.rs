use nalgebra::{Point3, UnitQuaternion};
use rayon::iter::ParallelIterator;
use rstar::{primitives::GeomWithData, RTree, AABB};

use crate::geometry::ellipsoid::Ellipsoid;

pub mod point_group;
pub mod volume_group;

pub enum SearchOrientation {
    Static,
    Dynamic,
}

pub trait ToPoints {
    fn to_points(&self) -> &[Point3<f32>];
}

impl ToPoints for Vec<Point3<f32>> {
    fn to_points(&self) -> &[Point3<f32>] {
        self.as_slice()
    }
}

impl ToPoints for Point3<f32> {
    fn to_points(&self) -> &[Point3<f32>] {
        std::slice::from_ref(self)
    }
}

#[derive(Copy, Clone, Debug, Hash)]
pub struct GroupRange {
    start: u32,
    end: u32,
}
/// Interface for a conditioning node provider for kriging.
///
/// Support is the data associated with the node.
///    Point support and block support NodeProviders are provided.
///    Point support may be used for point-point kriging
///    Block support is more flexible and may be used for point-block, block-block, and block-point kriging.
///
pub trait NodeProvider {
    type Support: ToPoints + Send + Sync;

    // The number of nodes in the provider.
    fn n_nodes(&self) -> usize;

    // The number of groups of nodes.
    // Groups are not necessarily the same size.
    fn n_groups(&self) -> usize;

    // The data associated with the group.
    fn get_group(&self, group: usize) -> &[Self::Support];

    // The orientation of the group.
    fn get_orientation(&self, group: usize) -> &UnitQuaternion<f32>;

    fn groups_and_orientations(
        &self,
    ) -> impl ParallelIterator<Item = (&[Self::Support], UnitQuaternion<f32>)>;

    fn indexed_groups_and_orientations(
        &self,
    ) -> impl ParallelIterator<Item = (usize, &[Self::Support], UnitQuaternion<f32>)>;
    fn randomize(&mut self, rng: &mut impl rand::Rng);

    fn get_group_range(&self, group: usize) -> GroupRange;
}

pub struct OrderedProvider {
    tree: RTree<GeomWithData<[f32; 3], u32>>,
}

impl OrderedProvider {
    pub fn new(provider: &impl NodeProvider) -> Self {
        let mut tree_data = Vec::with_capacity(provider.n_nodes());

        for i in 0..provider.n_nodes() {
            for point in provider
                .get_group(i)
                .iter()
                .map(|g| g.to_points())
                .flatten()
            {
                tree_data.push(GeomWithData::new([point.x, point.y, point.z], i as u32));
            }
        }

        let tree = RTree::bulk_load(tree_data);
        Self { tree }
    }

    pub fn fetch_preds(
        &self,
        ellipse: &Ellipsoid,
        filter: impl Fn(u32) -> bool,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        let min = ellipse.bounding_box().mins.into();
        let max = ellipse.bounding_box().maxs.into();
        let aabb = AABB::from_corners(min, max);
        for point in self.tree.locate_in_envelope(&aabb) {
            if filter(point.data)
                && ellipse.contains(&Point3::new(
                    point.geom()[0],
                    point.geom()[1],
                    point.geom()[2],
                ))
            {
                out.push(point.data);
            }
        }
    }
}
