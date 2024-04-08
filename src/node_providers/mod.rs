use nalgebra::UnitQuaternion;

pub mod point_group;
pub mod volume_group;

pub enum SearchOrientation {
    Static,
    Dynamic,
}

/// Interface for a conditioning node provider for kriging.
/// Support is the data associated with the node.
///    Point support and block support NodeProviders are provided.
///    Point support may be used for point-point kriging
///    Block support is more flexible and may be used for point-block, block-block, and block-point kriging.
///
pub trait NodeProvider {
    type Support: Send + Sync;

    // The number of groups of nodes.
    // Groups are not necessarily the same size.
    fn n_groups(&self) -> usize;

    // The data associated with the group.
    fn get_group(&self, group: usize) -> &[Self::Support];

    // The orientation of the group.
    fn get_orientation(&self, group: usize) -> &UnitQuaternion<f32>;

    //reimplement this when MSRV 1.75
    // fn groups_and_orientations(
    //     &self,
    // ) -> impl ParallelIterator<Item = (&[Self::Support], UnitQuaternion<f32>)>;
}
