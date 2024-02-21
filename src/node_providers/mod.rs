use nalgebra::UnitQuaternion;
use rayon::iter::ParallelIterator;

pub mod point_group;
pub mod volume_group;

pub enum SearchOrientation {
    Static,
    Dynamic,
}

pub trait NodeProvider {
    type Support;

    fn groups_and_orientations(
        &self,
    ) -> impl ParallelIterator<Item = (&[Self::Support], UnitQuaternion<f32>)>;
}
