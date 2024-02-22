use nalgebra::UnitQuaternion;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub mod point_group;
pub mod volume_group;

pub enum SearchOrientation {
    Static,
    Dynamic,
}

pub trait NodeProvider {
    type Support: Send + Sync;

    fn n_groups(&self) -> usize;

    fn get_group(&self, group: usize) -> &[Self::Support];

    fn get_orientation(&self, group: usize) -> &UnitQuaternion<f32>;

    //reimplement this when MSRV 1.75
    // fn groups_and_orientations(
    //     &self,
    // ) -> impl ParallelIterator<Item = (&[Self::Support], UnitQuaternion<f32>)>;
}
