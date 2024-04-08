use nalgebra::{Unit, UnitVector3, Vector3};

pub mod declustering;
pub mod estimators;
pub mod geometry;
pub mod node_providers;
pub mod spatial_database;
pub mod systems;
pub mod variography;

pub mod re_export {
    pub use nalgebra;
    pub use parry3d;
    pub use simba;
}

pub const FORWARD: UnitVector3<f32> = Unit::new_unchecked(Vector3::new(0.0, 1.0, 0.0));
pub const RIGHT: UnitVector3<f32> = Unit::new_unchecked(Vector3::new(1.0, 0.0, 0.0));
pub const UP: UnitVector3<f32> = Unit::new_unchecked(Vector3::new(0.0, 0.0, 1.0));
