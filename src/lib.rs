use ultraviolet::DVec3;

pub mod declustering;
pub mod geometry;
pub mod group_operators;
pub mod spatial_database;
pub mod systems;
pub mod variography;

pub mod prelude {

    pub mod re_exports {
        pub use rstar;
        pub use ultraviolet;
    }

    pub use crate::group_operators::{
        generalized_sequential_indicator_kriging::estimate as gsik_estimate,
        generalized_sequential_kriging::{estimate as gsk_estimate, simulate as gsk_simulate},
        inverse_distance::estimate as id_estimate,
    };
}

pub const FORWARD: DVec3 = DVec3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};
pub const RIGHT: DVec3 = DVec3 {
    x: 1.0,
    y: 0.0,
    z: 0.0,
};
pub const UP: DVec3 = DVec3 {
    x: 0.0,
    y: 0.0,
    z: 1.0,
};
