use std::fmt::Debug;

use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;

use crate::geometry::ellipsoid::Ellipsoid;

use self::gridded_databases::GriddedDataBaseInterface;

pub mod coordinate_system;
pub mod gridded_databases;
pub mod normalized;
pub mod qbvh;
pub mod rtree_point_set;
pub mod zero_mean;

pub trait PointProvider {
    fn points(&self) -> &[Point3<f32>];
}

pub trait SpatialQueryable<T, G> {
    fn query(&self, point: &Point3<f32>) -> (Vec<T>, Vec<Point3<f32>>);
    fn geometry(&self) -> &G;
}

pub trait ConditioningProvider<G, T, P> {
    type Shape;
    fn query(
        &self,
        point: &Point3<f32>,
        ellipsoid: &G,
        params: &P,
    ) -> (Vec<usize>, Vec<T>, Vec<Self::Shape>);

    fn points(&self) -> &[Point3<f32>];
    fn data(&self) -> &[T];
    fn data_mut(&mut self) -> &mut [T];
}

pub trait SpatialDataBase<T> {
    type INDEX: Debug;
    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX>;
    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32>;
    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T>;
    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>);
    fn data_and_inds(&self) -> (Vec<T>, Vec<Self::INDEX>);
    fn set_data_at_ind(&mut self, ind: &Self::INDEX, data: T);
}

macro_rules! impl_spatial_database_for_grid {
    ($( ($impl_type:ty, $data_type:ty) ),*) => {
        $(
            impl SpatialDataBase<$data_type> for $impl_type {
                type INDEX = [usize; 3];

                fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX> {
                    <Self as GriddedDataBaseInterface<$data_type>>::inds_in_bounding_box(self, bounding_box)
                }

                fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32> {
                    <Self as GriddedDataBaseInterface<$data_type>>::ind_to_point(self, &inds.map(|i| i as isize))
                }

                fn data_at_ind(&self, ind: &Self::INDEX) -> Option<$data_type> {
                    <Self as GriddedDataBaseInterface<$data_type>>::data_at_ind(self, ind)
                }

                fn data_and_points(&self) -> (Vec<$data_type>, Vec<Point3<f32>>) {
                    <Self as GriddedDataBaseInterface<$data_type>>::data_and_points(self)
                }

                fn data_and_inds(&self) -> (Vec<$data_type>, Vec<Self::INDEX>) {
                    <Self as GriddedDataBaseInterface<$data_type>>::data_and_inds(self)
                }

                fn set_data_at_ind(&mut self, ind: &Self::INDEX, data: $data_type) {
                    <Self as GriddedDataBaseInterface<$data_type>>::set_data_at_ind(self, ind, data)
                }
            }
        )*
    };
}

// impl_spatial_database_for_grid!(
//     (
//         gridded_databases::incomplete_grid::InCompleteGriddedDataBase<f32, f32>,
//         f32
//     ),
//     (
//         gridded_databases::incomplete_grid::InCompleteGriddedDataBase<f64, f32>,
//         f64
//     ),
//     (
//         gridded_databases::complete_grid::CompleteGriddedDataBase<f32, f32>,
//         f32
//     ),
//     (
//         gridded_databases::complete_grid::CompleteGriddedDataBase<f64, f32>,
//         f64
//     )
// );

pub trait DiscretiveVolume {
    fn discretize(&self, dx: f32, dy: f32, dz: f32) -> Vec<Point3<f32>>;
}

impl DiscretiveVolume for Aabb {
    fn discretize(&self, dx: f32, dy: f32, dz: f32) -> Vec<Point3<f32>> {
        //ceil gaurantees that the resulting discretization will have dimensions upperbounded by dx, dy, dz
        let nx = ((self.maxs.x - self.mins.x) / dx).ceil() as usize;
        let ny = ((self.maxs.y - self.mins.y) / dy).ceil() as usize;
        let nz = ((self.maxs.z - self.mins.z) / dz).ceil() as usize;

        //step size in each direction
        let step_x = (self.maxs.x - self.mins.x) / (nx as f32);
        let step_y = (self.maxs.y - self.mins.y) / (ny as f32);
        let step_z = (self.maxs.z - self.mins.z) / (nz as f32);

        //contains the discretized points
        let mut points = Vec::new();

        let mut x = self.mins.x + step_x / 2.0;
        while x <= self.maxs.x {
            let mut y = self.mins.y + step_y / 2.0;
            while y <= self.maxs.y {
                let mut z = self.mins.z + step_z / 2.0;
                while z <= self.maxs.z {
                    points.push(Point3::new(x, y, z));
                    z += step_z;
                }
                y += step_y;
            }
            x += step_x;
        }

        points
    }
}
