use ultraviolet::{DIsometry3, DRotor3, DVec3};

#[derive(Clone, Copy, Debug)]
pub struct NewCoordinateSystem {
    into_local: DIsometry3,
    into_global: DIsometry3,
}

impl NewCoordinateSystem {
    pub fn new(origin: DVec3, rotation: DRotor3) -> Self {
        let into_global = DIsometry3::new(origin, rotation);
        let into_local = into_global.inversed();

        Self {
            into_local,
            into_global,
        }
    }

    #[inline(always)]
    pub fn into_local(&self) -> &DIsometry3 {
        &self.into_local
    }

    #[inline(always)]
    pub fn into_global(&self) -> &DIsometry3 {
        &self.into_global
    }

    #[inline(always)]
    pub fn set_origin(&mut self, origin: DVec3) {
        self.into_global.translation = origin;
        self.into_local = self.into_global.inversed();
    }

    #[inline(always)]
    pub fn set_rotation(&mut self, rotation: DRotor3) {
        self.into_global.rotation = rotation;
        self.into_local = self.into_global.inversed();
    }
}

/// Octant of a point
pub fn octant(point: &DVec3) -> u8 {
    match (point.x >= 0.0, point.y >= 0.0, point.z >= 0.0) {
        (true, true, true) => 0,
        (false, true, true) => 1,
        (false, false, true) => 2,
        (true, false, true) => 3,
        (true, true, false) => 4,
        (false, true, false) => 5,
        (false, false, false) => 6,
        (true, false, false) => 7,
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use approx::assert_relative_eq;
    // use std::f32::consts::PI;
    // #[test]
    // fn no_offset_no_rotation() {
    //     let origin = Point3::new(0.0, 0.0, 0.0);
    //     let roll = 0.0;
    //     let pitch = 0.0;
    //     let yaw = 0.0;
    //     let coordinate_system =
    //         CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
    //     let point = Point3::new(1.0, 1.0, 1.0);
    //     let transformed_point = coordinate_system.world_to_local_point(&point);
    //     assert_relative_eq!(transformed_point, point, epsilon = 1e-6);
    // }

    // #[test]
    // fn offset_no_rotation() {
    //     let origin = Point3::new(1.0, 1.0, 1.0);
    //     let roll = 0.0;
    //     let pitch = 0.0;
    //     let yaw = 0.0;
    //     let coordinate_system =
    //         CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
    //     let point = Point3::new(2.0, 2.0, 2.0);
    //     let transformed_point = coordinate_system.world_to_local_point(&point);
    //     assert_relative_eq!(
    //         transformed_point,
    //         Point3::new(1.0, 1.0, 1.0),
    //         epsilon = 1e-6
    //     );
    // }

    // #[test]
    // fn no_offset_rotation() {
    //     let origin = Point3::new(0.0, 0.0, 0.0);
    //     let roll = 0.0;
    //     let pitch = 0.0;
    //     let yaw = PI / 2.0;
    //     let coordinate_system =
    //         CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
    //     let point = Point3::new(1.0, 1.0, 0.0);
    //     let transformed_point = coordinate_system.world_to_local_point(&point);
    //     assert_relative_eq!(
    //         transformed_point,
    //         Point3::new(1.0, -1.0, 0.0),
    //         epsilon = 1e-6
    //     );
    // }

    // #[test]
    // fn offset_1d_rotation() {
    //     let origin = Point3::new(1.0, 1.0, 1.0);
    //     let roll = 0.0;
    //     let pitch = 0.0;
    //     let yaw = PI / 2.0;
    //     let coordinate_system =
    //         CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
    //     let point = Point3::new(2.0, 1.0, 1.0);
    //     let transformed_point = coordinate_system.world_to_local_point(&point);
    //     assert_relative_eq!(
    //         transformed_point,
    //         Point3::new(0f32, -1f32, 0f32),
    //         epsilon = 1e-6
    //     );
    // }

    // //test 3d roation where point is not at origin
    // #[test]
    // fn offset_3d_rotation() {
    //     let origin = Point3::new(1.0, 2.0, 3.0);
    //     let roll = PI / 3.0;
    //     let pitch = PI / 4.0;
    //     let yaw = PI / 2.0;
    //     let coordinate_system =
    //         CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
    //     let point = Point3::new(2.0, 1.0, 1.0);
    //     let transformed_point = coordinate_system.world_to_local_point(&point);
    //     assert_relative_eq!(
    //         transformed_point,
    //         Point3::new(0.7071068, -2.3371172, -0.19463491),
    //         epsilon = 1e-6
    //     );
    // }

    // #[test]
    // fn back_offset_rotation() {
    //     let origin = Point3::new(1.0, 2.0, 3.0);
    //     let roll = PI / 3.0;
    //     let pitch = PI / 4.0;
    //     let yaw = PI / 2.0;
    //     let coordinate_system =
    //         CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
    //     let point = Point3::new(1.3660254, 4.38013939, 2.55171226);
    //     let transformed_point = coordinate_system.world_to_local_point(&point);
    //     let back_transformed_point = coordinate_system.local_world_point(&transformed_point);
    //     assert_relative_eq!(point, back_transformed_point, epsilon = 1e-6);
    // }

    use num_traits::Float;
    use ultraviolet::{DRotor3, DVec3};

    use crate::spatial_database::coordinate_system::NewCoordinateSystem;

    #[test]
    fn new_cs() {
        let origin = DVec3::new(10.0, 10.0, 10.0);
        let rotation = DRotor3::from_euler_angles(45.0.to_radians(), 0.0, 0.0);

        let mut cs = NewCoordinateSystem::new(DVec3::zero(), rotation);
        cs.set_origin(origin);

        let global_point = DVec3::new(11.0, 11.0, 11.0);
        let local_point = cs.into_local().transform_vec(global_point);

        println!("{:?}, {:?}", global_point, local_point);

        println!("{:?}", cs.into_global().transform_vec(local_point));
    }
}
