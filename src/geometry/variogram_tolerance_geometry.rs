use nalgebra::{distance, Point3, Translation3, UnitDualQuaternion};

use parry3d::bounding_volume::Aabb;

use crate::spatial_database::coordinate_system::CoordinateSystem;

/// Variogram tolerance geometry used to identify valid point pairs for variogram computation.
pub struct VariogramToleranceGeometry {
    pub coordinate_system: CoordinateSystem,
    pub current_location: Point3<f32>,
    pub lag_tolerance_backwards: f32,
    pub lag_tolerance_forwards: f32,
    pub bandwidth_vertical: f32,
    pub bandwidth_horizontal: f32,
    pub azimuth_tolerance: f32,
    pub dip_tolerance: f32,
    pub current_lag: f32,
}

impl VariogramToleranceGeometry {
    /// Create a new VariogramToleranceGeometry with the given parameters.
    /// # Arguments
    /// * `coordinate_system` - The coordinate system of the geometry (Location and orientation).
    /// * `lag_tolerance_backwards` - The maximum distance backwards along the current lag vector.
    /// * `lag_tolerance_forwards` - The maximum distance forwards along the current lag vector.
    /// * `bandwidth_vertical` - The maximum vertical distance from the current location.
    /// * `bandwidth_horizontal` - The maximum horizontal distance from the current location.
    /// * `azimuth_tolerance` - The maximum azimuthal angle from the current lag vector.
    /// * `dip_tolerance` - The maximum dip angle from the current lag vector.
    /// * `current_lag` - The current lag distance.
    pub fn new(
        coordinate_system: CoordinateSystem,
        lag_tolerance_backwards: f32,
        lag_tolerance_forwards: f32,
        bandwidth_vertical: f32,
        bandwidth_horizontal: f32,
        azimuth_tolerance: f32,
        dip_tolerance: f32,
    ) -> Self {
        Self {
            coordinate_system,
            current_location: coordinate_system.origin(),
            lag_tolerance_backwards,
            lag_tolerance_forwards,
            bandwidth_vertical,
            bandwidth_horizontal,
            azimuth_tolerance,
            dip_tolerance,
            current_lag: 0f32,
        }
    }

    /// Bounding box in world coordinates.
    pub fn bounding_box(&self) -> Aabb {
        fn max_rotation_within_threshold(dist: f32, angle_threshold: f32, y_theshold: f32) -> f32 {
            let angle = (y_theshold / dist).asin();
            angle.min(angle_threshold)
        }

        //radius between origin and current location
        let radius = distance(&self.current_location, &self.coordinate_system.origin())
            - self.lag_tolerance_backwards;

        //compute how much to move lag tolerance backwards due to radially dependent azimuth and dip tolerances
        let delta = if radius > 0f32 {
            let azimuth_tolerance = self.azimuth_tolerance.to_radians();
            let dip_tolerance = self.dip_tolerance.to_radians();

            let max_rotation_h =
                max_rotation_within_threshold(radius, azimuth_tolerance, self.bandwidth_horizontal);

            let max_rotation_v =
                max_rotation_within_threshold(radius, dip_tolerance, self.bandwidth_vertical);

            let delta_x_h = radius - radius * max_rotation_h.cos();
            let delta_x_v = radius - radius * max_rotation_v.cos();
            delta_x_h.max(delta_x_v)
        } else {
            0f32
        };

        let min = Point3::new(
            -self.lag_tolerance_backwards - delta,
            -self.bandwidth_horizontal,
            -self.bandwidth_vertical,
        );

        let max = Point3::new(
            self.lag_tolerance_forwards,
            self.bandwidth_horizontal,
            self.bandwidth_vertical,
        );

        let dual = UnitDualQuaternion::from_parts(
            self.current_location.into(),
            self.coordinate_system.rotation,
        );

        let bbox = Aabb::new(min, max);
        bbox.transform_by(&dual.into())

        //bounding_box.transform(&dual)
    }

    /// Translate the current location by the given translation.
    pub fn translate(&mut self, translation: &Translation3<f32>) {
        self.current_location = translation.transform_point(&self.current_location);
    }

    /// Translate the current location by the given step_size along the lag vector.
    pub fn step(&mut self, step_size: f32) {
        self.current_lag += step_size;
        let translation = Translation3::from(
            self.coordinate_system
                .rotation
                .transform_point(&Point3::new(step_size, 0.0, 0.0)),
        );
        self.translate(&translation);
    }

    /// Reset the current location to the origin of the coordinate system.
    pub fn reset_with_new_coordinate_system(&mut self, coordinate_system: CoordinateSystem) {
        self.coordinate_system = coordinate_system;
        self.current_location = coordinate_system.origin();
        self.current_lag = 0f32;
    }

    /// Check if the given point (world coordinates) is within the tolerance geometry.
    pub fn contains(&self, point: &Point3<f32>) -> bool {
        fn max_rotation_within_threshold(dist: f32, angle_threshold: f32, y_theshold: f32) -> f32 {
            let angle = (y_theshold / dist).asin();
            angle.min(angle_threshold)
        }

        //get distance between origin and point
        let dist_point = distance(&point, &self.coordinate_system.origin());

        //check if point is within lag tolerance
        let delta = dist_point - self.current_lag;
        if delta > self.lag_tolerance_forwards || delta < -self.lag_tolerance_backwards {
            return false;
        }

        let max_h_angle = max_rotation_within_threshold(
            dist_point,
            self.azimuth_tolerance,
            self.bandwidth_horizontal,
        );

        let max_v_angle =
            max_rotation_within_threshold(dist_point, self.dip_tolerance, self.bandwidth_vertical);

        let max_h_dist = dist_point * max_h_angle.sin();
        let max_v_dist = dist_point * max_v_angle.sin();

        //check if point is within ellipse
        let local_point = self.coordinate_system.global_to_local(&point);
        if local_point.x < 0f32
            || (local_point.y / max_h_dist).powi(2) + (local_point.z / max_v_dist).powi(2)
                > max_h_angle
        {
            return false;
        }
        true
    }
}

//TODO
#[cfg(test)]
mod tests {}
