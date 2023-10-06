// use crate::{
//     geometry::tolerance::ToleranceGeometry, spatial_database::coordinate_system::CoordinateSystem,
//     spatial_database::SpatialDataBase,
// };
// use nalgebra::UnitQuaternion;

// pub struct VariogramLagParamters {
//     pub lag: f32,
//     pub lag_tolerance: f32,
//     pub nlags: u32,
// }

// impl VariogramLagParamters {
//     pub fn new(lag: f32, lag_tolerance: f32, nlags: u32) -> Self {
//         Self {
//             lag,
//             lag_tolerance,
//             nlags,
//         }
//     }
// }

// pub struct VariogramBandWidthParamters {
//     pub horizontal: f32,
//     pub vertical: f32,
// }

// impl VariogramBandWidthParamters {
//     pub fn new(x: f32, y: f32) -> Self {
//         Self {
//             horizontal: x,
//             vertical: y,
//         }
//     }
// }

// pub struct VariogramToleranceParamters {
//     pub azimuth: f32,
//     pub dip: f32,
//     pub plunge: f32,
// }

// impl VariogramToleranceParamters {
//     pub fn new(azimuth: f32, dip: f32, plunge: f32) -> Self {
//         Self {
//             azimuth,
//             dip,
//             plunge,
//         }
//     }
// }

// pub struct ExperimentalVariogramParameters {
//     rotation: UnitQuaternion<f32>,
//     lag: VariogramLagParamters,
//     tolerance: VariogramToleranceParamters,
//     bandwidth: VariogramBandWidthParamters,
// }

// impl ExperimentalVariogramParameters {
//     pub fn new(
//         rotation: UnitQuaternion<f32>,
//         lag: VariogramLagParamters,
//         tolerance: VariogramToleranceParamters,
//         bandwidth: VariogramBandWidthParamters,
//     ) -> Self {
//         Self {
//             rotation,
//             lag,
//             tolerance,
//             bandwidth,
//         }
//     }

//     pub fn from_euler_angles(
//         azimuth: f32,
//         dip: f32,
//         plunge: f32,
//         lag: VariogramLagParamters,
//         tolerance: VariogramToleranceParamters,
//         bandwidth: VariogramBandWidthParamters,
//     ) -> Self {
//         let rotation = UnitQuaternion::from_euler_angles(azimuth, dip, plunge);
//         Self::new(rotation, lag, tolerance, bandwidth)
//     }
// }

// pub trait VariogramType {
//     type DATA;
//     fn new(lags: usize) -> Self;
//     fn update(&mut self, value_1: &Self::DATA, value_2: &Self::DATA, lag: usize);
//     fn values(&mut self) -> Vec<Self::DATA>;
//     fn counts(&self) -> Vec<u32>;
// }

// pub struct Direct<T> {
//     values: Vec<T>,
//     counts: Vec<u32>,
// }

// impl<T> VariogramType for Direct<T>
// where
//     T: num_traits::Float + num_traits::NumAssign,
// {
//     type DATA = T;

//     fn new(lags: usize) -> Self {
//         let values = vec![T::zero(); lags];
//         let counts = vec![0; lags];
//         Self { values, counts }
//     }

//     fn update(&mut self, value_1: &Self::DATA, value_2: &Self::DATA, lag: usize) {
//         self.values[lag] += (*value_1 - *value_2).powi(2);
//         self.counts[lag] += 1;
//     }

//     fn values(&mut self) -> Vec<Self::DATA> {
//         self.values
//             .iter()
//             .zip(self.counts.iter())
//             .map(|(v, c)| {
//                 let div = T::from(*c).unwrap();
//                 if div.is_zero() {
//                     T::zero()
//                 } else {
//                     *v / div
//                 }
//             })
//             .collect()
//     }

//     fn counts(&self) -> Vec<u32> {
//         self.counts.clone()
//     }
// }
// pub struct ExperimentalVariogram<T>
// where
//     T: VariogramType,
// {
//     pub parameters: ExperimentalVariogramParameters,
//     pub values: Vec<T::DATA>,
//     pub counts: Vec<u32>,
//     vgram: T,
// }

// impl<T> ExperimentalVariogram<T>
// where
//     T: VariogramType,
// {
//     pub fn new(parameters: ExperimentalVariogramParameters) -> Self {
//         let vgram = T::new(parameters.lag.nlags as usize);
//         Self {
//             parameters,
//             values: Vec::new(),
//             counts: Vec::new(),
//             vgram,
//         }
//     }

//     pub fn compute<S>(&mut self, database: &S)
//     where
//         S: SpatialDataBase<T::DATA>,
//     {
//         //get all pairs of points
//         let (values, points) = database.data_and_points();

//         //create a variogram tolerance geometry
//         let mut vgram_tolerance_geometry = ToleranceGeometry::new(
//             CoordinateSystem::default(),
//             self.parameters.lag.lag_tolerance,
//             self.parameters.lag.lag_tolerance,
//             self.parameters.bandwidth.vertical,
//             self.parameters.bandwidth.horizontal,
//             self.parameters.tolerance.azimuth,
//             self.parameters.tolerance.dip,
//         );

//         for (point, point_value) in points.into_iter().zip(values) {
//             //coordinate system for point
//             let point_cs = CoordinateSystem::new(point.into(), self.parameters.rotation);

//             //set coordinate system for variogram tolerance geometry
//             vgram_tolerance_geometry.reset_with_new_coordinate_system(point_cs);

//             //for each lag
//             for lag in 0..self.parameters.lag.nlags {
//                 //println!("__________Lag Step__________");
//                 //get bounding box
//                 let lag_bounding_box = vgram_tolerance_geometry.bounding_box();

//                 //get points in bounding box
//                 let inds_in_bounding_box = database.inds_in_bounding_box(&lag_bounding_box);
//                 let points_in_bounding_box = inds_in_bounding_box
//                     .iter()
//                     .map(|i| database.point_at_ind(i));
//                 let values_ind_in_bounding_box =
//                     inds_in_bounding_box.iter().map(|i| database.data_at_ind(i));

//                 for (lag_point, lag_value) in points_in_bounding_box.zip(values_ind_in_bounding_box)
//                 {
//                     //check if point is within lag tolerance geometry
//                     if lag_value.is_none() || !vgram_tolerance_geometry.contains(&lag_point) {
//                         continue;
//                     }

//                     self.vgram
//                         .update(&lag_value.unwrap(), &point_value, lag as usize);
//                 }

//                 //step variogram tolerance geometry
//                 vgram_tolerance_geometry.step(self.parameters.lag.lag);
//             }
//         }

//         self.values = self.vgram.values();
//         self.counts = self.vgram.counts();
//     }
// }
