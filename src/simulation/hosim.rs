// use std::{collections::HashMap, fs::File, io::Write};

// use itertools::Itertools;
// use mathru::algebra::abstr::Polynomial;
// use rand::prelude::*;
// use rand::seq::SliceRandom;

// use crate::{
//     geometry::{ellipsoid::Ellipsoid, template::Template},
//     spatial_database::{
//         gridded_databases::{
//             self, complete_grid::CompleteGriddedDataBase,
//             gridded_data_base_query_engine_mut::GriddedDataBaseOctantQueryEngineMut,
//             incomplete_grid::InCompleteGriddedDataBase,
//         },
//         SpatialDataBase,
//     },
// };

// struct CDFSampler<'a> {
//     cdf: &'a [f32],
//     val: &'a [f32],
// }

// impl<'a> CDFSampler<'a> {
//     pub fn new(cdf: &'a [f32], val: &'a [f32]) -> Self {
//         assert!(!cdf.is_empty(), "CDF cannot be empty.");

//         assert!(
//             cdf.len() == val.len(),
//             "CDF and values must be of same length."
//         );
//         CDFSampler { cdf, val }
//     }

//     pub fn sample(&self) -> f32 {
//         let random_val = rand::random::<f32>();

//         let Some(index) = self.cdf.iter().position(|&cdf_val| cdf_val > random_val) else {
//             return self.cdf[self.cdf.len() - 1];
//         };

//         if index == 0 {
//             return self.val[0];
//         }

//         let fraction = (random_val - self.cdf[index - 1]) / (self.cdf[index] - self.cdf[index - 1]);

//         self.val[index - 1] + fraction * (self.val[index] - self.val[index - 1])
//     }
// }

// pub struct ReplicateDatabase {
//     replicates: HashMap<Template, Vec<Vec<f32>>>,
// }

// pub struct HOSIM {
//     pub training_image: CompleteGriddedDataBase<f32>,
//     pub conditioning_data: InCompleteGriddedDataBase<f32>,
//     pub legendre_polynomials: Vec<Polynomial<f32>>,
// }

// impl HOSIM {
//     pub fn new(
//         training_image: CompleteGriddedDataBase<f32>,
//         conditioning_data: InCompleteGriddedDataBase<f32>,
//         order: usize,
//     ) -> Self {
//         let legendre_polynomials = Self::compute_legendre_polynomials(order + 1);
//         Self {
//             training_image,
//             conditioning_data,
//             legendre_polynomials,
//         }
//     }

//     pub fn compute_legendre_polynomials(order: usize) -> Vec<Polynomial<f32>> {
//         let mut polys = vec![Polynomial::from_coef(vec![0.0]); order];
//         let n_f64 = order as f64;

//         polys[0] = Polynomial::from_coef(vec![1.0]);
//         polys[1] = Polynomial::from_coef(vec![1.0, 0.0]);
//         println!("{}", polys[1]);
//         for n in 2..order {
//             let p_1 = Polynomial::from_coef(vec![2.0f32 * n as f32 - 1.0f32, 0.0]);
//             let p_2 = Polynomial::from_coef(vec![n as f32 - 1.0f32]);
//             let p_3 = Polynomial::from_coef(vec![n as f32]);
//             polys[n] =
//                 ((p_1 * polys[n as usize - 1].clone() - p_2 * polys[n as usize - 2].clone()) / p_3)
//                     .0;
//         }

//         polys
//     }

//     pub fn simulate(
//         &mut self,
//         simulation_grid: &mut InCompleteGriddedDataBase<f32>,
//         template_search_ellipsoid: Ellipsoid,
//     ) {
//         //simulation path
//         let mut path = simulation_grid
//             .raw_grid
//             .grid
//             .indexed_iter()
//             .map(|(ind, _)| ind)
//             .collect::<Vec<_>>();

//         //shuffle path order
//         path.shuffle(&mut rand::thread_rng());

//         //create mutable query engine
//         let mut query_engine =
//             GriddedDataBaseOctantQueryEngineMut::new(template_search_ellipsoid, simulation_grid, 2);

//         //sequentially simulate each node in patch
//         let len = path.len();
//         for (i, ind) in path.into_iter().enumerate() {
//             println!("Simulating node {} of {}.", i, len);
//             // hard data already present -> go to next node in path
//             if query_engine.db.data_at_ind(&ind.into()).is_some() {
//                 continue;
//             }

//             // construct data event
//             let (data_event_inds, data_event_values) =
//                 query_engine.nearest_inds_and_values_to_ind(&ind.into());

//             //create template
//             let data_event_template = Template::new(
//                 data_event_inds
//                     .iter()
//                     .map(|t| {
//                         [
//                             t[0] as isize - ind.0 as isize,
//                             t[1] as isize - ind.1 as isize,
//                             t[2] as isize - ind.2 as isize,
//                         ]
//                     })
//                     .collect_vec(),
//             );

//             //find replicates in ti
//             let replicates = self.find_replicates_in_ti(&data_event_template);

//             //compute cdf
//             let cdf = self.compute_cdf(data_event_values.clone(), replicates.clone());

//             //compute ccdf
//             let ccdf = Self::compute_ccdf(cdf);
//             // println!("CCDF: {:?}", ccdf);

//             //sample
//             let vals = (0..100).map(|i| -1f32 + i as f32 / 50f32).collect_vec();
//             let ccdf = vals
//                 .iter()
//                 .map(|v| {
//                     ccdf.iter().enumerate().fold(0f32, |mut acc, (ind, val)| {
//                         acc += self.legendre_polynomials[ind].eval(*v) * val;
//                         acc
//                     })
//                 })
//                 .collect_vec();
//             let mut sampler = CDFSampler::new(ccdf.as_slice(), vals.as_slice());

//             query_engine
//                 .db
//                 .set_data_at_ind(&ind.into(), sampler.sample());

//             if i % 500 == 0 {
//                 //save values to file for visualization
//                 let (values, points) =
//                     gridded_databases::GriddedDataBaseInterface::data_and_points(query_engine.db);

//                 let mut out =
//                     File::create(format!("./test_results/hosim_intermediate_{i}.txt")).unwrap();
//                 let _ = out.write_all(b"surfs\n");
//                 let _ = out.write_all(b"4\n");
//                 let _ = out.write_all(b"x\n");
//                 let _ = out.write_all(b"y\n");
//                 let _ = out.write_all(b"z\n");
//                 let _ = out.write_all(b"value\n");

//                 for (point, value) in points.iter().zip(values.iter()) {
//                     //println!("point: {:?}, value: {}", point, value);
//                     let _ = out.write_all(
//                         format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes(),
//                     );
//                 }
//             }
//         }

//         //find
//     }

//     #[inline(always)]
//     fn find_replicates_in_ti(&self, template: &Template) -> Vec<Vec<f32>> {
//         self.training_image
//             .raw_grid
//             .grid
//             .indexed_iter()
//             .filter_map(|(ind, val)| template.replicate_at_ind(&self.training_image, &ind.into()))
//             .collect_vec()
//     }

//     #[inline(always)]
//     fn compute_cdf(&self, data_event: Vec<f32>, replicates: Vec<Vec<f32>>) -> Vec<f32> {
//         let mut c = vec![0.0; self.legendre_polynomials.len() - 1];
//         let mut x = vec![0.0; self.legendre_polynomials.len() - 1];

//         for replicate in replicates {
//             for (w, p) in self.legendre_polynomials[0..self.legendre_polynomials.len() - 1]
//                 .iter()
//                 .enumerate()
//             {
//                 x[w] = (w as f32 + 0.5) * p.eval(replicate[0]);
//             }

//             let mut x_prod = 1.0;

//             for i in 0..data_event.len() {
//                 let mut x_t = 0.0;
//                 for (w, p) in self.legendre_polynomials.iter().enumerate() {
//                     x_t += (w as f32 + 0.5) * p.eval(replicate[i + 1]) * p.eval(data_event[i]);
//                 }

//                 x_prod *= x_t;
//             }

//             for w in 0..self.legendre_polynomials.len() - 1 {
//                 x[w] *= x_prod;
//                 c[w] += x[w]
//             }
//         }

//         let denom = 2.0 * c[0];

//         for w in 0..self.legendre_polynomials.len() - 1 {
//             c[w] /= denom
//         }

//         c
//     }

//     #[inline(always)]
//     fn compute_ccdf(c: Vec<f32>) -> Vec<f32> {
//         let mut ccdf = vec![0.0; c.len() + 1];

//         for i in 1..c.len() {
//             ccdf[i + 1] += c[i] / (2.0 * i as f32 + 1.0);
//             ccdf[i - 1] -= c[i] / (2.0 * i as f32 + 1.0);
//         }
//         ccdf[0] += 0.5;
//         ccdf[1] += 0.5;
//         ccdf
//     }
// }

// #[cfg(test)]
// mod tests {
//     use std::{fs::File, io::Write};

//     use nalgebra::{Point3, UnitQuaternion};
//     use ndarray::Array3;
//     use num_traits::Float;

//     use crate::spatial_database::{
//         coordinate_system::{CoordinateSystem, GridSpacing},
//         gridded_databases::{self, GriddedDataBaseInterface},
//     };

//     use super::*;

//     #[test]
//     fn cdf_sampler() {
//         let cdf = vec![0.1, 0.5, 0.8, 0.9, 1.0];
//         let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//         let sampler = CDFSampler::new(&cdf, &vals);

//         for _ in 0..10 {
//             println!("Sampled value: {}", sampler.sample());
//         }
//     }

//     fn map_range(from_range: (f32, f32), to_range: (f32, f32), s: f32) -> f32 {
//         to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
//     }

//     #[test]
//     fn hosim_test() {
//         // Define the coordinate system for the grid
//         // origing at x = 0, y = 0, z = 0
//         // azimuth = 0, dip = 0, plunge = 0
//         let coordinate_system = CoordinateSystem::new(
//             Point3::new(0.0, 0.0, 0.0).into(),
//             UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians()),
//         );

//         // create a training image from a csv file (walker lake)
//         let mut ti_gdb = CompleteGriddedDataBase::<f32>::from_csv_index(
//             "./data/Exhaustive_set.csv",
//             "X",
//             "Y",
//             "Z",
//             "V",
//             GridSpacing {
//                 x: 1.0,
//                 y: 1.0,
//                 z: 1.0,
//             },
//             coordinate_system,
//         )
//         .expect("Failed to create gdb");

//         let min = ti_gdb
//             .raw_grid
//             .grid
//             .iter()
//             .min_by(|a, b| a.partial_cmp(b).unwrap())
//             .unwrap()
//             .clone();

//         let max = ti_gdb
//             .raw_grid
//             .grid
//             .iter()
//             .max_by(|a, b| a.partial_cmp(b).unwrap())
//             .unwrap()
//             .clone();

//         //map data to [-1, 1]
//         ti_gdb
//             .raw_grid
//             .grid
//             .mapv_inplace(|v| map_range((min, max), (-1.0, 1.0), v));

//         // create a conditioning gridded database from a csv file (walker lake)
//         let mut conditioning_gdb = InCompleteGriddedDataBase::from_csv_index(
//             "./data/walker.csv",
//             "X",
//             "Y",
//             "Z",
//             "V",
//             GridSpacing {
//                 x: 1.0,
//                 y: 1.0,
//                 z: 1.0,
//             },
//             coordinate_system,
//         )
//         .expect("Failed to create gdb");

//         //map data to [-1, 1]
//         conditioning_gdb.raw_grid.grid.mapv_inplace(|v| {
//             if let Some(val) = v {
//                 return Some(map_range((min, max), (-1.0, 1.0), val));
//             } else {
//                 None
//             }
//         });

//         // create a grid to store the simulation values
//         let sim_grid_arr = Array3::<Option<f32>>::from_elem(ti_gdb.shape(), None);
//         let mut sim_db = InCompleteGriddedDataBase::new(
//             sim_grid_arr,
//             ti_gdb.grid_spacing().clone(),
//             ti_gdb.coordinate_system().clone(),
//         );

//         // create search ellipsoid
//         let search_ellipsoid = Ellipsoid::new(450.0, 150.0, 1.0, coordinate_system.clone());

//         //create HOSIM instance
//         let mut hosim = HOSIM::new(ti_gdb, conditioning_gdb, 8);
//         hosim.simulate(&mut sim_db, search_ellipsoid);

//         //save values to file for visualization
//         let (values, points) =
//             gridded_databases::GriddedDataBaseInterface::data_and_points(&sim_db);

//         let mut out = File::create("./test_results/hosim.txt").unwrap();
//         let _ = out.write_all(b"surfs\n");
//         let _ = out.write_all(b"4\n");
//         let _ = out.write_all(b"x\n");
//         let _ = out.write_all(b"y\n");
//         let _ = out.write_all(b"z\n");
//         let _ = out.write_all(b"value\n");

//         for (point, value) in points.iter().zip(values.iter()) {
//             //println!("point: {:?}, value: {}", point, value);
//             let _ = out
//                 .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
//         }
//     }
// }
