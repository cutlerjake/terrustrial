use std::marker::PhantomData;

use crate::{
    geometry::{ellipsoid::Ellipsoid, Geometry},
    kriging::KrigingParameters,
    spatial_database::{ConditioningProvider, SpatialQueryable},
    variography::model_variograms::VariogramModel,
};

use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_cholesky::llt::compute;
use faer_core::{mul::inner_prod::inner_prod_with_conj, Conj, Mat, MatRef, Parallelism};
use indicatif::ParallelProgressIterator;
use nalgebra::{Point3, SimdRealField, SimdValue, Vector3};
use num_traits::{Float, NumCast};
use rayon::prelude::*;
use simba::scalar::RealField;
use simba::simd::SimdPartialOrd;

use super::KrigingSystem;

pub struct ConditioningParams {
    pub max_n_cond: usize,
}

impl ConditioningParams {
    pub fn new(max_n_cond: usize) -> Self {
        Self { max_n_cond }
    }
}

pub struct SimpleKrigingSystem<T> {
    pub cond_cov_mat: Mat<f32>,
    pub krig_point_cov_vec: Mat<f32>,
    pub weights: Mat<f32>,
    pub values: Mat<f32>,
    pub c_0: f32,
    pub cholesky_compute_mem: GlobalMemBuffer,
    pub cholesky_solve_mem: GlobalMemBuffer,
    pub n_elems: usize,
    pub cond_x_buffer: T,
    pub cond_y_buffer: T,
    pub cond_z_buffer: T,
    pub cov_buffer: T,
    pub simd_vec_buffer: Vec<Vector3<T>>,
}

impl<T> Clone for SimpleKrigingSystem<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let n_elems = self.n_elems;
        Self {
            cond_cov_mat: self.cond_cov_mat.clone(),
            krig_point_cov_vec: self.krig_point_cov_vec.clone(),
            weights: self.weights.clone(),
            values: self.values.clone(),
            c_0: self.c_0,
            cholesky_compute_mem: GlobalMemBuffer::new(
                faer_cholesky::llt::compute::cholesky_in_place_req::<f32>(
                    n_elems,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            ),
            cholesky_solve_mem: GlobalMemBuffer::new(
                faer_cholesky::llt::solve::solve_req::<f32>(n_elems, 1, Parallelism::None).unwrap(),
            ),
            n_elems,
            cond_x_buffer: self.cond_x_buffer.clone(),
            cond_y_buffer: self.cond_y_buffer.clone(),
            cond_z_buffer: self.cond_z_buffer.clone(),
            cov_buffer: self.cov_buffer.clone(),
            simd_vec_buffer: Vec::with_capacity(n_elems * (n_elems + 1) / (2 * 16) + 1),
        }
    }
}

impl<T> SimpleKrigingSystem<T>
where
    T: SimdValue<Element = f32> + SimdRealField + Copy,
{
    /// Create a new simple kriging system
    /// # Arguments
    /// * `n_elems` - The maximum number of elements in the system
    /// # Returns
    /// * `Self` - The new simple kriging system
    /// * Requires zero mean data
    pub fn new(n_elems: usize) -> Self {
        let cholesky_compute_mem = GlobalMemBuffer::new(
            faer_cholesky::llt::compute::cholesky_in_place_req::<f32>(
                n_elems,
                Parallelism::None,
                Default::default(),
            )
            .unwrap(),
        );

        let cholesky_solve_mem = GlobalMemBuffer::new(
            faer_cholesky::llt::solve::solve_req::<f32>(n_elems, 1, Parallelism::None).unwrap(),
        );

        Self {
            cond_cov_mat: Mat::zeros(n_elems, n_elems),
            krig_point_cov_vec: Mat::zeros(n_elems, 1),
            weights: Mat::zeros(n_elems, 1),
            values: Mat::zeros(n_elems, 1),
            c_0: 0.0,
            cholesky_compute_mem,
            cholesky_solve_mem,
            n_elems,
            cond_x_buffer: T::zero(),
            cond_y_buffer: T::zero(),
            cond_z_buffer: T::zero(),
            cov_buffer: T::zero(),
            simd_vec_buffer: Vec::with_capacity(n_elems * (n_elems + 1) / (2 * 16) + 1),
        }
    }

    /// Build the covariance matrix and covariance vector for the system
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `kriging_point` - The kriging point
    /// * `vgram` - The variogram model
    // pub fn build_covariance_matrix_and_vector<V>(
    //     &mut self,
    //     cond_points: &[Point3<f32>],
    //     kriging_point: &Point3<f32>,
    //     vgram: &V,
    // ) where
    //     V: VariogramModel<MAYBE_SIMD = T>,
    // {
    //     let n_elems = cond_points.len();
    //     unsafe { self.cond_cov_mat.set_dims(n_elems, n_elems) };
    //     unsafe { self.krig_point_cov_vec.set_dims(n_elems, 1) };
    //     unsafe { self.weights.set_dims(n_elems, 1) };

    //     for i in 0..cond_points.len() {
    //         for j in 0..=i {
    //             let h = cond_points[i] - cond_points[j];
    //             let cov = vgram.covariogram(h);
    //             unsafe { self.cond_cov_mat.write_unchecked(i, j, cov) };
    //         }
    //         let h = kriging_point - cond_points[i];
    //         let cov = vgram.covariogram(h);
    //         unsafe { self.krig_point_cov_vec.write_unchecked(i, 0, cov) };
    //     }
    // }
    /// Set dimensions of the system for the given number of elements
    /// # Arguments
    /// * `n_elems` - The number of elements in the system
    /// # Safety
    /// This function is unsafe because it does not check that the number of elements is less than the maximum number of elements
    /// panics will occur if the number of elements is greater than the maximum number of elements
    #[inline(always)]
    pub fn set_dim(&mut self, n_elems: usize) {
        unsafe { self.cond_cov_mat.set_dims(n_elems, n_elems) };
        unsafe { self.krig_point_cov_vec.set_dims(n_elems, 1) };
        unsafe { self.weights.set_dims(n_elems, 1) };
        unsafe { self.values.set_dims(n_elems, 1) };
    }

    /// Build the covariance matrix for the system
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * 'vgram' - The variogram model
    #[inline(always)]
    pub fn vectorized_build_covariance_matrix<V>(&mut self, cond_points: &[Point3<f32>], vgram: &V)
    where
        V: VariogramModel<T>,
    {
        unsafe { self.simd_vec_buffer.set_len(0) };

        let mut cnt = 0;

        //compute lower triangle of cov matrix
        for (i, p1) in cond_points.iter().enumerate() {
            for p2 in cond_points[0..=i].iter() {
                let d = p1 - p2;
                // self.cond_x_buffer[cnt] = d.x;
                // self.cond_y_buffer[cnt] = d.y;
                // self.cond_z_buffer[cnt] = d.z;

                self.cond_x_buffer.replace(
                    cnt, d.x, //<<T as SimdValue>::Element as NumCast>::from(d.x).unwrap(),
                );
                self.cond_y_buffer.replace(
                    cnt, d.y, //<<T as SimdValue>::Element as NumCast>::from(d.y).unwrap(),
                );
                self.cond_z_buffer.replace(
                    cnt, d.z, //<<T as SimdValue>::Element as NumCast>::from(d.z).unwrap(),
                );

                cnt += 1;
                if cnt >= T::lanes() {
                    cnt = 0;
                    // let simd_x = f32x16::from(self.cond_x_buffer);
                    // let simd_y = f32x16::from(self.cond_y_buffer);
                    // let simd_z = f32x16::from(self.cond_z_buffer);
                    let vec_point =
                        Vector3::new(self.cond_x_buffer, self.cond_y_buffer, self.cond_z_buffer);
                    self.simd_vec_buffer.push(vec_point);
                }
            }
        }

        // let simd_x = f32x16::from(self.cond_x_buffer);
        // let simd_y = f32x16::from(self.cond_y_buffer);
        // let simd_z = f32x16::from(self.cond_z_buffer);
        let vec_point = Vector3::new(self.cond_x_buffer, self.cond_y_buffer, self.cond_z_buffer);
        self.simd_vec_buffer.push(vec_point);

        let mut simd_cov_iter = self
            .simd_vec_buffer
            .iter()
            .map(|simd_point| vgram.covariogram(*simd_point));

        let mut pop_cnt = 0;

        //populate covariance matrix using lower triangle value
        for i in 0..cond_points.len() {
            for j in 0..=i {
                if pop_cnt % T::lanes() == 0 {
                    self.cov_buffer = unsafe { simd_cov_iter.next().unwrap_unchecked().into() };
                }

                unsafe {
                    self.cond_cov_mat.write_unchecked(
                        i,
                        j,
                        self.cov_buffer.extract(pop_cnt % T::lanes()),
                    );
                };
                pop_cnt += 1;
            }
        }
    }

    /// Build the covariance vector for the system
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `kriging_point` - The kriging point
    /// * 'vgram' - The variogram model
    #[inline(always)]
    pub fn vectorized_build_covariance_vector<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        V: VariogramModel<T>,
    {
        unsafe { self.simd_vec_buffer.set_len(0) };

        let mut cnt = 0;

        //compute cov between kriging point and all conditioning points
        for p1 in cond_points.iter() {
            let d = p1 - kriging_point;
            // self.cond_x_buffer[cnt] = d.x;
            // self.cond_y_buffer[cnt] = d.y;
            // self.cond_z_buffer[cnt] = d.z;

            self.cond_x_buffer.replace(
                cnt, d.x, //<<T as SimdValue>::Element as NumCast>::from(d.x).unwrap(),
            );
            self.cond_y_buffer.replace(
                cnt, d.y, //<<T as SimdValue>::Element as NumCast>::from(d.y).unwrap(),
            );
            self.cond_z_buffer.replace(
                cnt, d.z, //<<T as SimdValue>::Element as NumCast>::from(d.z).unwrap(),
            );
            cnt += 1;
            if cnt >= T::lanes() {
                cnt = 0;
                // let simd_x = f32x16::from(self.cond_x_buffer);
                // let simd_y = f32x16::from(self.cond_y_buffer);
                // let simd_z = f32x16::from(self.cond_z_buffer);
                let vec_point =
                    Vector3::new(self.cond_x_buffer, self.cond_y_buffer, self.cond_z_buffer);
                self.simd_vec_buffer.push(vec_point);
            }
        }

        // let simd_x = f32x16::from(self.cond_x_buffer);
        // let simd_y = f32x16::from(self.cond_y_buffer);
        // let simd_z = f32x16::from(self.cond_z_buffer);
        let vec_point = Vector3::new(self.cond_x_buffer, self.cond_y_buffer, self.cond_z_buffer);
        self.simd_vec_buffer.push(vec_point);

        let mut simd_cov_iter = self
            .simd_vec_buffer
            .iter()
            .map(|simd_point| vgram.covariogram(*simd_point));

        let mut pop_cnt = 0;

        //populate covariance matrix using lower triangle value
        for i in 0..cond_points.len() {
            if pop_cnt % T::lanes() == 0 {
                self.cov_buffer = unsafe { simd_cov_iter.next().unwrap_unchecked().into() };
            }

            unsafe {
                self.krig_point_cov_vec.write_unchecked(
                    i,
                    0,
                    self.cov_buffer.extract(pop_cnt % T::lanes()),
                )
            };
            pop_cnt += 1;
        }
    }

    /// Build the covariance matrix and vector for the system
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `kriging_point` - The kriging point
    /// * 'vgram' - The variogram model
    #[inline(always)]
    pub fn vectorized_build_covariance_matrix_and_vector<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        V: VariogramModel<T>,
    {
        self.vectorized_build_covariance_matrix(cond_points, vgram);
        self.vectorized_build_covariance_vector(cond_points, kriging_point, vgram);
    }

    /// Compute SK weights
    #[inline(always)]
    pub fn compute_weights(&mut self) {
        //create dynstack
        let mut cholesky_compute_stack = DynStack::new(&mut self.cholesky_compute_mem);
        let mut cholesky_solve_stack = DynStack::new(&mut self.cholesky_solve_mem);

        //compute cholseky decomposition of covariance matrix
        let _ = compute::cholesky_in_place(
            self.cond_cov_mat.as_mut(),
            Parallelism::None,
            cholesky_compute_stack.rb_mut(),
            Default::default(),
        );

        //solve SK system
        faer_cholesky::llt::solve::solve_with_conj(
            self.weights.as_mut(),
            self.cond_cov_mat.as_ref(),
            Conj::No,
            self.krig_point_cov_vec.as_ref(),
            Parallelism::None,
            cholesky_solve_stack.rb_mut(),
        );
    }

    /// Build the system for SK and compute the weights
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `cond_values` - The conditioning values for the kriging point
    /// * `kriging_point` - The kriging point
    /// * 'vgram' - The variogram model
    #[inline(always)]
    pub fn build_system<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        cond_values: &[f32],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        V: VariogramModel<T>,
    {
        //set dimensions
        self.set_dim(cond_points.len());

        //build covariance matrix and vector
        self.vectorized_build_covariance_matrix_and_vector(cond_points, kriging_point, vgram);

        //store values
        unsafe { self.values.set_dims(cond_values.len(), 1) };
        for i in 0..cond_values.len() {
            unsafe { self.values.write_unchecked(i, 0, cond_values[i]) };
        }
        self.c_0 = vgram.c_0();

        //compute kriging weights
        self.compute_weights();
    }

    /// SK Estimate
    #[inline(always)]
    pub fn estimate(&self) -> f32 {
        inner_prod_with_conj(
            self.values.as_ref(),
            Conj::No,
            self.weights.as_ref(),
            Conj::No,
        )
    }
    /// SK Variance
    #[inline(always)]
    pub fn variance(&self) -> f32 {
        self.c_0
            - inner_prod_with_conj(
                self.weights.as_ref(),
                Conj::No,
                self.krig_point_cov_vec.as_ref(),
                Conj::No,
            )
    }

    /// Build the system for SK and compute the weights
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `cond_values` - The conditioning values for the kriging point
    /// * `kriging_point` - The kriging point
    /// * 'vgram' - The variogram model
    #[inline(always)]
    pub fn build_mini_system<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) -> MiniSKSystem
    where
        V: VariogramModel<T>,
    {
        //set dimensions
        self.set_dim(cond_points.len());

        //build covariance matrix and vector
        self.vectorized_build_covariance_matrix_and_vector(cond_points, kriging_point, vgram);

        self.c_0 = vgram.c_0();

        //compute kriging weights
        self.compute_weights();

        MiniSKSystem {
            c_0: self.c_0,
            weights: self.weights.clone(),
            cov_vec: self.krig_point_cov_vec.clone(),
        }
    }
}

pub struct MiniSKSystem {
    c_0: f32,
    weights: Mat<f32>,
    cov_vec: Mat<f32>,
}

impl MiniSKSystem {
    #[inline(always)]
    pub fn new(c_0: f32, weights: Mat<f32>, cov_vec: Mat<f32>) -> Self {
        Self {
            c_0,
            weights,
            cov_vec,
        }
    }

    #[inline(always)]
    pub fn estimate(&self, values: MatRef<f32>) -> f32 {
        inner_prod_with_conj(values, Conj::No, self.weights.as_ref(), Conj::No)
    }

    #[inline(always)]
    pub fn variance(&self) -> f32 {
        self.c_0
            - inner_prod_with_conj(
                self.weights.as_ref(),
                Conj::No,
                self.cov_vec.as_ref(),
                Conj::No,
            )
    }
}

impl<V, T> KrigingSystem<V, T> for SimpleKrigingSystem<T>
where
    T: SimdValue<Element = f32> + SimdRealField + Clone + Copy,
    T: SimdPartialOrd + SimdRealField,
    <T as SimdValue>::Element: SimdRealField + Float,
    V: VariogramModel<T>,
{
    fn new(n_elems: usize) -> Self {
        SimpleKrigingSystem::new(n_elems)
    }

    fn build_system(
        &mut self,
        conditioning_points: &[Point3<f32>],
        conditioning_values: &[f32],
        kriging_point: &Point3<f32>,
        variogram_model: &V,
    ) {
        SimpleKrigingSystem::build_system(
            self,
            conditioning_points,
            conditioning_values,
            kriging_point,
            variogram_model,
        );
    }

    fn estimate(&self) -> f32 {
        SimpleKrigingSystem::estimate(self)
    }

    fn variance(&self) -> f32 {
        SimpleKrigingSystem::variance(self)
    }
}

pub struct SimpleKriging<S, V, T>
where
    V: VariogramModel<T>,
    T: SimdPartialOrd + SimdRealField,
    <T as SimdValue>::Element: SimdRealField + Float,
{
    conditioning_data: S,
    variogram_model: V,
    search_ellipsoid: Ellipsoid,
    query_params: ConditioningParams,
    phantom: PhantomData<T>,
}

// impl<S, V, G> SimpleKriging<S, V, G>
// where
//     S: SpatialQueryable<f32, G> + Sync,
//     V: VariogramModel + Sync,
//     G: Sync,
// {
//     /// Create a new simple kriging estimator with the given parameters
//     /// # Arguments
//     /// * `conditioning_data` - The data to condition the kriging system on
//     /// * `variogram_model` - The variogram model to use
//     /// * `search_ellipsoid` - The search ellipsoid to use
//     /// * `kriging_parameters` - The kriging parameters to use
//     /// # Returns
//     /// A new simple kriging estimator
//     pub fn new(
//         conditioning_data: S,
//         variogram_model: V,
//         kriging_parameters: KrigingParameters,
//     ) -> Self {
//         Self {
//             conditioning_data,
//             variogram_model,
//             kriging_parameters,
//             phantom: PhantomData,
//         }
//     }

//     /// Perform simple kriging at all kriging points
//     pub fn krig(&self, kriging_points: &[Point3<f32>]) -> Vec<f32> {
//         //construct kriging system
//         let kriging_system = SimpleKrigingSystem::new(self.kriging_parameters.max_octant_data * 8);

//         kriging_points
//             .par_iter()
//             .progress()
//             .map_with(kriging_system.clone(), |local_system, kriging_point| {
//                 //let mut local_system = kriging_system.clone();
//                 //get nearest points and values
//                 let (cond_values, cond_points) = self.conditioning_data.query(kriging_point);

//                 //build kriging system for point
//                 local_system.build_system(
//                     &cond_points,
//                     cond_values.as_slice(),
//                     kriging_point,
//                     &self.variogram_model,
//                 );

//                 local_system.estimate()
//             })
//             .collect::<Vec<f32>>()
//     }
// }

impl<S, V, T> SimpleKriging<S, V, T>
where
    S: ConditioningProvider<Ellipsoid, f32, ConditioningParams> + Sync + std::marker::Send,
    V: VariogramModel<T> + Sync + std::marker::Send,
    T: SimdPartialOrd + SimdRealField,
    T: SimdValue<Element = f32> + Copy,
{
    /// Create a new simple kriging estimator with the given parameters
    /// # Arguments
    /// * `conditioning_data` - The data to condition the kriging system on
    /// * `variogram_model` - The variogram model to use
    /// * `search_ellipsoid` - The search ellipsoid to use
    /// * `kriging_parameters` - The kriging parameters to use
    /// # Returns
    /// A new simple kriging estimator
    pub fn new(
        conditioning_data: S,
        variogram_model: V,
        search_ellipsoid: Ellipsoid,
        query_params: ConditioningParams,
    ) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            search_ellipsoid,
            query_params,
            phantom: PhantomData,
        }
    }

    /// Perform simple kriging at all kriging points
    pub fn krig(&self, kriging_points: &[Point3<f32>]) -> Vec<f32> {
        //construct kriging system
        let kriging_system = SimpleKrigingSystem::new(self.query_params.max_n_cond * 8);

        kriging_points
            .par_iter()
            .progress()
            .map_with(
                (kriging_system.clone(), self.search_ellipsoid.clone()),
                |(local_system, ellipsoid), kriging_point| {
                    //translate search ellipsoid to kriging point
                    ellipsoid.translate_to(kriging_point);
                    //get nearest points and values
                    let (_, cond_values, cond_points) =
                        self.conditioning_data
                            .query(kriging_point, &ellipsoid, &self.query_params);

                    // if cond_points.len() > 0 {
                    //     println!("cond_points: {:?}", cond_points);
                    //     println!("search ellipsoid: {:?}", ellipsoid);
                    // }

                    //build kriging system for point
                    local_system.build_system(
                        &cond_points,
                        cond_values.as_slice(),
                        kriging_point,
                        &self.variogram_model,
                    );

                    local_system.estimate()
                },
            )
            .collect::<Vec<f32>>()
    }
}

#[cfg(test)]
mod tests {

    use std::{fs::File, io::Write};

    use itertools::Itertools;
    use nalgebra::{Translation3, UnitQuaternion, Vector3};
    use ndarray::Array3;
    use num_traits::Float;
    use simba::simd::WideF32x8;

    use crate::{
        geometry::ellipsoid::Ellipsoid,
        spatial_database::{
            coordinate_system::{CoordinateSystem, GridSpacing},
            normalized::Normalize,
            rtree_point_set::point_set::PointSet,
        },
        variography::model_variograms::spherical::SphericalVariogram,
    };

    use super::*;

    #[test]
    fn test_simple_kriging() {
        let cond_points = vec![
            Point3::new(2f32, 2f32, 0f32),
            Point3::new(3f32, 7f32, 0f32),
            Point3::new(9f32, 9f32, 0f32),
            Point3::new(6f32, 5f32, 0f32),
            Point3::new(5f32, 3f32, 0f32),
        ];
        let kriging_point = Point3::new(5f32, 5f32, 0f32);
        let values = vec![3f32, 4f32, 2f32, 4f32, 6f32];

        let rot = UnitQuaternion::identity();
        let vgram = SphericalVariogram::new(Vector3::new(10f32, 10f32, 10f32), 1f32, 0.25f32, rot);
        let mut system = SimpleKrigingSystem::new(cond_points.len());
        system.build_system(
            cond_points.as_slice(),
            values.as_slice(),
            &kriging_point,
            &vgram,
        );

        let mean = system.estimate();
        let variance = system.variance();

        assert_eq!(mean, 4.148547);
        assert_eq!(variance, 0.49257421);
    }

    #[test]
    fn sk_test() {
        // Define the coordinate system for the grid
        // origing at x = 0, y = 0, z = 0
        // azimuth = 0, dip = 0, plunge = 0
        let coordinate_system = CoordinateSystem::new(
            Point3::new(0.0, 0.0, 0.0).into(),
            UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians()),
        );

        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let mut gdb = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        // normalize the data
        gdb.normalize();

        // create a grid to store the simulation values
        let spacing = GridSpacing {
            x: 0.1,
            y: 0.1,
            z: 1.0,
        };
        // let krig_grid_arr = Array3::<Option<f32>>::from_elem([5000, 5000, 1], None);
        // let krig_db = InCompleteGriddedDataBase::new(
        //     krig_grid_arr,
        //     spacing.clone(),
        //     coordinate_system.clone(),
        // );

        // create a spherical variogram
        // azimuth = 0, dip = 0, plunge = 0
        // range_x = 150, range_y = 50, range_z = 1
        // sill = 1, nugget = 0
        let vgram_rot = UnitQuaternion::identity();
        let cs = CoordinateSystem::new(Default::default(), Default::default());
        let range = Vector3::new(
            WideF32x8::splat(150.0),
            WideF32x8::splat(50.0),
            WideF32x8::splat(10.0),
        );
        let sill = WideF32x8::splat(1.0f32);
        let nugget = WideF32x8::splat(0.2);

        let spherical_vgram = SphericalVariogram::new(range, sill, nugget, vgram_rot);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(150.0, 50.0, 10.0, cs.clone());

        // create a query engine for the conditioning data
        //let query_engine = GriddedDataBaseOctantQueryEngine::new(search_ellipsoid, &gdb, 16);

        // create a gsgs system
        let gsgs = SimpleKriging::new(
            gdb.clone(),
            spherical_vgram,
            search_ellipsoid,
            ConditioningParams::new(8),
        );

        //simulate values on grid
        // let points = krig_db
        //     .raw_grid
        //     .grid
        //     .indexed_iter()
        //     .map(|(ind, _)| krig_db.point_at_ind(&[ind.0, ind.1, ind.2]))
        //     .collect_vec();
        //"C:\Users\2jake\OneDrive - McGill University\Fall2022\MIME525\Project4\points_jacob.txt"
        println!("Reading Target Data");
        let targ = PointSet::<f32>::from_csv_index(
            "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
            "X",
            "Y",
            "Z",
            "V",
        )
        .unwrap();
        let points = targ.points.clone();
        // let points = (0..400)
        //     .cartesian_product(0..400)
        //     .map(|(x, y)| {
        //         let point = Point3::new(x as f32, y as f32, 0.0);
        //         point
        //     })
        //     .collect_vec();
        let values = gsgs.krig(points.as_slice());

        //save values to file for visualization

        let mut out = File::create("./test_results/sk.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in points.iter().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/sk_cond_data.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in gdb.points.iter().zip(gdb.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/sk.csv").unwrap();
        //write header
        out.write_all("X,Y,Z,XS,YS,ZS,V\n".as_bytes());

        //write each row

        for (point, value) in points.iter().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out.write_all(
                format!(
                    "{},{},{},{},{},{},{}\n",
                    point.x, point.y, point.z, 5, 5, 10, value
                )
                .as_bytes(),
            );
        }
    }
}
