use crate::{
    geometry::ellipsoid::Ellipsoid,
    kriging::KrigingParameters,
    spatial_database::{
        gridded_databases::{
            gridded_data_base_queary_engine::GriddedDataBaseOctantQueryEngine,
            GriddedDataBaseInterface,
        },
        SpatialDataBase, SpatialQueryable,
    },
    variography::model_variograms::VariogramModel,
};

use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_cholesky::llt::compute;
use faer_cholesky::llt::solve::solve_in_place_with_conj;
use faer_core::{mul::inner_prod::inner_prod_with_conj, Conj, Mat, Parallelism};
use indicatif::ParallelProgressIterator;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use simba::simd::f32x16;

pub struct SimpleKrigingSystem {
    pub(crate) cond_cov_mat: Mat<f32>,
    pub(crate) krig_point_cov_vec: Mat<f32>,
    pub(crate) weights: Mat<f32>,
    pub(crate) values: Mat<f32>,
    pub(crate) c_0: f32,
    pub(crate) cholesky_compute_mem: GlobalMemBuffer,
    pub(crate) cholesky_solve_mem: GlobalMemBuffer,
    pub(crate) n_elems: usize,
    cond_x_buffer: [f32; 16],
    cond_y_buffer: [f32; 16],
    cond_z_buffer: [f32; 16],
    cov_buffer: [f32; 16],
    simd_vec_buffer: Vec<Vector3<f32x16>>,
}

impl Clone for SimpleKrigingSystem {
    fn clone(&self) -> Self {
        let n_elems = self.n_elems;
        Self::new(n_elems)
    }
}

impl SimpleKrigingSystem {
    /// Create a new simple kriging system
    /// # Arguments
    /// * `n_elems` - The maximum number of elements in the system
    /// # Returns
    /// * `Self` - The new simple kriging system
    fn new(n_elems: usize) -> Self {
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
            cond_x_buffer: [0.0; 16],
            cond_y_buffer: [0.0; 16],
            cond_z_buffer: [0.0; 16],
            cov_buffer: [0.0; 16],
            simd_vec_buffer: Vec::with_capacity(n_elems * (n_elems + 1) / (2 * 16) + 1),
        }
    }

    /// Build the covariance matrix and covariance vector for the system
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `kriging_point` - The kriging point
    /// * `vgram` - The variogram model
    fn build_covariance_matrix_and_vector<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        V: VariogramModel,
    {
        let n_elems = cond_points.len();
        unsafe { self.cond_cov_mat.set_dims(n_elems, n_elems) };
        unsafe { self.krig_point_cov_vec.set_dims(n_elems, 1) };
        unsafe { self.weights.set_dims(n_elems, 1) };

        for i in 0..cond_points.len() {
            for j in 0..=i {
                let h = cond_points[i] - cond_points[j];
                let cov = vgram.covariogram(h);
                unsafe { self.cond_cov_mat.write_unchecked(i, j, cov) };
            }
            let h = kriging_point - cond_points[i];
            let cov = vgram.covariogram(h);
            unsafe { self.krig_point_cov_vec.write_unchecked(i, 0, cov) };
        }
    }
    /// Set dimensions of the system for the given number of elements
    /// # Arguments
    /// * `n_elems` - The number of elements in the system
    /// # Safety
    /// This function is unsafe because it does not check that the number of elements is less than the maximum number of elements
    /// panics will occur if the number of elements is greater than the maximum number of elements
    #[inline(always)]
    fn set_dim(&mut self, n_elems: usize) {
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
    fn vectorized_build_covariance_matrix<V>(&mut self, cond_points: &[Point3<f32>], vgram: &V)
    where
        V: VariogramModel,
    {
        unsafe { self.simd_vec_buffer.set_len(0) };

        let mut cnt = 0;

        //compute lower triangle of cov matrix
        for (i, p1) in cond_points.iter().enumerate() {
            for p2 in cond_points[0..=i].iter() {
                let d = p1 - p2;
                self.cond_x_buffer[cnt] = d.x;
                self.cond_y_buffer[cnt] = d.y;
                self.cond_z_buffer[cnt] = d.z;
                cnt += 1;
                if cnt >= 16 {
                    cnt = 0;
                    let simd_x = f32x16::from(self.cond_x_buffer);
                    let simd_y = f32x16::from(self.cond_y_buffer);
                    let simd_z = f32x16::from(self.cond_z_buffer);
                    let vec_point = Vector3::<f32x16>::new(simd_x, simd_y, simd_z);
                    self.simd_vec_buffer.push(vec_point);
                }
            }
        }

        let simd_x = f32x16::from(self.cond_x_buffer);
        let simd_y = f32x16::from(self.cond_y_buffer);
        let simd_z = f32x16::from(self.cond_z_buffer);
        let vec_point = Vector3::<f32x16>::new(simd_x, simd_y, simd_z);
        self.simd_vec_buffer.push(vec_point);

        let mut simd_cov_iter = self
            .simd_vec_buffer
            .iter()
            .map(|simd_point| vgram.vectorized_covariogram(*simd_point));

        let mut pop_cnt = 0;

        //populate covariance matrix using lower triangle value
        for i in 0..cond_points.len() {
            for j in 0..=i {
                if pop_cnt % 16 == 0 {
                    self.cov_buffer = unsafe { simd_cov_iter.next().unwrap_unchecked().into() };
                }

                unsafe {
                    self.cond_cov_mat
                        .write_unchecked(i, j, self.cov_buffer[pop_cnt % 16]);
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
    fn vectorized_build_covariance_vector<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        V: VariogramModel,
    {
        unsafe { self.simd_vec_buffer.set_len(0) };

        let mut cnt = 0;

        //compute cov between kriging point and all conditioning points
        for p1 in cond_points.iter() {
            let d = p1 - kriging_point;
            self.cond_x_buffer[cnt] = d.x;
            self.cond_y_buffer[cnt] = d.y;
            self.cond_z_buffer[cnt] = d.z;
            cnt += 1;
            if cnt >= 16 {
                cnt = 0;
                let simd_x = f32x16::from(self.cond_x_buffer);
                let simd_y = f32x16::from(self.cond_y_buffer);
                let simd_z = f32x16::from(self.cond_z_buffer);
                let vec_point = Vector3::<f32x16>::new(simd_x, simd_y, simd_z);
                self.simd_vec_buffer.push(vec_point);
            }
        }

        let simd_x = f32x16::from(self.cond_x_buffer);
        let simd_y = f32x16::from(self.cond_y_buffer);
        let simd_z = f32x16::from(self.cond_z_buffer);
        let vec_point = Vector3::<f32x16>::new(simd_x, simd_y, simd_z);
        self.simd_vec_buffer.push(vec_point);

        let mut simd_cov_iter = self
            .simd_vec_buffer
            .iter()
            .map(|simd_point| vgram.vectorized_covariogram(*simd_point));

        let mut pop_cnt = 0;

        //populate covariance matrix using lower triangle value
        for i in 0..cond_points.len() {
            if pop_cnt % 16 == 0 {
                self.cov_buffer = unsafe { simd_cov_iter.next().unwrap_unchecked().into() };
            }

            unsafe {
                self.krig_point_cov_vec
                    .write_unchecked(i, 0, self.cov_buffer[pop_cnt % 16])
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
    fn vectorized_build_covariance_matrix_and_vector<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        V: VariogramModel,
    {
        self.vectorized_build_covariance_matrix(cond_points, vgram);
        self.vectorized_build_covariance_vector(cond_points, kriging_point, vgram);
    }

    /// Compute SK weights
    #[inline(always)]
    fn compute_weights(&mut self) {
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
        V: VariogramModel,
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
    pub fn estimate(&mut self) -> f32 {
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
}

pub struct SimpleKriging<S, V> {
    conditioning_data: S,
    variogram_model: V,
    kriging_parameters: KrigingParameters,
}

impl<S, V> SimpleKriging<S, V>
where
    S: SpatialQueryable<f32> + Sync,
    V: VariogramModel + Sync,
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
        kriging_parameters: KrigingParameters,
    ) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            kriging_parameters,
        }
    }

    /// Perform simple kriging at all kriging points
    pub fn krige(&self, kriging_points: &[Point3<f32>]) -> Vec<f32> {
        //construct kriging system
        let kriging_system = SimpleKrigingSystem::new(self.kriging_parameters.max_octant_data * 8);

        kriging_points
            .par_iter()
            .progress()
            .map_with(kriging_system.clone(), |local_system, kriging_point| {
                //let mut local_system = kriging_system.clone();
                //get nearest points and values
                let (cond_values, cond_points) = self.conditioning_data.query(kriging_point);

                //build kriging system for point
                local_system.build_system(
                    &cond_points,
                    cond_values.as_slice(),
                    kriging_point,
                    &self.variogram_model,
                );

                local_system.estimate()
            })
            .collect::<Vec<f32>>()
    }
}

#[cfg(test)]
mod tests {

    use nalgebra::{Translation3, UnitQuaternion, Vector3};

    use crate::{
        spatial_database::coordinate_system::{self, CoordinateSystem},
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
        let coordinate_system = CoordinateSystem::new(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
        );
        let vgram = SphericalVariogram::new(
            Vector3::new(10f32, 10f32, 10f32),
            1f32,
            0.25f32,
            coordinate_system,
        );
        let mut system = SimpleKrigingSystem::new(cond_points.len());
        system.build_system(
            cond_points.as_slice(),
            values.as_slice(),
            &kriging_point,
            &vgram,
        );

        let mean = system.estimate();
        let variance = system.variance();

        assert_eq!(mean, 4.488526);
        assert_eq!(variance, 0.1558531);
    }
}
