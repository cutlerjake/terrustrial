use crate::{
    kriging::KrigingParameters, spatial_database::SpatialQueryable,
    variography::model_variograms::VariogramModel,
};

use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_cholesky::llt::compute;
use faer_core::{
    mul::{self, inner_prod::inner_prod_with_conj, matmul, triangular::BlockStructure},
    Conj, Mat, MatMut, MatRef, Parallelism,
};
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use nalgebra::{Point3, Vector3, LU};
use rand::{rngs::StdRng, Rng};
use rayon::prelude::*;
use simba::simd::f32x16;

pub struct LUSystem {
    pub l_mat: Mat<f32>,
    pub w_vec: Mat<f32>,
    pub intermediate_mat: Mat<f32>,
    pub cholesky_compute_mem: GlobalMemBuffer,
    pub cholesky_inv_mem: GlobalMemBuffer,
    pub n_sim: usize,
    pub n_cond: usize,
    pub n_total: usize,
    pub cond_x_buffer: [f32; 16],
    pub cond_y_buffer: [f32; 16],
    pub cond_z_buffer: [f32; 16],
    pub cov_buffer: [f32; 16],
    pub simd_vec_buffer: Vec<Vector3<f32x16>>,
}

impl LUSystem {
    pub fn new(n_sim: usize, n_cond: usize) -> Self {
        let n_total = n_sim + n_cond;
        let cholesky_compute_mem = GlobalMemBuffer::new(
            faer_cholesky::llt::compute::cholesky_in_place_req::<f32>(
                n_total,
                Parallelism::None,
                Default::default(),
            )
            .unwrap(),
        );

        let cholesky_inv_mem = GlobalMemBuffer::new(
            faer_cholesky::llt::inverse::invert_lower_in_place_req::<f32>(
                n_cond,
                Parallelism::None,
            )
            .unwrap(),
        );

        Self {
            l_mat: Mat::zeros(n_total, n_total),
            w_vec: Mat::zeros(n_total, 1),
            intermediate_mat: Mat::zeros(n_total, 1),
            cholesky_compute_mem,
            cholesky_inv_mem,
            n_sim,
            n_cond,
            n_total,
            cond_x_buffer: [0.0; 16],
            cond_y_buffer: [0.0; 16],
            cond_z_buffer: [0.0; 16],
            cov_buffer: [0.0; 16],
            simd_vec_buffer: Vec::with_capacity(n_total * (n_total + 1) / (2 * 16) + 1),
        }
    }

    #[inline(always)]
    fn vectorized_build_l_matrix<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        sim_points: &[Point3<f32>],
        vgram: &V,
    ) where
        V: VariogramModel,
    {
        unsafe { self.simd_vec_buffer.set_len(0) };

        let l_points = cond_points
            .iter()
            .chain(sim_points.iter())
            .map(|p| p.coords)
            .collect_vec();

        let mut cnt = 0;

        //compute lower triangle of L matrix
        for (i, p1) in l_points.iter().enumerate() {
            for p2 in l_points[0..=i].iter() {
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
        for i in 0..l_points.len() {
            for j in 0..=i {
                if pop_cnt % 16 == 0 {
                    self.cov_buffer = unsafe { simd_cov_iter.next().unwrap_unchecked().into() };
                }

                unsafe {
                    self.l_mat
                        .write_unchecked(i, j, self.cov_buffer[pop_cnt % 16]);
                };
                pop_cnt += 1;
            }
        }

        //create dynstack
        let mut cholesky_compute_stack = DynStack::new(&mut self.cholesky_compute_mem);
        let mut cholesky_inv_stack = DynStack::new(&mut self.cholesky_inv_mem);

        //compute cholseky decomposition of L matrix
        let _ = compute::cholesky_in_place(
            self.l_mat.as_mut(),
            Parallelism::None,
            cholesky_compute_stack.rb_mut(),
            Default::default(),
        );

        let m = self
            .l_mat
            .as_mut()
            .submatrix(0, 0, self.n_cond, self.n_cond);

        println!("{:?}", m);

        //invert dd portion of L matrix
        faer_cholesky::llt::inverse::invert_lower_in_place(
            m,
            Parallelism::None,
            cholesky_inv_stack.rb_mut(),
        );
    }

    #[inline(always)]
    fn populate_w_vec(&mut self, values: &[f32], rng: &mut StdRng) {
        //populate w vector
        for (i, v) in values.iter().enumerate() {
            unsafe { self.w_vec.write_unchecked(i, 0, values[i]) };
        }
        for i in values.len()..self.w_vec.nrows() {
            unsafe { self.w_vec.write_unchecked(i, 0, rng.gen::<f32>()) };
        }
    }

    #[inline(always)]
    fn compute_intermediate_mat(&mut self) {
        mul::triangular::matmul(
            self.intermediate_mat.as_mut(),
            BlockStructure::Rectangular,
            self.l_mat
                .as_ref()
                .submatrix(self.n_cond, 0, self.n_sim, self.n_cond),
            BlockStructure::Rectangular,
            self.l_mat
                .as_ref()
                .submatrix(0, 0, self.n_cond, self.n_cond),
            BlockStructure::TriangularLower,
            None,
            1.0,
            Parallelism::None,
        );
    }

    fn set_dims(&mut self, num_cond: usize, num_sim: usize) {
        self.n_cond = num_cond;
        self.n_sim = num_sim;
        self.n_total = num_cond + num_sim;
        unsafe {
            self.l_mat.set_dims(self.n_total, self.n_total);
            self.w_vec.set_dims(self.n_total, 1);
            self.intermediate_mat.set_dims(self.n_sim, self.n_cond);
        }
    }

    pub fn build_system<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        values: &[f32],
        sim_points: &[Point3<f32>],
        rng: &mut StdRng,
        vgram: &V,
    ) where
        V: VariogramModel,
    {
        let n_cond = cond_points.len();
        let n_sim = sim_points.len();
        self.set_dims(n_cond, n_sim);
        self.vectorized_build_l_matrix(cond_points, sim_points, vgram);
        self.populate_w_vec(values, rng);
        self.compute_intermediate_mat();
    }

    #[inline(always)]
    pub fn simulate(&self) -> Vec<f32> {
        let mut sim_mat = Mat::zeros(self.n_sim, 1);
        mul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.intermediate_mat.as_ref(),
            Conj::No,
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            Conj::No,
            None,
            1.0,
        );
        mul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.l_mat
                .as_ref()
                .submatrix(self.n_cond, self.n_cond, self.n_sim, self.n_sim),
            Conj::No,
            self.w_vec.as_ref().submatrix(self.n_cond, 0, self.n_sim, 1),
            Conj::No,
            None,
            1.0,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..sim_mat.nrows() {
            vals.push(unsafe { sim_mat.read_unchecked(i, 0) });
        }

        vals
    }

    pub fn create_mini_system<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        sim_points: &[Point3<f32>],
        vgram: &V,
    ) -> MiniLUSystem
    where
        V: VariogramModel,
    {
        let n_cond = cond_points.len();
        let n_sim = sim_points.len();
        self.set_dims(n_cond, n_sim);
        self.vectorized_build_l_matrix(cond_points, sim_points, vgram);
        self.compute_intermediate_mat();

        let l_gg = self
            .l_mat
            .as_ref()
            .submatrix(self.n_cond, self.n_cond, self.n_sim, self.n_sim)
            .to_owned();
        let intermediate = self.intermediate_mat.to_owned();
        let w = self.w_vec.to_owned();
        MiniLUSystem {
            n_sim,
            n_cond,
            l_gg,
            intermediate_mat: intermediate,
            w_vec: w,
        }
    }
}

impl Clone for LUSystem {
    fn clone(&self) -> Self {
        Self::new(self.n_sim, self.n_cond)
    }
}

pub struct MiniLUSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f32>,
    pub intermediate_mat: Mat<f32>,
    pub w_vec: Mat<f32>,
}

impl MiniLUSystem {
    #[inline(always)]
    pub fn populate_w_vec(&mut self, values: &[f32], rng: &mut StdRng) {
        //populate w vector
        for (i, v) in values.iter().enumerate() {
            unsafe { self.w_vec.write_unchecked(i, 0, values[i]) };
        }
        for i in values.len()..self.w_vec.nrows() {
            unsafe { self.w_vec.write_unchecked(i, 0, rng.gen::<f32>()) };
        }
    }

    #[inline(always)]
    pub fn simulate(&self) -> Vec<f32> {
        let mut sim_mat = Mat::zeros(self.n_sim, 1);
        mul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.intermediate_mat.as_ref(),
            Conj::No,
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            Conj::No,
            None,
            1.0,
        );
        mul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.l_gg.as_ref(),
            Conj::No,
            self.w_vec.as_ref().submatrix(self.n_cond, 0, self.n_sim, 1),
            Conj::No,
            None,
            1.0,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..sim_mat.nrows() {
            vals.push(unsafe { sim_mat.read_unchecked(i, 0) });
        }

        vals
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::UnitQuaternion;
    use num_traits::Float;
    use rand::SeedableRng;

    use crate::{
        spatial_database::coordinate_system::CoordinateSystem,
        variography::model_variograms::spherical::SphericalVariogram,
    };

    use super::*;
    #[test]
    fn test_lu() {
        let mut lu = LUSystem::new(2, 3);
        let cond_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let values = vec![0.0, 1.0, 2.0];
        let sim_points = vec![Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0)];
        let vgram_rot =
            UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
        let vgram_origin = Point3::new(0.0, 0.0, 0.0);
        let vgram_coordinate_system = CoordinateSystem::new(vgram_origin.into(), vgram_rot);
        let range = Vector3::new(2.0, 2.0, 2.0);
        let sill = 1.0;
        let nugget = 0.0;

        let vgram = SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());
        let mut rng = StdRng::seed_from_u64(0);
        lu.build_system(&cond_points, &values, &sim_points, &mut rng, &vgram);
        let vals = lu.simulate();
        println!("{:?}", vals);

        let mut rng = StdRng::seed_from_u64(0);
        let mut mini = lu.create_mini_system(&cond_points, &sim_points, &vgram);
        mini.populate_w_vec(values.as_slice(), &mut rng);
        let vals = mini.simulate();
        println!("{:?}", vals);
    }

    #[test]
    fn submatrix() {
        let mat = Mat::<f32>::zeros(5, 5);
        let sub = mat.as_ref().submatrix(0, 0, 1, 1);
        println!("{:?}", sub);
    }
}
