use core::panic;

use crate::variography::model_variograms::VariogramModel;

use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, StackReq};
use faer_cholesky::llt::{
    compute,
    inverse::invert_lower_in_place,
    solve::{solve_transpose_req, solve_transpose_with_conj},
    CholeskyError,
};

use faer_core::{
    mul::{self, triangular::BlockStructure},
    zipped, Conj, Mat, Parallelism,
};
use faer_core::{
    solve::{solve_lower_triangular_in_place, solve_upper_triangular_in_place},
    Scale,
};
use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;
use simba::simd::f32x16;

pub struct LUSystem {
    pub l_mat: Mat<f32>,
    pub w_vec: Mat<f32>,
    pub intermediate_mat: Mat<f32>,
    buffer: GlobalMemBuffer,
    pub n_sim: usize,
    pub n_cond: usize,
    pub n_total: usize,
    pub sim_size: usize,
    pub cond_size: usize,
    pub cond_x_buffer: [f32; 16],
    pub cond_y_buffer: [f32; 16],
    pub cond_z_buffer: [f32; 16],
    pub cov_buffer: [f32; 16],
    pub simd_vec_buffer: Vec<Vector3<f32x16>>,
}

impl LUSystem {
    pub fn new(n_sim: usize, n_cond: usize) -> Self {
        // L matrix will have dimensions of n_cond + n_Sim
        let n_total = n_sim + n_cond;

        //create a buffer large enough to compute cholesky in place
        let cholesky_required_mem = faer_cholesky::llt::compute::cholesky_in_place_req::<f32>(
            n_total,
            Parallelism::None,
            Default::default(),
        )
        .unwrap();

        let buffer = GlobalMemBuffer::new(cholesky_required_mem);

        Self {
            l_mat: Mat::zeros(n_total, n_total),
            w_vec: Mat::zeros(n_total, 1),
            intermediate_mat: Mat::zeros(n_sim, n_cond),
            buffer,
            n_sim,
            n_cond,
            n_total,
            sim_size: n_sim,
            cond_size: n_cond,
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
    ) -> Result<(), CholeskyError>
    where
        V: VariogramModel,
    {
        unsafe { self.simd_vec_buffer.set_len(0) };

        let l_points = cond_points
            .iter()
            .chain(sim_points.iter())
            .map(|p| p.coords)
            .collect_vec();

        let mut cnt = 0;

        //Compute Covariance values to populate lower triangle of L matrix
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

        //populate L matrix with covariance matrix
        for i in 0..l_points.len() {
            for j in 0..=i {
                if pop_cnt % 16 == 0 {
                    self.cov_buffer = simd_cov_iter.next().unwrap().into();
                }

                self.l_mat.write(i, j, self.cov_buffer[pop_cnt % 16]);

                pop_cnt += 1;
            }
        }

        //create dynstacks
        //let mut cholesky_compute_stack = DynStack::new(&mut self.cholesky_compute_mem);
        let mut cholesky_compute_stack = DynStack::new(&mut self.buffer);

        //compute cholseky decomposition of L matrix
        compute::cholesky_in_place(
            self.l_mat.as_mut(),
            Parallelism::None,
            cholesky_compute_stack.rb_mut(),
            Default::default(),
        )
    }

    #[inline(always)]
    fn populate_w_vec(&mut self, values: &[f32], rng: &mut StdRng) {
        //populate w vector with conditioning points
        for (i, v) in values.iter().enumerate() {
            self.w_vec.write(i, 0, *v);
        }
        // fill remaining values with random numbers
        for i in values.len()..self.w_vec.nrows() {
            self.w_vec.write(i, 0, rng.sample(StandardNormal));
        }
    }

    #[inline(always)]
    fn compute_intermediate_mat(&mut self) {
        //TODO: These are disjoint views of the same matrix, avoid copying
        let mut intermediate = self
            .l_mat
            .as_ref()
            .submatrix(self.n_cond, 0, self.n_sim, self.n_cond)
            .transpose()
            .to_owned();

        //Want to compute L_gd @ L_dd^-1
        //avoid inverting L_dd by solving L_dd^T * intermediate^T = L_dg^T
        solve_upper_triangular_in_place(
            self.l_mat
                .as_ref()
                .submatrix(0, 0, self.n_cond, self.n_cond)
                .transpose(),
            intermediate.as_mut(),
            Parallelism::None,
        );

        self.intermediate_mat = intermediate.transpose().to_owned();
    }

    #[inline(always)]
    fn set_dims(&mut self, num_cond: usize, num_sim: usize) {
        assert!(num_cond <= self.cond_size);
        assert!(num_sim <= self.sim_size);
        self.n_cond = num_cond;
        self.n_sim = num_sim;
        self.n_total = num_cond + num_sim;
        // unsafe {
        //     self.l_mat.set_dims(self.n_total, self.n_total);
        //     self.w_vec.set_dims(self.n_total, 1);
        //     self.intermediate_mat.set_dims(self.n_sim, self.n_cond);
        // }

        self.l_mat
            .resize_with(self.n_total, self.n_total, |_, _| 0.0);
        self.w_vec.resize_with(self.n_total, 1, |_, _| 0.0);
        self.intermediate_mat
            .resize_with(self.n_sim, self.n_cond, |_, _| 0.0);
    }

    #[inline(always)]
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

        //L_gd @ L_dd^-1 @ w_d
        mul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.intermediate_mat.as_ref(),
            Conj::No,
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            Conj::No,
            None,
            1.0,
        );
        //L_gg @ w_g
        mul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.l_mat
                .as_ref()
                .submatrix(self.n_cond, self.n_cond, self.n_sim, self.n_sim),
            Conj::No,
            self.w_vec.as_ref().submatrix(self.n_cond, 0, self.n_sim, 1),
            Conj::No,
            Some(1.0),
            1.0,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..sim_mat.nrows() {
            vals.push(sim_mat.read(i, 0));
        }

        vals
    }

    #[inline(always)]
    pub fn create_mini_sk_system<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        sim_points: &[Point3<f32>],
        vgram: &V,
    ) -> MiniLUSKSystem
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
            .to_owned()
            .clone();
        let intermediate = self.intermediate_mat.clone();
        let w = self.w_vec.clone();

        MiniLUSKSystem {
            n_sim,
            n_cond,
            l_gg,
            intermediate_mat: intermediate,
            w_vec: w,
        }
    }

    #[inline(always)]
    pub fn create_mini_ok_system<V>(
        &mut self,
        cond_points: &[Point3<f32>],
        cond_values: &[f32],
        sim_points: &[Point3<f32>],
        vgram: &V,
    ) -> MiniLUOKSystem
    where
        V: VariogramModel,
    {
        let n_cond = cond_points.len();
        let n_sim = sim_points.len();

        self.set_dims(n_cond, n_sim);
        self.vectorized_build_l_matrix(cond_points, sim_points, vgram);
        self.compute_intermediate_mat();

        println!("intermediate: {:?}", self.intermediate_mat);

        let ones = Mat::<f32>::from_fn(n_cond, 1, |_, _| 1.0);
        let i = Mat::<f32>::identity(self.intermediate_mat.nrows(), self.intermediate_mat.ncols());

        // //(I - L_gd @ L_dd^-1)
        // let a = i - &self.intermediate_mat;

        // //e^T @(L_11^-1)^T
        // let mut b = ones.clone();

        // solve_lower_triangular_in_place(
        //     self.l_mat
        //         .as_ref()
        //         .submatrix(0, 0, self.n_cond, self.n_cond)
        //         .transpose(),
        //     b.as_mut(),
        //     Parallelism::None,
        // );

        // println!("b: {:?}", b);

        // //L_11^-1 @ z
        // let mut c = Mat::from_fn(n_cond, 1, |i, _| cond_values[i]);

        // solve_lower_triangular_in_place(
        //     self.l_mat
        //         .as_ref()
        //         .submatrix(0, 0, self.n_cond, self.n_cond),
        //     c.as_mut(),
        //     Parallelism::None,
        // );

        // println!("c: {:?}", c);

        // let mut d = Mat::zeros(1, 1);
        // mul::matvec::matvec_with_conj(
        //     d.as_mut(),
        //     b.as_ref().transpose(),
        //     Conj::No,
        //     c.as_ref(),
        //     Conj::No,
        //     None,
        //     1.0,
        // );

        // println!("d: {:?}", d);

        // let mut e = Mat::zeros(1, 1);
        // mul::matvec::matvec_with_conj(
        //     e.as_mut(),
        //     b.as_ref().transpose(),
        //     Conj::No,
        //     b.as_ref(),
        //     Conj::No,
        //     None,
        //     1.0,
        // );

        // println!("e: {:?}", e);

        // let mut ok = Mat::zeros(n_sim, 1);
        // mul::matvec::matvec_with_conj(
        //     ok.as_mut(),
        //     a.as_ref(),
        //     Conj::No,
        //     ones.as_ref(),
        //     Conj::No,
        //     None,
        //     1.0,
        // );

        // let frac = Scale(d.read(0, 0) / e.read(0, 0));

        // ok *= frac;

        // print!("ok: {:?}", ok);

        // lambda_e - C_dd^-1 @ e
        let mut lambda_e = ones.clone();

        let mut stack = DynStack::new(&mut self.buffer);

        faer_cholesky::llt::solve::solve_in_place_with_conj(
            self.l_mat
                .as_ref()
                .submatrix(0, 0, self.n_cond, self.n_cond),
            Conj::No,
            lambda_e.as_mut(),
            Parallelism::None,
            stack.rb_mut(),
        );

        println!("lambda_e: {:?}", lambda_e);
        println!("intermediate: {:?}", self.intermediate_mat);

        let mut num = Mat::zeros(1, n_sim);
        mul::matvec::matvec_with_conj(
            num.as_mut().transpose(),
            self.intermediate_mat.as_ref(),
            Conj::No,
            ones.as_ref(),
            Conj::No,
            None,
            1.0,
        );

        print!("num: {:?}", num);

        zipped!(num.as_mut()).for_each(|mut v| v.write(1f32 - v.read()));

        //num = num + Scale(1.0);

        let mut denom = 0.0;
        zipped!(lambda_e.as_ref()).for_each(|v| denom += v.read());

        // let mut ok = Mat::from_fn(
        //     self.intermediate_mat.nrows(),
        //     self.intermediate_mat.ncols(),
        //     |i, j| self.intermediate_mat.read(i, j) + (1.0 - denom) / denom * lambda_e.read(j, 0),
        // );

        let temp = Mat::<f32>::from_fn(
            self.intermediate_mat.nrows(),
            self.intermediate_mat.ncols(),
            |i, j| num.read(0, i) / denom * lambda_e.read(j, 0),
        );

        println!("temp: {:?}", temp);

        println!("ok: {:?}", self.intermediate_mat.clone() + temp);

        //ok += Scale((1f32 - s) / s) * &lambda_e.transpose();

        //println!("ok: {:?}", ok);

        let l_gg = self
            .l_mat
            .as_ref()
            .submatrix(self.n_cond, self.n_cond, self.n_sim, self.n_sim)
            .to_owned()
            .clone();
        let intermediate = self.intermediate_mat.clone();
        let w = self.w_vec.clone();

        MiniLUOKSystem {
            n_sim,
            n_cond,
            l_gg,
            ok_weights: intermediate,
            w_vec: w,
        }
    }
}

impl Clone for LUSystem {
    fn clone(&self) -> Self {
        Self::new(self.sim_size, self.cond_size)
    }
}

pub struct MiniLUSKSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f32>,
    pub intermediate_mat: Mat<f32>,
    pub w_vec: Mat<f32>, // consider not storing w vec on this struct to avoid reallocating memory in hot loop
}

impl MiniLUSKSystem {
    #[inline(always)]
    pub fn populate_w_vec(&mut self, values: &[f32], rng: &mut StdRng) {
        //populate w vector
        for (i, v) in values.iter().enumerate() {
            self.w_vec.write(i, 0, *v);
        }
        for i in values.len()..self.w_vec.nrows() {
            self.w_vec.write(i, 0, rng.sample(StandardNormal));
        }
    }

    #[inline(always)]
    pub fn sk_values(&self) -> Vec<f32> {
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

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..sim_mat.nrows() {
            vals.push(sim_mat.read(i, 0));
        }

        vals
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

        mul::triangular::matmul(
            sim_mat.as_mut(),
            BlockStructure::Rectangular,
            self.l_gg.as_ref(),
            BlockStructure::TriangularLower,
            self.w_vec.as_ref().submatrix(self.n_cond, 0, self.n_sim, 1),
            BlockStructure::Rectangular,
            Some(1.0),
            1.0,
            Parallelism::None,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..sim_mat.nrows() {
            vals.push(sim_mat.read(i, 0));
        }

        vals
    }
}

pub struct MiniLUOKSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f32>,
    pub ok_weights: Mat<f32>,
    pub w_vec: Mat<f32>, // consider not storing w vec on this struct to avoid reallocating memory in hot loop
}

impl MiniLUOKSystem {
    //#[inline(always)]
    // pub fn ok_values(&self) -> Vec<f32> {
    //     //y_ok = y_sk + [I-\Lambda_sk] @ e @ (\lambda_e^T @ z)/ (\lambda_e^T @ e)
    //     //populate sk estimates
    //     let mut sim_mat = Mat::zeros(self.n_sim, 1);
    //     mul::matvec::matvec_with_conj(
    //         sim_mat.as_mut(),
    //         self.intermediate_mat.as_ref(),
    //         Conj::No,
    //         self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
    //         Conj::No,
    //         None,
    //         1.0,
    //     );
    //     Mat::<f32>::from_fn(self.intermediate_mat.nrows(), 1, |_, _| {
    //         1.0
    //     });
    //     let a = Mat::<f32>::identity(self.intermediate_mat.nrows(), self.intermediate_mat.ncols())
    //         - &self.intermediate_mat;

    //     let b = sol

    //     let mut vals = Vec::with_capacity(self.n_sim);
    //     for i in 0..sim_mat.nrows() {
    //         vals.push(sim_mat.read(i, 0));
    //     }

    //     vals
    // }
}

#[cfg(test)]
mod tests {

    use nalgebra::UnitQuaternion;
    use num_traits::Float;
    use rand::SeedableRng;

    use crate::{
        kriging::simple_kriging::SimpleKrigingSystem,
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
        let nugget = 0.1;

        let vgram = SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());
        let mut rng = StdRng::seed_from_u64(0);
        lu.build_system(&cond_points, &values, &sim_points, &mut rng, &vgram);
        let vals = lu.simulate();
        println!("{:?}", vals);

        let mut rng = StdRng::seed_from_u64(0);
        let mut mini = lu.create_mini_sk_system(&cond_points, &sim_points, &vgram);
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

    #[test]
    fn sgs_vs_lu() {
        let mut lu = LUSystem::new(1, 3);
        let cond_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let values = vec![0.0, 1.0, 2.0];
        let sim_points = vec![Point3::new(0.0, 0.0, 1.0)];
        let vgram_rot =
            UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
        let vgram_origin = Point3::new(0.0, 0.0, 0.0);
        let vgram_coordinate_system = CoordinateSystem::new(vgram_origin.into(), vgram_rot);
        let range = Vector3::new(200.0, 200.0, 200.0);
        let sill = 1.0;
        let nugget = 0.1;

        let vgram = SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());
        let mut rng = StdRng::seed_from_u64(0);

        let mut rng = StdRng::seed_from_u64(0);
        let mut mini = lu.create_mini_sk_system(&cond_points, &sim_points, &vgram);
        mini.populate_w_vec(values.as_slice(), &mut rng);
        let mut val = mini.sk_values()[0];

        //let val: f32 = (0..10000).map(|_| mini.simulate()[0]).sum::<f32>() / 10000.0;
        println!("{:?}", val);
        println!("{:?}", mini.w_vec);

        let mut system = SimpleKrigingSystem::new(cond_points.len());
        system.build_system(
            cond_points.as_slice(),
            values.as_slice(),
            &sim_points[0],
            &vgram,
        );

        println!("mean: {}", system.estimate());
    }

    #[test]
    fn test_lu_ok() {
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
        let nugget = 0.1;

        let vgram = SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());
        let mut rng = StdRng::seed_from_u64(0);

        lu.create_mini_ok_system(
            cond_points.as_slice(),
            values.as_slice(),
            sim_points.as_slice(),
            &vgram,
        );
    }
}
