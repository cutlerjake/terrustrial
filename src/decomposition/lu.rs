use std::{
    borrow::Borrow,
    ops::{Add, Sub},
};

use crate::{kriging::simple_kriging::SKBuilder, variography::model_variograms::VariogramModel};

use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};

use faer::{
    linalg::{
        matmul::{self, triangular::BlockStructure},
        triangular_solve::solve_upper_triangular_in_place,
    },
    modules::cholesky::{self, llt::compute::LltInfo},
    solvers::CholeskyError,
    unzipped, zipped, Conj, Mat, MatMut, MatRef, Parallelism,
};
use nalgebra::{SimdRealField, SimdValue};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;

use super::solved_systems::SolvedLUSystem;

pub struct LUSystem {
    pub l_mat: Mat<f32>,
    pub w_vec: Mat<f32>,
    pub intermediate_mat: Mat<f32>,
    pub buffer: GlobalPodBuffer,
    pub n_sim: usize,
    pub n_cond: usize,
    pub n_total: usize,
    pub sim_size: usize,
    pub cond_size: usize,
}

impl LUSystem {
    pub fn new(n_sim: usize, n_cond: usize) -> Self {
        // L matrix will have dimensions of n_cond + n_Sim
        let n_total = n_sim + n_cond;

        //create a buffer large enough to compute cholesky in place
        let cholesky_required_mem = cholesky::llt::compute::cholesky_in_place_req::<f32>(
            n_total,
            Parallelism::None,
            Default::default(),
        )
        .unwrap();

        let buffer = GlobalPodBuffer::new(cholesky_required_mem);

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
        }
    }

    #[inline(always)]
    pub(crate) fn build_cov_matrix<V, T, SKB>(
        &mut self,
        cond_points: &[SKB::Support],
        sim_points: &[SKB::Support],
        vgram: &V,
    ) where
        V: VariogramModel<T>,
        SKB: SKBuilder,
        T: SimdValue<Element = f32> + SimdRealField + Copy,
    {
        //set dimensions of LU system
        let n_cond = cond_points.len();
        let n_sim = sim_points.len();
        self.set_dims(n_cond, n_sim);

        let l_points = cond_points.iter().chain(sim_points);
        SKB::build_cov_mat(&mut self.l_mat, l_points, vgram)
    }

    #[inline(always)]
    pub(crate) fn compute_l_matrix(&mut self) {
        //create dynstacks
        let mut cholesky_compute_stack = PodStack::new(&mut self.buffer);

        //compute cholseky decomposition of L matrix
        cholesky::llt::compute::cholesky_in_place(
            self.l_mat.as_mut(),
            Default::default(),
            Parallelism::None,
            cholesky_compute_stack.rb_mut(),
            Default::default(),
        )
        .unwrap();
    }

    #[inline(always)]
    pub(crate) fn build_l_matrix<V, T, SKB>(
        &mut self,
        cond_points: &[SKB::Support],
        sim_points: &[SKB::Support],
        vgram: &V,
    ) -> Result<LltInfo, CholeskyError>
    where
        V: VariogramModel<T>,
        SKB: SKBuilder,
        T: SimdValue<Element = f32> + SimdRealField + Copy,
    {
        let l_points = cond_points.iter().chain(sim_points);

        SKB::build_cov_mat(&mut self.l_mat, l_points, vgram);
        //println!("cov_mat: {:?}", self.l_mat);

        //create dynstacks
        //let mut cholesky_compute_stack = DynStack::new(&mut self.cholesky_compute_mem);
        let mut cholesky_compute_stack = PodStack::new(&mut self.buffer);

        //compute cholseky decomposition of L matrix
        cholesky::llt::compute::cholesky_in_place(
            self.l_mat.as_mut(),
            Default::default(),
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
    pub(crate) fn compute_intermediate_mat(&mut self) {
        let (l_dd, _, l_gd, _) = self.l_mat.as_mut().split_at_mut(self.n_cond, self.n_cond);

        let mut l_gd_t = l_gd.transpose_mut();

        //Want to compute L_gd @ L_dd^-1
        //avoid inverting L_dd by solving L_dd^T * intermediate^T = L_dg^T
        solve_upper_triangular_in_place(
            l_dd.as_ref().transpose(),
            l_gd_t.as_mut(),
            Parallelism::None,
        );

        self.intermediate_mat = l_gd_t.transpose_mut().to_owned();
    }

    #[inline(always)]
    fn set_dims(&mut self, num_cond: usize, num_sim: usize) {
        assert!(num_cond <= self.cond_size);
        assert!(num_sim <= self.sim_size);
        self.n_cond = num_cond;
        self.n_sim = num_sim;
        self.n_total = num_cond + num_sim;

        self.l_mat
            .resize_with(self.n_total, self.n_total, |_, _| 0.0);
        self.w_vec.resize_with(self.n_total, 1, |_, _| 0.0);
        self.intermediate_mat
            .resize_with(self.n_sim, self.n_cond, |_, _| 0.0);
    }

    #[inline(always)]
    pub fn build_system<V, VT, SKB>(
        &mut self,
        cond_points: &[SKB::Support],
        values: &[f32],
        sim_points: &[SKB::Support],
        rng: &mut StdRng,
        vgram: &V,
    ) where
        V: VariogramModel<VT>,
        VT: SimdValue<Element = f32> + SimdRealField + Copy,
        SKB: SKBuilder,
    {
        let n_cond = cond_points.len();
        let n_sim = sim_points.len();
        self.set_dims(n_cond, n_sim);
        let _ = self.build_l_matrix::<_, _, SKB>(cond_points, sim_points, vgram);
        self.populate_w_vec(values, rng);
        self.compute_intermediate_mat();
    }

    #[inline(always)]
    pub fn simulate(&self) -> Vec<f32> {
        let mut sim_mat = Mat::zeros(self.n_sim, 1);

        //L_gd @ L_dd^-1 @ w_d
        matmul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.intermediate_mat.as_ref(),
            Conj::No,
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            Conj::No,
            None,
            1.0,
        );
        //L_gg @ w_g
        matmul::matvec::matvec_with_conj(
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
    pub fn set_dims_and_build_l_matrix<V, VT, SKB, MS>(
        &mut self,
        cond_points: &[SKB::Support],
        sim_points: &[SKB::Support],
        vgram: &V,
    ) where
        V: VariogramModel<VT>,
        VT: SimdValue<Element = f32> + SimdRealField + Copy,
        SKB: SKBuilder,
    {
        //set dimensions of LU system
        let n_cond = cond_points.len();
        let n_sim = sim_points.len();
        self.set_dims(n_cond, n_sim);

        //build L matrix
        // TODO: handle error
        let _ = self.build_l_matrix::<_, _, SKB>(cond_points, sim_points, vgram);
    }

    // #[inline(always)]
    // pub fn create_mini_system<V, VT, SKB, MS>(
    //     &mut self,
    //     cond_points: &[SKB::Support],
    //     sim_points: &[SKB::Support],
    //     vgram: &V,
    // ) -> MS
    // where
    //     V: VariogramModel<VT>,
    //     VT: SimdValue<Element = f32> + SimdRealField + Copy,
    //     SKB: SKBuilder,
    //     MS: SolvedLUSystem,
    // {
    //     //set dimensions of LU system
    //     let n_cond = cond_points.len();
    //     let n_sim = sim_points.len();
    //     self.set_dims(n_cond, n_sim);

    //     //build L matrix
    //     // TODO: handle error
    //     let _ = self.build_l_matrix::<_, _, SKB>(cond_points, sim_points, vgram);

    //     //Create mini system
    //     MS::from(self)
    // }
}

impl Clone for LUSystem {
    fn clone(&self) -> Self {
        Self::new(self.sim_size, self.cond_size)
    }
}

// impl<MS, VT> From<&mut LUSystem> for ModifiedMiniLUSystem<MS, VT>
// where
//     MS: MiniLUSystem + for<'a> From<&'a mut LUSystem>,
//     VT: ValueTransform<Vec<f32>>,
// {
//     fn from(lu: &mut LUSystem) -> Self {
//         Self {
//             system: MS::from(lu),
//             modifier,
//         }
//     }
// }

// pub struct NegativeFilteredMiniLUSystem<T>
// where
//     T: SolvedLUSystem,
// {
//     system: T,
// }

// impl<T> SolvedLUSystem for NegativeFilteredMiniLUSystem<T>
// where
//     T: SolvedLUSystem,
// {
//     fn populate_cond_values_est<I>(&mut self, values: I)
//     where
//         I: IntoIterator,
//         I::Item: Borrow<f32>,
//     {
//         self.system.populate_cond_values_est(values);
//     }

//     fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut StdRng)
//     where
//         I: IntoIterator,
//         I::Item: Borrow<f32>,
//     {
//         self.system.populate_cond_values_sim(values, rng)
//     }

//     fn estimate(&self) -> Vec<f32> {
//         self.system.estimate()
//     }

//     fn simulate(&self) -> Vec<f32> {
//         self.system.simulate()
//     }

//     fn weights(&self) -> MatRef<f32> {
//         self.system.weights()
//     }

//     fn weights_mut(&mut self) -> MatMut<f32> {
//         self.system.weights_mut()
//     }
// }

// impl<T> From<&mut LUSystem> for NegativeFilteredMiniLUSystem<T>
// where
//     T: SolvedLUSystem,
// {
//     fn from(lu: &mut LUSystem) -> Self {
//         let mut non_neg_sys = Self {
//             system: T::from(lu),
//         };

//         //set negative weights to zero
//         zipped!(non_neg_sys.weights_mut()).for_each(|unzipped!(mut v)| {
//             if v.read() < 0.0 {
//                 v.write(0.0);
//             }
//         });

//         //normalize weights so they sum to 1
//         for row in 0..non_neg_sys.weights().nrows() {
//             let mut sum = 0.0;
//             zipped!(non_neg_sys.weights_mut().row_mut(row))
//                 .for_each(|unzipped!(v)| sum += v.read());
//             if sum > 0.0 {
//                 zipped!(non_neg_sys.weights_mut().row_mut(row))
//                     .for_each(|unzipped!(mut v)| v.write(v.read() / sum));
//             }
//         }

//         non_neg_sys
//     }
// }

// #[cfg(test)]
// mod tests {

//     use nalgebra::{Point3, UnitQuaternion, Vector3};
//     use num_traits::Float;
//     use rand::SeedableRng;

//     use crate::{
//         kriging::simple_kriging::{SKPointSupportBuilder, SimpleKrigingSystem},
//         variography::model_variograms::spherical::SphericalVariogram,
//     };

//     use super::*;
//     #[test]
//     fn test_lu() {
//         let mut lu = LUSystem::new(2, 3);
//         let cond_points = vec![
//             Point3::new(0.0, 0.0, 0.0),
//             Point3::new(1.0, 0.0, 0.0),
//             Point3::new(0.0, 1.0, 0.0),
//         ];
//         let values = vec![0.0, 1.0, 2.0];
//         let sim_points = vec![Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0)];
//         let vgram_rot =
//             UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
//         let range = Vector3::new(2.0, 2.0, 2.0);
//         let sill = 1.0;

//         let vgram = SphericalVariogram::<f32>::new(range, sill, vgram_rot);
//         let mut rng = StdRng::seed_from_u64(0);
//         lu.build_system::<_, _, SKPointSupportBuilder>(
//             &cond_points,
//             &values,
//             &sim_points,
//             &mut rng,
//             &vgram,
//         );
//         let vals = lu.simulate();
//         println!("{:?}", vals);

//         let mut rng = StdRng::seed_from_u64(0);
//         let mut mini = lu.create_mini_system::<_, _, SKPointSupportBuilder, SolvedLUSKSystem>(
//             &cond_points,
//             &sim_points,
//             &vgram,
//         );
//         mini.populate_cond_values_sim(values.as_slice(), &mut rng);
//         let vals = mini.simulate();
//         println!("{:?}", vals);
//     }

//     #[test]
//     fn test_lu_ok_weights() {
//         let mut lu = LUSystem::new(2, 3);
//         let cond_points = vec![
//             Point3::new(0.0, 0.0, 0.0),
//             Point3::new(1.0, 0.0, 0.0),
//             Point3::new(0.0, 1.0, 0.0),
//         ];
//         let values = vec![0.0, 1.0, 2.0];
//         let sim_points = vec![Point3::new(1.0, 1.0, 1.0), Point3::new(0.0, 0.0, 1.0)];
//         let vgram_rot =
//             UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
//         let range = Vector3::new(2.0, 2.0, 2.0);
//         let sill = 1.0;

//         let vgram = SphericalVariogram::<f32>::new(range, sill, vgram_rot);

//         let mut rng = StdRng::seed_from_u64(0);
//         let mut mini = lu.create_mini_system::<_, _, SKPointSupportBuilder, SolvedLUOKSystem>(
//             &cond_points,
//             &sim_points,
//             &vgram,
//         );
//         mini.populate_cond_values_sim(values.as_slice(), &mut rng);
//         let vals = mini.estimate();
//         println!("{:?}", vals);
//     }

//     #[test]
//     fn submatrix() {
//         let mat = Mat::<f32>::zeros(5, 5);
//         let sub = mat.as_ref().submatrix(0, 0, 1, 1);
//         println!("{:?}", sub);
//     }

//     #[test]
//     fn sgs_vs_lu() {
//         let mut lu = LUSystem::new(1, 3);
//         let cond_points = vec![
//             Point3::new(0.0, 0.0, 0.0),
//             Point3::new(1.0, 0.0, 0.0),
//             Point3::new(0.0, 1.0, 0.0),
//         ];
//         let values = vec![0.0, 1.0, 2.0];
//         let sim_points = vec![Point3::new(0.0, 0.0, 1.0)];
//         let vgram_rot =
//             UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
//         let range = Vector3::new(200.0, 200.0, 200.0);
//         let sill = 1.0;

//         let vgram = SphericalVariogram::<f32>::new(range, sill, vgram_rot);

//         let mut rng = StdRng::seed_from_u64(0);
//         let mut mini = lu.create_mini_system::<_, _, SKPointSupportBuilder, SolvedLUSKSystem>(
//             &cond_points,
//             &sim_points,
//             &vgram,
//         );
//         mini.populate_cond_values_sim(values.as_slice(), &mut rng);
//         let val = mini.estimate()[0];

//         //let val: f32 = (0..10000).map(|_| mini.simulate()[0]).sum::<f32>() / 10000.0;
//         println!("{:?}", val);
//         println!("{:?}", mini.w_vec);

//         let mut system = SimpleKrigingSystem::new(cond_points.len());
//         system.build_system::<_, _, SKPointSupportBuilder>(
//             cond_points.as_slice(),
//             values.as_slice(),
//             &sim_points[0],
//             &vgram,
//         );

//         println!("mean: {}", system.estimate());
//     }

//     #[test]
//     fn test_lu_ok() {
//         let mut lu = LUSystem::new(2, 3);
//         let cond_points = vec![
//             Point3::new(0.0, 0.0, 0.0),
//             Point3::new(1.0, 0.0, 0.0),
//             Point3::new(0.0, 1.0, 0.0),
//         ];
//         let sim_points = vec![Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0)];
//         let vgram_rot =
//             UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
//         let range = Vector3::new(2.0, 2.0, 2.0);
//         let sill = 1.0;

//         let vgram = SphericalVariogram::<f32>::new(range, sill, vgram_rot);

//         let ok_sys = lu.create_mini_system::<_, _, SKPointSupportBuilder, SolvedLUOKSystem>(
//             cond_points.as_slice(),
//             sim_points.as_slice(),
//             &vgram,
//         );

//         println!("ok weights: {:?}", ok_sys.ok_weights);
//     }
// }
