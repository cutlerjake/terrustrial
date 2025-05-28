use crate::geometry::support::Support;
use crate::variography::model_variograms::composite::CompositeVariogram;
use dyn_stack::{MemBuffer, MemStack};

use faer::{
    linalg::{
        cholesky::{
            self,
            llt::factor::{LltInfo, LltRegularization},
        },
        matmul::{self},
        solvers::LltError,
        triangular_solve::solve_upper_triangular_in_place,
    },
    Accum, Mat, Par, Spec,
};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;
use ultraviolet::DVec3;

use super::system_builder::SKGeneralBuilder;

pub struct LUSystem {
    pub l_mat: Mat<f64>,
    pub w_vec: Mat<f64>,
    pub intermediate_mat: Mat<f64>,
    pub mem_buffer: MemBuffer,
    pub n_sim: usize,
    pub n_cond: usize,
    pub n_total: usize,
    pub sim_size: usize,
    pub cond_size: usize,
}

impl LUSystem {
    /// Creates a new LU system with the specified number of conditioning and simulation points
    /// The system will have dimensions of n_cond + n_sim
    pub fn new(n_sim: usize, n_cond: usize) -> Self {
        // L matrix will have dimensions of n_cond + n_Sim
        let n_total = n_sim + n_cond;

        //create a buffer large enough to compute cholesky in place
        let cholesky_required_mem = cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
            n_total,
            Par::Seq,
            Spec::default(),
        );
        let mem_buffer = MemBuffer::new(cholesky_required_mem);

        Self {
            l_mat: Mat::zeros(n_total, n_total),
            w_vec: Mat::zeros(n_total, 1),
            intermediate_mat: Mat::zeros(n_sim, n_cond),
            mem_buffer,
            n_sim,
            n_cond,
            n_total,
            sim_size: n_sim,
            cond_size: n_cond,
        }
    }

    /// Builds the covariance matrix for the system.
    /// The covariance matrix is built from the conditioning and simulation points according to the structure of the SKBuilder.
    /// The variogram model is used to compute the covariance between points.
    /// Note: only the lower triangle of the covariance matrix is required.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn build_cov_matrix(
        &mut self,
        n_cond: usize,
        n_sim: usize,
        supports: &[Support],
        vgram: &CompositeVariogram,
        h_buffer: &mut Vec<DVec3>,
        pt_buffer: &mut Vec<DVec3>,
        var_buffer: &mut Vec<f64>,
        ind_buffer: &mut Vec<usize>,
    ) {
        //set dimensions of LU system
        self.set_dims(n_cond, n_sim);

        SKGeneralBuilder::build_cov_mat(
            &mut self.l_mat,
            supports,
            vgram,
            h_buffer,
            pt_buffer,
            var_buffer,
            ind_buffer,
        );
    }

    /// Computes the L matrix from the covariance matrix.
    /// The L matrix is the Cholesky decomposition of the covariance matrix.
    /// The L matrix is stored in the lower triangle of the covariance matrix.
    /// The upper triangle of the covariance matrix is not used.
    #[inline(always)]
    pub(crate) fn compute_l_matrix(&mut self) -> Result<LltInfo, LltError> {
        let stack = MemStack::new(&mut self.mem_buffer);

        //compute cholseky decomposition of L matrix
        cholesky::llt::factor::cholesky_in_place(
            self.l_mat.as_mut(),
            LltRegularization::default(),
            Par::Seq,
            stack,
            Spec::default(),
        )
    }

    /// Build and compute the L matrix for the system.
    #[inline(always)]
    pub(crate) fn build_l_matrix(
        &mut self,
        supports: &[Support],
        vgram: &CompositeVariogram,
        h_buffer: &mut Vec<DVec3>,
        pt_buffer: &mut Vec<DVec3>,
        var_buffer: &mut Vec<f64>,
        ind_buffer: &mut Vec<usize>,
    ) -> Result<LltInfo, LltError> {
        SKGeneralBuilder::build_cov_mat(
            &mut self.l_mat,
            supports,
            vgram,
            h_buffer,
            pt_buffer,
            var_buffer,
            ind_buffer,
        );
        self.compute_l_matrix()
    }

    /// Populates the w vector with conditioning values and random numbers.
    /// The conditioning values are the first n_cond values in the w vector.
    /// The remaining values are filled with random numbers.
    #[inline(always)]
    fn populate_w_vec(&mut self, values: &[f64], rng: &mut StdRng) {
        //populate w vector with conditioning points
        for (i, v) in values.iter().enumerate() {
            *self.w_vec.get_mut(i, 0) = *v;
        }
        // fill remaining values with random numbers
        for i in values.len()..self.w_vec.nrows() {
            *self.w_vec.get_mut(i, 0) = rng.sample(StandardNormal);
        }
    }

    /// Solve the kriging system to compute weights for each datum for each estimation node.
    /// The weights are stored in the intermediate matrix.
    /// Each row of the intermediate matrix corresponds to the weights for a single estimation node.
    #[inline(always)]
    pub(crate) fn compute_intermediate_mat(&mut self) {
        let (l_dd, _, l_gd, _) = self.l_mat.as_mut().split_at_mut(self.n_cond, self.n_cond);

        let mut l_gd_t = l_gd.transpose_mut();

        //Want to compute L_gd @ L_dd^-1
        //avoid inverting L_dd by solving L_dd^T * intermediate^T = L_dg^T
        solve_upper_triangular_in_place(l_dd.as_ref().transpose(), l_gd_t.as_mut(), Par::Seq);

        self.intermediate_mat = l_gd_t.transpose_mut().to_owned();
    }

    /// Set the dimensions of the LU system.
    /// If the size of the system increases, new values are initialized to zero.
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

    /// Build and solve the LU system for the given conditioning points, values, and simulation points.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn build_and_solve_system(
        &mut self,
        n_cond: usize,
        n_sim: usize,
        supports: &[Support],
        values: &[f64],
        rng: &mut StdRng,
        vgram: &CompositeVariogram,
        h_buffer: &mut Vec<DVec3>,
        pt_buffer: &mut Vec<DVec3>,
        var_buffer: &mut Vec<f64>,
        ind_buffer: &mut Vec<usize>,
    ) {
        self.set_dims(n_cond, n_sim);
        let _ = self.build_l_matrix(supports, vgram, h_buffer, pt_buffer, var_buffer, ind_buffer);
        self.populate_w_vec(values, rng);
        self.compute_intermediate_mat();
    }

    /// Simulate the kriging system by sampling the distribution for each node.
    #[inline(always)]
    pub fn simulate(&self) -> Vec<f64> {
        let mut sim_mat = Mat::zeros(self.n_sim, 1);

        //L_gd @ L_dd^-1 @ w_d
        matmul::matmul(
            sim_mat.as_mut(),
            Accum::Replace,
            self.intermediate_mat.as_ref(),
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            1.0,
            Par::Seq,
        );
        //L_gg @ w_g
        matmul::matmul(
            sim_mat.as_mut(),
            Accum::Add,
            self.l_mat
                .as_ref()
                .submatrix(self.n_cond, self.n_cond, self.n_sim, self.n_sim),
            self.w_vec.as_ref().submatrix(self.n_cond, 0, self.n_sim, 1),
            1.0,
            Par::Seq,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..sim_mat.nrows() {
            vals.push(*sim_mat.get(i, 0));
        }

        vals
    }

    /// Size the system appropriately and build the L matrix for the given conditioning and simulation points.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn set_dims_and_build_l_matrix(
        &mut self,
        n_cond: usize,
        n_sim: usize,
        supports: &[Support],
        vgram: &CompositeVariogram,
        h_buffer: &mut Vec<DVec3>,
        pt_buffer: &mut Vec<DVec3>,
        var_buffer: &mut Vec<f64>,
        ind_buffer: &mut Vec<usize>,
    ) {
        //set dimensions of LU system
        self.set_dims(n_cond, n_sim);

        //build L matrix
        // TODO: handle error
        let _ = self.build_l_matrix(supports, vgram, h_buffer, pt_buffer, var_buffer, ind_buffer);
    }
}

impl Clone for LUSystem {
    fn clone(&self) -> Self {
        Self::new(self.sim_size, self.cond_size)
    }
}

// impl<MS, VT> From<&mut LUSystem> for ModifiedMiniLUSystem<MS, VT>
// where
//     MS: MiniLUSystem + for<'a> From<&'a mut LUSystem>,
//     VT: ValueTransform<Vec<f64>>,
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
