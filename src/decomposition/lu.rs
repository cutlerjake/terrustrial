use crate::{kriging::simple_kriging::SKBuilder, variography::model_variograms::VariogramModel};

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
use nalgebra::{Point3, SimdRealField, SimdValue, Vector3, LU};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;

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
        }
    }

    #[inline(always)]
    fn build_l_matrix<V, T, SKB>(
        &mut self,
        cond_points: &[SKB::Support],
        sim_points: &[SKB::Support],
        vgram: &V,
    ) -> Result<(), CholeskyError>
    where
        V: VariogramModel<T>,
        SKB: SKBuilder,
        T: SimdValue<Element = f32> + SimdRealField + Copy,
    {
        let l_points = cond_points.into_iter().chain(sim_points.into_iter());

        SKB::build_cov_mat(&mut self.l_mat, l_points, vgram);

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
        let [l_dd, _, l_gd, _] = self.l_mat.as_mut().split_at(self.n_cond, self.n_cond);

        let mut l_gd_t = l_gd.transpose();

        //Want to compute L_gd @ L_dd^-1
        //avoid inverting L_dd by solving L_dd^T * intermediate^T = L_dg^T
        solve_upper_triangular_in_place(
            l_dd.as_ref().transpose(),
            l_gd_t.as_mut(),
            Parallelism::None,
        );

        self.intermediate_mat = l_gd_t.transpose().to_owned();
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
        self.build_l_matrix::<_, _, SKB>(cond_points, sim_points, vgram);
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
    pub fn create_mini_system<V, VT, SKB, MS>(
        &mut self,
        cond_points: &[SKB::Support],
        sim_points: &[SKB::Support],
        vgram: &V,
    ) -> MS
    where
        V: VariogramModel<VT>,
        VT: SimdValue<Element = f32> + SimdRealField + Copy,
        SKB: SKBuilder,
        MS: MiniLUSystem,
    {
        //set dimensions of LU system
        let n_cond = cond_points.len();
        let n_sim = sim_points.len();
        self.set_dims(n_cond, n_sim);

        //build L matrix
        // TODO: handle error
        let _ = self.build_l_matrix::<_, _, SKB>(cond_points, sim_points, vgram);

        //Create mini system
        MS::from(self)
    }
}

impl Clone for LUSystem {
    fn clone(&self) -> Self {
        Self::new(self.sim_size, self.cond_size)
    }
}

pub trait MiniLUSystem: for<'a> From<&'a mut LUSystem> {
    fn populate_cond_values_est(&mut self, values: &[f32]);
    fn populate_cond_values_sim(&mut self, values: &[f32], rng: &mut StdRng);
    fn estimate(&self) -> Vec<f32>;
    fn simulate(&self) -> Vec<f32>;
}

pub trait ValueTransform<T> {
    fn transform(value: T) -> T;
}

pub struct AverageTransfrom;

impl ValueTransform<Vec<f32>> for AverageTransfrom {
    fn transform(value: Vec<f32>) -> Vec<f32> {
        let sum: f32 = value.iter().sum();
        vec![sum / (value.len() as f32)]
    }
}

pub struct ModifiedMiniLUSystem<MS, VT>
where
    MS: MiniLUSystem,
    VT: ValueTransform<Vec<f32>>,
{
    system: MS,
    modified: std::marker::PhantomData<VT>,
}

impl<MS, VT> MiniLUSystem for ModifiedMiniLUSystem<MS, VT>
where
    MS: MiniLUSystem,
    VT: ValueTransform<Vec<f32>>,
{
    fn populate_cond_values_est(&mut self, values: &[f32]) {
        self.system.populate_cond_values_est(values);
    }

    fn populate_cond_values_sim(&mut self, values: &[f32], rng: &mut StdRng) {
        self.system.populate_cond_values_sim(values, rng)
    }

    fn estimate(&self) -> Vec<f32> {
        VT::transform(self.system.estimate())
    }

    fn simulate(&self) -> Vec<f32> {
        VT::transform(self.system.simulate())
    }
}

impl<MS, VT> From<&mut LUSystem> for ModifiedMiniLUSystem<MS, VT>
where
    MS: MiniLUSystem + for<'a> From<&'a mut LUSystem>,
    VT: ValueTransform<Vec<f32>>,
{
    fn from(lu: &mut LUSystem) -> Self {
        Self {
            system: MS::from(lu),
            modified: std::marker::PhantomData::default(),
        }
    }
}

pub struct MiniLUSKSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f32>,
    pub sk_weights: Mat<f32>,
    pub w_vec: Mat<f32>, // consider not storing w vec on this struct to avoid reallocating memory in hot loop
}

impl MiniLUSystem for MiniLUSKSystem {
    #[inline(always)]
    fn populate_cond_values_est(&mut self, values: &[f32]) {
        for (i, v) in values.iter().enumerate() {
            self.w_vec.write(i, 0, *v);
        }
    }
    #[inline(always)]
    fn populate_cond_values_sim(&mut self, values: &[f32], rng: &mut StdRng) {
        //populate w vector
        for (i, v) in values.iter().enumerate() {
            self.w_vec.write(i, 0, *v);
        }
        for i in values.len()..self.w_vec.nrows() {
            self.w_vec.write(i, 0, rng.sample(StandardNormal));
        }
    }

    #[inline(always)]
    fn estimate(&self) -> Vec<f32> {
        let mut est_mat = Mat::zeros(self.n_sim, 1);
        mul::matvec::matvec_with_conj(
            est_mat.as_mut(),
            self.sk_weights.as_ref(),
            Conj::No,
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            Conj::No,
            None,
            1.0,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..est_mat.nrows() {
            vals.push(est_mat.read(i, 0));
        }

        vals
    }

    #[inline(always)]
    fn simulate(&self) -> Vec<f32> {
        let mut sim_mat = Mat::zeros(self.n_sim, 1);
        mul::matvec::matvec_with_conj(
            sim_mat.as_mut(),
            self.sk_weights.as_ref(),
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

impl From<&mut LUSystem> for MiniLUSKSystem {
    fn from(lu: &mut LUSystem) -> Self {
        lu.compute_intermediate_mat();
        let l_gg = lu
            .l_mat
            .as_ref()
            .submatrix(lu.n_cond, lu.n_cond, lu.n_sim, lu.n_sim)
            .to_owned()
            .clone();
        let intermediate = lu.intermediate_mat.clone();
        let w = lu.w_vec.clone();
        let n_sim = lu.n_sim;
        let n_cond = lu.n_cond;
        Self {
            n_sim,
            n_cond,
            l_gg,
            sk_weights: intermediate,
            w_vec: w,
        }
    }
}

pub struct MiniLUOKSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f32>,
    pub ok_weights: Mat<f32>,
    pub w_vec: Mat<f32>, // consider not storing w vec on this struct to avoid reallocating memory in hot loop
}

impl MiniLUSystem for MiniLUOKSystem {
    #[inline(always)]
    fn populate_cond_values_est(&mut self, values: &[f32]) {
        for (i, v) in values.iter().enumerate() {
            self.w_vec.write(i, 0, *v);
        }
    }
    #[inline(always)]
    fn populate_cond_values_sim(&mut self, values: &[f32], rng: &mut StdRng) {
        //populate w vector
        for (i, v) in values.iter().enumerate() {
            self.w_vec.write(i, 0, *v);
        }
        for i in values.len()..self.w_vec.nrows() {
            self.w_vec.write(i, 0, rng.sample(StandardNormal));
        }
    }

    #[inline(always)]
    fn estimate(&self) -> Vec<f32> {
        let mut est_mat = Mat::zeros(self.n_sim, 1);
        mul::matvec::matvec_with_conj(
            est_mat.as_mut(),
            self.ok_weights.as_ref(),
            Conj::No,
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            Conj::No,
            None,
            1.0,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..est_mat.nrows() {
            vals.push(est_mat.read(i, 0));
        }

        vals
    }

    fn simulate(&self) -> Vec<f32> {
        todo!()
    }
}

impl From<&mut LUSystem> for MiniLUOKSystem {
    fn from(lu: &mut LUSystem) -> Self {
        // Constructs OK weights from SK weights
        // full derivation can be found
        // Kriging in a Global Neighborhood: Direct Sequential Simulation with a Large Number of Conditioning Data (Davis and Grivet 1984)
        // slight modifications have been made to accomodate simultaneous estimation and/or simulation of multiple points
        lu.compute_intermediate_mat();

        let ones = Mat::<f32>::from_fn(lu.n_cond, 1, |_, _| 1.0);

        let mut lambda_e = ones.clone();

        let mut stack = DynStack::new(&mut lu.buffer);

        // solve C_dd @ \lambda_e = ones
        faer_cholesky::llt::solve::solve_in_place_with_conj(
            lu.l_mat.as_ref().submatrix(0, 0, lu.n_cond, lu.n_cond),
            Conj::No,
            lambda_e.as_mut(),
            Parallelism::None,
            stack.rb_mut(),
        );

        //denom is sum of lambda_e
        let mut denom = 0.0;
        zipped!(lambda_e.as_ref()).for_each(|v| denom += v.read());

        //stores 1 - ones^T @ \lambda_sk / denom
        let mut frac = Mat::zeros(lu.n_sim, 1);
        // frac = lambda_sk @ e
        mul::matvec::matvec_with_conj(
            frac.as_mut(),
            lu.intermediate_mat.as_ref(),
            Conj::No,
            ones.as_ref(),
            Conj::No,
            None,
            1.0,
        );

        //scale and shift frac
        zipped!(frac.as_mut()).for_each(|mut v| v.write((1f32 - v.read()) / denom));

        // lambda_ok = lmabda_sk + frac @ lambda e
        let mut ok = lu.intermediate_mat.clone();
        mul::matmul(
            ok.as_mut().transpose(),
            lambda_e.as_ref(),
            frac.as_ref().transpose(),
            Some(1.0),
            1.0,
            Parallelism::None,
        );

        let l_gg = lu
            .l_mat
            .as_ref()
            .submatrix(lu.n_cond, lu.n_cond, lu.n_sim, lu.n_sim)
            .to_owned()
            .clone();
        let w = lu.w_vec.clone();
        let n_sim = lu.n_sim;
        let n_cond = lu.n_cond;

        Self {
            n_sim,
            n_cond,
            l_gg,
            ok_weights: ok,
            w_vec: w,
        }
    }
}

#[cfg(test)]
mod tests {

    use nalgebra::UnitQuaternion;
    use num_traits::Float;
    use rand::SeedableRng;

    use crate::{
        kriging::simple_kriging::{SKPointSupportBuilder, SimpleKrigingSystem},
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
        let range = Vector3::new(2.0, 2.0, 2.0);
        let sill = 1.0;
        let nugget = 0.1;

        let vgram = SphericalVariogram::<f32>::new(range, sill, nugget, vgram_rot);
        let mut rng = StdRng::seed_from_u64(0);
        lu.build_system::<_, _, SKPointSupportBuilder>(
            &cond_points,
            &values,
            &sim_points,
            &mut rng,
            &vgram,
        );
        let vals = lu.simulate();
        println!("{:?}", vals);

        let mut rng = StdRng::seed_from_u64(0);
        let mut mini = lu.create_mini_system::<_, _, SKPointSupportBuilder, MiniLUSKSystem>(
            &cond_points,
            &sim_points,
            &vgram,
        );
        mini.populate_cond_values_sim(values.as_slice(), &mut rng);
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
        let range = Vector3::new(200.0, 200.0, 200.0);
        let sill = 1.0;
        let nugget = 0.1;

        let vgram = SphericalVariogram::<f32>::new(range, sill, nugget, vgram_rot);
        let mut rng = StdRng::seed_from_u64(0);

        let mut rng = StdRng::seed_from_u64(0);
        let mut mini = lu.create_mini_system::<_, _, SKPointSupportBuilder, MiniLUSKSystem>(
            &cond_points,
            &sim_points,
            &vgram,
        );
        mini.populate_cond_values_sim(values.as_slice(), &mut rng);
        let val = mini.estimate()[0];

        //let val: f32 = (0..10000).map(|_| mini.simulate()[0]).sum::<f32>() / 10000.0;
        println!("{:?}", val);
        println!("{:?}", mini.w_vec);

        let mut system = SimpleKrigingSystem::new(cond_points.len());
        system.build_system::<_, _, SKPointSupportBuilder>(
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
        let range = Vector3::new(2.0, 2.0, 2.0);
        let sill = 1.0;
        let nugget = 0.1;

        let vgram = SphericalVariogram::<f32>::new(range, sill, nugget, vgram_rot);
        let mut rng = StdRng::seed_from_u64(0);

        let ok_sys = lu.create_mini_system::<_, _, SKPointSupportBuilder, MiniLUOKSystem>(
            cond_points.as_slice(),
            sim_points.as_slice(),
            &vgram,
        );

        println!("ok weights: {:?}", ok_sys.ok_weights);
    }
}
