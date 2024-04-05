use std::borrow::Borrow;

use dyn_stack::{PodStack, ReborrowMut};
use faer::{
    linalg::{cholesky, matmul},
    unzipped, zipped, Conj, Mat, MatMut, MatRef, Parallelism,
};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;

use crate::decomposition::lu::LUSystem;

use super::{SolvedLUSystem, SolvedSystemBuilder};

#[derive(Clone)]
pub struct SolvedLUOKSystemBuilder;

impl SolvedSystemBuilder for SolvedLUOKSystemBuilder {
    type SolvedSystem = SolvedLUOKSystem;

    fn build(&self, system: &mut LUSystem) -> Self::SolvedSystem {
        SolvedLUOKSystem::from(system)
    }
}

#[derive(Clone)]
pub struct SolvedLUOKSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f32>,
    pub ok_weights: Mat<f32>,
    pub w_vec: Mat<f32>, // consider not storing w vec on this struct to avoid reallocating memory in hot loop
}

impl SolvedLUSystem for SolvedLUOKSystem {
    #[inline(always)]
    fn populate_cond_values_est<I>(&mut self, values: I)
    where
        I: IntoIterator,
        I::Item: Borrow<f32>,
    {
        for (i, v) in values.into_iter().enumerate() {
            self.w_vec.write(i, 0, *v.borrow());
        }
    }
    #[inline(always)]
    fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut StdRng)
    where
        I: IntoIterator,
        I::Item: Borrow<f32>,
    {
        //populate w vector
        let mut count = 0;
        for (i, v) in values.into_iter().enumerate() {
            self.w_vec.write(i, 0, *v.borrow());
            count += 1;
        }
        for i in count..self.w_vec.nrows() {
            self.w_vec.write(i, 0, rng.sample(StandardNormal));
        }
    }

    #[inline(always)]
    fn estimate(&self) -> Vec<f32> {
        let mut est_mat = Mat::zeros(self.n_sim, 1);
        matmul::matvec::matvec_with_conj(
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

    fn weights(&self) -> MatRef<f32> {
        self.ok_weights.as_ref()
    }

    fn weights_mut(&mut self) -> MatMut<f32> {
        self.ok_weights.as_mut()
    }
}

impl From<&mut LUSystem> for SolvedLUOKSystem {
    fn from(lu: &mut LUSystem) -> Self {
        // Constructs OK weights from SK weights
        // full derivation can be found
        // Kriging in a Global Neighborhood: Direct Sequential Simulation with a Large Number of Conditioning Data (Davis and Grivet 1984)
        // slight modifications have been made to accomodate simultaneous estimation and/or simulation of multiple points
        lu.compute_intermediate_mat();

        let ones = Mat::<f32>::from_fn(lu.n_cond, 1, |_, _| 1.0);

        let mut lambda_e = ones.clone();

        let mut stack = PodStack::new(&mut lu.buffer);

        // solve C_dd @ \lambda_e = ones
        cholesky::llt::solve::solve_in_place_with_conj(
            lu.l_mat.as_ref().submatrix(0, 0, lu.n_cond, lu.n_cond),
            Conj::No,
            lambda_e.as_mut(),
            Parallelism::None,
            stack.rb_mut(),
        );

        //denom is sum of lambda_e
        let mut denom = 0.0;
        zipped!(lambda_e.as_ref()).for_each(|unzipped!(v)| denom += v.read());

        //stores 1 - ones^T @ \lambda_sk / denom
        let mut frac = Mat::zeros(lu.n_sim, 1);
        // frac = lambda_sk @ e
        matmul::matvec::matvec_with_conj(
            frac.as_mut(),
            lu.intermediate_mat.as_ref(),
            Conj::No,
            ones.as_ref(),
            Conj::No,
            None,
            1.0,
        );

        //scale and shift frac
        zipped!(frac.as_mut()).for_each(|unzipped!(mut v)| v.write((1f32 - v.read()) / denom));

        // lambda_ok = lmabda_sk + frac @ lambda e
        let mut ok = lu.intermediate_mat.clone();
        matmul::matmul(
            ok.as_mut().transpose_mut(),
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
