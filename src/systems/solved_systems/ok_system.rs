use std::borrow::Borrow;

use dyn_stack::MemStack;
use faer::{
    linalg::{cholesky, matmul, solvers::LltError},
    unzip, zip, Accum, Mat, MatMut, MatRef, Par,
};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;

use crate::systems::lu::LUSystem;

use super::{SolvedLUSystem, SolvedSystemBuilder};

#[derive(Clone)]
pub struct SolvedLUOKSystemBuilder;

impl SolvedSystemBuilder for SolvedLUOKSystemBuilder {
    type SolvedSystem = SolvedLUOKSystem;
    type Error = LltError;
    fn build(&self, system: &mut LUSystem) -> Result<Self::SolvedSystem, Self::Error> {
        SolvedLUOKSystem::try_from(system)
    }
}

#[derive(Clone)]
pub struct SolvedLUOKSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f64>,
    pub ok_weights: Mat<f64>,
    pub w_vec: Mat<f64>, // consider not storing w vec on this struct to avoid reallocating memory in hot loop
}

impl SolvedLUSystem for SolvedLUOKSystem {
    #[inline(always)]
    fn populate_cond_values_est<I>(&mut self, values: I)
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        for (i, v) in values.into_iter().enumerate() {
            *self.w_vec.get_mut(i, 0) = *v.borrow();
        }
    }
    #[inline(always)]
    fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut StdRng)
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        //populate w vector
        let mut count = 0;
        for (i, v) in values.into_iter().enumerate() {
            *self.w_vec.get_mut(i, 0) = *v.borrow();
            count += 1;
        }
        for i in count..self.w_vec.nrows() {
            *self.w_vec.get_mut(i, 0) = rng.sample(StandardNormal);
        }
    }

    #[inline(always)]
    fn estimate(&self) -> Vec<f64> {
        let mut est_mat = Mat::zeros(self.n_sim, 1);
        matmul::matmul(
            est_mat.as_mut(),
            Accum::Replace,
            self.ok_weights.as_ref(),
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            1.0,
            Par::Seq,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..est_mat.nrows() {
            vals.push(*est_mat.get_mut(i, 0));
        }

        vals
    }

    fn simulate(&self) -> Vec<f64> {
        unimplemented!("simulate not implemented for OK system")
    }

    fn weights(&self) -> MatRef<f64> {
        self.ok_weights.as_ref()
    }

    fn weights_mut(&mut self) -> MatMut<f64> {
        self.ok_weights.as_mut()
    }
}

impl TryFrom<&mut LUSystem> for SolvedLUOKSystem {
    type Error = LltError;
    fn try_from(lu: &mut LUSystem) -> Result<Self, Self::Error> {
        // Constructs OK weights from SK weights
        // full derivation can be found
        // Kriging in a Global Neighborhood: Direct Sequential Simulation with a Large Number of Conditioning Data (Davis and Grivet 1984)
        // slight modifications have been made to accomodate simultaneous estimation and/or simulation of multiple points
        let _l = lu.l_mat.clone();
        let Ok(_) = lu.compute_l_matrix() else {
            print!("{:?}", _l);
            panic!()
        };
        lu.compute_intermediate_mat();

        let ones = Mat::<f64>::from_fn(lu.n_cond, 1, |_, _| 1.0);

        let mut lambda_e = ones.clone();
        let stack = MemStack::new(&mut lu.mem_buffer);

        // solve C_dd @ \lambda_e = ones
        cholesky::llt::solve::solve_in_place(
            lu.l_mat.as_ref().submatrix(0, 0, lu.n_cond, lu.n_cond),
            lambda_e.as_mut(),
            Par::Seq,
            stack,
        );

        //denom is sum of lambda_e
        let denom = lambda_e.sum();

        //stores 1 - ones^T @ \lambda_sk / denom
        let mut frac = Mat::zeros(lu.n_sim, 1);
        // frac = lambda_sk @ e
        matmul::matmul(
            frac.as_mut(),
            Accum::Replace,
            lu.intermediate_mat.as_ref(),
            ones.as_ref(),
            1.0,
            Par::Seq,
        );

        //scale and shift frac
        zip!(frac.as_mut()).for_each(|unzip!(v)| *v = (1.0 - *v) / denom);

        // lambda_ok = lmabda_sk + frac @ lambda e
        let mut ok = lu.intermediate_mat.clone();
        matmul::matmul(
            ok.as_mut().transpose_mut(),
            Accum::Add,
            lambda_e.as_ref(),
            frac.as_ref().transpose(),
            1.0,
            Par::Seq,
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

        Ok(Self {
            n_sim,
            n_cond,
            l_gg,
            ok_weights: ok,
            w_vec: w,
        })
    }
}
