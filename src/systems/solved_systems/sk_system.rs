use std::borrow::Borrow;

use faer::{
    linalg::{
        matmul::{self, triangular::BlockStructure},
        solvers::LltError,
    },
    Accum, Mat, MatMut, MatRef, Par,
};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;

use crate::systems::lu::LUSystem;

use super::{SolvedLUSystem, SolvedSystemBuilder};

#[derive(Clone)]
pub struct SolvedLUSKSystemBuilder;

impl SolvedSystemBuilder for SolvedLUSKSystemBuilder {
    type SolvedSystem = SolvedLUSKSystem;
    type Error = LltError;

    fn build(&self, system: &mut LUSystem) -> Result<Self::SolvedSystem, Self::Error> {
        SolvedLUSKSystem::try_from(system)
    }
}

#[derive(Clone)]
pub struct SolvedLUSKSystem {
    pub n_sim: usize,
    pub n_cond: usize,
    pub l_gg: Mat<f64>,
    pub sk_weights: Mat<f64>,
    pub w_vec: Mat<f64>, // consider not storing w vec on this struct to avoid reallocating memory in hot loop
}

impl SolvedLUSystem for SolvedLUSKSystem {
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
            self.sk_weights.as_ref(),
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            1.0,
            Par::Seq,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..est_mat.nrows() {
            vals.push(*est_mat.get(i, 0));
        }

        vals
    }

    #[inline(always)]
    fn simulate(&self) -> Vec<f64> {
        let mut sim_mat = Mat::zeros(self.n_sim, 1);
        matmul::matmul(
            sim_mat.as_mut(),
            Accum::Replace,
            self.sk_weights.as_ref(),
            self.w_vec.as_ref().submatrix(0, 0, self.n_cond, 1),
            1.0,
            Par::Seq,
        );

        faer::linalg::matmul::triangular::matmul(
            sim_mat.as_mut(),
            BlockStructure::Rectangular,
            Accum::Add,
            self.l_gg.as_ref(),
            BlockStructure::TriangularLower,
            self.w_vec.as_ref().submatrix(self.n_cond, 0, self.n_sim, 1),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );

        let mut vals = Vec::with_capacity(self.n_sim);
        for i in 0..sim_mat.nrows() {
            vals.push(*sim_mat.get(i, 0));
        }

        vals
    }

    fn weights(&self) -> MatRef<f64> {
        self.sk_weights.as_ref()
    }

    fn weights_mut(&mut self) -> MatMut<f64> {
        self.sk_weights.as_mut()
    }
}

impl TryFrom<&mut LUSystem> for SolvedLUSKSystem {
    type Error = LltError;
    fn try_from(lu: &mut LUSystem) -> Result<Self, Self::Error> {
        lu.compute_l_matrix()?;
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
        Ok(Self {
            n_sim,
            n_cond,
            l_gg,
            sk_weights: intermediate,
            w_vec: w,
        })
    }
}
