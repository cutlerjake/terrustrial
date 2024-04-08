use std::borrow::Borrow;

use faer::{MatMut, MatRef};
use rand::rngs::StdRng;

use super::lu::LUSystem;

pub mod negative_weight_filtered_system;
pub mod ok_system;
pub mod sk_system;

pub trait SolvedSystemBuilder: Clone + Send {
    type SolvedSystem: SolvedLUSystem;
    fn build(&self, system: &mut LUSystem) -> Self::SolvedSystem;
}

pub trait SolvedLUSystem: Clone {
    fn populate_cond_values_est<I>(&mut self, values: I)
    where
        I: IntoIterator,
        I::Item: Borrow<f32>;
    fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut StdRng)
    where
        I: IntoIterator,
        I::Item: Borrow<f32>;
    fn estimate(&self) -> Vec<f32>;
    fn simulate(&self) -> Vec<f32>;

    fn weights(&self) -> MatRef<f32>;
    fn weights_mut(&mut self) -> MatMut<f32>;
}
