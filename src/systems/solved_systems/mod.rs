use std::borrow::Borrow;

use faer::{MatMut, MatRef};
use rand::rngs::StdRng;

use super::lu::LUSystem;

pub mod negative_weight_filtered_system;
pub mod ok_system;
pub mod sk_system;

pub trait SolvedSystemBuilder: Clone + Send + Sync {
    type SolvedSystem: SolvedLUSystem + Send;
    type Error;
    fn build(&self, system: &mut LUSystem) -> Result<Self::SolvedSystem, Self::Error>;
}

pub trait SolvedLUSystem: Clone {
    fn populate_cond_values_est<I>(&mut self, values: I)
    where
        I: IntoIterator,
        I::Item: Borrow<f64>;
    fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut StdRng)
    where
        I: IntoIterator,
        I::Item: Borrow<f64>;
    fn estimate(&self) -> Vec<f64>;
    fn simulate(&self) -> Vec<f64>;

    fn weights(&self) -> MatRef<f64>;
    fn weights_mut(&mut self) -> MatMut<f64>;
}
