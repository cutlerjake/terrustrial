use std::borrow::Borrow;

use faer::{MatMut, MatRef};
use rand::rngs::StdRng;

use super::solved_systems::SolvedLUSystem;

pub mod mean_transform;

pub trait ValueTransform<T> {
    fn forward_transform(&self, value: &T) -> T;
    fn backward_transform(&self, value: &T) -> T;
}

#[derive(Clone)]
pub struct ModifiedMiniLUSystem<MS, VT>
where
    MS: SolvedLUSystem,
    VT: ValueTransform<f32>,
{
    system: MS,
    modifier: VT,
}

impl<MS, VT> SolvedLUSystem for ModifiedMiniLUSystem<MS, VT>
where
    MS: SolvedLUSystem + Clone,
    VT: ValueTransform<f32> + Clone,
{
    fn populate_cond_values_est<I>(&mut self, values: I)
    where
        I: IntoIterator,
        I::Item: Borrow<f32>,
    {
        self.system.populate_cond_values_est(
            values
                .into_iter()
                .map(|v| self.modifier.forward_transform(v.borrow())),
        );
    }

    fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut StdRng)
    where
        I: IntoIterator,
        I::Item: Borrow<f32>,
    {
        self.system.populate_cond_values_sim(
            values
                .into_iter()
                .map(|v| self.modifier.forward_transform(v.borrow())),
            rng,
        )
    }

    fn estimate(&self) -> Vec<f32> {
        self.system
            .estimate()
            .iter()
            .map(|v| self.modifier.backward_transform(v))
            .collect()
    }

    fn simulate(&self) -> Vec<f32> {
        self.system
            .simulate()
            .iter()
            .map(|v| self.modifier.backward_transform(v))
            .collect()
    }

    fn weights(&self) -> MatRef<f32> {
        self.system.weights()
    }

    fn weights_mut(&mut self) -> MatMut<f32> {
        self.system.weights_mut()
    }
}
