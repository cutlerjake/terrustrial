use std::borrow::Borrow;

use faer::{MatMut, MatRef};
use rand::rngs::StdRng;

use super::{
    lu::LUSystem,
    solved_systems::{SolvedLUSystem, SolvedSystemBuilder},
};

pub mod mean_transform;

pub trait ValueTransform<T> {
    fn forward_transform(&self, value: &T) -> T;
    fn backward_transform(&self, value: &T) -> T;
}

/// A builder for a solved system that applies a modifier to the values.
///
/// Forward transorms are applied to conditioining data before estimation and simulation.
/// Backward transforms are applied to the estimated and simulated values.
#[derive(Clone)]
pub struct ModifiedSolvedLUSystemBuilder<B, T> {
    builder: B,
    modifier: T,
}

impl<B, T> ModifiedSolvedLUSystemBuilder<B, T> {
    pub fn new(builder: B, modifier: T) -> Self {
        Self { builder, modifier }
    }
}

impl<B, VT> SolvedSystemBuilder for ModifiedSolvedLUSystemBuilder<B, VT>
where
    B: SolvedSystemBuilder + Send,
    VT: ValueTransform<f64> + Clone + Send + Sync,
{
    type SolvedSystem = ModifiedSolvedLUSystem<B::SolvedSystem, VT>;
    type Error = B::Error;

    fn build(&self, system: &mut LUSystem) -> Result<Self::SolvedSystem, Self::Error> {
        // Build the underlying system
        // Nothing more needs to happen here as the underlying kriging systems
        // depend only on the spatial configuration of the data, and not the values themselves.
        let system = self.builder.build(system)?;
        Ok(ModifiedSolvedLUSystem {
            system,
            modifier: self.modifier.clone(),
        })
    }
}

#[derive(Clone)]
pub struct ModifiedSolvedLUSystem<MS, VT>
where
    MS: SolvedLUSystem,
    VT: ValueTransform<f64>,
{
    system: MS,
    modifier: VT,
}

impl<MS, VT> SolvedLUSystem for ModifiedSolvedLUSystem<MS, VT>
where
    MS: SolvedLUSystem + Clone,
    VT: ValueTransform<f64> + Clone,
{
    /// Populate the conditional values for the estimator.
    /// The values are transformed using the forward transform of the modifier.
    fn populate_cond_values_est<I>(&mut self, values: I)
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        self.system.populate_cond_values_est(
            values
                .into_iter()
                .map(|v| self.modifier.forward_transform(v.borrow())),
        );
    }

    /// Populate the conditional values for the simulator.
    /// The values are transformed using the forward transform of the modifier.
    fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut StdRng)
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        self.system.populate_cond_values_sim(
            values
                .into_iter()
                .map(|v| self.modifier.forward_transform(v.borrow())),
            rng,
        )
    }

    /// Estimate the system.
    /// The values are transformed using the backward transform of the modifier.
    fn estimate(&self) -> Vec<f64> {
        self.system
            .estimate()
            .iter()
            .map(|v| self.modifier.backward_transform(v))
            .collect()
    }

    /// Simulate the system.
    /// The values are transformed using the backward transform of the modifier.
    fn simulate(&self) -> Vec<f64> {
        self.system
            .simulate()
            .iter()
            .map(|v| self.modifier.backward_transform(v))
            .collect()
    }

    fn weights(&self) -> MatRef<f64> {
        self.system.weights()
    }

    fn weights_mut(&mut self) -> MatMut<f64> {
        self.system.weights_mut()
    }
}
