use faer::{unzip, zip};

use crate::systems::lu::LUSystem;

use super::{SolvedLUSystem, SolvedSystemBuilder};

#[derive(Clone)]
pub struct SolvedNegativeWeightFilteredSystemBuilder<S> {
    system: S,
}

impl<S> SolvedNegativeWeightFilteredSystemBuilder<S> {
    pub fn new(system: S) -> Self {
        SolvedNegativeWeightFilteredSystemBuilder { system }
    }
}

impl<S> SolvedSystemBuilder for SolvedNegativeWeightFilteredSystemBuilder<S>
where
    S: SolvedSystemBuilder,
{
    type SolvedSystem = SolvedNegativeWeightFilteredSystem<S::SolvedSystem>;
    type Error = S::Error;

    /// Clayton V. Deutsch,
    /// Correcting for negative weights in ordinary kriging,
    /// Computers & Geosciences,
    /// Volume 22, Issue 7,
    /// 1996,
    /// Pages 765-773,
    /// ISSN 0098-3004,
    /// https://doi.org/10.1016/0098-3004(96)00005-2.
    /// (https://www.sciencedirect.com/science/article/pii/0098300496000052)
    fn build(&self, lu_system: &mut LUSystem) -> Result<Self::SolvedSystem, Self::Error> {
        // 1. Determine estimation node - data covariance values
        // 2. Compute weights
        // 3. Compute average absolute magnitude of negative weights
        // 4. Compute the average covariance between the estimation nodes and the nodes with negative weights
        // 5. Assign a weight of 0 to all negative weights
        // 6. Assign a weight of 0 to all nodes satifying the following conditions:
        //    - The absolute magnitude of the weight is less than the average absolute magnitude of negative weights
        //    - The node has a covariance less than the average covariance between the estimation nodes and the nodes with negative weights
        // 7 Normalize the weights

        // 1. Determine estimation node - data covariance values
        let cov = lu_system.l_mat.clone();

        // 2. Compute weights (stored in intermediate_mat)
        // lu_system.compute_l_matrix();

        let mut sys = self.system.build(lu_system)?;
        // 3, 4. Compute average absolute magnitude of negative weights and covariances (for each estimation node)
        let (avg_abs_neg_weights, avg_neg_cov): (Vec<_>, Vec<_>) = (0..lu_system.n_sim)
            .map(|i| {
                let node_weights = sys.weights().row(i);
                let node_covariances = cov
                    .as_ref()
                    .row(i + lu_system.n_cond)
                    .subcols(0, lu_system.n_cond);

                let mut weight_sum = 0.0;
                let mut covariance_sum = 0.0;
                let mut count = 0;
                zip!(node_weights, node_covariances).for_each(|unzip!(w, c)| {
                    if *w < 0.0 {
                        weight_sum += w.abs();
                        covariance_sum += *c;
                        count += 1;
                    }
                });

                (weight_sum / count as f64, covariance_sum / count as f64)
            })
            .unzip();

        // 5, 6, 7. Assign a weight of 0 to all nodes satifying the conditions noted above and compute new normalized weights
        (0..lu_system.n_sim).for_each(|i| {
            let node_weights = sys.weights_mut().row_mut(i);
            let node_covariances = cov
                .as_ref()
                .row(i + lu_system.n_cond)
                .subcols(0, lu_system.n_cond);

            let mut weight = 0.0;
            zip!(node_weights, node_covariances).for_each(|unzip!(w, c)| {
                if *w < 0.0 || w.abs() < avg_abs_neg_weights[i] && *c < avg_neg_cov[i] {
                    *w = 0.0;
                }
                weight += *w;
            });

            let node_weights = lu_system.intermediate_mat.as_mut().row_mut(i);
            zip!(node_weights).for_each(|unzip!(w)| *w /= weight);
        });

        Ok(SolvedNegativeWeightFilteredSystem { system: sys })
    }
}

#[derive(Clone)]
pub struct SolvedNegativeWeightFilteredSystem<S>
where
    S: Clone,
{
    system: S,
}

/// After construction of the SolvedNegativeWeightFilteredSystem, all negative weights have been addressedd
/// and we can simply forward all method to the underlying system.
impl<S> SolvedLUSystem for SolvedNegativeWeightFilteredSystem<S>
where
    S: SolvedLUSystem,
{
    fn populate_cond_values_est<I>(&mut self, values: I)
    where
        I: IntoIterator,
        I::Item: std::borrow::Borrow<f64>,
    {
        self.system.populate_cond_values_est(values);
    }

    fn populate_cond_values_sim<I>(&mut self, values: I, rng: &mut rand::prelude::StdRng)
    where
        I: IntoIterator,
        I::Item: std::borrow::Borrow<f64>,
    {
        self.system.populate_cond_values_sim(values, rng);
    }

    fn estimate(&self) -> Vec<f64> {
        self.system.estimate()
    }

    fn simulate(&self) -> Vec<f64> {
        self.system.simulate()
    }

    fn weights(&self) -> faer::prelude::MatRef<f64> {
        self.system.weights()
    }

    fn weights_mut(&mut self) -> faer::prelude::MatMut<f64> {
        self.system.weights_mut()
    }
}
