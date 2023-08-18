use mathru::algebra::abstr::Polynomial;

use crate::spatial_database::gridded_databases::{
    complete_grid::CompleteGriddedDataBase, incomplete_grid::InCompleteGriddedDataBase,
};

pub struct HOSIM {
    pub training_image: CompleteGriddedDataBase<f32>,
    pub conditioning_data: InCompleteGriddedDataBase<f32>,
    pub legendre_polynomials: Vec<Polynomial<f32>>,
}

impl HOSIM {
    fn simulate(&mut self, simulation_grid: &mut InCompleteGriddedDataBase<f32>) {}
}
