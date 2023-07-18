pub mod simple_kriging;

pub struct KrigingParameters {
    pub max_cond_data: usize,
    pub min_cond_data: usize,
    pub min_octant_data: usize,
    pub max_octant_data: usize,
}
