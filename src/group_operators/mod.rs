pub mod generalized_sequential_indicator_kriging;
pub mod generalized_sequential_kriging;
pub mod inverse_distance;

#[derive(Debug, Clone)]
pub struct ConditioningFilterMap {
    //clips extreme values if h > clip_h
    pub clip_h: Vec<f64>,
    pub clip_range: Vec<[f64; 2]>,

    //filter values outside of this value range
    pub valid_value_range: [f64; 2],
}

impl ConditioningFilterMap {
    pub fn filter_map(&self, value: f64, h: f64) -> Option<f64> {
        if value < self.valid_value_range[0] || value > self.valid_value_range[1] {
            return None;
        }
        for (i, clip_h) in self.clip_h.iter().enumerate().rev() {
            if h > *clip_h {
                let clip_range = self.clip_range[i];
                if value < clip_range[0] {
                    return Some(clip_range[0]);
                } else if value > clip_range[1] {
                    return Some(clip_range[1]);
                }
            }
        }

        Some(value)
    }
}

#[derive(Debug, Clone)]
pub struct ConditioningParams {
    //number of conditioning points
    pub max_n_cond: usize,
    pub min_n_cond: usize,

    //limit on the number of conditioning points per octant
    pub max_octant: usize,
    pub min_conditioned_octants: usize,

    //clips extreme values if h > clip_h
    pub clip_h: Vec<f64>,
    pub clip_range: Vec<[f64; 2]>,

    //filter values outside of this value range
    pub valid_value_range: [f64; 2],

    //limit number of points from same source group
    pub same_source_group_limit: usize,

    //dynamic orientation for seach ellipsoid
    pub orient_search: bool,
    //dynamic orientation for variogram
    pub orient_variogram: bool,
}

impl ConditioningParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_n_cond: usize,
        min_n_cond: usize,
        max_octant: usize,
        min_conditioned_octants: usize,
        clip_h: Vec<f64>,
        clip_range: Vec<[f64; 2]>,
        valid_value_range: [f64; 2],
        same_source_group_limit: usize,
        orient_search: bool,
        orient_variogram: bool,
    ) -> Self {
        Self {
            max_n_cond,
            min_n_cond,
            max_octant,
            min_conditioned_octants,

            clip_h,
            clip_range,

            valid_value_range,

            same_source_group_limit,

            orient_search,
            orient_variogram,
        }
    }
}

impl Default for ConditioningParams {
    fn default() -> Self {
        Self {
            max_n_cond: 32,
            min_n_cond: 4,
            max_octant: 8,
            min_conditioned_octants: 1,
            clip_h: vec![],
            clip_range: vec![],
            valid_value_range: [f64::NEG_INFINITY, f64::INFINITY],
            same_source_group_limit: usize::MAX,
            orient_search: false,
            orient_variogram: false,
        }
    }
}
