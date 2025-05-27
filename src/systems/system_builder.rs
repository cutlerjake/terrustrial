use crate::geometry::support::Support;
use crate::variography::model_variograms::composite::CompositeVariogram;
use crate::variography::model_variograms::VariogramModel;

use faer::{Col, Mat};

use ultraviolet::{DVec3, DVec3x4};
use wide::f64x4;

pub struct SKGeneralBuilder;

impl SKGeneralBuilder {
    const NUM_LANES: usize = 4;
    pub fn build_cov_mat(
        cov_mat: &mut Mat<f64>,
        cond: &[Support],
        vgram: &CompositeVariogram,
        h_buffer: &mut Vec<DVec3>,
        pt_buffer: &mut Vec<DVec3>,
        var_buffer: &mut Vec<f64>,
        ind_buffer: &mut Vec<usize>,
    ) {
        // 1. Clear input buffers.
        h_buffer.clear();
        pt_buffer.clear();
        var_buffer.clear();
        ind_buffer.clear();

        ind_buffer.push(0);

        //1. Compute pair-wise distances for lower triangular pairs.
        for (i, p1) in cond.iter().enumerate() {
            for p2 in cond.iter().take(i + 1) {
                p1.dists_to_other(p2, h_buffer, pt_buffer);
                ind_buffer.push(h_buffer.len());
            }
        }

        // 2. Compute the var
        for chunk in h_buffer.chunks_exact(Self::NUM_LANES) {
            let &[h1, h2, h3, h4] = chunk else {
                unreachable!()
            };

            let vec_h = DVec3x4::new(
                f64x4::new([h1.x, h2.x, h3.x, h4.x]),
                f64x4::new([h1.y, h2.y, h3.y, h4.y]),
                f64x4::new([h1.z, h2.z, h3.z, h4.z]),
            );

            let var = vgram.covariogram_simd(vec_h);

            var_buffer.extend_from_slice(var.as_array_ref());
        }

        for h in h_buffer.chunks_exact(Self::NUM_LANES).remainder() {
            var_buffer.push(vgram.covariogram(*h));
        }

        // 3. Populate lower traingular elements
        let mut cnt = 0;
        for i in 0..cond.len() {
            for j in 0..i + 1 {
                let low_idx = ind_buffer[cnt];
                let high_idx = ind_buffer[cnt + 1];
                let vars = &var_buffer[low_idx..high_idx];

                let avg_var = vars.iter().copied().sum::<f64>() / vars.len() as f64;

                *cov_mat.get_mut(i, j) = avg_var;

                cnt += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_cov_vec(
        cov_vec: &mut Col<f64>,
        cond: &[Support],
        kriging_point: &Support,
        vgram: &CompositeVariogram,
        h_buffer: &mut Vec<DVec3>,
        pt_buffer: &mut Vec<DVec3>,
        var_buffer: &mut Vec<f64>,
        ind_buffer: &mut Vec<usize>,
    ) {
        // 1. Clear input buffers.
        h_buffer.clear();
        pt_buffer.clear();
        var_buffer.clear();
        ind_buffer.clear();

        ind_buffer.push(0);

        //1. Compute pair-wise distances for lower triangular pairs.
        for p1 in cond.iter() {
            kriging_point.dists_to_other(p1, h_buffer, pt_buffer);
            ind_buffer.push(h_buffer.len());
        }

        // 2. Compute the var
        for chunk in h_buffer.chunks_exact(Self::NUM_LANES) {
            let &[h1, h2, h3, h4] = chunk else {
                unreachable!()
            };

            let vec_h = DVec3x4::new(
                f64x4::new([h1.x, h2.x, h3.x, h4.x]),
                f64x4::new([h1.y, h2.y, h3.y, h4.y]),
                f64x4::new([h1.z, h2.z, h3.z, h4.z]),
            );

            let var = vgram.covariogram_simd(vec_h);

            var_buffer.extend_from_slice(var.as_array_ref());
        }

        for h in h_buffer.chunks_exact(Self::NUM_LANES).remainder() {
            var_buffer.push(vgram.covariogram(*h));
        }

        // 3. Populate lower traingular elements
        for i in 0..cond.len() {
            let low_idx = ind_buffer[i];
            let high_idx = ind_buffer[i + 1];
            let vars = &var_buffer[low_idx..high_idx];

            let avg_var = vars.iter().copied().sum::<f64>() / vars.len() as f64;

            *cov_vec.get_mut(i) = avg_var;
        }
    }
}
