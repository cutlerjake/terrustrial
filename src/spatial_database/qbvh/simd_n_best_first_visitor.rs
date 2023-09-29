use nalgebra::{distance, Point3, SimdBool as _, SimdPartialOrd, SimdValue};
use parry3d::{
    bounding_volume::SimdAabb,
    math::{SimdBool, SimdReal, SIMD_WIDTH},
    partitioning::SimdBestFirstVisitStatus,
};

use crate::spatial_database::qbvh::conditioning_data_collector::{
    ConditioningDataCollector, InsertionResult,
};

pub trait SimdNBestFirstVisitor<LeafData, SimdBV> {
    type Result;
    fn visit(
        &mut self,
        threshold: &mut f32,
        bv: &SimdAabb,
        data: Option<[Option<&u32>; SIMD_WIDTH]>,
    ) -> SimdBestFirstVisitStatus<Self::Result>;
}

impl<'a, LeafData> SimdNBestFirstVisitor<LeafData, SimdAabb> for ConditioningDataCollector<'a> {
    type Result = ();

    fn visit(
        &mut self,
        threshold: &mut f32,
        bv: &SimdAabb,
        data: Option<[Option<&u32>; SIMD_WIDTH]>,
    ) -> SimdBestFirstVisitStatus<Self::Result> {
        //mask to select only element with dist less than current furthest point
        let dists = bv.distance_to_local_point(&Point3::splat(self.point));
        let mask = dists.simd_lt(SimdReal::splat(*threshold));

        if let Some(data) = data {
            let bitmask = mask.bitmask();
            let mut weights = [0.0; SIMD_WIDTH];
            let mut mask = [false; SIMD_WIDTH];
            //let mut results = [None; SIMD_WIDTH];

            for ii in 0..SIMD_WIDTH {
                if (bitmask & (1 << ii)) != 0 && data[ii].is_some() {
                    // get point index
                    let part_id = *data[ii].unwrap();
                    // get point
                    let point = self.point_set.points[part_id as usize];
                    // get distance from query point to point
                    let dist = distance(&point, &self.point);

                    //insert point if distance is less than current furthest point
                    match self.insert_octant_point(point, dist) {
                        InsertionResult::InsertedNotFull => {
                            //If inserted dist <= threshold
                            mask[ii] = true;
                        }
                        InsertionResult::InsertedFull => {
                            //Conditioning set full -> must updarte threshold
                            *threshold = self.max_accepted_dist;
                            //If inserted dist <= threshold
                            mask[ii] = true;
                        }
                        InsertionResult::NotInserted => {}
                        InsertionResult::NotInsertedOutOfRange => {
                            return SimdBestFirstVisitStatus::ExitEarly(None);
                        }
                    }

                    weights[ii] = dist;
                }
            }

            SimdBestFirstVisitStatus::MaybeContinue {
                weights: SimdReal::from(weights),
                mask: SimdBool::from(mask),
                results: [None; SIMD_WIDTH],
            }
        } else {
            SimdBestFirstVisitStatus::MaybeContinue {
                weights: dists,
                mask: mask,
                results: [None; SIMD_WIDTH],
            }
        }
    }
}
