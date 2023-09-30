use parry3d::{
    bounding_volume::SimdAabb, math::SIMD_WIDTH, partitioning::SimdBestFirstVisitStatus,
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
