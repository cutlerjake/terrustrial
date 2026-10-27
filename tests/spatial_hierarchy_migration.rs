use terrustrial::{
    geometry::{aabb::Aabb, support::Support},
    spatial_database::{group_provider::GroupProvider, IterNearest, SpatialAcceleratedDB},
};
use ultraviolet::DVec3;

#[test]
fn legacy_nearest_preserves_full_support_distance_and_payloads() {
    let supports: Vec<_> = (0..100)
        .map(|i| {
            let center = DVec3::new((i % 7) as f64, (i / 7) as f64, 0.);
            if i % 3 == 0 {
                Support::Aabb {
                    aabb: Aabb::new(center, DVec3::new(2., 0.5, 0.5)),
                    disc: DVec3::one(),
                }
            } else {
                Support::Point(center)
            }
        })
        .collect();
    let db = SpatialAcceleratedDB::new(supports.clone(), (0..100).collect::<Vec<u64>>());
    for p in [
        DVec3::zero(),
        DVec3::new(4., 3.5, 2.),
        DVec3::new(-5., -3., 0.),
    ] {
        let mut expected: Vec<_> = supports
            .iter()
            .map(|support| {
                let center = support.center();
                let extent = match support {
                    Support::Point(_) => DVec3::zero(),
                    Support::Aabb { aabb, .. } => aabb.half_extents,
                };
                let d = [
                    ((p.x - center.x).abs() - extent.x).max(0.),
                    ((p.y - center.y).abs() - extent.y).max(0.),
                    ((p.z - center.z).abs() - extent.z).max(0.),
                ];
                d.iter().map(|x| x * x).sum::<f64>()
            })
            .collect();
        expected.sort_unstable_by(f64::total_cmp);
        let found: Vec<_> = db.iter_nearest(&p).collect();
        assert_eq!(
            found.iter().map(|o| o.sq_dist).collect::<Vec<_>>(),
            expected
        );
        let mut ids: Vec<_> = found
            .iter()
            .map(|o| {
                assert_eq!(o.data, o.idx as u64);
                assert_eq!(o.shape.center(), supports[o.idx as usize].center());
                o.idx
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..100).collect::<Vec<_>>());
    }
}

#[test]
fn immutable_grouping_claims_boundaries_duplicates_and_blocks_once() {
    let mut supports: Vec<_> = (0..100)
        .map(|i| Support::Point(DVec3::new((i % 5) as f64, (i / 5 % 5) as f64, 0.)))
        .collect();
    supports.push(Support::Aabb {
        aabb: Aabb::new(DVec3::new(2., 2., 0.), DVec3::one()),
        disc: DVec3::one(),
    });
    let groups = GroupProvider::optimized_groups(&supports, 1., 1., 1., 2, 2, 2);
    assert_eq!(groups.n_nodes(), supports.len());
    let mut ids = Vec::new();
    for group in 0..groups.n_groups() {
        assert!(groups.get_group(group).len() <= 8);
        for (support, &id) in groups
            .get_group(group)
            .iter()
            .zip(groups.get_original_idxs(group))
        {
            assert_eq!(support.center(), supports[id].center());
            ids.push(id);
        }
    }
    ids.sort_unstable();
    assert_eq!(ids, (0..supports.len()).collect::<Vec<_>>());
    let empty = GroupProvider::optimized_groups(&[], 1., 1., 1., 2, 2, 2);
    assert_eq!(empty.n_nodes(), 0);
    assert_eq!(empty.n_groups(), 0);
}
