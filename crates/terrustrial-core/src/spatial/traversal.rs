//! Allocation-free closed-box traversal of contiguous spatial nodes.
use super::*;
pub(crate) fn visit_envelope<T: Bounded>(
    tree: &Hierarchy<T>,
    bounds: Bounds,
    visitor: impl FnMut(&T) -> Result<()>,
) -> Result<()> {
    tree.visit_bounds(bounds, visitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_traversal_matches_iterator_without_its_heap_storage() {
        let tree = Hierarchy::new(
            (0..8192)
                .map(|i| [(i % 23) as f64, ((i / 23) % 19) as f64, (i / 437) as f64])
                .collect(),
        )
        .unwrap();
        for envelope in [
            Bounds::from_corners([-1.; 3], [100.; 3]),
            Bounds::from_corners([3., 2., 1.], [19., 13., 8.]),
            Bounds::point([2., 2., 2.]),
            Bounds::point([-1.; 3]),
        ] {
            let mut expected: Vec<_> = tree.query(envelope).copied().collect();
            let mut actual = Vec::with_capacity(expected.len());
            let count = allocation_counter::measure(|| {
                visit_envelope(&tree, envelope, |point| {
                    actual.push(*point);
                    Ok(())
                })
                .unwrap()
            });
            assert_eq!(count.count_total, 0, "{count:?}");
            let compare = |a: &[f64; 3], b: &[f64; 3]| {
                a[0].total_cmp(&b[0])
                    .then(a[1].total_cmp(&b[1]))
                    .then(a[2].total_cmp(&b[2]))
            };
            expected.sort_unstable_by(compare);
            actual.sort_unstable_by(compare);
            assert_eq!(actual, expected);
        }
        let envelope = Bounds::from_corners([-1.; 3], [100.; 3]);
        let mut found = 0;
        let old = allocation_counter::measure(|| found = tree.query(envelope).count());
        assert_eq!(found, 8192);
        assert_eq!(
            old.count_total, 0,
            "range iterator also reuses stack storage"
        );
        let mut visited = 0;
        let result = visit_envelope(&tree, envelope, |_| {
            visited += 1;
            if visited == 7 {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(visited, 7);
    }
    #[test]
    fn indexed_range_and_exact_queries_preserve_membership_and_cancellation() {
        let records: Vec<_> = (0..8192)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new([(i % 257) as f64, 0., 0.]).unwrap(),
                        (i % 3) as usize,
                    ),
                    i as f64,
                    0.,
                )
                .unwrap()
            })
            .collect();
        let index = SpatialIndex::new(records.clone()).unwrap();
        let cancel = Cancellation::default();
        let mut count = 0;
        let measured = allocation_counter::measure(|| {
            index
                .visit_bounds([0.; 3], [256., 0., 0.], &cancel, &mut |_| {
                    count += 1;
                    Ok(())
                })
                .unwrap();
            let missing_variable = Site::point(Point3::new([0.; 3]).unwrap(), 99);
            assert!(index.at_site(&missing_variable).unwrap().is_empty());
        });
        assert_eq!(count, records.len());
        assert_eq!(measured.count_total, 0, "{measured:?}");
        let site = records[0].site();
        let expected: Vec<_> = records
            .iter()
            .filter(|o| o.site() == site)
            .map(Observation::id)
            .collect();
        let actual: Vec<_> = index
            .at_site(site)
            .unwrap()
            .iter()
            .map(Observation::id)
            .collect();
        assert_eq!(actual, expected);
        count = 0;
        let result = index.visit_bounds([0.; 3], [256., 0., 0.], &cancel, &mut |_| {
            count += 1;
            if count == 3 {
                cancel.cancel();
            }
            Ok(())
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(count, 3);
        let empty = SpatialIndex::new(vec![]).unwrap();
        assert!(matches!(
            empty.visit_bounds([0.; 3], [1.; 3], &cancel, &mut |_| Ok(())),
            Err(Error::Cancelled)
        ));
    }
}
