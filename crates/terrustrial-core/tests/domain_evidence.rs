use terrustrial_core::domains::evidence::{Hole, Membership, Record};
fn row(a: f64, b: f64, code: &str, line: u64) -> Record {
    Record {
        from: a,
        to: b,
        code: code.into(),
        source_rows: vec![line],
    }
}
#[test]
fn gaps_remain_unknown_and_short_intervals_survive() {
    let h = Hole::new(
        "A".into(),
        100.,
        "target",
        vec![
            row(10., 10.1, "target", 1),
            row(10.1, 20., "other", 2),
            row(40., 60., "target", 3),
        ],
    )
    .unwrap();
    assert_eq!(
        h.intervals
            .iter()
            .filter(|r| r.membership == Membership::Unknown)
            .count(),
        3
    );
    assert_eq!(h.contacts.len(), 4);
    assert!(!h.contacts[0].known_exterior);
    assert!(h.contacts[1].known_exterior);
    assert!(h
        .sample(5., 100)
        .unwrap()
        .iter()
        .any(|(d, m)| *d > 10. && *d < 10.1 && *m == Membership::Interior));
    assert!(h.sample(0.001, 10).is_err());
}
#[test]
fn provenance_merging_censoring_and_conflicts() {
    let h = Hole::new(
        "A".into(),
        100.,
        "target",
        vec![row(0., 20., "target", 1), row(20., 100., "target", 2)],
    )
    .unwrap();
    assert!(h.contacts.is_empty());
    assert_eq!(h.intervals[0].source_rows, vec![1, 2]);
    assert!(Hole::new(
        "A".into(),
        100.,
        "target",
        vec![row(0., 30., "target", 1), row(20., 50., "other", 2)]
    )
    .is_err());
    let h = Hole::new("A".into(), 100., "target", vec![]).unwrap();
    assert_eq!(h.intervals[0].membership, Membership::Unknown);
}
