use terrustrial_core::domains::{
    evidence::{ContactPolicy, Hole, Record},
    validation::*,
};
#[test]
fn coverage_limits_do_not_silently_become_contacts() {
    let record = |from, to, code: &str, row| Record {
        from,
        to,
        code: code.into(),
        source_rows: vec![row],
    };
    let h = Hole::new(
        "A".into(),
        100.,
        "ore",
        vec![record(10., 20., "ore", 1), record(20., 30., "host", 2)],
    )
    .unwrap();
    let confirmed: Vec<_> = h
        .fitting_contacts(ContactPolicy::KnownTransitions)
        .collect();
    assert_eq!(confirmed.len(), 1);
    assert_eq!(confirmed[0].depth, 20.);
    assert_eq!(confirmed[0].source_rows, vec![1, 2]);
    assert_eq!(
        h.fitting_contacts(ContactPolicy::InterpretedLimits).count(),
        2
    );
}
#[test]
fn spatial_blocks_are_contiguous_and_keep_colocated_holes_together() {
    let p = [
        [5., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [7., 0., 0.],
        [3., 0., 0.],
        [9., 0., 0.],
    ];
    let b = spatial_blocks(&p, 3).unwrap();
    assert_eq!(b[1], b[2]);
    for i in 0..p.len() {
        for j in 0..p.len() {
            if p[i][0] < p[j][0] {
                assert!(b[i] <= b[j]);
            }
        }
    }
    assert!(spatial_blocks(&[[0.; 3]; 4], 2).is_err());
}
#[test]
fn score_is_length_weighted_and_missing_classes_are_unresolved() {
    let mut s = IntervalScore::default();
    s.add(true, -1., 99.).unwrap();
    s.add(true, 1., 1.).unwrap();
    assert!(s.balanced_error().is_none());
    s.add(false, 1., 10.).unwrap();
    assert_eq!(s.balanced_error(), Some(0.005));
    assert!(s.add(false, f64::NAN, 1.).is_err());
}
#[test]
fn missing_failed_and_nonfinite_checks_cannot_pass_acceptance() {
    let limits = AcceptanceLimits {
        maximum_contact_distance_m: 1.,
        maximum_held_out_balanced_error: 0.1,
    };
    let mut e = AcceptanceEvidence::default();
    assert!(e.accept(limits).is_err());
    e = AcceptanceEvidence {
        numerical_complete: true,
        geometry_validated: true,
        boundaries_validated: true,
        intervals_validated: true,
        independent_spatial_validation: true,
        maximum_contact_distance_m: Some(0.5),
        held_out_balanced_error: Some(0.05),
        unresolved_checks: 0,
    };
    assert!(e.accept(limits).is_ok());
    e.unresolved_checks = 1;
    assert!(e.accept(limits).is_err());
    e.unresolved_checks = 0;
    e.maximum_contact_distance_m = Some(f64::NAN);
    assert!(e.accept(limits).is_err());
}

#[test]
fn adaptive_trace_audit_finds_thin_violations_and_never_accepts_unresolved_cells() {
    use terrustrial_core::{domains::validation::trace, execution::Cancellation};
    let cancel = Cancellation::default();
    let options = trace::Options {
        depth_tolerance_m: 0.001,
        maximum_cells: 10000,
    };
    // Negative except for a 2 cm lens between sample locations.
    let bounds = |a: f64, b: f64| {
        let near = if a <= 0.333 && b >= 0.333 {
            0.
        } else {
            (a - 0.333).abs().min((b - 0.333).abs())
        };
        let far = (a - 0.333).abs().max((b - 0.333).abs());
        Ok([0.01 - far, 0.01 - near])
    };
    let report = trace::audit_sign(0., 1., true, options, bounds, &cancel).unwrap();
    assert!(report.witnessed_violating_length_m > 0.018);
    assert!(
        (report.witnessed_violating_length_m
            + report.enclosed_consistent_length_m
            + report.unresolved_length_m
            - 1.)
            .abs()
            < 1e-12
    );
    let unresolved = trace::audit_sign(
        0.,
        1.,
        true,
        trace::Options {
            maximum_cells: 1,
            ..options
        },
        bounds,
        &cancel,
    )
    .unwrap();
    assert_eq!(unresolved.unresolved_length_m, 1.);
    let roots = trace::roots(
        0.,
        1.,
        options,
        bounds,
        |d| Ok(0.01 - (d - 0.333).abs()),
        &cancel,
    )
    .unwrap();
    assert_eq!(roots.brackets.len(), 2);
    assert!(roots.brackets.iter().all(|r| r[1] - r[0] <= 0.001));
    cancel.cancel();
    assert!(trace::audit_sign(0., 1., true, options, bounds, &cancel).is_err());
}
#[test]
fn unobserved_tangent_root_is_unresolved_not_absent() {
    use terrustrial_core::{domains::validation::trace, execution::Cancellation};
    let r = trace::roots(
        0.,
        1.,
        trace::Options {
            depth_tolerance_m: 0.01,
            maximum_cells: 10000,
        },
        |a, b| {
            let near = if a <= 0.333 && b >= 0.333 {
                0.
            } else {
                (a - 0.333).abs().min((b - 0.333).abs())
            };
            Ok([near * near, (a - 0.333).powi(2).max((b - 0.333).powi(2))])
        },
        |d| Ok((d - 0.333).powi(2)),
        &Cancellation::default(),
    )
    .unwrap();
    assert!(r.brackets.is_empty());
    assert!(!r.unresolved.is_empty());
}

#[test]
fn identically_zero_interval_has_no_interior_or_exterior_evidence() {
    use terrustrial_core::{domains::validation::trace, execution::Cancellation};
    let r = trace::audit_sign(
        0.,
        1.,
        true,
        trace::Options {
            depth_tolerance_m: 0.1,
            maximum_cells: 100,
        },
        |_, _| Ok([0., 0.]),
        &Cancellation::default(),
    )
    .unwrap();
    assert_eq!(r.enclosed_consistent_length_m, 0.);
    assert_eq!(r.unresolved_length_m, 1.);
}
