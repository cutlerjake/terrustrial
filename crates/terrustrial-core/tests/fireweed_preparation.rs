#[path = "../examples/support/shared/drillhole.rs"]
mod drillhole;
use drillhole::{composite, Interval, Survey, Trajectory};
use terrustrial_core::data::Point3;

fn survey(depth: f64, azimuth: f64, dip: f64) -> Survey {
    Survey {
        depth,
        azimuth,
        dip,
    }
}
fn close(a: [f64; 3], b: [f64; 3]) {
    for i in 0..3 {
        assert!((a[i] - b[i]).abs() < 1e-9, "{a:?} != {b:?}");
    }
}

#[test]
fn straight_arcs_wraparound_and_end_extension() {
    let origin = Point3::new([0.; 3]).unwrap();
    let straight = Trajectory::new(origin, &[survey(0., 270., -90.)], 100.).unwrap();
    close(straight.at(57.).unwrap().coordinates(), [0., 0., -57.]);
    assert!(straight.at(101.).is_err());
    assert!(straight.at(f64::NAN).is_err());
    // Quarter circle, north to east: exact analytic integral at intermediate depths.
    let arc = Trajectory::new(origin, &[survey(0., 0., 0.), survey(100., 90., 0.)], 120.).unwrap();
    let r = 200. / std::f64::consts::PI;
    for t in [0.1, 0.5, 0.9, 1.] {
        let angle = t * std::f64::consts::FRAC_PI_2;
        close(
            arc.at(t * 100.).unwrap().coordinates(),
            [r * (1. - angle.cos()), r * angle.sin(), 0.],
        );
    }
    close(arc.at(120.).unwrap().coordinates(), [r + 20., r, 0.]);
    let wrap =
        Trajectory::new(origin, &[survey(0., 359., 0.), survey(100., 1., 0.)], 100.).unwrap();
    let theta = 1_f64.to_radians();
    close(
        wrap.at(100.).unwrap().coordinates(),
        [0., 100. * theta.sin() / theta, 0.],
    );
    // Subdividing the same circular arc must not change its position.
    let subdivided = Trajectory::new(
        origin,
        &[
            survey(0., 0., 0.),
            survey(50., 45., 0.),
            survey(100., 90., 0.),
        ],
        120.,
    )
    .unwrap();
    for d in [10., 50., 75., 100., 120.] {
        close(
            arc.at(d).unwrap().coordinates(),
            subdivided.at(d).unwrap().coordinates(),
        );
    }
}

#[test]
fn invalid_surveys_do_not_invent_geometry() {
    let p = Point3::new([0.; 3]).unwrap();
    for surveys in [
        vec![],
        vec![survey(1., 0., 0.)],
        vec![survey(0., 0., 91.)],
        vec![survey(0., 0., 0.), survey(0., 0., 0.)],
        vec![survey(0., 0., 0.), survey(100., 180., 0.)],
    ] {
        assert!(Trajectory::new(p, &surveys, 100.).is_err());
    }
}

#[test]
fn interval_clipping_coverage_and_mass_conservation() {
    let samples = [
        Interval {
            from: 0.,
            to: 1.,
            values: [Some(2.), Some(4.)],
        },
        Interval {
            from: 1.,
            to: 3.,
            values: [Some(5.), None],
        },
        Interval {
            from: 4.,
            to: 5.,
            values: [Some(8.), Some(10.)],
        },
    ];
    let out = composite(&samples, [0.5, 5.], 2., 0.5).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].interval.values, [Some(4.25), None]);
    assert_eq!(out[0].coverage, [1., 0.25]);
    assert_eq!(out[1].interval.values, [Some(6.5), None]);
    assert_eq!(out[1].coverage, [0.5, 0.25]);
    assert_eq!(out[2].interval.values, [Some(8.), Some(10.)]);
    assert_eq!(out[2].interval.to, 5.);
    let mass: f64 = out
        .iter()
        .map(|c| c.interval.values[0].unwrap() * c.coverage[0] * (c.interval.to - c.interval.from))
        .sum();
    assert_eq!(mass, 19.);
    let strict = composite(&samples, [0.5, 5.], 2., 1.).unwrap();
    assert_eq!(strict[1].interval.values, [None, None]);
    let mut overlap = samples.to_vec();
    overlap[1].from = 0.5;
    assert!(composite(&overlap, [0., 5.], 1., 1.).is_err());
}

#[path = "../examples/support/fireweed/input.rs"]
#[allow(dead_code)]
mod input;

#[test]
fn package_reader_keeps_missingness_and_excludes_regressed_density() {
    let root = tempfile::tempdir().unwrap();
    let files = [
        ("MPA_Collar_20240227.csv", "HoleID,Grid,Easting,Northing,Elevation,Length_m,Dip,Azimuth\nA,NAD83 / UTM zone 9N,100,200,300,10,-90,0\n"),
        ("MPA_Survey_20240227.csv", "HoleID,Depth_m,Dip,Azimuth\nA,1,-90,0\n"),
        ("MPA_Interp_20240227.csv", "holeid,from,to,Code\nA,2,4,BZPZ\nA,4,8,BZPZ\n"),
        ("MPA_Samples_BD_20240227.csv", "\u{feff}HoleID,From_m,To_m,Zn_pct,Pb_pct,Ag_ppm,BD_tonnes_m3,Method,LowRecovery_<=85pct\nA,2,4,2,3,4,3.5,MEAS,N\nA,4,6,6,7,,4.5,REG,N\nA,6,8,8,9,10,3,MEAS,Y\n"),
    ];
    for (name, content) in files {
        std::fs::write(root.path().join(name), content).unwrap();
    }
    let data = input::load(root.path(), "BZPZ", 2.).unwrap();
    assert_eq!(data.holes, 1);
    assert_eq!(data.samples.len(), 2);
    assert_eq!(data.samples[0].hole, "A");
    assert_eq!((data.samples[0].from, data.samples[0].to), (2., 4.));
    close(data.samples[0].point.coordinates(), [100., 200., 297.]);
    assert_eq!(
        data.samples[0].values,
        [Some(2.), Some(3.), Some(4.), Some(3.5)]
    );
    assert_eq!(data.samples[1].values, [Some(6.), Some(7.), None, None]);
    assert_eq!(data.constraints.len(), 5); // adjacent interpretation rows were merged
    assert_eq!(data.constraint_holes, vec!["A"; 5]);
    assert_eq!(data.domain_checks.len(), 11);
    for c in &data.domain_checks {
        assert_eq!(c.hole, "A");
        close(c.point.coordinates(), [100., 200., 300. - c.depth]);
        let expected = (c.depth - 2.).min(8. - c.depth);
        assert_eq!(c.distance, expected);
    }
    assert!(data.audit.contains(&("collar_surveys_inserted".into(), 1)));
    let samples_only = input::load_samples(root.path(), "BZPZ", 2.).unwrap();
    assert!(samples_only.constraints.is_empty());
    assert!(samples_only.domain_checks.is_empty());
    assert_eq!(samples_only.samples.len(), data.samples.len());
    for (a, b) in samples_only.samples.iter().zip(&data.samples) {
        assert_eq!(a.hole, b.hole);
        assert_eq!((a.from, a.to), (b.from, b.to));
        assert_eq!(a.values, b.values);
        close(a.point.coordinates(), b.point.coordinates());
    }
    assert!(input::load(root.path(), "MISSING", 2.).is_err());
}

#[test]
fn topography_interpolates_faces_and_rejects_missing_coverage() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("topo.dxf");
    std::fs::write(&path,"0\nSECTION\n2\nENTITIES\n0\n3DFACE\n10\n0\n20\n0\n30\n10\n11\n10\n21\n0\n31\n20\n12\n0\n22\n10\n32\n30\n13\n0\n23\n10\n33\n30\n0\nENDSEC\n0\nEOF\n").unwrap();
    let topo = input::Topography::read(&path).unwrap();
    assert_eq!(topo.height([2., 3., 0.]), Some(18.));
    assert_eq!(topo.height([0., 0., 0.]), Some(10.));
    assert_eq!(topo.height([8., 8., 0.]), None);
    std::fs::write(&path, "0\n3DFACE\n10\n0\n0\nEOF\n").unwrap();
    assert!(input::Topography::read(&path).is_err());
}
