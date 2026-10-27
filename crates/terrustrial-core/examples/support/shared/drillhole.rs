//! Shared public drillhole example geometry and length-weighted compositing.
//! Coordinates are east, north, elevation; azimuth is clockwise from grid north,
//! dip is positive upward. Circular arcs integrate survey tangents, including at
//! intermediate assay depths, rather than interpolating station coordinates.
//! Minimum curvature reference: USGS WRI 83-4275, hole-survey methods.
use terrustrial_core::{data::Point3, Error, Result};

#[derive(Clone, Copy)]
pub struct Survey {
    pub depth: f64,
    pub azimuth: f64,
    pub dip: f64,
}

struct Station {
    depth: f64,
    point: Point3,
    direction: [f64; 3],
}

pub struct Trajectory {
    stations: Vec<Station>,
    end: f64,
}

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

impl Trajectory {
    /// Require an explicit collar orientation at zero and strictly ordered survey
    /// depths. Beyond the last survey, hold the last tangent to the recorded hole
    /// end. No query beyond the hole end or implicit unsurveyed collar is allowed.
    pub fn new(collar: Point3, surveys: &[Survey], end: f64) -> Result<Self> {
        if surveys.is_empty()
            || surveys[0].depth != 0.0
            || !end.is_finite()
            || end <= 0.0
            || surveys.windows(2).any(|s| s[0].depth >= s[1].depth)
        {
            return Err(invalid(
                "hole end and ordered survey starting at zero required",
            ));
        }
        let mut stations: Vec<Station> = Vec::with_capacity(surveys.len());
        for s in surveys {
            if !s.depth.is_finite()
                || !(0.0..=end).contains(&s.depth)
                || !s.azimuth.is_finite()
                || !(0.0..=360.0).contains(&s.azimuth)
                || !s.dip.is_finite()
                || !(-90.0..=90.0).contains(&s.dip)
            {
                return Err(invalid(
                    "invalid survey depth, azimuth or upward-positive dip",
                ));
            }
            let (sa, ca) = s.azimuth.to_radians().sin_cos();
            let (sd, cd) = s.dip.to_radians().sin_cos();
            let direction = [cd * sa, cd * ca, sd];
            let point = if let Some(previous) = stations.last() {
                advance(previous, direction, s.depth - previous.depth, 1.0)?
            } else {
                collar
            };
            stations.push(Station {
                depth: s.depth,
                point,
                direction,
            });
        }
        Ok(Self { stations, end })
    }

    pub fn at(&self, depth: f64) -> Result<Point3> {
        if !depth.is_finite() || !(0.0..=self.end).contains(&depth) {
            return Err(invalid("query outside recorded hole length"));
        }
        let i = self.stations.partition_point(|s| s.depth <= depth) - 1;
        let start = &self.stations[i];
        if depth == start.depth {
            return Ok(start.point);
        }
        if let Some(next) = self.stations.get(i + 1) {
            let length = next.depth - start.depth;
            advance(
                start,
                next.direction,
                length,
                (depth - start.depth) / length,
            )
        } else {
            advance(start, start.direction, depth - start.depth, 1.0)
        }
    }
}

fn advance(start: &Station, end: [f64; 3], length: f64, fraction: f64) -> Result<Point3> {
    let u = start.direction;
    let dot = u
        .iter()
        .zip(end)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    let perpendicular: [f64; 3] = std::array::from_fn(|i| end[i] - dot * u[i]);
    let sine = perpendicular[0]
        .hypot(perpendicular[1])
        .hypot(perpendicular[2]);
    if dot < -0.999999 {
        return Err(invalid(
            "near-antipodal survey tangents do not define a reliable arc",
        ));
    }
    let theta = sine.atan2(dot);
    let angle = fraction * theta;
    let along = if theta < 1e-8 {
        length * fraction * (1.0 - angle * angle / 6.0)
    } else {
        length * angle.sin() / theta
    };
    // 2 sin²(angle/2) avoids cancellation in 1-cos(angle).
    let across = if sine == 0.0 {
        0.0
    } else {
        length * 2.0 * (angle * 0.5).sin().powi(2) / theta / sine
    };
    let origin = start.point.coordinates();
    Point3::new(std::array::from_fn(|i| {
        origin[i] + along * u[i] + across * perpendicular[i]
    }))
}

#[derive(Clone, Debug)]
pub struct Interval<const N: usize> {
    pub from: f64,
    pub to: f64,
    pub values: [Option<f64>; N],
}

pub struct Composite<const N: usize> {
    pub interval: Interval<N>,
    pub coverage: [f64; N],
}

/// Length-weighted compositing within one explicit domain interval. Missing
/// attributes retain their own coverage and are never replaced with zero.
/// Partial final bins are retained with their actual support. Domain contacts
/// clip samples; this assumes each input value is constant over its interval.
pub fn composite<const N: usize>(
    samples: &[Interval<N>],
    domain: [f64; 2],
    length: f64,
    minimum_coverage: f64,
) -> Result<Vec<Composite<N>>> {
    if domain.iter().any(|v| !v.is_finite())
        || domain[0] < 0.0
        || domain[1] <= domain[0]
        || !length.is_finite()
        || length <= 0.0
        || !minimum_coverage.is_finite()
        || !(0.0..=1.0).contains(&minimum_coverage)
        || samples.iter().any(|s| {
            !s.from.is_finite()
                || !s.to.is_finite()
                || s.from < 0.0
                || s.to <= s.from
                || s.values.iter().flatten().any(|v| !v.is_finite())
        })
        || samples.windows(2).any(|s| s[0].to > s[1].from)
    {
        return Err(invalid(
            "invalid compositing configuration or overlapping/unordered samples",
        ));
    }
    let mut output = Vec::new();
    let mut from = domain[0];
    let mut cursor = samples.partition_point(|s| s.to <= from);
    while from < domain[1] {
        let to = (from + length).min(domain[1]);
        if to <= from {
            return Err(invalid("composite length below depth resolution"));
        }
        let mut covered = [0.0; N];
        let mut means = [0.0; N];
        while cursor < samples.len() && samples[cursor].to <= from {
            cursor += 1;
        }
        for sample in samples[cursor..].iter().take_while(|s| s.from < to) {
            let overlap = sample.to.min(to) - sample.from.max(from);
            if overlap <= 0.0 {
                continue;
            }
            for v in 0..N {
                if let Some(value) = sample.values[v] {
                    covered[v] += overlap;
                    means[v] += (value - means[v]) * (overlap / covered[v]);
                }
            }
        }
        let coverage = covered.map(|c| (c / (to - from)).clamp(0.0, 1.0));
        let values = std::array::from_fn(|v| {
            (covered[v] > 0.0 && coverage[v] + 16.0 * f64::EPSILON >= minimum_coverage)
                .then_some(means[v])
        });
        if values.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("composite arithmetic overflow"));
        }
        output.push(Composite {
            interval: Interval { from, to, values },
            coverage,
        });
        from = to;
    }
    Ok(output)
}

#[derive(Clone)]
#[allow(dead_code)] // Some geometry-only examples consume only the point field.
pub struct Sample<const N: usize = 4> {
    pub hole: String,
    pub from: f64,
    pub to: f64,
    pub point: Point3,
    pub values: [Option<f64>; N],
}
