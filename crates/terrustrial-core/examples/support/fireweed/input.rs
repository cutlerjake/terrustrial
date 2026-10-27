//! Strict readers for the February 2024 public package. No source data are embedded.
use super::drillhole::{composite, Interval, Survey, Trajectory};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::Path,
};
use terrustrial_core::data::Point3;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const ATTRIBUTES: [&str; 4] = ["Zn_pct", "Pb_pct", "Ag_ppm", "BD_tonnes_m3"];
pub type Sample = super::drillhole::Sample<4>;
/// Signed downhole-distance proxy for the interpreted domain at a point on a
/// hole: positive inside an interpreted interval, zero at its ends and negative
/// outside. These are not Euclidean signed distances.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub point: Point3,
    pub value: f64,
}
#[derive(Clone)]
pub struct DomainCheck {
    pub hole: String,
    pub depth: f64,
    pub point: Point3,
    pub distance: f64,
}
pub struct Data {
    pub samples: Vec<Sample>,
    pub constraints: Vec<Constraint>,
    pub domain_checks: Vec<DomainCheck>,
    pub constraint_holes: Vec<String>,
    pub holes: usize,
    pub audit: Vec<(String, usize)>,
}
pub(super) struct Table {
    columns: BTreeMap<String, usize>,
    pub(super) rows: Vec<csv::StringRecord>,
}
impl Table {
    pub(super) fn read(root: &Path, name: &str) -> Result<Self> {
        let mut r = csv::Reader::from_path(root.join(name))?;
        let columns = r
            .headers()?
            .iter()
            .enumerate()
            .map(|(i, v)| (v.trim_start_matches('\u{feff}').to_owned(), i))
            .collect();
        Ok(Self {
            columns,
            rows: r.records().collect::<std::result::Result<_, _>>()?,
        })
    }
    pub(super) fn text<'a>(&self, r: &'a csv::StringRecord, key: &str) -> Result<&'a str> {
        Ok(r.get(
            *self
                .columns
                .get(key)
                .ok_or_else(|| format!("missing column {key}"))?,
        )
        .ok_or("short CSV row")?
        .trim())
    }
    pub(super) fn number(&self, r: &csv::StringRecord, key: &str) -> Result<f64> {
        let v: f64 = self.text(r, key)?.parse()?;
        if !v.is_finite() {
            return Err(format!("nonfinite {key}").into());
        }
        Ok(v)
    }
}
pub fn load(root: &Path, code: &str, length: f64) -> Result<Data> {
    load_impl(root, code, length, true)
}
pub fn load_samples(root: &Path, code: &str, length: f64) -> Result<Data> {
    load_impl(root, code, length, false)
}
fn load_impl(root: &Path, code: &str, length: f64, legacy_geometry: bool) -> Result<Data> {
    let interp = Table::read(root, "MPA_Interp_20240227.csv")?;
    let mut domains: BTreeMap<String, Vec<[f64; 2]>> = BTreeMap::new();
    for r in &interp.rows {
        if interp.text(r, "Code")? == code {
            domains
                .entry(interp.text(r, "holeid")?.into())
                .or_default()
                .push([interp.number(r, "from")?, interp.number(r, "to")?]);
        }
    }
    if domains.is_empty() {
        return Err(format!("no interpretation for {code}").into());
    }
    for intervals in domains.values_mut() {
        intervals.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut merged: Vec<[f64; 2]> = Vec::new();
        for &p in intervals.iter() {
            if p[0] < 0. || p[1] <= p[0] {
                return Err("invalid interpreted interval".into());
            }
            if let Some(last) = merged.last_mut() {
                if p[0] < last[1] - 1e-6 {
                    return Err("overlapping interpreted intervals".into());
                }
                if (p[0] - last[1]).abs() <= 1e-6 {
                    last[1] = p[1];
                    continue;
                }
            }
            merged.push(p);
        }
        *intervals = merged;
    }
    let collar = Table::read(root, "MPA_Collar_20240227.csv")?;
    let surveys = Table::read(root, "MPA_Survey_20240227.csv")?;
    let assays = Table::read(root, "MPA_Samples_BD_20240227.csv")?;
    let mut paths = BTreeMap::new();
    let mut ends = BTreeMap::new();
    let mut inserted = 0;
    for r in &collar.rows {
        let id = collar.text(r, "HoleID")?;
        if !domains.contains_key(id) {
            continue;
        }
        if collar.text(r, "Grid")? != "NAD83 / UTM zone 9N" {
            return Err("unexpected coordinate reference system".into());
        }
        let end = collar.number(r, "Length_m")?;
        let point = Point3::new([
            collar.number(r, "Easting")?,
            collar.number(r, "Northing")?,
            collar.number(r, "Elevation")?,
        ])?;
        let mut s = Vec::new();
        for r in &surveys.rows {
            if surveys.text(r, "HoleID")? == id {
                s.push(Survey {
                    depth: surveys.number(r, "Depth_m")?,
                    azimuth: surveys.number(r, "Azimuth")?,
                    dip: surveys.number(r, "Dip")?,
                });
            }
        }
        s.sort_by(|a, b| a.depth.total_cmp(&b.depth));
        if s.first().is_none_or(|s| s.depth > 0.) {
            s.insert(
                0,
                Survey {
                    depth: 0.,
                    azimuth: collar.number(r, "Azimuth")?,
                    dip: collar.number(r, "Dip")?,
                },
            );
            inserted += 1;
        }
        if paths
            .insert(id.to_owned(), Trajectory::new(point, &s, end)?)
            .is_some()
        {
            return Err("duplicate collar".into());
        }
        ends.insert(id.to_owned(), end);
    }
    let mut byhole: BTreeMap<String, Vec<Interval<4>>> = BTreeMap::new();
    let mut low = 0;
    let mut density = 0;
    for r in &assays.rows {
        let id = assays.text(r, "HoleID")?;
        if !domains.contains_key(id) {
            continue;
        }
        let mut values = [None; 4];
        let low_recovery = match assays.text(r, "LowRecovery_<=85pct")? {
            "Y" => true,
            "N" => false,
            _ => return Err("unexpected recovery flag".into()),
        };
        if low_recovery {
            low += 1;
        }
        for v in 0..4 {
            let value = assays.text(r, ATTRIBUTES[v])?;
            if value.is_empty() || low_recovery {
                continue;
            }
            if v == 3 && assays.text(r, "Method")? != "MEAS" {
                density += 1;
                continue;
            }
            let x: f64 = value.parse()?;
            if !x.is_finite() || x < 0. || (v == 3 && x == 0.) {
                return Err(format!("invalid physical {} value", ATTRIBUTES[v]).into());
            }
            values[v] = Some(x);
        }
        byhole.entry(id.into()).or_default().push(Interval {
            from: assays.number(r, "From_m")?,
            to: assays.number(r, "To_m")?,
            values,
        });
    }
    let mut samples = Vec::new();
    let mut constraints = Vec::new();
    let mut domain_checks = Vec::new();
    let mut constraint_holes = Vec::new();
    let mut seen = BTreeSet::new();
    for (id, intervals) in &domains {
        let path = paths
            .get(id)
            .ok_or_else(|| format!("missing collar {id}"))?;
        if legacy_geometry {
            // Score the field throughout each logged interval, not only at fit controls.
            // Only a 40 m buffer around the selected interpretation is assumed exterior.
            let mut depths = BTreeSet::new();
            for &[from, to] in intervals {
                let lo = (from - 40.).max(0.);
                let hi = (to + 40.).min(ends[id]);
                let n = ((hi - lo) / 1.).ceil() as usize;
                for i in 0..=n {
                    depths.insert((lo + i as f64).min(hi).to_bits());
                }
                depths.insert(from.to_bits());
                depths.insert(to.to_bits());
            }
            // Nonnegative finite depths have the same ordering as their bit patterns.
            // Retain exact contacts and hole ends rather than rounding past coverage.
            for bits in depths {
                let depth = f64::from_bits(bits);
                let distance = intervals
                    .iter()
                    .map(|r| (depth - r[0]).abs().min((depth - r[1]).abs()))
                    .fold(f64::INFINITY, f64::min);
                let inside = intervals.iter().any(|r| depth >= r[0] && depth <= r[1]);
                domain_checks.push(DomainCheck {
                    hole: id.clone(),
                    depth,
                    point: path.at(depth)?,
                    distance: if inside { distance } else { -distance },
                });
            }
        }
        let s = byhole.entry(id.clone()).or_default();
        s.sort_by(|a, b| a.from.total_cmp(&b.from));
        for &[from, to] in intervals {
            if to > ends[id] {
                return Err("interpretation past hole end".into());
            }
            for c in composite(s, [from, to], length, 1.)? {
                if c.interval.values.iter().all(Option::is_none) {
                    continue;
                }
                if c.coverage.iter().any(|v| !v.is_finite()) {
                    return Err("invalid coverage".into());
                }
                samples.push(Sample {
                    hole: id.clone(),
                    from: c.interval.from,
                    to: c.interval.to,
                    point: path.at((c.interval.from + c.interval.to) * 0.5)?,
                    values: c.interval.values,
                });
            }
            if !legacy_geometry {
                continue;
            }
            // These are signed downhole-distance proxies, not Euclidean signed distances.
            // Contacts at hole endpoints are censored, so do not infer a boundary there.
            let offset = ((to - from) / 4.).min(5.);
            let mut points = vec![((from + to) * 0.5, offset)];
            if from > 0. {
                points.push((from, 0.));
                let d = (from - offset).max(0.);
                if !intervals.iter().any(|r| d >= r[0] && d <= r[1]) {
                    points.push((d, -(from - d)));
                }
            }
            if to < ends[id] {
                points.push((to, 0.));
                let d = (to + offset).min(ends[id]);
                if !intervals.iter().any(|r| d >= r[0] && d <= r[1]) {
                    points.push((d, -(d - to)));
                }
            }
            for (depth, value) in points {
                let point = path.at(depth)?;
                let key = point.coordinates().map(f64::to_bits);
                if !seen.insert(key) {
                    return Err("duplicate domain constraint location; reconcile intervals".into());
                }
                constraint_holes.push(id.clone());
                constraints.push(Constraint { point, value });
            }
        }
    }
    let audit = vec![
        ("domain_holes".into(), domains.len()),
        ("composites".into(), samples.len()),
        ("domain_constraints".into(), constraints.len()),
        ("collar_surveys_inserted".into(), inserted),
        ("low_recovery_rows_excluded_in_domain_holes".into(), low),
        (
            "unmeasured_density_values_excluded_in_domain_holes".into(),
            density,
        ),
    ];
    Ok(Data {
        samples,
        constraints,
        domain_checks,
        constraint_holes,
        holes: domains.len(),
        audit,
    })
}

/// XY bucket index over the supplied piecewise-planar topographic triangles.
pub struct Topography {
    triangles: Vec<[[f64; 3]; 3]>,
    buckets: BTreeMap<[i64; 2], Vec<usize>>,
}
impl Topography {
    pub fn read(path: &Path) -> Result<Self> {
        use std::io::{BufRead, BufReader};
        let mut lines = BufReader::new(File::open(path)?).lines();
        let mut triangles = Vec::new();
        let mut vertices = [[f64::NAN; 3]; 4];
        let mut face = false;
        while let Some(group) = lines.next() {
            let group: i32 = group?.trim().parse()?;
            let value = lines.next().ok_or("truncated DXF")??;
            if group == 0 {
                if face {
                    if vertices.iter().flatten().any(|x| !x.is_finite()) {
                        return Err("invalid DXF face".into());
                    }
                    triangles.push([vertices[0], vertices[1], vertices[2]]);
                    if vertices[3] != vertices[2] {
                        triangles.push([vertices[0], vertices[2], vertices[3]]);
                    }
                }
                face = value.trim() == "3DFACE";
                vertices = [[f64::NAN; 3]; 4];
            } else if face {
                for (axis, base) in [10, 20, 30].into_iter().enumerate() {
                    if (base..base + 4).contains(&group) {
                        vertices[(group - base) as usize][axis] = value.trim().parse()?;
                    }
                }
            }
        }
        if triangles.is_empty() {
            return Err("no topographic triangles".into());
        }
        let mut buckets: BTreeMap<[i64; 2], Vec<usize>> = BTreeMap::new();
        for (i, t) in triangles.iter().enumerate() {
            let lo: [i64; 2] = std::array::from_fn(|a| {
                (t.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min) / 50.).floor() as i64
            });
            let hi: [i64; 2] = std::array::from_fn(|a| {
                (t.iter().map(|p| p[a]).fold(f64::NEG_INFINITY, f64::max) / 50.).floor() as i64
            });
            if hi[0] - lo[0] > 1000 || hi[1] - lo[1] > 1000 {
                return Err("unexpected topographic triangle extent".into());
            }
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    buckets.entry([x, y]).or_default().push(i);
                }
            }
        }
        Ok(Self { triangles, buckets })
    }
    pub fn triangles(&self) -> &[[[f64; 3]; 3]] {
        &self.triangles
    }
    /// Height, horizontal slope and source triangle. Conflicting overlapping
    /// triangles are rejected rather than selecting one by file order.
    pub fn surface(&self, p: [f64; 3]) -> Result<Option<(f64, [f64; 2], usize)>> {
        if p.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite topography query".into());
        }
        let key = [(p[0] / 50.).floor() as i64, (p[1] / 50.).floor() as i64];
        let Some(indices) = self.buckets.get(&key) else {
            return Ok(None);
        };
        let mut found: Option<(f64, [f64; 2], usize)> = None;
        for &i in indices {
            let [a, b, c] = self.triangles[i];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let det = u[0] * v[1] - u[1] * v[0];
            if det.abs() < 1e-12 {
                continue;
            }
            let x = p[0] - a[0];
            let y = p[1] - a[1];
            let s = (x * v[1] - y * v[0]) / det;
            let t = (u[0] * y - u[1] * x) / det;
            if s < -1e-10 || t < -1e-10 || s + t > 1. + 1e-10 {
                continue;
            }
            let height = a[2] + s * u[2] + t * v[2];
            let slope = [
                (u[2] * v[1] - u[1] * v[2]) / det,
                (u[0] * v[2] - u[2] * v[0]) / det,
            ];
            if !height.is_finite() || slope.iter().any(|v| !v.is_finite()) {
                return Err("invalid topography interpolation".into());
            }
            if let Some((previous, _, _)) = found {
                if (height - previous).abs() > 1e-5 {
                    return Err("overlapping topography has inconsistent heights".into());
                }
            } else {
                found = Some((height, slope, i));
            }
        }
        Ok(found)
    }
    pub fn height(&self, p: [f64; 3]) -> Option<f64> {
        let key = [(p[0] / 50.).floor() as i64, (p[1] / 50.).floor() as i64];
        for &i in self.buckets.get(&key)? {
            let [a, b, c] = self.triangles[i];
            let d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
            if d.abs() < 1e-12 {
                continue;
            }
            let u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d;
            let v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d;
            if u >= -1e-10 && v >= -1e-10 && u + v <= 1. + 1e-10 {
                return Some(u * a[2] + v * b[2] + (1. - u - v) * c[2]);
            }
        }
        None
    }
}
