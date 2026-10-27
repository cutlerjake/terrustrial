//! All Boundary-prospect interpretation evidence, independent of assay eligibility.
use super::{
    drillhole::{Survey, Trajectory},
    input::{Result, Table},
};
use std::{collections::BTreeMap, path::Path};
use terrustrial_core::{
    data::Point3,
    domains::evidence::{Hole, Record},
};
pub struct GeologicalHole {
    pub evidence: Hole,
    pub trajectory: Trajectory,
}
pub fn load(root: &Path) -> Result<Vec<GeologicalHole>> {
    let interp = Table::read(root, "MPA_Interp_20240227.csv")?;
    let collar = Table::read(root, "MPA_Collar_20240227.csv")?;
    let surveys = Table::read(root, "MPA_Survey_20240227.csv")?;
    let mut records: BTreeMap<String, Vec<Record>> = BTreeMap::new();
    for (i, r) in interp.rows.iter().enumerate() {
        records
            .entry(interp.text(r, "holeid")?.into())
            .or_default()
            .push(Record {
                from: interp.number(r, "from")?,
                to: interp.number(r, "to")?,
                code: interp.text(r, "Code")?.into(),
                source_rows: vec![i as u64 + 2],
            });
    }
    let mut survey_map: BTreeMap<String, Vec<Survey>> = BTreeMap::new();
    for r in &surveys.rows {
        survey_map
            .entry(surveys.text(r, "HoleID")?.into())
            .or_default()
            .push(Survey {
                depth: surveys.number(r, "Depth_m")?,
                azimuth: surveys.number(r, "Azimuth")?,
                dip: surveys.number(r, "Dip")?,
            });
    }
    let mut holes = Vec::new();
    for r in &collar.rows {
        if collar.text(r, "Property")? != "NIDD"
            || !collar.text(r, "Prospect")?.starts_with("Boundary")
        {
            continue;
        }
        if collar.text(r, "Grid")? != "NAD83 / UTM zone 9N" {
            return Err("unexpected geological coordinate system".into());
        }
        let id = collar.text(r, "HoleID")?;
        let length = collar.number(r, "Length_m")?;
        let p = Point3::new([
            collar.number(r, "Easting")?,
            collar.number(r, "Northing")?,
            collar.number(r, "Elevation")?,
        ])?;
        let mut stations = survey_map.remove(id).unwrap_or_default();
        stations.sort_by(|a, b| a.depth.total_cmp(&b.depth));
        if stations.first().is_none_or(|s| s.depth > 0.) {
            stations.insert(
                0,
                Survey {
                    depth: 0.,
                    dip: collar.number(r, "Dip")?,
                    azimuth: collar.number(r, "Azimuth")?,
                },
            );
        }
        let evidence = Hole::new(
            id.into(),
            length,
            "BZPZ",
            records.remove(id).unwrap_or_default(),
        )?;
        holes.push(GeologicalHole {
            evidence,
            trajectory: Trajectory::new(p, &stations, length)?,
        });
    }
    holes.sort_by(|a, b| a.evidence.id.cmp(&b.evidence.id));
    if holes
        .windows(2)
        .any(|h| h[0].evidence.id == h[1].evidence.id)
    {
        return Err("duplicate geological hole".into());
    }
    Ok(holes)
}
