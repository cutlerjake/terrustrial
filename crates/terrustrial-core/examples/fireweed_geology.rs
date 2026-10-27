//! Evidence audit and staged geological reconstruction. Local output only.
#[path = "support/shared/drillhole.rs"]
#[allow(dead_code)]
mod drillhole;
#[path = "support/fireweed/geology.rs"]
mod geology;
#[path = "support/fireweed/input.rs"]
#[allow(dead_code)]
mod input;
use std::{
    fs::{self, File},
    path::Path,
};
use terrustrial_core::domains::evidence::Membership;
fn main() -> input::Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: fireweed_geology RAW_DIRECTORY NEW_OUTPUT_DIRECTORY".into());
    }
    let holes = geology::load(Path::new(&a[1]))?;
    let out = Path::new(&a[2]);
    fs::create_dir(out)?;
    let mut intervals = csv::Writer::from_writer(File::create_new(out.join("intervals.csv"))?);
    intervals.write_record(["hole", "from", "to", "membership", "code", "source_rows"])?;
    let mut checks = csv::Writer::from_writer(File::create_new(out.join("evidence.csv"))?);
    checks.write_record(["hole", "depth", "x", "y", "z", "membership"])?;
    let mut contacts = csv::Writer::from_writer(File::create_new(out.join("contacts.csv"))?);
    contacts.write_record([
        "hole",
        "depth",
        "x",
        "y",
        "z",
        "known_exterior",
        "source_rows",
    ])?;
    let mut counts = [0_usize; 3];
    let mut contact_count = 0;
    for h in &holes {
        for r in &h.evidence.intervals {
            intervals.write_record([
                h.evidence.id.clone(),
                r.from.to_string(),
                r.to.to_string(),
                format!("{:?}", r.membership),
                r.code.clone().unwrap_or_default(),
                r.source_rows
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(";"),
            ])?;
        }
        for (depth, m) in h.evidence.sample(2., 100_000)? {
            counts[match m {
                Membership::Interior => 0,
                Membership::Exterior => 1,
                Membership::Unknown => 2,
            }] += 1;
            let p = h.trajectory.at(depth)?.coordinates();
            checks.write_record([
                h.evidence.id.clone(),
                depth.to_string(),
                p[0].to_string(),
                p[1].to_string(),
                p[2].to_string(),
                format!("{m:?}"),
            ])?;
        }
        for c in &h.evidence.contacts {
            contact_count += 1;
            let p = h.trajectory.at(c.depth)?.coordinates();
            contacts.write_record([
                h.evidence.id.clone(),
                c.depth.to_string(),
                p[0].to_string(),
                p[1].to_string(),
                p[2].to_string(),
                c.known_exterior.to_string(),
                c.source_rows
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(";"),
            ])?;
        }
    }
    intervals.flush()?;
    checks.flush()?;
    contacts.flush()?;
    let report = serde_json::json!({"holes":holes.len(),"interior_checks":counts[0],"exterior_checks":counts[1],"unknown_checks":counts[2],"contacts":contact_count,"inside_convention":"negative","geometry_validated":false,"geologically_accepted":false});
    serde_json::to_writer_pretty(File::create_new(out.join("evidence-report.json"))?, &report)?;
    println!("{report}");
    Ok(())
}
