//! Import and query the repository example without owning the dataset in memory.
use std::{borrow::Cow, path::PathBuf, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::{NearestWorkspace, Neighborhood, ObservationSource},
    storage::SqliteStore,
    Error, Result,
};
struct CsvPoints(PathBuf);
fn invalid(error: impl std::fmt::Display) -> Error {
    Error::InvalidInput(error.to_string())
}
impl ObservationSource for CsvPoints {
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        let mut reader = csv::Reader::from_path(&self.0).map_err(invalid)?;
        let headers = reader.headers().map_err(invalid)?.clone();
        let mut indices = [0; 4];
        for (index, name) in indices.iter_mut().zip(["x", "y", "z", "porosity.dat"]) {
            *index = headers
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| invalid(format!("missing column {name}")))?;
        }
        let mut record = csv::StringRecord::new();
        let mut id = 0_u64;
        while reader.read_record(&mut record).map_err(invalid)? {
            let mut values = [0.0; 4];
            for (value, index) in values.iter_mut().zip(indices) {
                *value = record[index].parse().map_err(invalid)?;
            }
            visitor(&Observation::new(
                id,
                Site::point(Point3::new([values[0], values[1], values[2]])?, 0),
                values[3],
                0.0,
            )?)?;
            id = id.checked_add(1).ok_or(Error::SizeOverflow)?;
        }
        Ok(())
    }
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let mut metrics: Vec<(&str, String)> = Vec::new();
    match args.as_slice() {
        [command, csv, database] if command == "build" => {
            let mut store = SqliteStore::create(database, 16 << 20, &budget)?;
            let start = Instant::now();
            store.import(&CsvPoints(csv.into()), &cancel)?;
            let seconds = start.elapsed().as_secs_f64();
            let rows = store.len()?;
            metrics.extend([
                ("records", rows.to_string()),
                ("seconds", seconds.to_string()),
                ("records_per_second", (rows as f64 / seconds).to_string()),
                (
                    "database_bytes",
                    std::fs::metadata(database)?.len().to_string(),
                ),
            ]);
        }
        [command, database, count] if command == "query" || command == "query-reuse" => {
            let count: usize = count.to_str().ok_or("query count")?.parse()?;
            if count == 0 {
                return Err("positive query count required".into());
            }
            let store = SqliteStore::open(database, 16 << 20, &budget)?;
            let query = Neighborhood { maximum: 32, radius: 4.0 };
            let mut workspace = NearestWorkspace::new(query, &budget)?;
            let start = Instant::now();
            let mut checksum = 0_u64;
            let mut returned = 0_u64;
            for i in 0..count {
                let point = Point3::new([
                    ((i * 71) % 150) as f64 + 0.25,
                    ((i * 113) % 200) as f64 + 0.5,
                    ((i * 137) % 200) as f64 + 0.125,
                ])?;
                let result = if command == "query-reuse" {
                    Cow::Borrowed(store.nearest_into(point, &mut workspace, &cancel)?)
                } else {
                    Cow::Owned(store.nearest(point, query, &cancel)?)
                };
                for observation in result.iter() {
                    returned += 1;
                    checksum = checksum.wrapping_add(observation.id());
                }
            }
            let seconds = start.elapsed().as_secs_f64();
            metrics.extend([
                ("queries", count.to_string()),
                ("seconds", seconds.to_string()),
                ("queries_per_second", (count as f64 / seconds).to_string()),
                ("returned", returned.to_string()),
                ("id_checksum", checksum.to_string()),
            ]);
        }
        [command, database] if command == "digest" => {
            let store = SqliteStore::open(database, 16 << 20, &budget)?;
            let digest = terrustrial_core::identity::observations(&store, &cancel)?;
            metrics.push(("records", store.len()?.to_string()));
            metrics.push(("digest", digest.to_bytes().iter().map(|b| format!("{b:02x}")).collect()));
        }
        _ => {
            return Err(
                "usage: storage_benchmark build INPUT.csv OUTPUT.sqlite | query[-reuse] INPUT.sqlite COUNT | digest INPUT.sqlite"
                    .into(),
            )
        }
    }
    let mut writer = csv::Writer::from_writer(std::io::stdout().lock());
    writer.write_record(["metric", "value"])?;
    for (key, value) in metrics {
        writer.write_record([key, &value])?;
    }
    writer.flush()?;
    Ok(())
}
