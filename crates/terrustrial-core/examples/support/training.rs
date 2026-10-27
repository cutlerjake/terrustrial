//! Coordinate-checked loaders shared by the executable benchmark examples.
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};
use terrustrial_core::{
    grid::{Layout, Shape},
    storage::grid::ingest::Rows,
    Error,
};
pub type BenchmarkResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn read_training(path: &Path) -> BenchmarkResult<(Shape, Vec<Option<f64>>)> {
    if path.extension().is_some_and(|extension| extension == "csv") {
        let shape = Shape::new([150, 200, 200])?;
        return Ok((
            shape,
            read_reservoir(File::open(path)?, shape, &["facies.dat"])?,
        ));
    }
    let mut lines = BufReader::new(File::open(path)?).lines();
    let dimensions: Vec<_> = lines
        .next()
        .ok_or("missing dimensions")??
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let shape = Shape::new(
        dimensions
            .try_into()
            .map_err(|_| "three grid dimensions required")?,
    )?;
    if lines.next().ok_or("missing variables")??.trim() != "1" {
        return Err("one categorical attribute required".into());
    }
    lines.next().ok_or("missing variable name")??;
    let values: Vec<Option<f64>> = lines
        .map(|line| Ok(Some(line?.trim().parse::<f64>()?)))
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    Ok((shape, values))
}

/// The checked fixture geometry is explicit. Rows are placed by coordinates, not
/// file order: the example CSV advances z fastest, while Grid advances x fastest.
pub fn read_reservoir(
    reader: impl Read,
    shape: Shape,
    attributes: &[&str],
) -> BenchmarkResult<Vec<Option<f64>>> {
    let mut source = ReservoirRows::new(reader, shape, attributes)?;
    let mut values = vec![
        None;
        shape
            .cells()
            .checked_mul(attributes.len())
            .ok_or("grid storage overflow")?
    ];
    let mut row = vec![None; attributes.len()];
    let mut records = 0_usize;
    while let Some(cell) = source.next(&mut row)? {
        let index = usize::try_from(cell)?
            .checked_mul(attributes.len())
            .ok_or("grid storage overflow")?;
        if values[index].is_some() {
            return Err("duplicate reservoir cell".into());
        }
        values[index..index + attributes.len()].copy_from_slice(&row);
        records += 1;
    }
    if records != shape.cells() {
        return Err("incomplete reservoir grid".into());
    }
    Ok(values)
}

/// Shared reusable CSV row parser. Geometry/value validation is shared by the
/// in-memory reference and external-sort fixture. CSV record storage belongs to
/// this adapter, not the ingestion engine's memory reservation.
pub struct ReservoirRows<R: Read> {
    csv: csv::Reader<R>,
    record: csv::StringRecord,
    columns: Vec<usize>,
    categories: Vec<bool>,
    shape: Shape,
    layout: Layout,
}
impl<R: Read> ReservoirRows<R> {
    pub fn new(reader: R, shape: Shape, attributes: &[&str]) -> BenchmarkResult<Self> {
        if attributes.is_empty() {
            return Err("attributes required".into());
        }
        let mut csv = csv::Reader::from_reader(reader);
        let headers = csv.headers()?;
        let columns = ["x", "y", "z"]
            .iter()
            .chain(attributes)
            .map(|name| {
                let mut matches = headers.iter().enumerate().filter(|(_, h)| *h == *name);
                let column = matches
                    .next()
                    .ok_or("missing reservoir coordinate/attribute column")?
                    .0;
                if matches.next().is_some() {
                    return Err("ambiguous repeated reservoir column");
                }
                Ok(column)
            })
            .collect::<Result<Vec<_>, &str>>()?;
        Ok(Self {
            csv,
            record: csv::StringRecord::new(),
            columns,
            categories: attributes
                .iter()
                .map(|name| *name == "facies.dat")
                .collect(),
            shape,
            layout: Layout::new(shape.dimensions().map(|v| v as u64))?,
        })
    }
}
impl<R: Read> Rows for ReservoirRows<R> {
    fn next(&mut self, values: &mut [Option<f64>]) -> terrustrial_core::Result<Option<u64>> {
        let invalid = |message: &str| Error::InvalidInput(message.into());
        if values.len() != self.categories.len() {
            return Err(invalid("reservoir row dimension"));
        }
        if !self
            .csv
            .read_record(&mut self.record)
            .map_err(|e| Error::InvalidInput(format!("reservoir CSV: {e}")))?
        {
            return Ok(None);
        }
        let mut coordinates = [0; 3];
        for (axis, coordinate) in coordinates.iter_mut().enumerate() {
            let value = self.record[self.columns[axis]]
                .parse::<f64>()
                .map_err(|_| invalid("invalid reservoir coordinate"))?;
            if !value.is_finite()
                || value < 0.0
                || value.fract() != 0.0
                || value >= self.shape.dimensions()[axis] as f64
            {
                return Err(invalid("invalid reservoir grid coordinate"));
            }
            *coordinate = value as u64;
        }
        for (variable, out) in values.iter_mut().enumerate() {
            let value = self.record[self.columns[variable + 3]]
                .parse::<f64>()
                .map_err(|_| invalid("invalid reservoir attribute"))?;
            if !value.is_finite() {
                return Err(invalid("nonfinite reservoir attribute"));
            }
            if self.categories[variable]
                && (value < 0.0 || value.fract() != 0.0 || value >= u32::MAX as f64)
            {
                return Err(invalid("invalid reservoir category"));
            }
            *out = Some(value);
        }
        Ok(Some(self.layout.index(coordinates)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservoir_parser_rejects_ambiguous_headers_and_invalid_coordinates_or_categories() {
        let shape = Shape::new([1, 1, 1]).unwrap();
        for input in [
            "x,x,y,z,facies.dat\n0,0,0,0,1\n",
            "x,y,z,facies.dat\n0.5,0,0,1\n",
            "x,y,z,facies.dat\nNaN,0,0,1\n",
            "x,y,z,facies.dat\n0,0,0,0.5\n",
            "x,y,z,facies.dat\n0,0,0,-1\n",
            "x,y,z,facies.dat\n0,0,0,inf\n",
        ] {
            assert!(read_reservoir(input.as_bytes(), shape, &["facies.dat"]).is_err());
        }
    }
    #[test]
    fn reservoir_rows_are_reordered_by_coordinates_and_incomplete_or_duplicate_grids_fail() {
        let shape = Shape::new([2, 1, 2]).unwrap();
        let input = "x,y,z,facies.dat\n0,0,0,0\n0,0,1,1\n1,0,0,2\n1,0,1,3\n";
        assert_eq!(
            read_reservoir(input.as_bytes(), shape, &["facies.dat"]).unwrap(),
            vec![Some(0.0), Some(2.0), Some(1.0), Some(3.0)]
        );
        let joint =
            "x,y,z,facies.dat,porosity.dat\n0,0,0,0,0.1\n0,0,1,1,0.2\n1,0,0,2,0.3\n1,0,1,3,0.4\n";
        assert_eq!(
            read_reservoir(joint.as_bytes(), shape, &["porosity.dat", "facies.dat"]).unwrap(),
            vec![
                Some(0.1),
                Some(0.0),
                Some(0.3),
                Some(2.0),
                Some(0.2),
                Some(1.0),
                Some(0.4),
                Some(3.0)
            ]
        );
        assert!(read_reservoir(
            "x,y,z,facies.dat\n0,0,0,0\n".as_bytes(),
            shape,
            &["facies.dat"]
        )
        .is_err());
        assert!(read_reservoir(
            format!("{input}0,0,0,0\n").as_bytes(),
            shape,
            &["facies.dat"]
        )
        .is_err());
    }
}
