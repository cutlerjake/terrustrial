//! Versioned payload encoding. Format 3 writes only values after the immutable
//! full-record base; format 2 remains readable and writable without migration.
use super::*;

pub(super) struct Encoded {
    pub generation: u64,
    body: Vec<u8>,
    crc: u32,
    base: Option<(Vec<u8>, u32)>,
}
impl Encoded {
    pub fn bytes(&self) -> usize {
        self.body.len() + self.base.as_ref().map_or(0, |(body, _)| body.len())
    }
    pub fn blobs(&self) -> u64 {
        1 + u64::from(self.base.is_some())
    }
    pub fn materialize(self, capacity: usize) -> Result<Vec<u8>> {
        if crc32fast::hash(&self.body) != self.crc {
            return Err(invalid("packed payload checksum mismatch"));
        }
        let body = if let Some((mut base, crc)) = self.base {
            if crc32fast::hash(&base) != crc
                || base.is_empty()
                || base.len() % RECORD_BYTES != 0
                || base.len() / RECORD_BYTES > capacity
                || self.body.len() != base.len() / RECORD_BYTES * 8
            {
                return Err(invalid("packed value/base payload association"));
            }
            for (record, value) in base
                .as_chunks_mut::<RECORD_BYTES>()
                .0
                .iter_mut()
                .zip(self.body.as_chunks::<8>().0)
            {
                record[88..96].copy_from_slice(value);
            }
            base
        } else {
            self.body
        };
        if body.is_empty() || body.len() % RECORD_BYTES != 0 || body.len() / RECORD_BYTES > capacity
        {
            return Err(invalid("packed payload length"));
        }
        Ok(body)
    }
}

/// Fetch owned blobs, then return the connection before checksum/decoding work.
/// Length checks apply to the latest selected version, never filter it out and
/// accidentally substitute an earlier value when a latest blob is corrupt.
pub(super) fn fetch(
    c: &Connection,
    page: u64,
    generation: u64,
    capacity: usize,
    compact: bool,
) -> Result<Encoded> {
    let sql = if compact {
        "SELECT encoding,CASE WHEN length(body)<=CASE encoding WHEN 1 THEN ?3*8 ELSE ?3*104 END THEN body ELSE NULL END,crc,generation FROM payloads WHERE page=?1 AND generation<=?2 ORDER BY generation DESC LIMIT 1"
    } else {
        "SELECT 0,CASE WHEN length(body)<=?3*104 THEN body ELSE NULL END,crc,generation FROM payloads WHERE page=?1 AND generation<=?2 ORDER BY generation DESC LIMIT 1"
    };
    let (encoding, body, crc, stored): (u8, Vec<u8>, u32, u64) = c.prepare_cached(sql)?.query_row(
        params![sql_id(page)?, sql_id(generation)?, capacity as i64],
        |r| {
            let _span = Span::new(Metric::PageBlobCopy);
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, sql_u64(r, 3)?))
        },
    )?;
    let base = match encoding {
        0 => None,
        1 if compact => Some(c.prepare_cached("SELECT CASE WHEN length(body)<=?3 THEN body ELSE NULL END,crc FROM payloads WHERE page=?1 AND generation<=?2 AND encoding=0 ORDER BY generation LIMIT 1")?.query_row(
            params![sql_id(page)?, sql_id(stored)?, (capacity * RECORD_BYTES) as i64],
            |r| {
                let _span = Span::new(Metric::PageBlobCopy);
                Ok((r.get(0)?, r.get(1)?))
            },
        )?),
        _ => return Err(invalid("unknown packed payload encoding")),
    };
    Ok(Encoded {
        generation: stored,
        body,
        crc,
        base,
    })
}

pub(super) fn write_values(
    c: &Connection,
    page: u64,
    generation: u64,
    records: &[u8],
    compact: bool,
    values: &mut Vec<u8>,
) -> Result<()> {
    if !compact {
        return builder::write_payload(c, page, generation, records);
    }
    values.clear();
    for record in records.as_chunks::<RECORD_BYTES>().0 {
        values.extend_from_slice(&record[88..96]);
    }
    write_value_bytes(c, page, generation, values)
}

pub(super) fn write_value_bytes(
    c: &Connection,
    page: u64,
    generation: u64,
    values: &[u8],
) -> Result<()> {
    let _span = Span::new(Metric::PublishInsert);
    c.prepare_cached(
        "INSERT INTO payloads(page,generation,body,crc,encoding) VALUES(?1,?2,?3,?4,1)",
    )?
    .execute(params![
        sql_id(page)?,
        sql_id(generation)?,
        values,
        crc32fast::hash(values)
    ])?;
    Ok(())
}
