use super::*;
use crate::data::{Site, Support};

pub(super) const RECORD_BYTES: usize = 104;
pub(super) const FANOUT: usize = 8;
pub(super) const MAX_LEAF: usize = 256;

pub(super) use crate::spatial::hierarchy::Bounds;

pub(super) use crate::spatial::paged::Node;

pub(super) struct Leaf {
    pub coordinates: [Vec<f64>; 3],
    pub ranges: Vec<(usize, usize)>,
    pub ranks: Vec<u64>,
    pub ids: Vec<u64>,
}
pub(super) enum Page {
    Branch(Vec<Node>),
    Leaf(Leaf),
    Payload(Payload),
}

pub(super) fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn get_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset.checked_add(8).ok_or(Error::SizeOverflow)?;
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..end)
            .ok_or_else(|| invalid("truncated packed page"))?
            .try_into()
            .unwrap(),
    ))
}
pub(super) fn encode_record(o: &Observation) -> [u8; RECORD_BYTES] {
    let mut out = [0; RECORD_BYTES];
    let site = o.site();
    let mut words = [0u64; 13];
    words[0] = o.id();
    words[1] = site.variable as u64;
    for (i, x) in site
        .support
        .center()
        .coordinates()
        .iter()
        .chain(site.support.half_extents().iter())
        .enumerate()
    {
        words[2 + i] = x.to_bits();
    }
    for (i, x) in site.support.divisions().iter().enumerate() {
        words[8 + i] = *x as u64;
    }
    words[11] = o.value().to_bits();
    words[12] = o.noise_variance().to_bits();
    for (chunk, word) in out.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}
pub(super) fn decode_record(bytes: &[u8]) -> Result<Observation> {
    if bytes.len() != RECORD_BYTES {
        return Err(invalid("packed observation length"));
    }
    let mut w = [0; 13];
    for (i, v) in w.iter_mut().enumerate() {
        *v = get_u64(bytes, i * 8)?;
    }
    let p = Point3::new(std::array::from_fn(|i| f64::from_bits(w[2 + i])))?;
    let half = std::array::from_fn(|i| f64::from_bits(w[5 + i]));
    let mut divisions = [0; 3];
    for i in 0..3 {
        divisions[i] = usize::try_from(w[8 + i]).map_err(|_| Error::SizeOverflow)?;
    }
    Observation::new(
        w[0],
        Site {
            support: Support::block(p, half, divisions)?,
            variable: usize::try_from(w[1]).map_err(|_| Error::SizeOverflow)?,
        },
        f64::from_bits(w[11]),
        f64::from_bits(w[12]),
    )
}

pub(super) fn encode_nodes(nodes: &[Node]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + nodes.len() * 72);
    put_u64(&mut out, nodes.len() as u64);
    for n in nodes {
        put_u64(&mut out, n.id);
        for x in n.bounds.lower.iter().chain(&n.bounds.upper) {
            put_u64(&mut out, x.to_bits());
        }
        put_u64(&mut out, n.first_rank);
        put_u64(&mut out, n.records);
    }
    out
}
pub(super) fn decode_nodes(bytes: &[u8], limit: usize) -> Result<Vec<Node>> {
    let n = usize::try_from(get_u64(bytes, 0)?).map_err(|_| Error::SizeOverflow)?;
    if n > limit || bytes.len() != 8 + n * 72 {
        return Err(invalid("packed branch length"));
    }
    let mut out = Vec::with_capacity(n);
    for b in bytes[8..].as_chunks::<72>().0 {
        let bounds = Bounds {
            lower: std::array::from_fn(|i| f64::from_bits(get_u64(b, 8 + i * 8).unwrap())),
            upper: std::array::from_fn(|i| f64::from_bits(get_u64(b, 32 + i * 8).unwrap())),
        };
        if (0..3).any(|i| {
            !bounds.lower[i].is_finite()
                || !bounds.upper[i].is_finite()
                || bounds.lower[i] > bounds.upper[i]
        }) {
            return Err(invalid("packed branch bounds"));
        }
        let id = get_u64(b, 0)?;
        let records = get_u64(b, 64)?;
        if id == 0 || id > i64::MAX as u64 || records == 0 {
            return Err(invalid("packed branch identity"));
        }
        out.push(Node {
            id,
            bounds,
            first_rank: get_u64(b, 56)?,
            records,
        });
    }
    Ok(out)
}

pub(super) fn encode_leaf(records: &[(Observation, u64)]) -> Vec<u8> {
    let mut groups: Vec<([f64; 3], usize, usize)> = Vec::new();
    for (i, (o, _)) in records.iter().enumerate() {
        let p = o.site().support.center().coordinates();
        if let Some(last) = groups.last_mut().filter(|last| last.0 == p) {
            last.2 = i + 1;
        } else {
            groups.push((p, i, i + 1));
        }
    }
    let mut out = Vec::with_capacity(16 + groups.len() * 40 + records.len() * 16);
    put_u64(&mut out, records.len() as u64);
    put_u64(&mut out, groups.len() as u64);
    for (p, start, end) in groups {
        for x in p {
            put_u64(&mut out, x.to_bits());
        }
        put_u64(&mut out, start as u64);
        put_u64(&mut out, end as u64);
    }
    for (record, rank) in records {
        put_u64(&mut out, *rank);
        put_u64(&mut out, record.id());
    }
    out
}
pub(super) fn decode_payload(bytes: &[u8], capacity: usize) -> Result<Vec<Observation>> {
    if bytes.is_empty()
        || !bytes.len().is_multiple_of(RECORD_BYTES)
        || bytes.len() / RECORD_BYTES > capacity
    {
        return Err(invalid("packed payload length"));
    }
    let records = bytes.as_chunks::<RECORD_BYTES>().0;
    // Preserve the known size across fallible decoding to avoid buffer growth.
    let mut payload = Vec::with_capacity(records.len());
    for record in records {
        payload.push(decode_record(record)?);
    }
    Ok(payload)
}

pub(super) fn decode_page(kind: u8, bytes: &[u8], capacity: usize) -> Result<Page> {
    match kind {
        1 => {
            let nodes = decode_nodes(bytes, FANOUT)?;
            if nodes.is_empty() {
                return Err(invalid("empty packed branch"));
            }
            Ok(Page::Branch(nodes))
        }
        2 => {
            let n = usize::try_from(get_u64(bytes, 0)?).map_err(|_| Error::SizeOverflow)?;
            let groups = usize::try_from(get_u64(bytes, 8)?).map_err(|_| Error::SizeOverflow)?;
            if n == 0
                || n > capacity
                || groups == 0
                || groups > n
                || bytes.len() != 16 + groups * 40 + n * 16
            {
                return Err(invalid("packed geometry length"));
            }
            let mut coordinates: [Vec<f64>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(groups));
            let mut ranges = Vec::with_capacity(groups);
            let mut end = 0;
            for i in 0..groups {
                let off = 16 + i * 40;
                for (axis, coordinates) in coordinates.iter_mut().enumerate() {
                    let v = f64::from_bits(get_u64(bytes, off + axis * 8)?);
                    if !v.is_finite() {
                        return Err(invalid("packed nonfinite coordinate"));
                    }
                    coordinates.push(v);
                }
                let first =
                    usize::try_from(get_u64(bytes, off + 24)?).map_err(|_| Error::SizeOverflow)?;
                let last =
                    usize::try_from(get_u64(bytes, off + 32)?).map_err(|_| Error::SizeOverflow)?;
                if first != end || last <= first || last > n {
                    return Err(invalid("packed geometry association"));
                }
                ranges.push((first, last));
                end = last;
            }
            if end != n {
                return Err(invalid("packed unassociated observation"));
            }
            let mut ranks = Vec::with_capacity(n);
            let mut ids = Vec::with_capacity(n);
            for i in 0..n {
                let off = 16 + groups * 40 + i * 16;
                ranks.push(get_u64(bytes, off)?);
                ids.push(get_u64(bytes, off + 8)?);
            }
            Ok(Page::Leaf(Leaf {
                coordinates,
                ranges,
                ranks,
                ids,
            }))
        }
        _ => Err(invalid("packed page kind")),
    }
}

// This key determines layout only. Original coordinates and exact metric bounds
// determine membership. Half scaling avoids overflow in a wide coordinate span.
pub(super) fn morton(p: [f64; 3], bounds: Bounds) -> u64 {
    let mut key = 0;
    for (axis, coordinate) in p.iter().enumerate() {
        let lo = bounds.lower[axis] * 0.5;
        let span = bounds.upper[axis] * 0.5 - lo;
        let fraction = if span > 0. {
            ((*coordinate * 0.5 - lo) / span).clamp(0., 1.)
        } else {
            0.
        };
        let q = (fraction * ((1u64 << 21) - 1) as f64) as u64;
        for bit in 0..21 {
            key |= ((q >> bit) & 1) << (bit * 3 + axis);
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_payload_capacity_stays_within_decoded_memory_bound() {
        // A fallible collect can round this count up to 256, exceeding the
        // decoded-array allowance in page_charge even without realloc overlap.
        let count = 129;
        let records: Vec<_> = (0..count)
            .map(|id| {
                Observation::new(
                    id as u64,
                    Site {
                        support: Support::block(
                            Point3::new([id as f64, -0.0, 0.25]).unwrap(),
                            [0.5, 0.0, 0.25],
                            [2, 1, 3],
                        )
                        .unwrap(),
                        variable: id % 7,
                    },
                    if id % 2 == 0 { -0.0 } else { id as f64 },
                    0.125,
                )
                .unwrap()
            })
            .collect();
        let mut encoded = Vec::with_capacity(count * RECORD_BYTES);
        for record in &records {
            encoded.extend_from_slice(&encode_record(record));
        }
        // The serialized body already exists when page decoding begins. Count
        // every allocation during decoding, including temporary growth buffers.
        let allocations = allocation_counter::measure(|| {
            let decoded = decode_payload(&encoded, count).unwrap();
            assert_eq!(decoded.len(), records.len());
            for (actual, expected) in decoded.iter().zip(&records) {
                assert!(actual.same_bits(expected));
                assert_eq!(encode_record(actual), encode_record(expected));
            }
        });
        assert!(allocations.bytes_max <= (count * std::mem::size_of::<Observation>()) as u64);
        assert_eq!(allocations.bytes_current, 0);
    }
}
