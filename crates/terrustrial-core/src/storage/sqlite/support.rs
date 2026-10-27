//! Fixed-size little-endian support metadata. NULL is the ordinary point fast path.
use super::*;
use rusqlite::types::ValueRef;

pub(super) fn encode(support: &Support) -> Result<Option<[u8; 48]>> {
    let extents = support.half_extents();
    // Retain negative zero extents even for geometrically point-like supports.
    if extents.map(f64::to_bits) == [0; 3] {
        return Ok(None);
    }
    let mut bytes = [0; 48];
    for (i, extent) in extents.into_iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&extent.to_bits().to_le_bytes());
    }
    for (i, division) in support.divisions().into_iter().enumerate() {
        let count = u64::try_from(division).map_err(|_| Error::SizeOverflow)?;
        bytes[24 + i * 8..32 + i * 8].copy_from_slice(&count.to_le_bytes());
    }
    Ok(Some(bytes))
}
pub(super) fn decode(center: Point3, value: ValueRef<'_>) -> Result<Support> {
    match value {
        ValueRef::Null => Ok(Support::point(center)),
        ValueRef::Blob(bytes) if bytes.len() == 48 => {
            let extents = std::array::from_fn(|i| {
                f64::from_bits(u64::from_le_bytes(
                    bytes[i * 8..i * 8 + 8].try_into().unwrap(),
                ))
            });
            let mut divisions = [0; 3];
            for (i, division) in divisions.iter_mut().enumerate() {
                *division = usize::try_from(u64::from_le_bytes(
                    bytes[24 + i * 8..32 + i * 8].try_into().unwrap(),
                ))
                .map_err(|_| Error::SizeOverflow)?;
            }
            Support::block(center, extents, divisions)
        }
        _ => Err(invalid("support must be NULL or a 48-byte blob")),
    }
}
