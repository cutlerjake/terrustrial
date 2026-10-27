//! Fallible reusable buffer growth with pre-allocation admission and rollback.
use crate::{
    error::{checked_len, invalid},
    execution::Reservation,
    Error, Result,
};
// Buffer growth is charged before allocation. Failed growth preserves the prior
// charge and data, and successful smaller queries reuse the existing capacity.
pub(crate) fn grow<T>(
    buffer: &mut Vec<T>,
    requested: usize,
    reservation: &mut Reservation,
) -> Result<()> {
    if requested > buffer.capacity() {
        let previous = reservation.bytes();
        let bytes = checked_len(&[requested, std::mem::size_of::<T>()])?;
        reservation.resize(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        if buffer.try_reserve_exact(requested - buffer.len()).is_err() {
            reservation.resize(previous)?;
            return Err(invalid("group input buffer allocation"));
        }
    }
    Ok(())
}
