//! Shared power-of-two scaling without materializing overflowing scale factors.
/// Floor log2 for positive finite normal and subnormal values, without logarithms.
pub(crate) fn exponent(value: f64) -> i32 {
    let bits = value.to_bits();
    let encoded = ((bits >> 52) & 0x7ff) as i32;
    if encoded == 0 {
        63 - bits.leading_zeros() as i32 - 1074
    } else {
        encoded - 1023
    }
}
pub(crate) fn power(exponent: i32) -> f64 {
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (exponent + 1074))
    }
}
pub(crate) fn shift(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 1023 {
        value *= power(1023);
        exponent -= 1023;
    }
    while exponent < -1074 {
        value *= power(-1022);
        exponent += 1022;
    }
    value * power(exponent)
}
