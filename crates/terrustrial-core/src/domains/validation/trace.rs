//! Adaptive checks in downhole metres. The caller supplies an enclosure of the
//! field over the entire curved trace segment, not just its endpoints.
use crate::{error::invalid, execution::Cancellation, Result};
#[derive(Clone, Copy)]
pub struct Options {
    pub depth_tolerance_m: f64,
    pub maximum_cells: usize,
}
#[derive(Debug, Default)]
pub struct SignAudit {
    pub enclosed_consistent_length_m: f64,
    pub witnessed_violating_length_m: f64,
    pub unresolved_length_m: f64,
    pub visited_cells: usize,
}
#[derive(Debug, Default)]
pub struct Roots {
    /// Each bracket has a witnessed sign change or an evaluated zero endpoint.
    /// Its width is at most the requested downhole tolerance.
    pub brackets: Vec<[f64; 2]>,
    /// Cells whose enclosure contains zero without a witnessed root. These
    /// include unresolved tangencies and multiple crossings below resolution.
    pub unresolved: Vec<[f64; 2]>,
    pub visited_cells: usize,
}
fn validate(from: f64, to: f64, o: Options) -> Result<()> {
    if !from.is_finite()
        || !to.is_finite()
        || from < 0.
        || to <= from
        || !o.depth_tolerance_m.is_finite()
        || o.depth_tolerance_m <= 0.
        || o.maximum_cells == 0
    {
        return Err(invalid("trace audit interval or options"));
    }
    Ok(())
}
fn bounds(b: [f64; 2]) -> Result<[f64; 2]> {
    if b.iter().any(|x| x.is_nan()) || b[0] > b[1] {
        Err(invalid("invalid trace field enclosure"))
    } else {
        Ok(b)
    }
}
fn value(x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid("nonfinite trace field value"))
    }
}
pub fn audit_sign(
    from: f64,
    to: f64,
    inside: bool,
    options: Options,
    mut enclose: impl FnMut(f64, f64) -> Result<[f64; 2]>,
    cancel: &Cancellation,
) -> Result<SignAudit> {
    validate(from, to, options)?;
    let mut report = SignAudit::default();
    let mut stack = vec![[from, to]];
    while let Some([a, b]) = stack.pop() {
        cancel.check()?;
        if report.visited_cells >= options.maximum_cells {
            report.unresolved_length_m += b - a + stack.iter().map(|[a, b]| b - a).sum::<f64>();
            break;
        }
        report.visited_cells += 1;
        let [lo, hi] = bounds(enclose(a, b)?)?;
        let consistent = if inside { hi < 0. } else { lo > 0. };
        let violated = if inside { lo > 0. } else { hi < 0. };
        if consistent {
            report.enclosed_consistent_length_m += b - a;
        } else if violated {
            report.witnessed_violating_length_m += b - a;
        } else {
            let mid = a + (b - a) * 0.5;
            if b - a <= options.depth_tolerance_m || mid == a || mid == b {
                report.unresolved_length_m += b - a;
            } else {
                stack.push([mid, b]);
                stack.push([a, mid]);
            }
        }
    }
    Ok(report)
}
pub fn roots(
    from: f64,
    to: f64,
    options: Options,
    mut enclose: impl FnMut(f64, f64) -> Result<[f64; 2]>,
    mut sample: impl FnMut(f64) -> Result<f64>,
    cancel: &Cancellation,
) -> Result<Roots> {
    validate(from, to, options)?;
    let mut report = Roots::default();
    let mut stack = vec![[from, to]];
    while let Some([a, b]) = stack.pop() {
        cancel.check()?;
        if report.visited_cells >= options.maximum_cells {
            report.unresolved.push([a, b]);
            report.unresolved.extend(stack);
            break;
        }
        report.visited_cells += 1;
        let [lo, hi] = bounds(enclose(a, b)?)?;
        if lo > 0. || hi < 0. {
            continue;
        }
        let mid = a + (b - a) * 0.5;
        if b - a <= options.depth_tolerance_m || mid == a || mid == b {
            let x = value(sample(a)?)?;
            let y = value(sample(b)?)?;
            if x == 0. || y == 0. || x.is_sign_negative() != y.is_sign_negative() {
                report.brackets.push([a, b]);
            } else {
                report.unresolved.push([a, b]);
            }
        } else {
            stack.push([mid, b]);
            stack.push([a, mid]);
        }
    }
    report.brackets.sort_by(|a, b| a[0].total_cmp(&b[0]));
    report.unresolved.sort_by(|a, b| a[0].total_cmp(&b[0]));
    Ok(report)
}
