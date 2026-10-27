//! Spatially blocked selection of regularization for a Gaussian tensor field.
//!
//! Each evidence window belongs to one supplied spatial fold. All its bins are
//! removed together during validation. Sills, nuggets, window geometry, links
//! and folds must be specified independently of the validation bin values.
//! Selection evidence is not independent final acceptance evidence. This module
//! selects family and smoothing, not window geometry or marginal transforms.
use super::{
    field,
    field::{Edge, FieldFit, Options, Window},
    fitting::DirectionalBin,
    Family,
};
use crate::{
    data::Point3,
    error::{checked_len, invalid},
    Error, Result,
};

#[derive(Clone, Debug)]
pub struct Candidate {
    pub family: Family,
    pub regularization: f64,
    /// Equal-weight mean across spatial folds. For select_families this is
    /// normalized window variogram MSE; select_families_by defines its own loss.
    pub validation_loss: Option<f64>,
    pub fold_loss: Vec<Option<f64>>,
    /// A failed or unconverged fold disqualifies the whole candidate. Other
    /// candidates still run. Cancellation/resource failures propagate immediately.
    pub failures: Vec<Option<String>>,
}
pub struct Selection {
    pub fit: FieldFit,
    pub selected: usize,
    pub candidates: Vec<Candidate>,
}

/// Fit every candidate on every training partition, score held-out windows with
/// the *nonstationary* Gaussian covariance, then refit the selected regularization
/// on all evidence. No bins from a held-out window enter its tensor solve.
///
/// Folds must be contiguous labels starting at zero, with at least two folds.
/// Windows may overlap spatially, but their evidence must not share observations
/// across folds. The caller owns that provenance: it cannot be inferred from
/// already aggregated directional bins. Do not use random bin splitting.
///
/// Validation interprets a bin as point endpoints at center +/- lag/2, with a
/// common supplied sill and nugget at those endpoints. This is a diagnostic for
/// locally stationary experimental bins, not a block-support likelihood or a
/// substitute for drillhole-held-out prediction. Final refit convergence is
/// required. Ties retain caller candidate order. Fitting runs serially with one
/// admitted field workspace, independently of Rayon worker count.
pub fn select_regularization(
    windows: &[Window<'_>],
    edges: &[Edge],
    folds: &[usize],
    regularizations: &[f64],
    options: Options<'_>,
) -> Result<Selection> {
    select_families(
        windows,
        edges,
        folds,
        &[Family::Gaussian],
        regularizations,
        options,
    )
}

/// Select the covariance family and field smoothing jointly. Candidate order is
/// family-major, then regularization-major. Sill and nugget inputs remain fixed
/// independent priors; this does not select them using validation observations.
pub fn select_families(
    windows: &[Window<'_>],
    edges: &[Edge],
    folds: &[usize],
    families: &[Family],
    regularizations: &[f64],
    options: Options<'_>,
) -> Result<Selection> {
    let cancel = options.cancellation;
    select_families_by(
        windows,
        edges,
        folds,
        families,
        regularizations,
        options,
        |fit, fold| {
            let count = folds.iter().filter(|&&label| label == fold).count();
            let mut loss = 0.0;
            for (w, &label) in windows.iter().zip(folds) {
                if label == fold {
                    cancel.check()?;
                    loss += score(w, fit)? / count as f64;
                }
            }
            Ok(loss)
        },
    )
}

/// Select using caller-supplied spatial prediction loss. The scorer receives a
/// fit made without *any bins* from the identified fold. It must also withhold
/// that fold's raw observations from kriging conditioning, use independent
/// priors/transforms, and return a finite loss (smaller is better). Negative
/// losses, such as continuous negative log densities, are valid. Each fold has
/// equal weight. Errors disqualify the candidate except resource/cancellation
/// errors, which stop immediately. Compare these losses to a stationary baseline
/// on exactly the same folds before accepting a local model.
///
/// This function does not access raw observations, so it cannot enforce the
/// scorer's provenance. Selection loss is not independent final validation.
pub fn select_families_by(
    windows: &[Window<'_>],
    edges: &[Edge],
    folds: &[usize],
    families: &[Family],
    regularizations: &[f64],
    options: Options<'_>,
    mut evaluate: impl FnMut(&FieldFit, usize) -> Result<f64>,
) -> Result<Selection> {
    options.cancellation.check()?;
    if families.is_empty()
        || windows.len() != folds.len()
        || windows.is_empty()
        || regularizations.is_empty()
        || regularizations.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(invalid(
            "local anisotropy selection dimensions or positive regularizations",
        ));
    }
    let count = folds
        .iter()
        .copied()
        .max()
        .unwrap()
        .checked_add(1)
        .ok_or(Error::SizeOverflow)?;
    if count < 2 || count > windows.len() {
        return Err(invalid("at least two nonempty spatial folds required"));
    }
    // Admission precedes all size-dependent allocations, including diagnostics.
    let bytes = checked_len(&[windows.len(), 128])?
        .checked_add(checked_len(&[
            families.len(),
            regularizations.len(),
            count,
            256,
        ])?)
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let mut counts = vec![0usize; count];
    for (w, &fold) in windows.iter().zip(folds) {
        if !w.structured_sill.is_finite()
            || w.structured_sill <= 0.0
            || !w.nugget.is_finite()
            || w.nugget < 0.0
            || !(w.structured_sill + w.nugget).is_finite()
            || !w.bins.iter().any(|b| b.pairs > 0)
        {
            return Err(invalid(
                "selection window requires finite positive sill and validation evidence",
            ));
        }
        for bin in w.bins {
            validate_bin(w.position, bin)?;
        }
        counts[fold] += 1;
    }
    if counts.contains(&0) {
        return Err(invalid("spatial fold labels must be contiguous"));
    }
    let mut training: Vec<_> = windows
        .iter()
        .map(|w| Window {
            position: w.position,
            bins: w.bins,
            structured_sill: w.structured_sill,
            nugget: w.nugget,
        })
        .collect();
    let mut candidates = Vec::with_capacity(checked_len(&[families.len(), regularizations.len()])?);
    let mut selected = None;
    let mut best = f64::INFINITY;
    for (&family, &lambda) in families
        .iter()
        .flat_map(|family| regularizations.iter().map(move |lambda| (family, lambda)))
    {
        let mut candidate = Candidate {
            family,
            regularization: lambda,
            validation_loss: None,
            fold_loss: vec![None; count],
            failures: vec![None; count],
        };
        for (fold, _) in counts.iter().enumerate() {
            options.cancellation.check()?;
            for ((out, w), &label) in training.iter_mut().zip(windows).zip(folds) {
                out.bins = if label == fold { &[] } else { w.bins };
            }
            let result =
                field::fit_family(&training, edges, family, copy_options(&options, lambda));
            let fit = match result {
                Ok(fit) if fit.converged => fit,
                Ok(fit) => {
                    candidate.failures[fold] = Some(format!(
                        "unconverged field: stationarity {}",
                        fit.stationarity
                    ));
                    continue;
                }
                Err(e @ (Error::Cancelled | Error::MemoryBudget { .. } | Error::SizeOverflow)) => {
                    return Err(e)
                }
                Err(e) => {
                    candidate.failures[fold] = Some(e.to_string());
                    continue;
                }
            };
            match evaluate(&fit, fold) {
                Ok(loss) if loss.is_finite() => candidate.fold_loss[fold] = Some(loss),
                Ok(_) => candidate.failures[fold] = Some("nonfinite validation loss".into()),
                Err(e @ (Error::Cancelled | Error::MemoryBudget { .. } | Error::SizeOverflow)) => {
                    return Err(e)
                }
                Err(e) => candidate.failures[fold] = Some(e.to_string()),
            }
        }
        if candidate.failures.iter().all(Option::is_none) {
            let mse = candidate
                .fold_loss
                .iter()
                .map(|x| x.unwrap() / count as f64)
                .sum::<f64>();
            if mse < best {
                best = mse;
                selected = Some(candidates.len());
            }
            candidate.validation_loss = Some(mse);
        }
        candidates.push(candidate);
    }
    let selected = selected.ok_or_else(|| {
        invalid(format!(
            "no local anisotropy candidate passed every spatial fold: {:?}",
            candidates.iter().map(|c| &c.failures).collect::<Vec<_>>()
        ))
    })?;
    let fit = field::fit_family(
        windows,
        edges,
        candidates[selected].family,
        copy_options(&options, candidates[selected].regularization),
    )?;
    if !fit.converged {
        return Err(Error::Convergence {
            context: "selected local anisotropy refit",
            iterations: fit.iterations,
            residual: fit.stationarity,
        });
    }
    Ok(Selection {
        fit,
        selected,
        candidates,
    })
}

fn copy_options<'a>(o: &Options<'a>, regularization: f64) -> Options<'a> {
    Options {
        range_bounds: o.range_bounds,
        regularization,
        maximum_iterations: o.maximum_iterations,
        tolerance: o.tolerance,
        maximum_cg_iterations: o.maximum_cg_iterations,
        cg_tolerance: o.cg_tolerance,
        rank_tolerance: o.rank_tolerance,
        interpolation_neighbors: o.interpolation_neighbors,
        interpolation_power: o.interpolation_power,
        budget: o.budget,
        cancellation: o.cancellation,
    }
}
fn endpoints(center: Point3, lag: [f64; 3]) -> Result<[Point3; 2]> {
    let c = center.coordinates();
    Ok([
        Point3::new(std::array::from_fn(|i| c[i] - lag[i] * 0.5))?,
        Point3::new(std::array::from_fn(|i| c[i] + lag[i] * 0.5))?,
    ])
}
fn validate_bin(center: Point3, bin: &DirectionalBin) -> Result<()> {
    if !bin.semivariance.is_finite() || bin.semivariance < 0.0 {
        return Err(invalid("invalid selection semivariance"));
    }
    let [a, b] = endpoints(center, bin.lag)?;
    if bin.pairs > 0 && a == b {
        return Err(invalid(
            "validation lag endpoints must be distinct and representable",
        ));
    }
    Ok(())
}
fn score(window: &Window<'_>, fit: &FieldFit) -> Result<f64> {
    let maximum = window.bins.iter().map(|b| b.pairs).max().unwrap() as f64;
    let mut workspace = fit.field.workspace()?;
    let mut sum = 0.0;
    let mut weight = 0.0;
    let scale = window.structured_sill + window.nugget;
    for bin in window.bins.iter().filter(|b| b.pairs > 0) {
        let [a, b] = endpoints(window.position, bin.lag)?;
        let sa = fit.field.at_with_workspace(a, &mut workspace)?;
        let sb = fit.field.at_with_workspace(b, &mut workspace)?;
        let corr = super::correlation(a, &sa, b, &sb, fit.family)?;
        let prediction = window.nugget / scale + (window.structured_sill / scale) * (1.0 - corr);
        let residual = prediction - bin.semivariance / scale;
        let w = bin.pairs as f64 / maximum;
        sum += w * residual * residual;
        weight += w;
    }
    Ok(sum / weight)
}
