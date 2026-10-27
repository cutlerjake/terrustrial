//! Joint total-field and independent LMC-component covariance.
use super::*;

/// A jointly valid covariance for a total field and distinct LMC structures.
/// Total observation variables retain their original IDs. Use `component_site`
/// for component targets. Different components have zero PRIOR cross covariance;
/// their posterior errors generally correlate after conditioning on total data.
/// This model allows the ordinary joint Gaussian engine to retain those terms.
///
/// Component means/drift do not follow from covariances. Supply a corresponding
/// known-mean callback or drift with explicit responses for total and component
/// variables. Centered components use zero drift response, not an intercept of one.
pub struct Components<'a> {
    total: &'a Lmc,
    components: Vec<Lmc>,
    variables: usize,
}
impl<'a> Components<'a> {
    pub fn new(total: &'a Lmc, structures: &[usize]) -> Result<Self> {
        if structures.is_empty() {
            return Err(invalid("at least one component required"));
        }
        let variables = total
            .variables()
            .checked_mul(structures.len().checked_add(1).ok_or(Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
        let mut components = Vec::new();
        components
            .try_reserve_exact(structures.len())
            .map_err(|_| invalid("component model allocation"))?;
        for (i, &structure) in structures.iter().enumerate() {
            if structures[..i].contains(&structure) {
                return Err(invalid("component structures must be distinct"));
            }
            components.push(total.component(structure)?);
        }
        Ok(Self {
            total,
            components,
            variables,
        })
    }
    pub fn component_count(&self) -> usize {
        self.components.len()
    }
    pub fn attribute_count(&self) -> usize {
        self.total.variables()
    }
    /// Encode a base attribute/site as a selected component (zero-based selection
    /// order, not the original LMC structure index). Support is preserved exactly.
    pub fn component_site(&self, component: usize, site: Site) -> Result<Site> {
        if component >= self.component_count() || site.variable >= self.attribute_count() {
            return Err(invalid("component or base attribute out of range"));
        }
        Ok(Site {
            variable: (component + 1) * self.attribute_count() + site.variable,
            ..site
        })
    }
    /// Returns (selected component, base attribute); None denotes the total field.
    pub fn decode_variable(&self, variable: usize) -> Result<(Option<usize>, usize)> {
        if variable >= self.variables {
            return Err(invalid("component variable out of range"));
        }
        let field = variable / self.attribute_count();
        Ok((field.checked_sub(1), variable % self.attribute_count()))
    }
    fn field(&self, variable: usize) -> Result<&Lmc> {
        let (component, _) = self.decode_variable(variable)?;
        Ok(component.map_or(self.total, |i| &self.components[i]))
    }
}
impl Covariance for Components<'_> {
    fn variables(&self) -> usize {
        self.variables
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        let (ca, ia) = self.decode_variable(va)?;
        let (cb, ib) = self.decode_variable(vb)?;
        match (ca, cb) {
            (None, None) => self.total.point_covariance(a, ia, b, ib),
            (Some(i), None) | (None, Some(i)) => self.components[i].point_covariance(a, ia, b, ib),
            (Some(i), Some(j)) if i == j => self.components[i].point_covariance(a, ia, b, ib),
            _ => Ok(0.),
        }
    }
    fn covariance_row(
        &self,
        a: &Site,
        data: &[crate::data::Observation],
        out: &mut [f64],
    ) -> Result<()> {
        if out.len() != data.len() {
            return Err(invalid("component covariance row dimensions"));
        }
        // Total conditioning data use the original LMC's native SIMD kernels.
        if data
            .iter()
            .all(|o| o.site().variable < self.attribute_count())
        {
            let model = self.field(a.variable)?;
            let site = Site {
                variable: a.variable % self.attribute_count(),
                ..*a
            };
            return model.covariance_row(&site, data, out);
        }
        for (value, o) in out.iter_mut().zip(data) {
            *value = self.covariance(a, o.site())?;
        }
        Ok(())
    }
    fn covariance_column(
        &self,
        data: &[crate::data::Observation],
        b: &Site,
        out: &mut [f64],
    ) -> Result<()> {
        if out.len() != data.len() {
            return Err(invalid("component covariance column dimensions"));
        }
        if data
            .iter()
            .all(|o| o.site().variable < self.attribute_count())
        {
            let model = self.field(b.variable)?;
            let site = Site {
                variable: b.variable % self.attribute_count(),
                ..*b
            };
            return model.covariance_column(data, &site, out);
        }
        for (value, o) in out.iter_mut().zip(data) {
            *value = self.covariance(o.site(), b)?;
        }
        Ok(())
    }
}
