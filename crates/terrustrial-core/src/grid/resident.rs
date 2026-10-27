//! Validated local grid data shared by pattern and density engines.
use crate::{
    error::{checked_len, invalid},
    execution::{MemoryBudget, Reservation},
    Error, Result,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub(crate) dimensions: [usize; 3],
    pub(crate) cells: usize,
}
impl Shape {
    pub fn new(dimensions: [usize; 3]) -> Result<Self> {
        if dimensions.contains(&0) || dimensions.iter().any(|d| *d > i64::MAX as usize) {
            return Err(invalid("positive representable grid dimensions required"));
        }
        Ok(Self {
            dimensions,
            cells: checked_len(&dimensions)?,
        })
    }
    pub fn cells(self) -> usize {
        self.cells
    }
    pub fn dimensions(self) -> [usize; 3] {
        self.dimensions
    }
    pub fn coordinates(self, index: usize) -> Result<[usize; 3]> {
        if index >= self.cells {
            return Err(invalid("grid index"));
        }
        Ok([
            index % self.dimensions[0],
            (index / self.dimensions[0]) % self.dimensions[1],
            index / (self.dimensions[0] * self.dimensions[1]),
        ])
    }
    pub fn index(self, p: [usize; 3]) -> Result<usize> {
        if (0..3).any(|i| p[i] >= self.dimensions[i]) {
            return Err(invalid("grid coordinate"));
        }
        Ok(p[0] + self.dimensions[0] * (p[1] + self.dimensions[1] * p[2]))
    }
    pub(crate) fn offset(
        self,
        index: usize,
        offset: [i64; 3],
        boundary: Boundary,
    ) -> Option<usize> {
        let p = self.coordinates(index).ok()?;
        let mut q = [0; 3];
        for i in 0..3 {
            let value = p[i] as i128 + offset[i] as i128;
            let dimension = self.dimensions[i] as i128;
            q[i] = match boundary {
                Boundary::Periodic => value.rem_euclid(dimension) as usize,
                Boundary::Bounded => {
                    if value < 0 || value >= dimension {
                        return None;
                    }
                    value as usize
                }
            };
        }
        self.index(q).ok()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    Bounded,
    Periodic,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Variable {
    Continuous { scale: f64 },
    Categorical { classes: u32 },
}
impl Variable {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Continuous { scale }
                if !scale.is_finite() || *scale <= 0.0 || !(1.0 / scale).is_finite() =>
            {
                Err(invalid("continuous mismatch scale"))
            }
            Self::Categorical { classes: 0 } => Err(invalid("positive category count required")),
            _ => Ok(()),
        }
    }
    pub(crate) fn validate_value(&self, value: f64) -> Result<()> {
        if !value.is_finite()
            || matches!(self,Self::Categorical{classes} if value<0.0 || value>=*classes as f64 || value.fract()!=0.0)
        {
            return Err(invalid("training/conditioning variable value"));
        }
        Ok(())
    }
}
/// Cell-major values, with missing attributes distinct from every valid category.
/// One candidate supplies all requested attributes jointly.
pub struct Grid {
    pub(crate) shape: Shape,
    pub(crate) variables: Vec<Variable>,
    pub(crate) values: Vec<Option<f64>>,
    _reservation: Reservation,
}
impl Grid {
    pub fn new(
        shape: Shape,
        variables: Vec<Variable>,
        values: Vec<Option<f64>>,
        budget: &MemoryBudget,
    ) -> Result<Self> {
        if variables.is_empty() || values.len() != checked_len(&[shape.cells, variables.len()])? {
            return Err(invalid("grid variable/value dimensions"));
        }
        let bytes = checked_len(&[values.len(), std::mem::size_of::<Option<f64>>()])?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        for variable in &variables {
            variable.validate()?;
        }
        for (i, value) in values.iter().enumerate() {
            if let Some(value) = value {
                variables[i % variables.len()].validate_value(*value)?;
            }
        }
        Ok(Self {
            shape,
            variables,
            values,
            _reservation: reservation,
        })
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }
    pub fn values(&self) -> &[Option<f64>] {
        &self.values
    }
    pub fn get(&self, cell: usize, variable: usize) -> Result<Option<f64>> {
        if cell >= self.shape.cells || variable >= self.variables.len() {
            return Err(invalid("grid cell or variable"));
        }
        Ok(self.values[cell * self.variables.len() + variable])
    }
}
#[derive(Clone, Debug)]
pub struct Event {
    pub offset: [i64; 3],
    pub variable: usize,
    pub value: f64,
    pub weight: f64,
}
