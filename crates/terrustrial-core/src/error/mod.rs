use std::fmt;

#[derive(Debug)]
pub enum Error {
    InvalidInput(String),
    SizeOverflow,
    MemoryBudget {
        requested: u64,
        available: u64,
    },
    DiskBudget {
        requested: u64,
        available: u64,
    },
    Singular {
        context: &'static str,
    },
    Numerical {
        context: &'static str,
    },
    Convergence {
        context: &'static str,
        iterations: usize,
        residual: f64,
    },
    InsufficientNeighborhood,
    Cancelled,
    /// A cache-only preparation encountered a disk page. The execution driver
    /// retries that preparation on its bounded blocking service. Ordinary
    /// synchronous queries do not return this scheduling signal.
    WouldBlock,
    BackendUnavailable(String),
    Backend(String),
    Io(std::io::Error),
    #[cfg(feature = "sqlite")]
    Storage(rusqlite::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::SizeOverflow => write!(f, "dimension or byte count overflow"),
            Self::MemoryBudget {
                requested,
                available,
            } => write!(
                f,
                "requires {requested} bytes; {available} bytes remain in the budget"
            ),
            Self::DiskBudget { requested, available } => write!(f,
                "requires {requested} additional disk bytes; {available} bytes remain in the disk budget"),
            Self::Singular { context } => {
                write!(f, "singular or non-positive-definite system: {context}")
            }
            Self::Numerical { context } => write!(f, "numerical failure: {context}"),
            Self::Convergence { context, iterations, residual } => write!(f,"{context} did not converge after {iterations} iterations; relative residual {residual}"),
            Self::InsufficientNeighborhood => write!(f, "no neighborhood satisfies the configured minimum and maximum quotas"),
            Self::Cancelled => write!(f, "operation cancelled"),
            Self::WouldBlock => write!(f, "preparation requires a disk page"),
            Self::BackendUnavailable(message) => write!(f, "backend unavailable: {message}"),
            Self::Backend(message) => write!(f, "backend failure: {message}"),
            Self::Io(error) => error.fmt(f),
            #[cfg(feature = "sqlite")]
            Self::Storage(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            #[cfg(feature = "sqlite")]
            Self::Storage(e) => Some(e),
            _ => None,
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
#[cfg(feature = "sqlite")]
impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Self::Storage(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

pub(crate) fn checked_len(dimensions: &[usize]) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(1_usize, |a, b| a.checked_mul(*b).ok_or(Error::SizeOverflow))
}
