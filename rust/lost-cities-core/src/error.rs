use std::error::Error;
use std::fmt::{self, Display, Formatter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineErrorKind {
    AlreadyExists,
    NotFound,
    FailedPrecondition,
    InvalidArgument,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineError {
    kind: EngineErrorKind,
    message: String,
}

impl EngineError {
    pub fn new(kind: EngineErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> EngineErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn already_exists(message: impl Into<String>) -> Self {
        Self::new(EngineErrorKind::AlreadyExists, message)
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self::new(EngineErrorKind::NotFound, message)
    }

    pub(crate) fn failed_precondition(message: impl Into<String>) -> Self {
        Self::new(EngineErrorKind::FailedPrecondition, message)
    }

    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(EngineErrorKind::InvalidArgument, message)
    }
}

impl Display for EngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl Error for EngineError {}
