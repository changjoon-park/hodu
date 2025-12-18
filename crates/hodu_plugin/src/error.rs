//! Plugin error types

use std::fmt;
use std::sync::Arc;

/// Plugin error type
///
/// Note: This enum is `#[non_exhaustive]` - new error variants may be added in future versions.
/// Always include a wildcard pattern when matching.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PluginError {
    /// Capability not supported (e.g., runner, builder, load_model)
    NotSupported(String),
    /// Invalid input or argument
    InvalidInput(String),
    /// I/O error with source preserved
    Io {
        message: String,
        source: Option<Arc<std::io::Error>>,
    },
    /// Execution error
    Execution(String),
    /// Internal error
    Internal(String),
    /// Load error (failed to load file)
    Load(String),
    /// Save error (failed to save file)
    Save(String),
}

impl PluginError {
    /// Create an I/O error with just a message
    pub fn io(message: impl Into<String>) -> Self {
        Self::Io {
            message: message.into(),
            source: None,
        }
    }

    /// Create an I/O error with source
    pub fn io_with_source(source: std::io::Error) -> Self {
        Self::Io {
            message: source.to_string(),
            source: Some(Arc::new(source)),
        }
    }
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported(msg) => write!(f, "not supported: {}", msg),
            Self::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            Self::Io { message, .. } => write!(f, "io error: {}", message),
            Self::Execution(msg) => write!(f, "execution error: {}", msg),
            Self::Internal(msg) => write!(f, "internal error: {}", msg),
            Self::Load(msg) => write!(f, "load error: {}", msg),
            Self::Save(msg) => write!(f, "save error: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source: Some(e), .. } => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PluginError {
    fn from(e: std::io::Error) -> Self {
        PluginError::io_with_source(e)
    }
}

/// Result type for plugin operations
pub type PluginResult<T> = Result<T, PluginError>;
