//! Common utility functions for hodu-cli

use hodu_core::types::DType;
use hodu_plugin::PluginDType;
use std::path::Path;

/// Convert a path to a string, returning an error if the path is not valid UTF-8
pub fn path_to_str(path: &Path) -> Result<&str, Box<dyn std::error::Error>> {
    path.to_str()
        .ok_or_else(|| format!("Invalid UTF-8 in path: {}", path.display()).into())
}

/// Convert hodu_core DType to hodu_plugin PluginDType
pub fn core_dtype_to_plugin(dtype: DType) -> PluginDType {
    match dtype {
        DType::BOOL => PluginDType::BOOL,
        DType::F8E4M3 => PluginDType::F8E4M3,
        DType::F8E5M2 => PluginDType::F8E5M2,
        DType::BF16 => PluginDType::BF16,
        DType::F16 => PluginDType::F16,
        DType::F32 => PluginDType::F32,
        DType::F64 => PluginDType::F64,
        DType::U8 => PluginDType::U8,
        DType::U16 => PluginDType::U16,
        DType::U32 => PluginDType::U32,
        DType::U64 => PluginDType::U64,
        DType::I8 => PluginDType::I8,
        DType::I16 => PluginDType::I16,
        DType::I32 => PluginDType::I32,
        DType::I64 => PluginDType::I64,
    }
}

/// Error for unknown DType conversion
#[derive(Debug)]
pub struct UnknownDTypeError(pub PluginDType);

impl std::fmt::Display for UnknownDTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unknown PluginDType variant: {:?}", self.0)
    }
}

impl std::error::Error for UnknownDTypeError {}

/// Convert hodu_plugin PluginDType to hodu_core DType
pub fn plugin_dtype_to_core(dtype: PluginDType) -> Result<DType, UnknownDTypeError> {
    match dtype {
        PluginDType::BOOL => Ok(DType::BOOL),
        PluginDType::F8E4M3 => Ok(DType::F8E4M3),
        PluginDType::F8E5M2 => Ok(DType::F8E5M2),
        PluginDType::BF16 => Ok(DType::BF16),
        PluginDType::F16 => Ok(DType::F16),
        PluginDType::F32 => Ok(DType::F32),
        PluginDType::F64 => Ok(DType::F64),
        PluginDType::U8 => Ok(DType::U8),
        PluginDType::U16 => Ok(DType::U16),
        PluginDType::U32 => Ok(DType::U32),
        PluginDType::U64 => Ok(DType::U64),
        PluginDType::I8 => Ok(DType::I8),
        PluginDType::I16 => Ok(DType::I16),
        PluginDType::I32 => Ok(DType::I32),
        PluginDType::I64 => Ok(DType::I64),
        _ => Err(UnknownDTypeError(dtype)),
    }
}
