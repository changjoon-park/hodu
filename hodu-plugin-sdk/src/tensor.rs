//! Tensor data types for cross-plugin communication
//!
//! Re-exports from hodu_plugin with additional hodu_core integration.

// Re-export base types from hodu_plugin
pub use hodu_plugin::tensor::{PluginDType, TensorData};

// Keep SdkDType as alias for backwards compatibility
pub type SdkDType = PluginDType;

/// Convert hodu_core::DType to PluginDType
pub fn core_dtype_to_plugin(dtype: hodu_core::types::DType) -> PluginDType {
    use hodu_core::types::DType;
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

/// Convert PluginDType to hodu_core::DType
pub fn plugin_dtype_to_core(dtype: PluginDType) -> Result<hodu_core::types::DType, UnknownDTypeError> {
    use hodu_core::types::DType;
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

/// Extension trait for TensorData with hodu_core integration
pub trait TensorDataExt {
    /// Create new tensor data from hodu_core::DType
    fn from_core_dtype(data: Vec<u8>, shape: Vec<usize>, dtype: hodu_core::types::DType) -> TensorData;

    /// Get dtype as hodu_core::DType. Returns error for unknown dtype.
    fn core_dtype(&self) -> Result<hodu_core::types::DType, crate::PluginError>;

    /// Load tensor data from an HDT file
    fn load(path: impl AsRef<std::path::Path>) -> Result<TensorData, crate::PluginError>;

    /// Save tensor data to an HDT file
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), crate::PluginError>;
}

impl TensorDataExt for TensorData {
    fn from_core_dtype(data: Vec<u8>, shape: Vec<usize>, dtype: hodu_core::types::DType) -> TensorData {
        TensorData::new(data, shape, core_dtype_to_plugin(dtype))
    }

    fn core_dtype(&self) -> Result<hodu_core::types::DType, crate::PluginError> {
        plugin_dtype_to_core(self.dtype).map_err(|e| crate::PluginError::Load(e.to_string()))
    }

    fn load(path: impl AsRef<std::path::Path>) -> Result<TensorData, crate::PluginError> {
        use crate::hdt;
        let tensor = hdt::load(path).map_err(|e| crate::PluginError::Load(e.to_string()))?;
        let shape: Vec<usize> = tensor.shape().dims().to_vec();
        let dtype: PluginDType = core_dtype_to_plugin(tensor.dtype());
        let data = tensor.to_bytes().map_err(|e| crate::PluginError::Load(e.to_string()))?;
        Ok(TensorData::new(data, shape, dtype))
    }

    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), crate::PluginError> {
        use crate::{hdt, CoreDevice, Shape, Tensor};
        let shape = Shape::new(&self.shape);
        let dtype = self.core_dtype()?;
        let tensor = Tensor::from_bytes(&self.data, shape, dtype, CoreDevice::CPU)
            .map_err(|e| crate::PluginError::Save(e.to_string()))?;
        hdt::save(&tensor, path).map_err(|e| crate::PluginError::Save(e.to_string()))
    }
}
