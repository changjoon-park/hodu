//! Tensor loading and saving utilities

mod loader;
mod saver;

pub use loader::{load_tensor_file, str_to_plugin_dtype};
pub use saver::save_outputs;

use crate::utils::{core_dtype_to_plugin, plugin_dtype_to_core};
use hodu_core::format::hdt;
use hodu_core::tensor::Tensor;
use hodu_core::types::{Device as CoreDevice, Shape};
use hodu_plugin::TensorData;
use std::path::Path;

/// Load tensor data from an HDT file without validation
pub fn load_tensor_data(path: impl AsRef<Path>) -> Result<TensorData, Box<dyn std::error::Error>> {
    let tensor = hdt::load(path).map_err(|e| format!("Failed to load HDT: {}", e))?;
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let dtype = core_dtype_to_plugin(tensor.dtype());
    let data = tensor
        .to_bytes()
        .map_err(|e| format!("Failed to get tensor bytes: {}", e))?;
    Ok(TensorData::new(data, shape, dtype))
}

/// Save tensor data to an HDT file
pub fn save_tensor_data(tensor_data: &TensorData, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(&tensor_data.shape);
    let dtype = plugin_dtype_to_core(tensor_data.dtype)?;
    let tensor = Tensor::from_bytes(&tensor_data.data, shape, dtype, CoreDevice::CPU).map_err(|e| e.to_string())?;
    hdt::save(&tensor, path).map_err(|e| e.to_string())?;
    Ok(())
}
