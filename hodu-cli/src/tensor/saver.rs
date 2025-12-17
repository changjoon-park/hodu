//! Tensor saving utilities

use crate::utils::plugin_dtype_to_core;
use hodu_core::format::{hdt, json};
use hodu_core::tensor::Tensor;
use hodu_core::types::{Device as CoreDevice, Shape};
use hodu_plugin::TensorData;
use std::collections::HashMap;
use std::path::Path;

pub fn save_outputs(
    outputs: &HashMap<String, TensorData>,
    save_dir: &Path,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(save_dir)?;

    for (name, data) in outputs {
        let file_path = save_dir.join(format!("{}.{}", name, format.to_lowercase()));

        let dtype = plugin_dtype_to_core(data.dtype);
        let shape = Shape::new(&data.shape);
        let tensor = Tensor::from_bytes(&data.data, shape, dtype, CoreDevice::CPU)
            .map_err(|e| format!("Failed to create tensor: {}", e))?;

        match format.to_lowercase().as_str() {
            "hdt" => hdt::save(&tensor, &file_path)?,
            "json" => json::save(&tensor, &file_path)?,
            _ => return Err(format!("Unsupported save format: {}", format).into()),
        }
    }

    Ok(())
}
