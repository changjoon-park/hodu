//! Tensor saving utilities

use crate::utils::plugin_dtype_to_core;
use hodu_core::format::{hdt, json};
use hodu_core::tensor::Tensor;
use hodu_core::types::{Device as CoreDevice, Shape};
use hodu_plugin::TensorData;
use std::collections::HashMap;
use std::path::Path;

/// Validate tensor output name to prevent path traversal attacks
fn validate_output_name(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if name.is_empty() {
        return Err("Output tensor name cannot be empty".into());
    }
    // Check for path separators and traversal sequences
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!(
            "Output tensor name '{}' contains invalid characters (path separators not allowed)",
            name
        )
        .into());
    }
    // Check for null bytes
    if name.contains('\0') {
        return Err("Output tensor name contains null byte".into());
    }
    Ok(())
}

pub fn save_outputs(
    outputs: &HashMap<String, TensorData>,
    save_dir: &Path,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(save_dir)?;

    for (name, data) in outputs {
        // Validate name to prevent path traversal
        validate_output_name(name)?;

        let file_path = save_dir.join(format!("{}.{}", name, format.to_lowercase()));

        let dtype = plugin_dtype_to_core(data.dtype)?;
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
