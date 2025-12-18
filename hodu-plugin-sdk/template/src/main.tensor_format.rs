//! {{NAME}} - Tensor format plugin for Hodu

use hodu_plugin_sdk::{
    rpc::{LoadTensorParams, LoadTensorResult, RpcError},
    server::PluginServer,
    Context, TensorData,
};
use std::path::Path;

#[hodu_plugin_sdk::main]
async fn main() {
    let server = PluginServer::new("{{NAME}}", env!("CARGO_PKG_VERSION"))
        .tensor_extensions(vec!["ext"]) // TODO: Change to your format extension
        .method("format.load_tensor", handle_load_tensor);

    if let Err(e) = server.run().await {
        eprintln!("Plugin error: {}", e);
        std::process::exit(1);
    }
}

async fn handle_load_tensor(_ctx: Context, params: LoadTensorParams) -> Result<LoadTensorResult, RpcError> {
    let path = Path::new(&params.path);

    if !path.exists() {
        return Err(RpcError::invalid_params(format!("File not found: {}", params.path)));
    }

    // TODO: Implement your tensor parsing logic here
    // 1. Parse the tensor file format
    // 2. Create TensorData with shape, dtype, and data
    // 3. Save as .hdt and return path

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    Err(RpcError::internal_error(format!(
        "Tensor format '{}' parsing not implemented. File: {} ({} bytes)",
        path.extension().and_then(|e| e.to_str()).unwrap_or("unknown"),
        params.path,
        file_size
    )))
}
