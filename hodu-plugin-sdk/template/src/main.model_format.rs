//! {{NAME}} - Model format plugin for Hodu

use hodu_plugin_sdk::{
    rpc::{LoadModelParams, LoadModelResult, RpcError},
    server::PluginServer,
    Context,
};
use std::path::Path;

#[hodu_plugin_sdk::main]
async fn main() {
    let server = PluginServer::new("{{NAME}}", env!("CARGO_PKG_VERSION"))
        .model_extensions(vec!["ext"]) // TODO: Change to your format extension
        .method("format.load_model", handle_load_model);

    if let Err(e) = server.run().await {
        eprintln!("Plugin error: {}", e);
        std::process::exit(1);
    }
}

async fn handle_load_model(_ctx: Context, params: LoadModelParams) -> Result<LoadModelResult, RpcError> {
    let path = Path::new(&params.path);

    if !path.exists() {
        return Err(RpcError::invalid_params(format!("File not found: {}", params.path)));
    }

    // TODO: Implement your model parsing logic here
    // 1. Parse the model file
    // 2. Convert to Hodu Snapshot format
    // 3. Save snapshot and return path

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    Err(RpcError::internal_error(format!(
        "Model format '{}' parsing not implemented. File: {} ({} bytes)",
        path.extension().and_then(|e| e.to_str()).unwrap_or("unknown"),
        params.path,
        file_size
    )))
}
