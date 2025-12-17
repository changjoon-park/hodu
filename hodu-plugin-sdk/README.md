# hodu-plugin-sdk

[![Crates.io](https://img.shields.io/crates/v/hodu-plugin-sdk.svg)](https://crates.io/crates/hodu-plugin-sdk)
[![Doc.rs](https://docs.rs/hodu-plugin-sdk/badge.svg)](https://docs.rs/hodu-plugin-sdk)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/daminstudio/hodu#license)

SDK for building Hodu ML toolkit plugins.

## Overview

Plugins communicate with the Hodu CLI via JSON-RPC 2.0 over stdio. Each plugin runs as a separate process, providing isolation and language-agnostic extensibility.

## Plugin Types

| Type | Description | Capabilities |
|------|-------------|--------------|
| `backend` | Execute/compile models on devices | `backend.run`, `backend.build` |
| `model_format` | Load/save model files | `format.load_model`, `format.save_model` |
| `tensor_format` | Load/save tensor files | `format.load_tensor`, `format.save_tensor` |

## Quick Start

### Create a Plugin Project

```bash
$ curl -fsSL https://raw.githubusercontent.com/daminstudio/hodu/main/hodu-plugin-sdk/new.sh | sh
```

### Backend Plugin Example

```rust
use hodu_plugin_sdk::server::PluginServer;
use hodu_plugin_sdk::rpc::{RunParams, RunResult, BuildParams, TensorOutput, RpcError};
use hodu_plugin_sdk::Context;

async fn handle_run(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
    for i in 0..100 {
        // Check for cancellation periodically
        if ctx.is_cancelled() {
            return Err(RpcError::cancelled());
        }

        ctx.progress(Some(i as u8), &format!("Processing step {}/100", i));
        // ... do work ...
    }

    Ok(RunResult {
        outputs: vec![
            TensorOutput {
                name: "output".to_string(),
                path: "/tmp/output.hdt".to_string(),
            }
        ],
    })
}

async fn handle_build(ctx: Context, params: BuildParams) -> Result<serde_json::Value, RpcError> {
    if ctx.is_cancelled() {
        return Err(RpcError::cancelled());
    }
    // Compile model...
    Ok(serde_json::json!({}))
}

#[tokio::main]
async fn main() {
    PluginServer::new("my-backend", env!("CARGO_PKG_VERSION"))
        .devices(vec!["cpu"])
        .method("backend.run", handle_run)
        .method("backend.build", handle_build)
        .run()
        .await
        .unwrap();
}
```

### Model Format Plugin Example

```rust
use hodu_plugin_sdk::server::PluginServer;
use hodu_plugin_sdk::rpc::{LoadModelParams, LoadModelResult, SaveModelParams, RpcError};
use hodu_plugin_sdk::Context;

async fn handle_load_model(_ctx: Context, params: LoadModelParams) -> Result<LoadModelResult, RpcError> {
    // Parse model file from params.path
    // Convert to Hodu snapshot format
    Ok(LoadModelResult {
        snapshot_path: "/tmp/model.hdss".to_string(),
    })
}

async fn handle_save_model(_ctx: Context, params: SaveModelParams) -> Result<serde_json::Value, RpcError> {
    // Load snapshot from params.snapshot_path
    // Convert to target format
    Ok(serde_json::json!({}))
}

#[tokio::main]
async fn main() {
    PluginServer::new("my-format", env!("CARGO_PKG_VERSION"))
        .model_extensions(vec!["myformat", "mf"])
        .method("format.load_model", handle_load_model)
        .method("format.save_model", handle_save_model)
        .run()
        .await
        .unwrap();
}
```

## API Reference

### PluginServer

```rust
PluginServer::new(name: &str, version: &str) -> Self
    .model_extensions(exts: Vec<&str>) -> Self   // File extensions (model format)
    .tensor_extensions(exts: Vec<&str>) -> Self  // File extensions (tensor format)
    .devices(devs: Vec<&str>) -> Self            // Supported devices (backend)
    .method(name: &str, handler: F) -> Self      // Register handler
    .run() -> Result<(), Error>                  // Start server
```

### Context

```rust
impl Context {
    fn is_cancelled(&self) -> bool       // Check if cancelled
    async fn cancelled(&self)            // Wait until cancelled (for select!)
    fn request_id(&self) -> &RequestId   // Get request ID
    fn progress(&self, percent: Option<u8>, message: &str)
    fn log_info(&self, message: &str)
    fn log_warn(&self, message: &str)
    fn log_error(&self, message: &str)
    fn log_debug(&self, message: &str)
}
```

### RpcError

```rust
RpcError::invalid_params(msg: &str) -> Self
RpcError::internal_error(msg: &str) -> Self
RpcError::not_supported(feature: &str) -> Self
RpcError::file_not_found(path: &str) -> Self
RpcError::cancelled() -> Self
```

## Cancellation

Handlers receive a `Context` for cancellation support:

```rust
async fn handle_run(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
    // Method 1: Periodic check
    for step in 0..1000 {
        if ctx.is_cancelled() {
            return Err(RpcError::cancelled());
        }
        // ... do work ...
    }

    // Method 2: Race with select!
    tokio::select! {
        result = do_long_operation() => { /* handle result */ }
        _ = ctx.cancelled() => { return Err(RpcError::cancelled()); }
    }

    Ok(result)
}
```

## JSON-RPC Protocol

### Lifecycle

```
CLI                          Plugin
 |                              |
 |-- initialize --------------->|
 |<-- {name, version, caps} ----|
 |                              |
 |-- method.call -------------->|
 |<-- $/progress (optional) ----|
 |<-- $/log (optional) ---------|
 |<-- result/error -------------|
 |                              |
 |-- $/cancel (optional) ------>|
 |                              |
 |-- shutdown ----------------->|
 |                         [exit]
```

### Methods

| Method | Description |
|--------|-------------|
| `initialize` | Initialize plugin |
| `shutdown` | Graceful shutdown |
| `format.load_model` | Load model file |
| `format.save_model` | Save model file |
| `format.load_tensor` | Load tensor file |
| `format.save_tensor` | Save tensor file |
| `backend.run` | Run inference |
| `backend.build` | AOT compile |
| `$/progress` | Progress notification |
| `$/log` | Log notification |
| `$/cancel` | Cancel request |

### Error Codes

| Code | Name |
|------|------|
| -32700 | Parse Error |
| -32600 | Invalid Request |
| -32601 | Method Not Found |
| -32602 | Invalid Params |
| -32603 | Internal Error |
| -32001 | Not Supported |
| -32002 | File Not Found |
| -32007 | Request Cancelled |

## License

BSD-3-Clause
