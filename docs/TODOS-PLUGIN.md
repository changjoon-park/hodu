# TODOS-PLUGIN.md

## hodu_plugin

**Error Handling:** (🔴 Critical)
- [x] Replace `.unwrap()` with proper error handling in rpc.rs:343,351 (Notification::progress, Notification::log)

**API Consistency:** (🟡 Important)
- [x] Add `BACKEND_SUPPORTED_DEVICES` to capabilities module in rpc.rs (exists in methods but missing in capabilities)

**Validation:** (🟡 Important)
- [x] Add `TensorData::new_checked()` constructor with validation in tensor.rs
- [x] Consider Response struct validation (must have either result OR error per JSON-RPC spec) - added `is_valid()`, `is_success()`, `is_error()`

**Documentation:** (🟢 Nice-to-have)
- [ ] Add rustdoc for public functions in rpc.rs (Request::new, Response::success/error, Notification::*, RpcError::*)

**Testing:** (🟢 Nice-to-have)
- [ ] Add unit tests for PluginDType FromStr parsing
- [ ] Add unit tests for TensorData validation (is_valid, is_scalar)
- [ ] Add unit tests for JSON-RPC serialization/deserialization
- [ ] Add unit tests for device parsing logic
