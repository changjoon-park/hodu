# TODOS-PLUGIN.md

## hodu_plugin

**Error Handling:** (🔴 Critical)
- [x] Replace `.unwrap()` with proper error handling in rpc.rs:343,351 (Notification::progress, Notification::log)

**API Consistency:** (🟡 Important)
- [x] Add `BACKEND_SUPPORTED_DEVICES` to capabilities module in rpc.rs (exists in methods but missing in capabilities)
- [x] Standardize RpcError factory methods - all now use `impl Into<String>`
- [ ] Make PluginDType FromStr error implement std::error::Error (tensor.rs:115-137)

**Validation:** (🟡 Important)
- [x] Add `TensorData::new_checked()` constructor with validation in tensor.rs
- [x] Consider Response struct validation (must have either result OR error per JSON-RPC spec) - added `is_valid()`, `is_success()`, `is_error()`

**Error Chain:** (🟡 Important)
- [ ] Preserve error chain in `From<io::Error>` for PluginError (error.rs:44-47)

**Documentation:** (🟢 Nice-to-have)
- [ ] Add rustdoc for public struct fields in rpc.rs (40+ undocumented fields)
- [ ] Add rustdoc for public functions in rpc.rs (Request::new, Response::success/error, Notification::*, RpcError::*)

**Dead Code:** (🟡 Important)
- [ ] Remove or use `capabilities` module - currently unused
- [ ] Remove unused error codes: UNSUPPORTED_FORMAT, COMPILATION_ERROR, RUNTIME_ERROR, RESOURCE_EXHAUSTED

**Consistency:** (🟡 Important)
- [ ] Standardize RpcError factory message format - some use "X: {}" others use "X not found: {}"

**Testing:** (🟢 Nice-to-have)
- [ ] Add unit tests for PluginDType FromStr parsing
- [ ] Add unit tests for TensorData validation (is_valid, is_scalar)
- [ ] Add unit tests for JSON-RPC serialization/deserialization
- [ ] Add unit tests for device parsing logic
