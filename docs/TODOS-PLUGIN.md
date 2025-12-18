# TODOS-PLUGIN.md

## hodu_plugin

**Error Handling:** (🔴 Critical)
- [x] Replace `.unwrap()` with proper error handling in rpc.rs:343,351 (Notification::progress, Notification::log)

**API Consistency:** (🟡 Important)
- [x] Add `BACKEND_SUPPORTED_DEVICES` to capabilities module in rpc.rs (exists in methods but missing in capabilities)
- [x] Standardize RpcError factory methods - all now use `impl Into<String>`
- [x] Make PluginDType FromStr error implement std::error::Error - added ParseDTypeError type

**Validation:** (🟡 Important)
- [x] Add `TensorData::new_checked()` constructor with validation in tensor.rs
- [x] Consider Response struct validation (must have either result OR error per JSON-RPC spec) - added `is_valid()`, `is_success()`, `is_error()`

**Error Chain:** (🟡 Important)
- [x] Preserve error chain in `From<io::Error>` for PluginError - now uses Arc to preserve source

**Dead Code:** (🟡 Important)
- [x] Remove or use `capabilities` module - removed unused module

**Testing:** (🟢 Nice-to-have)
- [x] Add unit tests for PluginDType FromStr parsing
- [x] Add unit tests for TensorData validation (is_valid, is_scalar)
- [x] Add unit tests for JSON-RPC serialization/deserialization
- [x] Add unit tests for device parsing logic - tests in backend.rs

**Documentation:** (🟢 Nice-to-have)
- [x] Add rustdoc for public struct fields in rpc.rs - all params/result structs documented
- [x] Add rustdoc for public functions in rpc.rs - Request::new, Response::*, Notification::*, RpcError::* documented

---

## Newly Discovered Issues (2nd Analysis)

**Arithmetic Safety:** (🔴 Critical)
- [ ] Prevent shape product overflow - `tensor.rs:211,224` `shape.iter().product()` needs checked_mul

**Validation:** (🔴 Critical)
- [ ] Remove expect() calls - `rpc.rs:483,500` Notification serialization should return Result instead
- [ ] Validate percent range - `rpc.rs:361` u8 type but should only allow 0-100
- [ ] Validate BuildTarget - `backend.rs:33-38` allows empty strings/invalid triples
- [ ] Validate zero dimension shapes - `tensor.rs:202` allows invalid shapes like `[0, 5]`

**API Design:** (🟡 Important)
- [ ] Strengthen Response state validation - `rpc.rs:451` allows both result and error set (JSON-RPC violation)
- [ ] Validate Request method - `rpc.rs:24` allows empty strings/special characters

**Parsing:** (🟢 Nice-to-have)
- [ ] Strengthen device ID parsing - `backend.rs:13` allows malformed input like `cuda::0::extra`
- [ ] Remove unnecessary unwrap_or - `backend.rs:19` `split().next()` never returns None

**Documentation:** (🟢 Nice-to-have)
- [ ] Document parse_device_id - `backend.rs:13` no doc comment
- [ ] Document device_type - `backend.rs:18` no doc comment
- [ ] Document BuildTarget fields - `backend.rs:24` no doc comments on fields
