# TODOS-SDK.md

## hodu-plugin-sdk

**Core Features:** (🔴 Critical)
- [x] Implement cancellation handling - process `$/cancel` requests and allow handlers to check cancellation status
- [x] Implement async handler support - `async fn` handlers with tokio runtime
- [x] Implement graceful shutdown - cleanup callback before exit (`on_shutdown`)

**Validation:** (🔴 Critical)
- [x] Add percent validation (0-100) in notify_progress - values > 100 are now clamped
- [x] Add log level validation in notify_log - invalid levels default to "info"
- [x] Handle notification failures instead of silent discard - now logs to stderr on failure

**Server Improvements:** (🟡 Important)
- [x] Implement context/state passing - shared state across handlers (e.g., config, connections)
- [x] Implement JSON-RPC batch requests - handle array of requests per spec
- [x] Implement per-handler timeout - configurable timeout with auto-cancellation
- [x] Implement middleware/hooks - pre/post request processing (logging, auth, etc.)

**Code Deduplication:** (🟡 Important)
- [ ] Extract param deserialization helper - duplicated in server.rs:502, 545, 826
- [ ] Extract handler registration logic - duplicated 3 times in server.rs
- [ ] Reduce excessive cloning in request processing - consider borrowing where possible

**Plugin Metadata:** (🟡 Important)
- [x] Add plugin metadata fields - description, author, homepage, license, repository
- [x] Add supported OS/arch declaration - target triple patterns
- [x] Add minimum hodu version requirement - semver compatibility

**Code Quality:** (🟡 Important)
- [x] Fix DType fallback masking in tensor.rs:52 - now panics with descriptive message for unknown dtypes
- [ ] Remove dead code `with_state()` in context.rs:54 - unused method

**Developer Experience:** (🟢 Nice-to-have)
- [x] Implement health check endpoint - `$/ping` method for liveness probe
- [ ] Implement automatic profiling - execution time measurement per handler
- [ ] Implement streaming response - chunked output for large data
- [ ] Implement hot reload support - file watcher for development
- [ ] Add `#[derive(PluginMethod)]` macro - reduce boilerplate

**Error Handling:** (🟢 Nice-to-have)
- [ ] Implement error chain/cause - nested error information
- [ ] Implement error recovery hints - suggested actions for common errors
- [ ] Add structured error data - additional fields beyond code/message

**Testing:** (🟢 Nice-to-have)
- [ ] Add mock server for testing - simulate CLI requests
- [ ] Add integration test harness - end-to-end plugin testing
- [ ] Add request/response logging mode - debug flag for development
