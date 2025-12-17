# TODOS-SDK.md

## hodu-plugin-sdk

**Core Features:** (🔴 Critical)
- [x] Implement cancellation handling - process `$/cancel` requests and allow handlers to check cancellation status
- [x] Implement async handler support - `async fn` handlers with tokio runtime
- [x] Implement graceful shutdown - cleanup callback before exit (`on_shutdown`)

**Server Improvements:** (🟡 Important)
- [x] Implement context/state passing - shared state across handlers (e.g., config, connections)
- [x] Implement JSON-RPC batch requests - handle array of requests per spec
- [x] Implement per-handler timeout - configurable timeout with auto-cancellation
- [x] Implement middleware/hooks - pre/post request processing (logging, auth, etc.)

**Plugin Metadata:** (🟡 Important)
- [x] Add plugin metadata fields - description, author, homepage, license, repository
- [x] Add supported OS/arch declaration - target triple patterns
- [x] Add minimum hodu version requirement - semver compatibility

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
