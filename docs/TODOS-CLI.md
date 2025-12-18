# TODOS-CLI.md

## hodu-cli

**Code Deduplication:** (🔴 Critical)
- [x] Extract `path_to_str()` to shared utility - duplicated in run.rs, build.rs, inspect.rs, convert.rs
- [x] Remove duplicate `format_size()` in clean.rs:97 - use output.rs:188 instead
- [x] Extract `core_dtype_to_plugin()` to shared module - duplicated in run.rs, convert.rs, inspect.rs, loader.rs
- [x] Extract `plugin_dtype_to_core()` to shared module - duplicated in run.rs, convert.rs, saver.rs
- [x] Extract `load_tensor_data()` to tensor module - duplicated in run.rs, inspect.rs
- [x] Extract `save_tensor_data()` to tensor module - duplicated in run.rs, convert.rs
- [x] Remove duplicate `load_tensor_data()` in run.rs:464, inspect.rs:504, convert.rs:204
- [x] Remove duplicate `save_tensor_data()` in run.rs:474, convert.rs:217
- [x] Remove duplicate `format_bytes()` in inspect.rs:470 - use output.rs instead

**API Consistency:** (🟡 Important)
- [x] Remove `current_target_triple()` in run.rs:469 - use `hodu_plugin::current_host_triple()` instead
- [x] Replace `&PathBuf` with `&Path` parameters (clean.rs, tensor/loader.rs)
- [ ] Standardize path parameter types - mix of `&Path`, `&str`, `impl AsRef<Path>` across convert.rs, run.rs, inspect.rs

**Code Quality:** (🟡 Important)
- [ ] Fix DType fallback to F32 in utils.rs:52 - should panic like SDK does for unknown dtypes
- [ ] Extract registry loading pattern - duplicated ~15 times across commands
- [ ] Extract plugin name prefix constants - magic strings like "hodu-backend-", "hodu-format-" scattered
- [ ] Fix temp file PID collision risk - use UUID or atomic counter instead of PID

**Code Quality:** (🟢 Nice-to-have)
- [x] Remove unnecessary `#[allow(dead_code)]` in plugin/install.rs:30 - used the `description` field instead
- [x] Refactor `.expect()` calls with safety comments to proper error handling (plugins/process.rs:72,93 kept with safety comments, commands/plugin.rs:520 refactored to `if let Some`)
- [x] Remove unused `_snapshot` variable in build.rs:122 - kept validation call without variable
- [ ] Handle cleanup failures instead of ignoring (convert.rs:192, install.rs:220)
- [ ] version.rs duplicates host triple detection logic - consider using hodu_plugin::current_host_triple()
