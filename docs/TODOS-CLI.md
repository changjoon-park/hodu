# TODOS-CLI.md

## hodu-cli

**Code Deduplication:** (🔴 Critical)
- [x] Extract `path_to_str()` to shared utility - duplicated in run.rs, build.rs, inspect.rs, convert.rs
- [x] Remove duplicate `format_size()` in clean.rs:97 - use output.rs:188 instead
- [x] Extract `core_dtype_to_plugin()` to shared module - duplicated in run.rs, convert.rs, inspect.rs, loader.rs
- [x] Extract `plugin_dtype_to_core()` to shared module - duplicated in run.rs, convert.rs, saver.rs
- [x] Extract `load_tensor_data()` to tensor module - duplicated in run.rs, inspect.rs
- [x] Extract `save_tensor_data()` to tensor module - duplicated in run.rs, convert.rs

**API Consistency:** (🟡 Important)
- [x] Remove `current_target_triple()` in run.rs:469 - use `hodu_plugin::current_host_triple()` instead
- [x] Replace `&PathBuf` with `&Path` parameters (clean.rs, tensor/loader.rs)

**Code Quality:** (🟢 Nice-to-have)
- [x] Remove unnecessary `#[allow(dead_code)]` in plugin/install.rs:30 - used the `description` field instead
- [x] Refactor `.expect()` calls with safety comments to proper error handling (plugins/process.rs:72,93 kept with safety comments, commands/plugin.rs:520 refactored to `if let Some`)
- [x] Remove unused `_snapshot` variable in build.rs:122 - kept validation call without variable
