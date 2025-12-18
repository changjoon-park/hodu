//! Backend plugin types
//!
//! Types for backend plugins that execute models on various devices.

/// Target device for plugin execution
///
/// Using String for extensibility - plugins can define custom devices.
/// Convention: lowercase with `::` separator for device index.
/// Common values: "cpu", "cuda::0", "metal", "vulkan", "webgpu", "rocm::0"
pub type Device = String;

/// Parse device ID from device string (e.g., "cuda::0" -> 0)
pub fn parse_device_id(device: &str) -> Option<usize> {
    device.split("::").nth(1).and_then(|id| id.parse().ok())
}

/// Get device type from device string (e.g., "cuda::0" -> "cuda")
pub fn device_type(device: &str) -> &str {
    device.split("::").next().unwrap_or(device)
}

/// Build target specification for AOT compilation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuildTarget {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu", "aarch64-apple-darwin")
    pub triple: String,
    /// Target device (e.g., "cpu", "metal", "cuda::0")
    pub device: String,
}

impl BuildTarget {
    /// Create a new build target
    pub fn new(triple: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            triple: triple.into(),
            device: device.into(),
        }
    }

    /// Create a build target for the current host system
    pub fn host(device: impl Into<String>) -> Self {
        Self::new(current_host_triple(), device)
    }
}

/// Get the current host triple
pub fn current_host_triple() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return "x86_64-unknown-linux-gnu";
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    return "aarch64-unknown-linux-gnu";
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    return "x86_64-apple-darwin";
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return "aarch64-apple-darwin";
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    return "x86_64-pc-windows-msvc";
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "windows"),
    )))]
    return "unknown";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device_id() {
        assert_eq!(parse_device_id("cuda::0"), Some(0));
        assert_eq!(parse_device_id("cuda::1"), Some(1));
        assert_eq!(parse_device_id("rocm::2"), Some(2));
        assert_eq!(parse_device_id("cpu"), None);
        assert_eq!(parse_device_id("metal"), None);
        assert_eq!(parse_device_id("cuda::invalid"), None);
    }

    #[test]
    fn test_device_type() {
        assert_eq!(device_type("cuda::0"), "cuda");
        assert_eq!(device_type("cuda::1"), "cuda");
        assert_eq!(device_type("rocm::0"), "rocm");
        assert_eq!(device_type("cpu"), "cpu");
        assert_eq!(device_type("metal"), "metal");
        assert_eq!(device_type("webgpu"), "webgpu");
    }

    #[test]
    fn test_build_target_new() {
        let target = BuildTarget::new("x86_64-unknown-linux-gnu", "cuda::0");
        assert_eq!(target.triple, "x86_64-unknown-linux-gnu");
        assert_eq!(target.device, "cuda::0");
    }

    #[test]
    fn test_build_target_host() {
        let target = BuildTarget::host("cpu");
        assert_eq!(target.device, "cpu");
        assert!(!target.triple.is_empty());
    }

    #[test]
    fn test_current_host_triple() {
        let triple = current_host_triple();
        assert!(!triple.is_empty());
        // Should contain architecture and OS
        assert!(triple.contains('-'));
    }
}
