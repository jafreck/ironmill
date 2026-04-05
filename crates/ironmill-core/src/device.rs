//! Device selection for model execution (§10.1).

/// Specifies which compute device to use for inference.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Automatically select the best available device.
    #[default]
    Auto,
    /// Use Metal GPU acceleration (macOS).
    Metal,
    /// Use the Apple Neural Engine.
    Ane,
    /// Use Core ML for inference.
    CoreMl,
    /// Use CPU-only inference.
    Cpu,
}

impl Device {
    /// Returns the devices available on the current platform.
    pub fn available() -> Vec<Device> {
        let mut devices = vec![Device::Auto];
        #[cfg(target_os = "macos")]
        {
            devices.extend_from_slice(&[Device::Metal, Device::Ane, Device::CoreMl]);
        }
        devices.push(Device::Cpu);
        devices
    }
}
