//! Device selection for model execution (§10.1).

/// Specifies which compute device to use for inference.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Auto,
    Metal,
    Ane,
    CoreMl,
    Cuda,
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
