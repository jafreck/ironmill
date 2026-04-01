//! Safe wrappers for macOS process information APIs.

/// Returns the current process RSS (Resident Set Size) in bytes.
///
/// Uses `task_info(MACH_TASK_BASIC_INFO)` to query the Mach kernel.
/// Returns 0 on failure.
pub fn current_rss() -> u64 {
    use std::mem;

    use crate::objc;

    // MACH_TASK_BASIC_INFO = 20
    const MACH_TASK_BASIC_INFO: u32 = 20;

    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],   // time_value_t
        system_time: [u32; 2], // time_value_t
        policy: i32,
        suspend_count: i32,
    }

    // SAFETY: `mach_task_self_` is a valid Mach port for the current task.
    // `task_info` writes into `info` which is a properly aligned, zeroed
    // struct with `count` set to its size in u32 words. On success (kr == 0)
    // the struct is fully initialized by the kernel.
    let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
    let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
    let kr = unsafe {
        objc::task_info(
            objc::mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut u8,
            &mut count,
        )
    };
    if kr == 0 { info.resident_size } else { 0 }
}
