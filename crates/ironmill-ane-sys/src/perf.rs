//! ANE performance stats and QoS mapping wrappers.
//!
//! Wraps Apple's private `_ANEPerformanceStats`, `_ANEPerformanceStatsIOSurface`,
//! and `_ANEQoSMapper` classes from the `AppleNeuralEngine` framework.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{CFRelease, get_class, objc_msgSend, sel, sel_registerName};

// ---------------------------------------------------------------------------
// PerformanceStats — wraps _ANEPerformanceStats
// ---------------------------------------------------------------------------

/// ANE hardware performance statistics for a single evaluation.
///
/// Owns a retained `_ANEPerformanceStats` handle; released on drop.
pub struct PerformanceStats {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for PerformanceStats {}

impl PerformanceStats {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create stats from a hardware execution time in nanoseconds.
    ///
    /// Calls `+[_ANEPerformanceStats statsWithHardwareExecutionNS:]`.
    pub fn with_hw_execution_ns(ns: u64) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEPerformanceStats")?;
        type FactoryFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("statsWithHardwareExecutionNS:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, ns) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "statsWithHardwareExecutionNS: returned nil".into(),
            });
        }
        crate::objc::objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Hardware execution time in nanoseconds.
    pub fn hw_execution_time(&self) -> u64 {
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("hwExecutionTime")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Raw performance counter data (NSData).
    pub fn perf_counter_data(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("perfCounterData")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Performance counters dictionary (NSDictionary).
    pub fn performance_counters(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("performanceCounters")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Raw stats data (NSData).
    pub fn p_stats_raw_data(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("pStatsRawData")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }
}

impl Drop for PerformanceStats {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// PerformanceStatsIOSurface — wraps _ANEPerformanceStatsIOSurface
// ---------------------------------------------------------------------------

/// Performance stats tied to a specific IOSurface.
///
/// Owns a retained `_ANEPerformanceStatsIOSurface` handle; released on drop.
pub struct PerformanceStatsIOSurface {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for PerformanceStatsIOSurface {}

impl PerformanceStatsIOSurface {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create from an IOSurface object and stat type.
    ///
    /// Calls `+[_ANEPerformanceStatsIOSurface objectWithIOSurface:statType:]`.
    ///
    /// # Safety
    ///
    /// `surface` must be a valid `_ANEIOSurfaceObject` pointer.
    pub unsafe fn with_iosurface(
        surface: *mut c_void,
        stat_type: i64,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEPerformanceStatsIOSurface")?;
        type FactoryFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i64) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("objectWithIOSurface:statType:")) };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, surface, stat_type) };
        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "objectWithIOSurface:statType: returned nil".into(),
            });
        }
        crate::objc::objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// The underlying IOSurface stats object.
    pub fn stats(&self) -> *mut c_void {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("stats")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// The stat type identifier.
    pub fn stat_type(&self) -> i64 {
        type I64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let sel = unsafe { sel_registerName(sel!("statType")) };
        let f: I64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }
}

impl Drop for PerformanceStatsIOSurface {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// QoSMapper — all class methods, no instance state
// ---------------------------------------------------------------------------

/// Maps between ANE QoS levels, program priorities, and queue indices.
///
/// All methods are class methods on `_ANEQoSMapper` — no instance is needed.
pub struct QoSMapper;

impl QoSMapper {
    /// Default task QoS value.
    pub fn ane_default_task_qos() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("aneDefaultTaskQoS")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// Background task QoS value.
    pub fn ane_background_task_qos() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("aneBackgroundTaskQoS")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// Utility task QoS value.
    pub fn ane_utility_task_qos() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("aneUtilityTaskQoS")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// User-initiated task QoS value.
    pub fn ane_user_initiated_task_qos() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("aneUserInitiatedTaskQoS")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// User-interactive task QoS value.
    pub fn ane_user_interactive_task_qos() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("aneUserInteractiveTaskQoS")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// Real-time task QoS value.
    pub fn ane_real_time_task_qos() -> Result<u32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type U32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        let sel = unsafe { sel_registerName(sel!("aneRealTimeTaskQoS")) };
        let f: U32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// Map a QoS value to a program priority.
    pub fn program_priority_for_qos(qos: u32) -> Result<i32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type MapFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u32) -> i32;
        let sel = unsafe { sel_registerName(sel!("programPriorityForQoS:")) };
        let f: MapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel, qos) })
    }

    /// Map a program priority back to a QoS value.
    pub fn qos_for_program_priority(priority: i32) -> Result<u32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type MapFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> u32;
        let sel = unsafe { sel_registerName(sel!("qosForProgramPriority:")) };
        let f: MapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel, priority) })
    }

    /// Map a QoS value to a queue index.
    pub fn queue_index_for_qos(qos: u32) -> Result<u64, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type MapFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u32) -> u64;
        let sel = unsafe { sel_registerName(sel!("queueIndexForQoS:")) };
        let f: MapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel, qos) })
    }

    /// Program priority for real-time scheduling.
    pub fn real_time_program_priority() -> Result<i32, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type I32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
        let sel = unsafe { sel_registerName(sel!("realTimeProgramPriority")) };
        let f: I32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }

    /// Queue index for real-time scheduling.
    pub fn real_time_queue_index() -> Result<u64, AneSysError> {
        let cls = get_class("_ANEQoSMapper")?;
        type U64Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
        let sel = unsafe { sel_registerName(sel!("realTimeQueueIndex")) };
        let f: U64Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        Ok(unsafe { f(cls, sel) })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perf_stats_creation() {
        // May fail on machines without the framework; should not panic.
        let _ = PerformanceStats::with_hw_execution_ns(1000);
    }

    #[test]
    fn qos_mapper_default() {
        let _ = QoSMapper::ane_default_task_qos();
    }

    #[test]
    fn qos_mapper_real_time() {
        let _ = QoSMapper::real_time_program_priority();
    }
}
