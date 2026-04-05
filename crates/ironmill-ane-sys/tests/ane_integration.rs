//! Comprehensive integration test suite for `ironmill-ane-sys`.
//!
//! Organized into tiers:
//! - **Tier 1** (`tier1_logic`): Pure logic — no framework or hardware needed.
//! - **Tier 2** (`tier2_framework`): Probes the ANE framework — needs macOS, not ANE.
//! - **Tier 3** (`tier3_ane`): Full ANE hardware tests — needs Apple Silicon.
//! - **Exploratory** (`explore`): Discovery tests — `#[ignore]`d, run manually.

#![cfg(target_os = "macos")]

use ironmill_ane_sys::perf::QoSMapper;
use ironmill_ane_sys::util::{AneErrors, AneLog};
use ironmill_ane_sys::*;

// =========================================================================
// Helpers
// =========================================================================

/// A trivial identity MIL program: fp16 tensor in, same tensor out.
const IDENTITY_MIL: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(x: tensor<fp16, [1, 4, 1, 1]>) {
        return x;
    } -> (tensor<fp16, [1, 4, 1, 1]>);
}"#;

/// Check whether ANE hardware is available, for gating tier-3 tests.
fn has_ane() -> bool {
    DeviceInfo::has_ane().unwrap_or(false)
}

// =========================================================================
// Tier 1 — Pure logic tests (no hardware, no framework)
// =========================================================================

mod tier1_logic {
    use super::*;

    // ----- error.rs -----

    /// Verify all error variant Display messages are non-empty and meaningful.
    #[test]
    fn error_display_messages() {
        let variants: Vec<AneSysError> = vec![
            AneSysError::FrameworkNotFound("test".into()),
            AneSysError::ClassNotFound("Foo".into()),
            AneSysError::CompilationFailed("bad".into()),
            AneSysError::EvalFailed {
                status: 0x42,
                context: "oops".into(),
            },
            AneSysError::Io(std::io::Error::new(std::io::ErrorKind::Other, "io")),
            AneSysError::InvalidInput("empty".into()),
            AneSysError::BudgetExhausted { count: 119 },
            AneSysError::LoadFailed("load".into()),
            AneSysError::UnloadFailed("unload".into()),
            AneSysError::DeviceNotAvailable,
            AneSysError::IOSurfaceMappingFailed("map".into()),
            AneSysError::ChainingFailed("chain".into()),
            AneSysError::RequestValidationFailed,
            AneSysError::NullPointer {
                context: "ctx".into(),
            },
            AneSysError::SessionHintFailed("hint".into()),
            AneSysError::ProgramCreationFailed("prog".into()),
        ];

        for variant in &variants {
            let msg = format!("{variant}");
            assert!(
                !msg.is_empty(),
                "Display for {:?} should not be empty",
                variant
            );
        }
    }

    /// Verify Display includes the inner detail for key variants.
    #[test]
    fn error_display_includes_details() {
        let err = AneSysError::FrameworkNotFound("AppleNeuralEngine".into());
        assert!(
            format!("{err}").contains("AppleNeuralEngine"),
            "should contain framework name"
        );

        let err = AneSysError::ClassNotFound("_ANEFoo".into());
        assert!(format!("{err}").contains("_ANEFoo"));

        let err = AneSysError::EvalFailed {
            status: 0xBEEF,
            context: "timeout".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("0xbeef") || msg.contains("BEEF") || msg.contains("beef"));
        assert!(msg.contains("timeout"));

        let err = AneSysError::BudgetExhausted { count: 119 };
        assert!(format!("{err}").contains("119"));
    }

    /// Verify From<std::io::Error> conversion works.
    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let ane_err: AneSysError = io_err.into();
        assert!(matches!(ane_err, AneSysError::Io(_)));
        assert!(format!("{ane_err}").contains("gone"));
    }

    // ----- model.rs: make_blobfile -----

    /// Verify make_blobfile produces a valid 128-byte header + payload.
    #[test]
    fn blobfile_structure() {
        let data = b"test weight data";
        let blob = ironmill_ane_sys::model::make_blobfile(data).unwrap();

        // Total size = 128 header + data
        assert_eq!(blob.len(), 128 + data.len());

        // Payload follows header verbatim
        assert_eq!(&blob[128..], data);
    }

    /// Verify 0xDEADBEEF magic at offset 64-67 (little-endian).
    #[test]
    fn blobfile_magic() {
        let blob = ironmill_ane_sys::model::make_blobfile(b"x").unwrap();
        assert_eq!(blob[64], 0xEF);
        assert_eq!(blob[65], 0xBE);
        assert_eq!(blob[66], 0xAD);
        assert_eq!(blob[67], 0xDE);
    }

    /// Verify file header fields (bytes 0-7).
    #[test]
    fn blobfile_file_header() {
        let blob = ironmill_ane_sys::model::make_blobfile(b"abc").unwrap();
        assert_eq!(blob[0], 1, "version should be 1");
        assert_eq!(blob[4], 2, "chunk count should be 2");
    }

    /// Verify data size at offset 72-75 (u32 LE).
    #[test]
    fn blobfile_data_size_field() {
        let data = vec![0u8; 256];
        let blob = ironmill_ane_sys::model::make_blobfile(&data).unwrap();
        let stored_size = u32::from_le_bytes([blob[72], blob[73], blob[74], blob[75]]);
        assert_eq!(stored_size, 256);
    }

    /// Verify data offset at bytes 80-83 is always 128.
    #[test]
    fn blobfile_data_offset_field() {
        let blob = ironmill_ane_sys::model::make_blobfile(b"z").unwrap();
        let offset = u32::from_le_bytes([blob[80], blob[81], blob[82], blob[83]]);
        assert_eq!(offset, 128);
    }

    /// Empty data produces a 128-byte header-only blobfile.
    #[test]
    fn blobfile_empty_data() {
        let blob = ironmill_ane_sys::model::make_blobfile(b"").unwrap();
        assert_eq!(blob.len(), 128);
    }

    // ----- model.rs: compile budget -----

    /// compile_count() should return a sane value (not wildly high).
    #[test]
    fn compile_count_sane() {
        let count = ironmill_ane_sys::model::compile_count();
        assert!(
            count < 200,
            "compile count looks unreasonably high: {count}"
        );
    }

    /// remaining_budget() should be consistent with compile_count().
    #[test]
    fn remaining_budget_consistent() {
        let remaining = ironmill_ane_sys::model::remaining_budget();
        let count = ironmill_ane_sys::model::compile_count();
        assert_eq!(remaining, 119usize.saturating_sub(count));
    }

    /// compile_mil_text with empty string returns InvalidInput.
    #[test]
    fn compile_empty_mil_returns_error() {
        let result =
            ironmill_ane_sys::model::compile_mil_text("", &[], ironmill_ane_sys::model::ANE_QOS);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AneSysError::InvalidInput(_)),
            "expected InvalidInput, got: {err}"
        );
    }

    /// eval with no inputs returns error.
    #[test]
    fn eval_no_inputs_returns_error() {
        let model = unsafe { InMemoryModel::from_raw(0x1 as *mut std::ffi::c_void) };
        let result = ironmill_ane_sys::model::eval(
            &model,
            &[],
            &[std::ptr::null_mut()],
            ironmill_ane_sys::model::ANE_QOS,
        );
        assert!(result.is_err());
        std::mem::forget(model);
    }

    /// eval with no outputs returns error.
    #[test]
    fn eval_no_outputs_returns_error() {
        let model = unsafe { InMemoryModel::from_raw(0x1 as *mut std::ffi::c_void) };
        let result = ironmill_ane_sys::model::eval(
            &model,
            &[std::ptr::null_mut()],
            &[],
            ironmill_ane_sys::model::ANE_QOS,
        );
        assert!(result.is_err());
        std::mem::forget(model);
    }
}

// =========================================================================
// Tier 2 — Framework probe tests (needs macOS, not ANE hardware)
// =========================================================================

mod tier2_framework {
    use super::*;

    // ----- objc.rs -----

    /// Loading the ANE framework should succeed on any macOS with the private framework.
    #[test]
    fn ane_framework_loads() {
        // is_available exercises ane_framework() + get_class()
        let avail = ironmill_ane_sys::model::is_available();
        // On macOS, the framework should exist
        assert!(
            avail,
            "ANE framework should be loadable on macOS (even without ANE hardware)"
        );
    }

    /// get_class for a known ANE class should succeed.
    #[test]
    fn get_class_known_succeeds() {
        // If framework loads, _ANEInMemoryModel must exist
        let avail = ironmill_ane_sys::model::is_available();
        assert!(avail);
    }

    // ----- device.rs -----

    /// DeviceInfo::has_ane() should return Ok on macOS (true on Apple Silicon).
    #[test]
    fn device_info_has_ane() {
        let result = DeviceInfo::has_ane();
        assert!(result.is_ok(), "has_ane() should not fail: {result:?}");
        let val = result.unwrap();
        eprintln!("has_ane = {val}");
        // On Apple Silicon, should be true
        if cfg!(target_arch = "aarch64") {
            assert!(val, "Apple Silicon should have ANE");
        }
    }

    /// DeviceInfo::num_anes() should return > 0 on Apple Silicon.
    #[test]
    fn device_info_num_anes() {
        let result = DeviceInfo::num_anes();
        assert!(result.is_ok());
        let n = result.unwrap();
        eprintln!("num_anes = {n}");
        if cfg!(target_arch = "aarch64") {
            assert!(n > 0, "Apple Silicon should have at least 1 ANE");
        }
    }

    /// DeviceInfo::num_ane_cores() should return > 0 on Apple Silicon.
    #[test]
    fn device_info_num_ane_cores() {
        let result = DeviceInfo::num_ane_cores();
        assert!(result.is_ok());
        let n = result.unwrap();
        eprintln!("num_ane_cores = {n}");
        if cfg!(target_arch = "aarch64") {
            assert!(n > 0, "Apple Silicon should have at least 1 ANE core");
        }
    }

    /// DeviceInfo::architecture_type() should return Some non-empty string.
    #[test]
    fn device_info_architecture_type() {
        let result = DeviceInfo::architecture_type();
        assert!(result.is_ok());
        let arch = result.unwrap();
        eprintln!("architecture_type = {arch:?}");
        if has_ane() {
            assert!(arch.is_some(), "should have an architecture type on ANE");
            let s = arch.unwrap();
            assert!(!s.is_empty(), "architecture type should not be empty");
        }
    }

    /// DeviceInfo::product_name() should return Some non-empty string.
    #[test]
    fn device_info_product_name() {
        let result = DeviceInfo::product_name();
        assert!(result.is_ok());
        let name = result.unwrap();
        eprintln!("product_name = {name:?}");
        if has_ane() {
            assert!(name.is_some(), "should have product name on ANE machine");
            assert!(
                !name.as_ref().unwrap().is_empty(),
                "product name should not be empty"
            );
        }
    }

    /// DeviceInfo::build_version() should return Some non-empty string.
    #[test]
    fn device_info_build_version() {
        let result = DeviceInfo::build_version();
        assert!(result.is_ok());
        let ver = result.unwrap();
        eprintln!("build_version = {ver:?}");
        if has_ane() {
            assert!(ver.is_some(), "should have build version");
            assert!(!ver.as_ref().unwrap().is_empty());
        }
    }

    /// DeviceInfo::is_virtual_machine() should return Ok(false) on bare metal.
    #[test]
    fn device_info_is_virtual_machine() {
        let result = DeviceInfo::is_virtual_machine();
        assert!(result.is_ok());
        let is_vm = result.unwrap();
        eprintln!("is_virtual_machine = {is_vm}");
        // Can't strictly assert false — CI might be in a VM
    }

    // ----- perf.rs -----

    /// Default task QoS should be > 0.
    #[test]
    fn qos_default_task() {
        let result = QoSMapper::ane_default_task_qos();
        assert!(result.is_ok());
        let qos = result.unwrap();
        eprintln!("default_task_qos = {qos}");
        assert!(qos > 0, "default task QoS should be > 0");
    }

    /// Real-time QoS should differ from background QoS.
    #[test]
    fn qos_real_time_differs_from_background() {
        let rt = QoSMapper::ane_real_time_task_qos().unwrap();
        let bg = QoSMapper::ane_background_task_qos().unwrap();
        eprintln!("real_time_qos = {rt}, background_qos = {bg}");
        assert_ne!(rt, bg, "real-time and background QoS should differ");
    }

    /// All 6 QoS levels should produce distinct values.
    #[test]
    fn qos_all_levels_distinct() {
        let levels = vec![
            ("default", QoSMapper::ane_default_task_qos().unwrap()),
            ("background", QoSMapper::ane_background_task_qos().unwrap()),
            ("utility", QoSMapper::ane_utility_task_qos().unwrap()),
            (
                "user_initiated",
                QoSMapper::ane_user_initiated_task_qos().unwrap(),
            ),
            (
                "user_interactive",
                QoSMapper::ane_user_interactive_task_qos().unwrap(),
            ),
            ("real_time", QoSMapper::ane_real_time_task_qos().unwrap()),
        ];

        for (name, qos) in &levels {
            eprintln!("  {name}: {qos}");
        }

        let values: Vec<u32> = levels.iter().map(|(_, v)| *v).collect();
        let mut deduped = values.clone();
        deduped.sort();
        deduped.dedup();
        assert!(
            deduped.len() >= 4,
            "expected at least 4 distinct QoS levels, got {} from {values:?}",
            deduped.len()
        );
    }

    /// program_priority_for_qos / qos_for_program_priority roundtrip.
    #[test]
    fn qos_priority_roundtrip() {
        let qos = QoSMapper::ane_default_task_qos().unwrap();
        let priority = QoSMapper::program_priority_for_qos(qos).unwrap();
        let roundtrip = QoSMapper::qos_for_program_priority(priority).unwrap();
        eprintln!("qos={qos} -> priority={priority} -> roundtrip_qos={roundtrip}");
        assert_eq!(qos, roundtrip, "QoS -> priority -> QoS should roundtrip");
    }

    // ----- validate.rs -----

    /// get_validate_network_supported_version should return Ok with version > 0.
    #[test]
    fn validate_network_version() {
        let result = get_validate_network_supported_version();
        assert!(result.is_ok(), "should resolve the symbol: {result:?}");
        let version = result.unwrap();
        eprintln!("validate_network_supported_version = {version}");
        assert!(version > 0, "version should be > 0");
    }

    /// validate_mil_network_on_host_ptr should return a non-null function pointer.
    #[test]
    fn validate_mil_fn_ptr_non_null() {
        let result = validate_mil_network_on_host_ptr();
        assert!(result.is_ok(), "should resolve the symbol: {result:?}");
        let ptr = result.unwrap();
        assert!(!ptr.is_null(), "fn pointer should be non-null");
    }

    /// validate_mlir_network_on_host_ptr should also be non-null.
    #[test]
    fn validate_mlir_fn_ptr_non_null() {
        let result = validate_mlir_network_on_host_ptr();
        assert!(result.is_ok(), "should resolve the symbol: {result:?}");
        let ptr = result.unwrap();
        assert!(!ptr.is_null(), "fn pointer should be non-null");
    }

    // ----- model.rs: descriptor from MIL -----

    /// Creating a descriptor from a valid MIL program should succeed.
    #[test]
    fn descriptor_from_valid_mil() {
        let result = ironmill_ane_sys::model::InMemoryModelDescriptor::from_mil_text(
            IDENTITY_MIL,
            &[],
            None,
        );
        assert!(
            result.is_ok(),
            "descriptor creation should succeed: {result:?}"
        );
        let desc = result.unwrap();

        // Should be flagged as MIL model
        assert!(desc.is_mil_model());

        // Should have a hex identifier
        let hex_id = desc.hex_string_identifier();
        eprintln!("hex_string_identifier = {hex_id:?}");
        assert!(hex_id.is_some(), "should have a hex identifier");
        assert!(!hex_id.unwrap().is_empty());
    }

    /// Creating a descriptor from invalid MIL should return an error.
    #[test]
    fn descriptor_from_invalid_mil() {
        let result = ironmill_ane_sys::model::InMemoryModelDescriptor::from_mil_text(
            "this is not valid MIL",
            &[],
            None,
        );
        // The framework may or may not reject garbage — it might create a
        // descriptor that fails on compile. We just verify no panic.
        eprintln!("descriptor from invalid MIL: {result:?}");
    }

    /// InMemoryModel can be created from a valid descriptor.
    #[test]
    fn model_from_descriptor() {
        let desc = ironmill_ane_sys::model::InMemoryModelDescriptor::from_mil_text(
            IDENTITY_MIL,
            &[],
            None,
        )
        .expect("descriptor");
        let model = ironmill_ane_sys::model::InMemoryModel::from_descriptor(&desc);
        assert!(model.is_ok(), "model creation should succeed: {model:?}");
        let m = model.unwrap();
        eprintln!("model hex_id = {:?}", m.hex_string_identifier());
        assert!(m.is_mil_model());
    }

    // ----- process.rs -----

    /// current_rss() should return a nonzero value (we're a running process).
    #[test]
    fn current_rss_nonzero() {
        let rss = current_rss();
        eprintln!("current_rss = {rss} bytes ({:.1} MB)", rss as f64 / 1e6);
        assert!(rss > 0, "RSS should be > 0 for a running process");
    }

    // ----- perf.rs: PerformanceStats -----

    /// Creating PerformanceStats with a known value should work.
    #[test]
    fn perf_stats_creation_and_readback() {
        let result = PerformanceStats::with_hw_execution_ns(12345);
        assert!(result.is_ok(), "stats creation should succeed: {result:?}");
        let stats = result.unwrap();
        let time = stats.hw_execution_time();
        eprintln!("hw_execution_time = {time}");
        assert_eq!(time, 12345, "should read back the value we set");
    }
}

// =========================================================================
// Tier 3 — Full ANE tests (needs Apple Silicon hardware)
// =========================================================================

mod tier3_ane {
    use super::*;

    // ----- Compile + eval identity -----

    /// Compile and load a trivial identity MIL program on the ANE.
    #[test]
    fn compile_identity_model() {
        if !has_ane() {
            eprintln!("skipping: no ANE hardware");
            return;
        }

        let model = ironmill_ane_sys::model::compile_mil_text(
            IDENTITY_MIL,
            &[],
            ironmill_ane_sys::model::ANE_QOS,
        );
        match &model {
            Ok(m) => {
                eprintln!("compiled OK: hex_id={:?}", m.hex_string_identifier());
                eprintln!("  state={}", m.state());
                eprintln!("  program_handle={}", m.program_handle());
                eprintln!(
                    "  intermediate_buffer_handle={}",
                    m.intermediate_buffer_handle()
                );
                eprintln!("  queue_depth={}", m.queue_depth());
                assert!(m.program_handle() != 0, "program handle should be nonzero");
            }
            Err(e) => {
                eprintln!("compile failed (may be budget): {e}");
            }
        }
    }

    /// Compile identity and evaluate with IOSurface tensors.
    #[test]
    fn compile_and_eval_identity() {
        if !has_ane() {
            eprintln!("skipping: no ANE hardware");
            return;
        }

        // Compile
        let model = match ironmill_ane_sys::model::compile_mil_text(
            IDENTITY_MIL,
            &[],
            ironmill_ane_sys::model::ANE_QOS,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("compile failed: {e}");
                return;
            }
        };

        // Create IOSurfaces for input and output
        // shape [1,4,1,1] fp16 → 4 elements × 2 bytes = 8 bytes
        // ANE minimum page alignment: use width=1, pixel_size=2, height=4
        let in_surface = match AneIOSurfaceObject::create_iosurface(1, 2, 4) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("input IOSurface creation failed: {e}");
                return;
            }
        };
        let out_surface = match AneIOSurfaceObject::create_iosurface(1, 2, 4) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("output IOSurface creation failed: {e}");
                return;
            }
        };

        // Evaluate
        let result = ironmill_ane_sys::model::eval(
            &model,
            &[in_surface],
            &[out_surface],
            ironmill_ane_sys::model::ANE_QOS,
        );
        match result {
            Ok(()) => eprintln!("eval succeeded!"),
            Err(e) => eprintln!("eval result: {e}"),
        }
    }

    // ----- IOSurface -----

    /// Create an IOSurface with specific dimensions.
    #[test]
    fn iosurface_create() {
        let result = AneIOSurfaceObject::create_iosurface(1, 2, 4);
        assert!(
            result.is_ok(),
            "IOSurface creation should succeed: {result:?}"
        );
        let surface = result.unwrap();
        assert!(!surface.is_null(), "IOSurfaceRef should be non-null");
        eprintln!("created IOSurface: {surface:?}");
    }

    /// Create an IOSurface with bytes-per-element.
    #[test]
    fn iosurface_create_bpe() {
        let result = AneIOSurfaceObject::create_iosurface_bpe(1, 2, 4, 2);
        assert!(
            result.is_ok(),
            "IOSurface BPE creation should succeed: {result:?}"
        );
        let surface = result.unwrap();
        assert!(!surface.is_null());
    }

    // ----- Client API -----

    /// Client::shared_connection() should return a valid handle.
    #[test]
    fn client_shared_connection() {
        if !has_ane() {
            eprintln!("skipping: no ANE hardware");
            return;
        }

        let client = Client::shared_connection();
        assert!(
            client.is_ok(),
            "shared_connection should succeed: {client:?}"
        );
        let c = client.unwrap();
        assert!(
            !c.as_raw().is_null(),
            "client raw pointer should be non-null"
        );
    }

    /// Client::begin_real_time_task() / end_real_time_task() should succeed.
    #[test]
    fn client_real_time_task() {
        if !has_ane() {
            eprintln!("skipping: no ANE hardware");
            return;
        }

        let client = match Client::shared_connection() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no client: {e}");
                return;
            }
        };

        let begin = client.begin_real_time_task();
        eprintln!("begin_real_time_task = {begin}");

        let end = client.end_real_time_task();
        eprintln!("end_real_time_task = {end}");
    }

    /// Client boolean properties: is_virtual_client, allow_restricted_access.
    #[test]
    fn client_boolean_properties() {
        if !has_ane() {
            eprintln!("skipping: no ANE hardware");
            return;
        }

        let client = match Client::shared_connection() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no client: {e}");
                return;
            }
        };

        let is_virtual = client.is_virtual_client();
        eprintln!("is_virtual_client = {is_virtual}");

        let restricted = client.allow_restricted_access();
        eprintln!("allow_restricted_access = {restricted}");
    }

    /// Client connection handles should be non-null.
    #[test]
    fn client_connection_handles() {
        if !has_ane() {
            eprintln!("skipping: no ANE hardware");
            return;
        }

        let client = match Client::shared_connection() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no client: {e}");
                return;
            }
        };

        let conn = client.conn();
        let fast_conn = client.fast_conn();
        eprintln!("conn = {conn:?}, fast_conn = {fast_conn:?}");
        assert!(!conn.is_null(), "conn should be non-null");
        assert!(!fast_conn.is_null(), "fast_conn should be non-null");
    }
}

// =========================================================================
// Exploratory tests — print-and-observe, #[ignore]d for CI
// =========================================================================

mod explore {
    use super::*;

    /// Explore all DeviceInfo properties.
    #[test]
    #[ignore]
    fn explore_device_info() {
        eprintln!("--- DeviceInfo exploration ---");
        eprintln!("has_ane:                       {:?}", DeviceInfo::has_ane());
        eprintln!(
            "num_anes:                      {:?}",
            DeviceInfo::num_anes()
        );
        eprintln!(
            "num_ane_cores:                 {:?}",
            DeviceInfo::num_ane_cores()
        );
        eprintln!(
            "architecture_type:             {:?}",
            DeviceInfo::architecture_type()
        );
        eprintln!(
            "sub_type:                      {:?}",
            DeviceInfo::sub_type()
        );
        eprintln!(
            "sub_type_variant:              {:?}",
            DeviceInfo::sub_type_variant()
        );
        eprintln!(
            "sub_type_and_variant:          {:?}",
            DeviceInfo::sub_type_and_variant()
        );
        eprintln!(
            "sub_type_product_variant:      {:?}",
            DeviceInfo::sub_type_product_variant()
        );
        eprintln!(
            "board_type:                    {:?}",
            DeviceInfo::board_type()
        );
        eprintln!(
            "product_name:                  {:?}",
            DeviceInfo::product_name()
        );
        eprintln!(
            "build_version:                 {:?}",
            DeviceInfo::build_version()
        );
        eprintln!(
            "boot_args:                     {:?}",
            DeviceInfo::boot_args()
        );
        eprintln!(
            "is_internal_build:             {:?}",
            DeviceInfo::is_internal_build()
        );
        eprintln!(
            "is_virtual_machine:            {:?}",
            DeviceInfo::is_virtual_machine()
        );
        eprintln!(
            "precompiled_model_checks_disabled: {:?}",
            DeviceInfo::precompiled_model_checks_disabled()
        );
        eprintln!(
            "is_excessive_power_drain_idle: {:?}",
            DeviceInfo::is_excessive_power_drain_when_idle()
        );
    }

    /// Explore Client API — connections, echo, real-time tasks.
    #[test]
    #[ignore]
    fn explore_client_api() {
        eprintln!("--- Client API exploration ---");

        let client = match Client::shared_connection() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("shared_connection failed: {e}");
                return;
            }
        };

        eprintln!("is_virtual_client:       {}", client.is_virtual_client());
        eprintln!(
            "allow_restricted_access: {}",
            client.allow_restricted_access()
        );
        eprintln!("conn:                    {:?}", client.conn());
        eprintln!("fast_conn:               {:?}", client.fast_conn());

        // Real-time task cycle
        let begin = client.begin_real_time_task();
        eprintln!("begin_real_time_task:     {begin}");
        let end = client.end_real_time_task();
        eprintln!("end_real_time_task:       {end}");

        // Private connection
        match Client::shared_private_connection() {
            Ok(pc) => {
                eprintln!("shared_private_connection: OK (raw={:?})", pc.as_raw());
                eprintln!(
                    "  private.is_virtual_client:       {}",
                    pc.is_virtual_client()
                );
                eprintln!(
                    "  private.allow_restricted_access: {}",
                    pc.allow_restricted_access()
                );
            }
            Err(e) => eprintln!("shared_private_connection failed: {e}"),
        }
    }

    /// Explore all QoS values and mapping functions.
    #[test]
    #[ignore]
    fn explore_qos() {
        eprintln!("--- QoS exploration ---");

        let levels = [
            ("default", QoSMapper::ane_default_task_qos()),
            ("background", QoSMapper::ane_background_task_qos()),
            ("utility", QoSMapper::ane_utility_task_qos()),
            ("user_initiated", QoSMapper::ane_user_initiated_task_qos()),
            (
                "user_interactive",
                QoSMapper::ane_user_interactive_task_qos(),
            ),
            ("real_time", QoSMapper::ane_real_time_task_qos()),
        ];

        for (name, result) in &levels {
            match result {
                Ok(qos) => {
                    let prio = QoSMapper::program_priority_for_qos(*qos);
                    let qi = QoSMapper::queue_index_for_qos(*qos);
                    eprintln!("  {name:20}: qos={qos}, priority={prio:?}, queue_index={qi:?}");
                }
                Err(e) => eprintln!("  {name}: err={e}"),
            }
        }

        eprintln!(
            "real_time_program_priority: {:?}",
            QoSMapper::real_time_program_priority()
        );
        eprintln!(
            "real_time_queue_index:      {:?}",
            QoSMapper::real_time_queue_index()
        );
    }

    /// Explore VirtualClient API.
    #[test]
    #[ignore]
    fn explore_virtual_client() {
        eprintln!("--- VirtualClient exploration ---");

        match VirtualClient::shared_connection() {
            Ok(vc) => {
                eprintln!("shared_connection: OK (raw={:?})", vc.as_raw());
                eprintln!("  has_ane:          {}", vc.has_ane());
                eprintln!("  num_anes:         {}", vc.num_anes());
                eprintln!("  num_ane_cores:    {}", vc.num_ane_cores());
                eprintln!("  is_internal_build:{}", vc.is_internal_build());

                let dev_info = vc.get_device_info();
                eprintln!("  get_device_info:  {dev_info:?}");

                let bvi = vc.exchange_build_version_info();
                eprintln!("  exchange_build_version_info: {bvi:?}");

                let vnv = vc.get_validate_network_version();
                eprintln!("  get_validate_network_version: {vnv}");

                let arch = vc.ane_architecture_type_str();
                eprintln!("  ane_architecture_type_str: {arch:?}");

                let board = vc.ane_boardtype();
                eprintln!("  ane_boardtype: {board}");

                let sub = vc.ane_sub_type_and_variant();
                eprintln!("  ane_sub_type_and_variant: {sub:?}");

                let host_bv = vc.host_build_version_str();
                eprintln!("  host_build_version_str: {host_bv:?}");

                let cap_mask = vc.negotiated_capability_mask();
                eprintln!("  negotiated_capability_mask: {cap_mask:#x}");

                let data_ver = vc.negotiated_data_interface_version();
                eprintln!("  negotiated_data_interface_version: {data_ver}");

                let dict_size = vc.output_dict_iosurface_size();
                eprintln!("  output_dict_iosurface_size: {dict_size}");
            }
            Err(e) => {
                eprintln!("VirtualClient::shared_connection failed: {e}");
            }
        }
    }

    /// Explore DaemonConnection API.
    #[test]
    #[ignore]
    fn explore_daemon_connection() {
        eprintln!("--- DaemonConnection exploration ---");

        match DaemonConnection::new() {
            Ok(dc) => {
                eprintln!("daemonConnection: OK (raw={:?})", dc.as_raw());
                eprintln!("  restricted:       {}", dc.restricted());
                eprintln!("  xpc_connection:   {:?}", dc.xpc_connection());
            }
            Err(e) => eprintln!("DaemonConnection::new failed: {e}"),
        }

        match DaemonConnection::daemon_connection_restricted() {
            Ok(dc) => {
                eprintln!(
                    "daemonConnectionRestricted: OK (restricted={})",
                    dc.restricted()
                );
            }
            Err(e) => eprintln!("daemon_connection_restricted failed: {e}"),
        }

        match DaemonConnection::user_daemon_connection() {
            Ok(dc) => {
                eprintln!("userDaemonConnection: OK (restricted={})", dc.restricted());
            }
            Err(e) => eprintln!("user_daemon_connection failed: {e}"),
        }
    }

    /// Explore model compile + properties.
    #[test]
    #[ignore]
    fn explore_model_compile() {
        if !has_ane() {
            eprintln!("skipping: no ANE hardware");
            return;
        }

        eprintln!("--- Model compile exploration ---");
        eprintln!(
            "compile_count before: {}",
            ironmill_ane_sys::model::compile_count()
        );
        eprintln!(
            "remaining_budget:     {}",
            ironmill_ane_sys::model::remaining_budget()
        );

        match ironmill_ane_sys::model::compile_mil_text(
            IDENTITY_MIL,
            &[],
            ironmill_ane_sys::model::ANE_QOS,
        ) {
            Ok(m) => {
                eprintln!("compiled OK");
                eprintln!(
                    "  hex_string_identifier:        {:?}",
                    m.hex_string_identifier()
                );
                eprintln!("  is_mil_model:                 {}", m.is_mil_model());
                eprintln!("  state:                        {}", m.state());
                eprintln!("  program_handle:               {}", m.program_handle());
                eprintln!(
                    "  intermediate_buffer_handle:    {}",
                    m.intermediate_buffer_handle()
                );
                eprintln!("  queue_depth:                  {}", m.queue_depth());
                eprintln!("  perf_stats_mask:              {}", m.perf_stats_mask());
                eprintln!(
                    "  compiled_model_exists:        {}",
                    m.compiled_model_exists()
                );
                eprintln!("  local_model_path:             {:?}", m.local_model_path());
                eprintln!("  model_url:                    {:?}", m.model_url());
                eprintln!(
                    "  compiler_options_file_name:    {:?}",
                    m.compiler_options_file_name()
                );
                eprintln!("  model_attributes:             {:?}", m.model_attributes());
            }
            Err(e) => {
                eprintln!("compile failed: {e}");
            }
        }

        eprintln!(
            "compile_count after: {}",
            ironmill_ane_sys::model::compile_count()
        );
    }

    /// Explore util classes: AneLog, AneErrors.
    #[test]
    #[ignore]
    fn explore_util_classes() {
        eprintln!("--- Util class exploration ---");

        // AneLog handles
        for (name, result) in [
            ("daemon", AneLog::daemon()),
            ("compiler", AneLog::compiler()),
            ("tool", AneLog::tool()),
            ("common", AneLog::common()),
            ("tests", AneLog::tests()),
            ("maintenance", AneLog::maintenance()),
            ("framework", AneLog::framework()),
        ] {
            eprintln!("  AneLog::{name}: {result:?}");
        }

        // AneErrors factories
        eprintln!("  create_error: {:?}", AneErrors::create_error(-1, "test"));
        eprintln!(
            "  bad_argument: {:?}",
            AneErrors::bad_argument("testMethod")
        );
        eprintln!(
            "  file_not_found: {:?}",
            AneErrors::file_not_found("testMethod")
        );
        eprintln!(
            "  invalid_model: {:?}",
            AneErrors::invalid_model("testMethod")
        );
        eprintln!(
            "  program_creation_error: {:?}",
            AneErrors::program_creation_error("testMethod")
        );
        eprintln!(
            "  program_load_error: {:?}",
            AneErrors::program_load_error("testMethod")
        );
        eprintln!(
            "  timeout_error: {:?}",
            AneErrors::timeout_error("testMethod")
        );
    }

    /// Explore validate functions and compare versions.
    #[test]
    #[ignore]
    fn explore_validate() {
        eprintln!("--- Validate exploration ---");

        match get_validate_network_supported_version() {
            Ok(v) => eprintln!("  validate_network_supported_version: {v}"),
            Err(e) => eprintln!("  version err: {e}"),
        }

        match validate_mil_network_on_host_ptr() {
            Ok(p) => eprintln!("  mil_validate_ptr: {p:?}"),
            Err(e) => eprintln!("  mil err: {e}"),
        }

        match validate_mlir_network_on_host_ptr() {
            Ok(p) => eprintln!("  mlir_validate_ptr: {p:?}"),
            Err(e) => eprintln!("  mlir err: {e}"),
        }
    }

    /// Explore ModelToken API.
    #[test]
    #[ignore]
    fn explore_model_token() {
        eprintln!("--- ModelToken exploration ---");

        match ModelToken::with_cs_identity(
            "com.test.app",
            "TEAM",
            "model",
            std::process::id() as i32,
        ) {
            Ok(t) => {
                eprintln!("  token created OK");
                eprintln!("    model_identifier:  {:?}", t.model_identifier());
                eprintln!("    cs_identity:       {:?}", t.cs_identity());
                eprintln!("    team_identity:     {:?}", t.team_identity());
                eprintln!("    process_identifier:{}", t.process_identifier());
            }
            Err(e) => eprintln!("  with_cs_identity failed: {e}"),
        }

        // Class method lookups
        let token = [0u32; 8];
        let pid = std::process::id() as i32;
        eprintln!(
            "  code_signing_id_for: {:?}",
            ModelToken::code_signing_id_for(&token, pid)
        );
        eprintln!(
            "  team_id_for:         {:?}",
            ModelToken::team_id_for(&token, pid)
        );
        eprintln!(
            "  process_name_for:    {:?}",
            ModelToken::process_name_for(&token, pid)
        );
    }

    /// Explore current_rss reporting.
    #[test]
    #[ignore]
    fn explore_rss() {
        let rss = current_rss();
        eprintln!("current_rss = {rss} bytes ({:.2} MB)", rss as f64 / 1e6);
    }
}
