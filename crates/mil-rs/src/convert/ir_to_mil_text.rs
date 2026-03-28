//! Convert ironmill MIL IR to ANE MIL text format.
//!
//! This emitter produces the text-based MIL format consumed by `_ANECompiler`,
//! as reverse-engineered by Orion and maderix/ANE.
//!
//! Key differences from the CoreML protobuf emitter (`ir_to_proto`):
//! - Text format (UTF-8) instead of binary protobuf
//! - Tensor layout must be `[1, C, 1, S]`
//! - Weights use BLOBFILE with byte offsets
//! - I/O variables must be alphabetically ordered
//! - Const bools must be named const refs, not inline literals

// TODO: Implement program_to_mil_text — see spec Task 1
