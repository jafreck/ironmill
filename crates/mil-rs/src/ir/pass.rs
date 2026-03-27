use super::program::Program;

/// Trait for optimization passes that transform a MIL [`Program`].
///
/// Passes are applied sequentially to the program. Each pass can
/// modify operations, remove dead code, fuse ops, etc.
pub trait Pass {
    /// Human-readable name for this pass.
    fn name(&self) -> &str;

    /// Apply the pass to the program, modifying it in place.
    fn run(&self, program: &mut Program) -> crate::error::Result<()>;
}
