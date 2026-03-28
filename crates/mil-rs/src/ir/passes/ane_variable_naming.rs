//! ANE variable naming pass.
//!
//! ANE requires I/O IOSurfaces to be ordered alphabetically by their
//! MIL variable names (constraints #3, #13). This pass renames I/O
//! variables so alphabetical order matches the intended logical order.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;

/// Rename I/O variables for alphabetical ordering compatibility with ANE.
///
/// Uses naming scheme: `a_input0`, `a_input1` for inputs and
/// `z_output0`, `z_output1` for outputs.
pub struct AneVariableNamingPass;

impl Pass for AneVariableNamingPass {
    fn name(&self) -> &str {
        "ane-variable-naming"
    }

    fn run(&self, _program: &mut Program) -> Result<()> {
        // TODO: Implement variable renaming
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_name() {
        assert_eq!(AneVariableNamingPass.name(), "ane-variable-naming");
    }
}
