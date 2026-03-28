//! ANE concat elimination pass.
//!
//! ANE does not support the `concat` op (constraint #1). This pass
//! replaces concat operations with multi-output program patterns.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;

/// Replace `concat` ops with multi-output program equivalents.
pub struct AneConcatEliminationPass;

impl Pass for AneConcatEliminationPass {
    fn name(&self) -> &str {
        "ane-concat-elimination"
    }

    fn run(&self, _program: &mut Program) -> Result<()> {
        // TODO: Implement concat elimination
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_name() {
        assert_eq!(AneConcatEliminationPass.name(), "ane-concat-elimination");
    }
}
