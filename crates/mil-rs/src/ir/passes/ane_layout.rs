//! ANE layout pass — reshape all tensors to `[1, C, 1, S]`.
//!
//! ANE requires this specific 4D layout for all tensors. This pass
//! transforms the IR so all tensor shapes conform to the ANE convention.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;

/// Reshape all tensors in the program to ANE's required `[1, C, 1, S]` layout.
pub struct AneLayoutPass;

impl Pass for AneLayoutPass {
    fn name(&self) -> &str {
        "ane-layout"
    }

    fn run(&self, _program: &mut Program) -> Result<()> {
        // TODO: Implement tensor layout transformation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_name() {
        assert_eq!(AneLayoutPass.name(), "ane-layout");
    }
}
