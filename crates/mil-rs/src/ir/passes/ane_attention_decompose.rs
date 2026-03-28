//! Attention decomposition pass for ANE.
//!
//! ANE's SDPA op ignores causal masks (constraint #6), so we must
//! decompose `scaled_dot_product_attention` into explicit
//! matmul → mask → softmax → matmul chains.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;

/// Replace `scaled_dot_product_attention` with explicit matmul+mask+softmax+matmul.
pub struct AttentionDecomposePass;

impl Pass for AttentionDecomposePass {
    fn name(&self) -> &str {
        "ane-attention-decompose"
    }

    fn run(&self, _program: &mut Program) -> Result<()> {
        // TODO: Implement SDPA decomposition
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_name() {
        assert_eq!(AttentionDecomposePass.name(), "ane-attention-decompose");
    }
}
