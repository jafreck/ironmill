//! Structured generation — grammar/JSON-constrained sampling.
//!
//! This module implements XGrammar-style constrained generation that
//! forces model output to conform to a schema (JSON, function
//! signatures, BNF grammars) by masking invalid tokens at each decode
//! step.
//!
//! # Architecture
//!
//! ```text
//! Grammar (BNF / JSON Schema)
//!   ↓  compile
//! CompiledGrammar (flat rules + vocab mapping)
//!   ↓  instantiate
//! GrammarState (pushdown automaton)
//!   ↓  per token
//! TokenMask → apply_token_mask(logits) → sample
//! ```
//!
//! # Quick start
//!
//! ```ignore
//! use ironmill_inference::grammar::*;
//!
//! // From a BNF grammar:
//! let grammar = Grammar::from_bnf(r#"root ::= "true" | "false""#)?;
//! let compiled = CompiledGrammar::new(&grammar, vocab)?;
//! let mut state = GrammarState::new(Arc::new(compiled));
//!
//! // From a JSON schema:
//! let schema: serde_json::Value = serde_json::from_str(r#"{"type":"boolean"}"#)?;
//! let grammar = json_schema_to_grammar(&schema)?;
//! let compiled = CompiledGrammar::new(&grammar, vocab)?;
//! let mut state = GrammarState::new(Arc::new(compiled));
//!
//! // In the decode loop:
//! let mask = state.token_mask();
//! apply_token_mask(&mut logits, &mask);
//! let token = sample_token(&logits, temperature);
//! state.advance(token);
//! ```

pub mod automaton;
pub mod bnf;
pub mod compiler;
pub mod json_schema;
pub mod mask;

// Re-exports for convenience.
pub use automaton::GrammarState;
pub use bnf::{Element, Grammar, GrammarError, Rule};
pub use compiler::{CompiledGrammar, TokenClass};
pub use json_schema::json_schema_to_grammar;
pub use mask::TokenMask;

#[cfg(test)]
#[path = "_test_optional_comma.rs"]
mod test_optional_comma;
