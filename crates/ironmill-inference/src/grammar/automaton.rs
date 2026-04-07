//! Persistent pushdown automaton for grammar-constrained generation.
//!
//! [`GrammarState`] tracks one or more possible positions within a
//! compiled grammar. At each decode step it can:
//!
//! 1. Compute a [`TokenMask`] indicating which tokens are valid next.
//! 2. Advance the automaton after a token is accepted.
//! 3. Report whether the grammar is in an accepting (complete) state.

use std::collections::HashSet;
use std::sync::Arc;

use super::compiler::{CompiledElement, CompiledGrammar};
use super::mask::TokenMask;

/// Maximum expansion depth to guard against pathological recursive grammars.
const MAX_EXPAND_DEPTH: usize = 256;

// ── Stack types ──────────────────────────────────────────────────

/// A symbol on the pushdown stack, identifying which rule we will
/// return to and where.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StackSymbol {
    /// Rule index to return to.
    rule_idx: usize,
    /// Alternative index within that rule.
    alt_idx: usize,
    /// Next element index to process after the called rule completes.
    elem_idx: usize,
}

/// A single stack frame tracking the current position inside a rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StackFrame {
    rule_idx: usize,
    alt_idx: usize,
    elem_idx: usize,
    /// Character offset within a `Literal` element.
    char_idx: usize,
}

// ── Grammar state ────────────────────────────────────────────────

/// Runtime state for grammar-constrained generation.
///
/// Wraps a [`CompiledGrammar`] and maintains a set of possible positions
/// (branches) within the grammar. Multiple branches arise from
/// alternations and optional/repeated elements.
pub struct GrammarState {
    grammar: Arc<CompiledGrammar>,
    /// Current set of possible positions. Each position is a stack of
    /// frames representing nested rule invocations.
    positions: Vec<Vec<StackFrame>>,
}

impl GrammarState {
    /// Create a new grammar state at the beginning of the start rule.
    pub fn new(grammar: Arc<CompiledGrammar>) -> Self {
        let start = grammar.start_rule;
        let rule = &grammar.rules[start];
        let mut positions: Vec<Vec<StackFrame>> = rule
            .alternatives
            .iter()
            .enumerate()
            .map(|(alt_idx, _)| {
                vec![StackFrame {
                    rule_idx: start,
                    alt_idx,
                    elem_idx: 0,
                    char_idx: 0,
                }]
            })
            .collect();

        let mut state = GrammarState {
            grammar,
            positions: Vec::new(),
        };
        state.expand_positions(&mut positions);
        state.positions = positions;
        state
    }

    /// Compute the token mask for the current state.
    ///
    /// A token is allowed if its entire character sequence can be
    /// consumed by the automaton (in at least one branch) without
    /// reaching a dead state.
    pub fn token_mask(&self) -> TokenMask {
        let vocab_size = self.grammar.vocab.len();
        let mut mask = TokenMask::allow_none(vocab_size);

        for (token_id, token_text) in self.grammar.vocab.iter().enumerate() {
            if token_text.is_empty() {
                continue;
            }
            if self.is_token_valid(token_text) {
                mask.set_allowed(token_id, true);
            }
        }

        mask
    }

    /// Advance the automaton state after a token is accepted.
    ///
    /// Feeds each character of the token's text through the automaton
    /// and retains only branches that survive.
    ///
    /// # Panics
    /// Panics if `token_id` is out of range for the vocabulary.
    pub fn advance(&mut self, token_id: u32) {
        let idx = token_id as usize;
        assert!(
            idx < self.grammar.vocab.len(),
            "token_id {} out of range for vocabulary of size {}",
            token_id,
            self.grammar.vocab.len(),
        );
        let text = self.grammar.vocab[idx].clone();
        for ch in text.chars() {
            self.advance_char(ch);
        }
    }

    /// Check if the grammar is in an accepting state.
    ///
    /// The grammar is complete when at least one branch has an empty
    /// stack (all rules have been fully consumed).
    pub fn is_complete(&self) -> bool {
        self.positions.iter().any(|pos| pos.is_empty())
    }

    /// The compiled grammar backing this state.
    pub fn grammar(&self) -> &Arc<CompiledGrammar> {
        &self.grammar
    }

    // ── Internal helpers ─────────────────────────────────────────

    /// Check whether `text` can be fully consumed from the current state.
    fn is_token_valid(&self, text: &str) -> bool {
        let mut positions = self.positions.clone();

        for ch in text.chars() {
            positions = self.advance_positions(&positions, ch);
            if positions.is_empty() {
                return false;
            }
        }
        true
    }

    /// Advance all positions by one character, returning surviving branches.
    fn advance_positions(&self, positions: &[Vec<StackFrame>], ch: char) -> Vec<Vec<StackFrame>> {
        let mut next: Vec<Vec<StackFrame>> = Vec::new();

        for pos in positions {
            if pos.is_empty() {
                // Accepting position — cannot consume more characters, drop it.
                continue;
            }

            let frame = pos.last().unwrap();
            let rule = &self.grammar.rules[frame.rule_idx];
            let alt = &rule.alternatives[frame.alt_idx];

            if frame.elem_idx >= alt.len() {
                // Should have been expanded — skip.
                continue;
            }

            let elem = &alt[frame.elem_idx];
            match elem {
                CompiledElement::Literal(chars) => {
                    if frame.char_idx < chars.len() && chars[frame.char_idx] == ch {
                        let mut new_pos = pos.clone();
                        let f = new_pos.last_mut().unwrap();
                        f.char_idx += 1;
                        if f.char_idx >= chars.len() {
                            f.elem_idx += 1;
                            f.char_idx = 0;
                        }
                        next.push(new_pos);
                    }
                }
                CompiledElement::CharClass { ranges, negated } => {
                    let in_range = ranges.iter().any(|&(lo, hi)| ch >= lo && ch <= hi);
                    let matches = if *negated { !in_range } else { in_range };
                    if matches {
                        let mut new_pos = pos.clone();
                        let f = new_pos.last_mut().unwrap();
                        f.elem_idx += 1;
                        f.char_idx = 0;
                        next.push(new_pos);
                    }
                }
                CompiledElement::RuleRef(_) => {
                    // Should have been expanded — skip.
                }
            }
        }

        self.expand_positions(&mut next);
        next
    }

    /// Advance the live positions by one character in place.
    fn advance_char(&mut self, ch: char) {
        let current = std::mem::take(&mut self.positions);
        self.positions = self.advance_positions(&current, ch);
    }

    /// Expand positions: resolve `RuleRef` elements and completed
    /// alternatives until every position points at a terminal element
    /// (Literal or CharClass) or is an accepting state.
    fn expand_positions(&self, positions: &mut Vec<Vec<StackFrame>>) {
        // `seen` prevents re-expanding the same intermediate position,
        // guarding against infinite loops in recursive grammars.
        let mut seen: HashSet<Vec<StackFrame>> = HashSet::new();
        let mut iterations = 0;
        let max_iterations = MAX_EXPAND_DEPTH * (positions.len() + 1);
        let mut i = 0;

        while i < positions.len() && iterations < max_iterations {
            iterations += 1;
            let pos = positions[i].clone();

            if pos.is_empty() {
                // Accepting state — keep as-is.
                i += 1;
                continue;
            }

            let frame = pos.last().unwrap();
            let rule_idx = frame.rule_idx;
            let alt_idx = frame.alt_idx;
            let elem_idx = frame.elem_idx;

            let rule = &self.grammar.rules[rule_idx];
            let alt = &rule.alternatives[alt_idx];

            if elem_idx >= alt.len() {
                // Alternative completed — pop frame. The parent frame
                // already points past the RuleRef that invoked this rule,
                // so no further advancement is needed.
                let mut new_pos = pos;
                new_pos.pop();
                // Replace this position with the popped version if new.
                if seen.insert(new_pos.clone()) {
                    positions[i] = new_pos;
                    // Don't increment i — re-examine the replacement.
                } else {
                    positions.swap_remove(i);
                }
                continue;
            }

            match &alt[elem_idx] {
                CompiledElement::RuleRef(ref_rule_idx) => {
                    let ref_rule_idx = *ref_rule_idx;
                    let num_alts = self.grammar.rules[ref_rule_idx].alternatives.len();
                    positions.swap_remove(i);

                    for a in 0..num_alts {
                        let mut new_pos = pos.clone();
                        let f = new_pos.last_mut().unwrap();
                        f.elem_idx += 1;
                        f.char_idx = 0;
                        new_pos.push(StackFrame {
                            rule_idx: ref_rule_idx,
                            alt_idx: a,
                            elem_idx: 0,
                            char_idx: 0,
                        });
                        if seen.insert(new_pos.clone()) {
                            positions.push(new_pos);
                        }
                    }
                    // Don't increment i.
                    continue;
                }
                CompiledElement::Literal(chars) if chars.is_empty() => {
                    let mut new_pos = pos;
                    let f = new_pos.last_mut().unwrap();
                    f.elem_idx += 1;
                    f.char_idx = 0;
                    if seen.insert(new_pos.clone()) {
                        positions[i] = new_pos;
                    } else {
                        positions.swap_remove(i);
                    }
                    continue;
                }
                _ => {
                    // Terminal element (non-empty Literal or CharClass)
                    // — this position is ready. Keep it.
                    i += 1;
                }
            }
        }

        // Final dedup: keep the first occurrence of each position.
        let mut final_seen = HashSet::new();
        positions.retain(|pos| final_seen.insert(pos.clone()));
    }
}

impl Clone for GrammarState {
    fn clone(&self) -> Self {
        GrammarState {
            grammar: Arc::clone(&self.grammar),
            positions: self.positions.clone(),
        }
    }
}

impl std::fmt::Debug for GrammarState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarState")
            .field("positions", &self.positions.len())
            .field("complete", &self.is_complete())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::bnf::Grammar;
    use crate::grammar::compiler::CompiledGrammar;

    /// Build a grammar state with a tiny vocabulary for testing.
    fn make_state(bnf: &str, vocab: Vec<&str>) -> GrammarState {
        let g = Grammar::from_bnf(bnf).unwrap();
        let vocab: Vec<String> = vocab.into_iter().map(String::from).collect();
        let cg = CompiledGrammar::new(&g, vocab).unwrap();
        GrammarState::new(Arc::new(cg))
    }

    #[test]
    fn grammar_automaton_literal_only() {
        let state = make_state(
            r#"root ::= "true""#,
            vec!["t", "r", "u", "e", "true", "false"],
        );
        assert!(!state.is_complete());
        let mask = state.token_mask();
        // "t" and "true" should be allowed.
        assert!(mask.is_allowed(0)); // "t"
        assert!(mask.is_allowed(4)); // "true"
        assert!(!mask.is_allowed(5)); // "false"
    }

    #[test]
    fn grammar_automaton_advance_completes() {
        let mut state = make_state(r#"root ::= "ok""#, vec!["o", "k", "ok", "x"]);
        assert!(!state.is_complete());

        // Advance with "ok" token (id=2).
        state.advance(2);
        assert!(state.is_complete());
    }

    #[test]
    fn grammar_automaton_advance_char_by_char() {
        let mut state = make_state(r#"root ::= "ab""#, vec!["a", "b", "ab", "x"]);
        // Advance with "a" then "b".
        state.advance(0); // "a"
        assert!(!state.is_complete());
        state.advance(1); // "b"
        assert!(state.is_complete());
    }

    #[test]
    fn grammar_automaton_alternation() {
        let state = make_state(r#"root ::= "yes" | "no""#, vec!["y", "n", "yes", "no", "x"]);
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "y"
        assert!(mask.is_allowed(1)); // "n"
        assert!(mask.is_allowed(2)); // "yes"
        assert!(mask.is_allowed(3)); // "no"
        assert!(!mask.is_allowed(4)); // "x"
    }

    #[test]
    fn grammar_automaton_char_class() {
        let state = make_state(r#"root ::= [0-9]"#, vec!["0", "5", "9", "a", "x"]);
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "0"
        assert!(mask.is_allowed(1)); // "5"
        assert!(mask.is_allowed(2)); // "9"
        assert!(!mask.is_allowed(3)); // "a"
        assert!(!mask.is_allowed(4)); // "x"
    }

    #[test]
    fn grammar_automaton_sequence() {
        let state = make_state(r#"root ::= "a" "b""#, vec!["a", "b", "ab", "ba"]);
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "a" (partial match)
        assert!(!mask.is_allowed(1)); // "b" (doesn't start with "a")
        assert!(mask.is_allowed(2)); // "ab" (full match)
        assert!(!mask.is_allowed(3)); // "ba"
    }

    #[test]
    fn grammar_automaton_optional() {
        let mut state = make_state(r#"root ::= "-"? [0-9]"#, vec!["-", "0", "5", "-5", "x"]);
        let mask = state.token_mask();
        // "-" is valid (optional minus, then need digit)
        assert!(mask.is_allowed(0)); // "-"
        // "0" is valid (skip optional minus, match digit)
        assert!(mask.is_allowed(1)); // "0"
        // "-5" is valid (minus then digit)
        assert!(mask.is_allowed(3)); // "-5"
        assert!(!mask.is_allowed(4)); // "x"

        // Advance with "-", then only digits should be valid.
        state.advance(0);
        let mask = state.token_mask();
        assert!(!mask.is_allowed(0)); // "-" not valid after "-"
        assert!(mask.is_allowed(1)); // "0"
        assert!(mask.is_allowed(2)); // "5"
    }

    #[test]
    fn grammar_automaton_zero_or_more() {
        let state = make_state(r#"root ::= "a"*"#, vec!["a", "aa", "b", "aab"]);
        // At start, the grammar can match zero or more "a"s.
        // So it's immediately in an accepting state (zero matches).
        assert!(state.is_complete());
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "a"
        assert!(mask.is_allowed(1)); // "aa"
        assert!(!mask.is_allowed(2)); // "b"
        assert!(!mask.is_allowed(3)); // "aab"
    }

    #[test]
    fn grammar_automaton_one_or_more() {
        let state = make_state(r#"root ::= [0-9]+"#, vec!["0", "12", "abc", "1a"]);
        assert!(!state.is_complete());
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "0"
        assert!(mask.is_allowed(1)); // "12"
        assert!(!mask.is_allowed(2)); // "abc"
        assert!(!mask.is_allowed(3)); // "1a" — '1' ok but 'a' fails
    }

    #[test]
    fn grammar_automaton_rule_ref() {
        let state = make_state(
            r#"
root ::= <greeting>
greeting ::= "hi" | "bye"
"#,
            vec!["h", "hi", "bye", "x"],
        );
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "h"
        assert!(mask.is_allowed(1)); // "hi"
        assert!(mask.is_allowed(2)); // "bye"
        assert!(!mask.is_allowed(3)); // "x"
    }

    #[test]
    fn grammar_automaton_nested_rules() {
        let state = make_state(
            r#"
root ::= "{" <value> "}"
value ::= "true" | "false"
"#,
            vec!["{", "}", "true", "false", "{true}", "x"],
        );
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "{"
        assert!(!mask.is_allowed(1)); // "}" not yet
        assert!(!mask.is_allowed(2)); // "true" not yet (need "{" first)
        assert!(mask.is_allowed(4)); // "{true}" matches everything
    }

    #[test]
    fn grammar_automaton_boolean_json() {
        let bnf = r#"root ::= "true" | "false""#;
        let vocab: Vec<&str> = vec![
            "t", "r", "u", "e", "f", "a", "l", "s", "true", "false", "tr", "fa", "x",
        ];
        let mut state = make_state(bnf, vocab);

        // Initially: "t", "f", "true", "false", "tr", "fa" should be valid.
        let mask = state.token_mask();
        assert!(mask.is_allowed(0)); // "t"
        assert!(mask.is_allowed(4)); // "f"
        assert!(mask.is_allowed(8)); // "true"
        assert!(mask.is_allowed(9)); // "false"
        assert!(mask.is_allowed(10)); // "tr"
        assert!(mask.is_allowed(11)); // "fa"
        assert!(!mask.is_allowed(12)); // "x"

        // After "tr", only "u" and "ue" should be valid.
        state.advance(10); // "tr"
        let mask = state.token_mask();
        assert!(mask.is_allowed(2)); // "u"
        assert!(!mask.is_allowed(0)); // "t"
        assert!(!mask.is_allowed(4)); // "f"
    }
}
