//! Grammar compiler — transform a [`Grammar`] AST into a flat
//! [`CompiledGrammar`] suitable for pushdown-automaton execution.
//!
//! The compiler lowers `Optional`, `ZeroOrMore`, and `OneOrMore` nodes
//! into helper rules so the runtime only needs to handle three element
//! kinds: [`CompiledElement::Literal`], [`CompiledElement::CharClass`],
//! and [`CompiledElement::RuleRef`].

use std::collections::HashMap;

use super::bnf::{Element, Grammar, GrammarError};
use super::mask::TokenMask;

// ── Compiled representation ──────────────────────────────────────

/// A compiled element — the leaf types the automaton processes.
#[derive(Debug, Clone)]
pub(crate) enum CompiledElement {
    /// Match these characters in sequence.
    Literal(Vec<char>),
    /// Match one character in the given ranges (or not in them if negated).
    CharClass {
        ranges: Vec<(char, char)>,
        negated: bool,
    },
    /// Enter another rule.
    RuleRef(usize),
}

/// A compiled rule: a set of alternatives, each a sequence of
/// [`CompiledElement`]s.
#[derive(Debug, Clone)]
pub(crate) struct CompiledRule {
    pub(crate) alternatives: Vec<Vec<CompiledElement>>,
}

/// Pre-computed masks for context-independent token classes.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenClass {
    /// Tokens consisting entirely of decimal digits.
    Digit,
    /// Tokens consisting entirely of ASCII letters.
    Letter,
    /// Tokens consisting entirely of ASCII whitespace.
    Whitespace,
}

/// Compiled grammar ready for constrained generation.
///
/// Holds the flattened rule table, token vocabulary, and optional
/// pre-computed masks for common token classes.
pub struct CompiledGrammar {
    /// Pre-computed masks for context-independent token classes.
    pub(crate) precomputed_masks: HashMap<TokenClass, TokenMask>,
    /// Flattened rules (user-defined + compiler-generated helpers).
    pub(crate) rules: Vec<CompiledRule>,
    /// Map from user-visible rule name → rule index.
    pub(crate) rule_index: HashMap<String, usize>,
    /// Index of the start rule.
    pub(crate) start_rule: usize,
    /// Token vocabulary: `vocab[token_id]` is the token's text.
    pub(crate) vocab: Vec<String>,
}

impl CompiledGrammar {
    /// Compile a [`Grammar`] with the given token vocabulary.
    ///
    /// `vocab` maps each token id (by index) to its text representation.
    /// This is required for computing per-token masks at decode time.
    pub fn new(grammar: &Grammar, vocab: Vec<String>) -> Result<Self, GrammarError> {
        let mut rules: Vec<CompiledRule> = Vec::new();
        let mut rule_index: HashMap<String, usize> = HashMap::new();

        // First pass: reserve slots for user-defined rules.
        for rule in &grammar.rules {
            let idx = rules.len();
            rule_index.insert(rule.name.clone(), idx);
            rules.push(CompiledRule {
                alternatives: vec![],
            });
        }

        // Second pass: compile each rule body.
        for rule in &grammar.rules {
            let alts = compile_top_level(&rule.expr, &mut rules, &rule_index)?;
            let idx = rule_index[&rule.name];
            rules[idx].alternatives = alts;
        }

        let start_rule = rule_index[&grammar.start_rule];

        let precomputed_masks = precompute_token_class_masks(&vocab);

        Ok(CompiledGrammar {
            precomputed_masks,
            rules,
            rule_index,
            start_rule,
            vocab,
        })
    }

    /// Number of rules (user-defined + generated helpers).
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up a pre-computed mask for a token class.
    pub fn precomputed_mask(&self, class: TokenClass) -> Option<&TokenMask> {
        self.precomputed_masks.get(&class)
    }

    /// Look up a rule index by name.
    pub fn rule_by_name(&self, name: &str) -> Option<usize> {
        self.rule_index.get(name).copied()
    }
}

impl std::fmt::Debug for CompiledGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGrammar")
            .field("rules", &self.rules.len())
            .field("vocab_size", &self.vocab.len())
            .field("start_rule", &self.start_rule)
            .finish()
    }
}

// ── Compilation helpers ──────────────────────────────────────────

/// Compile a top-level rule expression into alternatives.
///
/// If the expression is an `Alternation`, each branch becomes its own
/// alternative. Otherwise the whole expression is a single alternative.
fn compile_top_level(
    expr: &Element,
    rules: &mut Vec<CompiledRule>,
    name_map: &HashMap<String, usize>,
) -> Result<Vec<Vec<CompiledElement>>, GrammarError> {
    match expr {
        Element::Alternation(alts) => alts
            .iter()
            .map(|alt| compile_element(alt, rules, name_map))
            .collect::<Result<_, _>>(),
        other => Ok(vec![compile_element(other, rules, name_map)?]),
    }
}

/// Recursively compile a grammar element into a flat list of
/// [`CompiledElement`]s, creating helper rules for repetitions.
fn compile_element(
    elem: &Element,
    rules: &mut Vec<CompiledRule>,
    name_map: &HashMap<String, usize>,
) -> Result<Vec<CompiledElement>, GrammarError> {
    match elem {
        Element::Literal(s) => {
            if s.is_empty() {
                Ok(vec![])
            } else {
                Ok(vec![CompiledElement::Literal(s.chars().collect())])
            }
        }

        Element::CharClass { ranges, negated } => Ok(vec![CompiledElement::CharClass {
            ranges: ranges.clone(),
            negated: *negated,
        }]),

        Element::RuleRef(name) => {
            let idx = name_map
                .get(name)
                .copied()
                .ok_or_else(|| GrammarError::UndefinedRule(name.clone()))?;
            Ok(vec![CompiledElement::RuleRef(idx)])
        }

        Element::Sequence(parts) => {
            let mut result = Vec::new();
            for part in parts {
                result.extend(compile_element(part, rules, name_map)?);
            }
            Ok(result)
        }

        Element::Alternation(alts) => {
            let alternatives: Vec<Vec<CompiledElement>> = alts
                .iter()
                .map(|alt| compile_element(alt, rules, name_map))
                .collect::<Result<_, _>>()?;
            let rule_idx = rules.len();
            rules.push(CompiledRule { alternatives });
            Ok(vec![CompiledElement::RuleRef(rule_idx)])
        }

        Element::Optional(inner) => {
            let inner_compiled = compile_element(inner, rules, name_map)?;
            let rule_idx = rules.len();
            rules.push(CompiledRule {
                alternatives: vec![inner_compiled, vec![]],
            });
            Ok(vec![CompiledElement::RuleRef(rule_idx)])
        }

        Element::ZeroOrMore(inner) => {
            // star → inner star | ε
            let rule_idx = rules.len();
            rules.push(CompiledRule {
                alternatives: vec![],
            });
            let mut body = compile_element(inner, rules, name_map)?;
            body.push(CompiledElement::RuleRef(rule_idx));
            rules[rule_idx].alternatives = vec![body, vec![]];
            Ok(vec![CompiledElement::RuleRef(rule_idx)])
        }

        Element::OneOrMore(inner) => {
            // plus → inner star, where star → inner star | ε
            let star_idx = rules.len();
            rules.push(CompiledRule {
                alternatives: vec![],
            });
            let inner_for_star = compile_element(inner, rules, name_map)?;
            let mut star_body = inner_for_star;
            star_body.push(CompiledElement::RuleRef(star_idx));
            rules[star_idx].alternatives = vec![star_body, vec![]];

            let mut result = compile_element(inner, rules, name_map)?;
            result.push(CompiledElement::RuleRef(star_idx));
            Ok(result)
        }
    }
}

// ── Pre-computed token class masks ───────────────────────────────

fn precompute_token_class_masks(vocab: &[String]) -> HashMap<TokenClass, TokenMask> {
    let n = vocab.len();
    let mut digit_mask = TokenMask::allow_none(n);
    let mut letter_mask = TokenMask::allow_none(n);
    let mut ws_mask = TokenMask::allow_none(n);

    for (i, text) in vocab.iter().enumerate() {
        if !text.is_empty() && text.chars().all(|c| c.is_ascii_digit()) {
            digit_mask.set_allowed(i, true);
        }
        if !text.is_empty() && text.chars().all(|c| c.is_ascii_alphabetic()) {
            letter_mask.set_allowed(i, true);
        }
        if !text.is_empty() && text.chars().all(|c| c.is_ascii_whitespace()) {
            ws_mask.set_allowed(i, true);
        }
    }

    let mut map = HashMap::new();
    map.insert(TokenClass::Digit, digit_mask);
    map.insert(TokenClass::Letter, letter_mask);
    map.insert(TokenClass::Whitespace, ws_mask);
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::bnf::Grammar;

    fn sample_vocab() -> Vec<String> {
        vec![
            "t".into(),
            "true".into(),
            "false".into(),
            "0".into(),
            "1".into(),
            "hello".into(),
            " ".into(),
        ]
    }

    #[test]
    fn grammar_compile_simple_literal() {
        let g = Grammar::from_bnf(r#"root ::= "true""#).unwrap();
        let cg = CompiledGrammar::new(&g, sample_vocab()).unwrap();
        assert_eq!(cg.rules[cg.start_rule].alternatives.len(), 1);
    }

    #[test]
    fn grammar_compile_alternation() {
        let g = Grammar::from_bnf(r#"root ::= "true" | "false""#).unwrap();
        let cg = CompiledGrammar::new(&g, sample_vocab()).unwrap();
        assert_eq!(cg.rules[cg.start_rule].alternatives.len(), 2);
    }

    #[test]
    fn grammar_compile_creates_helper_rules() {
        let g = Grammar::from_bnf(r#"root ::= [0-9]+"#).unwrap();
        let cg = CompiledGrammar::new(&g, sample_vocab()).unwrap();
        // User rule + star helper
        assert!(cg.rule_count() >= 2);
    }

    #[test]
    fn grammar_compile_optional_creates_helper() {
        let g = Grammar::from_bnf(r#"root ::= "-"?"#).unwrap();
        let cg = CompiledGrammar::new(&g, sample_vocab()).unwrap();
        assert!(cg.rule_count() >= 2);
    }

    #[test]
    fn grammar_compile_precomputed_masks() {
        let cg = CompiledGrammar::new(
            &Grammar::from_bnf(r#"root ::= "x""#).unwrap(),
            sample_vocab(),
        )
        .unwrap();
        let digit_mask = &cg.precomputed_masks[&TokenClass::Digit];
        // "0" and "1" are at indices 3 and 4.
        assert!(digit_mask.is_allowed(3));
        assert!(digit_mask.is_allowed(4));
        assert!(!digit_mask.is_allowed(0)); // "t"
    }

    #[test]
    fn grammar_compile_multi_rule() {
        let g = Grammar::from_bnf(
            r#"
value ::= <bool>
bool ::= "true" | "false"
"#,
        )
        .unwrap();
        let cg = CompiledGrammar::new(&g, sample_vocab()).unwrap();
        assert!(cg.rule_index.contains_key("value"));
        assert!(cg.rule_index.contains_key("bool"));
    }
}
