//! Tests for JSON Schema objects with optional properties — ensures
//! commas only appear between present members (no leading commas).

use super::automaton::GrammarState;
use super::compiler::CompiledGrammar;
use super::json_schema::json_schema_to_grammar;
use serde_json::Value;
use std::sync::Arc;

fn json_vocab() -> Vec<String> {
    vec![
        "{".into(),
        "}".into(),
        "[".into(),
        "]".into(),
        ":".into(),
        ",".into(),
        "\"".into(),
        " ".into(),
        "true".into(),
        "false".into(),
        "null".into(),
        "0".into(),
        "1".into(),
        "42".into(),
        "-".into(),
        ".".into(),
        "\"name\"".into(),
        "\"age\"".into(),
        "\"active\"".into(),
        "hello".into(),
        "a".into(),
        "b".into(),
        "\n".into(),
        "\"name\":".into(),
        "\"hello\"".into(),
        "\"a\"".into(),
        "\"b\"".into(),
        "\"c\"".into(),
    ]
}

fn make_state(schema_json: &str) -> GrammarState {
    let schema: Value = serde_json::from_str(schema_json).unwrap();
    let grammar = json_schema_to_grammar(&schema).unwrap();
    let compiled = CompiledGrammar::new(&grammar, json_vocab()).unwrap();
    GrammarState::new(Arc::new(compiled))
}

/// Feed a sequence of tokens by text, returning the final state.
fn feed_tokens(state: &mut GrammarState, vocab: &[String], tokens: &[&str]) {
    for tok_text in tokens {
        let idx = vocab
            .iter()
            .position(|v| v == tok_text)
            .unwrap_or_else(|| panic!("token {tok_text:?} not in vocab"));
        let mask = state.token_mask();
        assert!(
            mask.is_allowed(idx),
            "token {tok_text:?} (idx {idx}) was not allowed"
        );
        state.advance(idx as u32).unwrap();
    }
}

#[test]
fn optional_before_required_no_leading_comma() {
    // "a" is optional, "b" is required-by-schema but the grammar now
    // treats all properties uniformly — the point is that producing
    // {"b":true} must NOT generate a leading comma.
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"}
        },
        "required": ["b"]
    }"#;

    let mut state = make_state(schema);
    let vocab = json_vocab();

    // Produce: {"b":true}  — only the second property, no leading comma.
    // The grammar should allow "{" then a member directly (no comma).
    let mask = state.token_mask();
    assert!(mask.is_allowed(0), "'{{' should be allowed to open object");

    // After "{", a property key should be allowed (no comma required).
    state.advance(0).unwrap(); // "{"
    let mask = state.token_mask();
    // "\"b\"" is at index 27 in our vocab
    let b_key_idx = vocab.iter().position(|v| v == "\"b\"").unwrap();
    assert!(
        mask.is_allowed(b_key_idx),
        "property key '\"b\"' should be allowed right after open brace"
    );
}

#[test]
fn all_optional_empty_object_valid() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"}
        }
    }"#;

    let mut state = make_state(schema);
    let vocab = json_vocab();

    // Should allow producing an empty object: {}
    feed_tokens(&mut state, &vocab, &["{", "}"]);
    assert!(state.is_complete());
}

#[test]
fn two_properties_with_comma() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"}
        }
    }"#;

    let mut state = make_state(schema);
    let vocab = json_vocab();

    // Produce: {"a":true,"b":false}
    feed_tokens(
        &mut state,
        &vocab,
        &["{", "\"a\"", ":", "true", ",", "\"b\"", ":", "false", "}"],
    );
    assert!(state.is_complete());
}

#[test]
fn single_property_no_trailing_comma() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"}
        }
    }"#;

    let mut state = make_state(schema);
    let vocab = json_vocab();

    // Produce: {"a":true} — single property, no trailing comma needed.
    feed_tokens(&mut state, &vocab, &["{", "\"a\"", ":", "true", "}"]);
    assert!(state.is_complete());
}

#[test]
fn compiler_undefined_rule_returns_error() {
    use super::bnf::{Element, Grammar, GrammarError, Rule};

    let grammar = Grammar {
        start_rule: "root".into(),
        rules: vec![Rule {
            name: "root".into(),
            expr: Element::RuleRef("nonexistent".into()),
        }],
    };

    let result = CompiledGrammar::new(&grammar, vec!["x".into()]);
    assert!(result.is_err());
    match result.unwrap_err() {
        GrammarError::UndefinedRule(name) => assert_eq!(name, "nonexistent"),
        other => panic!("expected UndefinedRule, got {other:?}"),
    }
}
