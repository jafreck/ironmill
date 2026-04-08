//! JSON Schema → BNF grammar converter.
//!
//! Converts a JSON Schema (as a [`serde_json::Value`]) into a
//! [`Grammar`] whose start rule constrains output to valid JSON
//! matching the schema.

use serde_json::Value;

use super::bnf::{Element, Grammar, GrammarError, Rule};

/// Convert a JSON Schema into a [`Grammar`].
///
/// Supports the following JSON Schema types:
/// - `"string"` — JSON string (simplified: no escape handling)
/// - `"number"` / `"integer"` — JSON number
/// - `"boolean"` — `true` or `false`
/// - `"null"` — `null`
/// - `"object"` with `properties` — JSON object with known keys
/// - `"array"` with `items` — JSON array of homogeneous items
///
/// Returns a `Grammar` whose start rule is `"root"`.
pub fn json_schema_to_grammar(schema: &Value) -> Result<Grammar, GrammarError> {
    let mut ctx = ConvertCtx {
        rules: Vec::new(),
        counter: 0,
    };

    // Add shared helper rules.
    ctx.add_whitespace_rule();

    let root_expr = ctx.convert_schema(schema)?;
    ctx.rules.push(Rule {
        name: "root".into(),
        expr: root_expr,
    });

    // Move root to position 0 so it's the start rule.
    let root_idx = ctx.rules.len() - 1;
    ctx.rules.swap(0, root_idx);

    Ok(Grammar {
        start_rule: "root".into(),
        rules: ctx.rules,
    })
}

// ── Conversion context ───────────────────────────────────────────

struct ConvertCtx {
    rules: Vec<Rule>,
    counter: usize,
}

impl ConvertCtx {
    fn fresh_name(&mut self, prefix: &str) -> String {
        let name = format!("__{prefix}_{}", self.counter);
        self.counter += 1;
        name
    }

    fn add_whitespace_rule(&mut self) {
        // ws ::= " "*
        self.rules.push(Rule {
            name: "__ws".into(),
            expr: Element::ZeroOrMore(Box::new(Element::CharClass {
                ranges: vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
                negated: false,
            })),
        });
    }

    fn convert_schema(&mut self, schema: &Value) -> Result<Element, GrammarError> {
        let type_str = schema.get("type").and_then(Value::as_str);

        match type_str {
            Some("string") => Ok(self.convert_string()),
            Some("number") => Ok(self.convert_number()),
            Some("integer") => Ok(self.convert_integer()),
            Some("boolean") => Ok(self.convert_boolean()),
            Some("null") => Ok(Element::Literal("null".into())),
            Some("object") => self.convert_object(schema),
            Some("array") => self.convert_array(schema),
            Some(other) => Err(GrammarError::UnsupportedType(other.to_string())),
            None => Err(GrammarError::UnsupportedType("(missing)".to_string())),
        }
    }

    fn convert_string(&self) -> Element {
        // '"' [^"\\]* '"'
        Element::Sequence(vec![
            Element::Literal("\"".into()),
            Element::ZeroOrMore(Box::new(Element::CharClass {
                ranges: vec![('"', '"'), ('\\', '\\')],
                negated: true,
            })),
            Element::Literal("\"".into()),
        ])
    }

    fn convert_number(&self) -> Element {
        // '-'? [0-9]+ ('.' [0-9]+)?
        Element::Sequence(vec![
            Element::Optional(Box::new(Element::Literal("-".into()))),
            Element::OneOrMore(Box::new(Element::CharClass {
                ranges: vec![('0', '9')],
                negated: false,
            })),
            Element::Optional(Box::new(Element::Sequence(vec![
                Element::Literal(".".into()),
                Element::OneOrMore(Box::new(Element::CharClass {
                    ranges: vec![('0', '9')],
                    negated: false,
                })),
            ]))),
        ])
    }

    fn convert_integer(&self) -> Element {
        // '-'? [0-9]+
        Element::Sequence(vec![
            Element::Optional(Box::new(Element::Literal("-".into()))),
            Element::OneOrMore(Box::new(Element::CharClass {
                ranges: vec![('0', '9')],
                negated: false,
            })),
        ])
    }

    fn convert_boolean(&self) -> Element {
        Element::Alternation(vec![
            Element::Literal("true".into()),
            Element::Literal("false".into()),
        ])
    }

    fn convert_object(&mut self, schema: &Value) -> Result<Element, GrammarError> {
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        // Determine which properties are required.
        let required_set: std::collections::HashSet<String> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();

        let additional_props_allowed = schema
            .get("additionalProperties")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        // Warn about unsupported constraints.
        if schema.get("uniqueItems").is_some() {
            eprintln!(
                "warning: JSON Schema 'uniqueItems' constraint is not supported and will be ignored"
            );
        }
        if schema.get("minProperties").is_some() {
            eprintln!(
                "warning: JSON Schema 'minProperties' constraint is not supported and will be ignored"
            );
        }
        if schema.get("maxProperties").is_some() {
            eprintln!(
                "warning: JSON Schema 'maxProperties' constraint is not supported and will be ignored"
            );
        }

        if properties.is_empty() {
            if !additional_props_allowed {
                // No properties allowed, only empty object.
                return Ok(Element::Sequence(vec![
                    Element::Literal("{".into()),
                    Element::RuleRef("__ws".into()),
                    Element::Literal("}".into()),
                ]));
            }
            // Empty object: '{' ws '}'
            return Ok(Element::Sequence(vec![
                Element::Literal("{".into()),
                Element::RuleRef("__ws".into()),
                Element::Literal("}".into()),
            ]));
        }

        // Build a member expression for each property.
        // Required properties are emitted directly; optional ones are wrapped
        // in an alternation with the empty production.
        let mut required_members = Vec::new();
        let mut optional_members = Vec::new();
        let prop_keys: Vec<String> = properties.keys().cloned().collect();

        for key in &prop_keys {
            let prop_schema = &properties[key];
            let value_expr = self.convert_schema(prop_schema)?;

            let value_rule_name = self.fresh_name(&format!("prop_{key}"));
            self.rules.push(Rule {
                name: value_rule_name.clone(),
                expr: value_expr,
            });

            let member_rule_name = self.fresh_name(&format!("member_{key}"));
            self.rules.push(Rule {
                name: member_rule_name.clone(),
                expr: Element::Sequence(vec![
                    Element::Literal(format!("\"{key}\"")),
                    Element::RuleRef("__ws".into()),
                    Element::Literal(":".into()),
                    Element::RuleRef("__ws".into()),
                    Element::RuleRef(value_rule_name),
                ]),
            });

            if required_set.contains(key) {
                required_members.push(Element::RuleRef(member_rule_name));
            } else {
                optional_members.push(Element::RuleRef(member_rule_name));
            }
        }

        // Build the object body: required properties form a mandatory
        // fixed-order sequence; optional properties are each individually
        // wrapped in Optional and appended after.
        if !required_members.is_empty() {
            let mut body = Vec::new();
            for (i, member) in required_members.into_iter().enumerate() {
                if i > 0 {
                    body.push(Element::RuleRef("__ws".into()));
                    body.push(Element::Literal(",".into()));
                    body.push(Element::RuleRef("__ws".into()));
                }
                body.push(member);
            }
            for member in optional_members {
                body.push(Element::Optional(Box::new(Element::Sequence(vec![
                    Element::RuleRef("__ws".into()),
                    Element::Literal(",".into()),
                    Element::RuleRef("__ws".into()),
                    member,
                ]))));
            }
            Ok(Element::Sequence(vec![
                Element::Literal("{".into()),
                Element::RuleRef("__ws".into()),
                Element::Sequence(body),
                Element::RuleRef("__ws".into()),
                Element::Literal("}".into()),
            ]))
        } else {
            // No required properties — all are optional.
            // Structure: (opt1 ("," ws opt2)? ("," ws opt3)?)?
            let mut inner = Vec::new();
            for (i, member) in optional_members.into_iter().enumerate() {
                if i == 0 {
                    inner.push(member);
                } else {
                    inner.push(Element::Optional(Box::new(Element::Sequence(vec![
                        Element::RuleRef("__ws".into()),
                        Element::Literal(",".into()),
                        Element::RuleRef("__ws".into()),
                        member,
                    ]))));
                }
            }

            let body = if inner.len() == 1 {
                Element::Optional(Box::new(inner.pop().unwrap()))
            } else {
                Element::Optional(Box::new(Element::Sequence(inner)))
            };

            Ok(Element::Sequence(vec![
                Element::Literal("{".into()),
                Element::RuleRef("__ws".into()),
                body,
                Element::RuleRef("__ws".into()),
                Element::Literal("}".into()),
            ]))
        }
    }

    fn convert_array(&mut self, schema: &Value) -> Result<Element, GrammarError> {
        let items_schema =
            schema
                .get("items")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::from_iter([(
                    "type".to_string(),
                    Value::String("string".into()),
                )])));

        let item_expr = self.convert_schema(&items_schema)?;
        let item_rule_name = self.fresh_name("item");
        self.rules.push(Rule {
            name: item_rule_name.clone(),
            expr: item_expr,
        });

        // Build: more_items ::= ',' ws item
        let more_rule_name = self.fresh_name("more");
        self.rules.push(Rule {
            name: more_rule_name.clone(),
            expr: Element::Sequence(vec![
                Element::RuleRef("__ws".into()),
                Element::Literal(",".into()),
                Element::RuleRef("__ws".into()),
                Element::RuleRef(item_rule_name.clone()),
            ]),
        });

        // '[' ws (item (more_items)*)? ws ']'
        Ok(Element::Sequence(vec![
            Element::Literal("[".into()),
            Element::RuleRef("__ws".into()),
            Element::Optional(Box::new(Element::Sequence(vec![
                Element::RuleRef(item_rule_name),
                Element::ZeroOrMore(Box::new(Element::RuleRef(more_rule_name))),
            ]))),
            Element::RuleRef("__ws".into()),
            Element::Literal("]".into()),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::automaton::GrammarState;
    use crate::grammar::compiler::CompiledGrammar;
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
        ]
    }

    fn make_schema_state(schema_json: &str) -> GrammarState {
        let schema: Value = serde_json::from_str(schema_json).unwrap();
        let grammar = json_schema_to_grammar(&schema).unwrap();
        let compiled = CompiledGrammar::new(&grammar, json_vocab()).unwrap();
        GrammarState::new(Arc::new(compiled))
    }

    #[test]
    fn grammar_json_schema_boolean() {
        let state = make_schema_state(r#"{"type": "boolean"}"#);
        let mask = state.token_mask();
        assert!(mask.is_allowed(8)); // "true"
        assert!(mask.is_allowed(9)); // "false"
        assert!(!mask.is_allowed(10)); // "null"
        assert!(!mask.is_allowed(0)); // "{"
    }

    #[test]
    fn grammar_json_schema_number() {
        let state = make_schema_state(r#"{"type": "number"}"#);
        let mask = state.token_mask();
        assert!(mask.is_allowed(11)); // "0"
        assert!(mask.is_allowed(12)); // "1"
        assert!(mask.is_allowed(13)); // "42"
        assert!(mask.is_allowed(14)); // "-" (start of negative number)
        assert!(!mask.is_allowed(8)); // "true"
    }

    #[test]
    fn grammar_json_schema_string() {
        let state = make_schema_state(r#"{"type": "string"}"#);
        let mask = state.token_mask();
        assert!(mask.is_allowed(6)); // '"' (opening quote)
        assert!(mask.is_allowed(24)); // '"hello"' (complete string)
        assert!(!mask.is_allowed(8)); // "true"
    }

    #[test]
    fn grammar_json_schema_null() {
        let state = make_schema_state(r#"{"type": "null"}"#);
        let mask = state.token_mask();
        assert!(mask.is_allowed(10)); // "null"
        assert!(!mask.is_allowed(8)); // "true"
    }

    #[test]
    fn grammar_json_schema_array_boolean() {
        let schema = r#"{"type": "array", "items": {"type": "boolean"}}"#;
        let state = make_schema_state(schema);
        let mask = state.token_mask();
        assert!(mask.is_allowed(2)); // "[" to open array
        assert!(!mask.is_allowed(0)); // "{" not valid
    }

    #[test]
    fn grammar_json_schema_produces_valid_grammar() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }"#;
        let schema: Value = serde_json::from_str(schema).unwrap();
        let grammar = json_schema_to_grammar(&schema).unwrap();
        assert!(!grammar.rules.is_empty());
        assert_eq!(grammar.start_rule, "root");
    }

    #[test]
    fn grammar_json_schema_boolean_full_generation() {
        let schema = r#"{"type": "boolean"}"#;
        let mut state = make_schema_state(schema);

        // Generate "true" token by token.
        let mask = state.token_mask();
        assert!(mask.is_allowed(8)); // "true" is allowed
        state.advance(8).unwrap(); // accept "true"
        assert!(state.is_complete());
    }

    #[test]
    fn grammar_json_schema_empty_object() {
        let schema = r#"{"type": "object"}"#;
        let schema_val: Value = serde_json::from_str(schema).unwrap();
        let grammar = json_schema_to_grammar(&schema_val).unwrap();
        assert_eq!(grammar.start_rule, "root");
    }

    #[test]
    fn grammar_json_schema_compilation_speed() {
        // A moderately complex schema should compile quickly.
        let schema = r#"{
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "active": {"type": "boolean"},
                "score": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string"},
                        "updated": {"type": "string"}
                    }
                }
            },
            "required": ["id", "name"]
        }"#;
        let schema_val: Value = serde_json::from_str(schema).unwrap();
        let start = std::time::Instant::now();
        let grammar = json_schema_to_grammar(&schema_val).unwrap();
        let _compiled = CompiledGrammar::new(&grammar, json_vocab()).unwrap();
        let elapsed = start.elapsed();
        // Must complete in under 10ms.
        assert!(
            elapsed.as_millis() < 10,
            "compilation took {elapsed:?}, expected < 10ms"
        );
    }
}
