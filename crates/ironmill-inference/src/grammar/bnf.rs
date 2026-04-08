//! Grammar representation and BNF parser.
//!
//! A [`Grammar`] is a set of named rules written in a simplified BNF
//! notation. The first rule is the start symbol.
//!
//! # Supported syntax
//!
//! ```text
//! rule_name ::= expression
//! ```
//!
//! - Terminals: `"text"` or `'text'`
//! - Character classes: `[0-9]`, `[a-zA-Z]`, `[^"\\]`
//! - Non-terminals: `<rule_name>`
//! - Alternation: `|`
//! - Concatenation: whitespace-separated
//! - Repetition: `*` (zero or more), `+` (one or more), `?` (optional)
//! - Grouping: `( ... )`

use std::fmt;

// ── AST types ────────────────────────────────────────────────────

/// A grammar element (AST node).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    /// Literal string, e.g. `"true"`.
    Literal(String),
    /// Character class, e.g. `[0-9]` or `[^"\\]`.
    CharClass {
        /// Inclusive character ranges in this class.
        ranges: Vec<(char, char)>,
        /// Whether this class is negated (e.g. `[^...]`).
        negated: bool,
    },
    /// Reference to a named rule, e.g. `<value>`.
    RuleRef(String),
    /// Sequence of elements (concatenation).
    Sequence(Vec<Element>),
    /// One of several alternatives.
    Alternation(Vec<Element>),
    /// Zero or more repetitions.
    ZeroOrMore(Box<Element>),
    /// One or more repetitions.
    OneOrMore(Box<Element>),
    /// Optional (zero or one).
    Optional(Box<Element>),
}

/// A named grammar rule.
#[derive(Debug, Clone)]
pub struct Rule {
    /// The rule's identifier (e.g. `"value"`).
    pub name: String,
    /// The expression defining the rule body.
    pub expr: Element,
}

/// A complete grammar with named rules and a start symbol.
#[derive(Debug, Clone)]
pub struct Grammar {
    /// All named rules in this grammar.
    pub rules: Vec<Rule>,
    /// The rule name to start parsing from.
    pub start_rule: String,
}

// ── Errors ───────────────────────────────────────────────────────

/// Errors that can occur during grammar parsing.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    /// No rules found in the input.
    EmptyGrammar,
    /// Unexpected character during parsing.
    UnexpectedChar {
        /// Byte position where the unexpected character was found.
        pos: usize,
        /// The unexpected character.
        ch: char,
    },
    /// Unexpected end of input.
    UnexpectedEnd,
    /// Unterminated string literal.
    UnterminatedString {
        /// Byte position where the string literal started.
        pos: usize,
    },
    /// Unterminated character class.
    UnterminatedCharClass {
        /// Byte position where the character class started.
        pos: usize,
    },
    /// Unterminated group.
    UnterminatedGroup {
        /// Byte position where the group started.
        pos: usize,
    },
    /// Empty alternation branch.
    EmptyBranch,
    /// Unterminated rule reference.
    UnterminatedRuleRef {
        /// Byte position where the rule reference started.
        pos: usize,
    },
    /// Reference to an undefined rule name.
    UndefinedRule(String),
    /// Line does not contain `::=` separator.
    MalformedRule(String),
    /// Trailing content after a valid rule expression.
    TrailingContent {
        /// Byte position where trailing content starts.
        pos: usize,
    },
    /// Duplicate rule name.
    DuplicateRule(String),
    /// Unknown or unsupported JSON Schema type.
    UnsupportedType(String),
    /// Token ID is out of range for the vocabulary.
    InvalidTokenId {
        /// The out-of-range token ID.
        token_id: u32,
        /// Size of the vocabulary.
        vocab_size: usize,
    },
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGrammar => write!(f, "grammar contains no rules"),
            Self::UnexpectedChar { pos, ch } => {
                write!(f, "unexpected character '{ch}' at position {pos}")
            }
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::UnterminatedString { pos } => {
                write!(f, "unterminated string literal starting at position {pos}")
            }
            Self::UnterminatedCharClass { pos } => {
                write!(f, "unterminated character class starting at position {pos}")
            }
            Self::UnterminatedGroup { pos } => {
                write!(f, "unterminated group starting at position {pos}")
            }
            Self::EmptyBranch => write!(f, "empty alternation branch"),
            Self::UnterminatedRuleRef { pos } => {
                write!(f, "unterminated rule reference starting at position {pos}")
            }
            Self::UndefinedRule(name) => {
                write!(f, "undefined rule: {name}")
            }
            Self::MalformedRule(line) => {
                write!(f, "malformed rule (missing '::='): {line}")
            }
            Self::TrailingContent { pos } => {
                write!(f, "trailing content after expression at position {pos}")
            }
            Self::DuplicateRule(name) => {
                write!(f, "duplicate rule name: {name}")
            }
            Self::UnsupportedType(ty) => {
                write!(f, "unsupported JSON Schema type: {ty}")
            }
            Self::InvalidTokenId {
                token_id,
                vocab_size,
            } => {
                write!(
                    f,
                    "token_id {token_id} out of range for vocabulary of size {vocab_size}"
                )
            }
        }
    }
}

impl std::error::Error for GrammarError {}

// ── Grammar constructors ─────────────────────────────────────────

impl Grammar {
    /// Parse a grammar from BNF notation.
    ///
    /// The first rule defined becomes the start symbol.
    /// Returns an error for non-empty, non-comment lines that lack `::=`.
    pub fn from_bnf(input: &str) -> Result<Self, GrammarError> {
        let mut rules = Vec::new();

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((name, body)) = line.split_once("::=") {
                let name = name.trim().to_string();
                let body = body.trim();
                let expr = parse_expression(body)?;
                rules.push(Rule { name, expr });
            } else {
                return Err(GrammarError::MalformedRule(line.to_string()));
            }
        }

        if rules.is_empty() {
            return Err(GrammarError::EmptyGrammar);
        }
        let start_rule = rules[0].name.clone();
        Ok(Grammar { rules, start_rule })
    }
}

// ── Parser internals ─────────────────────────────────────────────

/// Cursor over a string for the recursive-descent parser.
struct Cursor<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.input.len()
    }
}

/// Parse a full expression (entry point).
///
/// Returns an error if there is trailing content after the expression.
fn parse_expression(input: &str) -> Result<Element, GrammarError> {
    let mut cursor = Cursor::new(input);
    let expr = parse_alternation(&mut cursor)?;
    cursor.skip_whitespace();
    if !cursor.at_end() {
        return Err(GrammarError::TrailingContent { pos: cursor.pos });
    }
    Ok(expr)
}

/// Parse alternation: `sequence ("|" sequence)*`
fn parse_alternation(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    let mut branches = vec![parse_sequence(cursor)?];

    loop {
        cursor.skip_whitespace();
        if cursor.peek() == Some('|') {
            cursor.advance();
            branches.push(parse_sequence(cursor)?);
        } else {
            break;
        }
    }

    if branches.len() == 1 {
        Ok(branches.pop().unwrap())
    } else {
        Ok(Element::Alternation(branches))
    }
}

/// Parse sequence: `postfix+`
fn parse_sequence(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    let mut elements = Vec::new();

    loop {
        cursor.skip_whitespace();
        match cursor.peek() {
            None | Some('|') | Some(')') => break,
            _ => elements.push(parse_postfix(cursor)?),
        }
    }

    if elements.len() == 1 {
        Ok(elements.pop().unwrap())
    } else {
        Ok(Element::Sequence(elements))
    }
}

/// Parse postfix: `atom ("*" | "+" | "?")?`
fn parse_postfix(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    let atom = parse_atom(cursor)?;

    match cursor.peek() {
        Some('*') => {
            cursor.advance();
            Ok(Element::ZeroOrMore(Box::new(atom)))
        }
        Some('+') => {
            cursor.advance();
            Ok(Element::OneOrMore(Box::new(atom)))
        }
        Some('?') => {
            cursor.advance();
            Ok(Element::Optional(Box::new(atom)))
        }
        _ => Ok(atom),
    }
}

/// Parse an atom: literal, char class, rule ref, or grouped expression.
fn parse_atom(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    cursor.skip_whitespace();

    match cursor.peek() {
        Some('"') | Some('\'') => parse_literal(cursor),
        Some('[') => parse_char_class(cursor),
        Some('<') => parse_rule_ref(cursor),
        Some('(') => parse_group(cursor),
        Some(ch) => Err(GrammarError::UnexpectedChar {
            pos: cursor.pos,
            ch,
        }),
        None => Err(GrammarError::UnexpectedEnd),
    }
}

/// Parse a quoted string literal.
fn parse_literal(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    let start = cursor.pos;
    let quote = cursor.advance().unwrap(); // opening quote
    let mut value = String::new();

    loop {
        match cursor.advance() {
            Some(ch) if ch == quote => return Ok(Element::Literal(value)),
            Some('\\') => match cursor.advance() {
                Some('n') => value.push('\n'),
                Some('t') => value.push('\t'),
                Some('\\') => value.push('\\'),
                Some('"') => value.push('"'),
                Some('\'') => value.push('\''),
                Some(ch) => {
                    value.push('\\');
                    value.push(ch);
                }
                None => return Err(GrammarError::UnterminatedString { pos: start }),
            },
            Some(ch) => value.push(ch),
            None => return Err(GrammarError::UnterminatedString { pos: start }),
        }
    }
}

/// Parse a character class like `[0-9]` or `[^"\\]`.
fn parse_char_class(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    let start = cursor.pos;
    cursor.advance(); // consume '['

    let negated = cursor.peek() == Some('^');
    if negated {
        cursor.advance();
    }

    let mut ranges: Vec<(char, char)> = Vec::new();

    loop {
        match cursor.peek() {
            Some(']') => {
                cursor.advance();
                return Ok(Element::CharClass { ranges, negated });
            }
            Some('\\') => {
                cursor.advance();
                let ch = parse_escape_char(cursor)?;
                if cursor.peek() == Some('-') {
                    cursor.advance();
                    let end = if cursor.peek() == Some('\\') {
                        cursor.advance();
                        parse_escape_char(cursor)?
                    } else {
                        cursor
                            .advance()
                            .ok_or(GrammarError::UnterminatedCharClass { pos: start })?
                    };
                    ranges.push((ch, end));
                } else {
                    ranges.push((ch, ch));
                }
            }
            Some(ch) => {
                cursor.advance();
                if cursor.peek() == Some('-') {
                    cursor.advance();
                    if cursor.peek() == Some(']') {
                        // Trailing dash: treat both as literal characters.
                        ranges.push((ch, ch));
                        ranges.push(('-', '-'));
                    } else {
                        let end = if cursor.peek() == Some('\\') {
                            cursor.advance();
                            parse_escape_char(cursor)?
                        } else {
                            cursor
                                .advance()
                                .ok_or(GrammarError::UnterminatedCharClass { pos: start })?
                        };
                        ranges.push((ch, end));
                    }
                } else {
                    ranges.push((ch, ch));
                }
            }
            None => return Err(GrammarError::UnterminatedCharClass { pos: start }),
        }
    }
}

/// Parse an escape character inside a character class.
fn parse_escape_char(cursor: &mut Cursor<'_>) -> Result<char, GrammarError> {
    match cursor.advance() {
        Some('n') => Ok('\n'),
        Some('t') => Ok('\t'),
        Some('\\') => Ok('\\'),
        Some('"') => Ok('"'),
        Some('\'') => Ok('\''),
        Some(']') => Ok(']'),
        Some('[') => Ok('['),
        Some('-') => Ok('-'),
        Some(ch) => Ok(ch),
        None => Err(GrammarError::UnexpectedEnd),
    }
}

/// Parse a rule reference like `<value>`.
fn parse_rule_ref(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    let start = cursor.pos;
    cursor.advance(); // consume '<'
    let mut name = String::new();

    loop {
        match cursor.advance() {
            Some('>') => return Ok(Element::RuleRef(name)),
            Some(ch) => name.push(ch),
            None => return Err(GrammarError::UnterminatedRuleRef { pos: start }),
        }
    }
}

/// Parse a grouped expression `( ... )`.
fn parse_group(cursor: &mut Cursor<'_>) -> Result<Element, GrammarError> {
    let start = cursor.pos;
    cursor.advance(); // consume '('
    let expr = parse_alternation(cursor)?;
    cursor.skip_whitespace();
    match cursor.advance() {
        Some(')') => Ok(expr),
        _ => Err(GrammarError::UnterminatedGroup { pos: start }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_parse_literal() {
        let g = Grammar::from_bnf(r#"root ::= "hello""#).unwrap();
        assert_eq!(g.rules.len(), 1);
        assert_eq!(g.rules[0].name, "root");
        assert_eq!(g.rules[0].expr, Element::Literal("hello".into()));
    }

    #[test]
    fn grammar_parse_alternation() {
        let g = Grammar::from_bnf(r#"bool ::= "true" | "false""#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::Alternation(vec![
                Element::Literal("true".into()),
                Element::Literal("false".into()),
            ])
        );
    }

    #[test]
    fn grammar_parse_sequence() {
        let g = Grammar::from_bnf(r#"pair ::= "a" "b""#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::Sequence(vec![
                Element::Literal("a".into()),
                Element::Literal("b".into()),
            ])
        );
    }

    #[test]
    fn grammar_parse_char_class() {
        let g = Grammar::from_bnf(r#"digit ::= [0-9]"#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::CharClass {
                ranges: vec![('0', '9')],
                negated: false,
            }
        );
    }

    #[test]
    fn grammar_parse_negated_char_class() {
        let g = Grammar::from_bnf(r#"non_quote ::= [^"\\]"#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::CharClass {
                ranges: vec![('"', '"'), ('\\', '\\')],
                negated: true,
            }
        );
    }

    #[test]
    fn grammar_parse_rule_ref() {
        let g = Grammar::from_bnf(
            r#"start ::= <value>
value ::= "x""#,
        )
        .unwrap();
        assert_eq!(g.rules[0].expr, Element::RuleRef("value".into()));
    }

    #[test]
    fn grammar_parse_repetition() {
        let g = Grammar::from_bnf(r#"digits ::= [0-9]+"#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::OneOrMore(Box::new(Element::CharClass {
                ranges: vec![('0', '9')],
                negated: false,
            }))
        );
    }

    #[test]
    fn grammar_parse_optional() {
        let g = Grammar::from_bnf(r#"sign ::= "-"?"#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::Optional(Box::new(Element::Literal("-".into())))
        );
    }

    #[test]
    fn grammar_parse_zero_or_more() {
        let g = Grammar::from_bnf(r#"spaces ::= " "*"#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::ZeroOrMore(Box::new(Element::Literal(" ".into())))
        );
    }

    #[test]
    fn grammar_parse_grouping() {
        let g = Grammar::from_bnf(r#"num ::= "-"? [0-9]+ ("." [0-9]+)?"#).unwrap();
        assert!(matches!(g.rules[0].expr, Element::Sequence(_)));
    }

    #[test]
    fn grammar_parse_multiple_rules() {
        let input = r#"
value ::= <bool> | <number>
bool ::= "true" | "false"
number ::= [0-9]+
"#;
        let g = Grammar::from_bnf(input).unwrap();
        assert_eq!(g.rules.len(), 3);
        assert_eq!(g.start_rule, "value");
    }

    #[test]
    fn grammar_parse_empty_input() {
        let err = Grammar::from_bnf("").unwrap_err();
        assert_eq!(err, GrammarError::EmptyGrammar);
    }

    #[test]
    fn grammar_parse_comments_skipped() {
        let input = r#"
# This is a comment
root ::= "ok"
# Another comment
"#;
        let g = Grammar::from_bnf(input).unwrap();
        assert_eq!(g.rules.len(), 1);
    }

    #[test]
    fn grammar_parse_multi_range_char_class() {
        let g = Grammar::from_bnf(r#"alnum ::= [a-zA-Z0-9]"#).unwrap();
        assert_eq!(
            g.rules[0].expr,
            Element::CharClass {
                ranges: vec![('a', 'z'), ('A', 'Z'), ('0', '9')],
                negated: false,
            }
        );
    }
}
