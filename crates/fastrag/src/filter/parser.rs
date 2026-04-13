use crate::filter::ast::FilterExpr;
use fastrag_store::schema::TypedValue;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FilterParseError {
    #[error("unexpected token at position {pos}: {message}")]
    Unexpected { pos: usize, message: String },
    #[error("unterminated string starting at position {pos}")]
    UnterminatedString { pos: usize },
    #[error("empty filter expression")]
    Empty,
}

/// Parse a filter expression from the string syntax.
pub fn parse(input: &str) -> Result<FilterExpr, FilterParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(FilterParseError::Empty);
    }

    // Legacy detection: `k=v,k=v` without any keyword tokens
    if is_legacy_kv(trimmed) {
        return parse_legacy_kv(trimmed);
    }

    let tokens = tokenize(trimmed)?;
    if tokens.is_empty() {
        return Err(FilterParseError::Empty);
    }
    let mut pos = 0;
    let expr = parse_or(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(FilterParseError::Unexpected {
            pos: tokens[pos].pos,
            message: format!("unexpected token '{}'", tokens[pos].text),
        });
    }
    Ok(expr)
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Token {
    text: String,
    pos: usize,
}

fn tokenize(input: &str) -> Result<Vec<Token>, FilterParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // skip whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        let start = i;

        // single-char operators / parens
        match chars[i] {
            '(' | ')' | ',' => {
                tokens.push(Token {
                    text: chars[i].to_string(),
                    pos: start,
                });
                i += 1;
                continue;
            }
            _ => {}
        }

        // two-char operators
        if i + 1 < chars.len() {
            let two: String = chars[i..=i + 1].iter().collect();
            if two == "!=" || two == ">=" || two == "<=" {
                tokens.push(Token {
                    text: two,
                    pos: start,
                });
                i += 2;
                continue;
            }
        }

        // single-char comparison operators
        if chars[i] == '=' || chars[i] == '>' || chars[i] == '<' {
            tokens.push(Token {
                text: chars[i].to_string(),
                pos: start,
            });
            i += 1;
            continue;
        }

        // quoted string
        if chars[i] == '"' {
            i += 1;
            let mut s = String::new();
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 1;
                    s.push(chars[i]);
                } else {
                    s.push(chars[i]);
                }
                i += 1;
            }
            if i >= chars.len() {
                return Err(FilterParseError::UnterminatedString { pos: start });
            }
            i += 1; // skip closing quote
            tokens.push(Token {
                text: format!("\"{s}\""),
                pos: start,
            });
            continue;
        }

        // bare word / number / date / dotted identifier
        let mut word = String::new();
        while i < chars.len()
            && !chars[i].is_whitespace()
            && !matches!(chars[i], '(' | ')' | ',' | '=' | '!' | '>' | '<' | '"')
        {
            word.push(chars[i]);
            i += 1;
        }
        if !word.is_empty() {
            tokens.push(Token {
                text: word,
                pos: start,
            });
        } else {
            // Unrecognized character that was not consumed by any branch above
            // (e.g. a bare `!` not followed by `=`). Return a parse error rather
            // than looping infinitely.
            return Err(FilterParseError::Unexpected {
                pos: start,
                message: format!("unexpected character '{}'", chars[i]),
            });
        }
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Recursive descent parser
// ---------------------------------------------------------------------------

fn parse_or(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, FilterParseError> {
    let mut left = parse_and(tokens, pos)?;

    while *pos < tokens.len() && tokens[*pos].text.eq_ignore_ascii_case("OR") {
        *pos += 1;
        let right = parse_and(tokens, pos)?;
        left = match left {
            FilterExpr::Or(mut v) => {
                v.push(right);
                FilterExpr::Or(v)
            }
            _ => FilterExpr::Or(vec![left, right]),
        };
    }

    Ok(left)
}

fn parse_and(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, FilterParseError> {
    let mut left = parse_not(tokens, pos)?;

    while *pos < tokens.len() && tokens[*pos].text.eq_ignore_ascii_case("AND") {
        *pos += 1;
        let right = parse_not(tokens, pos)?;
        left = match left {
            FilterExpr::And(mut v) => {
                v.push(right);
                FilterExpr::And(v)
            }
            _ => FilterExpr::And(vec![left, right]),
        };
    }

    Ok(left)
}

fn parse_not(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, FilterParseError> {
    if *pos < tokens.len() && tokens[*pos].text.eq_ignore_ascii_case("NOT") {
        // Peek ahead: if next tokens are `field NOT IN (...)`, this is not a unary NOT
        // but we handle NOT IN inside parse_primary/comparison, so only consume NOT
        // if the next token isn't a field followed by an operator.
        // Actually, top-level NOT is always unary. `field NOT IN` is handled as
        // a two-token operator inside comparison. So we just need to check: is the
        // token after NOT a valid start of a comparison or sub-expression?
        // If the next-next token is NOT (as in `NOT field NOT IN`), recurse.
        *pos += 1;
        let inner = parse_not(tokens, pos)?;
        return Ok(FilterExpr::Not(Box::new(inner)));
    }
    parse_primary(tokens, pos)
}

fn parse_primary(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, FilterParseError> {
    if *pos >= tokens.len() {
        return Err(FilterParseError::Unexpected {
            pos: 0,
            message: "unexpected end of expression".to_string(),
        });
    }

    // Parenthesized sub-expression
    if tokens[*pos].text == "(" {
        *pos += 1;
        let expr = parse_or(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos].text != ")" {
            let p = if *pos < tokens.len() {
                tokens[*pos].pos
            } else {
                tokens.last().map_or(0, |t| t.pos)
            };
            return Err(FilterParseError::Unexpected {
                pos: p,
                message: "expected ')'".to_string(),
            });
        }
        *pos += 1;
        return Ok(expr);
    }

    // Must be a comparison: field op value
    parse_comparison(tokens, pos)
}

fn parse_comparison(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, FilterParseError> {
    // field
    let field_tok = &tokens[*pos];
    let field = field_tok.text.clone();
    *pos += 1;

    if *pos >= tokens.len() {
        return Err(FilterParseError::Unexpected {
            pos: field_tok.pos + field_tok.text.len(),
            message: "expected operator after field".to_string(),
        });
    }

    let op_tok = &tokens[*pos];
    let op = op_tok.text.to_ascii_uppercase();

    match op.as_str() {
        "=" => {
            *pos += 1;
            let value = parse_value(tokens, pos)?;
            Ok(FilterExpr::Eq { field, value })
        }
        "!=" => {
            *pos += 1;
            let value = parse_value(tokens, pos)?;
            Ok(FilterExpr::Neq { field, value })
        }
        ">" => {
            *pos += 1;
            let value = parse_value(tokens, pos)?;
            Ok(FilterExpr::Gt { field, value })
        }
        ">=" => {
            *pos += 1;
            let value = parse_value(tokens, pos)?;
            Ok(FilterExpr::Gte { field, value })
        }
        "<" => {
            *pos += 1;
            let value = parse_value(tokens, pos)?;
            Ok(FilterExpr::Lt { field, value })
        }
        "<=" => {
            *pos += 1;
            let value = parse_value(tokens, pos)?;
            Ok(FilterExpr::Lte { field, value })
        }
        "IN" => {
            *pos += 1;
            let values = parse_value_list(tokens, pos)?;
            Ok(FilterExpr::In { field, values })
        }
        "NOT" => {
            // NOT IN
            *pos += 1;
            if *pos >= tokens.len() || !tokens[*pos].text.eq_ignore_ascii_case("IN") {
                return Err(FilterParseError::Unexpected {
                    pos: op_tok.pos,
                    message: "expected 'IN' after 'NOT'".to_string(),
                });
            }
            *pos += 1;
            let values = parse_value_list(tokens, pos)?;
            Ok(FilterExpr::NotIn { field, values })
        }
        "CONTAINS" => {
            *pos += 1;
            let value = parse_value(tokens, pos)?;
            Ok(FilterExpr::Contains { field, value })
        }
        "ALL" => {
            *pos += 1;
            let values = parse_value_list(tokens, pos)?;
            Ok(FilterExpr::All { field, values })
        }
        _ => Err(FilterParseError::Unexpected {
            pos: op_tok.pos,
            message: format!("expected operator, found '{}'", op_tok.text),
        }),
    }
}

fn parse_value_list(
    tokens: &[Token],
    pos: &mut usize,
) -> Result<Vec<TypedValue>, FilterParseError> {
    if *pos >= tokens.len() || tokens[*pos].text != "(" {
        let p = if *pos < tokens.len() {
            tokens[*pos].pos
        } else {
            tokens.last().map_or(0, |t| t.pos)
        };
        return Err(FilterParseError::Unexpected {
            pos: p,
            message: "expected '(' for value list".to_string(),
        });
    }
    *pos += 1; // skip '('

    let mut values = Vec::new();
    if *pos < tokens.len() && tokens[*pos].text != ")" {
        values.push(parse_value(tokens, pos)?);
        while *pos < tokens.len() && tokens[*pos].text == "," {
            *pos += 1; // skip ','
            values.push(parse_value(tokens, pos)?);
        }
    }

    if *pos >= tokens.len() || tokens[*pos].text != ")" {
        let p = if *pos < tokens.len() {
            tokens[*pos].pos
        } else {
            tokens.last().map_or(0, |t| t.pos)
        };
        return Err(FilterParseError::Unexpected {
            pos: p,
            message: "expected ')' to close value list".to_string(),
        });
    }
    *pos += 1; // skip ')'

    Ok(values)
}

fn parse_value(tokens: &[Token], pos: &mut usize) -> Result<TypedValue, FilterParseError> {
    if *pos >= tokens.len() {
        let p = tokens.last().map_or(0, |t| t.pos + t.text.len());
        return Err(FilterParseError::Unexpected {
            pos: p,
            message: "expected value".to_string(),
        });
    }

    let tok = &tokens[*pos];
    *pos += 1;

    let text = &tok.text;

    // Quoted string
    if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        return Ok(TypedValue::String(text[1..text.len() - 1].to_string()));
    }

    // Boolean
    if text.eq_ignore_ascii_case("true") {
        return Ok(TypedValue::Bool(true));
    }
    if text.eq_ignore_ascii_case("false") {
        return Ok(TypedValue::Bool(false));
    }

    // Date: YYYY-MM-DD
    if text.len() == 10
        && text.chars().nth(4) == Some('-')
        && text.chars().nth(7) == Some('-')
        && let Ok(date) = chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d")
    {
        return Ok(TypedValue::Date(date));
    }

    // Numeric
    if let Ok(n) = text.parse::<f64>() {
        return Ok(TypedValue::Numeric(n));
    }

    // Bare word → String
    Ok(TypedValue::String(text.clone()))
}

// ---------------------------------------------------------------------------
// Legacy k=v,k=v detection and parsing
// ---------------------------------------------------------------------------

/// Returns true if input looks like legacy `k=v,k=v` format:
/// no whitespace-separated keywords (AND/OR/NOT), no comparison operators
/// other than `=`, and contains at least one comma or a single `k=v`.
fn is_legacy_kv(input: &str) -> bool {
    // Must not contain whitespace (legacy format has no spaces)
    // unless inside quoted values — but legacy format doesn't use quotes.
    // Simple heuristic: if input contains spaces, it's not legacy.
    if input.contains(' ') || input.contains('\t') {
        return false;
    }

    // Must contain at least one '=' and no multi-char operators
    if !input.contains('=') {
        return false;
    }
    if input.contains("!=") || input.contains(">=") || input.contains("<=") {
        return false;
    }

    // All segments separated by ',' must be k=v
    input.split(',').all(|segment| {
        let parts: Vec<&str> = segment.splitn(2, '=').collect();
        parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty()
    })
}

fn parse_legacy_kv(input: &str) -> Result<FilterExpr, FilterParseError> {
    let pairs: Vec<FilterExpr> = input
        .split(',')
        .map(|segment| {
            let mut parts = segment.splitn(2, '=');
            let field = parts.next().unwrap().to_string();
            let raw_value = parts.next().unwrap();
            let value = parse_raw_value(raw_value);
            FilterExpr::Eq { field, value }
        })
        .collect();

    if pairs.len() == 1 {
        Ok(pairs.into_iter().next().unwrap())
    } else {
        Ok(FilterExpr::And(pairs))
    }
}

/// Parse a raw value string into a TypedValue (shared with legacy parser).
fn parse_raw_value(text: &str) -> TypedValue {
    if text.eq_ignore_ascii_case("true") {
        return TypedValue::Bool(true);
    }
    if text.eq_ignore_ascii_case("false") {
        return TypedValue::Bool(false);
    }
    if text.len() == 10
        && let Ok(date) = chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d")
    {
        return TypedValue::Date(date);
    }
    if let Ok(n) = text.parse::<f64>() {
        return TypedValue::Numeric(n);
    }
    TypedValue::String(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_eq() {
        let expr = parse("severity = HIGH").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Eq {
                field: "severity".to_string(),
                value: TypedValue::String("HIGH".to_string()),
            }
        );
    }

    #[test]
    fn parse_numeric_comparison() {
        let expr = parse("cvss_score >= 7.0").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Gte {
                field: "cvss_score".to_string(),
                value: TypedValue::Numeric(7.0),
            }
        );
    }

    #[test]
    fn parse_neq() {
        let expr = parse("status != resolved").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Neq {
                field: "status".to_string(),
                value: TypedValue::String("resolved".to_string()),
            }
        );
    }

    #[test]
    fn parse_in_operator() {
        let expr = parse("severity IN (HIGH, CRITICAL)").unwrap();
        assert_eq!(
            expr,
            FilterExpr::In {
                field: "severity".to_string(),
                values: vec![
                    TypedValue::String("HIGH".to_string()),
                    TypedValue::String("CRITICAL".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parse_not_in() {
        let expr = parse("status NOT IN (closed, resolved)").unwrap();
        assert_eq!(
            expr,
            FilterExpr::NotIn {
                field: "status".to_string(),
                values: vec![
                    TypedValue::String("closed".to_string()),
                    TypedValue::String("resolved".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parse_contains() {
        let expr = parse("tags CONTAINS rce").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Contains {
                field: "tags".to_string(),
                value: TypedValue::String("rce".to_string()),
            }
        );
    }

    #[test]
    fn parse_all() {
        let expr = parse("tags ALL (rce, remote)").unwrap();
        assert_eq!(
            expr,
            FilterExpr::All {
                field: "tags".to_string(),
                values: vec![
                    TypedValue::String("rce".to_string()),
                    TypedValue::String("remote".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parse_and_or_precedence() {
        // AND binds tighter than OR
        let expr = parse("a = 1 OR b = 2 AND c = 3").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Or(vec![
                FilterExpr::Eq {
                    field: "a".to_string(),
                    value: TypedValue::Numeric(1.0),
                },
                FilterExpr::And(vec![
                    FilterExpr::Eq {
                        field: "b".to_string(),
                        value: TypedValue::Numeric(2.0),
                    },
                    FilterExpr::Eq {
                        field: "c".to_string(),
                        value: TypedValue::Numeric(3.0),
                    },
                ]),
            ])
        );
    }

    #[test]
    fn parse_not() {
        let expr = parse("NOT false_positive = true").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Not(Box::new(FilterExpr::Eq {
                field: "false_positive".to_string(),
                value: TypedValue::Bool(true),
            }))
        );
    }

    #[test]
    fn parse_parentheses() {
        let expr = parse("(a = 1 OR b = 2) AND c = 3").unwrap();
        assert_eq!(
            expr,
            FilterExpr::And(vec![
                FilterExpr::Or(vec![
                    FilterExpr::Eq {
                        field: "a".to_string(),
                        value: TypedValue::Numeric(1.0),
                    },
                    FilterExpr::Eq {
                        field: "b".to_string(),
                        value: TypedValue::Numeric(2.0),
                    },
                ]),
                FilterExpr::Eq {
                    field: "c".to_string(),
                    value: TypedValue::Numeric(3.0),
                },
            ])
        );
    }

    #[test]
    fn parse_quoted_string() {
        let expr = parse(r#"title = "SQL Injection""#).unwrap();
        assert_eq!(
            expr,
            FilterExpr::Eq {
                field: "title".to_string(),
                value: TypedValue::String("SQL Injection".to_string()),
            }
        );
    }

    #[test]
    fn parse_boolean_value() {
        let expr = parse("published = true").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Eq {
                field: "published".to_string(),
                value: TypedValue::Bool(true),
            }
        );
    }

    #[test]
    fn parse_date_value() {
        let expr = parse("remediation_due <= 2024-06-01").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Lte {
                field: "remediation_due".to_string(),
                value: TypedValue::Date(chrono::NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
            }
        );
    }

    #[test]
    fn parse_legacy_kv_format() {
        let expr = parse("severity=high,status=open").unwrap();
        assert_eq!(
            expr,
            FilterExpr::And(vec![
                FilterExpr::Eq {
                    field: "severity".to_string(),
                    value: TypedValue::String("high".to_string()),
                },
                FilterExpr::Eq {
                    field: "status".to_string(),
                    value: TypedValue::String("open".to_string()),
                },
            ])
        );
    }

    #[test]
    fn parse_empty_is_error() {
        assert_eq!(parse(""), Err(FilterParseError::Empty));
        assert_eq!(parse("   "), Err(FilterParseError::Empty));
    }

    #[test]
    fn parse_error_has_position() {
        let err = parse("severity >= ").unwrap_err();
        match err {
            FilterParseError::Unexpected { pos, .. } => {
                assert!(pos > 0, "error position should be > 0, got {pos}");
            }
            other => panic!("expected Unexpected error, got: {other:?}"),
        }
    }

    #[test]
    fn parse_dotted_field() {
        let expr = parse("cvss.score >= 7.0").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Gte {
                field: "cvss.score".to_string(),
                value: TypedValue::Numeric(7.0),
            }
        );
    }
}
