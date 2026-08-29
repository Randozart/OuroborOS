pub mod node_state;
pub mod topology;

use anyhow::Result;

pub use node_state::{NodeState, NodeStatus, WorkloadAssignment};
pub use topology::{ClusterTopology, NodeEntry, WorkloadEntry};

/// Serialize any Beast-capable value to an S-expression string.
///
/// Beast format is nested lists of atoms: `(tag child1 child2 ...)`.
/// Atoms are strings, integers, floats, or booleans.
pub fn serialize(node: &impl serde::Serialize) -> Result<String> {
    let value = serde_json::to_value(node)?;
    let sexpr = json_to_sexpr(&value);
    Ok(sexpr)
}

/// Deserialize an S-expression string into a Beast-capable value.
pub fn deserialize<T: serde::de::DeserializeOwned>(sexpr: &str) -> Result<T> {
    let value = sexpr_to_json(sexpr)?;
    let json = serde_json::to_string(&value)?;
    let node = serde_json::from_str(&json)?;
    Ok(node)
}

/// Convert a JSON value to a nested-list S-expression string.
fn json_to_sexpr(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut parts = Vec::new();
            // Sort keys for deterministic output
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                let val = &map[key];
                parts.push(format!("({} {})", key, json_to_sexpr(val)));
            }
            format!("({})", parts.join(" "))
        }
        serde_json::Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(json_to_sexpr).collect();
            format!("({})", parts.join(" "))
        }
        serde_json::Value::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(f) = n.as_f64() {
                format!("{}", f)
            } else {
                n.to_string()
            }
        }
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

/// Convert an S-expression string to a JSON value.
fn sexpr_to_json(sexpr: &str) -> Result<serde_json::Value> {
    let tokens = tokenize(sexpr)?;
    let (value, _) = parse_list(&tokens, 0)?;
    Ok(value)
}

fn tokenize(input: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '(' | ')' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(c.to_string());
            }
            '"' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                let mut s = String::from("\"");
                loop {
                    match chars.next() {
                        Some('\\') => {
                            s.push('\\');
                            if let Some(n) = chars.next() {
                                s.push(n);
                            }
                        }
                        Some('"') => {
                            s.push('"');
                            break;
                        }
                        Some(c) => s.push(c),
                        None => anyhow::bail!("Unterminated string in Beast"),
                    }
                }
                tokens.push(s);
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    Ok(tokens)
}

fn parse_list(tokens: &[String], pos: usize) -> Result<(serde_json::Value, usize)> {
    if tokens.get(pos).map(|t| t.as_str()) != Some("(") {
        anyhow::bail!("Expected '(' at position {}", pos);
    }

    let mut items = Vec::new();
    let mut i = pos + 1;

    while i < tokens.len() {
        match tokens[i].as_str() {
            ")" => {
                return Ok((serde_json::Value::Array(items), i + 1));
            }
            "(" => {
                let (nested, next) = parse_list(tokens, i)?;
                items.push(nested);
                i = next;
            }
            tok => {
                items.push(parse_atom(tok)?);
                i += 1;
            }
        }
    }
    anyhow::bail!("Unbalanced parentheses in Beast")
}

fn parse_atom(tok: &str) -> Result<serde_json::Value> {
    if tok.starts_with('"') && tok.ends_with('"') {
        let inner = tok[1..tok.len() - 1].to_string();
        return Ok(serde_json::Value::String(inner));
    }
    if tok == "true" {
        return Ok(serde_json::Value::Bool(true));
    }
    if tok == "false" {
        return Ok(serde_json::Value::Bool(false));
    }
    if tok == "null" {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(n) = tok.parse::<i64>() {
        return Ok(serde_json::Value::Number(n.into()));
    }
    if let Ok(f) = tok.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(f) {
            return Ok(serde_json::Value::Number(num));
        }
        // f64 is NaN or infinite — store as string
        return Ok(serde_json::Value::String(tok.to_string()));
    }
    Ok(serde_json::Value::String(tok.to_string()))
}
