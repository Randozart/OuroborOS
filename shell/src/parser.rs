/// Lexical tokens produced by the shell lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Identifier: node names, properties, workload names
    Ident(String),
    /// Dot/period separator: `n3.power?` or `n3 assign branch_sort.`
    Dot,
    /// Question mark: query/discovery
    Question,
    /// Colon: shorthand separator
    Colon,
    /// Whitespace (discarded in most contexts)
    Whitespace,
    /// End of input
    Eof,
}

/// Lex a shell input string into tokens.
pub fn lex(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        match chars[i] {
            ' ' | '\t' => {
                tokens.push(Token::Whitespace);
                i += 1;
            }
            '.' => {
                tokens.push(Token::Dot);
                i += 1;
            }
            '?' => {
                tokens.push(Token::Question);
                i += 1;
            }
            ':' => {
                tokens.push(Token::Colon);
                i += 1;
            }
            '"' => {
                i += 1;
                let start = i;
                while i < len && chars[i] != '"' {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(word));
                if i < len {
                    i += 1; // skip closing quote
                }
            }
            _ => {
                let start = i;
                while i < len
                    && !matches!(
                        chars[i],
                        ' ' | '\t' | '.' | '?' | ';' | '"'
                    )
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                if !word.is_empty() {
                    tokens.push(Token::Ident(word));
                }
            }
        }
    }

    tokens.push(Token::Eof);
    tokens
}

/// Strip whitespace and eof tokens from a token stream.
pub fn strip_whitespace(tokens: Vec<Token>) -> Vec<Token> {
    tokens
        .into_iter()
        .filter(|t| *t != Token::Whitespace && *t != Token::Eof)
        .collect()
}

/// Parsed command types the shell understands.
#[derive(Debug, Clone)]
pub enum Command {
    /// Bare `?` — cluster summary
    ClusterSummary,
    /// `cluster?` — cluster summary
    ClusterQuery,
    /// `n3?` — node discovery
    NodeQuery { node: String },
    /// `n3.power?` — deep property query
    PropertyQuery { node: String, property: String },
    /// `power?` — query context's property
    ContextPropertyQuery { property: String },
    /// `cluster.active?` — bulk query
    BulkQuery { filter: String },
    /// `n3` — set context to node
    SetContext { node: String },
    /// `cluster` — reset context to cluster
    ResetContext,
    /// `n3 assign branch_sort.` — proposition
    AssignProposition { node: String, workload: String },
    /// `n3休眠.` — power state change
    PowerState { node: String, sleeping: bool },
    /// `budget 400w.` — set energy budget
    SetBudget { watts: u32 },
    /// `probe.` — probe all nodes
    Probe,
    /// `deploy.` — deploy node-agent
    Deploy,
    /// `deploy shards.` — push model shards (checksum-aware) to nodes
    DeployShards,
    /// `save.` — save cluster state
    Save,
    /// `load.` — load cluster state
    Load,
    /// `generate <text>.` — run BitNet generation on target nodes
    Generate { prompt: String },
    /// `shards.` — show pipeline plan + activation transport probe
    ShardStatus,
    /// `discover. [cidr] [port]` — sweep subnet for agents, absorb them
    Discover { cidr: Option<String>, port: Option<u16> },
    /// `register.` — register a node via probe
    Register,
    /// `unregister n3.` — unregister a node
    Unregister { node: String },
    /// `tasks.` — show task queue status
    Tasks,
    /// `poetry on.` / `poetry off.`
    Poetry { enabled: bool },
    /// `cluster?` with assignment check
    AssignCheck { node: String, workload: String },
    /// Unknown command
    Unknown(String),
}

/// Parse a stripped token stream into a Command.
pub fn parse(tokens: &[Token]) -> Command {
    // Filter out whitespace for easier matching
    let toks: Vec<&Token> = tokens
        .iter()
        .filter(|t| **t != Token::Whitespace && **t != Token::Eof)
        .collect();

    if toks.is_empty() {
        return Command::ClusterSummary;
    }

    match toks.as_slice() {
        // Bare `?` — cluster summary
        [Token::Question] => Command::ClusterSummary,

        // `cluster?`
        [Token::Ident(c), Token::Question] if c == "cluster" => Command::ClusterQuery,

        // `cluster.active?`, `cluster.idle?`, `cluster.power?`
        [Token::Ident(c), Token::Dot, Token::Ident(prop), Token::Question]
            if c == "cluster" =>
        {
            Command::BulkQuery {
                filter: prop.clone(),
            }
        }

        // `branch_sort on?` — workload assignment check
        [Token::Ident(wl), Token::Ident(pred), Token::Question] if pred == "on" => {
            Command::AssignCheck {
                node: String::new(),
                workload: wl.clone(),
            }
        }

        // `n3?` — node discovery
        [Token::Ident(name), Token::Question] if name.starts_with('n') => {
            Command::NodeQuery {
                node: name.clone(),
            }
        }

        // `n3.power?`, `n3.thermal?`, etc.
        [Token::Ident(name), Token::Dot, Token::Ident(prop), Token::Question]
            if name.starts_with('n') =>
        {
            Command::PropertyQuery {
                node: name.clone(),
                property: prop.clone(),
            }
        }

        // `n3 assign branch_sort?` — assignment check
        [Token::Ident(name), Token::Ident(pred), Token::Ident(wl), Token::Question]
            if name.starts_with('n') && pred == "assign" =>
        {
            Command::AssignCheck {
                node: name.clone(),
                workload: wl.clone(),
            }
        }

        // `n3 assign branch_sort.` — proposition
        [Token::Ident(name), Token::Ident(pred), Token::Ident(wl), Token::Dot]
            if name.starts_with('n') && pred == "assign" =>
        {
            Command::AssignProposition {
                node: name.clone(),
                workload: wl.clone(),
            }
        }

        // `n3休眠.` — power state
        [Token::Ident(name), Token::Ident(state), Token::Dot]
            if name.starts_with('n')
                && (state == "休眠" || state == "sleep") =>
        {
            Command::PowerState {
                node: name.clone(),
                sleeping: true,
            }
        }

        // `budget 400w.` or `budget 400.`
        [Token::Ident(b), Token::Ident(val), Token::Dot] if b == "budget" => {
            let watts_str = val.trim_end_matches('w').trim_end_matches('W');
            let watts = watts_str.parse::<u32>().unwrap_or(0);
            Command::SetBudget { watts }
        }

        // `poetry on.` / `poetry off.`
        [Token::Ident(p), Token::Ident(val), Token::Dot] if p == "poetry" => {
            Command::Poetry {
                enabled: val == "on",
            }
        }

        // `probe.`
        [Token::Ident(p), Token::Dot] if p == "probe" => Command::Probe,

        // `deploy.`
        [Token::Ident(d), Token::Dot] if d == "deploy" => Command::Deploy,

        // `save.`
        [Token::Ident(s), Token::Dot] if s == "save" => Command::Save,

        // `load.`
        [Token::Ident(l), Token::Dot] if l == "load" => Command::Load,

        // `cluster` — reset context
        [Token::Ident(c)] if c == "cluster" => Command::ResetContext,

        // `power?`, `thermal?`, etc. — bare property query on context
        [Token::Ident(prop), Token::Question] => Command::ContextPropertyQuery {
            property: prop.clone(),
        },

        // `n3` — set context
        [Token::Ident(name)] if name.starts_with('n') => Command::SetContext {
            node: name.clone(),
        },

        _ => Command::Unknown(format!("{:?}", toks)),
    }
}

/// Convenience: lex + strip whitespace + parse.
pub fn interpret(input: &str) -> Command {
    let trimmed = input.trim();
    // Raw-string shortcut: prompts may contain any characters except a trailing '.'
    if let Some(rest) = trimmed.strip_prefix("generate ") {
        let prompt = rest.strip_suffix('.').unwrap_or(rest).trim().to_string();
        return Command::Generate { prompt };
    }
    if trimmed == "shards." || trimmed.starts_with("shards ") {
        return Command::ShardStatus;
    }
    if trimmed.starts_with("deploy shards") {
        return Command::DeployShards;
    }
    if trimmed.starts_with("discover.") || trimmed.starts_with("discover ") {
        let rest = trimmed
            .trim_start_matches("discover")
            .trim_start_matches('.')
            .trim()
            .trim_end_matches('.');
        let mut it = rest.split_whitespace();
        let cidr = it.next().map(|s| s.to_string());
        let port = it.next().and_then(|s| s.parse().ok());
        return Command::Discover { cidr, port };
    }
    if trimmed == "register." || trimmed == "register" {
        return Command::Register;
    }
    if trimmed.starts_with("unregister ") || trimmed.starts_with("unregister.") {
        let rest = trimmed
            .trim_start_matches("unregister")
            .trim_start_matches('.')
            .trim()
            .trim_end_matches('.');
        let node = rest.split_whitespace().next().unwrap_or("").to_string();
        return Command::Unregister { node };
    }
    if trimmed == "tasks." || trimmed == "tasks" {
        return Command::Tasks;
    }
    let tokens = lex(input);
    let stripped = strip_whitespace(tokens);
    parse(&stripped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpret_shards_command() {
        assert!(matches!(interpret("shards."), Command::ShardStatus));
    }

    #[test]
    fn test_interpret_generate_command() {
        let cmd = interpret("generate hello brave world.");
        match cmd {
            Command::Generate { prompt } => assert_eq!(prompt, "hello brave world"),
            other => panic!("expected Generate, got {:?}", other),
        }
    }

    #[test]
    fn test_lex_bare_question() {
        let tokens = lex("?");
        let stripped = strip_whitespace(tokens);
        assert_eq!(stripped, vec![Token::Question]);
    }

    #[test]
    fn test_lex_node_query() {
        let tokens = lex("n3?");
        let stripped = strip_whitespace(tokens);
        assert_eq!(
            stripped,
            vec![Token::Ident("n3".into()), Token::Question]
        );
    }

    #[test]
    fn test_lex_property_query() {
        let tokens = lex("n3.power?");
        let stripped = strip_whitespace(tokens);
        assert_eq!(
            stripped,
            vec![
                Token::Ident("n3".into()),
                Token::Dot,
                Token::Ident("power".into()),
                Token::Question,
            ]
        );
    }

    #[test]
    fn test_interpret_cluster_summary() {
        assert!(matches!(interpret("?"), Command::ClusterSummary));
    }

    #[test]
    fn test_interpret_node_query() {
        let cmd = interpret("n3?");
        assert!(matches!(cmd, Command::NodeQuery { node } if node == "n3"));
    }

    #[test]
    fn test_interpret_property_query() {
        let cmd = interpret("n3.power?");
        assert!(
            matches!(cmd, Command::PropertyQuery { node, property } if node == "n3" && property == "power")
        );
    }

    #[test]
    fn test_interpret_assign_proposition() {
        let cmd = interpret("n3 assign branch_sort.");
        assert!(
            matches!(cmd, Command::AssignProposition { node, workload } if node == "n3" && workload == "branch_sort")
        );
    }

    #[test]
    fn test_interpret_context_reset() {
        assert!(matches!(interpret("cluster"), Command::ResetContext));
    }

    #[test]
    fn test_interpret_set_context() {
        let cmd = interpret("n3");
        assert!(matches!(cmd, Command::SetContext { node } if node == "n3"));
    }

    #[test]
    fn test_interpret_budget() {
        let cmd = interpret("budget 400w.");
        assert!(matches!(cmd, Command::SetBudget { watts: 400 }));
    }

    #[test]
    fn test_interpret_poetry() {
        let cmd = interpret("poetry on.");
        assert!(matches!(cmd, Command::Poetry { enabled: true }));
    }

    #[test]
    fn test_interpret_probe() {
        assert!(matches!(interpret("probe."), Command::Probe));
    }

    #[test]
    fn test_interpret_bare_property() {
        let cmd = interpret("power?");
        assert!(matches!(cmd, Command::ContextPropertyQuery { property } if property == "power"));
    }
}

#[cfg(test)]
mod discover_tests {
    use super::*;
    #[test]
    fn test_discover_forms() {
        assert!(matches!(interpret("discover."), Command::Discover { cidr: None, port: None }));
        assert!(matches!(interpret("discover. 127.0.0.1 9501"), Command::Discover { cidr: Some(c), port: Some(9501) } if c == "127.0.0.1"));
        assert!(matches!(interpret("discover 10.0.0"), Command::Discover { cidr: Some(c), port: None } if c == "10.0.0"));
    }
}
