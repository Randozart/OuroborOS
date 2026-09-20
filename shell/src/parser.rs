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
    /// `n3.lanes?` — live per-lane health (priced edges)
    Lanes { node: String },
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
    /// `ask [N] <text>.` — Bonsai greedy completion, local engine, N tokens
    Ask { max_tokens: usize, text: String },
    /// `shards.` — show pipeline plan + activation transport probe
    ShardStatus,
    /// `discover. [cidr] [port]` — sweep subnet for agents, absorb them
    Discover { cidr: Option<String>, port: Option<u16> },
    /// `register.` — register a node via probe
    Register,
    /// `help` — the command reference
    Help,
    /// `unregister n3.` — unregister a node
    Unregister { node: String },
    /// `tasks.` — show task queue status
    Tasks,
    /// `weights [n1 [i]].` — weight census / open a bulk handle (Rung B3)
    Weights { target: String },
    /// `bind weights.n1.0 [offset] [length].` — bind a span for fetch
    /// (Track C). No range = whole tensor.
    Bind { target: String, offset: Option<u64>, length: Option<u64> },
    /// `revoke b1.` — release a binding en-bloc (Track C)
    Revoke { binding: String },
    /// `fetch b1 [offset] [length].` or `fetch weights.n1.0 [offset]
    /// [length].` — fetch a bound span's bytes from the live agent (Track C).
    /// Target is a binding id or a weights path; bytes ride frames.
    Fetch { target: String, offset: Option<u64>, length: Option<u64> },
    /// `drift [rev]` — which tails don't run the expected versions
    Drift { expected: Option<String> },
    /// `recover.` — trigger error recovery sweep
    Recover,
    /// `reconcile` — converge the cluster toward the declared state now
    Reconcile,
    /// `desired` — show the declared state of the cluster
    Desired,
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

        // `cluster active?` — the space-separated twin (dots are internal
        // separators; the space form reads like prose).
        [Token::Ident(c), Token::Ident(prop), Token::Question]
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

        // `n3.lanes?` — live per-lane health (priced edges on the node).
        // Must precede the general `n3.power?` arm (lanes is a property too).
        [Token::Ident(name), Token::Dot, Token::Ident(prop), Token::Question]
            if name.starts_with('n') && prop == "lanes" =>
        {
            Command::Lanes { node: name.clone() }
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

        // `n3 lanes?` — the space-separated twin (prose form).
        [Token::Ident(name), Token::Ident(prop), Token::Question]
            if name.starts_with('n') && prop == "lanes" =>
        {
            Command::Lanes { node: name.clone() }
        }

        // `n3 power?` — space-separated property query.
        [Token::Ident(name), Token::Ident(prop), Token::Question]
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

        // `n3 assign branch_sort.` / `n3 assign branch_sort` — proposition
        [Token::Ident(name), Token::Ident(pred), Token::Ident(wl), Token::Dot]
            if name.starts_with('n') && pred == "assign" =>
        {
            Command::AssignProposition {
                node: name.clone(),
                workload: wl.clone(),
            }
        }
        [Token::Ident(name), Token::Ident(pred), Token::Ident(wl)]
            if name.starts_with('n') && pred == "assign" =>
        {
            Command::AssignProposition {
                node: name.clone(),
                workload: wl.clone(),
            }
        }

        // `n3休眠.` / `n3 sleep` / `n3 wake` — power state
        [Token::Ident(name), Token::Ident(state), Token::Dot]
            if name.starts_with('n')
                && (state == "休眠" || state == "sleep" || state == "wake") =>
        {
            Command::PowerState {
                node: name.clone(),
                sleeping: state != "wake",
            }
        }
        [Token::Ident(name), Token::Ident(state)]
            if name.starts_with('n')
                && (state == "休眠" || state == "sleep" || state == "wake") =>
        {
            Command::PowerState {
                node: name.clone(),
                sleeping: state != "wake",
            }
        }

        // `reconcile` — converge now (also runs on a 5s tick)
        [Token::Ident(r)] if r == "reconcile" => Command::Reconcile,

        // `desired` — the declared state
        [Token::Ident(d)] if d == "desired" => Command::Desired,

        // `budget 400w.` / `budget 400w`
        [Token::Ident(b), Token::Ident(val), Token::Dot] if b == "budget" => {
            let watts_str = val.trim_end_matches('w').trim_end_matches('W');
            let watts = watts_str.parse::<u32>().unwrap_or(0);
            Command::SetBudget { watts }
        }
        [Token::Ident(b), Token::Ident(val)] if b == "budget" => {
            let watts_str = val.trim_end_matches('w').trim_end_matches('W');
            let watts = watts_str.parse::<u32>().unwrap_or(0);
            Command::SetBudget { watts }
        }

        // `poetry on.` / `poetry on`
        [Token::Ident(p), Token::Ident(val), Token::Dot] if p == "poetry" => {
            Command::Poetry {
                enabled: val == "on",
            }
        }
        [Token::Ident(p), Token::Ident(val)] if p == "poetry" => {
            Command::Poetry {
                enabled: val == "on",
            }
        }

        // `probe.` / `probe`
        [Token::Ident(p), Token::Dot] if p == "probe" => Command::Probe,
        [Token::Ident(p)] if p == "probe" => Command::Probe,

        // `deploy.` / `deploy`
        [Token::Ident(d), Token::Dot] if d == "deploy" => Command::Deploy,
        [Token::Ident(d)] if d == "deploy" => Command::Deploy,

        // `save.` / `save`
        [Token::Ident(s), Token::Dot] if s == "save" => Command::Save,
        [Token::Ident(s)] if s == "save" => Command::Save,

        // `load.` / `load`
        [Token::Ident(l), Token::Dot] if l == "load" => Command::Load,
        [Token::Ident(l)] if l == "load" => Command::Load,

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
    // The sentence-period is decoration, never structure: strip one
    // trailing '.' so `budget 400w` and `budget 400w.` are the same
    // command. Property dots (`n1.power?`) are internal separators and
    // unaffected. `generate` keeps its own suffix handling below.
    let trimmed = trimmed.strip_suffix('.').unwrap_or(trimmed);
    // Raw-string shortcut: prompts may contain any characters except a trailing '.'
    if let Some(rest) = trimmed.strip_prefix("generate ") {
        let prompt = rest.strip_suffix('.').unwrap_or(rest).trim().to_string();
        return Command::Generate { prompt };
    }
    if let Some(rest) = trimmed.strip_prefix("ask ") {
        let rest = rest.strip_suffix('.').unwrap_or(rest).trim();
        // optional leading token budget: `ask 24 Hello world`
        let (max_tokens, text) = match rest.split_once(' ') {
            Some((n, t)) if n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty() => {
                (n.parse().unwrap_or(8), t.trim().to_string())
            }
            _ => (8, rest.to_string()),
        };
        return Command::Ask { max_tokens, text };
    }
    if trimmed == "shards" || trimmed == "shards." || trimmed.starts_with("shards ") {
        return Command::ShardStatus;
    }
    if trimmed.starts_with("deploy shards") {
        return Command::DeployShards;
    }
    if trimmed == "discover" || trimmed.starts_with("discover") {
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
    if trimmed == "help" || trimmed == "help?" {
        return Command::Help;
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
    if trimmed == "weights" || trimmed.starts_with("weights ") || trimmed.starts_with("weights.") {
        let target = trimmed
            .trim_start_matches("weights")
            .trim_start_matches('.')
            .trim()
            .trim_end_matches('.');
        return Command::Weights { target: target.to_string() };
    }
    if trimmed == "bind" || trimmed.starts_with("bind ") {
        let rest = trimmed.trim_start_matches("bind").trim().trim_end_matches('.');
        let mut parts = rest.split_whitespace();
        let target = parts.next().unwrap_or("").to_string();
        let offset = parts.next().and_then(|s| s.parse().ok());
        let length = parts.next().and_then(|s| s.parse().ok());
        return Command::Bind { target, offset, length };
    }
    if trimmed == "revoke" || trimmed.starts_with("revoke ") {
        let binding = trimmed.trim_start_matches("revoke").trim().trim_end_matches('.');
        return Command::Revoke { binding: binding.to_string() };
    }
    if trimmed == "fetch" || trimmed.starts_with("fetch ") {
        let rest = trimmed.trim_start_matches("fetch").trim().trim_end_matches('.');
        let mut parts = rest.split_whitespace();
        let target = parts.next().unwrap_or("").to_string();
        let offset = parts.next().and_then(|s| s.parse().ok());
        let length = parts.next().and_then(|s| s.parse().ok());
        return Command::Fetch { target, offset, length };
    }
    if trimmed == "recover." || trimmed == "recover" {
        return Command::Recover;
    }
    if trimmed == "drift." || trimmed == "drift" {
        return Command::Drift { expected: None };
    }
    if let Some(rest) = trimmed.strip_prefix("drift ") {
        let expected = rest.trim().trim_end_matches('.');
        return Command::Drift {
            expected: Some(expected.to_string()),
        };
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
        // ask: default budget, explicit budget, trailing-period strip
        match interpret("ask Hello") {
            Command::Ask { max_tokens, text } => {
                assert_eq!(max_tokens, 8);
                assert_eq!(text, "Hello");
            }
            other => panic!("expected Ask, got {:?}", other),
        }
        match interpret("ask 24 Hello, world.") {
            Command::Ask { max_tokens, text } => {
                assert_eq!(max_tokens, 24);
                assert_eq!(text, "Hello, world");
            }
            other => panic!("expected Ask, got {:?}", other),
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

    /// The sentence-period is decoration: every dotted form has an
    /// undotted twin (the property dot in `n1.power?` is untouched).
    #[test]
    fn test_interpret_undotted_equivalents() {
        assert!(
            matches!(interpret("n3 assign branch_sort"), Command::AssignProposition { node, workload } if node == "n3" && workload == "branch_sort")
        );
        assert!(matches!(interpret("n3 sleep"), Command::PowerState { sleeping: true, .. }));
        assert!(matches!(interpret("budget 400w"), Command::SetBudget { watts: 400 }));
        assert!(matches!(interpret("budget 400"), Command::SetBudget { watts: 400 }));
        assert!(matches!(interpret("poetry off"), Command::Poetry { enabled: false }));
        assert!(matches!(interpret("probe"), Command::Probe));
        assert!(matches!(interpret("deploy"), Command::Deploy));
        assert!(matches!(interpret("save"), Command::Save));
        assert!(matches!(interpret("load"), Command::Load));
        assert!(matches!(interpret("register"), Command::Register));
        assert!(matches!(interpret("tasks"), Command::Tasks));
        assert!(matches!(interpret("weights"), Command::Weights { .. }));
        assert!(matches!(interpret("weights.n1."), Command::Weights { target } if target == "n1"));
        assert!(
            matches!(interpret("weights.n1.0."), Command::Weights { target } if target == "n1.0")
        );
        assert!(
            matches!(interpret("bind weights.n1.0."), Command::Bind { target, offset, length }
                if target == "weights.n1.0" && offset.is_none() && length.is_none())
        );
        assert!(
            matches!(interpret("bind weights.n1.0 512 256."), Command::Bind { target, offset, length }
                if target == "weights.n1.0" && offset == Some(512) && length == Some(256))
        );
        assert!(
            matches!(interpret("revoke b1."), Command::Revoke { binding } if binding == "b1")
        );
        assert!(
            matches!(interpret("fetch b1."), Command::Fetch { target, offset, length }
                if target == "b1" && offset.is_none() && length.is_none())
        );
        assert!(
            matches!(interpret("fetch weights.n1.0 128 256."), Command::Fetch { target, offset, length }
                if target == "weights.n1.0" && offset == Some(128) && length == Some(256))
        );
        assert!(matches!(interpret("drift"), Command::Drift { expected: None }));
        assert!(
            matches!(interpret("drift abc1234."), Command::Drift { expected: Some(e) } if e == "abc1234")
        );
        assert!(matches!(interpret("recover"), Command::Recover));
        assert!(matches!(interpret("help"), Command::Help));
    }

    #[test]
    fn test_property_dot_survives_period_strip() {
        // `n1.power?` ends in '?' — unaffected. But a command whose
        // property query is written with a trailing period still
        // resolves: the strip happens once, before tokenizing.
        assert!(
            matches!(interpret("cluster.active?"), Command::BulkQuery { filter } if filter == "active")
        );
    }

    /// The space-separated prose forms are twins of the dotted queries.
    #[test]
    fn test_space_separated_queries() {
        assert!(
            matches!(interpret("n1 power?"), Command::PropertyQuery { node, property }
                if node == "n1" && property == "power")
        );
        assert!(
            matches!(interpret("n1 lanes?"), Command::Lanes { node } if node == "n1")
        );
        assert!(
            matches!(interpret("n1.lanes?"), Command::Lanes { node } if node == "n1")
        );
        assert!(
            matches!(interpret("cluster active?"), Command::BulkQuery { filter } if filter == "active")
        );
        // The dotted and space forms must agree.
        let dotted = interpret("n1.power?");
        let spaced = interpret("n1 power?");
        assert_eq!(format!("{:?}", dotted), format!("{:?}", spaced));
        let d2 = interpret("cluster.active?");
        let s2 = interpret("cluster active?");
        assert_eq!(format!("{:?}", d2), format!("{:?}", s2));
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
