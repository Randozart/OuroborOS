//! The named resource tree (docs/PLAN9.md §4.1).
//!
//! A `ResourcePath` is the dot-notation grammar every op speaks:
//! `n1`, `n1.power`, `n1.gpu`, `cluster.budget`, `queue`, `tasks`.
//! Paths are parsed here (grammar only); existence is resolved by a
//! `GraphBackend` (one writer, Art. 3).

use serde::{Deserialize, Serialize};

/// A dot-separated path into the named resource tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePath {
    pub segments: Vec<String>,
}

impl ResourcePath {
    /// Parse a dot-path. Grammar only: non-empty, alphanumeric-or-underscore
    /// segments separated by `.`. Existence is a backend question.
    pub fn parse(input: &str) -> Result<Self, String> {
        let raw: Vec<&str> = input.split('.').collect();
        if raw.is_empty() || raw.iter().any(|s| s.trim().is_empty()) {
            return Err(format!("invalid resource path: {:?}", input));
        }
        let segments: Vec<String> = raw.iter().map(|s| s.trim().to_string()).collect();
        if segments
            .iter()
            .any(|s| !s.chars().all(|c| c.is_alphanumeric() || c == '_'))
        {
            return Err(format!("invalid resource path: {:?}", input));
        }
        Ok(Self { segments })
    }

    /// The node segment, when the path names a node (`n1`, `n1.gpu`).
    pub fn node(&self) -> Option<&str> {
        self.segments
            .first()
            .filter(|s| s.starts_with('n'))
            .map(|s| s.as_str())
    }

    /// The property segment, when the path is `node.property`.
    pub fn property(&self) -> Option<&str> {
        self.segments.get(1).map(|s| s.as_str())
    }

    /// Whether the path is exactly a single known root segment.
    pub fn is(&self, root: &str) -> bool {
        self.segments.len() == 1 && self.segments[0] == root
    }

    /// Whether the path is `root` with a single sub-segment.
    pub fn is_child_of(&self, root: &str, child: &str) -> bool {
        self.segments.len() == 2 && self.segments[0] == root && self.segments[1] == child
    }
}

impl std::fmt::Display for ResourcePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.segments.join("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_node_prop() {
        let p = ResourcePath::parse("n1.gpu").unwrap();
        assert_eq!(p.segments, vec!["n1".to_string(), "gpu".to_string()]);
        assert_eq!(p.node(), Some("n1"));
        assert_eq!(p.property(), Some("gpu"));
    }

    #[test]
    fn test_parse_roots() {
        assert!(ResourcePath::parse("cluster.budget").unwrap().is_child_of("cluster", "budget"));
        assert!(ResourcePath::parse("queue").unwrap().is("queue"));
        assert!(ResourcePath::parse("tasks").unwrap().is("tasks"));
        assert_eq!(ResourcePath::parse("cluster").unwrap().to_string(), "cluster");
    }

    #[test]
    fn test_parse_rejects_garbage() {
        assert!(ResourcePath::parse("").is_err());
        assert!(ResourcePath::parse("..").is_err());
        assert!(ResourcePath::parse("n1..gpu").is_err());
        assert!(ResourcePath::parse("a b").is_err());
    }

    #[test]
    fn test_node_only_when_n_prefix() {
        assert_eq!(ResourcePath::parse("n1").unwrap().node(), Some("n1"));
        assert_eq!(ResourcePath::parse("cluster.budget").unwrap().node(), None);
    }
}