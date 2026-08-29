use std::collections::HashMap;

/// Persistent context memory for the shell REPL.
///
/// After typing `n3`, bare `power?` means `n3.power?`.
/// Typing `cluster` resets context to cluster level.
#[derive(Debug, Clone)]
pub struct Context {
    /// Current node context (None = cluster level).
    current_node: Option<String>,
    /// Node properties cache for fast lookup.
    properties: HashMap<String, HashMap<String, String>>,
    /// Poetry mode toggle.
    poetry: bool,
}

impl Context {
    pub fn new() -> Self {
        Self {
            current_node: None,
            properties: HashMap::new(),
            poetry: false,
        }
    }

    /// Set context to a specific node.
    pub fn set_node(&mut self, node: &str) {
        self.current_node = Some(node.to_string());
    }

    /// Reset context to cluster level.
    pub fn reset(&mut self) {
        self.current_node = None;
    }

    /// Get current context as a display string.
    pub fn current_label(&self) -> &str {
        match &self.current_node {
            Some(node) => node,
            None => "CLUSTER",
        }
    }

    /// Check if we're in node context.
    pub fn is_node_context(&self) -> bool {
        self.current_node.is_some()
    }

    /// Get the current node name (if in node context).
    pub fn current_node(&self) -> Option<&str> {
        self.current_node.as_deref()
    }

    /// Resolve a bare property to a full query.
    /// If in node context, prepends the node name.
    pub fn resolve_property(&self, property: &str) -> String {
        match &self.current_node {
            Some(node) => format!("{}.{}", node, property),
            None => property.to_string(),
        }
    }

    /// Cache a node's properties for lookup.
    pub fn cache_properties(&mut self, node: &str, props: HashMap<String, String>) {
        self.properties.insert(node.to_string(), props);
    }

    /// Look up a cached property.
    pub fn get_property(&self, node: &str, property: &str) -> Option<&str> {
        self.properties
            .get(node)
            .and_then(|props| props.get(property))
            .map(|s| s.as_str())
    }

    /// Toggle poetry mode.
    pub fn set_poetry(&mut self, enabled: bool) {
        self.poetry = enabled;
    }

    /// Check if poetry mode is enabled.
    pub fn is_poetry(&self) -> bool {
        self.poetry
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_cluster_context() {
        let ctx = Context::new();
        assert_eq!(ctx.current_label(), "CLUSTER");
        assert!(!ctx.is_node_context());
    }

    #[test]
    fn test_set_node() {
        let mut ctx = Context::new();
        ctx.set_node("n3");
        assert_eq!(ctx.current_label(), "n3");
        assert!(ctx.is_node_context());
        assert_eq!(ctx.current_node(), Some("n3"));
    }

    #[test]
    fn test_reset() {
        let mut ctx = Context::new();
        ctx.set_node("n3");
        ctx.reset();
        assert_eq!(ctx.current_label(), "CLUSTER");
        assert!(!ctx.is_node_context());
    }

    #[test]
    fn test_resolve_property_in_node_context() {
        let mut ctx = Context::new();
        ctx.set_node("n3");
        assert_eq!(ctx.resolve_property("power"), "n3.power");
    }

    #[test]
    fn test_resolve_property_in_cluster_context() {
        let ctx = Context::new();
        assert_eq!(ctx.resolve_property("power"), "power");
    }

    #[test]
    fn test_cache_and_get_property() {
        let mut ctx = Context::new();
        let mut props = HashMap::new();
        props.insert("power".to_string(), "12W".to_string());
        ctx.cache_properties("n3", props);
        assert_eq!(ctx.get_property("n3", "power"), Some("12W"));
        assert_eq!(ctx.get_property("n3", "thermal"), None);
        assert_eq!(ctx.get_property("n7", "power"), None);
    }

    #[test]
    fn test_poetry_toggle() {
        let mut ctx = Context::new();
        assert!(!ctx.is_poetry());
        ctx.set_poetry(true);
        assert!(ctx.is_poetry());
        ctx.set_poetry(false);
        assert!(!ctx.is_poetry());
    }
}
