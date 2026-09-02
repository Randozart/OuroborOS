pub mod auth;
pub mod ssh;
pub mod tcp;

use anyhow::Result;

/// Transport abstraction for dispatching tasks to nodes.
///
/// MVP uses SSH; Phase 2 adds a custom TCP protocol with lower overhead.
pub trait Transport: Send + Sync {
    /// Dispatch a task payload to a node, returning its result.
    fn dispatch(&self, node_ip: &str, payload: &str) -> Result<String>;

    /// Check whether a node is alive.
    fn heartbeat(&self, node_ip: &str) -> bool;

    /// Send a power-state command to a node.
    fn set_power(&self, node_ip: &str, sleeping: bool) -> Result<()>;
}