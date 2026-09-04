use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::registry::{Event, Registry};

/// Tracks per-node failure state for recovery decisions.
#[derive(Debug, Clone)]
pub struct NodeFailure {
    pub node_id: String,
    pub failure_count: u32,
    pub last_failure: Instant,
    pub reason: String,
    pub recovered: bool,
}

impl NodeFailure {
    fn new(node_id: &str, reason: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            failure_count: 0,
            last_failure: Instant::now(),
            reason: reason.to_string(),
            recovered: false,
        }
    }
}

/// Recovery policy configuration.
#[derive(Debug, Clone)]
pub struct RecoveryPolicy {
    /// Max failures before marking node offline.
    pub max_failures: u32,
    /// Cooldown before retrying an offline node.
    pub cooldown: Duration,
    /// Max time to wait for a node to recover.
    pub recovery_timeout: Duration,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            max_failures: 3,
            cooldown: Duration::from_secs(30),
            recovery_timeout: Duration::from_secs(300),
        }
    }
}

/// Error recovery manager: detects crashes, tracks failures, triggers requeue.
pub struct ErrorRecovery {
    failures: HashMap<String, NodeFailure>,
    policy: RecoveryPolicy,
}

impl ErrorRecovery {
    pub fn new() -> Self {
        Self {
            failures: HashMap::new(),
            policy: RecoveryPolicy::default(),
        }
    }

    pub fn with_policy(policy: RecoveryPolicy) -> Self {
        Self {
            failures: HashMap::new(),
            policy,
        }
    }

    /// Report a failure for a node. Returns true if node should be marked offline.
    pub fn report_failure(&mut self, node_id: &str, reason: &str) -> bool {
        let entry = self.failures
            .entry(node_id.to_string())
            .or_insert_with(|| NodeFailure::new(node_id, reason));

        entry.failure_count += 1;
        entry.last_failure = Instant::now();
        entry.reason = reason.to_string();

        entry.failure_count >= self.policy.max_failures
    }

    /// Report a successful heartbeat — resets failure count.
    pub fn report_success(&mut self, node_id: &str) -> Option<NodeFailure> {
        if let Some(mut failure) = self.failures.remove(node_id) {
            failure.recovered = true;
            Some(failure)
        } else {
            None
        }
    }

    /// Check if a node is in cooldown (recently failed, not yet retryable).
    pub fn is_in_cooldown(&self, node_id: &str) -> bool {
        if let Some(f) = self.failures.get(node_id) {
            f.last_failure.elapsed() < self.policy.cooldown
        } else {
            false
        }
    }

    /// Check if a node has exceeded recovery timeout.
    pub fn is_timed_out(&self, node_id: &str) -> bool {
        if let Some(f) = self.failures.get(node_id) {
            f.last_failure.elapsed() > self.policy.recovery_timeout
        } else {
            false
        }
    }

    /// Get all failed nodes.
    pub fn failed_nodes(&self) -> Vec<&NodeFailure> {
        self.failures.values().collect()
    }

    /// Get failure info for a specific node.
    pub fn get_failure(&self, node_id: &str) -> Option<&NodeFailure> {
        self.failures.get(node_id)
    }

    /// Process registry events and update failure tracking.
    /// Returns nodes that should be marked offline.
    pub fn process_events(&mut self, events: &[Event]) -> Vec<String> {
        let mut offline_nodes = Vec::new();
        for event in events {
            match event {
                Event::NodeLeft { node_id, reason } => {
                    if self.report_failure(node_id, reason) {
                        offline_nodes.push(node_id.clone());
                    }
                }
                Event::NodeStateChanged { node_id, from: _, to } => {
                    if to == "Offline" {
                        self.report_failure(node_id, "marked offline");
                    }
                }
                Event::NodeJoined { node_id, .. } => {
                    self.report_success(node_id);
                }
                _ => {}
            }
        }
        offline_nodes
    }

    /// Sweep registry for nodes that haven't been seen recently.
    /// Returns nodes that should be marked offline.
    pub fn sweep_stale(&self, registry: &Registry) -> Vec<String> {
        registry.offline_nodes()
            .iter()
            .map(|r| r.entry.id.clone())
            .collect()
    }

    /// Cleanup: remove entries for nodes no longer in the registry.
    pub fn cleanup(&mut self, registry: &Registry) {
        self.failures.retain(|id, _| registry.get(id).is_some());
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;
    use crate::probe::{NodeInfo, CpuInfo, MemoryInfo, EnergyInfo};

    fn test_info(hostname: &str, ip: &str) -> NodeInfo {
        NodeInfo {
            hostname: hostname.to_string(),
            ip: ip.to_string(),
            cpu: CpuInfo {
                model: "i5".into(), cores: 4, threads: 4,
                has_avx: true, has_avx2: true, has_sse42: true,
                has_bmi1: false, has_bmi2: false, tdp_watts: 35,
            },
            memory: MemoryInfo { total_mib: 8192, speed_mhz: None, memory_type: None },
            energy: EnergyInfo { current_watts: 35, rapl_available: false, power_limit_watts: None },
            network: None,
            status: crate::probe::NodeStatus::Idle,
            gpus: Vec::new(),
        }
    }

    #[test]
    fn test_report_failure_threshold() {
        let mut rec = ErrorRecovery::new();
        assert!(!rec.report_failure("n1", "timeout"));
        assert!(!rec.report_failure("n1", "timeout"));
        assert!(rec.report_failure("n1", "timeout")); // 3rd failure -> offline
    }

    #[test]
    fn test_report_success_resets() {
        let mut rec = ErrorRecovery::new();
        rec.report_failure("n1", "timeout");
        rec.report_failure("n1", "timeout");
        rec.report_success("n1");
        // After success, failure count resets
        assert!(!rec.report_failure("n1", "timeout")); // only 1st failure again
    }

    #[test]
    fn test_cooldown() {
        let policy = RecoveryPolicy {
            cooldown: Duration::from_millis(50),
            ..Default::default()
        };
        let mut rec = ErrorRecovery::with_policy(policy);
        rec.report_failure("n1", "timeout");
        assert!(rec.is_in_cooldown("n1"));
        std::thread::sleep(Duration::from_millis(60));
        assert!(!rec.is_in_cooldown("n1"));
    }

    #[test]
    fn test_process_events_node_left() {
        let mut rec = ErrorRecovery::new();
        let events = vec![Event::NodeLeft {
            node_id: "n1".to_string(),
            reason: "crash".to_string(),
        }];
        let offline = rec.process_events(&events);
        assert_eq!(offline.len(), 0); // 1 failure, below threshold
    }

    #[test]
    fn test_sweep_stale() {
        let mut reg = Registry::new().with_heartbeat(Duration::from_secs(1));
        let info = test_info("node-a", "192.168.1.10");
        reg.register(&info);
        // register() stamps last_seen = now; backdate it to look stale
        reg.touch_last_seen("n1", 0);
        let rec = ErrorRecovery::new();
        let stale = rec.sweep_stale(&reg);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0], "n1");
    }

    #[test]
    fn test_cleanup_removes_unknown() {
        let mut reg = Registry::new();
        let info = test_info("node-a", "192.168.1.10");
        reg.register(&info);
        let mut rec = ErrorRecovery::new();
        rec.report_failure("n1", "timeout");
        rec.report_failure("n2", "timeout"); // n2 doesn't exist
        assert_eq!(rec.failed_nodes().len(), 2);
        rec.cleanup(&reg);
        assert_eq!(rec.failed_nodes().len(), 1); // n2 removed
    }
}
