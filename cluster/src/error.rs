use std::fmt;

#[derive(Debug)]
pub enum ClusterError {
    NodeOffline(String),
    ProbeFailed(String, String),
    TransportFailed(String, String),
    EnergyBudgetExceeded { current: u32, budget: u32, requested: u32 },
    WorkloadClassUnknown(String),
    BeastParseError(String),
    SshError(String),
    IoError(std::io::Error),
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterError::NodeOffline(node) => write!(f, "Node {} is offline", node),
            ClusterError::ProbeFailed(node, reason) => write!(f, "Probe failed on {}: {}", node, reason),
            ClusterError::TransportFailed(node, reason) => {
                write!(f, "Transport failed to {}: {}", node, reason)
            }
            ClusterError::EnergyBudgetExceeded { current, budget, requested } => {
                write!(
                    f,
                    "Energy budget exceeded: current={}W, budget={}W, requested={}W",
                    current, budget, requested
                )
            }
            ClusterError::WorkloadClassUnknown(name) => {
                write!(f, "Unknown workload class: {}", name)
            }
            ClusterError::BeastParseError(msg) => write!(f, "Beast parse error: {}", msg),
            ClusterError::SshError(msg) => write!(f, "SSH error: {}", msg),
            ClusterError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for ClusterError {}

impl From<std::io::Error> for ClusterError {
    fn from(e: std::io::Error) -> Self {
        ClusterError::IoError(e)
    }
}
