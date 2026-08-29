use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Telemetry received from a node agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTelemetry {
    pub hostname: String,
    pub cpu_model: String,
    pub cores: u32,
    pub threads: u32,
    pub has_avx2: bool,
    pub ram_total_mib: u64,
    pub ram_used_mib: u64,
    pub power_watts: u32,
    pub temp_c: u32,
    pub load_avg: f64,
}

/// Task sent to a node agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub name: String,
    pub payload: String,
    pub estimated_watts: u32,
    pub estimated_seconds: u32,
}

/// Result received from a node agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskResult {
    pub task_id: String,
    pub status: String,
    pub output: String,
    pub elapsed_ms: u64,
    pub peak_watts: u32,
}

/// Send a raw message to an agent and receive a response.
fn send_raw(addr: &str, msg: &str) -> Result<String> {
    let stream = TcpStream::connect(addr)
        .with_context(|| format!("connect to {}", addr))?;
    stream.set_read_timeout(Some(DEFAULT_TIMEOUT))?;
    stream.set_write_timeout(Some(DEFAULT_TIMEOUT))?;

    let mut writer = stream.try_clone()?;
    writer.write_all(msg.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader
        .read_line(&mut response)
        .with_context(|| format!("read from {}", addr))?;

    Ok(response.trim().to_string())
}

/// Send a raw message to an agent and receive a response, with a custom timeout.
fn send_raw_timeout(addr: &str, msg: &str, timeout: Duration) -> Result<String> {
    let stream = TcpStream::connect(addr)
        .with_context(|| format!("connect to {}", addr))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let mut writer = stream.try_clone()?;
    writer.write_all(msg.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader
        .read_line(&mut response)
        .with_context(|| format!("read from {}", addr))?;

    Ok(response.trim().to_string())
}

/// Ping an agent to check if it's alive.
pub fn ping(addr: &str) -> Result<bool> {
    let resp = send_raw(addr, "ping")?;
    Ok(resp == "pong")
}

/// Request telemetry from an agent.
pub fn telemetry(addr: &str) -> Result<AgentTelemetry> {
    let resp = send_raw(addr, "telemetry")?;
    let tel: AgentTelemetry =
        serde_json::from_str(&resp).with_context(|| "parse telemetry response")?;
    Ok(tel)
}

/// Execute a task on an agent (long timeout, for inference).
pub fn execute(addr: &str, task: &AgentTask) -> Result<AgentTaskResult> {
    execute_timeout(addr, task, Duration::from_secs(120))
}

/// Execute a task on an agent with an explicit timeout.
pub fn execute_timeout(
    addr: &str,
    task: &AgentTask,
    timeout: Duration,
) -> Result<AgentTaskResult> {
    let json = serde_json::to_string(task)?;
    let resp = send_raw_timeout(addr, &json, timeout)?;
    let result: AgentTaskResult = serde_json::from_str(&resp)
        .with_context(|| format!("parse task result from {} (raw {:?})", addr, &resp[..resp.len().min(80)]))?;
    Ok(result)
}

/// Check if an agent is reachable, returning (addr, alive).
pub fn probe(addr: &str) -> (String, bool) {
    match ping(addr) {
        Ok(true) => (addr.to_string(), true),
        _ => (addr.to_string(), false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_raw_unreachable() {
        let result = send_raw("127.0.0.1:19999", "ping");
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_task_serializes() {
        let task = AgentTask {
            id: "t1".into(),
            name: "echo".into(),
            payload: "hello".into(),
            estimated_watts: 10,
            estimated_seconds: 1,
        };
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("echo"));
        assert!(json.contains("hello"));
    }

    #[test]
    fn test_agent_telemetry_deserializes() {
        let json = r#"{
            "hostname": "test-node",
            "cpu_model": "i5-4590",
            "cores": 4,
            "threads": 4,
            "has_avx2": true,
            "ram_total_mib": 32768,
            "ram_used_mib": 8192,
            "power_watts": 35,
            "temp_c": 45,
            "load_avg": 0.5
        }"#;
        let tel: AgentTelemetry = serde_json::from_str(json).unwrap();
        assert_eq!(tel.hostname, "test-node");
        assert_eq!(tel.ram_total_mib, 32768);
    }
}
