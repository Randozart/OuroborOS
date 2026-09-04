//! Push-based message bus: agents register on boot and heartbeat
//! telemetry periodically. One signed line each way — the same wire
//! format as the task channel (`<seq> <tag> <body>`), different verbs.
//!
//! Requests (agent → registry daemon):
//! - `register <telemetry-json>` → `registered <id>` (idempotent per IP)
//! - `heartbeat <telemetry-json>` → `ok <id>` | `unknown` (re-register)
//! - `ping` → `pong`
//!
//! The IP is taken from the connection peer, never self-reported.

use serde::{Deserialize, Serialize};

use crate::beast::NodeStatus as StateStatus;
use crate::error_recovery::ErrorRecovery;
use crate::probe::{CpuInfo, EnergyInfo, MemoryInfo, NodeInfo, NodeStatus};

use super::Registry;

/// Telemetry payload carried by register/heartbeat bodies.
/// Mirrors agent telemetry; `#[serde(default)]` keeps old agents working.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusTelemetry {
    pub hostname: String,
    #[serde(default)]
    pub cpu_model: String,
    #[serde(default)]
    pub cores: u32,
    #[serde(default)]
    pub threads: u32,
    #[serde(default)]
    pub has_avx: bool,
    #[serde(default)]
    pub has_avx2: bool,
    #[serde(default)]
    pub has_sse42: bool,
    #[serde(default)]
    pub ram_total_mib: u64,
    #[serde(default)]
    pub power_watts: u32,
    #[serde(default)]
    pub temp_c: u32,
    #[serde(default)]
    pub load_avg: f64,
    #[serde(default)]
    pub gpus: Vec<crate::probe::gpu::GpuInfo>,
}

impl BusTelemetry {
    /// Map to the probe-result shape Registry::register consumes.
    pub fn to_node_info(&self, ip: &str) -> NodeInfo {
        NodeInfo {
            hostname: self.hostname.clone(),
            ip: ip.to_string(),
            cpu: CpuInfo {
                model: self.cpu_model.clone(),
                cores: self.cores,
                threads: self.threads,
                has_avx: self.has_avx,
                has_avx2: self.has_avx2,
                has_sse42: self.has_sse42,
                has_bmi1: false,
                has_bmi2: false,
                tdp_watts: self.power_watts.max(15),
            },
            memory: MemoryInfo {
                total_mib: self.ram_total_mib,
                speed_mhz: None,
                memory_type: None,
            },
            energy: EnergyInfo {
                current_watts: self.power_watts,
                rapl_available: self.power_watts > 0,
                power_limit_watts: None,
            },
            network: None,
            status: NodeStatus::Idle,
            gpus: self.gpus.clone(),
        }
    }

    /// Working when the box is visibly busy; idle otherwise.
    pub fn status(&self) -> StateStatus {
        if self.load_avg > 2.0 {
            StateStatus::Working
        } else {
            StateStatus::Idle
        }
    }
}

/// Handle one bus message. `peer_ip` comes from the socket; the agent
/// cannot claim a different address. Returns the plain response body
/// (caller signs it).
pub fn handle_bus_message(
    reg: &mut Registry,
    recovery: &mut ErrorRecovery,
    peer_ip: &str,
    body: &str,
) -> String {
    let (verb, rest) = match body.split_once(' ') {
        Some((v, r)) => (v, r.trim()),
        None => (body, ""),
    };
    match verb {
        "ping" => "pong".to_string(),
        "register" => handle_register(reg, recovery, peer_ip, rest),
        "heartbeat" => handle_heartbeat(reg, recovery, peer_ip, rest),
        "status" => handle_status(reg),
        other => format!("err unknown-verb {}", other),
    }
}

/// Live census for the head's shell — the one-source-of-truth read
/// (Registry↔HISS unification). One JSON line: every record's entry
/// and live state plus online flag; the shell merges this over its
/// own topology, so `cluster?`/`n1?`/`n1.gpu` reflect what the bus
/// actually sees.
fn handle_status(reg: &Registry) -> String {
    let nodes: Vec<serde_json::Value> = reg
        .nodes
        .values()
        .map(|r| {
            serde_json::json!({
                "id": r.entry.id,
                "hostname": r.entry.hostname,
                "ip": r.entry.ip,
                "cpu_model": r.entry.cpu_model,
                "cores": r.entry.cores,
                "threads": r.entry.threads,
                "ram_mib": r.entry.ram_mib,
                "tdp_watts": r.entry.tdp_watts,
                "has_gpu": r.entry.has_gpu,
                "gpu_model": r.entry.gpu_model,
                "gpu_vram_mib": r.entry.gpu_vram_mib,
                "has_avx2": r.entry.has_avx2,
                "has_avx": r.entry.has_avx,
                "has_sse42": r.entry.has_sse42,
                "power_watts": r.state.power_watts,
                "temp_c": r.state.thermal_c,
                "load_avg": r.state.load_avg,
                "status": format!("{:?}", r.state.status),
                "online": r.is_alive(reg.heartbeat_threshold),
                "last_seen": r.last_seen,
            })
        })
        .collect();
    serde_json::json!({ "nodes": nodes, "event_count": reg.events.len() }).to_string()
}

fn parse_telemetry(json: &str) -> Option<BusTelemetry> {
    serde_json::from_str(json).ok()
}

fn handle_register(
    reg: &mut Registry,
    recovery: &mut ErrorRecovery,
    peer_ip: &str,
    json: &str,
) -> String {
    let Some(tel) = parse_telemetry(json) else {
        return "err bad-json".to_string();
    };
    // Idempotent per IP: same box re-registering keeps its id, refreshes
    // the live profile + last_seen, and reconciles hardware facts (the
    // entry must never stay frozen at first boot — GPU_CLAIM verification
    // depends on a reflashed tail updating its own record).
    if let Some(id) = reg.find_by_ip(peer_ip) {
        reg.refresh_entry(&id, &tel.to_node_info(peer_ip));
        let events = reg.heartbeat(&id, tel.power_watts, tel.temp_c, tel.load_avg, tel.status());
        recovery.process_events(&events);
        recovery.report_success(&id);
        return format!("registered {}", id);
    }
    let (id, events) = reg.register(&tel.to_node_info(peer_ip));
    recovery.process_events(&events);
    format!("registered {}", id)
}

fn handle_heartbeat(
    reg: &mut Registry,
    recovery: &mut ErrorRecovery,
    peer_ip: &str,
    json: &str,
) -> String {
    let Some(tel) = parse_telemetry(json) else {
        return "err bad-json".to_string();
    };
    let Some(id) = reg.find_by_ip(peer_ip) else {
        // Unknown node heartbeating: tell it to register.
        return "unknown".to_string();
    };
    // Heartbeats carry the full telemetry too — reconcile hardware
    // facts so a reflashed tail's new silicon lands within one beat,
    // no reboot-of-the-bus required.
    reg.refresh_entry(&id, &tel.to_node_info(peer_ip));
    let events = reg.heartbeat(&id, tel.power_watts, tel.temp_c, tel.load_avg, tel.status());
    recovery.process_events(&events);
    recovery.report_success(&id);
    format!("ok {}", id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn tel_json(hostname: &str, load: f64) -> String {
        format!(
            r#"{{"hostname":"{}","cpu_model":"i7-3770","cores":4,"threads":8,"has_avx":true,"has_avx2":true,"has_sse42":true,"ram_total_mib":16384,"power_watts":42,"temp_c":50,"load_avg":{}}}"#,
            hostname, load
        )
    }

    #[test]
    fn test_ping() {
        let mut reg = Registry::new();
        let mut rec = ErrorRecovery::new();
        assert_eq!(handle_bus_message(&mut reg, &mut rec, "10.0.0.5", "ping"), "pong");
    }

    #[test]
    fn test_register_assigns_id() {
        let mut reg = Registry::new();
        let mut rec = ErrorRecovery::new();
        let resp = handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &format!("register {}", tel_json("box-a", 0.4)));
        assert_eq!(resp, "registered n1");
        let record = reg.get("n1").unwrap();
        assert_eq!(record.entry.hostname, "box-a");
        assert_eq!(record.entry.ip, "10.0.0.5");
        assert_eq!(record.entry.tdp_watts, 42);
        assert!(record.entry.has_avx2);
    }

    #[test]
    fn test_register_idempotent_per_ip() {
        let mut reg = Registry::new();
        let mut rec = ErrorRecovery::new();
        let body = format!("register {}", tel_json("box-a", 0.4));
        assert_eq!(handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &body), "registered n1");
        assert_eq!(handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &body), "registered n1");
        assert_eq!(reg.len(), 1, "re-register must not duplicate");
    }

    #[test]
    fn test_heartbeat_updates_and_unknown_prompts_register() {
        let mut reg = Registry::new();
        let mut rec = ErrorRecovery::new();
        assert_eq!(
            handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &format!("heartbeat {}", tel_json("box-a", 0.4))),
            "unknown"
        );
        handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &format!("register {}", tel_json("box-a", 0.4)));
        let resp = handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &format!("heartbeat {}", tel_json("box-a", 0.4)));
        assert_eq!(resp, "ok n1");
        let record = reg.get("n1").unwrap();
        assert_eq!(record.state.power_watts, 42);
        assert_eq!(record.state.thermal_c, 50);
    }

    #[test]
    fn test_heartbeat_busy_marks_working() {
        let mut reg = Registry::new();
        let mut rec = ErrorRecovery::new();
        handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &format!("register {}", tel_json("box-a", 0.4)));
        handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &format!("heartbeat {}", tel_json("box-a", 5.0)));
        let record = reg.get("n1").unwrap();
        assert_eq!(record.state.status, StateStatus::Working);
    }

    #[test]
    fn test_unknown_verb() {
        let mut reg = Registry::new();
        let mut rec = ErrorRecovery::new();
        let resp = handle_bus_message(&mut reg, &mut rec, "10.0.0.5", "dance");
        assert!(resp.starts_with("err unknown-verb"));
    }

    #[test]
    fn test_bad_json() {
        let mut reg = Registry::new();
        let mut rec = ErrorRecovery::new();
        assert_eq!(handle_bus_message(&mut reg, &mut rec, "10.0.0.5", "register {nope"), "err bad-json");
    }

    #[test]
    fn test_stale_after_heartbeat_gap() {
        let mut reg = Registry::new().with_heartbeat(Duration::from_secs(30));
        let mut rec = ErrorRecovery::new();
        handle_bus_message(&mut reg, &mut rec, "10.0.0.5", &format!("register {}", tel_json("box-a", 0.4)));
        assert_eq!(reg.alive_nodes().len(), 1);
        reg.touch_last_seen("n1", 0);
        assert_eq!(reg.alive_nodes().len(), 0);
        assert_eq!(reg.offline_nodes().len(), 1);
    }
}
