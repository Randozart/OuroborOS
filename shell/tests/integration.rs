use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;

/// Find the ouro-agent binary in the target directory.
fn agent_binary() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent().unwrap();
    let target = workspace.join("target/debug/ouro-agent");
    assert!(
        target.exists(),
        "ouro-agent binary not found at {}. Run `cargo build` first.",
        target.display()
    );
    target
}

/// Start an agent process on a random port. Returns (child, port).
fn start_agent() -> (Child, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let bin = agent_binary();
    let child = Command::new(&bin)
        .env("OURO_PORT", port.to_string())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to start ouro-agent");

    // Wait for agent to be ready
    for _ in 0..50 {
        thread::sleep(Duration::from_millis(100));
        if TcpStream::connect(format!("127.0.0.1:{}", port)).is_ok() {
            return (child, port);
        }
    }
    panic!("agent on port {} did not start in time", port);
}

/// Send a message and read response.
fn send_and_receive(addr: &str, msg: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    stream.write_all(msg.as_bytes()).unwrap();
    stream.write_all(b"\n").unwrap();
    stream.flush().unwrap();

    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader.read_line(&mut response).unwrap();
    response.trim().to_string()
}

#[test]
fn test_agent_ping_roundtrip() {
    let (mut child, port) = start_agent();
    let addr = format!("127.0.0.1:{}", port);

    let resp = send_and_receive(&addr, "ping");
    assert_eq!(resp, "pong");

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn test_agent_telemetry_roundtrip() {
    let (mut child, port) = start_agent();
    let addr = format!("127.0.0.1:{}", port);

    let resp = send_and_receive(&addr, "telemetry");
    let tel: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert!(tel["hostname"].is_string());
    assert!(tel["ram_total_mib"].is_number());
    assert!(tel["ram_total_mib"].as_u64().unwrap() > 0);

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn test_agent_task_echo() {
    let (mut child, port) = start_agent();
    let addr = format!("127.0.0.1:{}", port);

    let task = r#"{"id":"t1","name":"echo","payload":"hello world","estimated_watts":10,"estimated_seconds":1}"#;
    let resp = send_and_receive(&addr, task);
    let result: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(result["status"], "Success");
    assert_eq!(result["output"], "hello world");

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn test_agent_task_bench_sum() {
    let (mut child, port) = start_agent();
    let addr = format!("127.0.0.1:{}", port);

    let task = r#"{"id":"t2","name":"bench_sum","payload":"10000","estimated_watts":10,"estimated_seconds":1}"#;
    let resp = send_and_receive(&addr, task);
    let result: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(result["status"], "Success");
    assert_eq!(result["output"], "49995000");

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn test_agent_multiple_tasks() {
    let (mut child, port) = start_agent();
    let addr = format!("127.0.0.1:{}", port);

    // Send 5 tasks sequentially
    for i in 0..5 {
        let task = format!(
            r#"{{"id":"t{}","name":"echo","payload":"msg{}","estimated_watts":10,"estimated_seconds":1}}"#,
            i, i
        );
        let resp = send_and_receive(&addr, &task);
        let result: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(result["status"], "Success");
        assert_eq!(result["output"], format!("msg{}", i));
    }

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn test_three_agent_cluster() {
    let mut agents = Vec::new();
    for _ in 0..3 {
        agents.push(start_agent());
    }

    // Verify all three are alive
    for (_, port) in &agents {
        let addr = format!("127.0.0.1:{}", port);
        let resp = send_and_receive(&addr, "ping");
        assert_eq!(resp, "pong");
    }

    // Verify all three have telemetry
    for (_, port) in &agents {
        let addr = format!("127.0.0.1:{}", port);
        let resp = send_and_receive(&addr, "telemetry");
        let tel: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(tel["hostname"].is_string());
    }

    // Cleanup
    for (mut child, _) in agents {
        child.kill().ok();
        child.wait().ok();
    }
}

#[test]
fn test_shell_client_agent_probe() {
    let (mut child, port) = start_agent();
    let addr = format!("127.0.0.1:{}", port);

    let alive = ouro_shell::agent_client::ping(&addr).unwrap();
    assert!(alive);

    let tel = ouro_shell::agent_client::telemetry(&addr).unwrap();
    assert!(!tel.hostname.is_empty());
    assert!(tel.ram_total_mib > 0);

    let task = ouro_shell::agent_client::AgentTask {
        id: "test1".into(),
        name: "echo".into(),
        payload: "shell_client_test".into(),
        estimated_watts: 10,
        estimated_seconds: 1,
    };
    let result = ouro_shell::agent_client::execute(&addr, &task).unwrap();
    assert_eq!(result.status, "Success");
    assert_eq!(result.output, "shell_client_test");

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn test_acts_and_shard_tasks_over_tcp() {
    use ouro_cluster::bmts::{write_shard, BmtsTensor};
    use ouro_cluster::pipeline::{to_hex, Activation};

    let (mut child, port) = start_agent();
    let addr = format!("127.0.0.1:{}", port);

    let act = Activation {
        sequence: 1,
        token_pos: 0,
        layer_start: 0,
        layer_end: 29,
        data: vec![0.5; 2560],
    };
    let task = ouro_shell::agent_client::AgentTask {
        id: "acts".into(),
        name: "acts_echo".into(),
        payload: to_hex(&act.encode()),
        estimated_watts: 5,
        estimated_seconds: 5,
    };
    let result = ouro_shell::agent_client::execute(&addr, &task).unwrap();
    assert_eq!(result.status, "Success");
    assert!(result.output.contains("elems=2560"));

    let dir = std::env::temp_dir().join("ouro_it_shard");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_1.bmts");
    write_shard(
        path.to_str().unwrap(),
        1,
        &[BmtsTensor { name: "blk.0.t".into(), shape: vec![2, 2], dtype: 34, offset: 0, length: 4 }],
        &[1, 2, 3, 4],
    )
    .unwrap();
    let task = ouro_shell::agent_client::AgentTask {
        id: "shard".into(),
        name: "load_shard".into(),
        payload: path.to_str().unwrap().into(),
        estimated_watts: 5,
        estimated_seconds: 5,
    };
    let result = ouro_shell::agent_client::execute(&addr, &task).unwrap();
    assert_eq!(result.status, "Success");
    assert!(result.output.contains("shard node=1"));

    std::fs::remove_dir_all(&dir).ok();
    child.kill().ok();
    child.wait().ok();
}
