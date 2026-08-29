use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A task dispatched by the master node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub payload: String,
    pub estimated_watts: u32,
    pub estimated_seconds: u32,
}

/// Result of executing a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub output: String,
    pub elapsed_ms: u64,
    pub peak_watts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Success,
    Failed,
    Timeout,
}

/// Execute a task locally.
pub fn execute(task: &Task) -> Result<TaskResult> {
    let start = Instant::now();
    let peak_watts = task.estimated_watts;

    let output = match task.name.as_str() {
        "echo" => task.payload.clone(),
        "bench_sum" => run_bench_sum(&task.payload),
        _ => format!("unknown task: {}", task.name),
    };

    let elapsed = start.elapsed();
    Ok(TaskResult {
        task_id: task.id.clone(),
        status: TaskStatus::Success,
        output,
        elapsed_ms: elapsed.as_millis() as u64,
        peak_watts,
    })
}

/// Simple benchmark: sum integers up to N.
fn run_bench_sum(payload: &str) -> String {
    let n: u64 = payload.trim().parse().unwrap_or(1_000_000);
    let sum: u64 = (0..n).sum();
    sum.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_echo() {
        let task = Task {
            id: "t1".into(),
            name: "echo".into(),
            payload: "hello".into(),
            estimated_watts: 10,
            estimated_seconds: 1,
        };
        let result = execute(&task).unwrap();
        assert_eq!(result.status, TaskStatus::Success);
        assert_eq!(result.output, "hello");
    }

    #[test]
    fn test_execute_bench_sum() {
        let task = Task {
            id: "t2".into(),
            name: "bench_sum".into(),
            payload: "1000".into(),
            estimated_watts: 10,
            estimated_seconds: 1,
        };
        let result = execute(&task).unwrap();
        assert_eq!(result.output, "499500");
    }

    #[test]
    fn test_execute_unknown() {
        let task = Task {
            id: "t3".into(),
            name: "unknown_task".into(),
            payload: "".into(),
            estimated_watts: 10,
            estimated_seconds: 1,
        };
        let result = execute(&task).unwrap();
        assert!(result.output.contains("unknown task"));
    }
}
