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

    let (status, output) = match task.name.as_str() {
        "echo" => (TaskStatus::Success, task.payload.clone()),
        "bench_sum" => (TaskStatus::Success, run_bench_sum(&task.payload)),
        #[cfg(feature = "bitnet")]
        "bitnet_generate" => match run_bitnet_generate(&task.payload) {
            Ok(out) => (TaskStatus::Success, out),
            Err(e) => (TaskStatus::Failed, format!("bitnet error: {}", e)),
        },
        _ => (
            TaskStatus::Failed,
            format!("unknown task: {}", task.name),
        ),
    };

    let elapsed = start.elapsed();
    Ok(TaskResult {
        task_id: task.id.clone(),
        status,
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

/// One-token-per-line generation guard: llama_decode is not thread-safe.
#[cfg(feature = "bitnet")]
fn model_slot() -> &'static std::sync::Mutex<Option<bitnet_rs::BitNetModel>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<bitnet_rs::BitNetModel>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

/// Run BitNet text generation. Payload: "prompt|max_tokens|temp".
/// Requires OURO_MODEL_PATH env var pointing to a BitNet GGUF file.
#[cfg(feature = "bitnet")]
fn run_bitnet_generate(payload: &str) -> Result<String> {
    use bitnet_rs::SamplingParams;

    let model_path = std::env::var("OURO_MODEL_PATH")
        .map_err(|_| anyhow::anyhow!("OURO_MODEL_PATH not set"))?;

    let mut parts = payload.splitn(3, '|');
    let prompt = parts.next().unwrap_or("");
    let max_tokens: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let temp: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let params = if temp > 0.0 {
        SamplingParams { temp, top_k: 40, top_p: 0.95, seed: 42 }
    } else {
        SamplingParams::greedy()
    };

    let n_ctx: u32 = std::env::var("OURO_N_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let n_threads: u32 = std::env::var("OURO_N_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let mut guard = model_slot()
        .lock()
        .map_err(|_| anyhow::anyhow!("model slot poisoned"))?;

    if guard.is_none() {
        let model = bitnet_rs::BitNetModel::load(&model_path, n_ctx, n_threads)?;
        *guard = Some(model);
    }

    let model = guard.as_ref().expect("model just loaded");
    model.generate_with(prompt, max_tokens, &params)
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
        assert_eq!(result.status, TaskStatus::Failed);
        assert!(result.output.contains("unknown task"));
    }

    #[test]
    #[cfg(feature = "bitnet")]
    fn test_bitnet_requires_model_path() {
        std::env::remove_var("OURO_MODEL_PATH");
        let task = Task {
            id: "t4".into(),
            name: "bitnet_generate".into(),
            payload: "Hello|4".into(),
            estimated_watts: 30,
            estimated_seconds: 10,
        };
        let result = execute(&task).unwrap();
        assert_eq!(result.status, TaskStatus::Failed);
        assert!(result.output.contains("OURO_MODEL_PATH"));
    }

    #[test]
    #[ignore] // Requires real model via OURO_MODEL_PATH
    #[cfg(feature = "bitnet")]
    fn test_bitnet_real_generation() {
        if std::env::var("OURO_MODEL_PATH").is_err() {
            eprintln!("OURO_MODEL_PATH unset, skipping");
            return;
        }
        let task = Task {
            id: "t5".into(),
            name: "bitnet_generate".into(),
            payload: "The capital of France is|12".into(),
            estimated_watts: 35,
            estimated_seconds: 30,
        };
        let result = execute(&task).unwrap();
        eprintln!("bitnet output: {}", result.output);
        assert_eq!(result.status, TaskStatus::Success);
        assert!(!result.output.is_empty());
    }
}
