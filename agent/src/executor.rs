use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A task dispatched by the head node.
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
        "load_shard" => match run_load_shard(&task.payload) {
            Ok(out) => (TaskStatus::Success, out),
            Err(e) => (TaskStatus::Failed, format!("shard error: {}", e)),
        },
        "acts_echo" => match run_acts_echo(&task.payload) {
            Ok(out) => (TaskStatus::Success, out),
            Err(e) => (TaskStatus::Failed, format!("acts error: {}", e)),
        },
        #[cfg(feature = "bitnet")]
        "bitnet_generate" => match run_bitnet_generate(&task.payload) {
            Ok(out) => (TaskStatus::Success, out),
            Err(e) => (TaskStatus::Failed, format!("bitnet error: {}", e)),
        },
        #[cfg(feature = "bitnet")]
        "tokenize" => match run_tokenize(&task.payload) {
            Ok(out) => (TaskStatus::Success, out),
            Err(e) => (TaskStatus::Failed, format!("tokenize error: {}", e)),
        },
        #[cfg(feature = "bitnet")]
        "detok" => match run_detok(&task.payload) {
            Ok(out) => (TaskStatus::Success, out),
            Err(e) => (TaskStatus::Failed, format!("detok error: {}", e)),
        },
        "stage_setup" | "stage_reset" | "stage_token" | "stage_step" | "stage_sample" => {
            match crate::stage::handle(task.name.as_str(), &task.payload) {
                Ok(out) => (TaskStatus::Success, out),
                Err(e) => (TaskStatus::Failed, format!("stage error: {}", e)),
            }
        }
        #[cfg(feature = "gpu")]
        "gpu_selftest" => match run_gpu_selftest() {
            Ok(out) => (TaskStatus::Success, out),
            Err(e) => (TaskStatus::Failed, format!("gpu error: {}", e)),
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

/// Tokenize using a vocab-only model load (cheap: no weights).
#[cfg(feature = "bitnet")]
fn vocab_slot() -> &'static std::sync::Mutex<Option<bitnet_rs::BitNetModel>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<bitnet_rs::BitNetModel>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(feature = "bitnet")]
fn with_vocab<T>(f: impl FnOnce(&bitnet_rs::BitNetModel) -> T) -> Result<T> {
    let model_path = std::env::var("OURO_MODEL_PATH")
        .map_err(|_| anyhow::anyhow!("OURO_MODEL_PATH not set"))?;
    let mut guard = vocab_slot()
        .lock()
        .map_err(|_| anyhow::anyhow!("vocab slot poisoned"))?;
    if guard.is_none() {
        *guard = Some(bitnet_rs::BitNetModel::load_vocab_only(&model_path)?);
    }
    Ok(f(guard.as_ref().unwrap()))
}

#[cfg(feature = "bitnet")]
fn run_tokenize(payload: &str) -> Result<String> {
    let text = payload.strip_prefix('|').unwrap_or(payload);
    let ids = with_vocab(|m| m.tokenize(text, true))?;
    Ok(ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","))
}

#[cfg(feature = "bitnet")]
fn run_detok(payload: &str) -> Result<String> {
    let mut out = String::new();
    for part in payload.split(',') {
        if let Ok(id) = part.trim().parse::<bitnet_rs::LlamaToken>() {
            out.push_str(&with_vocab(|m| m.token_to_piece(id))?);
        }
    }
    Ok(out)
}

/// Simple benchmark: sum integers up to N.
/// Run the Q6_K parity selftest on THIS node's GPU: deterministic
/// matrix through the OpenCL kernel vs the CPU reference, cos gate.
/// This is the on-node proof that a tail's GPU computes truthfully
/// (GPU_CLAIM.md WP-G4) — and the first distributed GPU compute.
#[cfg(feature = "gpu")]
fn run_gpu_selftest() -> Result<String> {
    use crate::gpu::{cosine, deterministic_payload, deterministic_x, GpuPool};
    use ouro_cluster::infer::{matvec_q, QuantKind};
    use std::time::Instant;

    let mut pool = GpuPool::new()?;
    let (out_len, in_len) = (16usize, 1024usize);
    let payload = deterministic_payload(out_len, in_len);
    pool.upload_q6k("selftest", &payload, out_len, in_len)?;
    let x = deterministic_x(in_len);

    let t0 = Instant::now();
    let gpu = pool.matvec("selftest", &x)?;
    let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let cpu = matvec_q(&payload, QuantKind::Q6K, out_len, in_len, &x);
    let cpu_ms = t1.elapsed().as_secs_f64() * 1e3;

    let cos = cosine(&gpu, &cpu);
    let pass = cos > 0.9999;
    let json = serde_json::json!({
        "adapter": pool.adapter_name,
        "out_len": out_len,
        "in_len": in_len,
        "cos": (cos * 1e8).round() / 1e8,
        "gate": 0.9999,
        "pass": pass,
        "gpu_ms": (gpu_ms * 1000.0).round() / 1000.0,
        "cpu_ms": (cpu_ms * 1000.0).round() / 1000.0,
    });
    if !pass {
        anyhow::bail!("parity gate failed: cos = {cos}");
    }
    Ok(json.to_string())
}

fn run_bench_sum(payload: &str) -> String {
    let n: u64 = payload.trim().parse().unwrap_or(1_000_000);
    let sum: u64 = (0..n).sum();
    sum.to_string()
}

/// Validate + summarize a BMTS shard at the given path.
fn run_load_shard(path: &str) -> Result<String> {
    use ouro_cluster::bmts::BmtsShard;
    let shard = BmtsShard::open(path)?;
    let first = shard.tensors.first().map(|t| t.name.as_str()).unwrap_or("-");
    let last = shard.tensors.last().map(|t| t.name.as_str()).unwrap_or("-");
    Ok(format!(
        "shard node={} tensors={} data={}MB first={} last={}",
        shard.node,
        shard.tensors.len(),
        shard.data_len() / 1_000_000,
        first,
        last
    ))
}

/// Decode an ACTS activation frame (hex), echo its stats, re-encode, return hex.
/// Proves activation transport end-to-end between pipeline stages.
fn run_acts_echo(hex_payload: &str) -> Result<String> {
    use ouro_cluster::pipeline::{from_hex, to_hex, Activation};
    let bytes = from_hex(hex_payload.trim())?;
    let act = Activation::decode(&bytes)?;
    let stats = format!(
        "seq={} pos={} layers={}-{} elems={}",
        act.sequence, act.token_pos, act.layer_start, act.layer_end, act.data.len()
    );
    let roundtrip = act.encode();
    if roundtrip != bytes {
        anyhow::bail!("ACTS roundtrip mismatch");
    }
    Ok(format!("{} hex_back={}", stats, to_hex(&roundtrip).len()))
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
    fn test_load_shard_task() {
        use ouro_cluster::bmts::{write_shard, BmtsTensor};
        let dir = std::env::temp_dir().join("ouro_agent_shard_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_1.bmts");
        let tensors = vec![BmtsTensor {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![4, 4],
            dtype: 34,
            offset: 0,
            length: 16,
        }];
        write_shard(path.to_str().unwrap(), 1, &tensors, &[7u8; 16]).unwrap();

        let task = Task {
            id: "s1".into(),
            name: "load_shard".into(),
            payload: path.to_str().unwrap().into(),
            estimated_watts: 5,
            estimated_seconds: 1,
        };
        let result = execute(&task).unwrap();
        assert_eq!(result.status, TaskStatus::Success);
        assert!(result.output.contains("shard node=1"));
        assert!(result.output.contains("blk.0.attn_q.weight"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_shard_missing_fails() {
        let task = Task {
            id: "s2".into(),
            name: "load_shard".into(),
            payload: "/nonexistent/shard.bmts".into(),
            estimated_watts: 5,
            estimated_seconds: 1,
        };
        let result = execute(&task).unwrap();
        assert_eq!(result.status, TaskStatus::Failed);
    }

    #[test]
    fn test_acts_echo_task() {
        use ouro_cluster::pipeline::{to_hex, Activation};
        let act = Activation {
            sequence: 1,
            token_pos: 2,
            layer_start: 0,
            layer_end: 9,
            data: vec![0.25, -0.5, 1.0, 2.0],
        };
        let task = Task {
            id: "a1".into(),
            name: "acts_echo".into(),
            payload: to_hex(&act.encode()),
            estimated_watts: 5,
            estimated_seconds: 1,
        };
        let result = execute(&task).unwrap();
        assert_eq!(result.status, TaskStatus::Success);
        assert!(result.output.contains("elems=4"));
        assert!(result.output.contains("layers=0-9"));
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
