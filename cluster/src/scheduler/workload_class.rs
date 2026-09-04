use serde::{Deserialize, Serialize};

/// Classification of a workload's computational character.
///
/// This drives scheduling: each class maps to nodes with the
/// architectural features that give CPUs an edge over GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadClass {
    /// Unpredictable branches — CPU branch predictor wins over GPU divergence.
    BranchHeavy,
    /// Deep recursion / pointer chasing — CPUs have stacks, GPUs do not.
    Recursive,
    /// Dense, contiguous parallelism — GPUs actually win here (baseline).
    SimdFriendly,
    /// Irregular access patterns — load imbalance hurts GPUs.
    Irregular,
    /// Many small independent tasks — GPU launch overhead dominates.
    SmallBatch,
    /// Ternary LLM matvec — AVX2 LUT kernels beat legacy GPUs.
    LlmInference,
    /// GPU compute — dense parallelism dispatched to a tail's GPU.
    GpuCompute,
    /// Not yet classified.
    Unknown,
}

impl WorkloadClass {
    /// Classify a workload from its metadata name/annotation.
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("branch") || lower.contains("sort") {
            WorkloadClass::BranchHeavy
        } else if lower.contains("recurs") || lower.contains("tree") || lower.contains("travers") {
            WorkloadClass::Recursive
        } else if lower.contains("matmul") || lower.contains("matrix") {
            WorkloadClass::SimdFriendly
        } else if lower.contains("graph") || lower.contains("bfs") || lower.contains("dfs") {
            WorkloadClass::Irregular
        } else if lower.contains("batch") || lower.contains("small") {
            WorkloadClass::SmallBatch
        } else if lower.contains("llm")
            || lower.contains("bitnet")
            || lower.contains("inference")
            || lower.contains("generate")
        {
            WorkloadClass::LlmInference
        } else if lower.contains("gpu") || lower.contains("cuda") || lower.contains("opencl") {
            WorkloadClass::GpuCompute
        } else {
            WorkloadClass::Unknown
        }
    }

    /// Human-readable label for shell output.
    pub fn label(&self) -> &'static str {
        match self {
            WorkloadClass::BranchHeavy => "BRANCH_HEAVY",
            WorkloadClass::Recursive => "RECURSIVE",
            WorkloadClass::SimdFriendly => "SIMD_FRIENDLY",
            WorkloadClass::Irregular => "IRREGULAR",
            WorkloadClass::SmallBatch => "SMALL_BATCH",
            WorkloadClass::LlmInference => "LLM_INFERENCE",
            WorkloadClass::GpuCompute => "GPU_COMPUTE",
            WorkloadClass::Unknown => "UNKNOWN",
        }
    }

    /// Whether GPUs lose to CPUs for this class (our thesis).
    pub fn cpu_advantage(&self) -> bool {
        !matches!(self, WorkloadClass::SimdFriendly | WorkloadClass::GpuCompute | WorkloadClass::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_branch_sort() {
        assert_eq!(WorkloadClass::from_name("branch_sort"), WorkloadClass::BranchHeavy);
    }

    #[test]
    fn test_classify_recursive_tree() {
        assert_eq!(WorkloadClass::from_name("recursive_tree"), WorkloadClass::Recursive);
    }

    #[test]
    fn test_classify_matmul() {
        assert_eq!(WorkloadClass::from_name("matrix_multiply"), WorkloadClass::SimdFriendly);
    }

    #[test]
    fn test_classify_graph() {
        assert_eq!(WorkloadClass::from_name("irregular_graph"), WorkloadClass::Irregular);
    }

    #[test]
    fn test_classify_batch() {
        assert_eq!(WorkloadClass::from_name("small_batch"), WorkloadClass::SmallBatch);
    }

    #[test]
    fn test_unknown() {
        assert_eq!(WorkloadClass::from_name("mystery_task"), WorkloadClass::Unknown);
    }

    #[test]
    fn test_cpu_advantage() {
        assert!(WorkloadClass::BranchHeavy.cpu_advantage());
        assert!(WorkloadClass::Recursive.cpu_advantage());
        assert!(!WorkloadClass::SimdFriendly.cpu_advantage());
        assert!(!WorkloadClass::GpuCompute.cpu_advantage());
    }
}

#[cfg(test)]
mod llm_tests {
    use super::*;

    #[test]
    fn test_classify_llm_inference() {
        assert_eq!(WorkloadClass::from_name("bitnet_generate"), WorkloadClass::LlmInference);
        assert_eq!(WorkloadClass::from_name("llm_prompt"), WorkloadClass::LlmInference);
        assert_eq!(WorkloadClass::LlmInference.label(), "LLM_INFERENCE");
        assert!(WorkloadClass::LlmInference.cpu_advantage());
    }

    #[test]
    fn test_classify_gpu_compute() {
        assert_eq!(WorkloadClass::from_name("gpu_matvec"), WorkloadClass::GpuCompute);
        assert_eq!(WorkloadClass::from_name("cuda_forward"), WorkloadClass::GpuCompute);
        assert_eq!(WorkloadClass::from_name("opencl_kernel"), WorkloadClass::GpuCompute);
        assert_eq!(WorkloadClass::GpuCompute.label(), "GPU_COMPUTE");
        assert!(!WorkloadClass::GpuCompute.cpu_advantage());
    }
}
