/// Format cluster and node data for terminal output.
pub struct Formatter {
    poetry: bool,
}

impl Formatter {
    pub fn new(poetry: bool) -> Self {
        Self { poetry }
    }

    pub fn set_poetry(&mut self, poetry: bool) {
        self.poetry = poetry;
    }

    /// Format cluster summary.
    pub fn cluster_summary(
        &self,
        total_nodes: usize,
        active_nodes: usize,
        power_watts: u32,
        budget_watts: u32,
        running_tasks: usize,
        queued_tasks: usize,
    ) -> String {
        if self.poetry {
            return self.poetry_cluster_summary(
                total_nodes,
                active_nodes,
                power_watts,
                budget_watts,
            );
        }

        let idle = total_nodes - active_nodes;
        let pct = if budget_watts > 0 {
            (power_watts as f64 / budget_watts as f64 * 100.0) as u32
        } else {
            0
        };

        format!(
            "CLUSTER\n  Nodes:  {} total | {} active | {} idle\n  Power:  {}W / {}W ({}% headroom)\n  Work:   {} running | {} queued",
            total_nodes, active_nodes, idle, power_watts, budget_watts, 100 - pct, running_tasks, queued_tasks
        )
    }

    fn poetry_cluster_summary(
        &self,
        total_nodes: usize,
        active_nodes: usize,
        power_watts: u32,
        budget_watts: u32,
    ) -> String {
        format!(
            "The cluster breathes. {} nodes. {} draw breath.\nThe rest dream of silicon and electrons.\nPower: {}W / {}W",
            total_nodes, active_nodes, power_watts, budget_watts
        )
    }

    /// Format node discovery output.
    pub fn node_query(&self, node: &NodeDisplay) -> String {
        format!(
            "NODE_{}\n  CPU:    {}\n  RAM:    {}MiB\n  SIMD:   {}\n  Status: {}\n  Power:  {}W | Temp: {}C",
            node.id,
            node.cpu_model,
            node.ram_mib,
            simd_list(node),
            node.status,
            node.power_watts,
            node.temp_c
        )
    }

    /// Format property query.
    pub fn property_query(&self, node: &str, property: &str, value: &str) -> String {
        format!("{}.{} = {}", node, property, value)
    }

    /// Format bulk query results.
    pub fn bulk_query(&self, filter: &str, nodes: &[NodeDisplay]) -> String {
        let mut out = format!("{}\n", filter.to_uppercase());
        for node in nodes {
            out.push_str(&format!(
                "  {}: {} | {} | {}W\n",
                node.id, node.cpu_model, node.status, node.power_watts
            ));
        }
        out
    }

    /// Format assignment proposition result.
    pub fn assign_result(&self, node: &str, workload: &str, success: bool, details: &[String]) -> String {
        if self.poetry {
            return self.poetry_assign_result(node, workload, success);
        }

        let mut out = "THE COUNSEL ACTS:\n".to_string();
        for (i, detail) in details.iter().enumerate() {
            out.push_str(&format!("  [{}] {}\n", i + 1, detail));
        }
        if success {
            out.push_str(&format!("RESULT: {} assigned to {}. [TRUE]", workload, node));
        } else {
            out.push_str("RESULT: Assignment failed. [FALSE]");
        }
        out
    }

    fn poetry_assign_result(&self, node: &str, workload: &str, success: bool) -> String {
        if success {
            format!(
                "The Counsel considers. The contract holds.\n{} finds a home in {}. A small warmth.",
                workload, node
            )
        } else {
            format!("The Counsel considers. The contract fails.\n{} cannot find rest.", workload)
        }
    }

    /// Format budget set confirmation.
    pub fn budget_set(&self, watts: u32) -> String {
        format!("Cluster power budget: {}W. [SET]", watts)
    }

    /// Format probe result.
    pub fn probe_result(&self, nodes: &[NodeDisplay]) -> String {
        let mut out = "Probing all nodes... [DONE]\n".to_string();
        for node in nodes {
            out.push_str(&format!(
                "  {}: {}, {}MiB, {} [FOUND]\n",
                node.id, node.cpu_model, node.ram_mib, simd_list(node)
            ));
        }
        out
    }

    /// Format context set message.
    pub fn context_set(&self, node: &str) -> String {
        format!("{} selected.", node)
    }

    /// Format context reset message.
    pub fn context_reset(&self) -> String {
        "CLUSTER context.".to_string()
    }

    /// Format poetry mode toggle.
    pub fn poetry_toggle(&self, enabled: bool) -> String {
        if enabled {
            "Poetry mode enabled.".to_string()
        } else {
            "Poetry mode disabled.".to_string()
        }
    }

    /// Format unknown command.
    pub fn unknown(&self, input: &str) -> String {
        format!("Unknown command: {}", input)
    }
}

/// Display-ready node data.
pub struct NodeDisplay {
    pub id: String,
    pub cpu_model: String,
    pub ram_mib: u64,
    pub has_avx2: bool,
    pub has_avx: bool,
    pub has_sse42: bool,
    pub status: String,
    pub power_watts: u32,
    pub temp_c: u32,
}

fn simd_list(node: &NodeDisplay) -> String {
    let mut parts = Vec::new();
    if node.has_avx2 {
        parts.push("AVX2");
    }
    if node.has_avx {
        parts.push("AVX");
    }
    if node.has_sse42 {
        parts.push("SSE4.2");
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_summary() {
        let fmt = Formatter::new(false);
        let out = fmt.cluster_summary(4, 2, 89, 500, 2, 0);
        assert!(out.contains("4 total"));
        assert!(out.contains("2 active"));
        assert!(out.contains("89W / 500W"));
    }

    #[test]
    fn test_poetry_cluster_summary() {
        let fmt = Formatter::new(true);
        let out = fmt.cluster_summary(4, 2, 89, 500, 2, 0);
        assert!(out.contains("The cluster breathes"));
    }

    #[test]
    fn test_node_query() {
        let fmt = Formatter::new(false);
        let node = NodeDisplay {
            id: "n3".to_string(),
            cpu_model: "i5-7200U".to_string(),
            ram_mib: 16384,
            has_avx2: true,
            has_avx: true,
            has_sse42: true,
            status: "IDLE".to_string(),
            power_watts: 12,
            temp_c: 42,
        };
        let out = fmt.node_query(&node);
        assert!(out.contains("i5-7200U"));
        assert!(out.contains("16384MiB"));
        assert!(out.contains("AVX2"));
    }

    #[test]
    fn test_budget_set() {
        let fmt = Formatter::new(false);
        assert_eq!(fmt.budget_set(400), "Cluster power budget: 400W. [SET]");
    }
}
