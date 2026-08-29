# OurobourOS — Agent Guidelines

## Quick Reference

### Commands
- **Build**: `cargo build --release`
- **Test**: `cargo test --lib`
- **Run shell**: `cargo run --bin ouro-shell`
- **Run agent**: `cargo run --bin ouro-agent`
- **Clippy**: `cargo clippy -- -D warnings`

### Workspace Structure
```
OurobourOS/
├── cluster/    — Core library (Beast, scheduler, probes, transport)
├── shell/      — Interactive REPL (dot notation, context memory)
├── agent/      — Node daemon (task execution, telemetry)
├── workloads/  — Briev workload files (.bv)
├── nixos/      — NixOS deployment configs
└── tools/      — Scripts (probe, deploy, bench)
```

## Architecture Philosophy

### The Cluster Is One Machine
Users interact with the cluster as a single entity. The shell hides
multi-node complexity behind dot notation and propositions.

### Waste Is Fuel
Old hardware is not trash. The scheduler routes work to the best available
node based on architecture, SIMD capability, and energy budget.

### Energy Is a First-Class Constraint
Every scheduling decision considers power draw. The cluster has a budget.
No assignment exceeds it.

### Self-Describing
The cluster topology is embedded in itself as Beast S-expressions.
The system knows what it is.

## Code Conventions

### Rust Style
- Early returns over else-if chains
- Bundle parameters into context structs when > 5 params
- Split functions exceeding 15 cyclomatic complexity
- Use `anyhow::Result` for error propagation
- Derive `Debug`, `Clone`, `Serialize`, `Deserialize` on data types

### Beast Format
- S-expression nested lists: `(tag child1 child2 ...)`
- Atoms are strings, integers, floats, or booleans
- Keys sorted alphabetically for deterministic output
- Quote strings with `"double quotes"`

### Transport Layer
- MVP uses TCP with custom binary protocol
- Phase 2 uses raw L2 Ethernet (EtherType 0x88B5)
- All frames start with magic bytes `0x4F55524F` ("OURO")
- Sequence IDs prevent frame reordering

### Probe Modules
- Local probes read `/proc/cpuinfo`, `/proc/meminfo`, `/sys/class/powercap/`
- Remote probes use SSH to read the same files
- GPU probes use `nvidia-smi --query-gpu=... --format=csv,noheader`

## Anti-Patterns (NEVER DO)

- Bypassing the energy budget scheduler
- Hardcoding node IPs (use discovery)
- Storing model weights in git
- Using shared libraries in worker nodes (static linking only)
- Modifying `.praetor/shadow-results.json` directly

## Correct Approach

- Route all scheduling through `Scheduler::schedule()`
- Use `probe_node()` for discovery
- Store weights in BMTS format on raw block devices
- Use Nix for reproducible builds
- Let Praetor enforce complexity and architecture checks

## Praetor Enforcement

Praetor is installed as an LSP server and runs on every keystroke.

**Active checks:**
- Cyclomatic complexity ≤ 15
- Cognitive complexity ≤ 15
- Nesting depth ≤ 6
- Parameter count ≤ 5
- Big-O flags O(n²) or worse
- Datalog rules for auth and data leaks

**Disabled:**
- Intent comment enforcement (Rust doc comments not recognized)

**Shadow escape hatch:**
If a check cannot be satisfied by refactoring:
```rust
// praetor-shadow: original=my_function
fn my_function_v2(...) { ... }
```
Then run `praetor verify --shadow <file>` to benchmark.

## Commands

```bash
praetor init              # Set up .praetor/ + pre-commit hook
praetor report --target . # Full project report
praetor validate --warn   # CI gate
praetor verify --shadow   # Benchmark shadow functions
```

## File Types

| Extension | Purpose | Location |
|-----------|---------|----------|
| `.rs` | Rust source | `cluster/`, `shell/`, `agent/` |
| `.bv` | Briev workload | `workloads/` |
| `.beast` | Cluster state | Generated at runtime |
| `.bmts` | Weight format | Raw block devices |
| `.nix` | NixOS config | `nixos/` |
