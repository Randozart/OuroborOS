# OurobourOS — Complete Specification

**Version:** 0.1.0-draft
**Date:** 2026-08-28
**Status:** Pre-implementation
**Enhanced Plan:** See `PLAN.md` for research findings, gap analysis, and revised implementation roadmap (2026-08-29)

---

## 1. Name and Philosophy

### The Name

**OurobourOS** — from the Ouroboros, the ancient symbol of a serpent devouring its own tail.

The snake eats itself and is reborn. Old CPUs are the food. Compute is the rebirth. The system describes itself in Beast, heals itself when nodes die, and never stops. It is self-referential, cyclical, and a little mythological.

**Casual name:** "Ouro" (as in "the Ouro is running").

**Logo concept:** A serpent coiled around a cluster of nodes, its tail entering its mouth. Each node is a vertebra.

### Design Principles

1. **Waste is fuel.** Old hardware is not trash — it is unrefined compute. The Ouro refines it.
2. **The cluster is one machine.** The user does not see 12 laptops. The user sees a single computational organism.
3. **Poke at it.** The shell is a dialogue, not a command line. You ask questions, it answers. You propose truths, it makes them real.
4. **CPUs over GPUs.** We seek workloads where branch prediction, recursion, and fine-grained synchronization beat SIMD parallelism. The CPU is not obsolete — it is misunderstood.
5. **Energy is a first-class constraint.** Every scheduling decision considers power draw. The cluster has a budget, and it sticks to it.
6. **Self-describing.** The cluster topology is embedded in itself as Beast. The system knows what it is.
7. **Self-healing.** When a node dies, the system reconfigures around it. No manual intervention.
8. **Playful.** Poetry mode. The system has a voice. Computing should be joyful.

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    OurobourOS Shell                          │
│  Propositional interface │ Context memory │ Dot notation     │
│  "What is your proposition?"                                │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cluster Beast                             │
│  Topology + node state as S-exprs                           │
│  Probed at boot │ Live-patchable │ Inspectable              │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Briev Reactor (Scheduler)                 │
│  Work arrives → precondition fires → route to node          │
│  Energy-bounded: [total_power < budget] as contract         │
│  Workload-class-aware: branch │ recursive │ SIMD │ irregular│
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Transport Layer                           │
│  SSH (MVP) │ Custom TCP (Phase 2)                           │
│  Task dispatch │ Result collection │ Heartbeat              │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Node Agent (per laptop)                   │
│  NixOS base │ Task executor │ Telemetry daemon              │
│  Intel RAPL │ Thermal monitoring │ SIMD capability report   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Descriptions

#### 2.2.1 Ouro Shell

The user-facing interface. A REPL that speaks in propositions and answers questions about the cluster.

**Key properties:**
- Dot notation: `n3.power?`
- Context memory: after `n3`, bare `power?` means `n3.power?`
- `?` queries, `.` declares
- Visual formatting: columns, pipes, status indicators
- Poetry mode: optional poetic responses

#### 2.2.2 Cluster Beast

The cluster's self-description. A Beast (S-expression) data structure that describes:
- Which nodes exist
- What each node can do (CPU, RAM, SIMD, energy)
- What each node is currently doing
- The cluster's energy budget
- Which workloads are running where

**Key properties:**
- Probed at boot from all nodes
- Live-patched when state changes (node joins, dies, starts work)
- Inspectable via the shell (`cluster?` reads the Beast)
- Serializable to disk for persistence

#### 2.2.3 Briev Reactor (Scheduler)

The scheduling engine. Built on the Briev reactor model: dependency-driven, reactive, contract-enforced.

**Key properties:**
- Work arrives → precondition fires → route to best node
- Workload class detection: determines if a task is branch-heavy, recursive, SIMD-friendly, etc.
- Energy budget as a contract: `[cluster_power < budget]`
- Fault tolerance: if a node dies, condition raised, work redistributed

#### 2.2.4 Transport Layer

The communication pipe between master and workers.

**MVP:** SSH-based task dispatch. ~1ms overhead per task. Proven, reliable, works everywhere.

**Phase 2:** Custom TCP protocol. Binary task dispatch (Beast → compressed → send → execute → return). ~10x lower overhead.

#### 2.2.5 Node Agent

A daemon running on each laptop. Responsibilities:
- Accept task dispatch from master
- Execute tasks (SIMD-optimized where possible)
- Report telemetry: power draw (Intel RAPL), temperature, load, availability
- Heartbeat: periodic "I'm alive" signal
- Graceful shutdown on休眠

---

## 3. Shell Syntax — Complete Reference

### 3.1 Discovery

```
# Cluster summary
> cluster?
CLUSTER
  Nodes:  12 total │ 3 active │ 9 idle
  Power:  72W / 500W (85% headroom)
  Work:   2 running │ 0 queued

# Node discovery
> n3?
NODE_3
  CPU:    Kaby Lake i5-7200U
  RAM:    16GB DDR4-2400
  SIMD:   SSE4.2, AVX2
  Status: IDLE
  Power:  12W │ Temp: 42°C
  Accepts: assign,休眠, status, remove

# Workload discovery
> branch_sort?
WORKLOAD: branch_sort.bv
  Class:  BRANCH_HEAVY
  Best on: Nodes with strong branch prediction
  Requires: { AVX2, 4GB RAM }
  Est. time: 45s on Haswell
```

### 3.2 Deep Queries (Dot Notation)

```
> n3.power?
12W

> n3.thermal?
42°C (threshold: 100°C)

> n3.simd?
SSE4.2, AVX2

> n3.ram?
16GB DDR4-2400

> n3.status?
IDLE

> n3.load?
0.00 (1-min avg)
```

### 3.3 Context Memory

```
> n3                    # set context to n3
NODE_3 selected.

> power?                # means n3.power?
12W

> thermal?              # means n3.thermal?
42°C

> status?               # means n3.status?
IDLE

> simd?                 # means n3.simd?
SSE4.2, AVX2

> cluster               # reset context to cluster
CLUSTER context.
```

### 3.4 Propositions (Make Things True — End with `.`)

```
# Assign a workload to a node
> n3 assign branch_sort.
THE COUNSEL ACTS:
  [1] Serialize branch_sort.bv.              [OK]
  [2] Check: n3 supports branch_prediction.  [YES]
  [3] Check: 72W + 28W < 500W.             [YES]
  [4] Dispatch to n3.                        [OK]
RESULT: branch_sort assigned to n3. [TRUE]

# Change power state
> n3休眠.
Node休眠. Power: 12W → 2W.

# Set energy budget
> budget 400w.
Cluster power budget: 400W. [SET]

# Remove a node from the cluster
> n7 remove.
Node_7 removed from cluster. [VACANT]
```

### 3.5 Queries (Ask If True — End with `?`)

```
# Would this assignment work?
> n3 assign branch_sort?
THE COUNSEL CONSIDERS:
  [1] n3 supports branch_prediction.         [YES]
  [2] Power budget allows +28W.             [YES]
  [3] n3 is available.                       [YES]
RESULT: This proposition CAN be satisfied. [TRUE]

# Is the power budget satisfied?
> cluster power < 500w?
TRUE │ 72W / 500W

# Which nodes can run this workload?
> branch_sort on?
  n3:  Kaby Lake  │ AVX2 │ AVAILABLE │ est. 45s
  n7:  Ivy Bridge │ AVX  │ AVAILABLE │ est. 62s
  n11: Haswell    │ AVX2 │ AVAILABLE │ est. 38s
```

### 3.6 Bulk Queries

```
# All active nodes
> cluster.active?
  n3:  Kaby Lake  │ WORKING │ 31W │ recursive_tree.bv
  n7:  Ivy Bridge │ WORKING │ 28W │ branch_sort.bv
  n11: Haswell    │ WORKING │ 23W │ matrix_multiply.bv

# All idle nodes
> cluster.idle?
  n1:  Haswell     │ IDLE │ 23W │ AVX2
  n2:  Sandy Bridge│ IDLE │ 18W │ SSE4.2
  n4:  Ivy Bridge  │ IDLE │ 25W │ AVX
  ...

# All nodes by power draw
> cluster.power?
  n2:  18W (Sandy Bridge)
  n1:  23W (Haswell)
  n4:  25W (Ivy Bridge)
  n7:  28W (Ivy Bridge) ← working
  n3:  31W (Kaby Lake) ← working
  ...
  TOTAL: 72W / 500W
```

### 3.7 System Commands

```
# Probe all nodes
> probe.
Probing all nodes... [DONE]
  n1:  Haswell i5-4200U, 8GB, AVX2         [FOUND]
  n2:  Sandy Bridge i5-2520M, 4GB, SSE4.2  [FOUND]
  n3:  Kaby Lake i5-7200U, 16GB, AVX2      [FOUND]
  ...

# Deploy node-agent to all laptops
> deploy.
Deploying node-agent to all nodes... [DONE]

# Save cluster state
> save.
Cluster state saved to cluster.beast. [DONE]

# Load cluster state
> load.
Cluster state loaded from cluster.beast. [DONE]
```

### 3.8 Poetry Mode

```
# Enable poetry mode
> poetry on.
Poetry mode enabled.

# Cluster summary (poetic)
> cluster?
The cluster breathes. 12 nodes. 3 draw breath.
The rest dream of silicon and electrons.

# Node休眠 (poetic)
> n3休眠.
Node休眠. Its fan slows. The heat fades. 12W → 2W. A small silence.

# Workload assignment (poetic)
> n3 assign branch_sort.
The Counsel considers. The contract holds.
branch_sort finds a home in Node_3. 28W added to the dream.

# Disable poetry mode
> poetry off.
Poetry mode disabled.
```

### 3.9 Shorthand Table

| Long | Short | Notes |
|---|---|---|
| `Node_3` | `n3` | `n` + node number |
| `Cluster` | `cl` | |
| `power` | `p` | |
| `thermal` | `t` | |
| `status` | `s` | |
| `branch_sort` | `bs` | Workloads only |
| `recursive_tree` | `rt` | Workloads only |
| `cluster.active` | `cl.a` | |
| `cluster.idle` | `cl.i` | |
| `cluster.power` | `cl.p` | |

### 3.10 Syntax Rules

| Pattern | Ends with | Meaning |
|---|---|---|
| `subject?` | `?` | Discovery — what is this? |
| `subject.property?` | `?` | Deep query — what is this property? |
| `subject predicate object.` | `.` | Proposition — make this true |
| `subject predicate object?` | `?` | Query — would this be true? |
| `subject休眠` | — | Power state change |
| `?` alone | `?` | Cluster summary |
| `property?` (bare) | `?` | Query current context's property |
| `poetry on/off` | — | Toggle poetry mode |
| `probe.` | — | Probe all nodes |
| `deploy.` | — | Deploy node-agent |
| `save.` | — | Save cluster state |
| `load.` | — | Load cluster state |
| `budget Nw.` | — | Set energy budget |

---

## 4. File Structure

```
OurobourOS/
├── CLUSTER.md                          # This specification
├── Cargo.toml                          # Workspace root
│
├── cluster/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       │
│       ├── beast/
│       │   ├── mod.rs
│       │   ├── topology.rs             # Cluster topology as Beast
│       │   └── node_state.rs           # Live node state
│       │
│       ├── scheduler/
│       │   ├── mod.rs                  # Cluster scheduler (reactor-based)
│       │   ├── workload_class.rs       # Workload classification
│       │   └── energy_budget.rs        # Power-aware scheduling
│       │
│       ├── transport/
│       │   ├── mod.rs                  # Transport abstraction trait
│       │   ├── ssh.rs                  # SSH-based task dispatch (MVP)
│       │   └── tcp.rs                  # Custom TCP protocol (Phase 2)
│       │
│       ├── probe/
│       │   ├── mod.rs                  # Node discovery + capability probing
│       │   ├── cpu.rs                  # CPUID detection
│       │   ├── memory.rs               # RAM detection
│       │   ├── energy.rs               # Intel RAPL power telemetry
│       │   └── network.rs              # Network latency/bandwidth
│       │
│       └── error.rs                    # Cluster error types
│
├── shell/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                     # Ouro Shell REPL entry point
│       ├── parser.rs                   # Cluster proposition parser
│       ├── context.rs                  # Context memory
│       ├── formatter.rs                # Visual output formatter
│       ├── inspector.rs                # Live state inspector
│       └── propositions.rs             # Proposition handlers
│
├── agent/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                     # Node agent daemon
│       ├── executor.rs                 # Task execution engine
│       └── telemetry.rs                # Telemetry daemon
│
├── workloads/
│   ├── branch_sort.bv                  # CPU-wins: branch-heavy merge sort
│   ├── recursive_tree.bv               # CPU-wins: tree traversal
│   ├── small_batch.bv                  # CPU-wins: small-batch operations
│   ├── irregular_graph.bv              # CPU-wins: graph BFS
│   └── matrix_multiply.bv              # GPU-wins control (baseline)
│
├── nixos/
│   ├── flake.nix                       # NixOS flake
│   ├── master.nix                      # Master node configuration
│   ├── worker.nix                      # Worker node configuration
│   └── common.nix                      # Shared configuration
│
└── tools/
    ├── probe_nodes.sh                  # Quick node discovery
    ├── deploy.sh                       # Deploy to all laptops
    └── bench.sh                        # Benchmark runner
```

---

## 5. Implementation Phases

### Phase 1: Probe + Discover (Weeks 1-2)

**Goal:** Run a command, get a Beast file describing your cluster.

| Step | File | Description |
|---|---|---|
| 1.1 | `Cargo.toml` | Create workspace with members: cluster, shell, agent |
| 1.2 | `cluster/probe/cpu.rs` | Read `/proc/cpuinfo`, detect AVX/SSE, core count |
| 1.3 | `cluster/probe/memory.rs` | Read `/proc/meminfo`, detect DDR type/speed |
| 1.4 | `cluster/probe/energy.rs` | Read Intel RAPL via `/sys/class/powercap/intel-rapl/` |
| 1.5 | `cluster/probe/network.rs` | Ping mesh, measure latency between nodes |
| 1.6 | `cluster/probe/mod.rs` | Combine probes, SSH into remote nodes |
| 1.7 | `cluster/beast/topology.rs` | Serialize cluster topology as Beast S-exprs |
| 1.8 | `tools/probe_nodes.sh` | Shell script: probe all nodes, output `cluster.beast` |

**Deliverable:** `bash tools/probe_nodes.sh` → `cluster.beast`

### Phase 2: Ouro Shell (Weeks 3-4)

**Goal:** Interactive shell with dot notation and context memory.

| Step | File | Description |
|---|---|---|
| 2.1 | `shell/parser.rs` | LL(1) lexer + parser for all proposition types |
| 2.2 | `shell/context.rs` | Context memory, shorthand expansion |
| 2.3 | `shell/formatter.rs` | Column formatting, status indicators, poetry mode |
| 2.4 | `shell/inspector.rs` | Read Cluster Beast, generate discovery responses |
| 2.5 | `shell/propositions.rs` | Handlers: assign,休眠, budget, probe, deploy, save, load |
| 2.6 | `shell/main.rs` | REPL loop, banner, error handling |

**Deliverable:** Run shell, see cluster state, poke at nodes.

### Phase 3: Scheduler + Transport (Weeks 5-8)

**Goal:** Submit workload, watch it route, get results.

| Step | File | Description |
|---|---|---|
| 3.1 | `cluster/scheduler/workload_class.rs` | Classify workloads: BRANCH_HEAVY, RECURSIVE, etc. |
| 3.2 | `cluster/scheduler/energy_budget.rs` | Enforce power budget as contract |
| 3.3 | `cluster/scheduler/mod.rs` | Reactor-based scheduler: work → precondition → route |
| 3.4 | `cluster/transport/mod.rs` | Transport trait: dispatch, heartbeat |
| 3.5 | `cluster/transport/ssh.rs` | SSH task dispatch, timeout, retry |
| 3.6 | `agent/executor.rs` | Load workload Beast, execute, return result |
| 3.7 | `agent/telemetry.rs` | RAPL power, thermal, load reporting |
| 3.8 | `agent/main.rs` | Daemon: accept tasks, heartbeat, graceful shutdown |

**Deliverable:** `n3 assign branch_sort.` → task dispatched → result returned.

### Phase 4: Workloads + Benchmarks (Weeks 9-12)

**Goal:** Prove CPUs beat GPUs on specific workloads.

| Step | File | Description |
|---|---|---|
| 4.1 | `workloads/branch_sort.bv` | Merge sort with unpredictable pivot |
| 4.2 | `workloads/recursive_tree.bv` | Binary tree traversal (pointer chasing) |
| 4.3 | `workloads/small_batch.bv` | 1000x small independent tasks |
| 4.4 | `workloads/irregular_graph.bv` | BFS on power-law graph |
| 4.5 | `workloads/matrix_multiply.bv` | Dense matmul (GPU-wins control) |
| 4.6 | `tools/bench.sh` | Run all workloads, output comparison table |

**Deliverable:** Benchmark results: CPU vs. GPU on each workload class.

### Phase 5: Custom Transport (Optional, Week 13+)

**Goal:** Replace SSH with faster protocol.

| Step | File | Description |
|---|---|---|
| 5.1 | `cluster/transport/tcp.rs` | TCP server, binary protocol, lz4 compression |
| 5.2 | `agent/main.rs` | Add TCP listener alongside SSH |
| 5.3 | `cluster/scheduler/mod.rs` | Switch to TCP transport |

**Deliverable:** ~10x lower dispatch overhead.

---

## 6. Reused Code

### 6.1 From briev-backend-foundation

| What | File | How We Use It |
|---|---|---|
| S-expression parser | `src/beast/sexpr.rs` | Copy `tokenize()`, `parse()`, `SExpr` types |
| Beast serializer | `src/beast/serialize.rs` | Reference for Beast format |
| Beast deserializer | `src/beast/deserialize.rs` | Reference for Beast parsing |
| Reactor | `src/reactor.rs` | Reference for scheduler design |

### 6.2 From moore-kernel

| What | File | How We Use It |
|---|---|---|
| msh parser | `kernel/msh/src/parser.rs` | Reference for LL(1) parser |
| msh main | `kernel/msh/src/main.rs` | Reference for REPL loop |
| Tether engine | `kernel/msh/src/tether.rs` | Reference for state queries |

### 6.3 From briev-compiler-baseline

| What | File | How We Use It |
|---|---|---|
| Queue stdlib | `lib/std/queue.bv` | Use directly in workloads |
| Telemetry pattern | `benchmarks/telemetry_stream.bv` | Reference for agent telemetry |

### 6.4 From VITRIOL

| What | File | How We Use It |
|---|---|---|
| Hardware probe | `libvitriol/src/probe.rs` | Adapt for cluster probing |

---

## 7. NixOS Configuration

### 7.1 Flake Structure

```nix
# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    nixosConfigurations = {
      master = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [ ./master.nix ];
      };
      worker = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [ ./worker.nix ];
      };
    };
  };
}
```

### 7.2 Deployment Steps

```bash
# 1. On master: generate SSH key
ssh-keygen -t ed25519 -f ~/.ssh/ouro-master

# 2. On each worker: install NixOS with worker.nix

# 3. On master: copy SSH key to each worker
for ip in 192.168.1.{101..112}; do
  ssh-copy-id -i ~/.ssh/ouro-master.pub ouro@$ip
done

# 4. On master: build and deploy node-agent
cargo build --release --bin node-agent
for ip in 192.168.1.{101..112}; do
  scp target/release/node-agent ouro@$ip:~/
  ssh ouro@$ip "sudo cp ~/node-agent /usr/local/bin/ouro-agent && sudo systemctl enable ouro-agent"
done

# 5. On master: run probe
cargo run --bin ouro-shell -- probe.
```

---

## 8. Workload Specifications

### 8.1 CPU-Wins Workloads

#### branch_sort.bv — Branch-Heavy Merge Sort

- **Why CPU wins:** GPU branch divergence kills throughput
- **Pattern:** Recursive merge sort with unpredictable pivot selection
- **Benchmark:** Sort 1M random integers

#### recursive_tree.bv — Tree Traversal

- **Why CPU wins:** Pointer chasing, no coalesced memory access
- **Pattern:** Binary tree in-order traversal
- **Benchmark:** Traverse 10M node tree

#### small_batch.bv — Small-Batch Operations

- **Why CPU wins:** GPU kernel launch overhead ~10-50μs, CPU ~1ns
- **Pattern:** 1000x independent encrypt/decrypt operations
- **Benchmark:** 1000x AES-256 blocks

#### irregular_graph.bv — Graph BFS

- **Why CPU wins:** Load imbalance on irregular degree distribution
- **Pattern:** BFS on power-law graph
- **Benchmark:** BFS on 1M node graph

### 8.2 GPU-Wins Control

#### matrix_multiply.bv — Dense Matrix Multiply

- **Why GPU wins:** Massive parallelism, contiguous memory
- **Pattern:** 1024x1024 dense matrix multiply
- **Baseline:** Compare single-node vs. GPU theoretical

---

## 9. Open Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | Master node | One of 12 laptops vs. main PC | Main PC |
| 2 | Network topology | Switch vs. direct | Switch |
| 3 | IP addressing | DHCP vs. static | Static |
| 4 | Node agent auth | SSH key vs. mTLS | SSH key |
| 5 | Beast format | Text S-exprs vs. binary | Text (MVP) |
| 6 | Workload input | .bv files vs. binaries | .bv files |
| 7 | Result format | Text vs. Beast | Beast |
| 8 | Poetry mode default | On vs. off | Off |
