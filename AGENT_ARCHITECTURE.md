# EcholoKernel Agent Architecture

## Overview

EcholoKernel is an iterative optimization agent designed to automatically improve Triton/NVSHMEM kernel implementations. The agent follows a bootstrap → evaluate → propose → refine cycle, using LLM-based code generation with optional RAG (Retrieval-Augmented Generation) support for both example kernels and documentation.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Configuration Layer"
        A[Experiment Config YAML] --> B[ExperimentConfig]
        B --> C[OpenAI Model]
        B --> D[Evaluation Settings]
        B --> E[Retrieval Settings]
        B --> F[Strategy Components]
    end
    
    subgraph "Core Optimizer"
        G[Optimizer] --> H[Bootstrap Stage]
        G --> I[Evaluator]
        G --> J[Proposer]
        G --> K[Memory Strategy]
        G --> L[Scorer]
    end
    
    subgraph "Retrieval Systems"
        M[LLM Retriever] --> N[Triton Corpus]
        O[RLM Doc Retriever] --> P[Docs Corpus]
    end
    
    B --> G
    F --> J
    F --> K
    F --> L
    E --> M
    E --> O
```

## Component Breakdown

### 1. Configuration System

The agent is driven by YAML configuration files that define:

- **Problems**: Problem IDs with dimensions (rows, cols) and data types
- **Strategies**: Which proposer, memory, and scorer to use
- **OpenAI Settings**: Model selection (e.g., gpt-5) and temperature
- **Evaluation**: Warmup/measurement iterations, timeouts, profiling
- **Retrieval**: Optional RAG for example kernels and documentation

**Example Configuration:**

```yaml
name: "problem30_agent_speedup"
max_iterations: 15
early_stop_speedup: 1.5
seed_from_backend: "triton"
candidate_dir: "solutions_agent"

problems:
  - problem_id: 30
    rows: 1024
    cols: 1024
    dtype: "float32"

retrieval:
  enabled: true
  top_k: 3
  docs_dir: "scraped_docs"
  rlm_max_iterations: 15
```

### 2. Bootstrap Stage

The bootstrap stage generates an initial candidate solution when none exists.

```mermaid
sequenceDiagram
    participant O as Optimizer
    participant B as Bootstrap
    participant C as OpenAI Client
    participant R as RLM Retriever
    
    O->>B: ensure_candidate(problem_id)
    
    alt Candidate file already exists
        B-->>O: Return existing file path
    else Seed from backend exists
        B->>B: Copy seed file
        B-->>O: Return copied file path
    else Generate new
        B->>B: Load reference implementation
        B->>B: Collect context examples from solved problems
        B->>R: Retrieve relevant NVSHMEM docs
        R-->>B: Documentation context
        B->>C: Generate initial candidate with context
        C-->>B: Generated code
        B->>B: Save candidate file
        B-->>O: Return generated file path
    end
```

**Bootstrap Process:**
1. Check if candidate file exists (skip if present)
2. Try to copy from seed backend (e.g., `solutions_triton/`)
3. If no seed, generate from scratch using:
   - Reference implementation (CPU/NumPy version)
   - Solved examples from other problems
   - Retrieved NVSHMEM documentation (if RLM enabled)

### 3. Main Optimization Loop

```mermaid
flowchart TD
    Start[Start Experiment] --> Bootstrap[Bootstrap All Problems]
    Bootstrap --> PrecompRef[Precompute Reference Results]
    PrecompRef --> IterStart{Iteration < Max?}
    
    IterStart -->|Yes| Evaluate[Evaluate All Candidates]
    Evaluate --> CheckStop{Score >= Early Stop?}
    
    CheckStop -->|Yes| EndExperiment[End Experiment]
    CheckStop -->|No| ProcessProblems[Process Each Problem]
    
    ProcessProblems --> Snapshot[Snapshot Current Code]
    Snapshot --> CheckFailed{Eval Failed?}
    
    CheckFailed -->|Yes| Rollback1[Rollback to Previous]
    CheckFailed -->|No| Propose
    
    Rollback1 --> Propose[Generate Proposal]
    Propose --> RLMRetrieval[RLM Doc Retrieval Optional]
    RLMRetrieval --> Apply[Apply New Code]
    
    Apply --> ValidateCode{Valid Code?}
    ValidateCode -->|No| Rollback2[Rollback]
    ValidateCode -->|Yes| SaveSnapshot[Save Snapshot]
    
    SaveSnapshot --> NextProblem{More Problems?}
    Rollback2 --> NextProblem
    
    NextProblem -->|Yes| ProcessProblems
    NextProblem -->|No| IterStart
    
    IterStart -->|No| EndExperiment
```

### 4. Evaluation System

The evaluator runs candidates on Modal (distributed GPU infrastructure) and compares them against reference implementations.

```mermaid
sequenceDiagram
    participant O as Optimizer
    participant E as ModalEvaluator
    participant M as Modal Infrastructure
    participant L as Logs
    
    O->>E: evaluate_pair(problem)
    
    E->>E: Check cached reference
    
    alt Reference not cached
        E->>M: Run reference backend
        M-->>L: Write logs & profiling
        M-->>E: Return summary
        E->>E: Cache reference result
    end
    
    E->>M: Run candidate backend with cached reference
    M-->>L: Write logs & profiling
    M-->>E: Return summary
    
    alt Timeout
        E->>E: Create timeout error result
    else Subprocess error
        E->>E: Create error result
    else Success
        E->>E: Parse summary JSON
        E->>E: Extract correctness & performance
    end
    
    E-->>O: Return (reference, candidate) results
```

**Evaluation Results Include:**
- **Correctness**: Whether outputs match reference
- **Performance**: Timing data (mean, min, max, p50, p90, p99)
- **Status**: Success, error, or timeout
- **Profiling Data**: Optional GPU traces
- **Error Details**: Crash messages, timeout indicators

### 5. Proposal System

The proposer generates code improvements based on evaluation feedback.

```mermaid
flowchart TD
    Start[Proposal Request] --> CheckMode{Correctness OK?}
    
    CheckMode -->|No| CorrectMode[Mode: correctness_fix]
    CheckMode -->|Yes| PerfMode[Mode: perf_opt]
    
    CorrectMode --> BuildContext1[Build Proposal Context]
    PerfMode --> BuildContext2[Build Proposal Context]
    
    BuildContext1 --> RLMQuery1[Build RLM Query from Errors]
    BuildContext2 --> RLMQuery2[Build RLM Query from Performance]
    
    RLMQuery1 --> RLMRetrieval{RLM Enabled?}
    RLMQuery2 --> RLMRetrieval
    
    RLMRetrieval -->|Yes| RLMCall[Retrieve Relevant Docs]
    RLMRetrieval -->|No| SkipRLM[No docs retrieved]
    
    RLMCall --> BuildPrompt
    SkipRLM --> BuildPrompt[Build Prompt with Template]
    
    BuildPrompt --> RAGCheck{RAG Enabled?}
    
    RAGCheck -->|Yes| ToolUse[LLM with search_examples tool]
    RAGCheck -->|No| DirectLLM[Direct LLM Call]
    
    ToolUse --> ToolLoop{Model calls tool?}
    ToolLoop -->|Yes| Retrieve[Retrieve from Triton Corpus]
    Retrieve --> ToolLoop
    ToolLoop -->|No| ParseResponse
    
    DirectLLM --> ParseResponse[Parse JSON Response]
    
    ParseResponse --> Return[Return Proposal]
```

**Proposal Context:**
- Current candidate code
- Latest evaluation metrics
- Memory summary (historical best results)
- Retrieved documentation (RLM)
- Proposal mode (correctness vs. performance)

**Proposal Output:**
- **Diagnosis**: What's wrong or could be improved
- **Hypotheses**: Potential optimization strategies
- **Candidate Code**: Complete rewritten file
- **Test Expectations**: What should change after applying this

### 6. Retrieval Systems

EcholoKernel uses two retrieval systems:

#### A. Triton Corpus Retrieval (RAG for Examples)

```mermaid
graph LR
    A[Query from LLM] --> B[LLM Retriever]
    B --> C[Rank Corpus Entries]
    C --> D[Select Top-K]
    D --> E[Format Code Pairs]
    E --> F[Return to LLM]
    
    G[Triton Corpus] --> H[Reference + Triton Solution Pairs]
    H --> B
```

- Corpus contains paired (reference, optimized) Triton solutions
- LLM ranks examples by relevance to query
- Top-K most relevant examples returned
- Available as `search_examples` tool during proposal

#### B. Documentation Retrieval (RLM for Docs)

```mermaid
graph TB
    A[Build Query from Eval Feedback] --> B[RLM Retriever]
    B --> C{Iteration < Max?}
    
    C -->|Yes| D[Select Relevant Doc Pages]
    D --> E[Read Page Content]
    E --> F[Append to Context]
    F --> G[Generate Follow-up Query]
    G --> C
    
    C -->|No| H[Concatenate Retrieved Docs]
    H --> I[Return to Proposer]
    
    J[Docs Corpus] --> K[Markdown Pages + Manifest]
    K --> D
```

- RLM (Recursive Learning Module) iteratively retrieves documentation
- Query built from evaluation feedback (errors, timeouts, correctness issues)
- Multiple iterations narrow down to most relevant pages
- Retrieved docs injected into proposal prompt

### 7. Memory & Scoring

#### Memory Strategy

```mermaid
graph LR
    A[Full Evaluation History] --> B[Memory Strategy]
    B --> C{Strategy Type}
    
    C -->|best_so_far| D[Find Best Score per Problem]
    C -->|full_history| E[Summarize All Results]
    
    D --> F[Format Summary]
    E --> F
    F --> G[Return to Proposer]
```

**Best-So-Far Strategy:**
- Tracks highest-scoring iteration for each problem
- Provides context of "what worked best" to guide optimization

#### Scorer

```mermaid
graph TB
    A[Reference Summary] --> C[Scorer]
    B[Candidate Summary] --> C
    
    C --> D{Scorer Type}
    
    D -->|speedup_mean| E[Calculate Mean Speedup]
    D -->|custom| F[Custom Metric]
    
    E --> G[Return Score]
    F --> G
```

**Speedup Mean Scorer:**
- Compares candidate vs. reference timing
- Score = reference_time / candidate_time
- Score > 1.0 means speedup
- Score < 1.0 means slowdown

### 8. Rollback & Error Handling

```mermaid
flowchart TD
    Start[Code Generation] --> Snapshot[Snapshot Current Code]
    Snapshot --> GenerateNew[Generate New Code]
    GenerateNew --> Validate{Valid?}
    
    Validate -->|No - Empty| Rollback1[Rollback]
    Validate -->|No - No solution| Rollback2[Rollback]
    Validate -->|Yes| Apply[Write to File]
    
    Apply --> Eval[Evaluate]
    Eval --> CheckResult{Eval Result?}
    
    CheckResult -->|Timeout| RollbackTimeout[Rollback Before Next Iteration]
    CheckResult -->|Error| RollbackError[Rollback Before Next Iteration]
    CheckResult -->|Success| Keep[Keep Changes]
    
    Rollback1 --> Log1[Log Failure]
    Rollback2 --> Log2[Log Failure]
    RollbackTimeout --> Log3[Log Eval Failure]
    RollbackError --> Log4[Log Eval Failure]
```

**Rollback Manager:**
- Takes snapshot before each code change
- Rolls back on:
  - Invalid code generation
  - Evaluation errors
  - Evaluation timeouts
- Prevents building on broken code

## Data Flow

### End-to-End Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Config
    participant Optimizer
    participant Bootstrap
    participant Evaluator
    participant Proposer
    participant RLM
    participant RAG
    participant Modal
    participant Logger
    
    User->>Config: Load YAML
    Config->>Optimizer: Initialize with config
    
    Optimizer->>Bootstrap: ensure_candidate(problem)
    Bootstrap->>RLM: Retrieve docs (optional)
    RLM-->>Bootstrap: NVSHMEM documentation
    Bootstrap->>Bootstrap: Generate initial code
    Bootstrap-->>Optimizer: Candidate file
    
    Optimizer->>Evaluator: precompute_references()
    Evaluator->>Modal: Run reference backend
    Modal-->>Evaluator: Reference results
    
    loop For each iteration (1 to max_iterations)
        Optimizer->>Evaluator: evaluate_all()
        Evaluator->>Modal: Run candidate backend
        Modal-->>Evaluator: Candidate results
        Evaluator-->>Optimizer: Scores & feedback
        
        Optimizer->>Optimizer: Check early stop
        
        alt Early stop achieved
            Optimizer->>Logger: Log early stop
            Optimizer-->>User: Experiment complete
        else Continue optimization
            loop For each problem
                Optimizer->>Optimizer: Snapshot code
                
                alt Eval failed
                    Optimizer->>Optimizer: Rollback code
                end
                
                Optimizer->>Proposer: Generate proposal
                Proposer->>RLM: Retrieve docs (optional)
                RLM-->>Proposer: Documentation
                Proposer->>RAG: search_examples (optional)
                RAG-->>Proposer: Similar kernels
                Proposer->>Proposer: Generate new code
                Proposer-->>Optimizer: Proposal
                
                Optimizer->>Optimizer: Apply code
                
                alt Code invalid
                    Optimizer->>Optimizer: Rollback
                else Code valid
                    Optimizer->>Optimizer: Save snapshot
                end
                
                Optimizer->>Logger: Log proposal & result
            end
        end
    end
    
    Optimizer->>Logger: Write summary
    Optimizer-->>User: Run directory with results
```

## Key Design Principles

### 1. **Iterative Refinement**
- Start with bootstrap or seed solution
- Evaluate → Propose → Apply → Repeat
- Each iteration builds on previous best results

### 2. **Dual-Mode Optimization**
- **Correctness Mode**: Fixes bugs, crashes, incorrect outputs
- **Performance Mode**: Optimizes speed while maintaining correctness
- Automatic mode switching based on evaluation feedback

### 3. **Failure Recovery**
- Rollback on invalid code generation
- Rollback after evaluation failures
- Continue optimization from last good state

### 4. **Context-Aware Retrieval**
- Dynamically retrieve relevant documentation based on errors
- Search example kernels on-demand during proposal
- Iterative documentation narrowing (RLM)

### 5. **Structured Logging**
- All proposals logged as JSONL
- Code snapshots for every iteration
- Complete evaluation traces
- Enables post-hoc analysis and replay

## File Organization

```
EcholoKernel/
├── agent/
│   ├── core/
│   │   ├── optimizer.py        # Main orchestration loop
│   │   ├── bootstrap.py        # Initial code generation
│   │   ├── rollback.py         # Snapshot & rollback manager
│   │   └── run_log.py          # JSONL logging
│   ├── strategies/
│   │   ├── proposers/
│   │   │   ├── base.py         # Proposer protocol
│   │   │   └── single_shot.py  # Standard LLM proposer
│   │   ├── memory/
│   │   │   ├── base.py         # Memory protocol
│   │   │   └── best_so_far.py  # Track best results
│   │   ├── scorers/
│   │   │   ├── base.py         # Scorer protocol
│   │   │   └── speedup_mean.py # Speedup metric
│   │   ├── retrieval/
│   │   │   ├── corpus.py       # Triton example corpus
│   │   │   ├── retriever.py    # LLM-based ranker
│   │   │   ├── docs_loader.py  # Documentation corpus
│   │   │   └── rlm_retriever.py # Recursive doc retriever
│   │   └── registry.py         # Strategy factory
│   ├── eval/
│   │   └── modal_evaluator.py  # Distributed evaluation
│   ├── config.py               # Configuration dataclasses
│   └── openai_client.py        # LLM API wrapper
├── experiments/
│   └── *.yaml                  # Experiment configs
├── runs/
│   └── <timestamp>_<name>/
│       ├── run_log.jsonl       # Event log
│       ├── llm_outputs.jsonl   # Proposal details
│       ├── snapshots/          # Code history
│       │   └── problem_<id>/
│       │       ├── iter_0_bootstrap.py
│       │       ├── iter_1.py
│       │       └── ...
│       └── summary.json        # Run metadata
├── solutions_agent/            # Current best candidates
├── solutions_triton/           # Optional seed solutions
├── reference/                  # Ground truth implementations
└── scraped_docs/               # NVSHMEM documentation
    ├── manifest.json
    └── markdown/
```

## Running the Agent

### Basic Execution

```bash
python -m agent.core.optimizer experiments/problem30.yaml
```

### Output Artifacts

After execution, the agent produces:

1. **Run Directory**: `runs/<timestamp>_<experiment_name>/`
2. **Event Log**: `run_log.jsonl` - all optimization events
3. **LLM Outputs**: `llm_outputs.jsonl` - detailed proposals
4. **Code Snapshots**: `snapshots/problem_<id>/` - evolution history
5. **Evaluation Logs**: `logs/problem_<id>/` - profiling & traces
6. **Final Candidates**: `solutions_agent/<id>_agent.py` - best solutions

### Monitoring Progress

The agent prints a progress bar showing:
- Current step / total steps
- Stage description (bootstrap, eval, patch cycle)
- Elapsed time
- Estimated time remaining

Example:
```
[############################] 45/50 completed iteration 3 patch cycle for problem 30 | elapsed 342.1s | eta 38.5s
```

## Configuration Options

### Essential Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `max_iterations` | Number of refinement cycles | 3 |
| `early_stop_speedup` | Stop if score exceeds this | None |
| `seed_from_backend` | Backend to copy seeds from | "triton" |
| `openai.model` | LLM model to use | "gpt-5" |
| `retrieval.enabled` | Enable RAG retrieval | False |
| `retrieval.top_k` | Number of examples to retrieve | 3 |
| `eval.eval_timeout_s` | Max evaluation time | 600 |

### Strategy Components

| Component | Options | Purpose |
|-----------|---------|---------|
| `proposer` | `single_shot` | Code generation strategy |
| `memory` | `best_so_far`, `none` | Historical context |
| `scorer` | `speedup_mean` | Performance metric |

## Extensibility

The agent uses protocol-based design for easy extension:

### Adding a New Proposer

```python
from agent.strategies.proposers.base import Proposer, ProposalContext

class MyProposer:
    def propose(self, ctx: ProposalContext) -> dict:
        # Your proposal logic here
        return {
            "diagnosis": [...],
            "hypotheses": [...],
            "candidate_code": "...",
            "test_expectations": [...]
        }
```

### Adding a New Scorer

```python
from agent.strategies.scorers.base import Scorer

class MyScorer:
    def score(self, reference_summary: dict, candidate_summary: dict) -> float:
        # Your scoring logic here
        return computed_score
```

### Registering New Strategies

In `agent/strategies/registry.py`:

```python
def make_proposer(name: str, *args, **kwargs):
    if name == "my_proposer":
        return MyProposer(*args, **kwargs)
    # ... existing proposers
```

## Debugging & Troubleshooting

### Common Issues

**1. Evaluation Timeouts**
- Increase `eval.eval_timeout_s` in config
- Check for deadlocks in candidate code
- Review RLM-retrieved docs for synchronization patterns

**2. Invalid Code Generation**
- Review `llm_outputs.jsonl` for proposal details
- Check if prompt templates are clear
- Increase temperature for more exploration

**3. No Improvement**
- Enable retrieval: `retrieval.enabled: true`
- Increase `max_iterations`
- Review memory strategy effectiveness

### Analyzing Results

```bash
# View event log
cat runs/<run_dir>/run_log.jsonl | jq

# View LLM proposals
cat runs/<run_dir>/llm_outputs.jsonl | jq

# Compare code evolution
diff snapshots/problem_30/iter_1.py snapshots/problem_30/iter_5.py

# View evaluation traces
cat logs/problem_30/agent/summary_rank0.json | jq
```

## Future Enhancements

Potential areas for extension:

1. **Multi-objective Optimization**: Balance speed, memory, accuracy
2. **Ensemble Proposals**: Generate multiple candidates per iteration
3. **Genetic Algorithms**: Crossover between successful solutions
4. **Active Learning**: Learn from evaluation patterns
5. **Hierarchical Search**: Coarse-grained then fine-grained optimization
6. **Constraint-based Generation**: Hard limits on memory, registers, etc.

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Maintained by**: EcholoKernel Team
