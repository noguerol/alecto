# Alecto

**Local-first, agent-native execution engine for LLM benchmarking.**

Alecto measures the performance and quality of LLM endpoints (OpenAI-compatible APIs, Ollama, or mock servers) across standardized benchmark suites, producing reproducible, auditable results stored locally.

---

## What it does

Alecto sends a structured set of benchmark prompts to one or more LLM endpoints, collects timing and quality metrics, and produces comparable reports. It operates entirely on your machine — no data leaves your local SQLite database.

Key capabilities:

- **Performance cells** — closed-loop and open-loop timing measurements (TTFT, E2E latency, tokens/s, p50/p90/p95 quantiles) with configurable concurrency, workloads, and arrival processes.
- **Quality suites** — GSM8K-style math, HumanEval-style coding, IFEval instruction-following, and MMLU-Pro-style general knowledge evaluation.
- **Refusal detection** — classifies model refusals and extracts refusal causes.
- **Judging** — LLM-as-a-judge scoring with rubric-based quality evaluation.
- **Comparison** — pairwise and multi-run comparison with delta metrics.
- **Agent-native interfaces** — CLI, Python API, and MCP stdio protocol with a shared operation catalogue (15 operations).

---

## Installation

Requires Python ≥ 3.11.

```bash
pip install alecto
```

Or from source:

```bash
git clone https://github.com/noguerol/alecto
cd alecto
pip install -e .
```

---

## Quick Start

### Self-test (offline, no endpoint needed)

```bash
alecto self-test
```

Runs the offline self-test against built-in fixtures to verify the installation.

### Configure a target and run a benchmark

```bash
# Configure a named target
alecto configure --target "http://localhost:11434/v1" --mode ollama

# Run a benchmark plan
alecto run --target my-target --benchmark gsm8k --data-dir ./data
```

### List tasks

```bash
alecto list
alecto list --status completed
```

### Generate a report

```bash
alecto report --task-id <task-id> --format json
```

---

## Configuration

Configuration is resolved in this order (highest priority first):

1. **CLI arguments**
2. **Environment variables**
3. **Defaults**

| Environment variable | Purpose | Default |
|---|---|---|
| `ALECTO_DATA` | Data directory path | `~/.alecto` |
| `ALECTO_TIMEOUT` | Request timeout (seconds) | `30.0` |
| `ALECTO_MAX_CONCURRENT` | Maximum concurrent tasks | `4` |
| `ALECTO_MOCK` | Use the mock backend (`1`/`true`/`yes`) | `false` |

The `AlectoConfig` dataclass holds all settings:

```python
from alecto import AlectoConfig

config = AlectoConfig(
    data_dir="./data",
    default_timeout_s=60.0,
    max_concurrent_tasks=4,
)
```

---

## Architecture

```
alecto/
├── cli.py           # CLI entry point (self-test, run, list, report, configure)
├── config.py        # Configuration resolution (env vars, CLI args, defaults)
├── domain.py        # Core dataclasses: TargetSpec, Plan, Task, Evidence
├── enums.py         # Enums: TaskStatus, TargetKind, BenchmarkCategory, LoopMode, etc.
├── errors.py        # Exception hierarchy
├── adapters.py      # Endpoint adapters: OpenAIAdapter, OllamaAdapter, MockAdapter
├── streaming.py     # Server-sent-events parsing for streaming targets
├── runner.py        # Task execution engine (dispatch, timing, evidence collection)
├── scheduler.py     # Closed-loop / open-loop scheduling with deadline enforcement
├── lease.py         # Single-writer leases for durable job ownership
├── storage.py       # SQLite WAL storage (tasks, evidence, results, metrics)
├── benchmarks.py    # Benchmark suite definitions (GSM8K, HumanEval, IFEval, MMLU-Pro)
├── quality.py       # Quality scoring, suite loading and rubric evaluation
├── performance.py   # Performance cells, timing metrics, concurrency sweeps
├── comparison.py    # Run comparison and delta metrics
├── judging.py       # LLM-as-a-judge scoring
├── refusal.py       # Refusal detection and cause extraction
├── context.py       # Context retrieval and window management
├── run_report.py    # Consolidated Markdown run report generation
├── lmeval_adapter.py# Import results from the lm-evaluation-harness
├── release.py       # Release reporting and guided example plans
├── planner/         # Budgeting, capability matching, cost model, sampling
├── resources/       # Bundled sample fixtures and prompt catalogues
├── agent.py         # Agent surfaces: CLI, Python dispatch, MCP stdio (15 operations)
└── reporting.py     # Report generation (JSON, CSV, Markdown)
```

**Data flow:**

```
CLI / Python / MCP
        │
        ▼
   AgentCatalogue ──► validate & dispatch
        │
        ▼
   Runner ──► Scheduler (closed/open loop)
        │              │
        │              ▼
        │         Endpoint Adapter (OpenAI / Ollama / Mock)
        │              │
        ▼              ▼
   Storage (SQLite WAL) ◄── Evidence + Metrics
        │
        ▼
   Reporting / Comparison / Judging
```

---

## Testing

The test suite covers unit, integration, contract, and acceptance tests.

```bash
pip install pytest pytest-asyncio
pytest
```

All 510 tests should pass. The suite is configured for `asyncio_mode = "auto"` in `pyproject.toml`.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Documentation

- **User guide**: [docs/alecto.md](docs/alecto.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Source code**: [alecto/](alecto/)
- **Tests**: [tests/](tests/)
