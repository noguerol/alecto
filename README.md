![Alecto — local-first, agent-native LLM benchmarking](readme.png)

# Alecto

**Local-first, agent-native execution engine for LLM benchmarking.**

[![CI](https://github.com/noguerol/alecto/actions/workflows/ci.yml/badge.svg)](https://github.com/noguerol/alecto/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Alecto measures how fast and how good an LLM endpoint actually is, and keeps the
evidence on your machine. Point it at anything that speaks the OpenAI-compatible
HTTP API — llama.cpp, vLLM, Ollama, LM Studio, a hosted provider — and it returns
latency, throughput, benchmark scores and refusal behaviour as auditable JSON
plus a human-readable report.

It is built to be driven by an agent: every capability is exposed through one
operation catalogue that the CLI, the Python API and an MCP stdio server all
share, so a coding harness can benchmark a model without a human transcribing
numbers between tools.

---

## Why Alecto

Benchmark numbers are easy to produce and easy to get wrong. A score without a
sample count, a "TTFT" that is really the whole request, a result that silently
drops the items the model failed on — all of these look like data and are not.

Alecto's design choices follow from taking that seriously:

- **A missing measurement is reported as missing.** `null` means "not measured";
  `0` means a measured zero. Alecto never substitutes an estimate for a value it
  did not observe.
- **Coverage is part of every result.** Scores ship with how many items were
  attempted, scored, failed to generate, or were truncated — so a good score
  over a shrunken denominator cannot hide.
- **Every run is reproducible.** Inputs, seeds and protocol ids are recorded
  alongside the numbers, and results are stored in a local SQLite database with
  evidence written before reports are rendered.
- **Nothing leaves your machine.** There is no cloud service, no hosted judge and
  no telemetry. The only network calls go to the endpoint you configure.

## What it does

| Capability | Detail |
| --- | --- |
| **Performance** | Closed-loop and open-loop cells measuring TTFB, TTFT, end-to-end latency, decode throughput and p50/p90/p95 quantiles, across configurable concurrency and arrival processes. |
| **Quality suites** | MMLU-Pro, GSM8K, HumanEval and IFEval, each with a declared protocol id and two-layer coverage. |
| **Refusal evaluation** | Orthogonal labels over benign, benign-edge and unsafe-contrast populations, with Wilson confidence intervals and worst/best-case bounds. |
| **Comparison** | Paired deltas, confidence intervals and factorial contrast analysis across runs. |
| **Judging** | A deterministic rubric judge for stored responses; no second endpoint required. |
| **Reports** | Detailed Markdown run reports, plus JSON, CSV and standalone HTML. |
| **Agent surfaces** | CLI, Python API and MCP stdio over one catalogue of 15 operations, plus a distributable skill. |
| **Sandboxed code execution** | HumanEval candidates run inside a container, never on the host. |

## Installation

Requires Python 3.11 or newer. Runtime dependencies are `httpx` and `jsonschema`.

```bash
pip install alecto
```

Or from source:

```bash
git clone https://github.com/noguerol/alecto
cd alecto
pip install -e ".[dev]"
```

HumanEval additionally needs `podman` or `docker` on `PATH`. Without a container
runtime the suite reports itself as `unsupported` rather than executing generated
code on your machine.

## Quick start

Verify the installation offline — no endpoint needed:

```bash
alecto self-test
```

Benchmark an endpoint:

```bash
alecto run --target openai \
           --endpoint http://localhost:8080/v1 \
           --model my-model \
           --benchmark math_basic \
           --data-dir ./data
```

Then inspect what was recorded:

```bash
alecto list --data-dir ./data
report=$(alecto list --data-dir ./data | cut -d: -f1 | head -1)
alecto report --task-id "$report" --data-dir ./data
```

Available `--benchmark` values: `math_basic`, `coding_basic`, `reasoning_basic`,
`language_basic`. Use `--target mock` to exercise the whole path with no endpoint
at all.

The academic suites (MMLU-Pro, GSM8K, HumanEval, IFEval) and named targets are
driven through the agent or Python surfaces described below.

## Drive it from an agent

Any harness that speaks MCP — Claude Code, pi, omp, and others — can use Alecto
directly.

```bash
# Serve the 15 operations over MCP stdio
alecto mcp

# Emit the tool schemas in the shape your harness expects
alecto catalogue --format mcp        # or canonical | openai | anthropic

# Install the bundled skill into a skills directory
alecto skill --install-dir ~/.claude/skills
```

**[`alecto/SKILL.md`](alecto/SKILL.md)** is the agent-facing contract: the
operation reference, error codes, metric semantics and the canonical workflow. It
ships inside the package, so `alecto skill` always prints the version that
matches your installation.

A typical session:

```jsonc
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"alecto_self_test","arguments":{}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"alecto_configure_target","arguments":{"target":"local","mode":"openai","endpoint":"http://localhost:8080/v1","model":"my-model"}}}
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"alecto_create_plan","arguments":{"target_ids":["local"],"profile":"smoke","suite":"gsm8k"}}}
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"alecto_start_run","arguments":{"plan_id":"<plan-id>","idempotency_key":"run-001"}}}
{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"alecto_get_results","arguments":{"run_id":"<run-id>"}}}
```

Every call returns an envelope — `{"ok": true, "data": {…}}` or
`{"ok": false, "error": {"code", "message", "retryable"}}`. Unknown arguments are
rejected rather than ignored, so a mistyped parameter fails loudly.

### The 15 operations

| Operation | Purpose |
| --- | --- |
| `alecto_self_test` | Offline installation check. |
| `alecto_list_suites` | Benchmark suites and their bundled sample counts. |
| `alecto_configure_target` | Create or update a named endpoint. |
| `alecto_list_targets` | List configured targets. |
| `alecto_capabilities` | Probe a target for observed capabilities. |
| `alecto_create_plan` | Build a bounded plan for a target, profile and suite. |
| `alecto_start_run` | Execute a plan and record its results. |
| `alecto_run_status` | State and progress of a run. |
| `alecto_cancel_run` | Cancel a run. |
| `alecto_resume_run` | Re-open a run that did not complete. |
| `alecto_get_results` | Metrics plus paginated evidence references. |
| `alecto_compare` | Compare runs pairwise or as a group. |
| `alecto_analyse_experiment` | Main and interaction contrasts. |
| `alecto_export_report` | Write a report artifact. |
| `alecto_judge_run` | Rubric-score a run's stored responses. |

## Python API

The same operations are available in-process:

```python
from alecto import AlectoConfig
from alecto.service import build_dispatcher

dispatcher = build_dispatcher(AlectoConfig(data_dir="./data"))
envelope = dispatcher.dispatch("alecto_list_suites", {})
print(envelope["data"]["suites"])
```

Measuring an endpoint directly:

```python
import asyncio

from alecto.adapters import get_adapter
from alecto.domain import TargetSpec
from alecto.enums import LoopMode, TargetKind
from alecto.performance import (
    PerformanceCell, make_streaming_request_fn, run_performance_cell,
)

target = TargetSpec(kind=TargetKind.OPENAI, endpoint="http://localhost:8080/v1", model="my-model")
cell = PerformanceCell(
    cell_id="chat_c1", mode=LoopMode.CLOSED, target=target,
    workload="chat", concurrency=1, samples=20, streaming=True,
)
step = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.0, "max_tokens": 256}

result = asyncio.run(run_performance_cell(cell, make_streaming_request_fn(lambda: get_adapter(target), step)))
print(result.metrics.ttft_ms, result.metrics.ttfb_ms, result.metrics.tokens_per_s)
```

## Configuration

Resolution order: CLI arguments, then environment variables, then defaults.

| Variable | Effect | Default |
| --- | --- | --- |
| `ALECTO_DATA` | Data directory | `~/.alecto` |
| `ALECTO_TIMEOUT` | Request timeout (seconds) | `30.0` |
| `ALECTO_MAX_CONCURRENT` | Maximum concurrent tasks | `4` |
| `ALECTO_MOCK` | Force the mock backend | `false` |

```python
from alecto import AlectoConfig

config = AlectoConfig(data_dir="./data", default_timeout_s=60.0, max_concurrent_tasks=4)
config.ensure_dirs()
```

## Reading the numbers

Two metrics are easy to conflate and reporting the wrong one produces a
misleading result:

- **`ttfb_ms`** — time to the first *byte* of the response stream.
- **`ttft_ms`** — time to the first *content, reasoning or tool* token. A
  role-only opener does not count.

Therefore **TTFB ≤ TTFT**. If they are equal, either the endpoint emits no
role-only opener or it is not streaming. A non-streaming request cannot produce a
real TTFT, and Alecto reports it as unavailable rather than deriving one from the
end-to-end time.

`tokens_per_s` comes from the endpoint's reported `usage` and is never invented.
Quality results carry both `capability_score` (over valid completed evaluations)
and `end_to_end_success` (over attempted items) so that generation failures and
truncations stay visible.

A single endpoint is a single measurement point. Running many concurrent requests
against a one-slot server reduces throughput; it does not increase it.

## Benchmark suites

| Suite | Protocol id | Notes |
| --- | --- | --- |
| `mmlu_pro` | `mmlu_pro.generative_budgeted.v1` | Multiple-choice general knowledge. |
| `gsm8k` | `gsm8k.exact_match.v1` | Grade-school maths, final-answer extraction. |
| `humaneval` | `humaneval.instruct_budgeted.v1` | Code generation, scored by execution in a container sandbox. |
| `ifeval` | `ifeval.strict.v1` | Instruction following, strict and loose accuracy. |

> **On the bundled fixtures.** Alecto ships small synthetic fixtures so the
> suites run offline and regressions are caught in CI. They are **not** official
> benchmark subsets and scores from them are **not** leaderboard comparable. For
> published numbers, run the real datasets with an external harness and import
> the results — `alecto.lmeval_adapter` converts `lm-evaluation-harness` output
> into Alecto's schema and marks it `comparable: true`.

## Architecture

```
alecto/
├── cli.py            # CLI entry point
├── config.py         # Configuration resolution
├── domain.py         # Core records: TargetSpec, Plan, Task, Evidence
├── enums.py          # Shared enumerations
├── adapters.py       # OpenAI / Ollama / mock endpoint adapters
├── streaming.py      # Server-sent-events parsing
├── runner.py         # Task execution engine
├── scheduler.py      # Closed-loop and open-loop scheduling
├── lease.py          # Single-writer leases for durable jobs
├── storage.py        # SQLite WAL storage
├── benchmarks.py     # Built-in benchmark definitions
├── quality.py        # Quality suites, scoring and the container sandbox
├── performance.py    # Performance cells and timing metrics
├── comparison.py     # Paired deltas and factorial contrasts
├── judging.py        # Rubric-based judging
├── refusal.py        # Refusal detection and metrics
├── context.py        # Context generation and retrieval tasks
├── service.py        # Agent service layer binding the operation catalogue
├── agent.py          # Operation catalogue, dispatcher and MCP server
├── run_report.py     # Markdown run reports
├── lmeval_adapter.py # Import lm-evaluation-harness results
├── release.py        # Release reporting
├── planner/          # Budgeting, capability probes, cost model, sampling
└── resources/        # Bundled fixtures and prompt catalogues
```

```
CLI / Python / MCP
        │
        ▼
   Operation catalogue ──►  validate & dispatch
        │
        ▼
     Service layer ──► adapters ──► your endpoint
        │
        ▼
   SQLite (evidence + metrics) ──► reports
```

## Development

```bash
pip install -e ".[dev]"
pytest                  # 581 tests
ruff check alecto tests conftest.py
```

The suite is configured with `asyncio_mode = "auto"`. CI runs the tests and lint
on Python 3.11, 3.12 and 3.13, plus a distribution build that asserts the bundled
fixtures ship with the wheel.

## Documentation

- **[Agent skill](alecto/SKILL.md)** — the contract for harnesses.
- **[User guide](docs/alecto.md)** — full reference.
- **[Changelog](CHANGELOG.md)** — what changed and why.

## License

MIT — see [LICENSE](LICENSE).
