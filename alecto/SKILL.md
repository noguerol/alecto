---
name: alecto
description: >
  Measure and compare LLM endpoints with Alecto: benchmark an OpenAI-compatible,
  Ollama or mock endpoint for latency/throughput, run academic quality suites
  (MMLU-Pro, GSM8K, HumanEval, IFEval), evaluate refusal behaviour, compare runs
  and produce Markdown/JSON reports. Use this skill when the user wants to
  benchmark a model or endpoint, measure TTFT/TTFB/tokens-per-second, score
  quality on a benchmark, compare two models or quantisations, audit refusals or
  safety behaviour, or export a benchmark report.
allowed-tools: Bash(alecto:*), Bash(python -m alecto:*)
---

# Alecto — LLM benchmarking for agent harnesses

Alecto measures the performance and quality of LLM endpoints and stores
reproducible, auditable results locally. It talks to anything that speaks the
OpenAI-compatible HTTP API (llama.cpp, vLLM, Ollama, LM Studio, hosted APIs) plus
a built-in offline `mock` target for testing the harness without a model.

Everything runs on the local machine. Alecto never sends your data anywhere
except to the endpoint you configure.

## Pick an integration surface

| Surface | Use when | Entry point |
| --- | --- | --- |
| **MCP stdio** | The harness supports MCP tools (Claude Code, pi, omp, …). Preferred: schemas, validation and typed errors. | `alecto mcp` |
| **CLI** | Shell, scripts, quick checks. | `alecto <command>` |
| **Python API** | You are writing code and want the modules directly. | `from alecto import …` |

All three resolve through one operation catalogue, so an operation behaves
identically whichever way you call it.

---

## 1. MCP stdio (preferred)

Start the server and speak JSON-RPC 2.0 over stdin/stdout:

```bash
alecto mcp [--data-dir DIR]
```

Supported methods: `initialize`, `tools/list`, `tools/call`,
`notifications/initialized`. Protocol version `2024-11-05`.

**Critical:** stdout is the protocol channel. Send one JSON object per line and
read exactly one response line per request. Alecto never writes non-JSON to
stdout, and redirects any handler output to stderr, but do not merge the two
streams in your client.

Typical session:

```
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}}}}

→ {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
← {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"alecto_self_test","description":"…","inputSchema":{}}]}}

→ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"alecto_list_suites","arguments":{}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"schema_version\":\"1.0\",\"ok\":true,\"operation\":\"alecto_list_suites\",\"data\":{},\"error\":null,\"diagnostics\":[]}"}]}}
```

The `text` field is a JSON-encoded **envelope**:

```json
{"schema_version":"1.0","ok":true,"operation":"…","data":{…},"error":null,"diagnostics":[]}
```

On failure `ok` is `false` and `error` carries `{code, message, retryable}`.

### Error codes

| Code | Meaning |
| --- | --- |
| `alecto.validation.failed` | Arguments do not match the tool schema (unknown field or wrong type). `details` lists every problem. Nothing was executed. |
| `alecto.tool.unknown_operation` | No such operation, or no handler registered. |
| `alecto.tool.unsupported` | The operation exists but this build cannot do that for these inputs (e.g. resuming a completed run). Not retryable — change the request. |
| `alecto.tool.error` | The handler raised. Read `message`. |

Unknown fields are **rejected, not ignored**, so a typo'd argument fails loudly
instead of silently doing the wrong thing.

### The 15 operations

`?` marks optional arguments. `target_ids` and `run_ids` are arrays of strings.

| Operation | Arguments | Notes |
| --- | --- | --- |
| `alecto_self_test` | `profile?` | Offline installation check. No endpoint. Good first call. |
| `alecto_list_suites` | `installed_only?` | Suites and bundled sample counts. |
| `alecto_configure_target` | `target`, `mode`, `endpoint?`, `model?`, `title?` | `mode` ∈ `openai`, `openai_compatible`, `ollama`, `llm`, `llmstudio`, `mock`. Persists a named target. |
| `alecto_list_targets` | `target_id_filter?` | Configured targets. |
| `alecto_capabilities` | `target_id`, `probe_level` | Observed capabilities only; unobserved fields stay `null` — never guessed. |
| `alecto_create_plan` | `target_ids`, `profile`, `budget_s?`, `suite?`, `source?`, `seed?`, `prompt_tps?`, `output_tps?`, `expected_output_tokens?` | `profile` ∈ `smoke`, `quick`, `compare`, `standard`. Default suite `mmlu_pro`. `source` ∈ `fixtures`, `official` (see below). |
| `alecto_start_run` | `plan_id`, `idempotency_key` | Runs the plan (synchronously) and records results. Same key ⇒ same run, no re-execution. |
| `alecto_run_status` | `run_id?` | Omit for the latest run. |
| `alecto_cancel_run` | `run_id` | Terminal runs are returned unchanged. |
| `alecto_resume_run` | `run_id`, `idempotency_key` | Refuses completed runs with `alecto.tool.unsupported`. |
| `alecto_get_results` | `run_id`, `suite?`, `cursor?`, `limit?` | `metrics` plus paginated `evidence_refs`; `next_cursor` is `null` at the end. |
| `alecto_compare` | `run_ids`, `mode` | `pairwise` compares the first two; `group` compares all and records an experiment when the runs carry factor labels. |
| `alecto_analyse_experiment` | `experiment_id` | Main/interaction contrasts. Ids come from `alecto_compare` with `mode="group"`. |
| `alecto_export_report` | `run_id`, `format`, `output_path?` | `format` ∈ `markdown`, `md`, `json`, `csv`, `html`. Returns `path` and a sha256 `digest`. |
| `alecto_judge_run` | `run_id`, `judge_target?`, `budget_s?` | Deterministic rubric scoring of stored responses. No endpoint needed. |

### Fixtures vs official benchmarks — read this before reporting a score

`alecto_list_suites` returns two separate lists, and they are not
interchangeable:

| | `suites` | `official_benchmarks` |
| --- | --- | --- |
| What runs | Bundled synthetic fixtures | The real dataset via an external harness |
| Comparable to a leaderboard | **No** | **Yes** (`comparable: true`) |
| Needs network | No | Yes (dataset download) |
| Cost | Seconds | Hours for a full split |

Control this with `source` on `alecto_create_plan`:

- `source: "fixtures"` — the offline set. Default for the four names that have
  fixtures (`mmlu_pro`, `gsm8k`, `humaneval`, `ifeval`), so quick checks stay
  fast and offline.
- `source: "official"` — the real benchmark. **Use this whenever the user wants
  a number they can compare with anything.**

Names that exist in both registries default to `fixtures`. That means
`suite: "gsm8k"` on its own does **not** give you a leaderboard number — pass
`source: "official"` explicitly. Names that only exist officially (`gpqa`,
`arc_challenge`, `hellaswag`, `truthfulqa`, `winogrande`, `bbh`, `drop`, `mmlu`,
`piqa`, `mathqa`, `mbpp`, `lambada_openai`) resolve to the official run
automatically.

Official runs need `lm-evaluation-harness` in a separate interpreter. Alecto does
not bundle it; set `ALECTO_EVAL_PYTHON` to such an interpreter. If none is found
the operation fails with `alecto.tool.unsupported` and instructions — it never
silently falls back to fixtures, because a fixture score presented as an official
one is the exact failure this distinction exists to prevent.

### Canonical workflow

```
alecto_self_test                                        # 1. verify install (offline)
alecto_configure_target {target, mode, endpoint, model}  # 2. name an endpoint
alecto_create_plan {target_ids:[target], profile:"smoke", suite:"gsm8k"}
alecto_start_run {plan_id, idempotency_key:"run-001"}    # 3. measure
alecto_get_results {run_id}                              # 4. read metrics + evidence
alecto_export_report {run_id, format:"markdown", output_path:"report.md"}
```

Start with `profile:"smoke"` to validate wiring, then raise to `quick` or
`compare` for real numbers.

### Inspecting the catalogue without MCP

```bash
alecto catalogue --format canonical   # pure JSON, no surrounding text
alecto catalogue --format mcp         # MCP tool definitions
alecto catalogue --format openai      # OpenAI function-calling tools
alecto catalogue --format anthropic   # Anthropic tool definitions
```

`canonical` emits parseable JSON with no banner, so it is safe to pipe into a
JSON parser.

---

## 2. CLI

```bash
alecto self-test                                     # offline check
alecto run --target mock --benchmark math_basic --data-dir ./data
alecto list [--status completed] [--data-dir DIR]
alecto report --task-id ID [--data-dir DIR]
alecto catalogue [--format …]
alecto mcp [--data-dir DIR]
```

`--data-dir` selects where state lives. When omitted, `ALECTO_DATA` is used,
falling back to `~/.alecto`.

| Environment variable | Effect | Default |
| --- | --- | --- |
| `ALECTO_DATA` | Data directory | `~/.alecto` |
| `ALECTO_TIMEOUT` | Request timeout (s) | `30.0` |
| `ALECTO_MAX_CONCURRENT` | Max concurrent tasks | `4` |
| `ALECTO_MOCK` | Force the mock backend | `false` |
| `ALECTO_EVAL_PYTHON` | Interpreter that has lm-evaluation-harness, for official benchmarks | *(auto-detected)* |

---

## 3. Python API

```python
from alecto import AlectoConfig, QUALITY_SUITES, load_quality_samples
from alecto.agent import MCPStdioAdapter
from alecto.service import build_catalogue, build_dispatcher

# Drive the same operations MCP exposes, in-process.
dispatcher = build_dispatcher(AlectoConfig(data_dir="./data"))
envelope = dispatcher.dispatch("alecto_list_suites", {})
assert envelope["ok"]

# MCP over any pair of file-like objects.
import sys

adapter = MCPStdioAdapter(build_catalogue(), dispatcher)
adapter.run(sys.stdin, sys.stdout)
```

Measuring an endpoint directly:

```python
import asyncio
from alecto.adapters import get_adapter
from alecto.domain import TargetSpec
from alecto.enums import LoopMode, TargetKind
from alecto.performance import PerformanceCell, make_streaming_request_fn, run_performance_cell

# streaming=True requires a target that actually streams. The mock adapter
# raises CapabilityUnavailable instead of reporting a fabricated TTFT.
target = TargetSpec(kind=TargetKind.OPENAI, endpoint="http://localhost:8080/v1", model="my-model")
cell = PerformanceCell(cell_id="chat_c1", mode=LoopMode.CLOSED, target=target,
                       workload="chat", concurrency=1, samples=20, streaming=True)
step = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.0, "max_tokens": 256}

result = asyncio.run(run_performance_cell(cell, make_streaming_request_fn(lambda: get_adapter(target), step)))
print(result.metrics.ttft_ms, result.metrics.ttfb_ms, result.metrics.tokens_per_s)
```

Running a quality suite:

```python
import asyncio

from alecto.adapters import get_adapter
from alecto.domain import TargetSpec
from alecto.enums import TargetKind
from alecto.quality import IFEvalBenchmark, load_quality_samples, run_quality_benchmark

target = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://local", model="my-model")
benchmark = IFEvalBenchmark(load_quality_samples("ifeval"))
result = asyncio.run(run_quality_benchmark(benchmark, get_adapter(target)))
print(result.benchmark_name, result.score)
```

Swap `TargetKind.MOCK` for the real target's kind to measure an endpoint. Or use
`run_quality_suite("ifeval", adapter)` for the higher-level call that also returns
coverage counts.

---

## Metric semantics — read before interpreting

Alecto distinguishes metrics that are easy to conflate. Reporting the wrong one
is the most common way to produce a misleading result.

| Metric | Meaning |
| --- | --- |
| `ttfb_ms` | Time to the **first byte** of the response stream. |
| `ttft_ms` | Time to the **first content/reasoning/tool token**. A role-only opener does not count. |
| `e2e_ms` | Total request duration. |
| `tokens_per_s` | Decode throughput, from reported `usage` when available. |
| `capability_score` | Score over **valid completed** evaluations. |
| `end_to_end_success` | Success over **attempted** items, so failures cannot be hidden. |

`None` means **not measured**; `0` means measured zero. Alecto never substitutes
an estimate for a missing measurement — the field is either populated or `null`
and the accompanying `*_status` explains why.

Two consequences worth internalising:

- **TTFB ≤ TTFT.** TTFB is the socket; TTFT additionally waits for the first
  token. If they are equal, either the endpoint emits no role-only opener or it
  is not streaming — check `streaming=True` was set.
- **Non-streaming cannot produce a real TTFT.** It is reported as unavailable
  rather than faked from the end-to-end time.

---

## Scope and honesty

Be precise about what a run proves.

- **Bundled fixtures are synthetic.** `load_quality_samples()` ships small
  fixtures for smoke and regression runs. They are **not** official benchmark
  subsets and scores from them are **not** leaderboard comparable. Official
  numbers require the real datasets and an external harness; Alecto can import
  those results (`alecto.lmeval_adapter`) and marks them `comparable: true`.
- **Always report `n`.** A score without a sample count is not evidence. A slice
  with n=8 has a standard error around ±0.19.
- **Check coverage.** Report `failed_generation`, `truncated` and `unsupported`
  alongside any score. A high score over a shrunken denominator is not a result.
- **Reasoning models need budget.** If the model spends tokens on hidden
  reasoning, too small a `max_tokens` yields empty answers scored as zero. Set
  an explicit budget before concluding a model "failed".
- **HumanEval runs sandboxed.** Generated code is only executed inside a
  container sandbox. If no sandbox is available the suite reports `unsupported`
  rather than executing model code on the host. Never set `HF_ALLOW_CODE_EVAL`
  to bypass this.
- **A single endpoint is a single measurement point.** Alecto is single-process
  per target; running many concurrent requests against a one-slot server reduces
  throughput, it does not increase it.

---

## Troubleshooting

| Symptom | Cause and action |
| --- | --- |
| `alecto.validation.failed` | Wrong or misspelled argument. Read `error.details`; the field is named there. |
| `alecto.tool.unsupported` | Not retryable. Usually a completed run being resumed, or a suite with no bundled samples. Change the request. |
| Empty model answers, score 0 | Output budget too small for a reasoning model. Raise `max_tokens`. |
| `TargetUnreachableError` | Endpoint is down or the URL/route is wrong. Verify with `alecto_capabilities`. |
| `ttft_ms` equals `ttfb_ms` | Not streaming, or the target does not emit incremental tokens. |
| JSON parse error reading MCP | Something wrote to stdout. Never `print()` in a handler; Alecto redirects its own output to stderr. |

## Requirements

Python ≥ 3.11. Runtime dependencies: `httpx`, `jsonschema`. The container sandbox
used for HumanEval additionally needs `podman` or `docker` available on `PATH`.
