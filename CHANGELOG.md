# Changelog

All notable changes to Alecto are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-09-28

Hardening release. Found by executing the CLI and configuration against a clean
checkout rather than by inspection; every item below has a test that fails on the
previous version.

### Fixed

- **`alecto run` could not persist a task.** The verdict was stored as a plain
  string where the storage contract requires the `Verdict` enum, so
  `Storage.create_task` raised `AttributeError: 'str' object has no attribute
  'value'`. Every `run` invocation failed and no task was ever written.
- **`alecto report` raised `NameError`.** The command referenced
  `generate_report` without importing it.
- **`--data-dir` was silently ignored** by `run`, and was not accepted at all by
  `list` and `report`, so a task written to an explicit directory could not be
  found again. All three commands now resolve configuration identically.
- **Completed runs stayed `pending`.** `run` set a verdict but never a terminal
  status, so `list --status completed` never matched.
- **`AlectoConfig(data_dir="...")` raised `AttributeError`.** The declared `Path`
  fields were not coerced, so the construction documented in the README crashed.
  `str` and `Path` inputs are now both accepted.
- **`AlectoConfig` was not importable from the package root** despite being the
  documented entry point. `AlectoConfig` and `default_config` are now exported.
- **README documented three environment variables the code never read**
  (`ALECTO_CONCURRENCY`, `ALECTO_MODE`, `ALECTO_TARGET`) and omitted two it did
  (`ALECTO_MAX_CONCURRENT`, `ALECTO_MOCK`). Following the README produced silent
  no-ops. A test now asserts the README and the code agree.
- **README configuration example used non-existent keyword arguments**
  (`timeout_s`, `concurrency`).

### Changed

- Removed dead code in `context.py` (a redundant insertion loop superseded by the
  single- and multi-needle branches) and in `planner/sampling.py` (an unused base
  RNG; determinism is preserved because all randomness derives from the seed).
- Added the missing `LICENSE` file (MIT), referenced by both the README and the
  package metadata.
- Added `CHANGELOG.md`.
- Documented `docs/` layout, the `streaming`, `run_report`, `lmeval_adapter`,
  `lease`, `release` and `planner` modules in the README architecture tree.

### Testing

- 527 tests pass; `ruff check` is clean across the package and tests.
- New suites: `tests/unit/test_cli.py` (the `run` → `list` → `report`
  round-trip) and configuration/public-API contract tests including the
  README-versus-code environment variable guard.

## [0.1.0] — 2026-09-28

First release.

- OpenAI-compatible, Ollama and mock endpoint adapters, with SSE streaming.
- Performance cells: closed-loop and open-loop timing (TTFB, TTFT, E2E,
  p50/p90/p95), configurable concurrency, workloads and arrival processes.
- Academic quality suites: MMLU-Pro, GSM8K, HumanEval and IFEval, with a
  sandboxed code-execution path for coding tasks.
- Refusal evaluation with orthogonal labels and Wilson confidence bounds.
- LLM-as-a-judge scoring, run comparison and delta metrics.
- SQLite WAL storage with single-writer leases for durable job ownership.
- CLI, Python API and MCP stdio surfaces over a shared operation catalogue.
- English Markdown run reports.
