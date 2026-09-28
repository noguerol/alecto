# Changelog

All notable changes to Alecto are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] — 2026-09-28

Documentation accuracy. The user guide advertised an API that does not exist, so
every example below failed with `ImportError` or `AttributeError` when followed.

### Fixed

- `docs/alecto.md` configuration example used `AlectoConfig.from_file()`,
  `config.targets` and `config.add_target()`. The real entry point is
  `load_config(path)`; targets are `TargetSpec` records, not config state.
- Performance example used `run_performance_suite(target, cells)` and
  `PerformanceCell(name=…, kind=…)`. The real API is
  `run_performance_cell(cell, request_fn)` with
  `PerformanceCell(cell_id=…, mode=…, target=…, workload=…)`.
- Quality examples used `QualitySuite("…")`. There is no such class; the suites
  are addressed by `load_quality_samples(suite_name)` with `QualityBenchmark`
  subclasses, or by `run_quality_suite(suite_name, adapter, …)`.
- Context example used `ContextSuite` / `run_context_benchmark`; the real API is
  `ContextGenerator(ContextConfig(...)).generate()` and
  `RetrievalTaskGenerator(...).generate_single_kv()` and friends.
- `make_streaming_request_fn` was documented with one argument; it takes an
  adapter factory and the request step.

All 22 constructs now shown in the guide are verified by execution.

### Testing

- New `TestDocumentedApiExists` guard: every `from alecto… import …` in the user
  guide must resolve, and the documented timing fields and quality-suite names
  must exist. This catches the whole drift family rather than one instance.

## [0.1.3] — 2026-09-28

Repository hygiene: removes project-specific names from versioned files and
keeps local working-set ignore rules out of the repository.

### Changed

- `.gitignore` now contains only generic Python/editor/OS patterns. It no longer
  enumerated this project's working files, because that list was itself project
  information. Local ignore rules now live in `.git/info/exclude`, which is not
  versioned.
- `alecto.run_report` no longer names internal stand-alone harnesses in its
  module docstring or in the generated "Reproduction Commands" appendix. The
  appendix describes the artifacts and their provenance instead.
- `alecto.lmeval_adapter` docstring uses a neutral output-directory placeholder.
- The shipped test suite no longer imports or asserts against harnesses that are
  not in the repository; the one report-generation test that depended on one now
  builds its fixture directly.

### Removed

- References to internal working documents in the changelog and docstrings.

## [0.1.2] — 2026-09-28

Release tooling and repository hygiene.

### Added

- Continuous integration (`.github/workflows/ci.yml`): the test suite and lint
  run on Python 3.11, 3.12 and 3.13, plus an offline CLI smoke test and a
  distribution build that asserts the bundled fixtures ship with the wheel and
  that the package version matches the repository.
- An explicit `[tool.ruff.lint]` rule set. Ruff's *default* rules changed
  between releases (0.16 enables many more than 0.15), so relying on them made
  lint results depend on whichever version the resolver picked — CI would have
  failed on its first run with 72 findings that were not in the working tree.

### Changed

- Pinned the tested Ruff range to `>=0.15,<0.17` and declared
  `target-version = "py311"` to match `requires-python`.
- Applied the resulting mechanical fixes (import ordering, quoted annotations,
  redundant open modes, `asyncio.TimeoutError` alias). No behavioural change:
  532 tests pass on 3.11 and 3.14.
- README documentation section now points at the user guide and changelog.

### Removed

- The internal design specification is no longer versioned; it is a private
  working document, kept out of the repository by a local exclude rule. It was
  never included in the built distributions, which is now asserted in CI.

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
