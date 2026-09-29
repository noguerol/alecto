# Changelog

All notable changes to Alecto are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] — 2026-09-29

Public release. The repository is now open.

### Added

- A banner and a rewritten `README.md`: what Alecto is for, the honesty rules it
  follows, install and quick start, the agent surfaces, the Python API, metric
  semantics, the benchmark suites and the architecture.
- README guards in the test suite: every `alecto <command>` shown in a bash block
  must be a registered command, every documented `run` flag must exist, every
  named benchmark must be registered, the banner must open the file, and the
  documented environment variables must match the code.

### Fixed

- **`alecto run` could not reach a real endpoint.** The target URL was hardcoded
  to `mock://test`, so `--target openai` built an adapter pointed at the mock.
  `--endpoint` and `--model` are now accepted, which is what the README had
  claimed all along.
- Documentation fixtures that resembled live provider credentials are replaced
  with self-evidently synthetic values, so secret scanners do not raise a false
  positive. The redaction tests still exercise the same code paths, and a
  history rewrite removes the old strings.

588 tests pass; ruff clean.

## [0.1.5] — 2026-09-28

Agent integration: the MCP surface is now real and documented.

Before this release the operation catalogue existed but nothing was wired to it —
no handler was registered anywhere outside the tests, and there was no way to
start the MCP server — so a harness could not use Alecto at all.

### Added

- **`alecto mcp`** — serves the operation catalogue over MCP stdio
  (JSON-RPC 2.0: `initialize`, `tools/list`, `tools/call`).
- **`alecto/service.py`** — the service layer that binds all 15 catalogued
  operations to real behaviour: named targets, planning, runs, results,
  comparison, experiments, report export, rubric judging and the offline
  self-test. `build_dispatcher()` is the Python entry point.
- **`alecto catalogue --format …`** — emits the tool schemas as canonical JSON or
  in MCP, OpenAI or Anthropic function-calling shapes.
- **`alecto/SKILL.md`** — a distributable agent skill describing the
  integration surfaces, the 15 operations, the error model, metric semantics and
  the canonical workflow. Exposed via `alecto skill [--path|--install-dir]` and
  shipped inside the wheel.
- Handlers may carry a stable error code; `alecto.tool.unsupported` now
  distinguishes "this build cannot do that" from a generic failure.

### Fixed

- **A timed-out sandbox leaked its container.** `subprocess.run(timeout=…)`
  kills the client (`podman run`), but the container keeps running under
  `conmon` until something removes it. Cleanup only ran on the timeout branch,
  so an interrupt or an unexpected error left a container running until the host
  was rebooted — dozens accumulated before it was noticed. Containers are now
  reaped in a `finally` block on every path.
- **Container names collided within a millisecond.** The name was derived from
  `time.time() * 1000 % 100000`, so two evaluations starting in the same
  millisecond shared a name and one run's `rm -f` could remove the other's
  container. Names now include a random suffix.
- A failure in the cleanup command itself no longer masks the real sandbox
  result.
- **A handler printing to stdout corrupted the MCP protocol channel.** The
  self-test operation called the CLI implementation, which printed four lines,
  so every response after it was unparseable. The self-test now has a silent
  implementation, and the server redirects `sys.stdout` to stderr while
  dispatching so no future handler can repeat the mistake.
- `BenchmarkResult.category` was assigned a bare string where the storage
  contract requires the enum, so a run could not be persisted.
- Several operation descriptions claimed behaviour that does not exist ("spawn a
  worker", "endpoint-judge job"); they now describe what the code does.
- `alecto_get_results` reported `n_samples: null`; it reads the coverage counts.
- `judge_run.budget_s` was typed as a string instead of a number.

### Testing

- `tests/unit/test_service.py` (23 tests): the full workflow against the offline
  mock target, idempotent replay, explicit refusals, and MCP transport integrity
  including the stdout-pollution regression.
- `tests/unit/test_skill.py` (19 tests): a bidirectional guard that fails if the
  skill omits a catalogued operation, documents one that does not exist, lists
  the wrong arguments, contains an unresolvable import, a non-executing example,
  a CLI command that is not registered, or an environment variable the code does
  not read.
- `tests/unit/test_quality.py::TestContainerSandboxReaping` (4 tests): drives the
  sandbox path with a stubbed runtime to assert the container is removed on
  timeout, on success, when cleanup itself fails, and that names never collide.
  These run without a container runtime.

581 tests pass; ruff clean.

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
