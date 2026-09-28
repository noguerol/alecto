"""Reporting and release for alecto (spec §18, §22 WP-10).

Release report generation (terminal/JSON/Markdown/standalone HTML), release
guide documentation and example plan generation. Stdlib-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0"
METRIC_STATUSES = ("measured", "estimated", "unsupported", "insufficient_data", "error")
PROFILES = ("smoke", "standard", "extended")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReleaseReport:
    """Release report generation (spec §18)."""

    def __init__(self, run_id: str, targets: list[dict[str, Any]]):
        self.run_id = run_id
        self.targets = list(targets)
        self.metrics: list[dict[str, Any]] = []
        self.sections: list[dict[str, Any]] = []
        self.elapsed_s: float | None = None
        self.budget_s: float | None = None
        self.generated_at = _now_iso()

    def set_budget(self, elapsed_s: float, budget_s: float) -> None:
        self.elapsed_s = elapsed_s
        self.budget_s = budget_s

    def add_metric(self, metric_id: str, value: Any, unit: str, status: str = "measured", target_id: str | None = None) -> None:
        if status not in METRIC_STATUSES:
            raise ValueError(f"invalid metric status: {status}")
        self.metrics.append({
            "metric_id": metric_id,
            "value": value,
            "unit": unit,
            "status": status,
            "target_id": target_id,
        })

    def add_section(self, title: str, lines: list[str]) -> None:
        self.sections.append({"title": title, "lines": list(lines)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "targets": [
                {"target_id": t.get("target_id"), "target_title": t.get("target_title")}
                for t in self.targets
            ],
            "metrics": self.metrics,
            "sections": self.sections,
            "elapsed_s": self.elapsed_s,
            "budget_s": self.budget_s,
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def to_markdown(self) -> str:
        lines = [f"# Alecto Report — {self.run_id}", ""]
        lines.append("## Targets")
        for t in self.targets:
            lines.append(f"- **{t.get('target_title')}** (`{t.get('target_id')}`)")
        lines.append("")
        if self.metrics:
            lines.append("## Metrics")
            for m in self.metrics:
                lines.append(f"- {m['metric_id']}: {m['value']} {m['unit']} ({m['status']})")
            lines.append("")
        for section in self.sections:
            lines.append(f"## {section['title']}")
            for line in section["lines"]:
                lines.append(line)
            lines.append("")
        lines.append(f"_Generated {self.generated_at}_")
        return "\n".join(lines)

    def to_html(self) -> str:
        targets_html = "".join(
            f"<li><strong>{_esc(t.get('target_title') or '')}</strong> ({_esc(t.get('target_id') or '')})</li>"
            for t in self.targets
        )
        metrics_html = "".join(
            f"<li>{_esc(m['metric_id'])}: {m['value']} {m['unit']} ({_esc(m['status'])})</li>"
            for m in self.metrics
        )
        sections_html = "".join(
            f"<h2>{_esc(s['title'])}</h2>\n<ul>{''.join('<li>' + _esc(line) + '</li>' for line in s['lines'])}</ul>"
            for s in self.sections
        )
        budget = ""
        if self.elapsed_s is not None and self.budget_s is not None:
            budget = f"<p>Elapsed: {self.elapsed_s}s / budget {self.budget_s}s</p>"
        return (
            "<!DOCTYPE html>\n"
            "<html lang='en'>\n"
            "<head>\n"
            "<meta charset='utf-8'>\n"
            f"<title>Alecto report {self.run_id}</title>\n"
            "<style>body{font-family:system-ui,sans-serif;margin:2rem;}h1,h2{color:#222;}li{margin:.25rem 0;}</style>\n"
            "</head>\n"
            "<body>\n"
            f"<h1>Alecto report — {self.run_id}</h1>\n"
            f"<h2>Targets</h2>\n<ul>{targets_html}</ul>\n"
            f"{budget}"
            f"<h2>Metrics</h2>\n<ul>{metrics_html}</ul>\n"
            f"{sections_html}"
            f"<p>Generated {self.generated_at}</p>\n"
            "</body>\n"
            "</html>"
        )

    def to_terminal(self) -> str:
        lines = [f"Alecto run {self.run_id}", ""]
        for t in self.targets:
            lines.append(f"  target: {t.get('target_title')} ({t.get('target_id')})")
        if self.elapsed_s is not None and self.budget_s is not None:
            lines.append(f"  elapsed: {self.elapsed_s:.1f}s / budget {self.budget_s:.1f}s")
        lines.append("")
        for title in ("Performance", "Capability", "Context", "Refusal/Format", "Comparison/Coverage"):
            lines.append(f"[{title}]")
            section = next((s for s in self.sections if s["title"].lower() == title.lower()), None)
            if section:
                lines.extend(f"  {line}" for line in section["lines"])
            else:
                lines.append("  (no data)")
        return "\n".join(lines)

    def default_filename(self, ext: str = "json") -> str:
        slugs = "-".join(_slugify(t.get("target_title") or t.get("target_id") or "untitled") for t in self.targets)
        return f"alecto-{_slugify(self.run_id)}-{slugs}.{ext}"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


class ReleaseGuide:
    """Release guide documentation (spec §22 WP-10)."""

    def __init__(self, title: str = "Alecto Release Guide"):
        self.title = title
        self.sections: list[dict[str, Any]] = []
        self.checklists: list[list[str]] = []

    def add_section(self, heading: str, body: str) -> None:
        self.sections.append({"heading": heading, "body": body})

    def add_checklist(self, items: list[str]) -> None:
        self.checklists.append(list(items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": self.sections,
            "checklists": self.checklists,
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for section in self.sections:
            lines.append(f"## {section['heading']}")
            lines.append("")
            lines.append(section["body"])
            lines.append("")
        for items in self.checklists:
            lines.append("### Checklist")
            lines.append("")
            for item in items:
                lines.append(f"- [ ] {item}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def default_guide(cls) -> "ReleaseGuide":
        guide = cls()
        guide.add_section(
            "Release checklist",
            "The release is complete only when the acceptance criteria are met with evidence, "
            "not when the CLI looks attractive.",
        )
        guide.add_checklist([
            "AC-01: installed wheel completes self-test offline without a GPU",
            "AC-02: CLI, Python dispatch and MCP produce equivalent canonical results",
            "AC-03: MCP tools/list schemas match the catalogue; invalid input rejected on every surface",
            "AC-04: manifest JSON parses directly; no logs/banner contaminate stdout",
        ])
        guide.add_section(
            "Agent surfaces",
            "CLI, Python API and MCP stdio are first-class and expose equivalent semantics. "
            "The machine manifest is pure JSON; documentation output is a separate option.",
        )
        guide.add_section(
            "Reports",
            "Terminal, JSON, Markdown and standalone HTML reports follow the frozen "
            "target-title naming contract. JSON is the canonical full report.",
        )
        guide.add_section(
            "Known limitations",
            "Deferred: distributed load generators, multi-host scheduling, a web application, "
            "automatic quantisation or abliteration, training, autonomous jailbreak search, "
            "benchmark leaderboard hosting, multimodal evaluation.",
        )
        return guide


class ExamplePlan:
    """Example plan generator (spec §8.2 smoke profile)."""

    SUITES = (
        {"suite_id": "mmlu_pro", "kind": "quality", "cap": 12},
        {"suite_id": "gsm8k", "kind": "quality", "cap": 12},
        {"suite_id": "humaneval", "kind": "quality", "cap": 12},
        {"suite_id": "ifeval", "kind": "quality", "cap": 12},
        {"suite_id": "context", "kind": "context", "cap": "5 positions x 1 length x 1 seed"},
        {"suite_id": "refusal", "kind": "refusal", "cap": 20},
        {"suite_id": "format", "kind": "format", "cap": 12},
        {"suite_id": "performance", "kind": "performance", "cap": "chat C=1/2/4; latency/prefill C=1"},
    )

    def __init__(self, targets: list[dict[str, Any]], profile: str = "smoke", budget_s: float = 480.0):
        self.targets = list(targets)
        self.profile = profile
        self.budget_s = budget_s

    def to_dict(self) -> dict[str, Any]:
        cells = [
            {"suite": s["suite_id"], "target_id": t.get("target_id"), "cap": s["cap"]}
            for t in self.targets
            for s in self.SUITES
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profile": self.profile,
            "budget_s": self.budget_s,
            "targets": [
                {"target_id": t.get("target_id"), "target_title": t.get("target_title")}
                for t in self.targets
            ],
            "suites": list(self.SUITES),
            "cells": cells,
        }
        payload["plan_hash"] = self._hash(payload)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.profile not in PROFILES:
            problems.append(f"unknown profile: {self.profile}")
        if self.budget_s <= 0:
            problems.append("budget_s must be positive")
        if not self.targets:
            problems.append("targets must not be empty")
        for t in self.targets:
            for key in ("target_id", "target_title"):
                if not t.get(key):
                    problems.append(f"target missing {key}: {t!r}")
        return problems

    @classmethod
    def default(cls) -> "ExamplePlan":
        return cls([{"target_id": "local_vllm", "target_title": "Workstation vLLM Q4"}])

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
