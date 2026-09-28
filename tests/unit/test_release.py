"""Unit tests for alecto.release (WP-10 reporting and release)."""

import json

import pytest

from alecto.release import ExamplePlan, ReleaseGuide, ReleaseReport

TARGETS = [{"target_id": "local_vllm", "target_title": "Workstation vLLM Q4"}]


@pytest.fixture
def report():
    r = ReleaseReport("run-abc", TARGETS)
    r.set_budget(120.0, 480.0)
    r.add_metric("alecto.perf.ttft_ms", 42.0, "ms", "measured", target_id="local_vllm")
    r.add_section("Performance", ["ttft 42 ms at C=1"])
    r.add_section("Capability", ["mmlu_pro 8.3/10"])
    r.add_section("Context", ["5 positions x 1 length x 1 seed"])
    r.add_section("Refusal/Format", ["refusal 20, format 12"])
    r.add_section("Comparison/Coverage", ["coverage 95%"])
    return r


class TestReleaseReport:
    def test_to_dict_keys(self, report):
        d = report.to_dict()
        assert d["schema_version"] == "1.0"
        assert d["run_id"] == "run-abc"
        assert d["targets"] == [{"target_id": "local_vllm", "target_title": "Workstation vLLM Q4"}]
        assert d["metrics"][0]["metric_id"] == "alecto.perf.ttft_ms"
        assert len(d["sections"]) == 5
        assert d["generated_at"]

    def test_to_json_pure(self, report):
        data = json.loads(report.to_json())
        assert data["run_id"] == "run-abc"

    def test_markdown_contains_titles(self, report):
        md = report.to_markdown()
        assert "run-abc" in md
        assert "Workstation vLLM Q4" in md

    def test_html_self_contained(self, report):
        html = report.to_html()
        assert "<meta charset" in html
        assert "run-abc" in html
        assert "Workstation vLLM Q4" in html
        assert "http://" not in html
        assert "https://" not in html

    def test_terminal_five_sections(self, report):
        term = report.to_terminal()
        assert "run-abc" in term
        assert "Workstation vLLM Q4" in term
        for label in ("Performance", "Capability", "Context", "Refusal/Format", "Comparison/Coverage"):
            assert label in term

    def test_default_filename(self, report):
        name = report.default_filename()
        assert name.endswith(".json")
        assert "run-abc" in name
        assert "workstation-vllm-q4" in name

    def test_invalid_metric_status(self):
        r = ReleaseReport("run-1", TARGETS)
        with pytest.raises(ValueError):
            r.add_metric("m", 1, "ms", status="bogus")


class TestReleaseGuide:
    def test_default_guide_sections(self):
        guide = ReleaseGuide.default_guide()
        headings = [s["heading"] for s in guide.sections]
        for expected in ("Release checklist", "Agent surfaces", "Reports", "Known limitations"):
            assert expected in headings
        assert guide.checklists and guide.checklists[0]

    def test_checklist_rendering(self):
        guide = ReleaseGuide()
        guide.add_checklist(["do A", "do B"])
        md = guide.to_markdown()
        assert "- [ ] do A" in md
        assert "- [ ] do B" in md

    def test_to_dict(self):
        guide = ReleaseGuide("My Guide")
        guide.add_section("S1", "body")
        d = guide.to_dict()
        assert d["title"] == "My Guide"
        assert d["sections"][0] == {"heading": "S1", "body": "body"}


class TestExamplePlan:
    def test_default_valid(self):
        plan = ExamplePlan.default()
        assert plan.validate() == []

    def test_to_dict_shape(self):
        d = ExamplePlan.default().to_dict()
        assert d["schema_version"] == "1.0"
        assert d["profile"] == "smoke"
        assert d["budget_s"] == 480.0
        assert d["targets"] == TARGETS
        assert d["suites"]
        assert d["cells"]
        assert d["plan_hash"]

    def test_plan_hash_stable(self):
        h1 = ExamplePlan.default().to_dict()["plan_hash"]
        h2 = ExamplePlan.default().to_dict()["plan_hash"]
        assert h1 == h2
        assert len(h1) == 64

    def test_validate_bad_profile(self):
        plan = ExamplePlan(TARGETS, profile="ultra")
        assert any("profile" in p for p in plan.validate())

    def test_validate_nonpositive_budget(self):
        plan = ExamplePlan(TARGETS, budget_s=0)
        assert any("budget" in p for p in plan.validate())

    def test_validate_empty_targets(self):
        plan = ExamplePlan([])
        assert any("targets" in p for p in plan.validate())

    def test_validate_missing_title(self):
        plan = ExamplePlan([{"target_id": "t1"}])
        assert any("target_title" in p for p in plan.validate())

    def test_to_json(self):
        data = json.loads(ExamplePlan.default().to_json())
        assert data["plan_hash"]
