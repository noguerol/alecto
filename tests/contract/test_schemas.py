"""Contract tests for JSON schemas."""



from alecto.domain import TargetSpec, Plan, Task, Evidence
from alecto.enums import TargetKind


class TestTargetSpecSchema:
    def test_serialization(self):
        spec = TargetSpec(
            kind=TargetKind.OPENAI,
            endpoint="https://api.openai.com",
            model="gpt-4",
            timeout_s=30.0,
        )
        data = {
            "kind": spec.kind.value,
            "endpoint": spec.endpoint,
            "model": spec.model,
            "timeout_s": spec.timeout_s,
        }
        assert data["kind"] == "openai"
        assert data["endpoint"] == "https://api.openai.com"

    def test_deserialization(self):
        data = {
            "kind": "openai",
            "endpoint": "https://api.openai.com",
            "model": "gpt-4",
            "timeout_s": 30.0,
        }
        spec = TargetSpec(**data)
        assert spec.kind == TargetKind.OPENAI
        assert spec.model == "gpt-4"


class TestPlanSchema:
    def test_serialization(self):
        plan = Plan(
            steps=[{"type": "noop", "name": "test"}],
            notes=["test plan"],
        )
        data = {
            "steps": plan.steps,
            "notes": plan.notes,
        }
        assert len(data["steps"]) == 1

    def test_deserialization(self):
        data = {
            "steps": [{"type": "noop", "name": "test"}],
            "notes": ["test plan"],
        }
        plan = Plan(**data)
        assert len(plan.steps) == 1
        assert plan.notes == ["test plan"]


class TestTaskSchema:
    def test_roundtrip(self):
        task = Task(
            plan=Plan(steps=[{"type": "noop"}]),
            target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"),
        )
        # Serialize
        data = {
            "id": task.id,
            "status": task.status.value,
            "target": {
                "kind": task.target.kind.value,
                "endpoint": task.target.endpoint,
            },
        }
        # Deserialize
        assert data["status"] == "pending"
        assert data["target"]["kind"] == "mock"


class TestEvidenceSchema:
    def test_text_evidence(self):
        from alecto.enums import EvidenceKind
        ev = Evidence(kind=EvidenceKind.TEXT, text="hello")
        assert ev.kind.value == "text"

    def test_metric_evidence(self):
        from alecto.enums import EvidenceKind
        ev = Evidence(kind=EvidenceKind.METRIC, metric_name="score", value=9.5)
        assert ev.kind.value == "metric"
        assert ev.value == 9.5
