"""Context and retrieval suite (spec §11) plus format/tool fixtures (spec §14).

Provides:
- ContextGenerator: synthetic context generation with configurable length, topics, difficulty.
- RetrievalTask: retrieval benchmark tasks with context passages and questions.
- FormatFixture: JSON schema fixtures for structured output validation.
- ToolFixture: tool-calling fixtures with schema definitions.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# ContextGenerator
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    """Difficulty levels for synthetic context generation."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ContextConfig:
    """Configuration for synthetic context generation."""
    target_length: int = 4096
    topics: list[str] = field(default_factory=lambda: ["technology", "science", "history"])
    difficulty: Difficulty = Difficulty.EASY
    num_needles: int = 1
    seed: int | None = None
    filler_type: str = "records"  # "records" | "text"
    collision_check: bool = True


@dataclass
class GeneratedContext:
    """Result of synthetic context generation."""
    haystack: str
    needles: list[dict[str, Any]]
    question: str
    expected_answer: str
    config: ContextConfig
    seed: int
    actual_length: int
    positions: list[float]


class ContextGenerator:
    """Generates synthetic context passages for retrieval benchmarking (spec §11.1).

    Produces deterministic, seeded contexts with embedded "needle" records at
    nominal positions. Supports configurable length, topics, and difficulty.
    """

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()

    def generate(self) -> GeneratedContext:
        """Generate a single synthetic context with embedded needles."""
        rng = random.Random(self.config.seed)
        positions = self._nominal_positions(self.config.num_needles, rng)
        needles = self._make_needles(rng)
        haystack = self._build_haystack(rng, needles, positions)
        question, expected = self._make_question(needles)
        return GeneratedContext(
            haystack=haystack,
            needles=needles,
            question=question,
            expected_answer=expected,
            config=self.config,
            seed=self.config.seed if self.config.seed is not None else 0,
            actual_length=len(haystack),
            positions=positions,
        )

    def _nominal_positions(self, n: int, rng: random.Random) -> list[float]:
        """Compute nominal positions (5%, 25%, 50%, 75%, 95%) for n needles."""
        nominal = [0.05, 0.25, 0.50, 0.75, 0.95]
        if n <= len(nominal):
            return nominal[:n]
        # For more needles, distribute evenly
        return [i / (n + 1) for i in range(1, n + 1)]

    def _make_needles(self, rng: random.Random) -> list[dict[str, Any]]:
        """Create needle key/value pairs with collision checks."""
        needles = []
        used_keys: set[str] = set()
        used_values: set[str] = set()
        topics = self.config.topics

        for i in range(self.config.num_needles):
            topic = topics[i % len(topics)]
            key = self._unique_key(rng, used_keys, topic)
            value = self._unique_value(rng, used_values)
            needles.append({
                "key": key,
                "value": value,
                "topic": topic,
                "record_id": f"rec-{i:03d}",
            })
        return needles

    def _unique_key(self, rng: random.Random, used: set[str], topic: str) -> str:
        """Generate a unique key related to the topic."""
        while True:
            key = f"{topic}_{rng.randint(1000, 9999)}_{rng.choice(['alpha', 'beta', 'gamma', 'delta'])}"
            if key not in used:
                used.add(key)
                return key

    def _unique_value(self, rng: random.Random, used: set[str]) -> str:
        """Generate a unique value that is not inferable from the question."""
        while True:
            value = "".join(rng.choices(string.ascii_lowercase + string.digits, k=12))
            if value not in used:
                used.add(value)
                return value

    def _build_haystack(self, rng: random.Random, needles: list[dict[str, Any]], positions: list[float]) -> str:
        """Build the haystack text with needles embedded at nominal positions."""
        target_len = self.config.target_length
        filler_len = target_len - len(needles) * 60  # reserve space for needle records

        if self.config.filler_type == "records":
            filler = self._generate_record_filler(rng, filler_len, needles)
        else:
            filler = self._generate_text_filler(rng, filler_len, needles)

        # Insert needles at their positions. The single- and multi-needle
        # cases are handled explicitly below.

        if len(needles) > 1:
            parts = []
            filler_chunks = self._split_filler(filler, len(needles) + 1)
            for i, needle in enumerate(needles):
                parts.append(filler_chunks[i])
                parts.append(self._format_record(needle))
            parts.append(filler_chunks[-1])
            return "".join(parts)

        if len(needles) == 1:
            insert_at = int(positions[0] * target_len)
            prefix = filler[:insert_at]
            suffix = filler[insert_at:]
            return prefix + self._format_record(needles[0]) + suffix

        return filler

    def _format_record(self, needle: dict[str, Any]) -> str:
        """Format a needle as a structured record."""
        return (
            f"\n[{needle['record_id']}] "
            f"key: {needle['key']} | value: {needle['value']} | topic: {needle['topic']}\n"
        )

    def _generate_record_filler(self, rng: random.Random, length: int, needles: list[dict[str, Any]]) -> str:
        """Generate homogeneous record filler."""
        records = []
        for i in range(max(1, length // 80)):
            topic = rng.choice(self.config.topics)
            key = f"filler_{topic}_{i:04d}"
            val = "".join(rng.choices(string.ascii_lowercase, k=8))
            records.append(f"[filler-{i:04d}] key: {key} | value: {val} | topic: {topic}")
        return "\n".join(records)

    def _generate_text_filler(self, rng: random.Random, length: int, needles: list[dict[str, Any]]) -> str:
        """Generate natural-text filler."""
        words = ["the", "data", "system", "process", "value", "record", "context", "model",
                 "endpoint", "target", "benchmark", "score", "result", "analysis", "report"]
        sentences = []
        for i in range(max(1, length // 60)):
            n_words = rng.randint(8, 15)
            sentence = " ".join(rng.choices(words, k=n_words))
            sentences.append(f"The {sentence} in the {rng.choice(words)} context.")
        return "\n".join(sentences)

    def _split_filler(self, filler: str, n_parts: int) -> list[str]:
        """Split filler into roughly equal parts."""
        chunk_size = max(1, len(filler) // n_parts)
        parts = []
        for i in range(n_parts):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_parts - 1 else len(filler)
            parts.append(filler[start:end])
        return parts

    def _make_question(self, needles: list[dict[str, Any]]) -> tuple[str, str]:
        """Generate the question and expected answer for the needles."""
        if len(needles) == 1:
            n = needles[0]
            return (
                f"What is the value for key '{n['key']}'?",
                n["value"],
            )
        keys = ", ".join(n["key"] for n in needles)
        return (
            f"What are the values for keys: {keys}?",
            "; ".join(n["value"] for n in needles),
        )


# ---------------------------------------------------------------------------
# RetrievalTask
# ---------------------------------------------------------------------------

class RetrievalTaskType(str, Enum):
    """Types of retrieval tasks (spec §11.2)."""
    SINGLE_KV = "single_kv"
    SIMILAR_KEY = "similar_key_interference"
    MULTI_NEEDLE = "multiple_needles"
    TWO_EVIDENCE = "two_evidence_composition"
    ABSENT_KEY = "absent_key"
    UPDATED_RECORD = "updated_record"


@dataclass
class Passage:
    """A context passage for retrieval."""
    id: str
    text: str
    topic: str
    position: float = 0.0


@dataclass
class RetrievalTask:
    """A retrieval benchmark task with context passages and a question (spec §11.2)."""
    task_id: str
    task_type: RetrievalTaskType
    passages: list[Passage]
    question: str
    expected_answer: str
    difficulty: Difficulty = Difficulty.EASY
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task to a dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "passages": [
                {"id": p.id, "text": p.text, "topic": p.topic, "position": p.position}
                for p in self.passages
            ],
            "question": self.question,
            "expected_answer": self.expected_answer,
            "difficulty": self.difficulty.value,
            "seed": self.seed,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize the task to JSON."""
        return json.dumps(self.to_dict(), indent=2)


class RetrievalTaskGenerator:
    """Generates retrieval benchmark tasks from synthetic contexts."""

    def __init__(self, generator: ContextGenerator | None = None):
        self.generator = generator or ContextGenerator()

    def generate_single_kv(self, seed: int | None = None) -> RetrievalTask:
        """Generate a single key/value retrieval task."""
        config = ContextConfig(num_needles=1, seed=seed)
        gen = ContextGenerator(config)
        ctx = gen.generate()
        needle = ctx.needles[0]
        return RetrievalTask(
            task_id=f"single_kv_{seed or 'auto'}",
            task_type=RetrievalTaskType.SINGLE_KV,
            passages=[Passage(id="ctx", text=ctx.haystack, topic=needle["topic"], position=ctx.positions[0])],
            question=ctx.question,
            expected_answer=ctx.expected_answer,
            difficulty=config.difficulty,
            seed=seed,
            metadata={"filler_type": config.filler_type, "actual_length": ctx.actual_length},
        )

    def generate_multi_needle(self, num_needles: int = 3, seed: int | None = None) -> RetrievalTask:
        """Generate a multiple independent needles task."""
        config = ContextConfig(num_needles=num_needles, seed=seed)
        gen = ContextGenerator(config)
        ctx = gen.generate()
        return RetrievalTask(
            task_id=f"multi_needle_{seed or 'auto'}",
            task_type=RetrievalTaskType.MULTI_NEEDLE,
            passages=[Passage(id="ctx", text=ctx.haystack, topic="mixed", position=0.0)],
            question=ctx.question,
            expected_answer=ctx.expected_answer,
            difficulty=config.difficulty,
            seed=seed,
            metadata={"num_needles": num_needles, "positions": ctx.positions},
        )

    def generate_absent_key(self, seed: int | None = None) -> RetrievalTask:
        """Generate an absent key task (correct abstention expected)."""
        config = ContextConfig(num_needles=1, seed=seed)
        gen = ContextGenerator(config)
        ctx = gen.generate()
        absent_key = "nonexistent_key_9999"
        return RetrievalTask(
            task_id=f"absent_key_{seed or 'auto'}",
            task_type=RetrievalTaskType.ABSENT_KEY,
            passages=[Passage(id="ctx", text=ctx.haystack, topic="mixed", position=0.0)],
            question=f"What is the value for key '{absent_key}'?",
            expected_answer="ABSTAIN",
            difficulty=config.difficulty,
            seed=seed,
            metadata={"absent_key": absent_key},
        )

    def generate_two_evidence(self, seed: int | None = None) -> RetrievalTask:
        """Generate a two-evidence composition task."""
        config = ContextConfig(num_needles=2, seed=seed)
        gen = ContextGenerator(config)
        ctx = gen.generate()
        n1, n2 = ctx.needles
        answer = f"{n1['value']}+{n2['value']}"
        return RetrievalTask(
            task_id=f"two_evidence_{seed or 'auto'}",
            task_type=RetrievalTaskType.TWO_EVIDENCE,
            passages=[Passage(id="ctx", text=ctx.haystack, topic="mixed", position=0.0)],
            question=(
                f"Concatenate the values for keys '{n1['key']}' and '{n2['key']}' "
                f"with a '+' separator."
            ),
            expected_answer=answer,
            difficulty=config.difficulty,
            seed=seed,
            metadata={"evidence_keys": [n1["key"], n2["key"]]},
        )


# ---------------------------------------------------------------------------
# FormatFixture
# ---------------------------------------------------------------------------

@dataclass
class FormatFixture:
    """JSON schema fixture for structured output validation (spec §14)."""
    fixture_id: str
    schema: dict[str, Any]
    valid_examples: list[Any]
    invalid_examples: list[Any]
    constraints: list[str] = field(default_factory=list)
    description: str = ""

    def validate(self, value: Any) -> tuple[bool, list[str]]:
        """Validate a value against the fixture schema. Returns (valid, errors)."""
        errors = []
        schema_type = self.schema.get("type")

        if schema_type == "object":
            if not isinstance(value, dict):
                errors.append("Expected object, got " + type(value).__name__)
                return False, errors
            required = self.schema.get("required", [])
            properties = self.schema.get("properties", {})
            for field_name in required:
                if field_name not in value:
                    errors.append(f"Missing required field: {field_name}")
            for prop_name, prop_schema in properties.items():
                if prop_name in value:
                    prop_val = value[prop_name]
                    prop_type = prop_schema.get("type")
                    if prop_type == "string" and not isinstance(prop_val, str):
                        errors.append(f"Field '{prop_name}' must be string")
                    elif prop_type == "number" and not isinstance(prop_val, (int, float)):
                        errors.append(f"Field '{prop_name}' must be number")
                    elif prop_type == "integer" and not isinstance(prop_val, int):
                        errors.append(f"Field '{prop_name}' must be integer")
                    elif prop_type == "boolean" and not isinstance(prop_val, bool):
                        errors.append(f"Field '{prop_name}' must be boolean")
                    elif prop_type == "array" and not isinstance(prop_val, list):
                        errors.append(f"Field '{prop_name}' must be array")
                    # Check enum constraint
                    enum_values = prop_schema.get("enum")
                    if enum_values is not None and prop_val not in enum_values:
                        errors.append(f"Field '{prop_name}' value not in enum {enum_values}")
            # Check forbidden fields
            forbidden = self.schema.get("additionalProperties", True)
            if forbidden is False:
                allowed = set(properties.keys())
                for key in value:
                    if key not in allowed:
                        errors.append(f"Unexpected field: {key}")
        elif schema_type == "array":
            if not isinstance(value, list):
                errors.append("Expected array, got " + type(value).__name__)
                return False, errors
            min_items = self.schema.get("minItems")
            if min_items is not None and len(value) < min_items:
                errors.append(f"Array must have at least {min_items} items")
            max_items = self.schema.get("maxItems")
            if max_items is not None and len(value) > max_items:
                errors.append(f"Array must have at most {max_items} items")
        elif schema_type == "string":
            if not isinstance(value, str):
                errors.append("Expected string")
                return False, errors
        elif schema_type == "number":
            if not isinstance(value, (int, float)):
                errors.append("Expected number")
                return False, errors

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixture."""
        return {
            "fixture_id": self.fixture_id,
            "schema": self.schema,
            "valid_examples": self.valid_examples,
            "invalid_examples": self.invalid_examples,
            "constraints": self.constraints,
            "description": self.description,
        }


class FormatFixtureSet:
    """Collection of format fixtures for structured output testing."""

    def __init__(self):
        self._fixtures: dict[str, FormatFixture] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default format fixtures."""
        # JSON object with required fields
        self.add(FormatFixture(
            fixture_id="json_object_basic",
            schema={
                "type": "object",
                "required": ["name", "score"],
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                    "category": {"type": "string"},
                },
                "additionalProperties": False,
            },
            valid_examples=[
                {"name": "test", "score": 85.5, "category": "math"},
                {"name": "coding", "score": 90.0},
            ],
            invalid_examples=[
                {"name": "test"},  # missing score
                {"name": "test", "score": "high"},  # wrong type
                {"name": "test", "score": 85, "extra": "field"},  # forbidden field
            ],
            constraints=["required_fields", "type_checking", "no_extra_fields"],
            description="Basic JSON object with required name and score fields",
        ))

        # Array with exactly N records
        self.add(FormatFixture(
            fixture_id="array_exact_n",
            schema={
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "object", "required": ["id", "label"]},
            },
            valid_examples=[
                [{"id": "a", "label": "first"}, {"id": "b", "label": "second"}, {"id": "c", "label": "third"}],
            ],
            invalid_examples=[
                [{"id": "a", "label": "first"}, {"id": "b", "label": "second"}],  # too few
                [{"id": "a", "label": "first"}, {"id": "b", "label": "second"},
                 {"id": "c", "label": "third"}, {"id": "d", "label": "fourth"}],  # too many
            ],
            constraints=["exact_count", "preserved_ids"],
            description="Array with exactly 3 records, each with id and label",
        ))

        # Enum selection
        self.add(FormatFixture(
            fixture_id="enum_selection",
            schema={
                "type": "object",
                "required": ["status"],
                "properties": {
                    "status": {"type": "string", "enum": ["pass", "fail", "inconclusive"]},
                    "detail": {"type": "string"},
                },
            },
            valid_examples=[
                {"status": "pass", "detail": "All tests passed"},
                {"status": "fail"},
            ],
            invalid_examples=[
                {"status": "maybe"},  # not in enum
                {"status": 42},  # wrong type
            ],
            constraints=["enum_values"],
            description="Status field restricted to pass/fail/inconclusive",
        ))

    def add(self, fixture: FormatFixture) -> None:
        """Add a fixture to the set."""
        self._fixtures[fixture.fixture_id] = fixture

    def get(self, fixture_id: str) -> FormatFixture:
        """Get a fixture by ID."""
        return self._fixtures[fixture_id]

    def all(self) -> list[FormatFixture]:
        """Get all fixtures."""
        return list(self._fixtures.values())

    def validate_all(self, fixture_id: str) -> tuple[bool, list[str]]:
        """Validate all examples in a fixture against its schema."""
        fixture = self.get(fixture_id)
        errors = []
        for i, example in enumerate(fixture.valid_examples):
            valid, errs = fixture.validate(example)
            if not valid:
                errors.append(f"valid_example[{i}] failed: {errs}")
        for i, example in enumerate(fixture.invalid_examples):
            valid, errs = fixture.validate(example)
            if valid:
                errors.append(f"invalid_example[{i}] unexpectedly passed")
        return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# ToolFixture
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Definition of a tool for tool-calling fixtures."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters
    required: list[str] = field(default_factory=list)


@dataclass
class ToolFixture:
    """Tool-calling fixture with schema definitions (spec §14)."""
    fixture_id: str
    tool: ToolDefinition
    expected_calls: list[dict[str, Any]]
    expected_no_call: bool = False
    description: str = ""

    def validate_call(self, call: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a tool call against the expected schema."""
        errors = []
        tool_name = call.get("name", "")
        if tool_name != self.tool.name:
            errors.append(f"Expected tool '{self.tool.name}', got '{tool_name}'")

        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            errors.append("Arguments must be an object")
            return False, errors

        # Check required parameters
        for param in self.tool.required:
            if param not in arguments:
                errors.append(f"Missing required parameter: {param}")

        # Check parameter types against schema
        props = self.tool.parameters.get("properties", {})
        for param_name, param_schema in props.items():
            if param_name in arguments:
                val = arguments[param_name]
                ptype = param_schema.get("type")
                if ptype == "string" and not isinstance(val, str):
                    errors.append(f"Parameter '{param_name}' must be string")
                elif ptype == "integer" and not isinstance(val, int):
                    errors.append(f"Parameter '{param_name}' must be integer")
                elif ptype == "number" and not isinstance(val, (int, float)):
                    errors.append(f"Parameter '{param_name}' must be number")
                elif ptype == "boolean" and not isinstance(val, bool):
                    errors.append(f"Parameter '{param_name}' must be boolean")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool fixture."""
        return {
            "fixture_id": self.fixture_id,
            "tool": {
                "name": self.tool.name,
                "description": self.tool.description,
                "parameters": self.tool.parameters,
                "required": self.tool.required,
            },
            "expected_calls": self.expected_calls,
            "expected_no_call": self.expected_no_call,
            "description": self.description,
        }


class ToolFixtureSet:
    """Collection of tool-calling fixtures."""

    def __init__(self):
        self._fixtures: dict[str, ToolFixture] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default tool fixtures."""
        # Search tool
        self.add(ToolFixture(
            fixture_id="search_tool",
            tool=ToolDefinition(
                name="search_documents",
                description="Search documents by keyword",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                },
                required=["query"],
            ),
            expected_calls=[
                {"name": "search_documents", "arguments": {"query": "benchmark results", "max_results": 5}},
                {"name": "search_documents", "arguments": {"query": "context retrieval"}},
            ],
            expected_no_call=False,
            description="Search tool with required query and optional max_results",
        ))

        # Calculator tool
        self.add(ToolFixture(
            fixture_id="calculator_tool",
            tool=ToolDefinition(
                name="calculate",
                description="Perform arithmetic calculation",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "precision": {"type": "integer"},
                    },
                },
                required=["expression"],
            ),
            expected_calls=[
                {"name": "calculate", "arguments": {"expression": "2 + 2", "precision": 4}},
                {"name": "calculate", "arguments": {"expression": "sqrt(16)"}},
            ],
            expected_no_call=False,
            description="Calculator tool with expression and optional precision",
        ))

        # No-tool scenario
        self.add(ToolFixture(
            fixture_id="no_tool_needed",
            tool=ToolDefinition(
                name="send_email",
                description="Send an email",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
                required=["to", "subject", "body"],
            ),
            expected_calls=[],
            expected_no_call=True,
            description="Scenario where no tool should be called (e.g., simple greeting)",
        ))

    def add(self, fixture: ToolFixture) -> None:
        """Add a tool fixture."""
        self._fixtures[fixture.fixture_id] = fixture

    def get(self, fixture_id: str) -> ToolFixture:
        """Get a tool fixture by ID."""
        return self._fixtures[fixture_id]

    def all(self) -> list[ToolFixture]:
        """Get all tool fixtures."""
        return list(self._fixtures.values())

    def validate_expected_calls(self, fixture_id: str) -> tuple[bool, list[str]]:
        """Validate that all expected calls pass schema validation."""
        fixture = self.get(fixture_id)
        errors = []
        for i, call in enumerate(fixture.expected_calls):
            valid, errs = fixture.validate_call(call)
            if not valid:
                errors.append(f"expected_call[{i}] failed: {errs}")
        return len(errors) == 0, errors
