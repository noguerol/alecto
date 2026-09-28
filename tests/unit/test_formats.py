"""Unit tests for alecto.context — FormatFixture and ToolFixture."""


from alecto.context import (
    FormatFixture,
    FormatFixtureSet,
    ToolDefinition,
    ToolFixture,
    ToolFixtureSet,
)


class TestFormatFixture:
    def test_validate_valid_object(self):
        fixture = FormatFixture(
            fixture_id="test_obj",
            schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "score": {"type": "number"}},
                "additionalProperties": False,
            },
            valid_examples=[{"name": "test", "score": 90.5}],
            invalid_examples=[{"score": 90.5}],
        )
        valid, errors = fixture.validate({"name": "test", "score": 90.5})
        assert valid is True
        assert errors == []

    def test_validate_missing_required(self):
        fixture = FormatFixture(
            fixture_id="test_missing",
            schema={"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
            valid_examples=[],
            invalid_examples=[{}],
        )
        valid, errors = fixture.validate({})
        assert valid is False
        assert any("name" in e for e in errors)

    def test_validate_wrong_type(self):
        fixture = FormatFixture(
            fixture_id="test_type",
            schema={"type": "object", "required": ["score"], "properties": {"score": {"type": "number"}}},
            valid_examples=[],
            invalid_examples=[{"score": "high"}],
        )
        valid, errors = fixture.validate({"score": "high"})
        assert valid is False
        assert any("score" in e for e in errors)

    def test_validate_forbidden_field(self):
        fixture = FormatFixture(
            fixture_id="test_forbidden",
            schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
            valid_examples=[],
            invalid_examples=[{"name": "x", "extra": "y"}],
        )
        valid, errors = fixture.validate({"name": "x", "extra": "y"})
        assert valid is False
        assert any("extra" in e for e in errors)

    def test_validate_array(self):
        fixture = FormatFixture(
            fixture_id="test_array",
            schema={"type": "array", "minItems": 2, "maxItems": 5},
            valid_examples=[[1, 2, 3]],
            invalid_examples=[[1]],
        )
        valid, errors = fixture.validate([1, 2, 3])
        assert valid is True
        valid, errors = fixture.validate([1])
        assert valid is False

    def test_validate_string_type(self):
        fixture = FormatFixture(
            fixture_id="test_string",
            schema={"type": "string"},
            valid_examples=["hello"],
            invalid_examples=[42],
        )
        valid, errors = fixture.validate("hello")
        assert valid is True
        valid, errors = fixture.validate(42)
        assert valid is False

    def test_to_dict(self):
        fixture = FormatFixture(
            fixture_id="test_dict",
            schema={"type": "object"},
            valid_examples=[{}],
            invalid_examples=[],
            description="Test fixture",
        )
        d = fixture.to_dict()
        assert d["fixture_id"] == "test_dict"
        assert d["schema"] == {"type": "object"}
        assert d["description"] == "Test fixture"


class TestFormatFixtureSet:
    def test_default_fixtures(self):
        fs = FormatFixtureSet()
        fixtures = fs.all()
        assert len(fixtures) >= 3
        ids = [f.fixture_id for f in fixtures]
        assert "json_object_basic" in ids
        assert "array_exact_n" in ids
        assert "enum_selection" in ids

    def test_get_fixture(self):
        fs = FormatFixtureSet()
        fixture = fs.get("json_object_basic")
        assert fixture.fixture_id == "json_object_basic"

    def test_validate_all(self):
        fs = FormatFixtureSet()
        for fixture in fs.all():
            valid, errors = fs.validate_all(fixture.fixture_id)
            assert valid, f"Fixture {fixture.fixture_id} failed: {errors}"

    def test_add_custom_fixture(self):
        fs = FormatFixtureSet()
        custom = FormatFixture(
            fixture_id="custom_test",
            schema={"type": "string"},
            valid_examples=["ok"],
            invalid_examples=[123],
        )
        fs.add(custom)
        assert fs.get("custom_test") is custom


class TestToolFixture:
    def test_validate_correct_call(self):
        fixture = ToolFixture(
            fixture_id="test_search",
            tool=ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                required=["query"],
            ),
            expected_calls=[{"name": "search", "arguments": {"query": "test"}}],
        )
        valid, errors = fixture.validate_call({"name": "search", "arguments": {"query": "test"}})
        assert valid is True
        assert errors == []

    def test_validate_wrong_tool_name(self):
        fixture = ToolFixture(
            fixture_id="test_wrong_name",
            tool=ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                required=["query"],
            ),
            expected_calls=[],
        )
        valid, errors = fixture.validate_call({"name": "other_tool", "arguments": {"query": "test"}})
        assert valid is False
        assert any("search" in e for e in errors)

    def test_validate_missing_required_param(self):
        fixture = ToolFixture(
            fixture_id="test_missing_param",
            tool=ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                required=["query"],
            ),
            expected_calls=[],
        )
        valid, errors = fixture.validate_call({"name": "search", "arguments": {}})
        assert valid is False
        assert any("query" in e for e in errors)

    def test_validate_wrong_param_type(self):
        fixture = ToolFixture(
            fixture_id="test_wrong_type",
            tool=ToolDefinition(
                name="calc",
                description="Calculate",
                parameters={"type": "object", "properties": {"count": {"type": "integer"}}},
                required=["count"],
            ),
            expected_calls=[],
        )
        valid, errors = fixture.validate_call({"name": "calc", "arguments": {"count": "five"}})
        assert valid is False
        assert any("count" in e for e in errors)

    def test_to_dict(self):
        fixture = ToolFixture(
            fixture_id="test_dict",
            tool=ToolDefinition(name="test", description="Test", parameters={"type": "object"}, required=[]),
            expected_calls=[{"name": "test", "arguments": {}}],
            description="Test fixture",
        )
        d = fixture.to_dict()
        assert d["fixture_id"] == "test_dict"
        assert d["tool"]["name"] == "test"
        assert d["expected_calls"] == [{"name": "test", "arguments": {}}]


class TestToolFixtureSet:
    def test_default_fixtures(self):
        fs = ToolFixtureSet()
        fixtures = fs.all()
        assert len(fixtures) >= 3
        ids = [f.fixture_id for f in fixtures]
        assert "search_tool" in ids
        assert "calculator_tool" in ids
        assert "no_tool_needed" in ids

    def test_get_fixture(self):
        fs = ToolFixtureSet()
        fixture = fs.get("search_tool")
        assert fixture.tool.name == "search_documents"

    def test_validate_expected_calls(self):
        fs = ToolFixtureSet()
        for fixture in fs.all():
            valid, errors = fs.validate_expected_calls(fixture.fixture_id)
            assert valid, f"Tool fixture {fixture.fixture_id} failed: {errors}"

    def test_no_tool_fixture(self):
        fs = ToolFixtureSet()
        fixture = fs.get("no_tool_needed")
        assert fixture.expected_no_call is True
        assert fixture.expected_calls == []

    def test_add_custom_fixture(self):
        fs = ToolFixtureSet()
        custom = ToolFixture(
            fixture_id="custom_tool",
            tool=ToolDefinition(
                name="custom",
                description="Custom tool",
                parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
                required=["arg"],
            ),
            expected_calls=[{"name": "custom", "arguments": {"arg": "value"}}],
        )
        fs.add(custom)
        assert fs.get("custom_tool") is custom
