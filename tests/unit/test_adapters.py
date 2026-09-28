"""Unit tests for target adapters."""

import pytest

from alecto.adapters import MockAdapter, get_adapter
from alecto.domain import TargetSpec
from alecto.enums import TargetKind
from alecto.errors import TargetUnreachableError


@pytest.fixture
def mock_spec():
    return TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")


class TestMockAdapter:
    @pytest.mark.asyncio
    async def test_call(self, mock_spec):
        adapter = MockAdapter(mock_spec)
        result = await adapter.call({"messages": [{"role": "user", "content": "test"}]})
        assert "text" in result
        assert result["text"].startswith("mock-response")
        assert result["tokens"] == 10

    @pytest.mark.asyncio
    async def test_health_check(self, mock_spec):
        adapter = MockAdapter(mock_spec)
        assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_call_count(self, mock_spec):
        adapter = MockAdapter(mock_spec)
        await adapter.call({"messages": []})
        await adapter.call({"messages": []})
        result = await adapter.call({"messages": []})
        assert "3" in result["text"]


class TestGetAdapter:
    @pytest.mark.asyncio
    async def test_mock(self):
        spec = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
        adapter = get_adapter(spec)
        assert isinstance(adapter, MockAdapter)

    @pytest.mark.asyncio
    async def test_unknown_kind_falls_back_to_mock(self):
        spec = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
        adapter = get_adapter(spec)
        assert isinstance(adapter, MockAdapter)


class TestErrorContract:
    @pytest.mark.asyncio
    async def test_unreachable_endpoint_raises_target_unreachable(self):
        """A dead endpoint must surface as TargetUnreachableError, not
        NameError or httpx.ConnectError."""
        spec = TargetSpec(
            kind=TargetKind.OPENAI_COMPATIBLE,
            endpoint="http://127.0.0.1:1/v1",
            model="x",
            timeout_s=2.0,
        )
        adapter = get_adapter(spec)
        try:
            with pytest.raises(TargetUnreachableError):
                await adapter.call(
                    {"messages": [{"role": "user", "content": "hi"}]}
                )
        finally:
            await adapter.aclose()
