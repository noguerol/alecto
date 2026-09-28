"""Capability probing for targets (spec §8, §10).

Probe levels: 'basic' and 'extended'. Unknown capabilities stay
None (never guessed).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProbeLevel(str, Enum):
    BASIC = "basic"
    EXTENDED = "extended"


@dataclass
class CapabilityReport:
    """Report of probed target capabilities.

    Any capability that was not observed stays None — never guessed.
    """

    streaming: bool | None = None
    tool_calling: bool | None = None
    reasoning: bool | None = None
    max_context: int | None = None
    max_output_tokens: int | None = None
    level: str = ProbeLevel.BASIC.value
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "streaming": self.streaming,
            "tool_calling": self.tool_calling,
            "reasoning": self.reasoning,
            "max_context": self.max_context,
            "max_output_tokens": self.max_output_tokens,
            "level": self.level,
            "notes": list(self.notes),
        }


def probe_target(target: Any, level: ProbeLevel | str = ProbeLevel.BASIC) -> CapabilityReport:
    """Probe a target for capabilities.

    The probe only records what is actually observed. Anything not
    observed remains None. The level is recorded on the report.
    """
    lvl = level if isinstance(level, ProbeLevel) else ProbeLevel(level)
    report = CapabilityReport(level=lvl.value)

    # Basic probe: check attributes/flags the target exposes, if any.
    # A mock or unknown target exposes nothing, so all fields stay None.
    if hasattr(target, "capabilities"):
        caps = target.capabilities
        if isinstance(caps, dict):
            if "streaming" in caps:
                report.streaming = bool(caps["streaming"])
            if "tool_calling" in caps:
                report.tool_calling = bool(caps["tool_calling"])
            if "reasoning" in caps:
                report.reasoning = bool(caps["reasoning"])
            if "max_context" in caps:
                report.max_context = int(caps["max_context"])
            if "max_output_tokens" in caps:
                report.max_output_tokens = int(caps["max_output_tokens"])
        else:
            report.notes.append("target exposes non-dict capabilities; not interpreted")
    else:
        report.notes.append("no capability attributes observed; all unknown")

    if lvl == ProbeLevel.EXTENDED:
        report.notes.append("extended probe requested; only observed fields populated")

    return report
