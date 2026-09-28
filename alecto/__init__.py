"""Alecto — Local-first, agent-native execution engine for LLM benchmarking.

This module exposes the stable public API.  Heavy or optional integrations are
imported lazily by their own modules; the names below are safe to import from
the base installation (spec §4.1, AC-22).
"""

from .adapters import TargetAdapter, get_adapter
from .config import AlectoConfig, default_config
from .domain import Plan, TargetSpec, Task
from .enums import LoopMode, TargetKind
from .errors import AlectoError
from .performance import PerformanceCell, run_performance_cell
from .quality import QUALITY_SUITES, load_quality_samples, run_quality_suite
from .refusal import compute_refusal_metrics
from .streaming import StreamEvent

__version__ = "0.1.3"

__all__ = [
    "__version__",
    # adapters
    "TargetAdapter",
    "get_adapter",
    # domain / enums
    "Plan",
    "Task",
    "TargetSpec",
    "LoopMode",
    "TargetKind",
    # errors
    "AlectoError",
    # configuration
    "AlectoConfig",
    "default_config",
    # performance
    "PerformanceCell",
    "run_performance_cell",
    # quality
    "QUALITY_SUITES",
    "load_quality_samples",
    "run_quality_suite",
    # refusal
    "compute_refusal_metrics",
    # streaming
    "StreamEvent",
]
