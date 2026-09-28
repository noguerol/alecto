"""Enumerations and error codes for the alecto system."""

from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class LoopMode(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


class TargetKind(str, Enum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"
    LLM = "llm"
    LLMSTUDIO = "llmstudio"
    MOCK = "mock"


class EvidenceKind(str, Enum):
    FILE = "file"
    TEXT = "text"
    METRIC = "metric"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class RefusalReason(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    UNVERIFIABLE = "unverifiable"
    RESOURCE_LIMIT = "resource_limit"
    POLICY_VIOLATION = "policy_violation"


class ErrorKind(str, Enum):
    TARGET_UNREACHABLE = "target_unreachable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class RetryPolicy(str, Enum):
    NEVER = "never"
    ONCE = "once"
    EXPONENTIAL = "exponential"


class ComparisonMode(str, Enum):
    PAIRWISE = "pairwise"
    GROUP = "group"


class LoopStrategy(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"


class BenchmarkCategory(str, Enum):
    REASONING = "reasoning"
    CODING = "coding"
    MATH = "math"
    LANGUAGE = "language"
    GENERAL = "general"
