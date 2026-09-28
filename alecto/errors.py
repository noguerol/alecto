"""Exception hierarchy for alecto."""

from .enums import ErrorKind


class AlectoError(Exception):
    """Base exception for all alecto errors.

    ``code`` is the stable machine-readable error code from spec §19 (for
    example ``alecto.stream.malformed``). ``kind`` is the coarse
    :class:`ErrorKind` used by the runner/reporting layers.
    """

    code: str = "alecto.error.unknown"

    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.UNKNOWN,
        code: str | None = None,
        **context,
    ):
        super().__init__(message)
        self.kind = kind
        self.code = code or type(self).code
        self.context = context


class TargetUnreachableError(AlectoError):
    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.TARGET_UNREACHABLE, **context)


class TimeoutError(AlectoError):
    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.TIMEOUT, **context)


class InvalidResponseError(AlectoError):
    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.INVALID_RESPONSE, **context)


class ResourceExhaustedError(AlectoError):
    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.RESOURCE_EXHAUSTED, **context)


class CancelledError(AlectoError):
    def __init__(self, message: str = "Task cancelled", **context):
        super().__init__(message, kind=ErrorKind.CANCELLED, **context)


class ConfigurationError(AlectoError):
    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.UNKNOWN, **context)


class ValidationError(AlectoError):
    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.INVALID_RESPONSE, **context)


class RefusalError(AlectoError):
    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.UNKNOWN, **context)


class CapabilityUnavailable(AlectoError):
    """An optional adapter capability is not available on this target.

    Stable code ``alecto.backend.capability_missing`` (spec §5.2, §19).
    Raising this is how optional methods avoid inventing estimates and
    labelling them native measurements.
    """

    code = "alecto.backend.capability_missing"

    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.UNKNOWN, **context)


class StreamMalformedError(AlectoError):
    """The SSE stream carried malformed data.

    Stable code ``alecto.stream.malformed`` (spec §19): preserve a bounded
    raw excerpt as evidence and fail the attempt. Callers must never
    silently truncate or discard the evidence.
    """

    code = "alecto.stream.malformed"

    def __init__(self, message: str, **context):
        super().__init__(message, kind=ErrorKind.INVALID_RESPONSE, **context)
