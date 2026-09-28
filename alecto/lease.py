"""Resource lease management (spec §4.6)."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class ResourceLease:
    resource_id: str
    holder_id: str
    acquired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    released: bool = False

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def renew(self, duration_s: float) -> None:
        self.expires_at = datetime.now(timezone.utc)
        self.expires_at += timedelta(seconds=duration_s)


class LeaseManager:
    """Manages resource leases for concurrent task execution."""

    def __init__(self, default_lease_duration_s: float = 30.0):
        self.default_duration = default_lease_duration_s
        self._leases: dict[str, ResourceLease] = {}
        self._resources: dict[str, str | None] = {}  # resource_id -> holder_id

    def acquire(self, resource_id: str, holder_id: str, duration_s: float | None = None) -> ResourceLease:
        duration = duration_s or self.default_duration
        lease = ResourceLease(
            resource_id=resource_id,
            holder_id=holder_id,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=duration),
        )
        self._leases[lease.resource_id] = lease
        self._resources[resource_id] = holder_id
        return lease

    def release(self, resource_id: str) -> None:
        if resource_id in self._leases:
            self._leases[resource_id].released = True
            self._resources[resource_id] = None

    def is_available(self, resource_id: str) -> bool:
        holder = self._resources.get(resource_id)
        if holder is None:
            return True
        lease = self._leases.get(resource_id)
        if lease and lease.is_expired():
            return True
        return False

    def get_active_leases(self) -> list[ResourceLease]:
        return [lease for lease in self._leases.values()
                if not lease.released and not lease.is_expired()]
