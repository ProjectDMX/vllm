# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional DMI PCIe hint bridge for KV transfer connectors."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

try:
    from monitoring.governor import emit_hint as _emit_hint
except Exception:
    _emit_hint = None


def emit_dmi_pcie_hint(
    *,
    direction: str,
    source: str,
    est_bytes: int = 0,
    valid_until_ns: int = 0,
) -> None:
    """Emit a best-effort hint without coupling vLLM to DMI."""

    if _emit_hint is None:
        return
    with suppress(Exception):
        _emit_hint(
            direction=direction,
            source=source,
            est_bytes=est_bytes,
            valid_until_ns=valid_until_ns,
        )


def connector_manages_dmi_pcie_hints(connector: Any) -> bool:
    """Return whether a connector emits its own transfer-lifecycle hints."""

    try:
        return bool(getattr(connector, "manages_dmi_pcie_hints", False))
    except Exception:
        return False


class D2HHintLease:
    """Aggregate one connector's D2H activity into begin/renew/end hints.

    Connector lifecycle methods run on the worker thread, so this helper does
    not add locking or a background heartbeat. Long-lived asynchronous paths
    renew the lease from their existing completion poll.
    """

    def __init__(
        self,
        source: str,
        *,
        renew_interval_ns: int = 100_000_000,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not source:
            raise ValueError("source must be non-empty")
        self.source = source
        self._renew_interval_ns = max(0, int(renew_interval_ns))
        self._clock_ns = clock_ns
        self._active = False
        self._last_emit_ns = 0
        self._est_bytes = 0

    @property
    def active(self) -> bool:
        return self._active

    def start(self, *, est_bytes: int = 0) -> None:
        """Start the lease, or renew it if it is already active."""

        if self._active:
            self.renew(est_bytes=est_bytes)
            return

        now = self._clock_ns()
        self._active = True
        self._last_emit_ns = now
        self._est_bytes = max(0, int(est_bytes))
        emit_dmi_pcie_hint(
            direction="D2H",
            source=self.source,
            est_bytes=self._est_bytes,
        )

    def renew(self, *, est_bytes: int = 0) -> None:
        """Refresh an active lease at a bounded frequency."""

        if not self._active:
            self.start(est_bytes=est_bytes)
            return

        now = self._clock_ns()
        if now - self._last_emit_ns < self._renew_interval_ns:
            return

        self._last_emit_ns = now
        self._est_bytes = max(self._est_bytes, int(est_bytes or 0))
        emit_dmi_pcie_hint(
            direction="D2H",
            source=self.source,
            est_bytes=self._est_bytes,
        )

    def finish(self) -> None:
        """End the lease immediately after the tracked D2H operation finishes."""

        if not self._active:
            return

        self._active = False
        self._last_emit_ns = 0
        self._est_bytes = 0
        emit_dmi_pcie_hint(
            direction="D2H",
            source=self.source,
            valid_until_ns=self._clock_ns(),
        )
