"""Time budget tracking for constrained execution."""

from __future__ import annotations

import time
from typing import Callable


class BudgetClock:
    """Track elapsed time against a budget using an injected clock.

    Attributes:
        skipped: List of {"label": str, "skipped": True} for refused operations.
    """

    def __init__(self, seconds: float, clock: Callable[[], float] | None = None) -> None:
        """Initialize a time budget.

        Args:
            seconds: Budget duration in seconds; must be > 0.
            clock: Callable returning monotonic seconds; defaults to time.time.

        Raises:
            ValueError: If seconds <= 0.
        """
        if seconds <= 0:
            raise ValueError(f"Budget must be positive, got {seconds}")
        self.seconds = float(seconds)
        self.clock = clock if clock is not None else time.time
        self._start_time: float | None = None
        self.skipped: list[dict[str, object]] = []

    def start(self) -> None:
        """Start the budget clock. Idempotent."""
        if self._start_time is None:
            self._start_time = self.clock()

    def elapsed(self) -> float:
        """Return elapsed seconds since start."""
        if self._start_time is None:
            return 0.0
        return self.clock() - self._start_time

    def remaining(self) -> float:
        """Return remaining seconds; never negative."""
        return max(0.0, self.seconds - self.elapsed())

    def exceeded(self) -> bool:
        """Return True if budget has been exhausted."""
        return self.elapsed() >= self.seconds

    def guard(self, label: str) -> bool:
        """Check if an operation is allowed within budget.

        Returns True if budget remains; False if exceeded.
        Records refusals in self.skipped.
        Implicitly starts the budget on first call if not already started.
        """
        if self._start_time is None:
            self.start()
        if self.exceeded():
            self.skipped.append({"label": label, "skipped": True})
            return False
        return True
