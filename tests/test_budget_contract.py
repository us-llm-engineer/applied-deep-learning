from __future__ import annotations

import pytest

from src.budget import BudgetClock


class FakeClock:
    """Manually driven monotonic clock; never sleeps."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TickingClock:
    """Advances by a fixed step on every read."""

    def __init__(self, step: float = 10.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _started(seconds: float, clock: FakeClock) -> BudgetClock:
    budget = BudgetClock(seconds, clock=clock)
    budget.start()
    return budget


def test_elapsed_and_remaining_follow_the_injected_clock() -> None:
    clock = FakeClock()
    budget = _started(60.0, clock)
    clock.now += 25.0
    assert budget.elapsed() == pytest.approx(25.0)
    assert budget.remaining() == pytest.approx(35.0)
    assert budget.exceeded() is False


def test_exceeded_flips_exactly_at_the_limit() -> None:
    clock = FakeClock()
    budget = _started(60.0, clock)
    clock.now += 59.999
    assert budget.exceeded() is False
    clock.now += 0.001
    assert budget.exceeded() is True


def test_remaining_never_goes_negative() -> None:
    clock = FakeClock()
    budget = _started(60.0, clock)
    clock.now += 500.0
    assert budget.remaining() == 0.0
    assert budget.exceeded() is True


def test_start_is_idempotent_and_does_not_restart_the_clock() -> None:
    clock = FakeClock()
    budget = BudgetClock(60.0, clock=clock)
    budget.start()
    clock.now += 5.0
    budget.start()
    clock.now += 5.0
    assert budget.elapsed() == pytest.approx(10.0)


@pytest.mark.parametrize("seconds", [0, 0.0, -1.0, -45 * 60])
def test_non_positive_budget_is_rejected(seconds: float) -> None:
    with pytest.raises(ValueError):
        BudgetClock(seconds, clock=FakeClock())


def test_guard_true_while_budget_remains_records_nothing() -> None:
    clock = FakeClock()
    budget = _started(60.0, clock)
    assert budget.guard("run-a") is True
    clock.now += 10.0
    assert budget.guard("run-b") is True
    assert budget.skipped == []


def test_guard_false_after_exhaustion_appends_one_record_per_refusal() -> None:
    clock = FakeClock()
    budget = _started(60.0, clock)
    clock.now += 60.0
    assert budget.guard("run-c") is False
    assert budget.skipped == [{"label": "run-c", "skipped": True}]
    assert budget.guard("run-d") is False
    assert budget.skipped == [
        {"label": "run-c", "skipped": True},
        {"label": "run-d", "skipped": True},
    ]


def test_guard_sequence_with_clock_advancing_ten_seconds_per_call() -> None:
    """Given a 45 s budget and a clock that jumps 10 s per read; Then guards go True... then False forever, and only Falses are recorded."""
    budget = BudgetClock(45.0, clock=TickingClock(10.0))
    results = [budget.guard(f"run-{i}") for i in range(10)]

    assert results[0] is True
    first_false = results.index(False)
    assert all(r is True for r in results[:first_false])
    assert all(r is False for r in results[first_false:]), "guard must not recover once exhausted"
    assert first_false <= 4, "a 10 s/read clock must exhaust a 45 s budget within 4 guards"
    assert budget.skipped == [
        {"label": f"run-{i}", "skipped": True} for i in range(first_false, 10)
    ]
