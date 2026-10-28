"""Check result container and the Pass / Watch / Fail classification rule."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Status(IntEnum):
    PASS = 0
    WATCH = 1
    FAIL = 2

    @property
    def label(self) -> str:
        return self.name.capitalize()


def classify(
    value: float,
    watch: float,
    fail: float,
    higher_is_worse: bool = True,
    inclusive: bool = True,
) -> Status:
    """Map a metric to a status.

    With ``higher_is_worse`` a value at or above ``watch`` is Watch and at or above ``fail``
    is Fail (``inclusive=False`` makes the comparison strict). The mirror image applies when
    lower values are worse. NaN is treated as Fail: a check that cannot be computed is not a pass.
    """
    if value is None or value != value:
        return Status.FAIL
    v, w, f = (value, watch, fail) if higher_is_worse else (-value, -watch, -fail)
    if inclusive:
        return Status.FAIL if v >= f else Status.WATCH if v >= w else Status.PASS
    return Status.FAIL if v > f else Status.WATCH if v > w else Status.PASS


def worst(statuses: list[Status]) -> Status:
    return max(statuses, default=Status.PASS)


@dataclass
class CheckResult:
    check_id: str
    name: str
    metric: str
    value: float | None
    status: Status
    threshold: str
    rationale: str
    observation: str
    recommendation: str
    details: dict[str, Any] = field(default_factory=dict)
