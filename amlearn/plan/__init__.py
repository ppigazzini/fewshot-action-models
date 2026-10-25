"""Two planners over one semantics, and the outcome they share.

Breadth-first search is the measuring planner: it proves a problem unsolvable by
exhausting the reachable states, which an ASP solver cannot do without a horizon bound.
The ASP planner exists to disagree with it. Both return plans of minimal length, so on
every problem both solve, the lengths must be equal; `tests/test_plan.py` holds them to it.
"""

from dataclasses import dataclass
from typing import Literal

from amlearn.strips import GroundAction

type Status = Literal["solved", "unsolvable", "limit"]


@dataclass(frozen=True, slots=True)
class Outcome:
    """A plan, or why there is none.

    ``unsolvable`` is a proof: every reachable state was expanded. ``limit`` is not: the
    node or horizon bound stopped the search first, and the problem may be solvable.
    """

    status: Status
    plan: tuple[GroundAction, ...] | None
    expanded: int
