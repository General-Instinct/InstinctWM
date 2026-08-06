"""The pass registry.

Registration order is evaluation order, and it is load-bearing where one pass is a
precondition for another. The current order removes substrate overhead first — those passes
are unconditional and bit-exact — and then runs the passes derived from model declarations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from instinctwm.optimizer.passes.cfg_elision import CFGBranchElision
from instinctwm.optimizer.passes.conditioning_prefill import ConditioningPrefill
from instinctwm.optimizer.passes.operator_fusion import OperatorFusion
from instinctwm.optimizer.passes.substrate import (
    AllocatorChurnElision,
    DebugDumpElision,
    FSDPElision,
    ObsDecodeElision,
)

if TYPE_CHECKING:
    from instinctwm.optimizer.base import OptimizationPass


def default_passes() -> "list[OptimizationPass]":
    """Every pass InstinctWM ships, in evaluation order.

    Returns fresh instances rather than a shared module-level list: passes are stateless
    today, but a shared mutable default is the kind of thing that stops being true quietly.
    """
    return [
        FSDPElision(),
        AllocatorChurnElision(),
        DebugDumpElision(),
        ObsDecodeElision(),
        ConditioningPrefill(),
        CFGBranchElision(),
        # Last: it is the only pass that rewrites the block body, and it composes with the two
        # other rewrites by detecting them rather than by ordering. Evaluating it last keeps
        # `explain()` reading in layer order (substrate -> graph -> cache -> kernel).
        OperatorFusion(),
    ]


__all__ = [
    "AllocatorChurnElision",
    "CFGBranchElision",
    "ConditioningPrefill",
    "DebugDumpElision",
    "FSDPElision",
    "ObsDecodeElision",
    "OperatorFusion",
    "default_passes",
]
