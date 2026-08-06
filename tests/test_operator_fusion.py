#!/usr/bin/env python3
"""The L5 pass: what it takes, what it refuses, and what it must not import.

`test_kernel_framework.py` covers the kernel layer in isolation. This covers the seam that layer
was missing until now — an adapter declaring regions, a pass reading them, and a plan whose tier
survives the result.

The interesting assertions are the negative ones. A pass that fuses everything it can is easy;
this one has to decline the LayerNorm region without dropping it silently, and it has to reach
its verdict without loading a kernel, because deciding what is legal must work on a laptop.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctwm.adapter.lingbot_va import lingbot_va_spec  # noqa: E402
from instinctwm.deployment import DeploymentSpec  # noqa: E402
from instinctwm.kernels.lingbot_regions import POST_ATTENTION, PRE_ATTENTION  # noqa: E402
from instinctwm.kernels.regions import FusibleRegion, OpKind, OpSpec  # noqa: E402
from instinctwm.optimizer.base import Optimizer, Tier  # noqa: E402
from instinctwm.optimizer.passes.operator_fusion import (  # noqa: E402
    OperatorFusion, admissible_tier)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_adapter_declares_its_regions():
    """Without this the pass has nothing to read, which is what 'not wired in' looked like."""
    spec = lingbot_va_spec()
    assert spec.fusion is not None, "lingbot-va declares no fusible regions"
    names = {r.name for r in spec.fusion.regions}
    assert names == {"pre_attention_modulated_norm", "post_attention_gated_residual"}, names


def test_tier_comes_from_structure():
    assert admissible_tier(POST_ATTENTION)[0] is Tier.BITEXACT
    # the LayerNorm is the whole reason, so the reason has to say so
    tier, why = admissible_tier(PRE_ATTENTION)
    assert tier is Tier.NUMERIC, tier
    assert "reduction" in why, why

    effectful = FusibleRegion(
        name="x", ops=(OpSpec("rng", OpKind.EFFECTFUL),),
        boundary_in=("a",), boundary_out=("b",))
    assert admissible_tier(effectful)[0] is Tier.BEHAVIORAL


def test_the_residual_region_has_one_rounding_point_and_two_sites():
    """Both were wrong before, in ways that cancelled.

    `gate` was declared as materialising in bf16 on the belief that `attn_out * gate` rounds
    there. It does not — `gate_msa` is a chunk of `temb.float()` — and the kernel that claimed to
    reproduce that rounding does not reproduce it either. Two errors agreeing is not a check.
    """
    assert POST_ATTENTION.rounding_points() == ("residual_add",), \
        POST_ATTENTION.rounding_points()
    assert not POST_ATTENTION.has_reduction()
    for region in (POST_ATTENTION, PRE_ATTENTION):
        assert region.occurrences_per_forward == 60, \
            f"{region.name}: 30 layers x 2 sites per block, not {region.occurrences_per_forward}"


def test_it_takes_the_elementwise_region_and_declines_the_reduction():
    r = OperatorFusion().evaluate(lingbot_va_spec(), DeploymentSpec())
    assert r.applies and r.tier is Tier.BITEXACT, (r.applies, r.tier)
    assert "post_attention_gated_residual" in r.params["regions"]
    assert "pre_attention_modulated_norm" not in r.params["regions"]
    # declined, not vanished
    assert "pre_attention_modulated_norm" in r.reason, r.reason


def test_a_numeric_ceiling_opts_into_both():
    r = OperatorFusion(tier_ceiling=Tier.NUMERIC).evaluate(lingbot_va_spec(), DeploymentSpec())
    assert r.applies and r.tier is Tier.NUMERIC, (r.applies, r.tier)
    assert set(r.params["regions"]) == {"pre_attention_modulated_norm",
                                        "post_attention_gated_residual"}


def test_an_adapter_without_regions_declines():
    class NoFusion:
        pass

    spec = lingbot_va_spec()
    r = OperatorFusion().evaluate(
        type(spec)(**{**spec.__dict__, "fusion": None}), DeploymentSpec())
    assert not r.applies and "no fusible regions" in r.reason, r.reason


def test_the_default_plan_stays_bitexact():
    """The point of the BITEXACT ceiling. If this flips, every quoted number needs a re-gate."""
    plan = Optimizer(tier_ceiling=Tier.BITEXACT).compile(lingbot_va_spec())
    assert plan.tier() is Tier.BITEXACT, plan.tier()
    assert "operator_fusion" in {r.name for r in plan.applied}


def test_evaluating_the_pass_does_not_import_torch():
    """Same contract as test_adapter.py, asserted against THIS pass specifically.

    It is the only pass whose subject matter is a CUDA kernel, so it is the one that will grow a
    convenient `DeviceProfile.probe()` at plan time and take the analysis path down with it.
    """
    code = (
        "import sys; "
        "from instinctwm.adapter.lingbot_va import lingbot_va_spec; "
        "from instinctwm.deployment import DeploymentSpec; "
        "from instinctwm.optimizer.passes.operator_fusion import OperatorFusion; "
        "OperatorFusion().evaluate(lingbot_va_spec(), DeploymentSpec()); "
        "assert 'torch' not in sys.modules, 'the L5 pass pulled in torch at plan time'"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
