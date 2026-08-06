"""OperatorFusion (L5) — run a registered kernel over a declared fusible region.

This is the pass that connects `instinctwm/kernels/` to a plan. Before it existed the kernel
layer was a framework with no caller: regions were declared in `kernels/lingbot_regions.py`,
kernels registered themselves against those region names, and nothing outside `tests/` ever
imported either. The measured 3.38x came entirely from GRAPH and CACHE.

WHAT IT DECIDES, AND WHAT IT REFUSES TO DECIDE
----------------------------------------------
Legality and tier are decided here, from the adapter's declaration plus the kernel's declared
properties — `kernels/registry.py:derive_tier` does the work and this pass supplies the region.
Profitability is NOT decided here. The pass reports the launch count it expects to remove, and
the *installer* re-measures on the target's real shapes and refuses to arm a kernel that loses.
Both halves are load-bearing, and they fail independently: on this A100 the same bit-exact
kernel is 1.26x on the video stream and 0.93x on the action stream, so a pass that fired on
legality alone would have shipped a regression on 51 of the 77 denoise forwards.

TIER, AND WHY THIS PASS ONLY TAKES BITEXACT REGIONS BY DEFAULT
---------------------------------------------------------------
`Plan.tier()` is the weakest claim in the plan. LingBot-VA declares two fusible regions and
they land on opposite sides of that line:

  * `post_attention_gated_residual` — pure elementwise, no reduction. A kernel that reproduces
    the eager rounding points and does not contract the multiply-add reaches BITEXACT.
  * `pre_attention_modulated_norm` — contains a LayerNorm. Fusing a reduction re-trees it unless
    the kernel is written not to, so BITEXACT there is a claim about a kernel that does not
    exist yet rather than about the region.

Taking both would cost the whole plan its bit-exactness, and with it the gate that makes every
other number in `RESULTS.md` quotable, to buy a fusion in a region whose kernel has not been
written yet. So the ceiling defaults to BITEXACT and the NUMERIC region is reported rather than
silently dropped. `OperatorFusion(tier_ceiling=Tier.NUMERIC)` opts in.

WHY PLAN TIME NEVER TOUCHES THE KERNEL REGISTRY
------------------------------------------------
The obvious implementation asks `KernelRegistry.candidates(region, DeviceProfile.probe())` here
and reports the winning kernel by name. It is also wrong: populating the registry means
importing the kernel modules, which import torch, and `test_adapter.py:
test_planning_does_not_import_torch` exists because a plan has to be inspectable on a laptop.

So the split runs along the same seam as every other pass. What the *region* admits is decided
here, from declared structure — effects, reductions, rounding points — with no kernel in sight.
Which kernel, whether it is legal on this device, and whether it beats eager are decided by the
installer, where the registry, the device and the real shapes all exist. The installer then
re-derives the tier from the kernel it picked and refuses if it is weaker than what this pass
claimed, so the two halves cannot drift apart silently.
"""

from __future__ import annotations

from instinctwm.adapter.base import AdapterSpec
from instinctwm.deployment import DeploymentSpec
from instinctwm.kernels.regions import FusibleRegion
from instinctwm.optimizer.base import PassResult, Tier


def admissible_tier(region: FusibleRegion) -> tuple[Tier, str]:
    """The strongest claim a fusion of this region could earn, from structure alone.

    Deliberately the same rule order as `kernels/registry.py:derive_tier`, minus the two checks
    that are properties of a kernel rather than of the region (`matches_reference_contraction`
    and `preserves_intermediate_rounding`). Those can only weaken the result, which is why the
    installer re-derives with the real kernel and refuses on a downgrade.
    """
    if region.has_effects():
        return Tier.BEHAVIORAL, (
            f"{region.name!r} contains an effectful op; fusing reorders a side effect")
    if region.has_reduction():
        return Tier.NUMERIC, (
            f"{region.name!r} contains a reduction, so a fused kernel re-trees it unless it is "
            f"written to preserve the eager order; that is a property of a kernel that does not "
            f"exist yet, not of the region")
    rp = region.rounding_points()
    return Tier.BITEXACT, (
        f"{region.name!r} is elementwise with no reduction and no effects; a kernel that "
        f"reproduces its {len(rp)} rounding point(s) {list(rp)} and does not contract the "
        f"multiply-add can be bit-exact")


class OperatorFusion:
    name = "operator_fusion"

    def __init__(self, tier_ceiling: Tier = Tier.BITEXACT):
        self._ceiling = tier_ceiling

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        fusion = getattr(spec, "fusion", None)
        if fusion is None or not fusion.regions:
            return PassResult(self.name, False, Tier.BITEXACT,
                              "the adapter declares no fusible regions")

        selected: dict[str, dict] = {}
        declined: list[str] = []
        for region in fusion.regions:
            tier, why = admissible_tier(region)
            if tier > self._ceiling:
                # Say WHY, not just that it did not fire: a region that is one tier too weak
                # reads identically to a region with no kernel in a bare skip line.
                declined.append(f"{region.name} (admits {tier.name} at best > ceiling "
                                f"{self._ceiling.name}: {why})")
                continue
            selected[region.name] = {
                "tier": tier.name,
                "why": why,
                "eager_launches": fusion.launches_per_region.get(region.name, 0),
                "occurrences_per_forward": region.occurrences_per_forward,
            }

        if not selected:
            return PassResult(self.name, False, Tier.BITEXACT,
                              f"no region fused: {'; '.join(declined)}")

        tier = max(Tier[v["tier"]] for v in selected.values())
        forwards = spec.total_forwards()
        removed = sum(
            max(v["eager_launches"] - 1, 0) * v["occurrences_per_forward"] * forwards
            for v in selected.values()
        )
        took = ", ".join(f"{n} [{v['tier']}]" for n, v in sorted(selected.items()))
        reason = f"fusible regions admitting a fusion at or below the ceiling: {took}"
        if declined:
            reason += f"; not taken: {'; '.join(declined)}"

        return PassResult(
            name=self.name,
            applies=True,
            tier=tier,
            reason=reason,
            params={"regions": selected, "tier_ceiling": self._ceiling.name,
                    "requires": ["triton"]},
            expected_win=(
                f"~{removed:,} kernel launches per control cycle "
                f"({forwards} forwards x {sum(v['occurrences_per_forward'] for v in selected.values())} "
                f"occurrences), and ~70% of the region's memory traffic. The realized win is "
                f"bounded by the launch-vs-traffic mix at each shape, so the installer re-measures "
                f"and leaves shapes below the measured break-even eager"
            ),
        )

    # ---- install -------------------------------------------------------------------------
    def install(self, server_module, server_cls) -> list[str]:
        from instinctwm.runtime.fused_residual import install_gated_residual_fusion

        return install_gated_residual_fusion(server_module, server_cls)
