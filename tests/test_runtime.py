"""Tests for plan installation.

The property under test is narrow and important: a server must never come up claiming a pass
that was not applied. `plan.explain()` is what every measurement is labelled with, so an
installer that silently skips a pass turns a real number into a mislabelled one.

Skipped when torch is absent — the runtime layer needs it, the rest of the package does not.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctwm import Optimizer, Tier, load  # noqa: E402  (needs the insert above)

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_TORCH = False


class _FakeServerModule:
    """Enough of `wan_va_server` for the installers to bind to."""

    def __init__(self):
        self.save_async = lambda obj, path: ("wrote", path)
        self._configure_model = lambda *a, **k: None
        self.VA_Server = type("VA_Server", (), {"_infer": lambda self, obs, frame_st_id=0: None})


def test_install_plan_applies_the_bitexact_substrate_passes():
    if not HAVE_TORCH:
        return
    from instinctwm.runtime.lingbot_install import install_plan

    model = load("lingbot-va-posttrain-robotwin")
    plan = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
    server = _FakeServerModule()

    # Both dropped for the same reason: their installers patch the REAL `modules.model`, which a
    # fake server module cannot stand in for. `operator_fusion` joined `default_passes()` after
    # this test was written, so the default plan started carrying an installer that needs the
    # lingbot tree and this test began failing on any box without it. It is named for the
    # SUBSTRATE passes; a kernel pass is not one of them.
    applied = install_plan(server, server.VA_Server,
                           plan.without("conditioning_prefill", "operator_fusion"))

    assert "fsdp_elision" in applied
    assert "debug_dump_elision" in applied
    assert "allocator_churn_elision" in applied
    # obs_decode_elision has no runtime action on this backend; it must still be REPORTED.
    assert any(a.startswith("obs_decode_elision") for a in applied)
    assert server.save_async(None, "x") is None, "debug dump should be neutered"


def test_install_plan_refuses_a_pass_it_cannot_install():
    if not HAVE_TORCH:
        return
    from instinctwm.runtime.lingbot_install import install_plan

    model = load("lingbot-va-posttrain-robotwin")
    # cfg_branch_elision is analysed but has no installer yet.
    plan = Optimizer(tier_ceiling=Tier.NUMERIC).compile(model.spec())
    assert "cfg_branch_elision" in [r.name for r in plan.applied]

    server = _FakeServerModule()
    try:
        install_plan(server, server.VA_Server, plan)
    except NotImplementedError as exc:
        assert "cfg_branch_elision" in str(exc)
        assert "plan.without(" in str(exc)
        return
    raise AssertionError("install_plan must refuse a plan it cannot fully apply")


def test_installers_refuse_a_server_that_changed_shape():
    if not HAVE_TORCH:
        return
    from instinctwm.runtime.lingbot_install import install_debug_dump_elision

    class _Renamed:
        pass

    try:
        install_debug_dump_elision(_Renamed())
    except RuntimeError as exc:
        assert "save_async" in str(exc)
        return
    raise AssertionError("an installer must raise rather than no-op on an unknown server")


def test_resolve_lingbot_root_reports_what_is_missing():
    if not HAVE_TORCH:
        return
    from instinctwm.runtime.lingbot_install import resolve_lingbot_root

    try:
        resolve_lingbot_root("/definitely/not/here")
    except FileNotFoundError as exc:
        assert "LINGBOT_ROOT" in str(exc)
        return
    raise AssertionError("resolve_lingbot_root should raise on a missing checkout")


if __name__ == "__main__":
    # Script-style entry, matching the rest of this directory. pytest still collects the
    # test_* functions above directly.
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
