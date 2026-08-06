"""Backend Adapter — what a world-action model DECLARES about itself.

InstinctWM is an optimization framework, not just a runtime. The layering is

    Backend Adapter  ->  Optimizer/Compiler  ->  Runtime

and the load-bearing idea is that the adapter states *facts*, never *optimizations*. A model
author writes "the action stream uses positive-only guidance" and "the text conditioning is a
pure function of the instruction". They do not write "skip the negative branch" or "cache the
cross-attention K/V" — the optimizer derives those. That is the difference between a runtime
you configure and a framework that makes your model fast because it understands it.

The declarations below were chosen by diffing the per-control-step execution graphs of six
model families (LingBot-VA, DreamZero, Cosmos3-Edge, InternVLA-A1, GR00T, pi-0/pi-0.5); see
`docs/EXECUTION_GRAPH.md`. Two findings shaped them:

  * KV persistence is a LIFETIME, not a boolean. pi-0 builds a prefix cache, commits it, reads
    it from all 10 denoise forwards and drops it — structurally identical to LingBot-VA's
    episode-scoped stream, differing only in how long it lives. A boolean `is_stateful` would
    have excluded Cosmos3-Edge (chunk-scoped) and every VLA.
  * The clean seam is `commit_context`: everything above it is prefill, everything below is
    decode. Five of six models already draw that line in their own code without naming it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

# `kernels.regions` is pure dataclasses — no torch, no registry — so importing it here keeps
# `AdapterSpec` readable on a laptop. The kernel *implementations* stay behind the pass.
from instinctwm.kernels.regions import FusionDescriptor


class KVLifetime(enum.Enum):
    """How long a committed KV stream survives.

    The single most important declaration: it is what lets one optimizer serve a stateless VLA
    and an episode-scoped WAM without an `if is_vla` branch anywhere.
    """

    NONE = "none"        # GR00T: no KV persists past a forward
    CHUNK = "chunk"      # pi-0 prefix, Cosmos3-Edge text K/V, InternVLA-A1
    WINDOW = "window"    # DreamZero: N frames, hard reset at the boundary
    EPISODE = "episode"  # LingBot-VA: both streams, for the whole episode


class CommitMode(enum.Enum):
    SCRATCH = "scratch"          # written then rolled back within a step
    PROVISIONAL = "provisional"  # survives the step, invalidated at the next commit
    CONFIRMED = "confirmed"      # permanent for the declared lifetime


class GuidanceMode(enum.Enum):
    NONE = "none"                    # GR00T, InternVLA-A1, pi-0: no negative branch at all
    CFG = "cfg"                      # both branches computed AND combined
    POSITIVE_ONLY = "positive_only"  # a negative branch exists in the batch but is DISCARDED


@dataclass(frozen=True)
class KVStreamSpec:
    """One named, independently committed KV stream.

    Generalizes vLLM-Omni's `ARDiffusionKVCacheSpec` — which is the right dataclass — along the
    two axes it lacks. Theirs has a single `tokens_per_frame` (also the paged block size) and
    demotes everything else to `max_scratch_tokens_per_branch`, documented as "non-video KV (for
    example, action/state registers) that must coexist with an uncommitted video block". That
    models one video stream with registers bolted on. LingBot-VA commits action K/V permanently
    (`update_cache=2` on both streams) and attends it in every later cycle, so it needs two
    CO-EQUAL streams with different token densities.
    """

    name: str
    tokens_per_frame: int
    lifetime: KVLifetime
    commit_mode: CommitMode = CommitMode.CONFIRMED
    window_frames: int | None = None
    sink_frames: int = 0
    supports_provisional: bool = False


@dataclass(frozen=True)
class GuidanceRule:
    """Per-output-stream guidance. Declarative, because the three models that use guidance all
    implement it differently for reasons that fit in these fields."""

    mode: GuidanceMode
    scale: float = 1.0
    # True when both branches can share one forward at batch 2 (LingBot-VA); False when the
    # model cannot batch-duplicate and must run branches as separate forwards (Cosmos3-Edge).
    batchable: bool = True


@dataclass(frozen=True)
class PhaseSpec:
    """A run of forwards with homogeneous shape — the unit the optimizer schedules.

    `nfe` is mutable per control step by design: that is the whole point of declaring the loop
    instead of hiding it inside `forward()`.
    """

    name: str
    nfe: int
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    #: Indices within the phase whose forwards COMMIT KV. A set, not a single index: LingBot-VA's
    #: kv_refresh phase commits on BOTH of its forwards (one per stream, each update_cache=2),
    #: and a single-index field silently marked one of them elidable — which would corrupt the
    #: episode several chunks later. Empty = this phase never commits.
    commit_steps: frozenset[int] = frozenset()
    truncatable: bool = False
    min_nfe: int = 1
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class PurityKey:
    """A model ASSERTION that some conditioning artifact is a pure function of `fields`.

    This is the highest-value and riskiest declaration in the interface: it is what licenses the
    optimizer to hoist work from per-forward to per-episode. It is an assertion, so it needs a
    verifier rather than trust — see `instinctwm/verify/`.
    """

    artifact: str
    fields: tuple[str, ...]
    scope: KVLifetime


@dataclass(frozen=True)
class AdapterSpec:
    """Everything the optimizer reads. Facts only — no optimizations."""

    model_id: str
    param_bytes: int
    streams: tuple[KVStreamSpec, ...]
    phases: tuple[PhaseSpec, ...]
    guidance: Mapping[str, GuidanceRule]
    purity: tuple[PurityKey, ...] = ()
    #: modules whose output is only needed when the caller asks for predicted pixels. All four
    #: WAMs surveyed agree the observation-decode tail is optional at serving time — Cosmos3-Edge
    #: denoises 550 of 567 tokens as future video and discards them.
    obs_decode_modules: tuple[str, ...] = ()
    #: op sequences the adapter offers for fusion, read out of the model's source. A DECLARATION
    #: of what the eager path does — which ops, in which order, materialising in which dtype —
    #: and never a request to fuse. `OperatorFusion` decides whether any registered kernel is
    #: legal for the declared structure, and `derive_tier` decides what claim it earns.
    #:
    #: Optional because it is the one declaration a model author cannot make from the config
    #: alone: it needs the block source. An adapter that omits it simply gets no L5 pass.
    fusion: FusionDescriptor | None = None
    notes: Mapping[str, str] = field(default_factory=dict)

    def phase(self, name: str) -> PhaseSpec:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(name)

    def total_forwards(self) -> int:
        """Every transformer forward in one control step, across all phases.

        Note this is NOT the same number as the "77 forwards" quoted throughout the LingBot-VA
        write-ups: that figure is the *denoise* loop alone (26 video + 51 action) and excludes
        the 2 kv_refresh forwards. Both counts are correct for different questions, so anything
        user-facing should say which one it means — `forwards_breakdown()` exists so a pass can
        show its work instead of quoting a bare total that disagrees with the docs.
        """
        return sum(p.nfe for p in self.phases)

    def forwards_breakdown(self) -> str:
        """`total_forwards()` with its per-phase terms, e.g. `kv_refresh=2 + video=26 + action=51`."""
        return " + ".join(f"{p.name}={p.nfe}" for p in self.phases)


class BackendAdapter(Protocol):
    """The contract a world-action model implements to be optimized by InstinctWM."""

    def spec(self) -> AdapterSpec:
        """Immutable declarations, read once at load."""
        ...

    def install(self, server_module: object, plan: "object") -> Sequence[str]:
        """Apply an optimization plan to a concrete serving object.

        Today this patches the upstream server at runtime rather than replacing it. That is
        deliberate and temporary: it keeps every pass verifiable against the existing
        bit-exactness gate before anything is rewritten, and it keeps the vendored upstream
        tree clean so `git diff` stays reviewable.

        Returns what it actually applied, and raises on any applied pass it cannot install.
        Reporting a pass as installed when it was skipped would invalidate every number
        measured against the resulting server.
        """
        ...

    def serve(self, plan: "object", port: int, **kwargs) -> object:
        """Import this model's server, install `plan`, and start serving on `port`."""
        ...
