"""Move the KV extent out of the graph capture key. Layers ON TOP of P003, never edits it.

WHY THIS IS A SEPARATE FILE

`ring_kv` is a FROZEN released pass (`released.py`: P003 v1.0.0, "change only for a correctness
bug"). Making the extent device-resident is a performance change, so it does not belong inside
it. This installs after P003 and overrides three of its seams; with the flag off, P003 behaves
byte-for-byte as v1.0.0.

WHAT IT COSTS TODAY, measured on this box before writing a line of it

    single CUDA graph capture        172.0 ms   (n=120, 162.8-220.4)
    captures                         6.0 /cycle, hit rate 92.5%, 204 evictions/episode
    capturing run vs warm run        3206.9 -> 2171.8 ms = ~1020 ms/cycle

For contrast, attention's own cost is 201 ms/cycle -- 3.6% of GPU busy, measured with
`profile_cycle.py`. The EXTENT is worth five times more than the arithmetic it describes, which
is the whole reason this file is about a capture key and not about attention.

WHAT IT DELIVERS, measured after (probe_episode, 45 cycles, ONE reset, sequential arms, one GPU)

    whole episode        3463.0 -> 2633.0 ms   1.32x   (-830.0 ms/cycle)
    late episode (36+)   3801.1 -> 2822.4 ms   1.35x   (-978.7 ms/cycle)
    captures / cycle     6.0    -> 0.1
    unique graph keys    270    -> 6
    evictions / episode  204    -> 0
    cache hit rate       92.457% -> 99.831%
    late-episode spread  0.6%   -> 0.2%

The late-episode -978.7 ms lands within 5% of the -1032 ms predicted from 6 captures x 172 ms,
which is the strongest evidence that the mechanism is the intended one rather than a coincidence.
Six keys is exactly 2 token shapes x 3 `update_cache` values, i.e. the whole key space once the
ring is gone.

TIER: NUMERIC, and it needs a certificate before it ships.

    probe_bitexact, 6 paired seeded cycles:  max|d action| = 1.184
    reference chunk-to-chunk movement     :  1.055
    ratio                                 :  112%

A ~5e-4 per-call difference compounds through the KV cache, because the actions this kernel
helps produce are themselves written back into the pool the next cycle reads. So this is not a
"small numeric difference" that can be waved through -- it is a different trajectory, and the
only honest route is `certify.py`: paired non-inferiority on pinned seeds, margin declared before
the run. Worth buying at -830 ms/cycle; the same certificate was NOT worth buying for the
attention-backend pass at -15 ms/cycle.

THE THREE PLACES `count` LEAKS INTO A CAPTURED GRAPH, and what replaces each

    read extent    kp[:, start:start+count]  a SHAPE   -> ring_attention(pool, extent_dev)
    write offset   kp[:, head:head+n] = key  a SLICE   -> index_copy_ at a device-computed index
    graph key      _iwm_ring_signature                 -> returns None in this mode

Miss any one and the key still has to contain the ring, which is why the first two are done
together rather than in sequence.

ONE EXTENT FOR ALL 30 LAYERS

Every layer's ring advances by the same `key_size` on the same forwards, so all 30 hold identical
`(start, count)`. They therefore share ONE device tensor, and the host->device mirror costs 79
small copies per cycle instead of 2,370. That is a claim about the model, so `_sync_ext` ASSERTS
it rather than assuming it -- a layer that ever disagreed would otherwise read another layer's
extent and produce a plausible wrong answer.

WRAPAROUND, and why the first version died at cycle 36

The pool holds 9792 slots and the ring advances 272 per cycle, so it fills at exactly cycle 36 --
and the first version of this file crashed there with an illegal memory access. Two causes, both
in the kernel and both now fixed and gated:

  * the read extent was `count + key_size` UNCLAMPED, so at a full pool it addressed past the
    end of the allocation. P003 writes the same clamp as `if count >= total: use the whole pool`.
  * the read walked `[start, start+count)` as one flat interval, which stops being the live set
    the moment `start` is non-zero.

Both are now handled in the kernel (`min(count, CAP)` and `pos % CAP`). "Real episodes never
wrap" was the reasoning that let the first version ship without them -- the same reasoning that
produced P003's own wraparound bug, and wrong for the same reason: RoboTwin runs 400-1700 steps,
which is 12-53 cycles.
"""

from __future__ import annotations


def install_ring_attention(server_module, va_server_cls=None) -> list[str]:
    """Install after `RingKVAddressing`. Returns the pass names applied."""
    import torch

    import modules.model as M
    from instinctwm.kernels.ring_attention import HAVE_TRITON, ring_attention

    if not HAVE_TRITON:
        raise RuntimeError("ring_attention requires triton, which this environment lacks")

    Attn = M.WanAttention
    if not hasattr(Attn, "_iwm_commit"):
        raise RuntimeError(
            "ring_attention requires the ring_kv pass (P003) to be installed FIRST: it owns the "
            "ring bookkeeping this rides on, and without it there is no interval to make "
            "device-resident.")

    _orig_forward = Attn.forward
    _orig_commit = Attn._iwm_commit
    _orig_clear_pred = Attn.clear_pred_cache

    #: cache_name -> {"dev": int32[2] cuda, "host": int32[2] pinned, "idx": {key_size: int64 cuda}}
    #: Allocated once and never reallocated: a captured graph reads `dev` at a fixed address, so
    #: replacing the tensor would be the same silent-staleness bug `stable_pools` exists to stop.
    _state: dict = {}

    def _slot(cache_name, device):
        s = _state.get(cache_name)
        if s is None:
            s = _state[cache_name] = {
                "dev": torch.zeros(2, dtype=torch.int32, device=device),
                "host": torch.zeros(2, dtype=torch.int32).pin_memory(),
                "idx": {},
            }
        return s

    def _arange(slot, key_size, total, device):
        a = slot["idx"].get(key_size)
        if a is None:
            a = slot["idx"][key_size] = torch.arange(key_size, dtype=torch.int64, device=device)
        return a

    def _sync_ext(self, cache_name):
        """Mirror the host ring into the device tensor. HOST side, after replay. Never in a graph."""
        c = (self.attn_caches or {}).get(cache_name) or {}
        r = c.get("_ring")
        s = _state.get(cache_name)
        if r is None or s is None:
            return
        start, count = int(r["start"]), int(r["count"])
        total = int(r["total"])
        # The kernel wraps with `pos % CAP` and clamps with `min(count, CAP)`, so a wrapped ring
        # is fine. What is NOT fine is a live set larger than the pool -- that would mean the
        # ring bookkeeping itself is broken, and the kernel would silently read some slots twice.
        if count > total:
            raise RuntimeError(
                f"ring_attention: live set {count} exceeds pool capacity {total}. The ring "
                f"bookkeeping is inconsistent; refusing to serve a wrong value.")
        s["host"][0] = start
        s["host"][1] = count
        s["dev"].copy_(s["host"], non_blocking=True)

    def _commit(self, cache_name, key_size, update_cache):
        _orig_commit(self, cache_name, key_size, update_cache)
        _sync_ext(self, cache_name)

    def clear_pred_cache(self, cache_name):
        _orig_clear_pred(self, cache_name)
        _sync_ext(self, cache_name)

    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
        if not getattr(type(self), "_iwm_ring_attn", False):
            return _orig_forward(self, q, k, v, rotary_emb, update_cache, cache_name)

        kvc = (self.attn_caches or {}).get(cache_name)
        r = (kvc or {}).get("_ring")
        if kvc is None or r is None or kvc.get("k") is None:
            return _orig_forward(self, q, k, v, rotary_emb, update_cache, cache_name)

        key_size = k.shape[1]
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query).unflatten(2, (self.heads, -1))
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
                return torch.view_as_real(x_out * freqs).flatten(3).to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        kp, vp = kvc["k"], kvc["v"]
        total = r["total"]
        slot = _slot(cache_name, kp.device)
        ext = slot["dev"]

        # WRITE at a DEVICE-computed offset. `head` used to be a Python int read out of the ring
        # dict, and that read is exactly what a captured graph baked in. Everything here is a
        # device op, so replay recomputes it from whatever `ext` holds at replay time.
        head = (ext[0] + ext[1]) % total
        idx = (head.to(torch.int64) + _arange(slot, key_size, total, kp.device)) % total
        kp.index_copy_(1, idx, key)
        vp.index_copy_(1, idx, value)

        # READ the live set INCLUDING the block just written -- `count + key_size`, matching
        # P003's `r["count"] + key_size`. Passed as a constexpr, not a second tensor.
        hidden_states = ring_attention(query, kp, vp, ext, count_extra=key_size)

        if not self._iwm_defer_commit:
            _commit(self, cache_name, key_size, update_cache)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](hidden_states))

    def ring_signature(self, cache_name):
        """None: the ring is no longer baked into a capture.

        This is the entire point of the file. `graph_block_stack` puts whatever this returns in
        its key; returning None makes the key `(shapes, update_cache, cache_name)`, all of which
        are constant within a phase -- so the cache converges after the first cycle instead of
        capturing 6 graphs per cycle forever.
        """
        return None

    # A reset rewinds the ring to (0, 0) via init_kv_cache. Nothing else syncs the device tensor
    # before the first forward of the new episode, so without this the first cycle of episode 2
    # would attend using episode 1's extent -- a wrong answer, not a slow one.
    _orig_init = Attn.init_kv_cache

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        _orig_init(self, cache_name, total_tolen, num_head, head_dim,
                   device, dtype, batch_size)
        _sync_ext(self, cache_name)

    Attn.init_kv_cache = init_kv_cache

    Attn._iwm_ring_attn = True
    Attn.forward = forward
    Attn._iwm_commit = _commit
    Attn.clear_pred_cache = clear_pred_cache
    Attn._iwm_ring_signature = ring_signature
    Attn._iwm_sync_ext = _sync_ext
    Attn._iwm_ext_state = _state
    return ["ring_attention"]
