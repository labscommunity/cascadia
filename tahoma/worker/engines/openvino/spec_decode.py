"""Speculative decoding with mask-based KV cache rewind.

Ported from rainier (`cascadia/pipeline/spec_decode.py`, DISCOVERY #20):

  - `MaskedReq`: wraps an OpenVINO `InferRequest` for a stateful causal LM
    and adds a `rewind(k)` op that invalidates the last `k` tokens of the
    KV cache by zeroing `attention_mask[...]` on future feeds, instead of
    physically trimming via `query_state()` + `set_state()`. The physical
    trim costs ~40 ms per call on Intel iGPU; mask-based is ~free.
  - `spec_decode_greedy()`: a self-contained greedy spec decode loop.

Inputs to both target and draft must follow the OpenVINO causal-LM
convention with named inputs `input_ids`, `attention_mask`, `position_ids`,
plus optional `beam_idx`. IRs exported via `optimum-cli export openvino`
or rainier's `export_cached_shards_v5.py` work directly.

Per rainier benchmarks: 1.36x over monolithic target (LAN, K=3, 128 tok),
1.55x at 512 tok with 90%+ accept rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _feed_raw(
    req: Any,
    has_beam: bool,
    input_ids: np.ndarray,
    attn_mask: np.ndarray,
    pos_ids: np.ndarray,
) -> np.ndarray:
    feed = {"input_ids": input_ids, "attention_mask": attn_mask, "position_ids": pos_ids}
    if has_beam:
        feed["beam_idx"] = np.zeros(1, dtype=np.int32)
    req.infer(feed)
    return req.get_output_tensor(0).data


class MaskedReq:
    """OpenVINO stateful causal-LM `InferRequest` wrapper with mask-based rewind.

    Tracks `cache_len` (physical KV length, monotonic) and `logical_pos`
    (logical sequence length after rewinds). `valid_mask[i]` is 1 if cache
    position `i` is live, 0 if it was rewound.

    `rewind(k)` zeros the last `k` valid positions and decreases `logical_pos`.
    Subsequent `feed()` builds attention_mask from `valid_mask`.
    """

    __slots__ = ("req", "has_beam", "valid_mask", "cache_len", "logical_pos")

    def __init__(self, req: Any, has_beam: bool, initial_capacity: int = 4096):
        self.req = req
        self.has_beam = has_beam
        self.valid_mask = np.ones(initial_capacity, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self) -> None:
        self.req.reset_state()
        self.valid_mask[:] = 1
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids: np.ndarray) -> np.ndarray:
        """Feed `input_ids` (shape [1, n]). Returns logits (1, n, vocab)."""
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new_size = max(total * 2, len(self.valid_mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[: len(self.valid_mask)] = self.valid_mask
            self.valid_mask = new_mask

        att = np.empty((1, total), dtype=np.int64)
        att[0, : self.cache_len] = self.valid_mask[: self.cache_len]
        att[0, self.cache_len :] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)

        out = _feed_raw(self.req, self.has_beam, input_ids, att, pos)
        self.cache_len += n
        self.logical_pos += n
        return out

    def rewind(self, k: int) -> None:
        """Invalidate the last `k` cache positions logically."""
        if k <= 0:
            return
        self.valid_mask[self.cache_len - k : self.cache_len] = 0
        self.logical_pos -= k


@dataclass
class SpecDecodeStats:
    n_steps: int = 0
    total_drafts: int = 0
    total_accepted: int = 0

    @property
    def accept_rate(self) -> float:
        return self.total_accepted / max(self.total_drafts, 1)


def spec_decode_greedy(
    target: MaskedReq,
    draft: MaskedReq,
    prompt_ids: np.ndarray,
    max_tokens: int,
    k: int = 3,
) -> tuple[list[int], SpecDecodeStats]:
    """Greedy speculative decoding.

    Bit-exact with target's own greedy decode given the same prompt.
    Returns generated token ids + per-run telemetry.
    """
    target.reset()
    draft.reset()

    t_logits = target.feed(prompt_ids)
    draft.feed(prompt_ids)

    first = int(np.argmax(t_logits[0, -1, :]))
    gens: list[int] = [first]
    prev_correction = first

    d_logits = draft.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_logits[0, -1, :].copy()

    stats = SpecDecodeStats()

    while len(gens) < max_tokens:
        stats.n_steps += 1

        # 1. Draft K candidates.
        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens:
                break
            d_logits = draft.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_logits[0, -1, :])))
        d_advanced = len(drafts) - 1
        stats.total_drafts += len(drafts)

        # 2. Target verifies [prev_correction, drafts] in one forward.
        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t_logits = target.feed(verify)
        t_greedy = np.argmax(t_logits[0], axis=-1)  # [K+1]

        # 3. Acceptance: longest matching prefix.
        accepted = 0
        for i in range(len(drafts)):
            if int(t_greedy[i]) == drafts[i]:
                accepted += 1
            else:
                break
        stats.total_accepted += accepted

        # 4. Correction.
        if accepted < len(drafts):
            correction = int(t_greedy[accepted])
        else:
            correction = int(t_greedy[len(drafts)])

        for tk in drafts[:accepted] + [correction]:
            if len(gens) >= max_tokens:
                break
            gens.append(tk)

        # 5. Rewind target by the rejected drafts.
        target.rewind(len(drafts) - accepted)

        # 5b. Draft rewind / catch-up.
        if accepted < len(drafts):
            draft.rewind(d_advanced - accepted)
            d_logits = draft.feed(np.array([[correction]], dtype=np.int64))
        else:
            d_logits = draft.feed(np.array([[drafts[-1]]], dtype=np.int64))
            d_logits = draft.feed(np.array([[correction]], dtype=np.int64))
        d_last_logit = d_logits[0, -1, :].copy()
        prev_correction = correction

    return gens[:max_tokens], stats


def make_masked_req(compiled_model: Any) -> MaskedReq:
    """Wrap a compiled model in MaskedReq, auto-detecting the beam_idx input."""
    req = compiled_model.create_infer_request()
    has_beam = any(
        any("beam_idx" in n for n in inp.get_names()) for inp in compiled_model.inputs
    )
    return MaskedReq(req, has_beam)
