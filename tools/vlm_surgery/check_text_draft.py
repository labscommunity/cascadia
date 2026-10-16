#!/usr/bin/env python3
"""Check a make_text_draft.py draft against its VLM export, the way
ov-dist-spec drives it.

Runs greedy decoding three ways on the same ids and prints where they part:

  reference  the export's text embeddings IR, then its language model IR
  draft      the composed draft IR (`openvino_model.xml`, input_ids in)
  rewound    the draft again, but after every few tokens it is fed junk ids
             that are then "rejected" the way ov-dist-spec rejects drafts:
             their attention_mask entries go to 0 and the positions are reused.
             This must match `draft`; if it does not, the IR ignores mask holes
             and every round after the first rejection drafts from a broken
             cache (acceptance ~0).

It also prints the prompt ids (does the tokenizer, called the way Cascadia
calls it, give a BOS?) and, with `--image-tokens N`, what the draft predicts
after a run of N `[IMG]` placeholder ids (ov-dist-spec used to feed image
turns' placeholders to the draft as text).

Usage (cluster):
  python tools/vlm_surgery/check_text_draft.py \
      --export <ministral3-3b-int4-ov> --draft <ministral3-3b-text-draft> \
      [--tokenizer <dir with tokenizer.json>] [--prompt "..."] [--image-tokens 784]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import openvino as ov

EMBED_XML = "openvino_text_embeddings_model.xml"
LM_XML = "openvino_language_model.xml"
DRAFT_XML = "openvino_model.xml"


class Stateful:
    """One stateful request fed like ov-dist-spec's MaskedReq: a host-side
    valid mask over the cache, positions that rewind on rejection."""

    def __init__(self, compiled: ov.CompiledModel, embed: ov.CompiledModel | None = None):
        self.req = compiled.create_infer_request()
        self.embed = embed.create_infer_request() if embed is not None else None
        self.names = {i.get_any_name() for i in compiled.inputs}
        self.mask: list[int] = []
        self.pos = 0

    def reset(self) -> None:
        self.req.reset_state()
        self.mask, self.pos = [], 0

    def feed(self, ids: list[int]) -> np.ndarray:
        n = len(ids)
        self.mask += [1] * n
        feed = {
            "attention_mask": np.array([self.mask], np.int64),
            "position_ids": np.arange(self.pos, self.pos + n, dtype=np.int64)[None],
        }
        if "beam_idx" in self.names:
            feed["beam_idx"] = np.array([0], np.int32)
        arr = np.array([ids], np.int64)
        if self.embed is not None:
            ein = self.embed.model_inputs[0]
            rows = self.embed.infer({ein: arr.astype(ein.get_element_type().to_dtype())})
            feed["inputs_embeds"] = rows[self.embed.model_outputs[0]]
        else:
            feed["input_ids"] = arr
        out = self.req.infer(feed)[self.req.model_outputs[0]]
        self.pos += n
        return out[0, -1]

    def rewind(self, k: int) -> None:
        for i in range(len(self.mask) - k, len(self.mask)):
            self.mask[i] = 0
        self.pos -= k


def greedy(m: Stateful, ids: list[int], max_new: int, junk_every: int = 0,
           junk: tuple[int, ...] = (), stop: set[int] = frozenset()) -> list[int]:
    m.reset()
    nxt = int(np.argmax(m.feed(ids)))
    out = [nxt]
    while len(out) < max_new and nxt not in stop:
        if junk_every and len(out) % junk_every == 0 and junk:
            m.feed(list(junk))
            m.rewind(len(junk))
        nxt = int(np.argmax(m.feed([nxt])))
        out.append(nxt)
    return out


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True, help="the OpenVINO VLM export the draft came from")
    ap.add_argument("--draft", required=True, help="make_text_draft.py output dir")
    ap.add_argument("--tokenizer", default=None,
                    help="dir with tokenizer.json (default: --draft); pass the target tree's "
                         "tokenizer/ to tokenize as ov-dist-spec does")
    ap.add_argument("--prompt", default="<s>[INST]Write one sentence about the sea.[/INST]")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--image-tokens", type=int, default=0,
                    help="also show the draft's continuation after N [IMG] ids")
    ap.add_argument("--image-token-id", type=int, default=10)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.tokenizer or args.draft, "tokenizer.json"))
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    with_special = tok.encode(args.prompt, add_special_tokens=True).ids
    print(f"prompt ids (as Cascadia tokenizes, no added specials): {ids[:12]}... ({len(ids)})")
    print(f"with the tokenizer's added specials:                   {with_special[:12]}... "
          f"({len(with_special)})")
    if not ids or ids[0] != 1:
        print("NOTE: the prompt does not start with BOS (1); the chat template should emit <s>")

    core = ov.Core()
    ref = Stateful(core.compile_model(os.path.join(args.export, LM_XML), args.device),
                   core.compile_model(os.path.join(args.export, EMBED_XML), args.device))
    draft = Stateful(core.compile_model(os.path.join(args.draft, DRAFT_XML), args.device))
    stop = {2}
    r = greedy(ref, ids, args.max_new, stop=stop)
    d = greedy(draft, ids, args.max_new, stop=stop)
    w = greedy(draft, ids, args.max_new, junk_every=3, junk=(1000, 2000, 3000), stop=stop)
    for name, seq in (("reference", r), ("draft", d), ("rewound", w)):
        print(f"{name:9} {seq}\n          {tok.decode(seq)!r}")
    ok = True
    for name, a, b in (("draft vs reference", d, r), ("rewound vs draft", w, d)):
        i = first_diff(a, b)
        print(f"{name}: " + ("MATCH" if i is None else f"DIFFER at token {i}"))
        ok &= i is None
    if args.image_tokens:
        img = ids[:-1] + [args.image_token_id] * args.image_tokens + ids[-1:]
        p = greedy(draft, img, args.max_new, stop=stop)
        print(f"after {args.image_tokens} [IMG] ids: {tok.decode(p)!r}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
