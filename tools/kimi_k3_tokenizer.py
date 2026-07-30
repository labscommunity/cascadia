#!/usr/bin/env python3
"""Convert K3's tiktoken BPE into a HF `tokenizer.json`, and prove it matches.

K3 ships `tiktoken.model` plus a custom `TikTokenTokenizer`, not a
`tokenizer.json`. Every engine in this crate loads `tokenizer.json` via the HF
`tokenizers` crate, so the API rank cannot start without one.

The pre-tokenizer pattern uses Java/ICU character-class intersection
(`[\\p{Lu}...&&[^\\p{Han}]]` — "letters and marks except Han"). Oniguruma, which
`tokenizers` is built with here, accepts it verbatim, so the pattern is carried
across unchanged rather than rewritten. `tests/k3_tokenizer_pattern.rs` pins
that: it splits `hello世界world` into `["hello", "世界", "world"]` in the engine's
own regex flavour.

A tokenizer that is subtly wrong looks like a model quality problem, so
`--validate` is not optional in spirit: it encodes a multilingual corpus through
both the reference tiktoken encoding and the generated `tokenizer.json` and
requires the id streams to be identical.

    python tools/kimi_k3_tokenizer.py --model-dir <ckpt> --out <export>/tokenizer.json
    python tools/kimi_k3_tokenizer.py --model-dir <ckpt> --out /tmp/t.json --validate
"""
import argparse
import base64
import json
from pathlib import Path

# Special tokens, from `tokenization_kimi.py`. 163584 is the first special id;
# the block is 256 reserved slots wide.
FIRST_SPECIAL = 163584
NUM_RESERVED = 256
NAMED_SPECIALS = {
    "[BOS]": 163584,
    "<|im_end|>": 163585,
    "[EOS]": 163586,
    "<|im_user|>": 163587,
    "<|im_assistant|>": 163588,
    "<|start_header_id|>": 163590,
    "<|end_header_id|>": 163591,
    "<|im_system|>": 163594,
    "<|im_middle|>": 163601,
    "<|kimi_image_placeholder|>": 163605,
}

# Verbatim `TikTokenTokenizer.pat_str`.
PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


def byte_to_unicode() -> dict:
    """GPT-2's reversible byte -> printable-codepoint map, as HF ByteLevel uses."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def read_ranks(model_dir: Path) -> dict:
    """`tiktoken.model` is `base64(token_bytes) rank` per line."""
    p = model_dir / "tiktoken.model"
    if not p.exists():
        raise SystemExit(f"[k3-tok] missing {p}")
    ranks = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        tok, rank = line.split()
        ranks[base64.b64decode(tok)] = int(rank)
    return ranks


def _bpe_upto(ranks: dict, token: bytes, ceiling: int) -> list:
    """BPE-encode `token`'s bytes using only merges ranked below `ceiling`."""
    parts = [bytes([b]) for b in token]
    while len(parts) > 1:
        best_i, best_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < ceiling and (best_rank is None or r < best_rank):
                best_i, best_rank = i, r
        if best_i is None:
            break
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return parts


def derive_merges(ranks: dict) -> list:
    """Recover the ordered BPE merge list from tiktoken's rank table.

    For each token, re-run BPE over its bytes using only merges ranked below it;
    the two surviving parts are the pair that produced it.

    Do not instead pick the split minimising `max(rank(l), rank(r))`: several
    splits can have both halves in the vocab, so it emits a merge that never
    fires while the real one goes missing and the token becomes unreachable.
    """
    merges = []
    unreachable = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) < 2:
            continue
        parts = _bpe_upto(ranks, token, rank)
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
        else:
            unreachable.append((token, len(parts)))
    if unreachable:
        print(f"[k3-tok] WARNING {len(unreachable)} tokens did not reduce to a "
              f"single pair, e.g. {unreachable[:3]}")
    return merges


def build_tokenizer_json(ranks: dict) -> dict:
    b2u = byte_to_unicode()

    def enc(bs: bytes) -> str:
        return "".join(b2u[b] for b in bs)

    vocab = {enc(t): r for t, r in ranks.items()}
    merges = [f"{enc(a)} {enc(b)}" for a, b in derive_merges(ranks)]

    added = []
    for name, tid in sorted(NAMED_SPECIALS.items(), key=lambda kv: kv[1]):
        added.append(
            {
                "id": tid,
                "content": name,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        vocab[name] = tid
    # the rest of the reserved block, so ids stay dense and decode round-trips
    for i in range(FIRST_SPECIAL, FIRST_SPECIAL + NUM_RESERVED):
        if i in NAMED_SPECIALS.values():
            continue
        name = f"<|reserved_token_{i}|>"
        added.append(
            {
                "id": i,
                "content": name,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        vocab[name] = i

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        # Split on the tiktoken pattern, then byte-encode each piece.
        # `use_regex: false` matters: ByteLevel's own GPT-2 regex would re-split
        # the pieces and change the tokenisation.
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": PAT_STR},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


SAMPLES = [
    "The capital of France is Paris.",
    "hello世界world 123",
    "def f(x):\n    return x ** 2  # comment\n",
    "  leading and   multiple   spaces\t\ttabs\n\nblank lines\n",
    "Mixed 漢字 and かな and 한글 and Ελληνικά and кириллица",
    "punctuation!!! ...???  (parens) [brackets] {braces} <tags/>",
    "numbers 1 12 123 1234 12345 3.14159 -42 1e10",
    "emoji 🙂🚀🇯🇵 and ZWJ 👨‍👩‍👧‍👦",
    "it's don't we've I'll they'd he's",
    "URL https://example.com/a?b=c&d=e#f",
    "",
    " ",
    "\n",
    "a" * 200,
]


def validate(ranks: dict, out: Path) -> int:
    """Encode the corpus both ways and require identical ids."""
    import tiktoken
    from tokenizers import Tokenizer

    ref = tiktoken.Encoding(
        name="kimi_k3",
        pat_str=PAT_STR,
        mergeable_ranks=ranks,
        special_tokens={},
    )
    got = Tokenizer.from_file(str(out))

    bad = 0
    for s in SAMPLES:
        a = ref.encode(s, allowed_special=set(), disallowed_special=set())
        b = got.encode(s, add_special_tokens=False).ids
        if a != b:
            bad += 1
            if bad <= 4:
                print(f"[validate] MISMATCH {s!r}")
                print(f"    tiktoken : {a}")
                print(f"    tokenizer: {b}")
                # first divergence, to make the cause obvious
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        print(
                            f"    first diff at {i}: {ref.decode([x])!r} vs "
                            f"{got.decode([y])!r}"
                        )
                        break
    if bad:
        print(f"[validate] FAILED — {bad}/{len(SAMPLES)} samples differ")
        return 1
    print(f"[validate] OK — {len(SAMPLES)} samples encode identically")

    # decode round-trip, so the ByteLevel decoder is exercised too
    for s in SAMPLES:
        ids = got.encode(s, add_special_tokens=False).ids
        if got.decode(ids) != s:
            print(f"[validate] FAILED decode round-trip: {s!r} -> {got.decode(ids)!r}")
            return 1
    print("[validate] OK — decode round-trips")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="checkpoint dir holding tiktoken.model")
    ap.add_argument("--out", type=Path, required=True, help="tokenizer.json to write")
    ap.add_argument("--validate", action="store_true",
                    help="require the output to match reference tiktoken exactly")
    a = ap.parse_args()

    ranks = read_ranks(a.model_dir)
    tj = build_tokenizer_json(ranks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(tj, ensure_ascii=False))
    print(f"[k3-tok] {len(ranks):,} ranks, {len(tj['model']['merges']):,} merges, "
          f"{len(tj['added_tokens'])} special -> {a.out}")

    return validate(ranks, a.out) if a.validate else 0


if __name__ == "__main__":
    raise SystemExit(main())
