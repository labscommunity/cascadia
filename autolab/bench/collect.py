#!/usr/bin/env python3
"""Collect a corpus of the fleet's own greedy outputs for offline drafter studies.

    collect.py OUT.jsonl [--prompts 192] [--streams 48] [--tokens 160] [--seed TAG]

Single-stream speed on this fleet is 1 / (a*T + (1-a)*L): stage time T, one token's trip through all ranks L,
and a = how often the drafter names the model's next token. What a drafter is worth can be measured without
the fleet, by replaying it over text the model really wrote. This writes that text: one JSON line per request
with the prompt and the output (temperature 0). The prompts are wider than lab.py's `fresh` family on purpose
(explanations, code, arithmetic, stories, lists, advice, comparisons, rewriting).
"""
import argparse, concurrent.futures, hashlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402

TOPICS = ["tides", "a bicycle", "sourdough bread", "the stock market", "volcanoes", "a violin", "vaccines", "black holes",
          "the Roman empire", "coffee", "honey bees", "the internet", "glaciers", "a jet engine", "chess openings", "jazz",
          "the immune system", "rainbows", "compound interest", "the printing press", "coral reefs", "a lighthouse",
          "electric cars", "the moon", "sleep", "rice farming", "earthquakes", "a submarine", "origami", "the Silk Road"]
CODE = ["reverses a linked list", "checks whether a string is a palindrome", "computes the nth Fibonacci number iteratively",
        "merges two sorted lists", "counts word frequencies in a text file", "finds the largest element of a binary tree",
        "parses a CSV line with quoted fields", "implements binary search", "removes duplicates from a list while keeping order",
        "validates an email address with a regular expression", "computes the greatest common divisor of two integers",
        "flattens a nested list"]
LANGS = ["Python", "JavaScript", "Rust", "Go", "C"]
FAMILIES = [
    lambda h: "Explain %s to a curious ten-year-old in one short paragraph." % TOPICS[h % len(TOPICS)],
    lambda h: "Write a %s function that %s. Include a short comment." % (LANGS[h % len(LANGS)], CODE[(h >> 8) % len(CODE)]),
    lambda h: "A shop sells pencils at %d cents each and notebooks at %d cents each. How much do %d pencils and %d notebooks cost? Show the steps." % (
        15 + h % 40, 120 + (h >> 8) % 200, 3 + (h >> 16) % 9, 2 + (h >> 20) % 5),
    lambda h: "Write the opening paragraph of a short story that takes place near %s." % TOPICS[h % len(TOPICS)],
    lambda h: "List five practical tips about %s, one line each." % TOPICS[h % len(TOPICS)],
    lambda h: "Compare %s and %s in a short table, then give a one-sentence conclusion." % (
        TOPICS[h % len(TOPICS)], TOPICS[(h >> 8) % len(TOPICS)]),
    lambda h: "Rewrite this sentence to sound more formal: 'So basically %s is kind of a big deal and people should really care more about it.'" % TOPICS[h % len(TOPICS)],
    lambda h: "What are the three most important facts about %s? Answer briefly." % TOPICS[h % len(TOPICS)],
    lambda h: "Write a four-line poem about %s." % TOPICS[h % len(TOPICS)],
    lambda h: "Give step-by-step instructions for a beginner who wants to learn about %s this weekend." % TOPICS[h % len(TOPICS)],
    lambda h: "Translate into French and then into Spanish: 'The history of %s is longer than most people think.'" % TOPICS[h % len(TOPICS)],
    lambda h: "Is it true that %s? Answer yes or no, then justify in two sentences." % [
        "lightning never strikes the same place twice", "humans use only ten percent of their brains",
        "the Great Wall of China is visible from space", "glass is a slow-moving liquid", "bats are blind",
        "goldfish have a three-second memory", "sugar makes children hyperactive", "water conducts electricity"][h % 8],
]


def prompt_for(seed, i):
    h = int(hashlib.sha256(("%s/%d" % (seed, i)).encode()).hexdigest(), 16)
    return FAMILIES[i % len(FAMILIES)](h >> 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--prompts", type=int, default=192); ap.add_argument("--streams", type=int, default=48)
    ap.add_argument("--tokens", type=int, default=160); ap.add_argument("--seed", default="corpus-a")
    ap.add_argument("--cap", type=int, default=900)
    a = ap.parse_args()
    prompts = [prompt_for(a.seed, i) for i in range(a.prompts)]
    done = 0; t0 = time.time()
    with open(a.out, "a") as f:
        for lo in range(0, len(prompts), a.streams):
            batch = prompts[lo:lo + a.streams]

            def one(j):
                time.sleep(0.015 * j)
                return lab.chat(j, a.tokens, 0, a.cap, batch[j])

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
                res = list(ex.map(one, range(len(batch))))
            for j, r in enumerate(res):
                if "error" in r or not r.get("text"):
                    lab.log("request %d failed: %s" % (lo + j, r.get("error")))
                    continue
                f.write(json.dumps(dict(seed=a.seed, i=lo + j, prompt=batch[j], tokens=r["tokens"], text=r["text"])) + "\n")
                done += 1
            f.flush()
            lab.log("collected %d/%d responses, %.0f s" % (done, len(prompts), time.time() - t0))


if __name__ == "__main__":
    main()
