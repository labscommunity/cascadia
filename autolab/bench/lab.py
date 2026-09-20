#!/usr/bin/env python3
"""Autolab harness for the 11-box Inkling fleet. Runs on the operator's Mac; standard library only.

The fleet has no shell. Everything goes through two doors:
  in:   ~/inkling-release/bin/release.py publish ...   (signed release channel -> rank 0 -> all boxes)
  out:  http://localhost:18000  = rank 0 :8000 (OpenAI API, /api/stats, /api/fleet/telemetry)
        release.py status --json  (signed: fleet table, rank 0 worker log tail)

    lab.py status                                  one-line fleet state
    lab.py publish  --note WHY [cascadia=PATH] [fleet-overrides.env=PATH] [beacon.py=PATH run.sh=PATH --allow-infra]
    lab.py settle   [--expect name=sha ...]        wait until 11 workers serve one files version, restarts steady
    lab.py reference                               record greedy reference texts from the fleet as it is NOW
    lab.py gate                                    greedy outputs vs the reference (exit 1 = output broken)
    lab.py bench EXP --phases single:1:48 s16:16:48 [--prompt-words N]   timed phases + telemetry -> experiments/EXP/
    lab.py run EXP --note WHY [files...] --phases ...   publish + settle + warm + gate + bench

Hard rule: no phase waits longer than --cap seconds (default 900).
"""
import argparse, concurrent.futures, hashlib, json, os, re, subprocess, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
API = os.environ.get("INKLING_API", "http://localhost:18000")
RELEASE = [sys.executable if sys.version_info >= (3, 9) else "/usr/bin/python3", os.path.expanduser("~/inkling-release/bin/release.py")]
RELEASE[0] = "/usr/bin/python3"
LOCK = os.path.expanduser("~/inkling-release/publisher.lock/owner")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPTS = ["Explain in three sentences why the sky is blue.", "What is the capital of France? Answer in one word.",
           "Write two sentences about the Pacific Ocean.", "List three prime numbers and say why they are prime.",
           "Describe a cat in two sentences.", "What is 6 times 7? Explain briefly.", "Name two planets and one fact about each.",
           "Give one tip for sleeping better, in two sentences.", "Why is the ocean salty? Two sentences.",
           "Summarise photosynthesis in two sentences.", "What does a compiler do? Two sentences.", "Describe rain in one sentence.",
           "What is a haiku? Give one.", "Explain gravity to a child in two sentences.", "Name a famous painting and its painter.",
           "What is the boiling point of water? Explain."]
FILLER = ("The harbour town woke slowly that morning: gulls over the quay, a baker carrying trays, two "
          "fishermen arguing about the tide, and a child counting the boats as they left one by one. ")
GATE_PROMPTS = [0, 1, 5]
GATE_TOKENS = 32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_json(path, timeout=15):
    with OPENER.open(API + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def release(*args, timeout=120):
    p = subprocess.run(RELEASE + list(args), capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def status():
    rc, out, err = release("status", "--json")
    if rc != 0 or not out.strip():
        raise RuntimeError("release.py status failed: %s" % (err or out)[:300])
    return json.loads(out)


def fleet_rows(st):
    rows = {}
    for line in (st.get("beacon") or "").splitlines():
        m = re.match(r"\s+rank\s+(\d+)\s+(\S+)\s+(\S+)\s+files (\S+ \S+)\s+\| worker (\w+), (\d+) restarts: (.*)", line)
        if m:
            rows[int(m.group(1))] = dict(ip=m.group(2), host=m.group(3), files=m.group(4), state=m.group(5),
                                         restarts=int(m.group(6)), phase=m.group(7).strip())
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def cmd_status(a):
    st = status()
    rows = fleet_rows(st)
    age = time.time() - st.get("time", 0)
    print("report %.0f s old; release %s; files version %s; %d/11 ranks active; restarts %s" % (
        age, st.get("poller", {}).get("applied"), st.get("fleet", {}).get("version"),
        sum(1 for r in rows.values() if r["state"] == "active"), " ".join(str(rows[k]["restarts"]) for k in sorted(rows))))
    bad = {k: v["phase"] for k, v in rows.items() if v["phase"] != "serving"}
    if bad:
        print("not serving:", bad)
    if a.log:
        for l in st.get("worker_log", []):
            print("   ", re.sub(r"\x1b\[[0-9;]*m", "", l)[:260])
    return 0


def settle(expect, cap=900, steady=3):
    """Wait until the fleet runs the expected file hashes, 11 workers serve, and restart counts stop moving."""
    t0 = time.time(); last = None; same = 0; seen_time = 0
    while time.time() - t0 < cap:
        try:
            st = status()
        except Exception as e:  # noqa: BLE001
            log("status:", str(e)[:120]); time.sleep(10); continue
        if st.get("time", 0) == seen_time:
            time.sleep(5); continue
        seen_time = st.get("time", 0)
        files = st.get("fleet", {}).get("files", {})
        rows = fleet_rows(st)
        ok_files = all(files.get(n) == h for n, h in expect.items())
        versions = {r["files"] for r in rows.values()}
        active = sum(1 for r in rows.values() if r["state"] == "active" and r["phase"].startswith("serving"))
        restarts = tuple(rows[k]["restarts"] for k in sorted(rows))
        same = same + 1 if (restarts == last and ok_files and active == 11 and len(versions) == 1) else 0
        last = restarts
        log("settle: files %s, %d/11 serving, %d version(s), restarts %s, steady %d/%d" % (
            "ok" if ok_files else "PENDING", active, len(versions), " ".join(map(str, restarts)), same, steady))
        if same >= steady:
            return True
        time.sleep(10)
    return False


def cmd_settle(a):
    expect = dict(x.split("=", 1) for x in a.expect)
    return 0 if settle(expect, a.cap) else 1


def cmd_publish(a):
    owner = open(LOCK).read() if os.path.exists(LOCK) else ""
    if "tahoma-6d" not in owner:
        sys.exit("publisher lock is not ours (%s): not publishing" % LOCK)
    args, expect = ["publish"], {}
    for spec in a.files:
        name, path = spec.split("=", 1)
        expect[name] = sha256(path)
        args.append("%s=%s" % (name, os.path.abspath(path)))
    if a.allow_infra:
        args.append("--allow-infra")
    args += ["--note", a.note]
    rc, out, err = release(*args, timeout=300)
    log("publish:", (out + err).strip().replace("\n", " | ")[:400])
    if rc != 0:
        return 2
    json.dump(expect, open(os.path.join(LAB, ".autolab", "expect.json"), "w"))
    return 0 if settle(expect, a.cap) else 1


def chat(i, tokens, prompt_words=0, timeout=900, prompt=None):
    prompt = prompt or PROMPTS[i % len(PROMPTS)]
    if prompt_words:
        words = (FILLER * (prompt_words // len(FILLER.split()) + 1)).split()[:prompt_words]
        prompt = "Read this, then answer the question after it.\n\n" + " ".join(words) + "\n\n" + prompt
    body = json.dumps({"model": "inkling", "stream": True, "max_tokens": tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; n = 0; text = []; deadline = t0 + timeout
    try:
        r = None
        for attempt in range(40):
            try:
                r = OPENER.open(req, timeout=timeout); break
            except urllib.error.HTTPError as e:
                if e.code != 503 or attempt == 39:
                    raise
                time.sleep(0.5 + 0.25 * attempt)
        with r:
            for line in r:
                if time.time() > deadline:
                    raise TimeoutError("phase cap")
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    v = json.loads(data)
                except ValueError:
                    continue
                if "error" in v and not v.get("choices"):
                    raise RuntimeError("server error: %s" % str(v["error"])[:160])
                ch = v.get("choices", [{}])[0]; d = ch.get("delta") or {}
                if ch.get("finish_reason") is None and d != {}:
                    now = time.time(); first = first or now; last = now; n += 1
                    text.append(d.get("content") or d.get("reasoning_content") or d.get("reasoning") or "")
        return dict(i=i, wall_s=round(time.time() - t0, 2), ttft_s=round((first or t0) - t0, 2), tokens=n,
                    tok_s=round((n - 1) / (last - first), 3) if n > 1 and last > first else None, text="".join(text))
    except Exception as e:  # noqa: BLE001
        return dict(i=i, wall_s=round(time.time() - t0, 2), error=str(e)[:200], tokens=n, text="".join(text))


def cmd_reference(a):
    ref = {}
    for i in GATE_PROMPTS:
        r = chat(i, GATE_TOKENS)
        if "error" in r:
            sys.exit("reference request failed: %s" % r["error"])
        ref[str(i)] = r["text"]; log("ref %d: %r" % (i, r["text"][:90]))
    json.dump(ref, open(os.path.join(HERE, "reference.json"), "w"), indent=1)
    return 0


def gate():
    """Greedy outputs vs the recorded reference. Returns (ok, details). Garbage has fine tok/s, so this runs before every timing."""
    ref = json.load(open(os.path.join(HERE, "reference.json")))
    res = []
    for i in GATE_PROMPTS:
        r = chat(i, GATE_TOKENS, timeout=600)
        got, want = r.get("text", ""), ref[str(i)]
        common = 0
        for x, y in zip(got, want):
            if x != y:
                break
            common += 1
        bad = ("error" in r) or not got.strip() or "!!!!" in got or len(set(got)) < 6
        res.append(dict(i=i, match_chars=common, of=len(want), exact=got == want, broken=bad, tok_s=r.get("tok_s"),
                        ttft_s=r.get("ttft_s"), text=got[:120], error=r.get("error")))
    ok = all(not x["broken"] for x in res) and sum(1 for x in res if x["match_chars"] >= min(40, x["of"])) >= 2
    return ok, res


def gate_multi(n=8):
    """The same check with the reference prompts decoding side by side with others: multi-row kernels (expert GEMM,
    batched dense MLP, batched admission) only run when a frame carries several rows, which a lone request never does."""
    ref = json.load(open(os.path.join(HERE, "reference.json")))
    order = [GATE_PROMPTS[i % len(GATE_PROMPTS)] if i % 2 == 0 else 6 + i for i in range(n)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        outs = list(ex.map(lambda i: chat(i, GATE_TOKENS, timeout=900), order))
    res = []
    for i, r in zip(order, outs):
        got = r.get("text", "")
        bad = ("error" in r) or not got.strip() or "!!!!" in got or len(set(got)) < 6
        item = dict(i=i, broken=bad, text=got[:80], error=r.get("error"))
        if str(i) in ref:
            want = ref[str(i)]; common = 0
            for x, y in zip(got, want):
                if x != y:
                    break
                common += 1
            item.update(match_chars=common, of=len(want))
        res.append(item)
    judged = [x for x in res if "of" in x]
    ok = all(not x["broken"] for x in res) and all(x["match_chars"] >= min(40, x["of"]) for x in judged)
    return ok, res


QUALITY = [("What is the capital of France? Answer in one word.", ["paris"]),
           ("What is 6 times 7? Answer with the number only.", ["42"]),
           ("What is 17 multiplied by 23? Answer with the number only.", ["391"]),
           ("List the first five prime numbers, separated by commas.", ["2", "3", "5", "7", "11"]),
           ("Which planet is known as the Red Planet? One word.", ["mars"]),
           ("What is the chemical symbol for gold? Answer with the symbol only.", ["au"]),
           ("Who wrote the play Romeo and Juliet? Give the name only.", ["shakespeare"]),
           ("What is the largest ocean on Earth? One word.", ["pacific"]),
           ("How many days are there in a leap year? Number only.", ["366"]),
           ("What is the square root of 144? Number only.", ["12"]),
           ("Translate 'good morning' into Spanish. Two words only.", ["buenos"]),
           ("What gas do plants absorb from the air for photosynthesis? Give its chemical formula.", ["co2", "co₂"])]


def cmd_quality(a):
    """Answers, not prefixes: 12 questions with known answers decoded side by side to the end of the answer.
    A change that keeps the first tokens but damages the model (half-precision paths) shows here."""
    d = os.path.join(LAB, "experiments", a.exp); os.makedirs(d, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(QUALITY)) as ex:
        outs = list(ex.map(lambda q: chat(0, a.tokens, timeout=a.cap, prompt=q[0]), QUALITY))
    res, good = [], 0
    for (q, want), r in zip(QUALITY, outs):
        text = r.get("text", ""); answer = text.split("</think>")[-1] if "</think>" in text else ""
        hit = bool(answer) and (all(w in answer.lower() for w in want) if len(want) > 2 else any(w in answer.lower() for w in want))
        words = text.split(); rep = len(words) > 30 and len(set(words)) < len(words) * 0.2
        good += hit
        res.append(dict(q=q, want=want, ok=hit, finished="</think>" in text, degenerate=rep, tokens=r.get("tokens"), answer=answer.strip()[:160],
                        error=r.get("error")))
        log("%s  %-62s -> %r%s" % ("ok " if hit else "BAD", q[:62], answer.strip()[:60], "  (no answer within the token budget)" if "</think>" not in text else ""))
    json.dump(dict(correct=good, of=len(QUALITY), items=res), open(os.path.join(d, "quality.json"), "w"), indent=1)
    log("QUALITY %d/%d correct" % (good, len(QUALITY)))
    return 0 if good >= len(QUALITY) - 1 else 1


SEED_ASKS = ["Explain how {} works, step by step.", "What are the main advantages and disadvantages of {}?",
             "Write a short paragraph about the history of {}.", "Give three practical tips related to {}.",
             "Compare {} with something similar and say which is better for a beginner.", "Summarise what a child should know about {}.",
             "What commonly goes wrong with {}, and how do you fix it?", "Write a four-line poem about {}."]


def cmd_seed(a):
    """Traffic whose only purpose is to teach the cross-request drafter how this model phrases things: many different
    prompts, decoded side by side. None of them is used in a timed phase ("fresh" prompts come from other templates)."""
    topics = THINGS + ["a sourdough starter", "the water cycle", "a chess opening", "compound interest", "a volcano", "a coral reef",
                       "the stock market", "a violin", "machine learning", "a marathon", "vaccination", "recycling", "a telescope",
                       "the Roman Empire", "a jet engine", "photosynthesis", "an electric car", "a symphony orchestra", "DNA", "a glacier"]
    prompts = [q.format(t) for t in topics for q in SEED_ASKS][: a.prompts]
    t0 = time.time(); done = tokens = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.streams) as ex:
        for r in ex.map(lambda p: chat(0, a.tokens, timeout=a.cap, prompt=p), prompts):
            done += "error" not in r; tokens += r.get("tokens", 0)
    log("SEED: %d of %d prompts, %d tokens in %.0f s (%.1f tok/s)" % (done, len(prompts), tokens, time.time() - t0, tokens / (time.time() - t0)))
    return 0


def cmd_gate(a):
    ok, res = gate()
    for x in res:
        log("gate prompt %d: %d/%d chars match%s  %s tok/s ttft %s  %r" % (x["i"], x["match_chars"], x["of"],
            " (exact)" if x["exact"] else "", x["tok_s"], x["ttft_s"], x["text"][:70]))
    log("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


class Telemetry(threading.Thread):
    """Polls /api/fleet/telemetry (when the fleet serves it) and /api/stats into a JSONL file."""

    def __init__(self, path, every=2.0):
        super().__init__(daemon=True); self.path, self.every, self.stop_ev = path, every, threading.Event()
        self.seen = {}; self.available = None

    def run(self):
        with open(self.path, "a") as f:
            while not self.stop_ev.is_set():
                t = time.time()
                try:
                    tel = get_json("/api/fleet/telemetry", timeout=10)
                    self.available = "ranks" in tel
                    for rank, e in (tel.get("ranks") or {}).items():
                        rec = {"lt": t, "rank": int(rank), "rt": e.get("rt"), "sys": e.get("sys"), "host": e.get("host")}
                        new = [p for p in e.get("profs", []) if p.get("at") not in self.seen.setdefault(rank, set())]
                        for p in new:
                            self.seen[rank].add(p.get("at"))
                        if new:
                            rec["profs"] = new
                        if e.get("static"):
                            rec["static"] = e["static"]
                        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                except Exception as e:  # noqa: BLE001
                    self.available = False if self.available is None else self.available
                try:
                    f.write(json.dumps({"lt": t, "stats": get_json("/api/stats", timeout=10)}) + "\n")
                except Exception:  # noqa: BLE001
                    pass
                f.flush()
                self.stop_ev.wait(max(0.2, self.every - (time.time() - t)))


THINGS = ["a bicycle gear", "a refrigerator", "a suspension bridge", "a sailboat tacking upwind", "a pendulum clock", "a zipper",
          "a solar panel", "a vaccine", "a compost heap", "a barometer", "a microwave oven", "a lighthouse lens", "yeast in bread",
          "a heat pump", "a tuning fork", "a canal lock", "a parachute", "a thermos flask", "a ballpoint pen", "a wind turbine",
          "a pressure cooker", "noise-cancelling headphones", "a gyroscope", "a camera shutter", "a water tower", "a fuse",
          "an escalator", "a sundial", "a violin bow", "a smoke detector", "a siphon", "a telescope mirror", "a lock and key",
          "a steam whistle", "a magnifying glass", "a hot-air balloon", "a loom", "a sewing machine", "a rain gauge", "a kite"]
ASKS = ["Explain in three sentences how {} works.", "What are two common misconceptions about {}? Be brief.",
        "Describe {} to someone who has never seen one, in two sentences.", "Give one surprising fact about {} and explain it."]


def raw_telemetry_path(exp):
    """Raw telemetry names the lab's hosts and this repository is public: it lives OUTSIDE the work tree (a forced
    `git add` of an experiment folder once published seven of these files). Records keep analysis.json only."""
    d = os.path.join(os.path.expanduser("~/inkling-release/autolab-telemetry"), exp)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "telemetry.jsonl")


def fresh_prompt(tag, i):
    """A prompt this fleet has (almost certainly) not seen: what the cross-request drafter is worth has to be measured on
    text it could not have memorised. Deterministic per (experiment tag, stream), different across experiments."""
    h = int(hashlib.sha256(("%s/%d" % (tag, i)).encode()).hexdigest(), 16)
    return ASKS[h % len(ASKS)].format(THINGS[(h >> 8) % len(THINGS)])


def family_prompt(name, tag, i):
    """`famK...` phases: an unseen prompt of collect.py's family K (0 explain, 1 code, 2 arithmetic, 3 story, 4 tips,
    5 table, 6 rewrite, 7 facts, 8 poem, 9 how-to, 10 translate, 11 true/false): how predictable the model's text is
    to a drafter depends on the kind of task far more than on anything else (013: 0.39 to 0.78 for the same drafter)."""
    import collect
    k = int(re.match(r"fam(\d+)", name).group(1)) % len(collect.FAMILIES)
    h = int(hashlib.sha256(("%s/%d" % (tag, i)).encode()).hexdigest(), 16)
    return collect.FAMILIES[k](h >> 4)


ECHO = "Repeat the following paragraph exactly, word for word, two times, and write nothing else:\n\n" + FILLER


def run_phase(name, streams, tokens, prompt_words, cap):
    t0 = time.time()
    # "echo..." phases ask for a copy of the prompt: the output repeats the input, the best case for n-gram drafts
    # "fresh..." phases use prompts derived from the phase's tag (RUN_TAG + phase name), unseen by earlier experiments
    prompts = [ECHO if name.startswith("echo") else fresh_prompt(RUN_TAG[0] + "/" + name, i) if name.startswith("fresh")
               else family_prompt(name, RUN_TAG[0] + "/" + name, i) if name.startswith("fam") else None
               for i in range(streams)]
    def start(i):
        # Hundreds of connections opened in the same millisecond get reset somewhere along the tunnel
        # (13 of 264 in exp 006); 15 ms apart they all arrive within a few seconds and none is lost.
        time.sleep(0.015 * i)
        return chat(i, tokens, prompt_words, cap, prompts[i])

    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as ex:
        futs = [ex.submit(start, i) for i in range(streams)]
        out = [f.result() for f in futs]
    wall = time.time() - t0
    ok = [x for x in out if "error" not in x]
    tot = sum(x["tokens"] for x in out)
    res = dict(phase=name, streams=streams, tokens_req=tokens, prompt_words=prompt_words, start=t0, end=t0 + wall, wall_s=round(wall, 1),
               completed=len(ok), tokens=tot, aggregate_tok_s=round(tot / wall, 3),
               sum_stream_tok_s=round(sum(x["tok_s"] or 0 for x in ok), 3),
               ttft_mean_s=round(sum(x["ttft_s"] for x in ok) / max(1, len(ok)), 2), ttft_max_s=max([x["ttft_s"] for x in ok] or [0]),
               errors=[x["error"] for x in out if "error" in x][:5], sample=(ok[0]["text"][:100] if ok else ""))
    log("PHASE %s: %d/%d done, %d tok in %.0f s = %.2f tok/s aggregate (sum of streams %.2f), ttft mean %.1f max %.1f%s" % (
        name, len(ok), streams, tot, wall, res["aggregate_tok_s"], res["sum_stream_tok_s"], res["ttft_mean_s"], res["ttft_max_s"],
        "  ERRORS: %s" % res["errors"][:2] if res["errors"] else ""))
    return res


RUN_TAG = [""]


def cmd_bench(a):
    RUN_TAG[0] = a.exp
    d = os.path.join(LAB, "experiments", a.exp); os.makedirs(d, exist_ok=True)
    tel = Telemetry(raw_telemetry_path(a.exp)); tel.start()
    results = []
    try:
        for spec in a.phases:
            f = spec.split(":"); name, streams, tokens = f[0], int(f[1]), int(f[2]); pw = int(f[3]) if len(f) > 3 else a.prompt_words
            results.append(run_phase(name, streams, tokens, pw, a.cap))
            json.dump(results, open(os.path.join(d, "phases.json"), "w"), indent=1)
            time.sleep(12)  # the ranks report a request's last profile window when they go idle
    finally:
        tel.stop_ev.set(); tel.join(5)
    log("telemetry endpoint available: %s" % tel.available)
    return 0


def cmd_run(a):
    if a.files:
        rc = cmd_publish(a)
        if rc != 0:
            log("publish/settle failed (rc %d)" % rc); return rc
    d = os.path.join(LAB, "experiments", a.exp); os.makedirs(d, exist_ok=True)
    if a.warm_streams:
        # A restart empties every rank's resident expert copy: touch many experts before anything is timed.
        run_phase("warmup", a.warm_streams, 24, 0, a.cap)
    for w in range(a.warm):
        r = chat(3 + w, 24, timeout=600)
        log("warm %d: %s tok/s, ttft %s, %s" % (w, r.get("tok_s"), r.get("ttft_s"), r.get("error") or repr(r.get("text", "")[:50])))
    ok, res = gate()
    json.dump(dict(ok=ok, prompts=res), open(os.path.join(d, "gate.json"), "w"), indent=1)
    for x in res:
        log("gate prompt %d: %d/%d chars match  %s tok/s ttft %s  %r" % (x["i"], x["match_chars"], x["of"], x["tok_s"], x["ttft_s"], x["text"][:60]))
    log("GATE", "PASS" if ok else "FAIL")
    if not ok and not a.force:
        return 3
    ok2, res2 = gate_multi()
    json.dump(dict(ok=ok, prompts=res, multi_ok=ok2, multi=res2), open(os.path.join(d, "gate.json"), "w"), indent=1)
    for x in res2:
        log("gate (8 side by side) prompt %d: %s%s  %r" % (x["i"], "%d/%d chars match" % (x["match_chars"], x["of"]) if "of" in x else "no reference",
            "  BROKEN" if x["broken"] else "", x["text"][:50]))
    log("GATE MULTI", "PASS" if ok2 else "FAIL")
    if not ok2 and not a.force:
        return 4
    return cmd_bench(a)


def main():
    try:  # macOS gives a process 256 descriptors: phases of more than ~250 streams lost the rest to "connection reset"
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, ValueError, OSError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("--log", action="store_true"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("settle"); s.add_argument("--expect", nargs="*", default=[]); s.add_argument("--cap", type=int, default=900); s.set_defaults(fn=cmd_settle)
    for name, fn in (("publish", cmd_publish), ("run", cmd_run)):
        s = sub.add_parser(name)
        if name == "run":
            s.add_argument("exp")
        s.add_argument("files", nargs="*"); s.add_argument("--note", required=True); s.add_argument("--allow-infra", action="store_true")
        s.add_argument("--cap", type=int, default=900)
        if name == "run":
            s.add_argument("--phases", nargs="+", required=True); s.add_argument("--prompt-words", type=int, default=0)
            s.add_argument("--warm", type=int, default=2); s.add_argument("--force", action="store_true")
            s.add_argument("--warm-streams", type=int, default=16)
        s.set_defaults(fn=fn)
    s = sub.add_parser("quality"); s.add_argument("exp"); s.add_argument("--tokens", type=int, default=320)
    s.add_argument("--cap", type=int, default=900); s.set_defaults(fn=cmd_quality)
    s = sub.add_parser("seed"); s.add_argument("--prompts", type=int, default=240); s.add_argument("--streams", type=int, default=96)
    s.add_argument("--tokens", type=int, default=128); s.add_argument("--cap", type=int, default=900); s.set_defaults(fn=cmd_seed)
    s = sub.add_parser("reference"); s.set_defaults(fn=cmd_reference)
    s = sub.add_parser("gate"); s.set_defaults(fn=cmd_gate)
    s = sub.add_parser("bench"); s.add_argument("exp"); s.add_argument("--phases", nargs="+", required=True)
    s.add_argument("--prompt-words", type=int, default=0); s.add_argument("--cap", type=int, default=900); s.set_defaults(fn=cmd_bench)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
