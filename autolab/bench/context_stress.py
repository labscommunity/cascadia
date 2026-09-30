#!/usr/bin/env python3
"""context_stress.py EXP [--sizes 1024,2048,...] [--budget-s 10800] [--repeats 3,2,1] [--streams 1,4] [--new-tokens 96]

The fleet's performance against context length, stepped up until the budget ends (autolab 047, for the paper).

Every measurement is one request (or `--streams` requests at once) whose prompt is natural text of about N tokens
with a needle sentence near the start and, at the end, a question asking for it. Recorded per request: exact prompt
tokens (the API's `usage`), time to first token (the prefill), prefill tokens/s, decode tokens/s at that context,
tokens produced, whether the needle's code appears anywhere in the output (thinking included: the model reasons
first), and, sampled every 5 s during the request from /api/fleet/telemetry, every box's CPU, GPU, package power,
memory, and its stage profile (attention and expert time per frame).

Order: for each concurrency in `--streams`, sizes ascending; `--repeats a,b,c` = repeats for sizes <= 16k, <= 64k,
above. Before each measurement the harness predicts its time from the prefill rates measured so far (the slowest
rate at or above that size; 40 tok/s before any) and stops the pass when the remaining budget cannot hold it; the
next concurrency starts where the budget allows. Everything is written to experiments/EXP/context_stress.json
after EVERY request, plus a CSV (context_stress.csv) for plotting. Standard library only; no fleet state changes.
"""
import argparse, concurrent.futures, csv, json, os, random, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
API = os.environ.get("INKLING_API", "http://localhost:18000")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHARS_PER_TOKEN = 4.4

sys.path.insert(0, HERE)
from context_scan import text_pool, build_prompt, log  # noqa: E402


def chat(prompt, new_tokens, cap):
    body = json.dumps({"model": "inkling", "stream": True, "max_tokens": new_tokens, "temperature": 0,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; n = 0; text = []; usage = None; err = None; stamps = []
    try:
        with OPENER.open(req, timeout=cap) as r:
            for line in r:
                if time.time() - t0 > cap:
                    err = "cap"; break
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
                if v.get("usage"):
                    usage = v["usage"]
                if "error" in v and not v.get("choices"):
                    err = str(v["error"])[:200]; break
                ch = (v.get("choices") or [{}])[0]; d = ch.get("delta") or {}
                if ch.get("finish_reason") is None and d != {} and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    now = time.time(); first = first or now; last = now; n += 1; stamps.append(round(now - t0, 3))
                    text.append(d.get("content") or d.get("reasoning_content") or d.get("reasoning") or "")
    except Exception as e:  # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, str(e)[:160])
    return dict(wall_s=round(time.time() - t0, 2), ttft_s=round((first - t0), 2) if first else None, tokens=n,
                decode_tok_s=round((n - 1) / (last - first), 3) if n > 1 and last > first else None,
                prompt_tokens=(usage or {}).get("prompt_tokens"), text="".join(text), error=err,
                token_times=stamps[:200])


class Sampler(threading.Thread):
    """Every 5 s: each box's load and its latest stage profile, while a request runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                d = json.loads(OPENER.open(API + "/api/fleet/telemetry", timeout=10).read())
                row = {"t": round(time.time(), 1), "ranks": {}}
                for r, e in d.get("ranks", {}).items():
                    s = e.get("sys", {})
                    prof = [p for p in e.get("profs", []) if isinstance(p, dict) and p.get("frames")]
                    p = prof[-1] if prof else {}
                    row["ranks"][r] = {k: s.get(k) for k in ("cpu", "gpu", "pkg_w", "psys_w", "mem_avail", "mhz", "temp", "swap")}
                    row["ranks"][r].update({k: p.get(k) for k in ("frames", "rows", "compute_ms", "attn_ms", "mlp_ms", "ov_attn_ms", "ov_moe_ms", "prefill_ms", "head_ms", "window_ms") if k in p})
                self.samples.append(row)
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(5)


def summarize_samples(samples):
    """Per box: mean CPU %, GPU %, package W, min free memory over the request."""
    out = {}
    for s in samples:
        for r, v in s["ranks"].items():
            o = out.setdefault(r, {"cpu": [], "gpu": [], "pkg_w": [], "mem_avail": [], "attn_ms": [], "mlp_ms": []})
            for k in o:
                if v.get(k) is not None:
                    o[k].append(v[k])
    def mean(xs): return round(sum(xs) / len(xs), 2) if xs else None
    return {r: {"cpu_mean": mean(v["cpu"]), "gpu_mean": mean(v["gpu"]), "pkg_w_mean": mean(v["pkg_w"]),
                "mem_avail_min_mb": min(v["mem_avail"]) if v["mem_avail"] else None,
                "attn_ms_last": v["attn_ms"][-1] if v["attn_ms"] else None, "mlp_ms_last": v["mlp_ms"][-1] if v["mlp_ms"] else None}
            for r, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--sizes", default="1024,2048,4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--budget-s", type=int, default=10800)
    ap.add_argument("--repeats", default="3,2,1", help="repeats for sizes <= 16k, <= 64k, above")
    ap.add_argument("--streams", default="1,4", help="concurrencies, each a full ascending pass")
    ap.add_argument("--new-tokens", type=int, default=96)
    ap.add_argument("--cap", type=int, default=21600, help="cap per request (s): a 256k prompt can take hours")
    a = ap.parse_args()
    sizes = sorted({int(s) for s in a.sizes.split(",") if s.strip()})
    reps = [int(x) for x in a.repeats.split(",")]
    streams_list = [int(x) for x in a.streams.split(",")]
    pool = text_pool()
    out_dir = os.path.join(LAB, "experiments", a.exp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "context_stress.json")
    csv_path = os.path.join(out_dir, "context_stress.csv")
    results, t_start = [], time.time()
    state = {"note": "started", "stopped_reason": None}

    def save():
        json.dump(dict(exp=a.exp, sizes=sizes, budget_s=a.budget_s, new_tokens=a.new_tokens, streams=streams_list,
                       repeats=reps, results=results, wall_s=round(time.time() - t_start), **state),
                  open(out_path + ".tmp", "w"), indent=1)
        os.replace(out_path + ".tmp", out_path)
        with open(csv_path + ".tmp", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["streams", "size", "repeat", "prompt_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s", "tokens", "wall_s",
                        "needle_found", "error", "aggregate_decode_tok_s", "started_at"])
            for r in results:
                if r.get("skipped") or r.get("in_flight"):
                    continue
                w.writerow([r["streams"], r["size"], r["repeat"], r.get("prompt_tokens"), r.get("ttft_s"), r.get("prefill_tok_s"),
                            r.get("decode_tok_s"), r.get("tokens"), r.get("wall_s"), r.get("needle_found"), r.get("error") or "",
                            r.get("aggregate_decode_tok_s"), r.get("started_at")])
        os.replace(csv_path + ".tmp", csv_path)

    def rate_for(size):
        # the slowest prefill rate measured at this size or above (the rate falls with context), else at any size
        rs = [r["prefill_tok_s"] for r in results if r.get("prefill_tok_s") and r["size"] >= size]
        if not rs:
            rs = [r["prefill_tok_s"] for r in results if r.get("prefill_tok_s")]
        return min(rs) if rs else 40.0

    save()
    budget_file = os.path.join(out_dir, "budget_s")   # the operator can change the budget while the run is on

    def budget():
        try:
            v = int(open(budget_file).read().strip())
            if v != a.budget_s:
                log("budget changed to %d s (from %s)" % (v, budget_file))
                a.budget_s = v
        except (OSError, ValueError):
            pass
        return a.budget_s

    for streams in streams_list:
        for size in sizes:
            n_rep = reps[0] if size <= 16384 else reps[1] if size <= 65536 else reps[2]
            for rep in range(n_rep):
                left = budget() - (time.time() - t_start)
                rate = rate_for(size)
                # prefill grows with the square of the context on the CPU attention: assume the rate at a larger
                # size is at best the rate measured, and with `streams` prompts at once the pipeline is shared
                predicted = streams * size / rate + a.new_tokens / 1.5 + 10
                if predicted > left:
                    log("streams %d size %d rep %d: predicted %.0f s at %.0f tok/s, %.0f s left: stopping this pass" % (streams, size, rep, predicted, rate, left))
                    results.append(dict(streams=streams, size=size, repeat=rep, skipped=True, predicted_s=round(predicted), left_s=round(left)))
                    save()
                    break
                prompt, code = build_prompt(pool, size, seed=size * 7 + rep)
                started = time.strftime("%H:%M:%S")
                results.append(dict(streams=streams, size=size, repeat=rep, in_flight=True, started_at=started))
                state["note"] = "streams %d size %d rep %d in flight since %s" % (streams, size, rep, started)
                save(); results.pop()
                log("streams %d size %d rep %d: %d chars, predicted %.0f s" % (streams, size, rep, len(prompt), predicted))
                sampler = Sampler(); sampler.start()
                cap = min(a.cap, max(120, left + 60))
                if streams > 1:
                    with concurrent.futures.ThreadPoolExecutor(streams) as ex:
                        rs = list(ex.map(lambda _: chat(prompt, a.new_tokens, cap), range(streams)))
                else:
                    rs = [chat(prompt, a.new_tokens, cap)]
                sampler.stop.set(); sampler.join(timeout=10)
                r = rs[0]
                pt = r.get("prompt_tokens")
                prefill_rate = round(pt / r["ttft_s"], 1) if pt and r.get("ttft_s") else None
                agg = round(sum(x.get("decode_tok_s") or 0 for x in rs), 2) if streams > 1 else r.get("decode_tok_s")
                row = dict(streams=streams, size=size, repeat=rep, started_at=started, prompt_tokens=pt, ttft_s=r.get("ttft_s"),
                           prefill_tok_s=prefill_rate, decode_tok_s=r.get("decode_tok_s"), aggregate_decode_tok_s=agg,
                           tokens=r.get("tokens"), wall_s=r.get("wall_s"), needle_found=code in (r.get("text") or ""),
                           error=r.get("error"), answer=(r.get("text") or "")[-300:], token_times=r.get("token_times"),
                           others=[dict(prompt_tokens=x.get("prompt_tokens"), ttft_s=x.get("ttft_s"), decode_tok_s=x.get("decode_tok_s"),
                                        tokens=x.get("tokens"), needle_found=code in (x.get("text") or ""), error=x.get("error")) for x in rs[1:]],
                           boxes=summarize_samples(sampler.samples), samples=sampler.samples[-40:])
                results.append(row)
                state["note"] = "last: streams %d size %d rep %d" % (streams, size, rep)
                save()
                log("streams %d size %d rep %d: prompt %s tok, first token %s s (%s tok/s prefill), decode %s tok/s%s, needle %s%s" % (
                    streams, size, rep, pt, r.get("ttft_s"), prefill_rate, r.get("decode_tok_s"),
                    (" (aggregate %s)" % agg) if streams > 1 else "", "FOUND" if row["needle_found"] else "missing",
                    (", error: " + r["error"]) if r.get("error") else ""))
                if r.get("error"):
                    state["stopped_reason"] = "error at streams %d size %d: %s" % (streams, size, r["error"])
                    save()
                    log("stopping after an error")
                    print_table(results)
                    return
    state["note"] = "finished"
    save()
    print_table(results)


def print_table(results):
    print("\n| streams | context (tokens) | first token (s) | prefill tok/s | decode tok/s (per stream) | aggregate | needle | note |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("skipped"):
            print("| %d | %d | - | - | - | - | - | stopped: %d s predicted, %d s left |" % (r["streams"], r["size"], r["predicted_s"], r["left_s"]))
        elif not r.get("in_flight"):
            print("| %d | %s | %s | %s | %s | %s | %s | %s |" % (r["streams"], r.get("prompt_tokens") or r["size"], r.get("ttft_s"), r.get("prefill_tok_s"),
                                                             r.get("decode_tok_s"), r.get("aggregate_decode_tok_s"),
                                                             "yes" if r.get("needle_found") else "NO", r.get("error") or ""))


if __name__ == "__main__":
    main()
