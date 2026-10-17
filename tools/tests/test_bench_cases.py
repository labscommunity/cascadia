"""bench_cases: typed-decisions latency runs, run-to-run parity, engine log summary.

    python -m pytest tools/tests/test_bench_cases.py -v
"""
import json
import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "decision"))
import bench_cases as bc  # noqa: E402


def rec(cid, wf, model, lat, answers, status=200):
    return {"id": cid, "workflow": wf, "model": model, "status": status, "latency_s": lat,
            "answers": answers, "usage": {}, "error": None}


NOUL = lambda p: {"type": "noul", "noul": p}  # noqa: E731
CHOICE = lambda d: {"type": "choice", "choice": max(d, key=d.get), "probabilities": d}  # noqa: E731


def test_select_takes_the_first_n_per_workflow_in_file_order():
    cases = [{"id": f"{w}{i}", "workflow": w, "body": {}} for w in "ab" for i in range(3)]
    assert [c["id"] for c in bc.select(cases, 2)] == ["a0", "a1", "b0", "b1"]
    assert len(bc.select(cases, None)) == 6


def test_summary_percentiles_and_errors():
    rs = [rec(str(i), "w", "m", float(i), {}) for i in range(1, 11)]
    rs.append(rec("x", "w", "m", 99.0, {}, status=503))
    s = bc.summary(rs)
    assert s["m"]["n"] == 10 and s["m"]["errors"] == 1
    assert s["m"]["p50"] == 5.5 and abs(s["m"]["p90"] - 9.1) < 1e-9


def test_compare_reports_max_dp_and_flips_with_margins():
    a = [rec("1", "w", "m", 1.0, {"q": NOUL(0.80), "c": CHOICE({"x": 0.51, "y": 0.49})})]
    b = [rec("1", "w", "m", 0.9, {"q": NOUL(0.78), "c": CHOICE({"x": 0.48, "y": 0.52})})]
    r = bc.compare(a, b, confident=0.05, max_dp=0.05)
    assert abs(r["max_dp"] - 0.03) < 1e-9
    assert r["flips"] == 1 and r["confident_flips"] == 0 and r["ok"]


def test_compare_fails_a_confident_flip_or_a_large_dp():
    a = [rec("1", "w", "m", 1.0, {"q": NOUL(0.90)})]
    b = [rec("1", "w", "m", 1.0, {"q": NOUL(0.40)})]
    r = bc.compare(a, b, confident=0.05, max_dp=0.05)
    assert r["confident_flips"] == 1 and not r["ok"]


def test_compare_refuses_mismatched_case_sets():
    a = [rec("1", "w", "m", 1.0, {"q": NOUL(0.9)})]
    b = [rec("2", "w", "m", 1.0, {"q": NOUL(0.9)})]
    r = bc.compare(a, b, confident=0.05, max_dp=0.05)
    assert not r["ok"] and r["missing"] == ["1", "2"]


def test_logsum_parses_tracing_lines_with_ansi():
    lines = [
        "\x1b[2m2026-10-17T10:00:00Z\x1b[0m \x1b[34mDEBUG\x1b[0m cascadia::perf: qwen36 decision pass done "
        'task=dec-1 head="pointer" prompt=250 restore_ms=301.2 recreate=true',
        "2026-10-17T10:00:01Z DEBUG cascadia::perf: qwen36 decision pass done task=dec-2 head=\"pointer\" "
        "prompt=80 restore_ms=299.0 recreate=true",
        "2026-10-17T10:00:01Z INFO other line restore_ms=5",
    ]
    s = bc.logsum(lines, "qwen36 decision pass done")
    assert s["restore_ms"]["n"] == 2 and abs(s["restore_ms"]["p50"] - 300.1) < 1e-9
    assert s["prompt"]["p50"] == 165.0


# The baseline client (td_perf.py, 2026-10-17): rows grouped by workflow in file order, then
# `order = [by[w][i] for i in range(10) for w in sorted(by)]`.
def baseline_order(rows, n):
    by = {}
    for r in rows:
        by.setdefault(r["workflow"], []).append(r)
    return [by[w][i] for i in range(n) for w in sorted(by)]


def split_rows():
    # Interleaved, unsorted workflow names, 12 rows each: file order is not workflow order.
    rows = []
    for i in range(12):
        for w in ("triage", "billing", "routing"):
            rows.append({"id": f"{w}-{i}", "workflow": w,
                         "state": json.dumps({"n": i, "w": w}),
                         "questions": json.dumps({"q": {"type": "noul", "question": "?"}})})
    return rows


def test_cases_reproduce_the_baseline_client_order_and_body():
    rows = split_rows()
    cases = bc.to_cases(rows, 10)
    assert [c["id"] for c in cases] == [r["id"] for r in baseline_order(rows, 10)]
    assert [c["id"] for c in cases[:4]] == ["billing-0", "routing-0", "triage-0", "billing-1"]
    assert len(cases) == 30
    assert cases[0]["body"] == {"state": {"n": 0, "w": "billing"},
                                "questions": {"q": {"type": "noul", "question": "?"}}}
    # run keeps the converted order (no per-workflow filter re-sorts it).
    assert bc.select(cases, None) == cases


def test_cases_cli_reads_a_json_array_and_writes_private(tmp_path):
    split, out = tmp_path / "td-test.json", tmp_path / "td40.jsonl"
    split.write_text(json.dumps(split_rows()))
    assert bc.main(["cases", str(split), str(out)]) == 0
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    assert [c["id"] for c in bc.load(out)] == [r["id"] for r in baseline_order(split_rows(), 10)]


def test_compare_reads_the_baseline_client_lines():
    # td_perf.py lines: {"id", "latency_s", "probs"}, a noul as {"false": 1-p, "true": p}.
    base = [{"id": "1", "latency_s": 2.0,
             "probs": {"q": {"false": 0.2, "true": 0.8}, "c": {"x": 0.7, "y": 0.3}}}]
    new = [rec("1", "w", "kev-9b", 1.5, {"q": NOUL(0.8), "c": CHOICE({"x": 0.7, "y": 0.3})})]
    r = bc.compare(base, new, confident=0.05, max_dp=1e-4)
    assert r["ok"] and r["max_dp"] < 1e-12 and r["flips"] == 0
    sa, sb = bc.summary(base), bc.summary(new)
    assert bc.model_pairs(sa, sb) == [(None, "kev-9b")]
    assert sa[None]["p50"] == 2.0 and sa[None]["errors"] == 0


def test_compare_cli_exit_code(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    bc.write_private(a, [rec("1", "w", "m", 1.0, {"q": NOUL(0.90)})])
    bc.write_private(b, [rec("1", "w", "m", 1.1, {"q": NOUL(0.40)})])
    assert bc.main(["compare", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "max |dp| = 0.5000" in out and "flips = 1 (confident 1)" in out and "FAIL" in out
    assert bc.main(["compare", str(a), str(a)]) == 0


def test_compare_fails_when_both_runs_errored_everywhere():
    # Server down for both runs: nothing was compared, so it must not read as a PASS.
    a = [rec(str(i), "w", "m", 0.0, {}, status=0) for i in range(3)]
    b = [rec(str(i), "w", "m", 0.0, {}, status=0) for i in range(3)]
    r = bc.compare(a, b, confident=0.05, max_dp=0.05)
    assert not r["ok"] and r["compared"] == 0
    assert r["errors_a"] == r["errors_b"] == ["0", "1", "2"]
    # --allow-errors relaxes only the error rule: an empty comparison still fails.
    assert not bc.compare(a, b, confident=0.05, max_dp=0.05, allow_errors=True)["ok"]


def test_compare_fails_on_a_case_errored_in_one_run_unless_allowed():
    a = [rec("1", "w", "m", 1.0, {"q": NOUL(0.9)}), rec("2", "w", "m", 1.0, {"q": NOUL(0.2)})]
    b = [rec("1", "w", "m", 1.0, {"q": NOUL(0.9)}), rec("2", "w", "m", 900.0, {}, status=0)]
    r = bc.compare(a, b, confident=0.05, max_dp=0.05)
    assert not r["ok"] and r["errors_a"] == [] and r["errors_b"] == ["2"] and r["compared"] == 1
    assert r["missing"] == []
    assert bc.compare(a, b, confident=0.05, max_dp=0.05, allow_errors=True)["ok"]


def test_compare_fails_when_nothing_is_compared():
    assert not bc.compare([], [], confident=0.05, max_dp=0.05)["ok"]


def test_compare_checks_questions_on_both_sides():
    a = [rec("1", "w", "m", 1.0, {"q": NOUL(0.9)})]
    b = [rec("1", "w", "m", 1.0, {"q": NOUL(0.9), "extra": NOUL(0.5)})]
    r = bc.compare(a, b, confident=0.05, max_dp=0.05)
    assert not r["ok"] and r["missing"] == ["1.extra"]


def test_compare_cli_reports_errors_and_exits_nonzero(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    bc.write_private(a, [rec("1", "w", "m", 0.0, {}, status=0)])
    bc.write_private(b, [rec("1", "w", "m", 0.0, {}, status=0)])
    assert bc.main(["compare", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "compared 0 cases; errored: 1 in" in out and "nothing compared" in out and "FAIL" in out


def test_baseline_noul_is_ordered_false_true_whatever_the_key_order():
    r = {"id": "1", "latency_s": 1.0, "probs": {"q": {"true": 0.8, "false": 0.2}}}
    assert bc.probs_of(r) == {"q": [0.2, 0.8]}


def test_post_records_invalid_json_as_an_error(monkeypatch):
    class Resp:
        status = 200

        def read(self):
            return b"not json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(bc.urllib.request, "urlopen", lambda req, timeout: Resp())
    status, _, data, err = bc.post("http://x", {}, 1.0)
    assert status == 200 and data is None and "invalid JSON" in err
    assert not bc.ok({"status": status, "error": err})


def test_post_survives_an_incomplete_read(monkeypatch):
    def boom(req, timeout):
        raise bc.http.client.IncompleteRead(b"")

    monkeypatch.setattr(bc.urllib.request, "urlopen", boom)
    status, _, data, err = bc.post("http://x", {}, 1.0)
    assert status == 0 and data is None and "IncompleteRead" in err
