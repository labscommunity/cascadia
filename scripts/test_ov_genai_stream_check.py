"""Offline tests for scripts/ov_genai_stream_check.py (the verdict rules, and
the client against a scripted loopback server; no worker needed).

    python3 -m unittest scripts/test_ov_genai_stream_check.py
"""

import http.server
import json
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ov_genai_stream_check as c  # noqa: E402


def rec(deltas=(), tokens=None, t_end=None, **kw):
    r = {
        "deltas": [list(d) for d in deltas],
        "content": "".join(t for _, t in deltas),
        "completion_tokens": tokens,
        "t_end": t_end,
    }
    r.update(kw)
    return r


def frame(content=None, finish=None, usage=None):
    j = {"choices": []}
    if content is not None or finish is not None:
        j["choices"].append({"delta": {"content": content} if content is not None else {}, "finish_reason": finish})
    if usage is not None:
        j["usage"] = usage
    return "data: " + json.dumps(j)


class Median(unittest.TestCase):
    def test_odd_even_empty(self):
        self.assertEqual(c.median([3, 1, 2]), 2)
        self.assertEqual(c.median([4, 1, 2, 3]), 2.5)
        self.assertEqual(c.median([]), 0.0)


class ParseSse(unittest.TestCase):
    def test_deltas_usage_finish_done(self):
        r = c.parse_sse(
            [
                (0.01, ": keep-alive"),
                (0.02, frame(content="")),  # a role-only frame is not a delta
                (0.05, frame(content="One")),
                (0.06, ""),
                (0.08, frame(content=", two")),
                (0.09, frame(finish="length")),
                (0.10, frame(usage={"completion_tokens": 7})),
                (0.11, "data: [DONE]"),
            ]
        )
        self.assertEqual(r["deltas"], [[0.05, "One"], [0.08, ", two"]])
        self.assertEqual(r["content"], "One, two")
        self.assertEqual(r["finish"], "length")
        self.assertEqual(r["completion_tokens"], 7)
        self.assertTrue(r["done"])
        self.assertEqual(r["frames"], 5)

    def test_error_frame_and_garbage(self):
        r = c.parse_sse([(0.1, "data: not json"), (0.2, 'data: {"error": {"message": "boom"}}')])
        self.assertEqual(r["error"], {"message": "boom"})
        self.assertEqual(r["deltas"], [])

    def test_non_streamed_message(self):
        r = {"content": "", "deltas": [], "finish": None, "completion_tokens": None, "error": None}
        c.take(r, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}})
        self.assertEqual((r["content"], r["finish"], r["completion_tokens"], r["deltas"]), ("hi", "stop", 2, []))


class FirstByte(unittest.TestCase):
    def test_median_against_the_bar(self):
        runs = [rec([(0.10, "a")]), rec([(0.20, "a")]), rec([(0.11, "a")])]
        ok, m, xs = c.first_byte_verdict(runs, 0.12)
        self.assertTrue(ok)
        self.assertAlmostEqual(m, 0.11)
        self.assertFalse(c.first_byte_verdict(runs, 0.105)[0])

    def test_a_run_without_a_delta_is_infinitely_late(self):
        ok, m, _ = c.first_byte_verdict([rec(), rec(), rec([(0.05, "a")])], 0.12)
        self.assertFalse(ok)
        self.assertEqual(m, float("inf"))


class Deltas(unittest.TestCase):
    def test_one_delta_per_eight_tokens(self):
        thirty_two = rec([(i * 0.1, "x") for i in range(32)], tokens=256)
        thirty_one = rec([(i * 0.1, "x") for i in range(31)], tokens=256)
        self.assertTrue(c.deltas_verdict([thirty_two], 8)[0])
        self.assertFalse(c.deltas_verdict([thirty_one], 8)[0])

    def test_speculative_k_plus_one(self):
        # --speculative 3: one delta per 4 tokens at most.
        r = rec([(i * 0.1, "x") for i in range(64)], tokens=256)
        self.assertTrue(c.deltas_verdict([r], 4)[0])
        self.assertFalse(c.deltas_verdict([r], 3)[0])

    def test_one_final_burst_fails_and_missing_usage_fails(self):
        self.assertFalse(c.deltas_verdict([rec([(1.0, "all")], tokens=256)], 8)[0])
        self.assertFalse(c.deltas_verdict([rec([(0.1, "x")] * 40, tokens=None)], 8)[0])

    def test_median_over_runs(self):
        good = rec([(0, "x")] * 64, tokens=256)
        bad = rec([(0, "x")], tokens=256)
        self.assertTrue(c.deltas_verdict([good, good, bad], 8)[0])
        self.assertFalse(c.deltas_verdict([good, bad, bad], 8)[0])


class TokS(unittest.TestCase):
    def test_tokens_over_total(self):
        runs = [rec(tokens=256, t_end=5.0), rec(tokens=256, t_end=6.4), rec(tokens=256, t_end=7.0)]
        ok, m, _ = c.tok_s_verdict(runs, 40)
        self.assertTrue(ok)
        self.assertAlmostEqual(m, 256 / 6.4)  # 40.0, the median run
        self.assertFalse(c.tok_s_verdict(runs, 41)[0])
        self.assertFalse(c.tok_s_verdict([rec(tokens=None, t_end=1.0)], 41)[0])


class Parity(unittest.TestCase):
    def test_equal_and_unequal(self):
        s = rec([(0.1, "One, "), (0.2, "two")])
        self.assertEqual(c.parity_verdict(s, [{"content": "One, two"}, {"content": "One, two"}])[0], "ok")
        v, detail = c.parity_verdict(s, [{"content": "One,  two"}, {"content": "One,  two"}])
        self.assertEqual(v, "bad")
        self.assertIn("char 5", detail)

    def test_byte_exact_not_normalised(self):
        s = rec([(0.1, "café")])
        self.assertEqual(c.parity_verdict(s, [{"content": "café"}])[0], "bad")
        self.assertEqual(c.parity_verdict(rec([(0.1, " a")]), [{"content": "a"}])[0], "bad")

    def test_references_that_disagree_make_it_unverifiable(self):
        s = rec([(0.1, "x")])
        self.assertEqual(c.parity_verdict(s, [{"content": "y"}, {"content": "z"}])[0], "varies")
        self.assertEqual(c.parity_verdict(s, [{"content": "y"}, {"content": "x"}])[0], "ok")

    def test_first_diff(self):
        self.assertEqual(c.first_diff("abc", "abc"), -1)
        self.assertEqual(c.first_diff("abc", "abd"), 2)
        self.assertEqual(c.first_diff("ab", "abc"), 2)


class Headers(unittest.TestCase):
    def test_bar_status_and_type(self):
        ok_rec = {"t_headers": 0.4, "status": 200, "content_type": "text/event-stream; charset=utf-8"}
        self.assertTrue(c.headers_verdict(ok_rec, 1.0)[0])
        self.assertFalse(c.headers_verdict(dict(ok_rec, t_headers=1.2), 1.0)[0])
        self.assertFalse(c.headers_verdict(dict(ok_rec, status=503), 1.0)[0])
        self.assertFalse(c.headers_verdict(dict(ok_rec, content_type="application/json"), 1.0)[0])
        ok, t, detail = c.headers_verdict({"t_headers": None, "error": "timeout"}, 1.0)
        self.assertFalse(ok)
        self.assertIn("timeout", detail)


class Gap(unittest.TestCase):
    def test_even_stream_passes(self):
        r = rec([(0.1 + i * 0.025, "x") for i in range(40)])
        self.assertTrue(c.gap_verdict(r, 3)[0])

    def test_a_burst_fails(self):
        times = [0.1 + i * 0.025 for i in range(20)] + [2.0 + i * 0.025 for i in range(20)]
        ok, mx, md, n = c.gap_verdict(rec([(t, "x") for t in times]), 3)
        self.assertFalse(ok)
        self.assertGreater(mx, 1.0)
        self.assertEqual(n, 39)

    def test_too_few_deltas_cannot_be_judged(self):
        self.assertFalse(c.gap_verdict(rec([(0.1, "all")]), 3)[0])


class Log(unittest.TestCase):
    LOG = "\n".join(
        [
            "2026-11-13T00:00:00Z  INFO cascadia_engine_openvino::genai: task done task=a tokens=256 elapsed_s=5.1 tok_s=50.2 ttft_ms=88.5",
            "2026-11-13T00:00:01Z  INFO cascadia_engine_openvino::genai: task done task=b tokens=12 elapsed_s=0.3 tok_s=40",
            "2026-11-13T00:00:02Z  WARN cascadia_engine_openvino::genai: ov_genai_stream_divergence: the streamed text is not a prefix of the answer task=c",
            "2026-11-13T00:00:03Z  INFO cascadia_engine_openvino::genai: task stopped at stop string task=d tokens=3 ttft_ms=1.0",
        ]
    )

    def test_ttft_only_from_task_done_lines_that_carry_it(self):
        self.assertEqual(c.ttft_values(self.LOG), [88.5])

    def test_ttft_through_terminal_colours(self):
        line = "\x1b[32m INFO\x1b[0m task done \x1b[3mttft_ms\x1b[0m\x1b[2m=\x1b[0m42.5"
        self.assertEqual(c.ttft_values(line), [42.5])

    def test_divergence_count(self):
        self.assertEqual(c.count_divergence(self.LOG), 1)
        self.assertEqual(c.count_divergence(""), 0)


class Body(unittest.TestCase):
    def test_streamed_greedy_with_usage(self):
        b = c.body("count", "256", True, True, "m")
        self.assertEqual(b["temperature"], 0)
        self.assertEqual(b["max_tokens"], 256)
        self.assertTrue(b["stream"])
        self.assertEqual(b["stream_options"], {"include_usage": True})
        self.assertIs(b["enable_thinking"], False)
        b = c.body("count", "256", False, False, "m")
        self.assertNotIn("stream", b)
        self.assertNotIn("enable_thinking", b)


class Client(unittest.TestCase):
    """`send` against a loopback server that streams deltas 50 ms apart."""

    @classmethod
    def setUpClass(cls):
        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers["content-length"])
                req = json.loads(self.rfile.read(n))
                if req.get("stream"):
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("transfer-encoding", "chunked")
                    self.end_headers()

                    def chunk(s):
                        b = (s + "\n\n").encode()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
                        self.wfile.flush()

                    for word in ["One", ", two", ", three"]:
                        time.sleep(0.05)
                        chunk(frame(content=word))
                    chunk(frame(finish="length"))
                    chunk(frame(usage={"completion_tokens": 3}))
                    chunk("data: [DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    out = json.dumps(
                        {
                            "choices": [{"message": {"content": "One, two, three"}, "finish_reason": "length"}],
                            "usage": {"completion_tokens": 3},
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)

        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        cls.url = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_streamed_timings_and_parity(self):
        s = c.send(self.url, c.body("count", 3, True, True, "m"), 10)
        self.assertIsNone(s["error"])
        self.assertEqual(s["status"], 200)
        self.assertEqual([t for _, t in s["deltas"]], ["One", ", two", ", three"])
        times = [t for t, _ in s["deltas"]]
        self.assertGreaterEqual(times[0], 0.04)
        self.assertTrue(all(b - a >= 0.04 for a, b in zip(times, times[1:])))
        self.assertLess(s["t_headers"], times[0])
        self.assertEqual(s["completion_tokens"], 3)
        self.assertTrue(s["done"])
        self.assertTrue(c.headers_verdict(s, 1.0)[0])
        n = c.send(self.url, c.body("count", 3, False, True, "m"), 10)
        self.assertEqual(c.parity_verdict(s, [n])[0], "ok")
        self.assertIn("first delta", c.summary(s))

    def test_refused_connection_is_an_error_not_a_raise(self):
        r = c.send("http://127.0.0.1:9", c.body("short", 1, True, True, "m"), 2)
        self.assertIsNotNone(r["error"])
        self.assertIsNone(r["t_headers"])


if __name__ == "__main__":
    unittest.main()
