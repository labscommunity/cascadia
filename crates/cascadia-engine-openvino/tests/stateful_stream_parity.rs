//! Byte parity of the streamed stateful ov-genai path with its non-streamed
//! answer, on a real pipeline.
//!
//! With `OvGenaiBuilder::stream_writes(true)` the default (stateful) engine
//! hands out the text as the pipeline writes it: each `step()` returns the
//! text two consecutive decodes agree on (one write late), holds back an
//! echoed prompt, the whitespace after it and a character split across
//! tokens, and the final chunk carries the rest. The deltas plus that rest
//! must be exactly the answer the same engine returns with streaming off.
//! The unit tests prove the stream state on scripted decodes; only a real
//! tokenizer shows whether its detokenizer ever rewrites text it already
//! decoded (a cleanup rule such as `it '` -> `it's`, a byte-fallback
//! character, a leading-space strip). The engine logs such a rewrite as
//! `ov_genai_stream_divergence`; this test fails on that line too.
//!
//! For each prompt (ASCII, code, CJK, emoji, a counting list, an answer that
//! starts with whitespace), greedy, 256 new tokens:
//!
//! * the reference: the full engine built with `stream_writes(false)`, twice
//!   (some devices are not reproducible at temperature 0 over long answers;
//!   when the two differ the prompt's byte compare accepts either and says
//!   so, and the test fails only if every prompt is unverifiable);
//! * the stream: the full engine built with `stream_writes(true)`;
//! * asserts: deltas + final rest == the reference byte for byte, no
//!   `ov_genai_stream_divergence` line, at least one delta per 8 completion
//!   tokens (`CASCADIA_STREAM_PARITY_TOKENS_PER_DELTA`, k+1 for a speculative
//!   round of k), and prints each prompt's first delta, total, tok/s and the
//!   engine's `ttft_ms`.
//!
//! Hardware + a real IR are required:
//!
//!   CASCADIA_STREAM_PARITY_MODEL=<ir-dir> [CASCADIA_STREAM_PARITY_DEVICE=GPU] \
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!     --test stateful_stream_parity -- --nocapture
//!
//! Skips when `CASCADIA_STREAM_PARITY_MODEL` is unset. Other knobs:
//! `CASCADIA_STREAM_PARITY_DEVICE` (default GPU), `CASCADIA_STREAM_PARITY_CACHE`
//! (compiled-blob cache dir), `CASCADIA_STREAM_PARITY_DRAFT` (a draft IR dir;
//! `CASCADIA_STREAM_PARITY_SPEC_K`, default 3, its round length) or
//! `CASCADIA_STREAM_PARITY_PROMPT_LOOKUP` (the n-gram size), and
//! `CASCADIA_STREAM_PARITY_TOKENS_PER_DELTA` (default 8).
#![cfg(feature = "openvino")]

use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cascadia_engine::{Builder, Engine};
use cascadia_engine_openvino::OvGenaiBuilder;
use cascadia_types::{Chunk, GenerationTask, PeerLayout, ShardSpec};
use futures::StreamExt;

const MAX_NEW: u32 = 256;
const DIVERGENCE: &str = "ov_genai_stream_divergence";

/// (name, prompt). Each one stresses a different part of the stream state.
const PROMPTS: &[(&str, &str)] = &[
    ("ascii", "Write a short paragraph about the sea."),
    (
        "code",
        "Write a Python function that returns the n-th Fibonacci number, with a docstring, \
         type hints and a short usage example in a code block.",
    ),
    // "Write three sentences about the sea, in Chinese." Multi-byte characters
    // split across tokens.
    (
        "cjk",
        "\u{7528}\u{4e2d}\u{6587}\u{5199}\u{4e09}\u{53e5}\u{8bdd}\u{4ecb}\u{7ecd}\u{5927}\u{6d77}\u{3002}",
    ),
    // Byte-fallback tokenizers spend several tokens on one emoji.
    (
        "emoji",
        "Reply with ten sea-themed emoji separated by spaces, then name each one in a few words.",
    ),
    ("count", "Count from 1 to 100000 in words, separated by commas."),
    // The answer's leading whitespace is held and dropped (the API's
    // trim_start), so the first delta comes only after visible text.
    (
        "leading-space",
        "Start your reply with a blank line and four spaces, then write a haiku about rain.",
    ),
];

/// Every log line of the test binary, from any thread (the engine logs the
/// final chunk's lines on the thread that calls `step()`, but the worker's
/// lines go here too).
fn log_buf() -> &'static Arc<Mutex<Vec<u8>>> {
    static BUF: OnceLock<Arc<Mutex<Vec<u8>>>> = OnceLock::new();
    BUF.get_or_init(|| {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let sink = buf.clone();
        let sub = tracing_subscriber::fmt()
            .with_writer(move || LogSink(sink.clone()))
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .finish();
        tracing::subscriber::set_global_default(sub).expect("one test subscriber");
        buf
    })
}

struct LogSink(Arc<Mutex<Vec<u8>>>);

impl Write for LogSink {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        // Echo, so a --nocapture run shows the engine's own lines.
        let _ = std::io::stderr().write_all(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn log_len() -> usize {
    log_buf().lock().unwrap().len()
}

fn log_since(off: usize) -> String {
    String::from_utf8_lossy(&log_buf().lock().unwrap()[off..]).into_owned()
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn build_engine(
    rt: &tokio::runtime::Runtime,
    model: &str,
    device: &str,
    stream: bool,
) -> Box<dyn Engine> {
    let mut b = OvGenaiBuilder::new(model, device).stream_writes(stream);
    if let Some(dir) = env("CASCADIA_STREAM_PARITY_CACHE") {
        b = b.with_cache_dir(dir);
    }
    if let Some(draft) = env("CASCADIA_STREAM_PARITY_DRAFT") {
        let k = env("CASCADIA_STREAM_PARITY_SPEC_K")
            .map(|v| {
                v.parse()
                    .expect("CASCADIA_STREAM_PARITY_SPEC_K is a number")
            })
            .unwrap_or(3);
        b = b.with_draft(draft, device, k);
    }
    if let Some(n) = env("CASCADIA_STREAM_PARITY_PROMPT_LOOKUP") {
        b = b.with_prompt_lookup(
            n.parse()
                .expect("CASCADIA_STREAM_PARITY_PROMPT_LOOKUP is a number"),
        );
    }
    rt.block_on(async {
        b.connect(PeerLayout::single_stage())
            .await
            .expect("connect");
        let mut load = b
            .load(ShardSpec::single_stage(model, device))
            .await
            .expect("load");
        while let Some(ev) = load.next().await {
            eprintln!("load: {ev:?}");
        }
    });
    let mut engine = Box::new(b).build().expect("build");
    engine.warmup();
    engine
}

/// One generation through `step()`, as the runner drives it.
struct Run {
    /// Non-empty token chunks before the final, with their arrival time.
    deltas: Vec<(Duration, String)>,
    /// The final chunk's text.
    rest: String,
    completion_tokens: u32,
    total: Duration,
    /// The log lines written while it ran.
    log: String,
}

impl Run {
    fn joined(&self) -> String {
        let mut s: String = self.deltas.iter().map(|(_, t)| t.as_str()).collect();
        s.push_str(&self.rest);
        s
    }

    fn first_delta(&self) -> Option<Duration> {
        self.deltas.first().map(|(t, _)| *t)
    }

    fn tok_s(&self) -> f64 {
        let s = self.total.as_secs_f64();
        if s > 0.0 {
            self.completion_tokens as f64 / s
        } else {
            0.0
        }
    }

    /// The engine's `ttft_ms` on this run's `task done` line, if it logged one.
    fn ttft_ms(&self) -> Option<f64> {
        self.log
            .lines()
            .filter(|l| l.contains("task done"))
            .find_map(|l| {
                l.split_whitespace()
                    .find_map(|w| w.strip_prefix("ttft_ms="))
            })
            .and_then(|v| v.parse().ok())
    }
}

fn generate(engine: &mut dyn Engine, id: &str, prompt: &str) -> Run {
    let off = log_len();
    // A streaming client: with `stream_writes(true)` only such a task takes
    // the streamed route (a non-streamed one gets the one-shot answer). The
    // reference engine ignores it.
    let task = GenerationTask::new(id.to_string(), prompt)
        .with_max_tokens(MAX_NEW)
        .with_client_streams(true);
    let started = Instant::now();
    engine.submit(task).expect("submit");
    let mut deltas = Vec::new();
    let fin: Chunk = 'outer: loop {
        for (_, c) in engine.step().expect("step") {
            if let Some(err) = &c.error {
                panic!("{id}: the generation failed: {err}");
            }
            if c.is_final {
                break 'outer c;
            }
            if !c.text.is_empty() {
                deltas.push((started.elapsed(), c.text));
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(600),
            "{id}: no final chunk within 600 s"
        );
    };
    let total = started.elapsed();
    Run {
        deltas,
        completion_tokens: fin.completion_tokens.or(fin.n_tokens).unwrap_or(0),
        rest: fin.text,
        total,
        log: log_since(off),
    }
}

fn ms(d: Option<Duration>) -> String {
    d.map_or("-".into(), |d| format!("{:.1}", d.as_secs_f64() * 1e3))
}

/// The first differing char index and a little context on both sides.
fn diff(a: &str, b: &str) -> String {
    let (ac, bc): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let i = ac.iter().zip(&bc).take_while(|(x, y)| x == y).count();
    let ctx = |v: &[char]| -> String {
        v[i.saturating_sub(20)..(i + 40).min(v.len())]
            .iter()
            .collect()
    };
    format!(
        "first difference at char {i}: streamed {:?} vs non-streamed {:?} ({} vs {} bytes)",
        ctx(&ac),
        ctx(&bc),
        a.len(),
        b.len()
    )
}

#[test]
fn streamed_deltas_join_to_the_non_streamed_answer() {
    let Some(model) = env("CASCADIA_STREAM_PARITY_MODEL") else {
        eprintln!("CASCADIA_STREAM_PARITY_MODEL not set; skipping");
        return;
    };
    let device = env("CASCADIA_STREAM_PARITY_DEVICE").unwrap_or_else(|| "GPU".into());
    let per: u32 = env("CASCADIA_STREAM_PARITY_TOKENS_PER_DELTA")
        .map(|v| {
            v.parse()
                .expect("CASCADIA_STREAM_PARITY_TOKENS_PER_DELTA is a number")
        })
        .unwrap_or(8);
    log_buf();
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    eprintln!(
        "model {model} device {device}, {MAX_NEW} new tokens, greedy, <= {per} tokens per delta"
    );

    // References first, then the stream, so only one pipeline is resident.
    let mut refs = Vec::new();
    {
        let mut engine = build_engine(&rt, &model, &device, false);
        for (name, prompt) in PROMPTS {
            let a = generate(engine.as_mut(), &format!("ref-{name}"), prompt);
            let b = generate(engine.as_mut(), &format!("ref2-{name}"), prompt);
            assert!(
                a.deltas.is_empty(),
                "{name}: stream_writes(false) sent deltas"
            );
            eprintln!(
                "reference {name:<14} {:>4} tokens total {:>7.3} s {:>6.2} tok/s ttft_ms {}",
                a.completion_tokens,
                a.total.as_secs_f64(),
                a.tok_s(),
                a.ttft_ms().map_or("-".into(), |v| format!("{v:.1}")),
            );
            refs.push((a, b));
        }
    }

    let mut engine = build_engine(&rt, &model, &device, true);
    let mut failures = Vec::new();
    let mut unverifiable = 0;
    for ((name, prompt), (ref_a, ref_b)) in PROMPTS.iter().zip(&refs) {
        let s = generate(engine.as_mut(), &format!("stream-{name}"), prompt);
        let joined = s.joined();
        eprintln!(
            "stream    {name:<14} {:>4} tokens {:>4} deltas first delta {:>7} ms total {:>7.3} s {:>6.2} tok/s ttft_ms {}",
            s.completion_tokens,
            s.deltas.len(),
            ms(s.first_delta()),
            s.total.as_secs_f64(),
            s.tok_s(),
            s.ttft_ms().map_or("-".into(), |v| format!("{v:.1}")),
        );
        eprintln!(
            "          {name:<14} text {:?}",
            joined.chars().take(120).collect::<String>()
        );

        // The stream-on run really streamed, so the comparison below is not
        // one-shot against one-shot.
        assert!(
            s.completion_tokens <= 1 || s.deltas.len() > 1,
            "{name}: stream_writes(true) sent {} content chunks before the final for {} tokens; \
             the task took the one-shot route",
            s.deltas.len(),
            s.completion_tokens
        );
        let (ra, rb) = (ref_a.joined(), ref_b.joined());
        if joined != ra && joined != rb {
            if ra == rb {
                failures.push(format!("{name}: {}", diff(&joined, &ra)));
            } else {
                unverifiable += 1;
                eprintln!(
                    "UNVERIFIABLE {name}: the two non-streamed answers differ ({}) and the stream equals neither",
                    diff(&ra, &rb)
                );
            }
        } else if joined != ra || ra != rb {
            eprintln!(
                "note {name}: the two non-streamed answers differ; the stream equals one of them"
            );
        }
        if s.log.contains(DIVERGENCE) {
            let line = s
                .log
                .lines()
                .find(|l| l.contains(DIVERGENCE))
                .unwrap_or_default();
            failures.push(format!("{name}: the engine logged a divergence: {line}"));
        }
        if s.completion_tokens == 0 {
            failures.push(format!("{name}: the final chunk has no completion_tokens"));
        } else if (s.deltas.len() as u32) < s.completion_tokens / per {
            failures.push(format!(
                "{name}: {} deltas for {} tokens, fewer than one per {per}",
                s.deltas.len(),
                s.completion_tokens
            ));
        }
        if s.completion_tokens != ref_a.completion_tokens && joined == ra {
            failures.push(format!(
                "{name}: completion_tokens {} streamed vs {} non-streamed for the same text",
                s.completion_tokens, ref_a.completion_tokens
            ));
        }
    }
    assert!(
        unverifiable < PROMPTS.len(),
        "every prompt's non-streamed answer varied between two calls; nothing was compared"
    );
    assert!(
        failures.is_empty(),
        "stream parity failures:\n{}",
        failures.join("\n")
    );
}
