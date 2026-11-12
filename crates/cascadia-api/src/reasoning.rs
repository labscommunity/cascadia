//! Splitting a generation into reasoning and answer.
//!
//! A thinking model writes its reasoning inside `<think>…</think>` before the
//! answer. [`ReasoningSplit`] routes the text inside the tags to reasoning and
//! the rest to the answer, chunk by chunk, so the chat completions and the
//! responses endpoints, streamed or not, split a reply the same way. It starts
//! inside reasoning when the generation prompt already opened `<think>`;
//! otherwise it holds leading whitespace and any prefix of `<think>` until the
//! first non-whitespace text decides. A reply that never opens `<think>` is
//! released unchanged, so its text stays byte-identical to a reply with
//! thinking off.
//!
//! [`ReasoningCounter`] drives a split and attributes each chunk's tokens to
//! reasoning or answer, for `reasoning_tokens` in the usage block.
//! [`ReasoningFormat`] selects whether replies carry reasoning separately or
//! leave `<think>` in the answer text.

use std::fmt;
use std::str::FromStr;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// A run of text routed to one side of the split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seg {
    Reasoning(String),
    Answer(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Thinking off: everything is answer.
    Off,
    /// Thinking on, nothing but whitespace or a prefix of `<think>` seen yet.
    Undecided,
    /// Inside `<think>`.
    Reasoning,
    /// Past `</think>`, or the reply never opened `<think>`.
    Answer,
}

/// Splits generated text into reasoning and answer.
#[derive(Debug)]
pub struct ReasoningSplit {
    state: State,
    /// Undecided: the whitespace and `<think>` prefix seen so far.
    /// Reasoning: a trailing prefix of `</think>` split across feeds.
    held: String,
    /// The answer's leading whitespace is still being dropped (right after
    /// `</think>`).
    trim_answer: bool,
    consumed_close: bool,
}

impl ReasoningSplit {
    /// `thinking`: the request has thinking on. `opened`: the generation
    /// prompt already opened `<think>`, so the model starts inside reasoning.
    pub fn new(thinking: bool, opened: bool) -> Self {
        let state = match (thinking, opened) {
            (false, _) => State::Off,
            (true, true) => State::Reasoning,
            (true, false) => State::Undecided,
        };
        Self {
            state,
            held: String::new(),
            trim_answer: false,
            consumed_close: false,
        }
    }

    /// Route one chunk of generated text.
    pub fn feed(&mut self, text: &str) -> Vec<Seg> {
        let mut out = Vec::new();
        let mut rest = text.to_owned();
        loop {
            match self.state {
                State::Off => {
                    push(&mut out, Seg::Answer, rest);
                    return out;
                }
                State::Undecided => {
                    let buf = std::mem::take(&mut self.held) + &rest;
                    let lead = buf.len() - buf.trim_start().len();
                    let body = &buf[lead..];
                    if let Some(after) = body.strip_prefix(OPEN) {
                        // The model opened reasoning: the tag and the
                        // whitespace before it are not text.
                        self.state = State::Reasoning;
                        rest = after.to_owned();
                    } else if OPEN.starts_with(body) {
                        // Whitespace only, or a prefix of `<think>`: wait.
                        self.held = buf;
                        return out;
                    } else {
                        // No reasoning: everything held goes out unchanged.
                        self.state = State::Answer;
                        push(&mut out, Seg::Answer, buf);
                        return out;
                    }
                }
                State::Reasoning => {
                    let buf = std::mem::take(&mut self.held) + &rest;
                    if let Some(at) = buf.find(CLOSE) {
                        push(&mut out, Seg::Reasoning, buf[..at].to_owned());
                        self.consumed_close = true;
                        self.trim_answer = true;
                        self.state = State::Answer;
                        rest = buf[at + CLOSE.len()..].to_owned();
                    } else {
                        let keep = partial_suffix(&buf, CLOSE);
                        self.held = buf[buf.len() - keep..].to_owned();
                        push(&mut out, Seg::Reasoning, buf[..buf.len() - keep].to_owned());
                        return out;
                    }
                }
                State::Answer => {
                    if self.trim_answer {
                        let trimmed = rest.trim_start();
                        if trimmed.is_empty() {
                            return out;
                        }
                        self.trim_answer = false;
                        rest = trimmed.to_owned();
                    }
                    push(&mut out, Seg::Answer, rest);
                    return out;
                }
            }
        }
    }

    /// End of the generation: release whatever is held (an undecided prefix
    /// is answer text; a partial `</think>` inside reasoning is reasoning).
    /// Final: the state is left as it was, so no [`Self::feed`] may follow.
    pub fn finish(&mut self) -> Vec<Seg> {
        let held = std::mem::take(&mut self.held);
        let mut out = Vec::new();
        match self.state {
            State::Undecided => push(&mut out, Seg::Answer, held),
            State::Reasoning => push(&mut out, Seg::Reasoning, held),
            State::Off | State::Answer => {}
        }
        out
    }

    /// The split is inside `<think>`.
    pub fn in_reasoning(&self) -> bool {
        self.state == State::Reasoning
    }

    /// A `</think>` was consumed (even around empty reasoning).
    pub fn consumed_close(&self) -> bool {
        self.consumed_close
    }

    /// Bytes fed but not yet released or consumed.
    pub fn held_len(&self) -> usize {
        self.held.len()
    }
}

/// Append a non-empty run of text as `kind`.
fn push(out: &mut Vec<Seg>, kind: fn(String) -> Seg, text: String) {
    if !text.is_empty() {
        out.push(kind(text));
    }
}

/// Length of the longest proper prefix of `tag` that `buf` ends with.
fn partial_suffix(buf: &str, tag: &str) -> usize {
    (1..tag.len().min(buf.len() + 1))
        .rev()
        .find(|&k| buf.is_char_boundary(buf.len() - k) && tag.starts_with(&buf[buf.len() - k..]))
        .unwrap_or(0)
}

/// Splits generated text into reasoning and answer and counts the tokens
/// that went to reasoning.
#[derive(Debug)]
pub struct ReasoningCounter {
    split: ReasoningSplit,
    armed: bool,
    /// Bytes the reasoning split is holding.
    split_held: usize,
    /// Tokens of feeds that released nothing yet.
    carried: u32,
    /// Reasoning tokens by the per-chunk counts.
    reasoning: f64,
    /// The per-chunk counts' sum.
    observed: u32,
    /// A feed released bytes without a token count (its own or carried
    /// from a held feed), so the per-chunk counts miss part of the reply.
    uncounted: bool,
    /// Bytes released or consumed, and the reasoning share of them
    /// (`<think>`/`</think>` included).
    out_bytes: usize,
    reasoning_bytes: usize,
}

impl ReasoningCounter {
    /// `thinking`: the split is armed. `opened`: the generation prompt
    /// already opened `<think>`, so the model starts inside reasoning.
    pub fn new(thinking: bool, opened: bool) -> Self {
        Self {
            split: ReasoningSplit::new(thinking, opened),
            armed: thinking,
            split_held: 0,
            carried: 0,
            reasoning: 0.0,
            observed: 0,
            uncounted: false,
            out_bytes: 0,
            reasoning_bytes: 0,
        }
    }

    /// One chunk of generated text and the tokens it carried: the
    /// `(reasoning, answer)` text it released.
    pub fn feed(&mut self, text: &str, tokens: u32) -> (String, String) {
        let segs = self.split.feed(text);
        self.count(text.len(), &segs, tokens);
        join(segs)
    }

    /// End of the generation: the `(reasoning, answer)` text still held.
    /// Final: no [`Self::feed`] may follow.
    pub fn finish(&mut self) -> (String, String) {
        let segs = self.split.finish();
        self.count(0, &segs, 0);
        // Tokens of trailing feeds that released no text (an end-of-sequence
        // token, text a marker filter removed) belong to the side the reply
        // ended on.
        let carried = std::mem::take(&mut self.carried);
        if self.split.in_reasoning() {
            self.reasoning += f64::from(carried);
        }
        join(segs)
    }

    /// The split is on (thinking on): reasoning is separated and counted.
    pub fn armed(&self) -> bool {
        self.armed
    }

    /// The split is inside `<think>`.
    pub fn in_reasoning(&self) -> bool {
        self.split.in_reasoning()
    }

    /// A `</think>` was consumed (even around empty reasoning).
    pub fn consumed_close(&self) -> bool {
        self.split.consumed_close()
    }

    /// `completion_tokens_details.reasoning_tokens` of a `completion_tokens`
    /// total, clamped to it. A total other than the per-chunk counts' sum
    /// (an engine's authoritative final count) scales the counted reasoning
    /// to it; when any feed released text without a count, the reasoning
    /// share of all released bytes splits it.
    pub fn reasoning_tokens(&self, completion_tokens: u32) -> u32 {
        let total = f64::from(completion_tokens);
        let reasoning = if self.uncounted || self.observed == 0 {
            if self.out_bytes > 0 {
                total * self.reasoning_bytes as f64 / self.out_bytes as f64
            } else {
                0.0
            }
        } else if self.observed == completion_tokens {
            self.reasoning
        } else {
            self.reasoning * total / f64::from(self.observed)
        };
        (reasoning.round() as u32).min(completion_tokens)
    }

    /// Split a feed's tokens by the share of the bytes it released or
    /// consumed that went to reasoning (`<think>`/`</think>` included); a
    /// feed that only held text carries its tokens to the next one. The
    /// end of the generation (`fed == 0`) releases text whose tokens its
    /// feed already counted.
    fn count(&mut self, fed: usize, segs: &[Seg], tokens: u32) {
        self.observed = self.observed.saturating_add(tokens);
        let pool = self.split_held + fed;
        self.split_held = self.split.held_len();
        let out = pool - self.split_held;
        let tokens = self.carried.saturating_add(tokens);
        if out == 0 {
            self.carried = tokens;
            return;
        }
        self.carried = 0;
        if tokens == 0 && fed > 0 {
            self.uncounted = true;
        }
        let answer: usize = segs
            .iter()
            .map(|s| match s {
                Seg::Answer(t) => t.len(),
                Seg::Reasoning(_) => 0,
            })
            .sum();
        self.out_bytes += out;
        self.reasoning_bytes += out - answer;
        self.reasoning += f64::from(tokens) * (out - answer) as f64 / out as f64;
    }
}

/// Concatenate segments into `(reasoning, answer)`.
fn join(segs: Vec<Seg>) -> (String, String) {
    let (mut reasoning, mut answer) = (String::new(), String::new());
    for seg in segs {
        match seg {
            Seg::Reasoning(t) => reasoning.push_str(&t),
            Seg::Answer(t) => answer.push_str(&t),
        }
    }
    (reasoning, answer)
}

/// How a reply carries the model's reasoning (`--reasoning-format`, with
/// llama-server's names).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// `deepseek`: reasoning leaves the answer and is returned as
    /// `reasoning_content`.
    #[default]
    ReasoningContent,
    /// `none`: no split; `<think>…</think>` stays in the answer text.
    None,
}

impl FromStr for ReasoningFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "deepseek" => Ok(Self::ReasoningContent),
            "none" => Ok(Self::None),
            other => Err(format!(
                "unknown reasoning format `{other}` (expected `deepseek` or `none`)"
            )),
        }
    }
}

impl fmt::Display for ReasoningFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ReasoningContent => "deepseek",
            Self::None => "none",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `chunks` then finish; merge adjacent segments of one kind.
    fn run(thinking: bool, opened: bool, chunks: &[&str]) -> (Vec<Seg>, ReasoningSplit) {
        let mut s = ReasoningSplit::new(thinking, opened);
        let mut segs = Vec::new();
        for c in chunks {
            segs.extend(s.feed(c));
        }
        segs.extend(s.finish());
        (merge(segs), s)
    }

    fn merge(segs: Vec<Seg>) -> Vec<Seg> {
        let mut out: Vec<Seg> = Vec::new();
        for seg in segs {
            match (out.last_mut(), seg) {
                (Some(Seg::Reasoning(a)), Seg::Reasoning(b)) => a.push_str(&b),
                (Some(Seg::Answer(a)), Seg::Answer(b)) => a.push_str(&b),
                (_, seg) => out.push(seg),
            }
        }
        out
    }

    fn r(s: &str) -> Seg {
        Seg::Reasoning(s.into())
    }

    fn a(s: &str) -> Seg {
        Seg::Answer(s.into())
    }

    #[test]
    fn thinking_off_is_all_answer_with_think_verbatim() {
        let (segs, s) = run(false, false, &["<think>x", "</think>y"]);
        assert_eq!(segs, vec![a("<think>x</think>y")]);
        assert!(!s.consumed_close());
        // even when the prompt opened it: thinking off means no split
        let (segs, _) = run(false, true, &["r</think>y"]);
        assert_eq!(segs, vec![a("r</think>y")]);
    }

    #[test]
    fn prompt_opened_splits_at_a_close_split_across_feeds() {
        let mut s = ReasoningSplit::new(true, true);
        assert!(s.in_reasoning());
        let mut segs = Vec::new();
        for c in ["ab", "c</th", "ink>\n", "\nhi"] {
            segs.extend(s.feed(c));
        }
        assert!(!s.in_reasoning());
        assert!(s.consumed_close());
        segs.extend(s.finish());
        assert_eq!(merge(segs), vec![r("abc"), a("hi")]);
    }

    #[test]
    fn partial_close_is_held_not_released_as_reasoning() {
        let mut s = ReasoningSplit::new(true, true);
        assert_eq!(merge(s.feed("ab</th")), vec![r("ab")]);
        assert!(s.in_reasoning());
    }

    #[test]
    fn output_opened_think_after_whitespace() {
        let (segs, s) = run(true, false, &["  <th", "ink>r</think>x"]);
        assert_eq!(segs, vec![r("r"), a("x")]);
        assert!(s.consumed_close());
    }

    #[test]
    fn undecided_resolves_on_the_first_non_whitespace_text() {
        let mut s = ReasoningSplit::new(true, false);
        assert!(!s.in_reasoning());
        assert_eq!(s.feed("hello"), vec![a("hello")]);
        assert_eq!(s.feed(" <think>"), vec![a(" <think>")]);
        assert!(s.finish().is_empty());
        assert!(!s.consumed_close());
    }

    #[test]
    fn a_longer_tag_is_answer() {
        let (segs, s) = run(true, false, &["<thinking"]);
        assert_eq!(segs, vec![a("<thinking")]);
        assert!(!s.in_reasoning());
    }

    #[test]
    fn unclosed_reasoning_flushes_as_reasoning() {
        let mut s = ReasoningSplit::new(true, false);
        let mut segs = s.feed("<think>r");
        assert!(s.in_reasoning());
        segs.extend(s.finish());
        assert_eq!(merge(segs), vec![r("r")]);
        // a held partial close is reasoning at the end
        let (segs, _) = run(true, true, &["r</thi"]);
        assert_eq!(segs, vec![r("r</thi")]);
    }

    #[test]
    fn whitespace_only_flushes_as_answer() {
        let mut s = ReasoningSplit::new(true, false);
        assert!(s.feed("  ").is_empty());
        assert_eq!(s.finish(), vec![a("  ")]);
        // a held `<think>` prefix is answer at the end too
        let (segs, _) = run(true, false, &["\n<thi"]);
        assert_eq!(segs, vec![a("\n<thi")]);
    }

    #[test]
    fn non_thinking_reply_is_byte_identical() {
        let (segs, _) = run(true, false, &["\n", "\nhello"]);
        assert_eq!(segs, vec![a("\n\nhello")]);
    }

    #[test]
    fn empty_reasoning_still_consumes_the_close() {
        let (segs, s) = run(true, false, &["<think></think>"]);
        assert!(segs.is_empty(), "{segs:?}");
        assert!(s.consumed_close());
        let (segs, s) = run(true, true, &["</think>\n\nanswer"]);
        assert_eq!(segs, vec![a("answer")]);
        assert!(s.consumed_close());
    }

    #[test]
    fn answer_whitespace_dropped_only_at_its_start() {
        let (segs, _) = run(true, true, &["r</think>", " ", "\n", "a b", " c"]);
        assert_eq!(segs, vec![r("r"), a("a b c")]);
    }

    #[test]
    fn multibyte_text_around_the_tags() {
        let (segs, _) = run(true, false, &["<think>é</", "think>ü"]);
        assert_eq!(segs, vec![r("é"), a("ü")]);
    }

    // ---- ReasoningCounter ----

    #[test]
    fn counter_splits_and_counts_a_streamed_think_block() {
        let mut c = ReasoningCounter::new(true, false);
        // reasoning bytes are kept verbatim
        assert_eq!(c.feed("<think> plan ", 2), (" plan ".into(), String::new()));
        assert_eq!(c.feed("more </think> ", 2), ("more ".into(), String::new()));
        assert_eq!(c.feed("answer", 1), (String::new(), "answer".into()));
        assert_eq!(c.finish(), (String::new(), String::new()));
        assert_eq!(c.reasoning_tokens(5), 4);
        assert!(c.armed());
    }

    #[test]
    fn counter_off_passes_everything_and_counts_nothing() {
        let mut c = ReasoningCounter::new(false, false);
        assert_eq!(
            c.feed("<think>x</think>y", 3),
            (String::new(), "<think>x</think>y".into())
        );
        assert!(!c.armed());
        assert_eq!(c.reasoning_tokens(3), 0);
    }

    #[test]
    fn counter_opened_prompt_starts_in_reasoning_and_byte_share_when_uncounted() {
        let mut c = ReasoningCounter::new(true, true);
        let (r, a) = c.feed("abcd</think>efgh", 0);
        assert_eq!((r.as_str(), a.as_str()), ("abcd", "efgh"));
        // 12 of 16 bytes are reasoning (tag included): 12/16 of 8.
        assert_eq!(c.reasoning_tokens(8), 6);
    }

    #[test]
    fn counter_length_cut_inside_reasoning_keeps_it_all_reasoning() {
        let mut c = ReasoningCounter::new(true, false);
        let _ = c.feed("<think>never closes </thi", 3);
        assert_eq!(c.finish(), ("</thi".into(), String::new()));
        assert_eq!(c.reasoning_tokens(3), 3);
    }

    #[test]
    fn reasoning_format_parses_llama_server_names() {
        assert_eq!(
            "deepseek".parse::<ReasoningFormat>().unwrap(),
            ReasoningFormat::ReasoningContent
        );
        assert_eq!(
            "none".parse::<ReasoningFormat>().unwrap(),
            ReasoningFormat::None
        );
        assert!("auto2".parse::<ReasoningFormat>().is_err());
    }

    /// Chunks that carried no tokens, then an authoritative total: the
    /// reasoning share of every released byte (tags included) splits it.
    #[test]
    fn counter_uncounted_chunks_split_the_total_by_bytes() {
        let mut c = ReasoningCounter::new(true, false);
        c.feed("<think>plan</think>", 0);
        c.feed("answer", 0);
        c.finish();
        // 19 of 25 bytes are reasoning: 7.6 → 8
        assert_eq!(c.reasoning_tokens(10), 8);
        assert_eq!(c.reasoning_tokens(0), 0);
        // nothing released at all: none
        let mut c = ReasoningCounter::new(true, false);
        c.finish();
        assert_eq!(c.reasoning_tokens(3), 0);
    }

    /// Per-chunk counts that disagree with the authoritative total are
    /// scaled to it.
    #[test]
    fn counter_counted_chunks_are_scaled_to_the_total() {
        let mut c = ReasoningCounter::new(true, false);
        // 1 token all reasoning, then 1 token 8 of 10 bytes reasoning: 1.8
        c.feed("<think>ab", 1);
        c.feed("</think>xy", 1);
        c.finish();
        assert_eq!(c.reasoning_tokens(2), 2);
        // ×5: 9
        assert_eq!(c.reasoning_tokens(10), 9);
        // ×0.5: 0.9 → 1
        assert_eq!(c.reasoning_tokens(1), 1);
    }

    #[test]
    fn counter_held_tokens_are_carried_to_the_next_release() {
        // the `<th` feed holds everything; its token goes with the next feed
        let mut c = ReasoningCounter::new(true, false);
        for (text, tokens) in [("<th", 1), ("ink>ab", 1), ("</think>x", 1)] {
            c.feed(text, tokens);
        }
        c.finish();
        // feed 2: 3 + 4 bytes all reasoning, 2 tokens; feed 3: 8 of 9 bytes
        // of 1 token: 2.89 → 3
        assert_eq!(c.reasoning_tokens(3), 3);
        // finish attributes what is still carried
        let mut c = ReasoningCounter::new(true, false);
        c.feed("\n", 4);
        c.finish();
        assert_eq!(c.reasoning_tokens(4), 0);
        let mut c = ReasoningCounter::new(true, true);
        c.feed("ab</thi", 4);
        c.finish();
        assert_eq!(c.reasoning_tokens(4), 4);
    }

    /// A last chunk that carries tokens but no text (an end-of-sequence
    /// token, or text a marker filter removed) counts on the side the
    /// reply ended on.
    #[test]
    fn counter_tokens_of_a_final_empty_chunk_count_where_the_reply_ended() {
        let mut c = ReasoningCounter::new(true, true);
        c.feed("abc", 1);
        c.feed("", 1);
        c.finish();
        assert_eq!(c.reasoning_tokens(2), 2);
        let mut c = ReasoningCounter::new(true, false);
        c.feed("<think>r</think>", 1);
        c.feed("answer", 1);
        c.feed("", 1);
        c.finish();
        assert_eq!(c.reasoning_tokens(3), 1);
    }

    /// A chunk that released text without a token count leaves the
    /// per-chunk counts incomplete: the byte share of the whole reply
    /// splits the total instead.
    #[test]
    fn counter_an_uncounted_chunk_switches_to_the_byte_share() {
        let mut c = ReasoningCounter::new(true, false);
        c.feed("<think>R</think>", 1);
        c.feed("long answer text", 0);
        c.finish();
        // 16 of 32 bytes are reasoning
        assert_eq!(c.reasoning_tokens(100), 50);
    }

    /// Every chunk that released text reported tokens: the per-chunk
    /// counts decide, scaled to the total.
    #[test]
    fn counter_all_counted_chunks_use_the_per_chunk_counts() {
        let mut c = ReasoningCounter::new(true, false);
        c.feed("<think>R</think>", 1);
        c.feed("long answer text", 3);
        c.finish();
        assert_eq!(c.reasoning_tokens(4), 1);
        assert_eq!(c.reasoning_tokens(8), 2);
    }
}
