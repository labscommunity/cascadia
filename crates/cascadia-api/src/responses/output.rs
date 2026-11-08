//! The Responses output path: splitting a generation into reasoning and answer.
//!
//! [`ReasoningSplit`] is stage 1 of the output builder: with thinking on it
//! routes the text inside `<think>…</think>` to reasoning and the rest to the
//! answer. It starts inside reasoning when the generation prompt already
//! opened `<think>` (see [`crate::opens_think`]); otherwise it holds leading
//! whitespace and any prefix of `<think>` until the first non-whitespace text
//! decides. A reply that never opens `<think>` is released unchanged, so its
//! text stays byte-identical to a plain chat reply.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// A run of text routed to one side of the split.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum Seg {
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

/// Splits generated text into reasoning and answer (spec "Output", stage 1).
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct ReasoningSplit {
    state: State,
    /// Undecided: the whitespace and `<think>` prefix seen so far.
    /// Reasoning: a trailing prefix of `</think>` split across feeds.
    held: String,
    /// The answer's leading whitespace is still being dropped (right after
    /// `</think>`).
    trim_answer: bool,
    consumed_close: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
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
}

/// Append a non-empty run of text as `kind`.
#[cfg_attr(not(test), allow(dead_code))]
fn push(out: &mut Vec<Seg>, kind: fn(String) -> Seg, text: String) {
    if !text.is_empty() {
        out.push(kind(text));
    }
}

/// Length of the longest proper prefix of `tag` that `buf` ends with.
#[cfg_attr(not(test), allow(dead_code))]
fn partial_suffix(buf: &str, tag: &str) -> usize {
    (1..tag.len().min(buf.len() + 1))
        .rev()
        .find(|&k| buf.is_char_boundary(buf.len() - k) && tag.starts_with(&buf[buf.len() - k..]))
        .unwrap_or(0)
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
}
