//! The text a streaming stateful ov-genai task hands out as it generates.
//!
//! The shim gives every streamer write the raw decode of all the ids generated so far (the engine
//! never confirms a window, so each one starts at the answer's first id). [`StreamText`] turns
//! those decodes into deltas whose concatenation, plus [`Finish::rest`], is exactly the visible
//! answer the non-streamed path returns:
//!
//! - **Visible rule.** The returned text is the decode with the prompt stripped and then
//!   `trim_start`ed when it starts with the prompt (an empty prompt counts), else the decode as is.
//!   While the decode is still a prefix of the prompt that choice is open, so nothing is handed out;
//!   once it is made, each decode goes through the same rule, and leading whitespace after a
//!   stripped echo is held simply because the trim removes it.
//! - **UTF-8.** A trailing incomplete UTF-8 sequence is held as raw bytes (never decoded lossily),
//!   and [`advance_emitted`] holds a trailing run of U+FFFD, which a detokenizer in replace mode
//!   gives for a character the next id completes.
//! - **Rewrites.** A detokenizer is not prefix-stable: a later decode may not start with what was
//!   handed out. [`advance_emitted`] then re-anchors and hands out nothing for that write;
//!   [`Finish::diverged`] reports it, and the final adds nothing.
//! - **Tokens.** Each write's token count is carried until a delta goes out, so the deltas' counts
//!   add up to the tokens generated even when some writes were held.

use cascadia_ov_genai_shim::{advance_emitted, PollWrite};

/// The per-task stream state (see the module docs).
#[derive(Debug)]
pub(crate) struct StreamText {
    prompt: String,
    /// Whether the decode has stopped being a prefix of the prompt, so the visible rule applies.
    decided: bool,
    /// The visible bytes handed out (after a rewrite: the re-anchored decode).
    emitted: Vec<u8>,
    /// Tokens of writes whose text has not gone out yet.
    carried_tokens: usize,
    /// A decode stopped extending what was handed out.
    diverged: bool,
}

/// What is left to send when the generation ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Finish {
    /// The visible answer after what was handed out; empty when the stream diverged.
    pub rest: String,
    /// The handed-out text is not a prefix of the visible answer (a rewrite mid-stream, or a final
    /// decode that disagrees with the window decodes).
    pub diverged: bool,
    /// Tokens of writes whose text never went out on a delta.
    pub carried_tokens: usize,
    /// Bytes handed out, for the divergence log.
    pub emitted_bytes: usize,
}

/// The answer the stateful engine returns for a decode `raw` of the generated ids: the prompt
/// stripped and the rest `trim_start`ed when `raw` starts with the prompt, else `raw` as is.
pub(crate) fn visible_text<'a>(raw: &'a str, prompt: &str) -> &'a str {
    match raw.strip_prefix(prompt) {
        Some(stripped) => stripped.trim_start(),
        None => raw,
    }
}

/// The length of a trailing UTF-8 sequence in `bytes` that the next bytes may still complete
/// (0 when the bytes end on a complete character or on bytes no continuation can fix).
fn incomplete_tail(bytes: &[u8]) -> usize {
    for back in 1..=bytes.len().min(3) {
        let lead = bytes[bytes.len() - back];
        if lead & 0xC0 == 0x80 {
            continue;
        }
        let need = match lead {
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => 1,
        };
        if need <= back {
            return 0;
        }
        return match std::str::from_utf8(&bytes[bytes.len() - back..]) {
            Err(e) if e.valid_up_to() == 0 && e.error_len().is_none() => back,
            _ => 0,
        };
    }
    0
}

impl StreamText {
    pub(crate) fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            decided: false,
            emitted: Vec::new(),
            carried_tokens: 0,
            diverged: false,
        }
    }

    /// Take one streamer write. Returns the text to hand out and the tokens it stands for (this
    /// write's plus any carried), or `None` when the write's text is held.
    pub(crate) fn on_write(&mut self, w: &PollWrite<'_>) -> Option<(String, usize)> {
        debug_assert_eq!(w.window_from, 0, "the engine never confirms a window");
        self.carried_tokens += w.n_tokens;
        if w.n_tokens == 0 {
            // The end call: no window.
            return None;
        }
        let cut = w.window.len() - incomplete_tail(w.window);
        let text = String::from_utf8_lossy(&w.window[..cut]);
        if !self.decided {
            // Still possibly an echo of the prompt, or a character the next id may make one.
            if self.prompt.starts_with(text.as_ref()) || text.ends_with('\u{FFFD}') {
                return None;
            }
            self.decided = true;
        }
        let visible = visible_text(&text, &self.prompt);
        let (delta, diverged) = advance_emitted(&mut self.emitted, visible.as_bytes(), true);
        self.diverged |= diverged;
        if delta.is_empty() {
            return None;
        }
        Some((delta, std::mem::take(&mut self.carried_tokens)))
    }

    /// The generation ended and `visible` is the answer it returns: what is left to send.
    pub(crate) fn finish(&mut self, visible: &str) -> Finish {
        let carried_tokens = std::mem::take(&mut self.carried_tokens);
        let emitted_bytes = self.emitted.len();
        if self.diverged || !visible.as_bytes().starts_with(&self.emitted) {
            self.diverged = true;
            return Finish {
                rest: String::new(),
                diverged: true,
                carried_tokens,
                emitted_bytes,
            };
        }
        let rest = String::from_utf8_lossy(&visible.as_bytes()[self.emitted.len()..]).into_owned();
        self.emitted = visible.as_bytes().to_vec();
        Finish {
            rest,
            diverged: false,
            carried_tokens,
            emitted_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One write of `window` (the whole decode so far) carrying `n` tokens.
    fn write(window: &[u8], n: usize) -> PollWrite<'_> {
        PollWrite {
            tail: "",
            window,
            window_from: 0,
            n_tokens: n,
        }
    }

    /// Feed the cumulative decodes of `tokens` (one write per entry, one token each) and finish on
    /// their whole decode: (each delta with its tokens, the finish).
    fn run(prompt: &str, tokens: &[&[u8]]) -> (Vec<(String, usize)>, Finish, String) {
        let mut s = StreamText::new(prompt);
        let mut window = Vec::new();
        let mut sent = Vec::new();
        for t in tokens {
            window.extend_from_slice(t);
            if let Some(d) = s.on_write(&write(&window, 1)) {
                sent.push(d);
            }
        }
        assert_eq!(
            s.on_write(&write(b"", 0)),
            None,
            "the end call hands out nothing"
        );
        let raw = String::from_utf8_lossy(&window).into_owned();
        let visible = visible_text(&raw, prompt).to_string();
        let fin = s.finish(&visible);
        (sent, fin, visible)
    }

    /// Today's rule: what was sent, plus the rest, is the visible answer, and the tokens add up.
    fn assert_parity(sent: &[(String, usize)], fin: &Finish, visible: &str, tokens: usize) {
        let mut all: String = sent.iter().map(|(t, _)| t.as_str()).collect();
        all.push_str(&fin.rest);
        assert_eq!(all, visible);
        assert!(!fin.diverged);
        let n: usize = sent.iter().map(|(_, n)| n).sum::<usize>() + fin.carried_tokens;
        assert_eq!(n, tokens, "every write's tokens are counted once");
    }

    fn texts(sent: &[(String, usize)]) -> Vec<&str> {
        sent.iter().map(|(t, _)| t.as_str()).collect()
    }

    /// A chat prompt the answer doesn't start with: no strip, no trim, streamed at once.
    #[test]
    fn a_chat_answer_streams_as_written() {
        let toks: [&[u8]; 3] = [b"\n", b" Hi", b" there"];
        let (sent, fin, visible) = run("<|user|>hi<|assistant|>", &toks);
        assert_eq!(visible, "\n Hi there");
        assert_eq!(texts(&sent), ["\n", " Hi", " there"]);
        assert_eq!(fin.rest, "");
        assert_parity(&sent, &fin, &visible, 3);
    }

    /// The model repeats the prompt: the echo and the whitespace after it never go out.
    #[test]
    fn an_echo_is_stripped_and_trimmed() {
        let toks: [&[u8]; 5] = [b"PRO", b"MPT", b"\n", b" Hi", b"!"];
        let (sent, fin, visible) = run("PROMPT", &toks);
        assert_eq!(visible, "Hi!");
        assert_eq!(texts(&sent), ["Hi", "!"]);
        // The held echo and whitespace writes ride on the first delta.
        assert_eq!(sent[0].1, 4);
        assert_parity(&sent, &fin, &visible, 5);
    }

    /// An empty prompt counts as echoed: leading whitespace is trimmed.
    #[test]
    fn an_empty_prompt_trims_leading_whitespace() {
        let toks: [&[u8]; 4] = [b" ", b" ", b"Hi", b" x"];
        let (sent, fin, visible) = run("", &toks);
        assert_eq!(visible, "Hi x");
        assert_eq!(texts(&sent), ["Hi", " x"]);
        assert_eq!(sent[0].1, 3);
        assert_parity(&sent, &fin, &visible, 4);
    }

    /// An answer that starts like the prompt and then leaves it is sent whole, unstripped.
    #[test]
    fn a_partial_echo_is_released_unchanged() {
        let toks: [&[u8]; 3] = [b"Hel", b"p", b" me"];
        let (sent, fin, visible) = run("Hello", &toks);
        assert_eq!(visible, "Help me");
        assert_eq!(texts(&sent), ["Help", " me"]);
        assert_eq!(sent[0].1, 2);
        assert_parity(&sent, &fin, &visible, 3);
    }

    /// An answer that ends while still inside the echo: everything is held, and the final carries
    /// the whole answer (the decode as is: it does not start with the whole prompt).
    #[test]
    fn an_answer_ending_inside_a_long_echo_comes_on_the_final() {
        let words: Vec<String> = (0..40).map(|i| format!("w{i} ")).collect();
        let prompt = words.concat();
        let toks: Vec<&[u8]> = words[..30].iter().map(|w| w.as_bytes()).collect();
        let (sent, fin, visible) = run(&prompt, &toks);
        assert!(sent.is_empty(), "{sent:?}");
        assert_eq!(fin.rest, visible);
        assert_eq!(visible, words[..30].concat());
        assert_parity(&sent, &fin, &visible, 30);
    }

    /// A long echo is held until it ends, then the answer streams.
    #[test]
    fn a_long_echo_is_held_then_the_answer_streams() {
        let words: Vec<String> = (0..40).map(|i| format!("w{i} ")).collect();
        let prompt = words.concat();
        let mut toks: Vec<&[u8]> = words.iter().map(|w| w.as_bytes()).collect();
        toks.extend([b"ans".as_slice(), b"wer".as_slice()]);
        let (sent, fin, visible) = run(&prompt, &toks);
        assert_eq!(visible, "answer");
        assert_eq!(texts(&sent), ["ans", "wer"]);
        assert_parity(&sent, &fin, &visible, 42);
    }

    /// A character split across two writes goes out once, whole.
    #[test]
    fn a_split_character_goes_out_whole() {
        let snow = "\u{2603}".as_bytes(); // 3 bytes
        let toks: [&[u8]; 4] = [b"a", &snow[..1], &snow[1..], b"b"];
        let (sent, fin, visible) = run("<chat>", &toks);
        assert_eq!(visible, "a\u{2603}b");
        assert_eq!(texts(&sent), ["a", "\u{2603}", "b"]);
        assert_eq!(
            sent[1].1, 2,
            "the held write's token rides on the character"
        );
        assert_parity(&sent, &fin, &visible, 4);
    }

    /// A 4-byte character over three writes, and a detokenizer that gives U+FFFD for the part it
    /// cannot decode yet: neither the raw bytes nor the U+FFFD go out early.
    #[test]
    fn a_split_character_or_its_replacement_is_held() {
        let crab = "\u{1F980}".as_bytes(); // 4 bytes
        let toks: [&[u8]; 4] = [b"x", &crab[..2], &crab[2..3], &crab[3..]];
        let (sent, fin, visible) = run("<chat>", &toks);
        assert_eq!(texts(&sent), ["x", "\u{1F980}"]);
        assert_parity(&sent, &fin, &visible, 4);

        // Replace mode: the window shows U+FFFD until the character completes.
        let mut s = StreamText::new("<chat>");
        assert_eq!(s.on_write(&write(b"x", 1)), Some(("x".into(), 1)));
        assert_eq!(s.on_write(&write("x\u{FFFD}".as_bytes(), 1)), None);
        assert_eq!(s.on_write(&write("x\u{FFFD}\u{FFFD}".as_bytes(), 1)), None);
        let done = "x\u{1F980}";
        assert_eq!(
            s.on_write(&write(done.as_bytes(), 1)),
            Some(("\u{1F980}".into(), 3))
        );
        assert_eq!(s.finish(done).rest, "");
    }

    /// A lone invalid byte inside the text reads as U+FFFD, as the returned text does.
    #[test]
    fn an_invalid_byte_inside_the_text_matches_the_lossy_answer() {
        let toks: [&[u8]; 3] = [b"a", b"\xFF", b"b"];
        let (sent, fin, visible) = run("<chat>", &toks);
        assert_eq!(visible, "a\u{FFFD}b");
        assert_parity(&sent, &fin, &visible, 3);
    }

    /// A decode that no longer starts with what went out: that write is held, later writes go on
    /// from the new decode, and the finish reports the divergence and adds nothing.
    #[test]
    fn a_rewrite_is_held_and_reported() {
        let mut s = StreamText::new("<chat>");
        assert_eq!(s.on_write(&write(b"abc", 1)), Some(("abc".into(), 1)));
        assert_eq!(s.on_write(&write(b"abX d", 1)), None);
        assert_eq!(s.on_write(&write(b"abX de", 1)), Some(("e".into(), 2)));
        let fin = s.finish("abX def");
        assert!(fin.diverged);
        assert_eq!(fin.rest, "");
        assert_eq!(fin.emitted_bytes, 6);
    }

    /// A final decode that disagrees with what went out is a divergence too.
    #[test]
    fn a_final_that_disagrees_is_a_divergence() {
        let mut s = StreamText::new("<chat>");
        assert_eq!(s.on_write(&write(b"abc", 1)), Some(("abc".into(), 1)));
        let fin = s.finish("abd");
        assert!(fin.diverged);
        assert_eq!(fin.rest, "");
    }

    /// A speculative round writes several ids at once: its count rides with its text.
    #[test]
    fn a_multi_id_write_carries_its_count() {
        let mut s = StreamText::new("<chat>");
        assert_eq!(
            s.on_write(&write(b"one two three", 3)),
            Some(("one two three".into(), 3))
        );
        assert_eq!(s.on_write(&write(b"one two three\xE2", 4)), None);
        assert_eq!(
            s.on_write(&write("one two three\u{2603} x".as_bytes(), 2)),
            Some(("\u{2603} x".into(), 6))
        );
        let fin = s.finish("one two three\u{2603} x");
        assert_eq!(fin.carried_tokens, 0);
        assert_eq!(fin.rest, "");
    }

    #[test]
    fn incomplete_tail_finds_only_a_completable_sequence() {
        assert_eq!(incomplete_tail(b""), 0);
        assert_eq!(incomplete_tail(b"abc"), 0);
        assert_eq!(incomplete_tail(b"a\xE2"), 1);
        assert_eq!(incomplete_tail(b"a\xE2\x98"), 2);
        assert_eq!(incomplete_tail("a\u{2603}".as_bytes()), 0);
        assert_eq!(incomplete_tail(b"a\xF0\x9F\xA6"), 3);
        // Not completable: a stray continuation byte, an invalid lead.
        assert_eq!(incomplete_tail(b"a\x98"), 0);
        assert_eq!(incomplete_tail(b"a\xFF"), 0);
        assert_eq!(incomplete_tail(b"a\xE2\x28"), 0);
    }
}
