//! A bounded `stop`-string check for a one-shot engine to run while it generates (M18 D6).
//!
//! An engine that runs a whole generation in one `step()` can end it early at a `stop` match
//! instead of generating to `max_tokens` and leaving the cut to the API. This is only an early
//! exit: the API's `StopMatcher` still makes the authoritative cut on the final text, so the engine
//! keeps the stop string in what it returns.
//!
//! The per-token cost is bounded. At most [`MAX_STOPS`] stops of at most [`MAX_STOP_BYTES`] bytes
//! each are checked, against a rolling tail of the last [`TAIL_BYTES`] bytes. Longer stops, and any
//! past the first [`MAX_STOPS`], are left to the API's linear cut at the end.

/// How many stops are checked (OpenAI's maximum).
pub const MAX_STOPS: usize = 4;
/// The longest stop checked, in bytes.
pub const MAX_STOP_BYTES: usize = 256;
/// How much decoded text is kept, in bytes. Twice the longest stop, so every match that ends in the
/// newest piece is still inside it.
pub const TAIL_BYTES: usize = 512;

/// Stop strings checked against the text a one-shot generation has produced so far.
#[derive(Debug, Clone, Default)]
pub struct OneShotStop {
    stops: Vec<String>,
    max_len: usize,
    tail: String,
    matched: Option<usize>,
}

impl OneShotStop {
    /// Keeps the first [`MAX_STOPS`] stops that are non-empty and at most [`MAX_STOP_BYTES`] bytes
    /// long, in the order given; the rest are ignored here.
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops
            .iter()
            .filter(|s| !s.is_empty() && s.len() <= MAX_STOP_BYTES)
            .take(MAX_STOPS)
            .cloned()
            .collect();
        let max_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            max_len,
            tail: String::new(),
            matched: None,
        }
    }

    /// Whether any stop is checked. When `false`, [`push`](Self::push) always returns `false` and a
    /// caller can skip all the work of feeding it.
    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// The longest active stop in bytes (0 when none is active; never more than
    /// [`MAX_STOP_BYTES`]). A caller that decodes its own tail ([`matches_tail`](Self::matches_tail))
    /// sizes it from this.
    pub fn max_stop_bytes(&self) -> usize {
        self.max_len
    }

    /// Adds the next decoded piece. Returns `true` once any active stop occurs in the text pushed so
    /// far, and keeps returning `true` after that. Only the last [`TAIL_BYTES`] bytes are kept; the
    /// work per call is bounded by that tail plus the piece, not by the stop lengths or the total
    /// text.
    pub fn push(&mut self, piece: &str) -> bool {
        if self.matched.is_some() {
            return true;
        }
        if self.stops.is_empty() {
            return false;
        }
        // Every earlier match was already found, so only a match that ends in `piece` is new. It
        // starts at most `max_len - 1` bytes before the piece.
        let from = floor_char_boundary(
            &self.tail,
            self.tail
                .len()
                .saturating_sub(self.max_len.saturating_sub(1)),
        );
        self.tail.push_str(piece);
        self.matched = earliest_match(&self.stops, &self.tail[from..]);
        let cut = ceil_char_boundary(&self.tail, self.tail.len().saturating_sub(TAIL_BYTES));
        self.tail.drain(..cut);
        self.matched.is_some()
    }

    /// The stop that matched first (the one whose occurrence ends earliest in the text; on a tie,
    /// the one given first), once [`push`](Self::push) has returned `true`.
    pub fn matched(&self) -> Option<&str> {
        self.matched.map(|i| self.stops[i].as_str())
    }

    /// Stateless check of a freshly decoded tail of the output, for a caller that re-decodes its
    /// newest ids each time instead of feeding pieces. A match counts only if the whole stop lies
    /// inside `tail`, so the caller must pass a tail that holds every stop ending in its newest text:
    /// at least [`max_stop_bytes`](Self::max_stop_bytes) bytes back from that text, and more if a
    /// token can decode to zero bytes. Nothing is remembered between calls.
    pub fn matches_tail(&self, tail: &str) -> bool {
        self.stops.iter().any(|stop| tail.contains(stop.as_str()))
    }

    #[cfg(test)]
    fn tail_len(&self) -> usize {
        self.tail.len()
    }
}

/// The stop whose first occurrence in `text` ends earliest; on a tie, the one listed first.
fn earliest_match(stops: &[String], text: &str) -> Option<usize> {
    stops
        .iter()
        .enumerate()
        .filter_map(|(i, stop)| text.find(stop.as_str()).map(|at| (at + stop.len(), i)))
        .min()
        .map(|(_, i)| i)
}

/// The largest char boundary of `s` at or below `i`.
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The smallest char boundary of `s` at or above `i`.
fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn stops(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn a_match_inside_one_piece() {
        let mut s = OneShotStop::new(&stops(&["twenty"]));
        assert!(s.is_active());
        assert!(!s.push("one two "));
        assert!(s.push("nineteen twenty twenty-one"));
        assert_eq!(s.matched(), Some("twenty"));
        // Latched.
        assert!(s.push("anything"));
    }

    #[test]
    fn a_match_across_piece_boundaries() {
        let mut s = OneShotStop::new(&stops(&["twenty"]));
        assert!(!s.push("nineteen tw"));
        assert!(!s.push("en"));
        assert!(s.push("ty"));
        assert_eq!(s.matched(), Some("twenty"));
    }

    #[test]
    fn a_multibyte_stop_split_between_chars() {
        let mut s = OneShotStop::new(&stops(&["😀x"]));
        assert!(!s.push("a 😀"));
        assert!(s.push("x"));
        let mut s = OneShotStop::new(&stops(&["é😀"]));
        assert!(!s.push("caf"));
        assert!(!s.push("é"));
        assert!(s.push("😀 after"));
    }

    #[test]
    fn several_stops_the_first_match_wins() {
        // "lo" ends before "world" does, in the same piece.
        let mut s = OneShotStop::new(&stops(&["world", "lo"]));
        assert!(s.push("hello world"));
        assert_eq!(s.matched(), Some("lo"));
        // Same end: the stop given first wins.
        let mut s = OneShotStop::new(&stops(&["b", "ab"]));
        assert!(s.push("xab"));
        assert_eq!(s.matched(), Some("b"));
        // The second stop can match alone.
        let mut s = OneShotStop::new(&stops(&["never", "end"]));
        assert!(!s.push("the e"));
        assert!(s.push("nd"));
        assert_eq!(s.matched(), Some("end"));
    }

    #[test]
    fn a_stop_over_256_bytes_is_ignored_and_stays_cheap() {
        let long = "x".repeat(MAX_STOP_BYTES + 1);
        let mut s = OneShotStop::new(&[long.clone(), "ok".to_string()]);
        assert_eq!(s.max_stop_bytes(), 2);
        assert!(!s.push(&long), "an over-long stop was checked");
        assert!(s.push("ok"));

        let huge = "y".repeat(200_000);
        let mut s = OneShotStop::new(&[huge, "zz".to_string()]);
        let piece = "y".repeat(20);
        let start = Instant::now();
        for _ in 0..10_000 {
            assert!(!s.push(&piece));
        }
        let took = start.elapsed();
        assert!(s.push("zz"), "the short stop next to a huge one was lost");
        assert!(
            took < Duration::from_millis(50),
            "10 000 pushes took {took:?}"
        );
    }

    #[test]
    fn exactly_256_bytes_is_kept() {
        let edge = "q".repeat(MAX_STOP_BYTES);
        let mut s = OneShotStop::new(std::slice::from_ref(&edge));
        assert_eq!(s.max_stop_bytes(), MAX_STOP_BYTES);
        assert!(!s.push(&edge[..100]));
        assert!(s.push(&edge[100..]));
    }

    #[test]
    fn more_than_4_stops_keeps_the_first_4() {
        let s = OneShotStop::new(&stops(&["a1", "a2", "a3", "a4", "a5"]));
        for (i, x) in ["a1", "a2", "a3", "a4"].iter().enumerate() {
            let mut c = s.clone();
            assert!(c.push(&format!("-{x}-")), "stop {i} not active");
        }
        let mut c = s.clone();
        assert!(!c.push("-a5-"), "the fifth stop was checked");
    }

    #[test]
    fn empty_stops_are_ignored_and_none_active_never_matches() {
        let mut s = OneShotStop::new(&stops(&["", "", "end"]));
        assert!(s.is_active());
        assert!(s.push("the end"));

        let mut none = OneShotStop::new(&stops(&["", ""]));
        assert!(!none.is_active());
        assert_eq!(none.max_stop_bytes(), 0);
        assert!(!none.push(""));
        assert!(!none.push("anything at all"));
        assert!(!none.matches_tail("anything"));
        assert!(!OneShotStop::new(&[]).is_active());
    }

    #[test]
    fn a_match_after_the_tail_rolled_many_times() {
        let mut s = OneShotStop::new(&stops(&["STOP-HERE"]));
        let filler = "abc 😀 def ";
        let mut fed = 0;
        while fed < TAIL_BYTES * 10 {
            assert!(!s.push(filler));
            fed += filler.len();
            assert!(s.tail_len() <= TAIL_BYTES, "tail grew to {}", s.tail_len());
        }
        assert!(!s.push("STOP-"));
        assert!(s.push("HERE"));
        assert_eq!(s.matched(), Some("STOP-HERE"));
    }

    #[test]
    fn a_piece_longer_than_the_tail_is_checked_whole() {
        let mut s = OneShotStop::new(&stops(&["needle"]));
        let mut big = "n".repeat(TAIL_BYTES * 3);
        big.insert_str(10, "needle");
        assert!(s.push(&big), "a match early in a long piece was missed");
    }

    #[test]
    fn matches_tail_checks_only_the_given_tail() {
        let s = OneShotStop::new(&stops(&["twenty", "😀x"]));
        assert!(s.matches_tail("nineteen twenty"));
        assert!(s.matches_tail("a 😀x b"));
        assert!(!s.matches_tail("nineteen twent"));
        // Stateless: a stop split across two calls is not a match.
        assert!(!s.matches_tail("tw"));
        assert!(!s.matches_tail("enty"));
    }
}
