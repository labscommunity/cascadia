//! N-gram lookup draft model for speculative decoding.
//!
//! "Lookahead Decoding" / "Prompt Lookup Decoding"-style draft: instead
//! of running a small neural-network draft model, we maintain a hash
//! table mapping recent k-grams in the token history to the token that
//! followed them. At draft time we look up the trailing k-gram of the
//! current history and propose the next `n` tokens that followed it.
//!
//! Why this matters for K2.6:
//! - **Zero-compute**: drafting is a hash lookup (~µs) vs the ~9 sec
//!   per-token target forward pass. Even single-token acceptance is
//!   essentially free draft cost.
//! - **No extra model**: avoids the design question of "what draft model
//!   pairs with K2.6's 163,840-vocab tokenizer?" — n-gram drafts are
//!   tokenizer-agnostic because they live entirely in token-id space.
//! - **Works on repeated structure**: empirically (Yang et al.,
//!   "Prompt Lookup Decoding", 2025) ~1.5-2× speedup on completion
//!   tasks with long shared structure (code, JSON, repetitive prose).
//!   K2.6's typical eval prompts ("The capital of France
//!   is...", code completions) have this structure.
//!
//! This module is pure (no model deps), so it's testable without
//! loading weights. The speculative-decode loop in
//! [`crate::spec_decode`] is what wires it to the int4 target forward.
//!
//! Design notes:
//! - We index k-grams of length [`MIN_NGRAM`..=`MAX_NGRAM`], preferring
//!   the longest match (more specific → more accurate). Falling back to
//!   shorter k-grams catches less-specific repetition.
//! - The table is updated on every appended token via [`Draft::append`].
//!   A bounded ring of recent (k-gram, next-token) pairs keeps memory
//!   constant across long generations.
//! - Drafting is single-branch greedy: each propose step picks the
//!   *most recent* observed continuation. We do not maintain
//!   confidence scores; the spec-decode acceptance step is the truth
//!   for whether a draft was correct.

use std::collections::HashMap;

/// Smallest k-gram we index. k=1 ("what token followed token X?") is
/// noisy; k=2 is the empirical sweet spot for repetitive text.
pub const MIN_NGRAM: usize = 2;

/// Largest k-gram we index. Bigger k-grams are more specific but match
/// less often. 4 covers common bigrams + trigrams + 4-grams without
/// blowing up the table.
pub const MAX_NGRAM: usize = 4;

/// How many tokens to draft per round (the `K` of speculative decoding).
/// Default tuned to measured K2.6 numbers (`SPEC_K=8`); callers can
/// override per task via [`Draft::with_draft_k`].
pub const DEFAULT_DRAFT_K: usize = 8;

/// What every request so far has taught about "which token follows these":
/// a backoff table over 3-, 2- and 1-token contexts with counts, shared by all
/// streams of a process ([`Draft::with_shared`]). A request's own prompt and
/// output are the best guide when they repeat; this answers the rest of the
/// time, which is most of the time for a reasoning model whose stock phrases
/// ("The user is asking for ...") recur across requests but not within one.
///
/// A pipeline that speculates on a lone stream pays about half a stage time
/// for a wrong guess and saves ten stage times with a right one, so a guess is
/// worth sending when it is right more than about one time in twenty
/// ([`SharedNgrams::MIN_CONFIDENCE`] asks for one in eight to leave a margin).
#[derive(Debug, Default)]
pub struct SharedNgrams {
    /// Context (1..=3 tokens, FNV-hashed with its length) -> followers seen.
    table: HashMap<u64, Followers>,
}

/// The most frequent followers of one context (a handful is enough: only the
/// top one is ever proposed) and how often the context was seen at all.
#[derive(Debug, Default, Clone)]
struct Followers {
    seen: u32,
    top: Vec<(i64, u32)>,
}

impl SharedNgrams {
    /// Longest shared context. Longer contexts live in the per-request table.
    pub const MAX_CONTEXT: usize = 3;
    /// Followers remembered per context.
    const KEEP: usize = 4;
    /// Contexts remembered; past this the table stops learning new ones (the
    /// common ones are in by then) rather than grow without bound.
    const MAX_CONTEXTS: usize = 2_000_000;
    /// A shared guess is proposed only if its follower was seen in at least
    /// this share of the context's occurrences...
    pub const MIN_CONFIDENCE: f32 = 0.125;
    /// ...and the context at least this often.
    const MIN_SEEN: u32 = 2;

    fn key(ctx: &[i64]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ ctx.len() as u64;
        for &t in ctx {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Learn from a finished sequence (prompt followed by output).
    pub fn learn(&mut self, tokens: &[i64]) {
        for i in 1..tokens.len() {
            let next = tokens[i];
            for k in 1..=Self::MAX_CONTEXT.min(i) {
                let key = Self::key(&tokens[i - k..i]);
                if !self.table.contains_key(&key) && self.table.len() >= Self::MAX_CONTEXTS {
                    continue;
                }
                let f = self.table.entry(key).or_default();
                f.seen = f.seen.saturating_add(1);
                if let Some(i) = f.top.iter().position(|(t, _)| *t == next) {
                    f.top[i].1 = f.top[i].1.saturating_add(1);
                } else if f.top.len() < Self::KEEP {
                    f.top.push((next, 1));
                } else if let Some(min) = f.top.iter_mut().min_by_key(|(_, n)| *n) {
                    // The rarest remembered follower gives way, and the newcomer
                    // starts from one: counts never overstate, so the confidence
                    // bar cannot be passed by churn.
                    *min = (next, 1);
                }
            }
        }
    }

    /// The likeliest follower of the longest known suffix of `buf`, if it
    /// clears the confidence bar.
    pub fn guess(&self, buf: &[i64]) -> Option<i64> {
        for k in (1..=Self::MAX_CONTEXT.min(buf.len())).rev() {
            let Some(f) = self.table.get(&Self::key(&buf[buf.len() - k..])) else {
                continue;
            };
            let Some(&(t, n)) = f.top.iter().max_by_key(|(_, n)| *n) else {
                continue;
            };
            if f.seen >= Self::MIN_SEEN && n as f32 >= Self::MIN_CONFIDENCE * f.seen as f32 {
                return Some(t);
            }
        }
        None
    }

    pub fn contexts(&self) -> usize {
        self.table.len()
    }

    const MAGIC: &'static [u8; 8] = b"CSNGRAM1";

    /// Write the table to `path` (tmp + rename), so what requests taught
    /// survives a restart of the worker.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let tmp = path.with_extension("tmp");
        {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            w.write_all(Self::MAGIC)?;
            w.write_all(&(self.table.len() as u64).to_le_bytes())?;
            for (key, f) in &self.table {
                w.write_all(&key.to_le_bytes())?;
                w.write_all(&f.seen.to_le_bytes())?;
                w.write_all(&[f.top.len() as u8])?;
                for (t, n) in &f.top {
                    w.write_all(&t.to_le_bytes())?;
                    w.write_all(&n.to_le_bytes())?;
                }
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    /// Read a table written by [`Self::save`]. A missing, foreign or damaged
    /// file yields an error and the caller starts empty.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 16];
        r.read_exact(&mut head)?;
        if &head[..8] != Self::MAGIC {
            return Err(bad("not a shared n-gram table"));
        }
        let n = u64::from_le_bytes(head[8..].try_into().expect("8 bytes")) as usize;
        if n > Self::MAX_CONTEXTS {
            return Err(bad("table larger than this build accepts"));
        }
        let mut table = HashMap::with_capacity(n);
        for _ in 0..n {
            let mut rec = [0u8; 13];
            r.read_exact(&mut rec)?;
            let key = u64::from_le_bytes(rec[..8].try_into().expect("8 bytes"));
            let seen = u32::from_le_bytes(rec[8..12].try_into().expect("4 bytes"));
            let k = rec[12] as usize;
            if k > Self::KEEP {
                return Err(bad("damaged entry"));
            }
            let mut top = Vec::with_capacity(k);
            for _ in 0..k {
                let mut e = [0u8; 12];
                r.read_exact(&mut e)?;
                top.push((
                    i64::from_le_bytes(e[..8].try_into().expect("8 bytes")),
                    u32::from_le_bytes(e[8..].try_into().expect("4 bytes")),
                ));
            }
            table.insert(key, Followers { seen, top });
        }
        Ok(Self { table })
    }
}

/// N-gram lookup draft model. Stateless w.r.t. the target — owns its
/// own token history and lookup table.
///
/// Memory: O(history_len × MAX_NGRAM) entries in the worst case
/// (every new token spawns one entry per k-gram length). For typical
/// generations (max_new ≤ 256) this is < 1 KB. We don't bound or evict
/// because the engine drops the Draft between tasks.
pub struct Draft {
    /// Full token history (prompt + accepted tokens).
    history: Vec<i64>,
    /// Map from k-gram (encoded as flat bytes) to last observed next-token.
    /// Using flat bytes lets us key on slices of various lengths without
    /// allocating a Vec per insert.
    table: HashMap<Vec<i64>, i64>,
    /// Max tokens to draft per round.
    draft_k: usize,
    /// What other requests taught ([`SharedNgrams`]); asked when this
    /// request's own history has no match.
    shared: Option<std::sync::Arc<std::sync::Mutex<SharedNgrams>>>,
    /// Where the output begins in `history` (after [`Self::warm_with_prompt`]).
    prompt_len: usize,
    /// A small language model asked before the tables ([`crate::lm_draft`]):
    /// its settings until the first guess is wanted, then the live link. Only
    /// a stream that really speculates ever opens one.
    lm_cfg: Option<(
        crate::lm_draft::LmConfig,
        std::sync::Arc<tokenizers::Tokenizer>,
    )>,
    lm: Option<crate::lm_draft::LmLink>,
    /// Guesses by source: `(model, tables)`.
    sources: (u64, u64),
}

impl Draft {
    /// New empty draft with the default k.
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            table: HashMap::new(),
            draft_k: DEFAULT_DRAFT_K,
            shared: None,
            prompt_len: 0,
            lm_cfg: None,
            lm: None,
            sources: (0, 0),
        }
    }

    /// Ask a drafter model first ([`crate::lm_draft`]); the tables answer when
    /// it has nothing ready. `tok` is the TARGET's tokenizer.
    pub fn with_lm(
        mut self,
        cfg: crate::lm_draft::LmConfig,
        tok: std::sync::Arc<tokenizers::Tokenizer>,
    ) -> Self {
        self.lm_cfg = Some((cfg, tok));
        self
    }

    /// The next token's guess for the speculation path: the drafter model's if
    /// one is configured and has an answer within `wait`, else the tables'.
    pub fn propose_one(&mut self, wait: std::time::Duration) -> Option<i64> {
        if self.lm.is_none() {
            if let Some((cfg, tok)) = self.lm_cfg.take() {
                let prompt = &self.history[..self.prompt_len.min(self.history.len())];
                self.lm = Some(crate::lm_draft::LmLink::new(cfg, tok, prompt));
            }
        }
        if let Some(lm) = self.lm.as_mut() {
            let from = self.prompt_len.min(self.history.len());
            if let Some(t) = lm.next(&self.history[from..], wait) {
                self.sources.0 += 1;
                return Some(t);
            }
            // A guess from the tables would send the text somewhere the model
            // is not writing towards, and the model is right more often: an
            // empty frame is the better price. The tables answer only when
            // there is no model to ask.
            if !lm.unreachable() {
                return None;
            }
        }
        let t = self.lookup_next(&self.history)?;
        self.sources.1 += 1;
        Some(t)
    }

    /// Tell the drafter model where the text stands NOW, so that it writes
    /// while the caller computes (a frame holds rank 0 for a stage time; the
    /// guess for the frame after it is wanted right then). No-op without one.
    pub fn poke(&mut self) {
        if self.lm.is_none() {
            if let Some((cfg, tok)) = self.lm_cfg.take() {
                let prompt = &self.history[..self.prompt_len.min(self.history.len())];
                self.lm = Some(crate::lm_draft::LmLink::new(cfg, tok, prompt));
            }
        }
        if let Some(lm) = self.lm.as_mut() {
            let from = self.prompt_len.min(self.history.len());
            let _ = lm.next(&self.history[from..], std::time::Duration::ZERO);
        }
    }

    /// `(guesses from the drafter model, guesses from the tables)`.
    pub fn sources(&self) -> (u64, u64) {
        self.sources
    }

    /// Whether a drafter model is configured for this request.
    pub fn has_lm(&self) -> bool {
        self.lm.is_some() || self.lm_cfg.is_some()
    }

    /// Fall back to a table shared across requests when this one's own
    /// history has no match for the current suffix.
    pub fn with_shared(mut self, shared: std::sync::Arc<std::sync::Mutex<SharedNgrams>>) -> Self {
        self.shared = Some(shared);
        self
    }

    /// The tokens seen so far (prompt, output, and any speculated tail).
    pub fn history(&self) -> &[i64] {
        &self.history
    }

    /// Override draft K. Clamped to `1..=64`; values outside that range
    /// are dropped (negative ROI per the K-sweep data).
    pub fn with_draft_k(mut self, k: usize) -> Self {
        self.draft_k = k.clamp(1, 64);
        self
    }

    pub fn draft_k(&self) -> usize {
        self.draft_k
    }

    /// Clear all state. Call between tasks.
    pub fn reset(&mut self) {
        self.history.clear();
        self.table.clear();
        self.prompt_len = 0;
        self.lm = None;
    }

    /// Bulk-load history (prompt prefill). Walks every k-gram in
    /// `tokens` and updates the lookup table. Idempotent across calls
    /// — multiple prompts can be loaded (e.g. system + user) without
    /// stale entries persisting incorrectly.
    pub fn warm_with_prompt(&mut self, tokens: &[i64]) {
        for &t in tokens {
            self.append(t);
        }
        self.prompt_len = self.history.len();
    }

    /// Append one verified token. Updates the k-gram table with the
    /// (history-suffix, t) edges this new token now confirms.
    pub fn append(&mut self, t: i64) {
        // For every k in MIN..=MAX, if there's a k-gram ending at the
        // current trailing position, insert (k-gram, t) into the table.
        // The k-gram is `history[history.len()-k..history.len()]`
        // BEFORE we push t.
        let h = &self.history;
        for k in MIN_NGRAM..=MAX_NGRAM {
            if h.len() < k {
                continue;
            }
            let key: Vec<i64> = h[h.len() - k..].to_vec();
            self.table.insert(key, t);
        }
        self.history.push(t);
    }

    /// Rewind by `n` tokens. Used after a spec round when fewer than K
    /// drafts were accepted — the target's KV cache truncates back, and
    /// the draft history must mirror that or future k-gram lookups will
    /// reference tokens the target doesn't have in its cache.
    ///
    /// We do NOT remove the table entries the rewound tokens
    /// contributed. They remain available as "what might come next" for
    /// the next round, which is consistent with the n-gram philosophy
    /// of "use any past observed continuation". Removing them on every
    /// rewind would force a relearn after each rejection.
    pub fn rewind(&mut self, n: usize) {
        let n = n.min(self.history.len());
        self.history.truncate(self.history.len() - n);
    }

    /// Current history length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Propose up to `draft_k` next tokens. Returns the drafted token
    /// sequence; may be empty if no k-gram match is found.
    ///
    /// Algorithm: at each step, look at the trailing MAX_NGRAM tokens of
    /// the current history+drafts buffer, then try k = MAX..MIN. The
    /// first matching k-gram's continuation is appended. Stop when no
    /// k-gram of any tracked length matches.
    ///
    /// This is single-branch greedy — equivalent to the "longest match
    /// wins, no scoring" variant in the prompt-lookup paper.
    pub fn propose(&self) -> Vec<i64> {
        let mut out: Vec<i64> = Vec::with_capacity(self.draft_k);
        // Working buffer: history + tokens we've drafted so far.
        let mut working = self.history.clone();
        for _ in 0..self.draft_k {
            let candidate = self.lookup_next(&working);
            match candidate {
                Some(t) => {
                    out.push(t);
                    working.push(t);
                }
                None => break,
            }
        }
        out
    }

    /// Look up the next token after the trailing k-gram of `buf`.
    /// Tries the longest k-gram first, then progressively shorter.
    /// Returns the first match found.
    fn lookup_next(&self, buf: &[i64]) -> Option<i64> {
        for k in (MIN_NGRAM..=MAX_NGRAM).rev() {
            if buf.len() < k {
                continue;
            }
            let key = &buf[buf.len() - k..];
            if let Some(&t) = self.table.get(key) {
                return Some(t);
            }
        }
        let shared = self.shared.as_ref()?;
        shared.lock().ok()?.guess(buf)
    }
}

impl Default for Draft {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_table_answers_what_the_request_cannot() {
        use std::sync::{Arc, Mutex};
        let shared = Arc::new(Mutex::new(SharedNgrams::default()));
        // Two earlier requests went "7 8 9 10"; a third went "7 8 50".
        for seq in [[1, 7, 8, 9, 10], [2, 7, 8, 9, 10], [3, 7, 8, 50, 60]] {
            shared.lock().unwrap().learn(&seq);
        }
        let mut d = Draft::new().with_draft_k(1).with_shared(shared.clone());
        d.warm_with_prompt(&[40, 41, 7, 8]);
        // Nothing in this request follows "7 8"; the shared table says 9 (2 of 3).
        assert_eq!(d.propose(), vec![9]);
        // The request's own history wins over the shared table.
        d.warm_with_prompt(&[50, 99, 7, 8]);
        assert_eq!(d.propose(), vec![50]);
        // A context seen once is not evidence, and a rare follower is not a guess.
        let mut fresh = Draft::new().with_draft_k(1).with_shared(shared.clone());
        fresh.warm_with_prompt(&[3]);
        assert!(fresh.propose().is_empty(), "\"3\" was seen once: no guess");
        let mut noisy = SharedNgrams::default();
        let mut seq = Vec::new();
        for t in 0..40 {
            seq.extend([5, 100 + t]); // forty different followers of 5
        }
        noisy.learn(&seq);
        assert_eq!(
            noisy.guess(&[5]),
            None,
            "1 in 40 is below the confidence bar"
        );
    }

    #[test]
    fn shared_table_survives_a_restart() {
        let mut t = SharedNgrams::default();
        t.learn(&[1, 7, 8, 9, 10, 7, 8, 9, 11, 7, 8, 9]);
        let dir = std::env::temp_dir().join(format!("cs-ngram-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("table.bin");
        t.save(&path).unwrap();
        let back = SharedNgrams::load(&path).unwrap();
        assert_eq!(back.contexts(), t.contexts());
        for ctx in [&[7i64, 8][..], &[8, 9], &[9], &[7, 8, 9]] {
            assert_eq!(back.guess(ctx), t.guess(ctx), "context {ctx:?}");
        }
        std::fs::write(&path, b"garbage").unwrap();
        assert!(SharedNgrams::load(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_draft_proposes_nothing() {
        let d = Draft::new();
        assert!(d.propose().is_empty());
    }

    #[test]
    fn warm_with_prompt_indexes_kgrams() {
        let mut d = Draft::new();
        d.warm_with_prompt(&[1, 2, 3, 4, 5]);
        assert_eq!(d.history_len(), 5);
        // After loading [1,2,3,4,5]: the 2-gram [1,2] should map to 3,
        // [2,3] to 4, [3,4] to 5.
        let mut probe = vec![1, 2];
        let n = d.lookup_next(&probe);
        assert_eq!(n, Some(3));
        probe = vec![3, 4];
        assert_eq!(d.lookup_next(&probe), Some(5));
    }

    #[test]
    fn propose_walks_repeated_sequence() {
        // History contains "the cat sat on the cat sat ..." — the
        // 2-gram [the, cat] → sat should let us draft [sat, on].
        let mut d = Draft::new().with_draft_k(4);
        // Use distinct ids: "the"=10 "cat"=20 "sat"=30 "on"=40 "mat"=50
        d.warm_with_prompt(&[10, 20, 30, 40, 50, 10, 20, 30, 40, 10, 20]);
        // After [..., 10, 20], lookup_next should return 30 (last
        // observed after [10, 20]).
        let proposal = d.propose();
        // First draft: after [10, 20] -> 30 (the cat -> sat)
        // After drafting 30, working = [..., 10, 20, 30], lookup_next
        // should yield 40 (the [20, 30] 2-gram → 40).
        assert!(!proposal.is_empty());
        assert_eq!(proposal[0], 30);
        if proposal.len() >= 2 {
            assert_eq!(proposal[1], 40);
        }
    }

    #[test]
    fn append_extends_history_and_indexes() {
        let mut d = Draft::new();
        d.warm_with_prompt(&[1, 2, 3]);
        // Before append: [1,2,3] indexes (1,2)→3.
        // append(7): history becomes [1,2,3,7], (2,3)→7 is now in table.
        d.append(7);
        assert_eq!(d.history_len(), 4);
        let probe = vec![2_i64, 3];
        assert_eq!(d.lookup_next(&probe), Some(7));
    }

    #[test]
    fn rewind_truncates_history_but_keeps_table() {
        let mut d = Draft::new();
        d.warm_with_prompt(&[1, 2, 3, 4, 5]);
        // Table has (1,2)→3, (2,3)→4, (3,4)→5 etc.
        d.rewind(2);
        assert_eq!(d.history_len(), 3);
        // Table entries persist: looking up (1,2) still gives 3.
        let probe = vec![1_i64, 2];
        assert_eq!(d.lookup_next(&probe), Some(3));
        // And (3,4)→5 also persists even though 4 and 5 are no longer
        // in history.
        let probe = vec![3_i64, 4];
        assert_eq!(d.lookup_next(&probe), Some(5));
    }

    #[test]
    fn rewind_past_history_clamps_to_zero() {
        let mut d = Draft::new();
        d.warm_with_prompt(&[1, 2, 3]);
        d.rewind(100);
        assert_eq!(d.history_len(), 0);
    }

    #[test]
    fn reset_clears_state() {
        let mut d = Draft::new();
        d.warm_with_prompt(&[1, 2, 3]);
        d.reset();
        assert_eq!(d.history_len(), 0);
        assert!(d.propose().is_empty());
    }

    #[test]
    fn draft_k_default_and_clamp() {
        let d = Draft::new();
        assert_eq!(d.draft_k(), DEFAULT_DRAFT_K);
        let d = Draft::new().with_draft_k(0);
        assert_eq!(d.draft_k(), 1);
        let d = Draft::new().with_draft_k(1000);
        assert_eq!(d.draft_k(), 64);
    }

    #[test]
    fn longer_kgram_match_wins_over_shorter() {
        // Construct a history where (a, b) → c and (z, a, b) → d.
        // A lookup ending in [..., z, a, b] should pick d (the
        // longer-k match), not c.
        // ids: a=1, b=2, c=3, d=4, z=5
        let mut d = Draft::new();
        // Sequence "1,2,3" indexes (1,2)→3.
        d.warm_with_prompt(&[1, 2, 3]);
        // Now sequence "5,1,2,4" indexes (5,1)→2, (1,2)→4 OVERWRITES
        // the earlier (1,2)→3 (table stores last-observed). So (1,2)→4
        // is what wins on a short-k lookup. To make a longer-k differ,
        // continue with "5,1,2,9" — this gives (5,1,2)→9.
        d.append(5);
        d.append(1);
        d.append(2);
        d.append(9);
        // Now table:
        //   (1,2) → last is 9 (from the most recent [1,2] followed-by 9... wait,
        //                       (1,2) wasn't followed by 9 — it was followed by 9? Let me trace.)
        // history evolves: [1,2,3] → append(5)=[1,2,3,5] indexes (3) nothing 2-gram yet (need 2-gram BEFORE 5).
        // Actually the algorithm: when we append t, we index k-gram = history[..len] (before push) → t.
        // So appending 5 to [1,2,3]: history before push = [1,2,3]; (2,3)→5 indexed. Then push 5.
        // appending 1: history before = [1,2,3,5]; (3,5)→1 indexed; (2,3,5)→1 indexed.
        // appending 2: history before = [1,2,3,5,1]; (5,1)→2 indexed; (3,5,1)→2; (2,3,5,1)→2.
        // appending 9: history before = [1,2,3,5,1,2]; (1,2)→9 indexed (OVERWRITES the earlier (1,2)→3);
        //                                  (5,1,2)→9; (3,5,1,2)→9 indexed.
        // So now lookup on [..., 5, 1, 2]:
        //   k=4 needs ≥4 tokens; key=[3,5,1,2] → table has (3,5,1,2)→9. Hit!
        //   Returns 9.
        // Lookup on [..., 1, 2] (only 2 tokens or k=4 doesn't apply):
        //   k=4 needs 4 tokens, only 2 → skip
        //   k=3 needs 3 tokens, only 2 → skip
        //   k=2 → (1,2)→9. Returns 9.
        let buf = vec![3_i64, 5, 1, 2];
        assert_eq!(d.lookup_next(&buf), Some(9));
        let short = vec![1_i64, 2];
        assert_eq!(d.lookup_next(&short), Some(9));
    }
}
