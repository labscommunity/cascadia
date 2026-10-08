//! Typed decisions over decision models (docs/decisions.md): POST /v1/decisions (OpenAI format,
//! openai.rs) and POST /v1/systemone (Jev/SystemOne format, handler.rs), sharing one core.

pub mod clef;
pub mod handler;
pub mod jev;
pub mod kev;
pub mod openai;
pub mod profile;
pub mod schema;

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::http::StatusCode;
use schema::{DecisionRequest, Limits};

#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    TooLong(String),
    #[error("this model does not serve decisions")]
    NotServed,
    #[error("{0}")]
    Engine(String),
}

impl DecisionError {
    pub fn status(&self) -> StatusCode {
        match self {
            DecisionError::BadRequest(_) => StatusCode::BAD_REQUEST,
            DecisionError::TooLong(_) => StatusCode::PAYLOAD_TOO_LARGE,
            DecisionError::NotServed => StatusCode::NOT_IMPLEMENTED,
            DecisionError::Engine(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

/// One prefill-only engine pass and which slice of its logits answers which question.
#[derive(Clone, Debug)]
pub struct Pass {
    pub readout: cascadia_types::DecisionReadout,
    pub parts: Vec<PassPart>,
}

#[derive(Clone, Debug)]
pub struct PassPart {
    pub question_id: String,
    pub logits: Range<usize>,
}

pub trait Codec: Send + Sync {
    fn limits(&self) -> Limits;
    /// Encode the request into engine passes. `cancel` is set when the client is gone (the
    /// request future was dropped): codecs check it between questions and stop early, so an
    /// orphaned encode does not keep tokenizing (SEC-2). The error is never seen by anyone.
    fn encode_until(
        &self,
        req: &DecisionRequest,
        cancel: &AtomicBool,
    ) -> Result<Vec<Pass>, DecisionError>;
    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> {
        self.encode_until(req, &AtomicBool::new(false))
    }
    fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64>;
    /// A score answer's `legend` values, one per level; `None` echoes the request's raw levels.
    fn legend(&self, _q: &schema::Question) -> Option<Vec<schema::OrderedValue>> {
        None
    }
}

pub struct DecisionService {
    pub family: profile::Family,
    pub codec: Box<dyn Codec>,
    pub max_tokens: usize,
    pub max_total_tokens: usize,
    pub max_questions: usize,
    /// The engine can resume a shared prefix (single box, prefix cache on). Only then does the
    /// budget count Kev's state once; pipeline ranks re-prefill it per question.
    pub prefix_reuse: bool,
}

// `cascadia_api::Config` derives Debug, and `Box<dyn Codec>` cannot.
impl std::fmt::Debug for DecisionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionService")
            .field("family", &self.family)
            .field("max_tokens", &self.max_tokens)
            .field("max_total_tokens", &self.max_total_tokens)
            .field("max_questions", &self.max_questions)
            .field("prefix_reuse", &self.prefix_reuse)
            .finish_non_exhaustive()
    }
}

impl DecisionService {
    /// `Ok(None)` when the model directory has no `decision.json` (a plain chat model).
    pub fn load(model_dir: &Path) -> Result<Option<Arc<DecisionService>>, String> {
        let path = model_dir.join("decision.json");
        if !path.exists() {
            return Ok(None);
        }
        let prof = profile::DecisionProfile::load(&path)?;
        let tok = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("{}: tokenizer.json: {e}", model_dir.display()))?;
        Self::from_profile(&prof, tok, tree_hidden(model_dir))
            .map(|s| Some(Arc::new(s)))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Range-check the profile and clamp it to the limits the engine enforces (EDGE-6, COR-1):
    /// a limit the engine would refuse at admission turns a too-large request into a 503 instead
    /// of the API's own 400/413.
    fn from_profile(
        prof: &profile::DecisionProfile,
        tok: tokenizers::Tokenizer,
        hidden: Option<usize>,
    ) -> Result<DecisionService, String> {
        use cascadia_types::{MAX_DECISION_PROMPT, MAX_JOINT_QUESTIONS};
        if prof.max_tokens == 0 {
            return Err("max_tokens must be at least 1".into());
        }
        if prof.max_questions == Some(0) || prof.max_total_tokens == Some(0) {
            return Err("max_questions and max_total_tokens must be at least 1".into());
        }
        let mut max_tokens = prof.max_tokens;
        if max_tokens > MAX_DECISION_PROMPT {
            tracing::warn!(
                max_tokens,
                cap = MAX_DECISION_PROMPT,
                "decision.json max_tokens exceeds the engine's decision prompt cap; clamping"
            );
            max_tokens = MAX_DECISION_PROMPT;
        }
        let max_total_tokens = prof.max_total_tokens.unwrap_or(max_tokens * 4);
        let mut max_questions = prof.max_questions.unwrap_or(32);
        let codec: Box<dyn Codec> = match prof.family {
            profile::Family::Jev => Box::new(
                jev::JevCodec::new(tok, prof.jev.clone().unwrap())?.with_pass_cap(max_tokens),
            ),
            profile::Family::Kev => Box::new(kev::KevCodec::new(
                tok,
                prof.kev.clone().unwrap(),
                max_tokens,
                max_total_tokens,
            )?),
            profile::Family::Clef => {
                let c = clef::ClefCodec::new(tok, max_tokens, hidden)?;
                // Clef truncates the state to fit its one pass, so the pass cap is the joint
                // prompt cap, and the request budget must cover one full pass.
                max_tokens = c.max_tokens();
                if max_total_tokens < max_tokens {
                    return Err(format!(
                        "clef max_total_tokens {max_total_tokens} is below its {max_tokens}-token pass; \
                         long states would be refused instead of truncated"
                    ));
                }
                if max_questions > MAX_JOINT_QUESTIONS {
                    tracing::warn!(
                        max_questions,
                        cap = MAX_JOINT_QUESTIONS,
                        "decision.json max_questions exceeds the joint head's question cap; clamping"
                    );
                    max_questions = MAX_JOINT_QUESTIONS;
                }
                Box::new(c)
            }
        };
        Ok(DecisionService {
            family: prof.family,
            codec,
            max_tokens,
            max_total_tokens,
            max_questions,
            prefix_reuse: false,
        })
    }

    /// The codec's option limits with the deployment's question cap.
    pub fn limits(&self) -> Limits {
        Limits {
            max_questions: self.max_questions,
            ..self.codec.limits()
        }
    }
}

/// The tree's activation width, as the engine reads it: manifest.json `hidden_size`, else
/// config.json's (`text_config` first). `None` when neither states it; the engine then enforces
/// the joint input limits alone.
fn tree_hidden(model_dir: &Path) -> Option<usize> {
    let read = |f: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(model_dir.join(f)).ok()?).ok()
    };
    let width = |v: &serde_json::Value| v.get("hidden_size")?.as_u64().map(|h| h as usize);
    read("manifest.json")
        .as_ref()
        .and_then(width)
        .or_else(|| {
            let c = read("config.json")?;
            c.get("text_config").and_then(width).or_else(|| width(&c))
        })
        .filter(|&h| h > 0)
}

/// A tokenizer failure inside a codec's `encode`. The tokenizer's own text can carry vocab or path
/// detail, so it is logged and the client gets a fixed message. Every other `BadRequest` from a
/// codec is a real validation reason and reaches the client unchanged.
pub(crate) fn tokenize_failed(e: impl std::fmt::Display) -> DecisionError {
    tracing::warn!(error = %e, "decision tokenization failed");
    DecisionError::BadRequest(TOKENIZE_FAILED.into())
}

/// The 413 for a pass over the per-pass cap (the handler's check, or a codec's early stop).
pub(crate) fn pass_too_long(len: usize, max_tokens: usize) -> DecisionError {
    DecisionError::TooLong(format!(
        "a decision pass is {len} tokens; the limit is {max_tokens}"
    ))
}

/// `Err` once `cancel` is set (see [`Codec::encode_until`]).
pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<(), DecisionError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(DecisionError::Engine("request cancelled".into()));
    }
    Ok(())
}

pub(crate) const TOKENIZE_FAILED: &str = "request could not be tokenized";

/// kev.model.user_tokens, shared by every codec: `<|name|>` (name = [A-Za-z0-9_]+) → `<¦name¦>`.
/// HF tokenizers match added/special tokens inside text even with add_special_tokens=false, so
/// without this, state or option text could forge `<|im_end|>`, `<|fim_suffix|>` etc. and steer
/// the decision. Template pieces are tokenized separately and keep their real special tokens.
pub(crate) fn sanitize_user_text(text: &str) -> String {
    // One linear scan (SEC-1). At each "<|", the name is the maximal [A-Za-z0-9_] run after it,
    // rewritten only when "|>" follows the run immediately. This equals "find the first `|>` and
    // check the text before it is a non-empty name": a name holds no '|', so the first "|>" can
    // only sit at the run's end. Re-searching the remainder for "|>" at every "<|" was O(n^2).
    let mut safe = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        safe.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let run = after
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        if run > 0 && after[run..].starts_with("|>") {
            safe.push_str("<¦");
            safe.push_str(&after[..run]);
            safe.push_str("¦>");
            rest = &after[run + 2..];
        } else {
            safe.push_str("<|");
            rest = after;
        }
    }
    safe.push_str(rest);
    safe
}

/// Tokens a request prefills. With `prefix_reuse` (single box, prefix cache on) a prefix every pass
/// shares (`cache_prefix`) is prefilled once and resumed, so it counts once; without it (pipeline,
/// cache off) every pass is prefilled in full. This is an upper bound only when reuse is on: an
/// evicted or over-budget snapshot re-prefills, which the per-pass `max_tokens` cap still bounds.
pub fn prefill_work(passes: &[Pass], prefix_reuse: bool) -> usize {
    let full: usize = passes.iter().map(|p| p.readout.prompt_ids.len()).sum();
    if !prefix_reuse {
        return full;
    }
    let shared = passes
        .iter()
        .filter_map(|p| p.readout.cache_prefix)
        .max()
        .unwrap_or(0) as usize;
    let tails: usize = passes
        .iter()
        .map(|p| {
            p.readout
                .prompt_ids
                .len()
                .saturating_sub(p.readout.cache_prefix.unwrap_or(0) as usize)
        })
        .sum();
    (shared + tails).min(full)
}

/// Mark the passes that must resume their shared prefix from the prefix cache (SEC-3), so the
/// tokens a request actually prefills never exceed `max_total`, whatever the cache evicts.
///
/// With `prefix_reuse` the budget (`prefill_work`) counts a shared prefix once, but the engine
/// re-prefills it whenever the snapshot is gone (another request's passes evicted it, or it never
/// fit the cache). A pass may run cold only while `spent + its full length + every later pass's
/// tail` still fits; the rest carry `require_warm`, and the engine refuses one whose snapshot is
/// gone (`decision_cold`, a 503 to retry) instead of prefilling past the budget. Invariant:
/// `spent + remaining tails <= max_total`, which admission (`prefill_work <= max_total`)
/// establishes. A request that fits without the discount marks nothing and behaves as before.
pub fn mark_required_reuse(passes: &mut [Pass], max_total: usize, prefix_reuse: bool) {
    if !prefix_reuse {
        return;
    }
    let tail = |p: &Pass| {
        p.readout
            .prompt_ids
            .len()
            .saturating_sub(p.readout.cache_prefix.unwrap_or(0) as usize)
    };
    let mut later: usize = passes.iter().map(tail).sum();
    let mut spent = 0usize;
    for p in passes.iter_mut() {
        let (full, t) = (p.readout.prompt_ids.len(), tail(p));
        later -= t;
        if p.readout.cache_prefix.is_none() || spent + full + later <= max_total {
            spent += full;
        } else {
            p.readout.require_warm = true;
            spent += t;
        }
    }
}

/// Caller-text neutralizer bound to one tokenizer: the `<|name|>` rule plus every other
/// angle-bracket added token (Qwen registers `<think>`, `</think>`, `<tool_call>`, … as added tokens,
/// and HF tokenizers match those inside plain text too). Longest tokens are rewritten first.
pub(crate) struct Sanitizer {
    added: Vec<String>,
}

impl Sanitizer {
    pub(crate) fn from_tokenizer(tok: &tokenizers::Tokenizer) -> Self {
        let mut added: Vec<String> = tok
            .get_added_tokens_decoder()
            .values()
            .map(|t| t.content.clone())
            .filter(|c| c.len() > 2 && c.starts_with('<') && c.ends_with('>'))
            // `sanitize_user_text` already rewrites `<|name|>` with name in [A-Za-z0-9_]+; any
            // other `<|…|>` name (`<|box-start|>`) is covered only by this list, so keep it.
            .filter(|c| !is_plain_pipe_token(c))
            .collect();
        added.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        added.dedup();
        Self { added }
    }

    /// Linear in the text: `sanitize_user_text` is one scan, and each added token is one
    /// `contains` + `replace` pass (both linear two-way searches), a fixed number per tokenizer.
    pub(crate) fn apply(&self, text: &str) -> String {
        let mut s = sanitize_user_text(text);
        for a in &self.added {
            if s.contains(a.as_str()) {
                s = s.replace(a.as_str(), &format!("<¦{}¦>", &a[1..a.len() - 1]));
            }
        }
        s
    }
}

/// `<|name|>` with a non-empty name in [A-Za-z0-9_]: the shape `sanitize_user_text` rewrites.
fn is_plain_pipe_token(c: &str) -> bool {
    c.strip_prefix("<|")
        .and_then(|r| r.strip_suffix("|>"))
        .is_some_and(|n| {
            !n.is_empty() && n.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        })
}

/// Numerically stable softmax. Inputs must be finite and non-empty (all -inf or any NaN gives NaN);
/// `handler::evaluate` rejects non-finite engine logits before any codec calls this.
pub(crate) fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn tree_hidden_reads_manifest_then_config() {
        let d = std::env::temp_dir().join(format!("tree-hidden-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(super::tree_hidden(&d), None);
        std::fs::write(
            d.join("config.json"),
            r#"{"text_config": {"hidden_size": 4096}}"#,
        )
        .unwrap();
        assert_eq!(super::tree_hidden(&d), Some(4096));
        std::fs::write(
            d.join("manifest.json"),
            r#"{"hidden_size": 5120, "stages": []}"#,
        )
        .unwrap();
        assert_eq!(super::tree_hidden(&d), Some(5120));
        std::fs::remove_dir_all(&d).unwrap();
    }

    use super::*;
    use cascadia_types::{DecisionReadout, ReadoutHead};

    fn pass(n: usize, cache_prefix: Option<u32>) -> Pass {
        Pass {
            readout: DecisionReadout {
                prompt_ids: vec![1; n],
                head: ReadoutHead::Verbalizer { token_ids: vec![1] },
                cache_prefix,
                require_warm: false,
            },
            parts: Vec::new(),
        }
    }

    /// The pre-SEC-1 implementation (quadratic on `<|<|<|…`), kept as the equivalence oracle.
    fn sanitize_oracle(text: &str) -> String {
        let mut safe = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find("<|") {
            safe.push_str(&rest[..start]);
            let after = &rest[start + 2..];
            match after.find("|>") {
                Some(end)
                    if end > 0
                        && after[..end]
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_') =>
                {
                    safe.push_str("<¦");
                    safe.push_str(&after[..end]);
                    safe.push_str("¦>");
                    rest = &after[end + 2..];
                }
                _ => {
                    safe.push_str("<|");
                    rest = after;
                }
            }
        }
        safe.push_str(rest);
        safe
    }

    /// xorshift64*: deterministic, dependency-free.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    #[test]
    fn sanitize_is_byte_identical_to_the_oracle() {
        // Adversarial alphabet: the delimiters, name chars, a pipe/angle on their own, a non-ASCII
        // char, a char outside the name set and whitespace.
        let atoms = [
            "<|", "|>", "<", "|", ">", "a", "Z", "9", "_", "-", "é", " ", "¦", "<|a|>", "|<",
        ];
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..200_000 {
            let n = (rng.next() % 24) as usize;
            let s: String = (0..n)
                .map(|_| atoms[(rng.next() % atoms.len() as u64) as usize])
                .collect();
            assert_eq!(sanitize_user_text(&s), sanitize_oracle(&s), "{s:?}");
        }
        for s in [
            "",
            "<|",
            "|>",
            "<||>",
            "<|a|>",
            "<|a|><|b|>",
            "<|<|a|>",
            "<|a<|b|>",
            "<|a|b|>",
            "<|a b|>",
            "<|é|>",
            "x<|im_end|>y",
            "<|a",
            "<|a|",
            "<|_|>|>",
        ] {
            assert_eq!(sanitize_user_text(s), sanitize_oracle(s), "{s:?}");
        }
    }

    #[test]
    fn sanitize_is_linear_on_adversarial_input() {
        // SEC-1: 1 MiB of "<|" (with and without a trailing "|>") took minutes when every "<|"
        // rescanned the remainder for "|>".
        for tail in ["", "|>", "a|>"] {
            let s = format!("{}{tail}", "<|".repeat(512 * 1024));
            let t = std::time::Instant::now();
            let out = sanitize_user_text(&s);
            assert!(
                t.elapsed() < std::time::Duration::from_millis(500),
                "{tail:?}: {:?}",
                t.elapsed()
            );
            assert!(out.len() >= s.len());
        }
        let s = format!("{}x", "<|a".repeat(350_000));
        let t = std::time::Instant::now();
        sanitize_user_text(&s);
        assert!(
            t.elapsed() < std::time::Duration::from_millis(500),
            "{:?}",
            t.elapsed()
        );
    }

    #[test]
    fn sanitizer_apply_is_linear_on_adversarial_input() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[unk]".to_string(), 0u32)].into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.add_tokens(&[
            tokenizers::AddedToken::from("<think>", false),
            tokenizers::AddedToken::from("</think>", false),
            tokenizers::AddedToken::from("<tool_call>", false),
        ]);
        let san = Sanitizer::from_tokenizer(&tok);
        for unit in ["<|", "<", "<think", "<think><", "<|<think>"] {
            let s = unit.repeat(1024 * 1024 / unit.len());
            let t = std::time::Instant::now();
            san.apply(&s);
            assert!(
                t.elapsed() < std::time::Duration::from_secs(1),
                "{unit:?}: {:?}",
                t.elapsed()
            );
        }
    }

    fn word_tok() -> tokenizers::Tokenizer {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[unk]".to_string(), 0u32)].into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        tokenizers::Tokenizer::new(model)
    }

    fn profile(json: &str) -> profile::DecisionProfile {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn profile_limits_are_range_checked_and_clamped_at_load() {
        use cascadia_types::{MAX_DECISION_PROMPT, MAX_JOINT_PROMPT, MAX_JOINT_QUESTIONS};
        let load = |j: &str| DecisionService::from_profile(&profile(j), word_tok(), None);
        // Zero limits would fail every request confusingly: refuse the profile.
        for j in [
            r#"{"family":"clef","max_tokens":0}"#,
            r#"{"family":"clef","max_tokens":100,"max_questions":0}"#,
            r#"{"family":"clef","max_tokens":100,"max_total_tokens":0}"#,
        ] {
            assert!(load(j).is_err(), "{j}");
        }
        // Clef: the joint head takes at most MAX_JOINT_QUESTIONS questions; more would be an
        // engine 503 instead of a 400.
        let s = load(r#"{"family":"clef","max_tokens":16384,"max_questions":1000}"#).unwrap();
        assert_eq!(s.max_questions, MAX_JOINT_QUESTIONS);
        assert_eq!(
            s.max_tokens, MAX_JOINT_PROMPT,
            "the pass cap is the clamped joint cap"
        );
        // ... and its budget must cover one full pass, or truncation-to-fit turns into 413s.
        let e = load(r#"{"family":"clef","max_tokens":1000,"max_total_tokens":999}"#).unwrap_err();
        assert!(e.contains("max_total_tokens"), "{e}");
        // Every family: max_tokens is clamped to the engine's decision prompt cap.
        let jev = r#"{"family":"jev","max_tokens":999999,"jev":{"verbalizers":["false","true","0","1","2","3","4","5","A"],"verbalizer_ids":[1,2,3,4,5,6,7,8,9],"bias":[0,0,0,0,0,0,0,0,0],"slots":{"noul":[0,2],"score":[2,8],"choice":[8,9]},"temperature":{"noul":1,"choice":1,"score":1}}}"#;
        let s = load(jev).unwrap();
        assert_eq!(s.max_tokens, MAX_DECISION_PROMPT);
        assert_eq!(s.max_total_tokens, MAX_DECISION_PROMPT * 4);
        assert_eq!(s.max_questions, 32);
        assert_eq!(s.family, profile::Family::Jev);
    }

    #[test]
    fn only_passes_the_budget_cannot_cover_cold_must_resume() {
        let warm = |p: &[Pass]| p.iter().map(|x| x.readout.require_warm).collect::<Vec<_>>();
        // 3 Kev passes: prefix 21, tail 8 (full 29). Reuse work 45, full work 87.
        let kev = || (0..3).map(|_| pass(29, Some(21))).collect::<Vec<_>>();
        let mut p = kev();
        mark_required_reuse(&mut p, 100, true);
        assert_eq!(
            warm(&p),
            [false; 3],
            "fits without the discount: nothing changes"
        );
        let mut p = kev();
        mark_required_reuse(&mut p, 60, true);
        assert_eq!(warm(&p), [false, true, true]);
        let mut p = kev();
        mark_required_reuse(&mut p, 66, true);
        assert_eq!(warm(&p), [false, false, true], "29 + 29 + 8 = 66 fits");
        let mut p = kev();
        mark_required_reuse(&mut p, 45, true);
        assert_eq!(warm(&p), [false, true, true]);
        // Without reuse (pipeline, cache off) the budget already counted every pass in full.
        let mut p = kev();
        mark_required_reuse(&mut p, 45, false);
        assert_eq!(warm(&p), [false; 3]);
        // Worst case honoured: cold where planned cold, tails where warm.
        for budget in 45..=90 {
            let mut p = kev();
            mark_required_reuse(&mut p, budget, true);
            let worst: usize = p
                .iter()
                .map(|x| if x.readout.require_warm { 8 } else { 29 })
                .sum();
            assert!(worst <= budget, "{budget}: {worst}");
        }
    }

    #[test]
    fn prefill_work_counts_shared_prefix_once_only_with_reuse() {
        let kev: Vec<Pass> = (0..5).map(|_| pass(1020, Some(1000))).collect();
        assert_eq!(prefill_work(&kev, true), 1100);
        assert_eq!(prefill_work(&kev, false), 5100);
        let plain = vec![pass(300, None), pass(200, None)];
        assert_eq!(prefill_work(&plain, true), 500);
        assert_eq!(prefill_work(&plain, false), 500);
    }
}
