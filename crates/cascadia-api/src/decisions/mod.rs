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
    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError>;
    fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64>;
}

pub struct DecisionService {
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
            .field("max_tokens", &self.max_tokens)
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
        let codec: Box<dyn Codec> = match prof.family {
            profile::Family::Jev => Box::new(jev::JevCodec::new(tok, prof.jev.clone().unwrap())?),
            profile::Family::Kev => Box::new(kev::KevCodec::new(tok, prof.kev.clone().unwrap())?),
            profile::Family::Clef => Box::new(clef::ClefCodec::new(tok, prof.max_tokens)?),
        };
        let max_total_tokens = prof.max_total_tokens.unwrap_or(prof.max_tokens * 4);
        let max_questions = prof.max_questions.unwrap_or(32);
        Ok(Some(Arc::new(DecisionService {
            codec,
            max_tokens: prof.max_tokens,
            max_total_tokens,
            max_questions,
            prefix_reuse: false,
        })))
    }

    /// The codec's option limits with the deployment's question cap.
    pub fn limits(&self) -> Limits {
        Limits {
            max_questions: self.max_questions,
            ..self.codec.limits()
        }
    }
}

/// A tokenizer failure inside a codec's `encode`. The tokenizer's own text can carry vocab or path
/// detail, so it is logged and the client gets a fixed message. Every other `BadRequest` from a
/// codec is a real validation reason and reaches the client unchanged.
pub(crate) fn tokenize_failed(e: impl std::fmt::Display) -> DecisionError {
    tracing::warn!(error = %e, "decision tokenization failed");
    DecisionError::BadRequest(TOKENIZE_FAILED.into())
}

pub(crate) const TOKENIZE_FAILED: &str = "request could not be tokenized";

/// kev.model.user_tokens, shared by every codec: `<|name|>` (name = [A-Za-z0-9_]+) → `<¦name¦>`.
/// HF tokenizers match added/special tokens inside text even with add_special_tokens=false, so
/// without this, state or option text could forge `<|im_end|>`, `<|fim_suffix|>` etc. and steer
/// the decision. Template pieces are tokenized separately and keep their real special tokens.
pub(crate) fn sanitize_user_text(text: &str) -> String {
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
            .filter(|c| {
                c.len() > 2
                    && c.starts_with('<')
                    && c.ends_with('>')
                    && !(c.starts_with("<|") && c.ends_with("|>"))
            })
            .collect();
        added.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        added.dedup();
        Self { added }
    }

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

pub(crate) fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::{DecisionReadout, ReadoutHead};

    fn pass(n: usize, cache_prefix: Option<u32>) -> Pass {
        Pass {
            readout: DecisionReadout {
                prompt_ids: vec![1; n],
                head: ReadoutHead::Verbalizer { token_ids: vec![1] },
                cache_prefix,
            },
            parts: Vec::new(),
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
