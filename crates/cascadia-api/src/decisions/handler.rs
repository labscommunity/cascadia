//! The decision core shared by both wire formats (`evaluate`) and `POST /v1/systemone` (`decide`,
//! the Jev/SystemOne format).

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use indexmap::IndexMap;

use super::schema::{self, answer_with_legend, Answer, DecisionRequest, DecisionResponse, Usage};
use super::DecisionError;
use crate::AppState;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// What one admitted decision request holds until both its handler future and its blocking
/// encode are finished.
struct Held {
    _permit: tokio::sync::OwnedSemaphorePermit,
    _inflight: crate::InFlightGuard,
}

/// Sets the flag when dropped: the request future ended (finished, failed or disconnected).
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn err(e: DecisionError) -> Response {
    (
        e.status(),
        Json(serde_json::json!({ "error": e.to_string() })),
    )
        .into_response()
}

/// `generate_async` failures: log the EngineError (it can carry paths, peer and OpenVINO text),
/// return a fixed message per category. The chat handlers' `engine_error_response` echoes
/// `err.to_string()` and must not be reused here.
fn decision_engine_error(e: cascadia_engine::EngineError) -> DecisionError {
    use cascadia_engine::EngineError as E;
    tracing::warn!(error = %e, "decision submit failed");
    match e {
        E::PromptTooLong(_) => DecisionError::TooLong("decision prompt too long".into()),
        E::QueueFull { .. } => DecisionError::Engine("engine at capacity or unavailable".into()),
        _ => DecisionError::Engine("decision engine error".into()),
    }
}

/// Engine/head failures: log the detail (paths, OpenVINO text), return a fixed message.
fn engine_failure(reason: &str) -> DecisionError {
    tracing::warn!(reason, "decision pass failed");
    if reason.starts_with("decision_unsupported") {
        return DecisionError::NotServed;
    }
    // The engine's joint input limit (MAX_HEAD_INPUT_BYTES): a too-long request, refused before
    // any prefill. The codec clamps first, so this only fires for a width the API did not know.
    if reason.starts_with("decision_too_long") {
        return DecisionError::TooLong("decision prompt too long for the model's head".into());
    }
    DecisionError::Engine("decision engine error".into())
}

/// POST /v1/systemone: the Jev/SystemOne wire format.
pub async fn decide(State(state): State<AppState>, Json(req): Json<DecisionRequest>) -> Response {
    match evaluate(&state, &req).await {
        Ok((answers, input_tokens)) => Json(DecisionResponse {
            model: state.model_id.clone(),
            answers,
            usage: Usage {
                input_tokens,
                output_tokens: 0,
            },
        })
        .into_response(),
        Err(e) => err(e),
    }
}

/// The core both wire formats share: validate, encode, budget, run every pass, and return the
/// answers in request order plus the prompt tokens prefilled. Errors are `DecisionError`s, so each
/// route renders them in its own format (SystemOne `{"error": msg}`, OpenAI `{"error": {...}}`).
pub(crate) async fn evaluate(
    state: &AppState,
    req: &DecisionRequest,
) -> Result<(IndexMap<String, Answer>, u64), DecisionError> {
    let Some(svc) = state.decisions.clone() else {
        return Err(DecisionError::NotServed);
    };
    req.validate(&svc.limits())?;
    // Take the permit BEFORE tokenizing: encode() is CPU work proportional to the request.
    // Counted like chat (SEC-5): a capacity refusal, the request total and the in-flight gauge.
    let permit = match state.permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            crate::count_rejected("capacity");
            return Err(DecisionError::Engine(
                "engine at capacity; retry after current requests complete".into(),
            ));
        }
    };
    state
        .stats
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // The permit (and the in-flight gauge) is shared with the blocking encode below: a client
    // disconnect drops this future but not a running `spawn_blocking` closure, so the permit is
    // released only once both are done (SEC-2). Orphaned encodes stay bounded by the permit count.
    let held = Arc::new(Held {
        _permit: permit,
        _inflight: crate::InFlightGuard::new(state.stats.clone()),
    });
    // Set when this future is dropped (client gone): the codec stops between questions.
    let cancel = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    // Cheap pre-check before any tokenization: a token is at least one byte, and this bound is
    // generous (~8 bytes per token) so honest requests never hit it.
    let state_bytes = match &req.state {
        schema::OrderedValue::String(s) => s.len(),
        other => other.py_json(false, true).len(),
    };
    let byte_limit = svc.max_total_tokens.saturating_mul(8);
    if state_bytes > byte_limit {
        return Err(DecisionError::TooLong(format!(
            "state is {state_bytes} bytes; the limit is {byte_limit}"
        )));
    }
    // Tokenization is CPU work proportional to the request: run it off the async executor.
    let (svc2, req2, held2, cancel2) = (svc.clone(), req.clone(), held.clone(), cancel.0.clone());
    let encoded = match tokio::task::spawn_blocking(move || {
        let _held = held2;
        svc2.codec.encode_until(&req2, &cancel2)
    })
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "decision encode task panicked");
            return Err(DecisionError::Engine("decision engine error".into()));
        }
    };
    // Codecs mask tokenizer failures themselves (`tokenize_failed`); every other encode error is
    // a reason the client should see (Clef's option cap as 400; Kev's render budget and state
    // length, Clef's schema size and JEV's early pass cap as 413).
    let passes = encoded?;
    if let Some(p) = passes
        .iter()
        .find(|p| p.readout.prompt_ids.len() > svc.max_tokens)
    {
        return Err(super::pass_too_long(
            p.readout.prompt_ids.len(),
            svc.max_tokens,
        ));
    }
    let total = super::prefill_work(&passes, svc.prefix_reuse);
    if total > svc.max_total_tokens {
        return Err(DecisionError::TooLong(format!(
            "this request needs {total} prefill tokens; the per-request limit is {}",
            svc.max_total_tokens
        )));
    }
    if let Some(f) = &state.decision_dump {
        use std::io::Write;
        // A poisoned lock (a panic mid-write) must not take the endpoint down: keep writing.
        let mut f = f.lock().unwrap_or_else(|e| e.into_inner());
        for p in &passes {
            let parts: Vec<_> = p
                .parts
                .iter()
                .map(|x| (&x.question_id, x.logits.start, x.logits.end))
                .collect();
            if let Err(e) = writeln!(
                f,
                "{}",
                serde_json::json!({"parts": parts, "readout": p.readout})
            ) {
                tracing::warn!(error = %e, "--decision-dump write failed");
                break;
            }
        }
    }
    let mut probs: IndexMap<String, Vec<f64>> = IndexMap::new();
    let mut input_tokens: u64 = 0;
    for pass in &passes {
        let task = cascadia_types::GenerationTask {
            task_id: format!("dec-{}", uuid::Uuid::new_v4().simple()),
            prompt: String::new(),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            max_tokens: 1,
            temperature: 0.0,
            logprobs: 0,
            sampling: Default::default(),
            enable_thinking: false,
            trust_remote_code: false,
            // Never client-settable (see the chat handler): a KV-namespace boundary.
            tenant: String::new(),
            resume_token_ids: None,
            decision: Some(pass.readout.clone()),
        };
        let mut stream = match state.runner.generate_async(task).await {
            Ok(s) => s,
            Err(e) => return Err(decision_engine_error(e)),
        };
        let mut logits = None;
        while let Some(chunk) = stream.next().await {
            if let Some(reason) = &chunk.error {
                return Err(engine_failure(reason));
            }
            if chunk.is_final {
                input_tokens += chunk
                    .prompt_tokens
                    .unwrap_or(pass.readout.prompt_ids.len() as u32)
                    as u64;
                logits = chunk.decision_logits.clone();
            }
        }
        let Some(logits) = logits.filter(|l| l.len() == pass.readout.head.output_len()) else {
            return Err(engine_failure(
                "engine returned no (or mis-sized) decision logits",
            ));
        };
        // NaN/±inf (an fp16 overflow, a degenerate head) would softmax to NaN, serialise as
        // `null` and argmax to the first option: a silent wrong 200. Fail the request instead.
        if !logits.iter().all(|x| x.is_finite()) {
            return Err(engine_failure("non-finite decision logits"));
        }
        for part in &pass.parts {
            probs.insert(
                part.question_id.clone(),
                svc.codec
                    .probabilities(req, part, &logits[part.logits.clone()]),
            );
        }
    }
    let mut answers = IndexMap::with_capacity(req.questions.len());
    for (id, q) in &req.questions {
        let Some(p) = probs.get(id) else {
            return Err(engine_failure("codec produced no answer for a question"));
        };
        // A codec bug (an off-by-one logits range, a rescale to the wrong level count) would
        // otherwise truncate the distribution silently or panic on `probs[1]` (QUA-6).
        if p.len() != q.option_keys().len() {
            return Err(engine_failure(&format!(
                "codec returned {} probabilities for {} options",
                p.len(),
                q.option_keys().len()
            )));
        }
        let legend = match q.qtype {
            schema::QuestionType::Score => svc.codec.legend(q),
            _ => None,
        };
        answers.insert(id.clone(), answer_with_legend(q, p, legend));
    }
    Ok((answers, input_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_joint_input_refusal_is_413_not_503() {
        let e = engine_failure("decision_too_long: joint readout of 13108 tokens exceeds 13107");
        assert!(matches!(e, DecisionError::TooLong(_)), "{e:?}");
        assert_eq!(e.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        assert!(matches!(
            engine_failure("decision_invalid: x"),
            DecisionError::Engine(_)
        ));
    }
}
