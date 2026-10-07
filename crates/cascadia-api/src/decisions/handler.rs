//! The decision core shared by both wire formats (`evaluate`) and `POST /v1/systemone` (`decide`,
//! the Jev/SystemOne format).

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use indexmap::IndexMap;

use super::schema::{self, answer_from_probs, Answer, DecisionRequest, DecisionResponse, Usage};
use super::DecisionError;
use crate::AppState;

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
    let _permit = match state.permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return Err(DecisionError::Engine(
                "engine at capacity; retry after current requests complete".into(),
            ))
        }
    };
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
    let (svc2, req2) = (svc.clone(), req.clone());
    let encoded = match tokio::task::spawn_blocking(move || svc2.codec.encode(&req2)).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "decision encode task panicked");
            return Err(DecisionError::Engine("decision engine error".into()));
        }
    };
    // Codecs mask tokenizer failures themselves (`tokenize_failed`); every other encode error is
    // a validation reason the client should see (Clef's option cap, JEV's image parts).
    let passes = encoded?;
    if let Some(p) = passes
        .iter()
        .find(|p| p.readout.prompt_ids.len() > svc.max_tokens)
    {
        return Err(DecisionError::TooLong(format!(
            "a decision pass is {} tokens; the limit is {}",
            p.readout.prompt_ids.len(),
            svc.max_tokens
        )));
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
        answers.insert(id.clone(), answer_from_probs(q, p));
    }
    Ok((answers, input_tokens))
}
