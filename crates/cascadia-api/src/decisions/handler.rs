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
    // The engine's joint input limit (MAX_HEAD_INPUT_BYTES): a too-long request. The codec
    // clamps first, so this only fires for a width the API did not know, or in a mixed pipeline
    // where the last rank caps lower than this rank 0's codec: there it is refused before the
    // head runs but after the chain prefill. A single box refuses it before any prefill.
    if reason.starts_with("decision_too_long") {
        return DecisionError::TooLong("decision prompt too long for the model's head".into());
    }
    // A `require_warm` pass whose shared-prefix snapshot was evicted (SEC-3): the engine refused
    // to prefill past the request's budget. Busy, not broken: the capacity 503, retry. (A
    // snapshot that can never fit is `decision_uncacheable`, a 413, handled by the caller.)
    if reason.starts_with("decision_cold") {
        return DecisionError::Engine(CAPACITY.into());
    }
    DecisionError::Engine("decision engine error".into())
}

const CAPACITY: &str = "engine at capacity; retry after current requests complete";

/// The 413 for a request admitted on the shared-prefix discount whose prefix the engine's cache
/// can never hold (`decision_uncacheable`): without reuse it needs every pass in full.
fn uncacheable(passes: &[super::Pass], max_total: usize) -> DecisionError {
    let full: usize = passes.iter().map(|p| p.readout.prompt_ids.len()).sum();
    DecisionError::TooLong(format!(
        "this request needs {full} prefill tokens without prefix reuse (the shared prefix cannot \
         fit the prefix cache); the per-request limit is {max_total}"
    ))
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
    // `strategy` picks among JEV's multi-pass plans; Kev and Clef read any option count in one
    // pass and have none.
    let fixed = req.strategy.and_then(schema::RequestStrategy::fixed);
    if fixed.is_some() && svc.family != super::profile::Family::Jev {
        return Err(DecisionError::BadRequest(
            "strategy applies to JEV only".into(),
        ));
    }
    // Take the permit BEFORE tokenizing: encode() is CPU work proportional to the request.
    // Decisions have their own pool: they never take chat's permits, nor chat theirs.
    // Counted like chat (SEC-5): a capacity refusal, the request total and the in-flight gauge.
    let permit = match state.decision_permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            crate::count_rejected("capacity");
            return Err(DecisionError::Engine(CAPACITY.into()));
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
    // The request is shared by both rounds' blocking encodes.
    let req = Arc::new(req.clone());
    let (svc2, req2, cancel2) = (svc.clone(), req.clone(), cancel.0.clone());
    let encoded = off_executor(&svc, &held, move || {
        svc2.codec.encode_rounds(&req2, &cancel2)
    })
    .await?;
    // Codecs mask tokenizer failures themselves (`tokenize_failed`); every other encode error is
    // a reason the client should see (Clef's option cap as 400; Kev's render budget and state
    // length, Clef's schema size and JEV's early pass cap as 413).
    let (mut passes, reserve) = encoded?;
    check_pass_lengths(&passes, svc.max_tokens)?;
    // A second round (JEV's tournament finals) is counted now, at its worst case, so a request
    // is refused before its first pass rather than half way.
    let total = super::prefill_work(&passes, svc.prefix_reuse) + reserve;
    if total > svc.max_total_tokens {
        return Err(DecisionError::TooLong(format!(
            "this request needs {total} prefill tokens; the per-request limit is {}",
            svc.max_total_tokens
        )));
    }
    dump_passes(state, &svc, &req, &held, &passes, 1).await;
    super::mark_required_reuse(&mut passes, svc.max_total_tokens, svc.prefix_reuse);
    // Every question's per-pass probabilities, in submission order.
    let mut probs: IndexMap<String, Vec<Vec<f64>>> = IndexMap::new();
    let mut input_tokens = run_passes(state, &svc, &req, &passes, &mut probs).await?;
    // Only a request that reserved tokens has a second round (a final is never empty), so every
    // other request skips this encode.
    if reserve > 0 {
        let (svc2, req2, cancel2, first) =
            (svc.clone(), req.clone(), cancel.0.clone(), probs.clone());
        let second = off_executor(&svc, &held, move || {
            svc2.codec.second_round(&req2, &first, &cancel2)
        })
        .await??;
        check_pass_lengths(&second, svc.max_tokens)?;
        // Within the reserve whenever the codec's per-line costs add up (they do for Qwen); a
        // tokenizer where they do not could still exceed it, and then the request stops here
        // rather than overspend. Second-round passes share no prefix, so every one counts in full.
        let spent = super::prefill_work(&passes, svc.prefix_reuse)
            + second
                .iter()
                .map(|p| p.readout.prompt_ids.len())
                .sum::<usize>();
        if spent > svc.max_total_tokens {
            return Err(DecisionError::TooLong(format!(
                "this request needs {spent} prefill tokens; the per-request limit is {}",
                svc.max_total_tokens
            )));
        }
        dump_passes(state, &svc, &req, &held, &second, 2).await;
        input_tokens += run_passes(state, &svc, &req, &second, &mut probs).await?;
    }
    let mut answers = IndexMap::with_capacity(req.questions.len());
    for (id, q) in &req.questions {
        let Some(p) = probs
            .get(id)
            .and_then(|passes| svc.codec.combine(&req, id, passes))
        else {
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
        // Finite logits can still give NaN/inf probabilities if a calibration overflows
        // (`(l + b) / t`, H17); the profile bounds prevent it, and this keeps it from ever
        // reaching a client as `null` in a 200.
        if !p.iter().all(|x| x.is_finite()) {
            return Err(engine_failure("non-finite decision probabilities"));
        }
        let legend = match q.qtype {
            schema::QuestionType::Score => svc.codec.legend(q),
            _ => None,
        };
        answers.insert(id.clone(), answer_with_legend(q, &p, legend));
    }
    Ok((answers, input_tokens))
}

/// Run tokenization work `f` off the async executor (it is CPU work proportional to the
/// request), in one of the service's encode slots. At most `encode_slots` tokenizations run at
/// once, whatever the decision permits allow (each can hold a few hundred MB of tokenizer
/// transients, P2-SEC-3); this waits rather than refusing, since the request already holds its
/// permit. The closure keeps the slot and the request's permit (SEC-2), so an orphaned encode
/// keeps both until it ends.
async fn off_executor<T: Send + 'static>(
    svc: &Arc<super::DecisionService>,
    held: &Arc<Held>,
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, DecisionError> {
    let Ok(slot) = svc.encode_slots.clone().acquire_owned().await else {
        return Err(DecisionError::Engine("decision engine error".into()));
    };
    let held = held.clone();
    match tokio::task::spawn_blocking(move || {
        let _held = held;
        let _slot = slot;
        f()
    })
    .await
    {
        Ok(r) => Ok(r),
        Err(e) => {
            tracing::warn!(error = %e, "decision encode task panicked");
            Err(DecisionError::Engine("decision engine error".into()))
        }
    }
}

/// The 413 for the first pass over the per-pass cap.
fn check_pass_lengths(passes: &[super::Pass], max_tokens: usize) -> Result<(), DecisionError> {
    match passes
        .iter()
        .find(|p| p.readout.prompt_ids.len() > max_tokens)
    {
        Some(p) => Err(super::pass_too_long(p.readout.prompt_ids.len(), max_tokens)),
        None => Ok(()),
    }
}

/// Append `passes` to `--decision-dump`, if it is open. A JEV choice pass's line also names its
/// plan (`strategy`) and `round`, so the tools can rebuild the combination.
async fn dump_passes(
    state: &AppState,
    svc: &super::DecisionService,
    req: &DecisionRequest,
    held: &Arc<Held>,
    passes: &[super::Pass],
    round: u32,
) {
    let Some(dump) = state.decision_dump.clone().filter(|d| !d.is_closed()) else {
        return;
    };
    let tags: Vec<Option<&'static str>> = passes
        .iter()
        .map(|p| match p.parts.as_slice() {
            [part] => svc.codec.dump_strategy(req, &part.question_id),
            _ => None,
        })
        .collect();
    // Serialize and write off the async workers (P2-SEC-5): a request's lines can reach
    // ~16 MB, and a synchronous write under the lock would stall chat streams and timers.
    // The write carries the request's permit, so a client that leaves cannot orphan
    // request-sized writes past the permits (P3-SEC-1). A closed dump (cap reached, write
    // failed) costs nothing: no clone, no serialization.
    let dumped = passes.to_vec();
    let held = held.clone();
    let wrote = tokio::task::spawn_blocking(move || {
        let _held = held;
        if dump.is_closed() {
            return Ok(());
        }
        let mut lines = Vec::new();
        for (p, tag) in dumped.iter().zip(tags) {
            let parts: Vec<_> = p
                .parts
                .iter()
                .map(|x| (&x.question_id, x.logits.start, x.logits.end))
                .collect();
            let mut line = serde_json::json!({"parts": parts, "readout": p.readout});
            if let Some(strategy) = tag {
                line["strategy"] = strategy.into();
                line["round"] = round.into();
            }
            if serde_json::to_writer(&mut lines, &line).is_err() {
                return Err(std::io::Error::other("serialize"));
            }
            lines.push(b'\n');
        }
        dump.append(&lines)
    })
    .await;
    match wrote {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(error = %e, "--decision-dump write failed"),
        Err(e) => tracing::warn!(error = %e, "--decision-dump writer panicked"),
    }
}

/// Run `passes` one after another (the engine runs one decision pass at a time) and append
/// each part's probabilities to its question's list. Returns the prompt tokens prefilled.
async fn run_passes(
    state: &AppState,
    svc: &super::DecisionService,
    req: &DecisionRequest,
    passes: &[super::Pass],
    probs: &mut IndexMap<String, Vec<Vec<f64>>>,
) -> Result<u64, DecisionError> {
    let mut input_tokens: u64 = 0;
    for pass in passes {
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
                // The cache can never hold this request's shared prefix (refused before the first
                // pass prefilled anything): the request needs its full length, a 413.
                if reason.starts_with("decision_uncacheable") {
                    tracing::warn!(reason, "decision pass failed");
                    return Err(uncacheable(passes, svc.max_total_tokens));
                }
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
            probs
                .entry(part.question_id.clone())
                .or_default()
                .push(
                    svc.codec
                        .probabilities(req, part, &logits[part.logits.clone()]),
                );
        }
    }
    Ok(input_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_joint_input_refusal_is_413_not_503() {
        let e = engine_failure(
            "decision_too_long: joint readout of 10241 tokens exceeds 10240 at width 8192",
        );
        assert!(matches!(e, DecisionError::TooLong(_)), "{e:?}");
        assert_eq!(e.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        assert!(matches!(
            engine_failure("decision_invalid: x"),
            DecisionError::Engine(_)
        ));
        match engine_failure("decision_cold: the shared prefix snapshot (21 tokens) is not cached")
        {
            DecisionError::Engine(m) => assert_eq!(m, CAPACITY),
            e => panic!("{e:?}"),
        }
    }
}
