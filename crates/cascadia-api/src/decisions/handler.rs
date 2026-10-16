//! The decision core shared by both wire formats (`evaluate`) and `POST /v1/systemone` (`decide`,
//! the Jev/SystemOne format).

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use indexmap::IndexMap;

use super::schema::{
    self, answer_with_legend, Answer, AnswerOut, DecisionRequest, DecisionResponse, QuestionType,
    ThinkingMode, Usage,
};
use super::DecisionError;
use crate::AppState;
use cascadia_types::{DecisionReadout, ReadoutHead, MAX_DECISION_PROMPT};
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
        Ok(Evaluated {
            answers,
            mut thinking,
            input_tokens,
            output_tokens,
        }) => Json(DecisionResponse {
            model: state.model_id.clone(),
            answers: answers
                .into_iter()
                .map(|(id, answer)| {
                    let thinking = thinking.shift_remove(&id);
                    (id, AnswerOut { answer, thinking })
                })
                .collect(),
            usage: Usage {
                input_tokens,
                output_tokens,
            },
        })
        .into_response(),
        Err(e) => err(e),
    }
}

/// What [`evaluate`] returns: the answers in request order, System 2's report for each answer
/// that has one, the prompt tokens prefilled and the tokens System 2 generated.
pub(crate) struct Evaluated {
    pub answers: IndexMap<String, Answer>,
    pub thinking: IndexMap<String, schema::Thinking>,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A request's System 2 settings resolved against the profile (serve_decide `decide`), and the
/// chat template's frame its prompts go in.
struct ThinkPlan {
    mode: ThinkingMode,
    threshold: f64,
    budget: usize,
    reasoning: bool,
    mix: f64,
    pre: String,
    post: String,
}

/// One question's System 2 prompt and its length in tokens.
struct ThinkPrompt {
    qid: String,
    text: String,
    len: usize,
    /// Its worst case was reserved up front. Always true under `on` (or the request is a 413);
    /// under `auto` a question that did not fit answers from System 1 (`reason: "budget"`).
    fits: bool,
}

/// The default `think_budget`: 4096 tokens, or `max_tokens` when smaller. The reference's
/// default is the context window, which on our hardware is hours of decoding per question and
/// more than a request's `max_total_tokens` can hold.
pub const DEFAULT_THINK_BUDGET: usize = 4096;

/// The request's [`ThinkPlan`], or `None` when nothing thinks: the mode is `off` (the default),
/// or the codec has no System 2 and no System 2 field was set. Cheap checks only, run before the
/// request takes a permit; every refusal is a 400.
fn plan_thinking(
    state: &AppState,
    svc: &super::DecisionService,
    req: &DecisionRequest,
) -> Result<Option<ThinkPlan>, DecisionError> {
    let bad = |m: String| Err(DecisionError::BadRequest(m));
    let Some(jev) = svc.codec.as_jev() else {
        let set = req.thinking.is_some_and(|m| m != ThinkingMode::Off)
            || req.threshold.is_some()
            || req.think_budget.is_some()
            || req.return_reasoning == Some(true);
        if set {
            return bad(
                "thinking, threshold, think_budget and return_reasoning apply to JEV only".into(),
            );
        }
        return Ok(None);
    };
    let profile = jev.system2();
    let mode = req.thinking.unwrap_or(profile.thinking);
    if mode == ThinkingMode::Off {
        return Ok(None);
    }
    // A tree that cannot think at all (a profile default that thinks is refused at load).
    if let Some(why) = jev.system2_refusal() {
        return bad(why);
    }
    // Disabled at startup (with a warning) when the template is missing or does not render: a
    // profile default then serves System 1 alone; asking for thinking is a 400.
    let Some(frame) = state.think_frame.clone() else {
        if req.thinking.is_some() {
            return bad(
                "thinking is disabled on this server: the model's chat template did not load or \
                 render (see the startup log)"
                    .into(),
            );
        }
        return Ok(None);
    };
    // serve_decide refuses a score question asked to think; under the profile's default it just
    // answers without thinking.
    if req.thinking.is_some() {
        if let Some((id, _)) = req
            .questions
            .iter()
            .find(|(_, q)| q.qtype == QuestionType::Score)
        {
            return bad(format!(
                "thinking is supported for noul and choice questions; {id} is a score question"
            ));
        }
    }
    let labels = jev.answer_label_count();
    if let Some((id, n)) = req
        .questions
        .iter()
        .filter(|(_, q)| q.qtype == QuestionType::Choice)
        .map(|(id, q)| (id, q.choice_criteria().len()))
        .find(|&(_, n)| n > labels)
    {
        return bad(format!(
            "{id}: thinking labels at most {labels} options, got {n}"
        ));
    }
    let threshold = req.threshold.unwrap_or(profile.threshold);
    if !(0.0..=1.0).contains(&threshold) {
        return bad(format!("threshold must be in [0, 1], got {threshold}"));
    }
    let budget = req
        .think_budget
        .unwrap_or(DEFAULT_THINK_BUDGET.min(svc.max_tokens));
    if budget == 0 || budget > svc.max_tokens {
        return bad(format!(
            "think_budget must be 1..={} tokens, got {budget}",
            svc.max_tokens
        ));
    }
    let (pre, post) = (frame.0.clone(), frame.1.clone());
    Ok(Some(ThinkPlan {
        mode,
        threshold,
        budget,
        reasoning: req.return_reasoning.unwrap_or(false),
        mix: profile.mix,
        pre,
        post,
    }))
}

/// Every question that may think (all but score questions) with its System 2 prompt, and the
/// tokens the read-back appends after the thought. Encoded up front, so the request's budget
/// counts System 2 before its first pass.
fn think_prompts(
    svc: &super::DecisionService,
    req: &DecisionRequest,
    pre: &str,
    post: &str,
    cancel: &AtomicBool,
) -> Result<(Vec<ThinkPrompt>, usize), DecisionError> {
    let Some(jev) = svc.codec.as_jev() else {
        return Err(DecisionError::Engine("decision engine error".into()));
    };
    let state = jev.rendered_state(req);
    let tail = jev.encode_text(&jev.read_back_tail())?.len();
    let mut out = Vec::new();
    for (qid, q) in &req.questions {
        super::check_cancel(cancel)?;
        if q.qtype == QuestionType::Score {
            continue;
        }
        let text = jev.think_prompt(req, qid, &state, pre, post);
        let len = jev.encode_text(&text)?.len();
        out.push(ThinkPrompt {
            qid: qid.clone(),
            text,
            len,
            fits: false,
        });
    }
    Ok((out, tail))
}

/// The core both wire formats share: validate, encode, budget, run every pass (System 2's
/// included), and return the answers in request order. Errors are `DecisionError`s, so each
/// route renders them in its own format (SystemOne `{"error": msg}`, OpenAI `{"error": {...}}`).
pub(crate) async fn evaluate(
    state: &AppState,
    req: &DecisionRequest,
) -> Result<Evaluated, DecisionError> {
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
    let plan = plan_thinking(state, &svc, req)?;
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
    // System 2's prompts, for every question that may think (under `auto`, before System 1 says
    // which will).
    let thinks = match &plan {
        Some(p) => {
            let (svc2, req2, cancel2) = (svc.clone(), req.clone(), cancel.0.clone());
            let (pre, post) = (p.pre.clone(), p.post.clone());
            let prompts = off_executor(&svc, &held, move || {
                think_prompts(&svc2, &req2, &pre, &post, &cancel2)
            })
            .await??;
            Some(prompts)
        }
        None => None,
    };
    // A second round (JEV's tournament finals) is counted now, at its worst case, so a request
    // is refused before its first pass rather than half way.
    let total = super::prefill_work(&passes, svc.prefix_reuse) + reserve;
    if total > svc.max_total_tokens {
        return Err(DecisionError::TooLong(format!(
            "this request needs {total} prefill tokens; the per-request limit is {}",
            svc.max_total_tokens
        )));
    }
    // System 2 too, per question that may think, at its worst case: its prompt and
    // `think_budget` generated, then the read-back over both plus the tail. Its prompt part is
    // capped like any pass, the read-back by the engine's decision prompt cap (the model's
    // window, or MAX_DECISION_PROMPT). `on` refuses a request that cannot afford every one
    // (413); `auto` reserves question by question and lets one that does not fit answer from
    // System 1 alone, so `auto` never makes a request fail.
    let mut thinks = thinks;
    if let (Some(p), Some((prompts, tail))) = (&plan, &mut thinks) {
        let cap = svc
            .codec
            .as_jev()
            .map_or(MAX_DECISION_PROMPT, |j| j.read_back_cap());
        let strict = p.mode == ThinkingMode::On;
        let mut left = svc.max_total_tokens - total;
        let mut think_need = 0usize;
        for t in prompts.iter_mut() {
            let read_back = t.len + p.budget + *tail;
            let need = t.len + p.budget + read_back;
            t.fits = t.len + *tail <= svc.max_tokens && read_back <= cap && need <= left;
            if t.fits {
                left -= need;
            } else if strict {
                if t.len + *tail > svc.max_tokens {
                    return Err(super::pass_too_long(t.len + *tail, svc.max_tokens));
                }
                if read_back > cap {
                    return Err(DecisionError::TooLong(format!(
                        "with think_budget {} a read-back pass can reach {read_back} tokens; \
                         the engine's limit (the model's window, at most \
                         {MAX_DECISION_PROMPT}) is {cap}",
                        p.budget
                    )));
                }
            }
            think_need += need;
        }
        if strict && prompts.iter().any(|t| !t.fits) {
            return Err(DecisionError::TooLong(format!(
                "this request needs up to {} tokens, {think_need} of them for thinking at \
                 think_budget {}; the per-request limit is {} (lower think_budget)",
                total + think_need,
                p.budget,
                svc.max_total_tokens
            )));
        }
    }
    dump_passes(state, &svc, &req, &held, &passes, 1).await;
    super::mark_required_reuse(&mut passes, svc.max_total_tokens, svc.prefix_reuse);
    // Every question's per-pass probabilities, in submission order.
    let mut probs: IndexMap<String, Vec<Vec<f64>>> = IndexMap::new();
    let mut input_tokens = run_passes(state, &svc, &req, &passes, &mut probs).await?;
    // The tokens spent so far by our own count (System 2's checks continue from it).
    let mut counted = super::prefill_work(&passes, svc.prefix_reuse);
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
        counted = spent;
    }
    let mut dists: IndexMap<String, Vec<f64>> = IndexMap::with_capacity(req.questions.len());
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
        dists.insert(id.clone(), p);
    }
    let mut thinking = IndexMap::new();
    let mut output_tokens = 0u64;
    if let (Some(plan), Some((prompts, _))) = (&plan, thinks) {
        for t in prompts {
            let report = think(
                state,
                &svc,
                &held,
                &req,
                plan,
                &t,
                &mut dists,
                &mut counted,
                &mut input_tokens,
                &mut output_tokens,
            )
            .await?;
            thinking.insert(t.qid, report);
        }
    }
    let mut answers = IndexMap::with_capacity(req.questions.len());
    for (id, q) in &req.questions {
        let legend = match q.qtype {
            schema::QuestionType::Score => svc.codec.legend(q),
            _ => None,
        };
        answers.insert(id.clone(), answer_with_legend(q, &dists[id], legend));
    }
    Ok(Evaluated {
        answers,
        thinking,
        input_tokens,
        output_tokens,
    })
}

/// System 2 for one question (serve_decide `decide` and `s2_dist`): when the mode is `on`, or
/// `auto` and System 1's leading probability is below the threshold, the base model thinks
/// ([`generate_thought`]), one readout scores the answer labels after the thought, and the
/// question's distribution becomes `(1 - mix) * p1 + mix * p2`. Returns the answer's report.
#[allow(clippy::too_many_arguments)]
async fn think(
    state: &AppState,
    svc: &Arc<super::DecisionService>,
    held: &Arc<Held>,
    req: &Arc<DecisionRequest>,
    plan: &ThinkPlan,
    t: &ThinkPrompt,
    dists: &mut IndexMap<String, Vec<f64>>,
    spent: &mut usize,
    input_tokens: &mut u64,
    output_tokens: &mut u64,
) -> Result<schema::Thinking, DecisionError> {
    let Some(jev) = svc.codec.as_jev() else {
        return Err(DecisionError::Engine("decision engine error".into()));
    };
    let p1 = dists[&t.qid].clone();
    let lead = p1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let wanted = plan.mode == ThinkingMode::On || lead < plan.threshold;
    let mut report = schema::Thinking {
        mode: plan.mode,
        used: wanted && t.fits,
        reason: (wanted && !t.fits).then_some(BUDGET),
        threshold: plan.threshold,
        budget: plan.budget,
        think_tokens: None,
        finished_within_budget: None,
        reasoning: None,
    };
    if !report.used {
        return Ok(report);
    }
    let Some(end_think) = jev.end_think_id() else {
        return Err(DecisionError::Engine("decision engine error".into()));
    };
    let thought = generate_thought(state, &t.text, plan.budget, end_think).await?;
    *spent += t.len + thought.ids.len();
    *input_tokens += thought.prompt_tokens.unwrap_or(t.len as u64);
    *output_tokens += thought.ids.len() as u64;
    state.stats.tokens_total.fetch_add(
        thought.ids.len() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    // The thought without its closing token, as text (the read-back writes `end_think` itself,
    // as serve_decide does after a thought that ran out of budget too).
    let body: Vec<u32> = match thought.ids.split_last() {
        Some((&last, rest)) if last == end_think => rest.to_vec(),
        _ => thought.ids.clone(),
    };
    let (svc2, prompt) = (svc.clone(), t.text.clone());
    let (text, prompt_ids) = off_executor(svc, held, move || {
        let jev = svc2
            .codec
            .as_jev()
            .ok_or_else(|| DecisionError::Engine("decision engine error".into()))?;
        let text = jev.decode(&body)?;
        let ids = jev.encode_text(&jev.read_back_text(&prompt, &text))?;
        Ok::<_, DecisionError>((text, ids))
    })
    .await??;
    // Within the worst case admission counted whenever re-encoding the thought gives back no
    // more tokens than were generated; a tokenizer where it does not stops here rather than
    // overspend: `auto` answers from System 1 (the thought is still counted in usage), `on` is
    // a 413.
    let cap = jev.read_back_cap();
    let over = prompt_ids.len() > cap || *spent + prompt_ids.len() > svc.max_total_tokens;
    if over && plan.mode == ThinkingMode::Auto {
        report.used = false;
        report.reason = Some(BUDGET);
        return Ok(report);
    }
    if prompt_ids.len() > cap {
        return Err(super::pass_too_long(prompt_ids.len(), cap));
    }
    if *spent + prompt_ids.len() > svc.max_total_tokens {
        return Err(DecisionError::TooLong(format!(
            "this request needs {} tokens with thinking; the per-request limit is {}",
            *spent + prompt_ids.len(),
            svc.max_total_tokens
        )));
    }
    let q = &req.questions[&t.qid];
    let n = q.option_keys().len();
    let readout = DecisionReadout {
        prompt_ids,
        head: ReadoutHead::Verbalizer {
            token_ids: jev.answer_ids(n),
        },
        cache_prefix: None,
        require_warm: false,
        needs_snapshot: false,
    };
    // The base model reads the answer, as it thought.
    let (logits, prefilled) = run_readout(state, &readout, Some(0.0), || {
        engine_failure("decision_uncacheable on a System 2 read-back")
    })
    .await?;
    *spent += readout.prompt_ids.len();
    *input_tokens += prefilled;
    let p2 = super::jev::system2_probabilities(q.qtype, &logits);
    dists.insert(t.qid.clone(), super::jev::blend(&p1, &p2, plan.mix));
    report.think_tokens = Some(thought.ids.len());
    report.finished_within_budget = Some(thought.finished);
    if plan.reasoning {
        report.reasoning = Some(text.trim().to_string());
    }
    Ok(report)
}

/// `thinking.reason` of a question whose System 2 the request's budget could not cover.
const BUDGET: &str = "budget";

/// What System 2's generation produced.
struct Thought {
    /// The generated ids, the closing `end_think` included when the thought closed.
    ids: Vec<u32>,
    /// It closed, or the model stopped, before the budget ran out.
    finished: bool,
    /// The engine's prompt count, when its final chunk arrived (it does not after a close).
    prompt_tokens: Option<u64>,
}

/// System 2's generation: the base model (`lora_scale = 0`, a switchable tree) decodes greedily
/// from `prompt` (thinking on, so the engine appends no empty think block) for at most `budget`
/// tokens, until `end_think`. The engine does not stop at a token id, so the stream is dropped
/// there, which cancels the task (serve_decide's `stop_token_ids`).
async fn generate_thought(
    state: &AppState,
    prompt: &str,
    budget: usize,
    end_think: u32,
) -> Result<Thought, DecisionError> {
    let task = cascadia_types::GenerationTask {
        task_id: format!("think-{}", uuid::Uuid::new_v4().simple()),
        prompt: prompt.to_string(),
        max_tokens: u32::try_from(budget).unwrap_or(u32::MAX),
        temperature: 0.0,
        logprobs: 0,
        sampling: Default::default(),
        enable_thinking: true,
        trust_remote_code: false,
        // Never client-settable (see the chat handler): a KV-namespace boundary.
        tenant: String::new(),
        resume_token_ids: None,
        decision: None,
        lora_scale: Some(0.0),
    };
    let mut stream = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(e) => return Err(decision_engine_error(e)),
    };
    let mut ids = Vec::new();
    while let Some(chunk) = stream.next().await {
        if let Some(reason) = &chunk.error {
            return Err(engine_failure(reason));
        }
        let got = if chunk.token_ids.is_empty() && !chunk.is_final {
            vec![chunk.token_id]
        } else {
            chunk.token_ids.clone()
        };
        for id in got {
            let Ok(id) = u32::try_from(id) else {
                return Err(engine_failure("engine returned a negative token id"));
            };
            ids.push(id);
            if id == end_think {
                return Ok(Thought {
                    ids,
                    finished: true,
                    prompt_tokens: None,
                });
            }
        }
        if chunk.is_final {
            let finished = chunk.finish_reason != Some(cascadia_types::FinishReason::Length)
                && ids.len() < budget;
            return Ok(Thought {
                ids,
                finished,
                prompt_tokens: chunk.prompt_tokens.map(u64::from),
            });
        }
    }
    Err(engine_failure(
        "System 2 generation ended without a final chunk",
    ))
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
        // System 1 runs at the tree's trained scale.
        let (logits, prefilled) = run_readout(state, &pass.readout, None, || {
            uncacheable(passes, svc.max_total_tokens)
        })
        .await?;
        input_tokens += prefilled;
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

/// Run one readout at `lora_scale` (`None`: the tree's trained scale) and return its checked
/// logits and the prompt tokens it prefilled. `uncacheable` is the error for a shared prefix the
/// engine's cache can never hold.
async fn run_readout(
    state: &AppState,
    readout: &DecisionReadout,
    lora_scale: Option<f32>,
    uncacheable: impl FnOnce() -> DecisionError,
) -> Result<(Vec<f32>, u64), DecisionError> {
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
        decision: Some(readout.clone()),
        lora_scale,
    };
    let mut stream = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(e) => return Err(decision_engine_error(e)),
    };
    let mut logits = None;
    let mut prefilled = 0u64;
    while let Some(chunk) = stream.next().await {
        if let Some(reason) = &chunk.error {
            // The cache can never hold this request's shared prefix (refused before the first
            // pass prefilled anything): the request needs its full length, a 413.
            if reason.starts_with("decision_uncacheable") {
                tracing::warn!(reason, "decision pass failed");
                return Err(uncacheable());
            }
            return Err(engine_failure(reason));
        }
        if chunk.is_final {
            prefilled = chunk
                .prompt_tokens
                .unwrap_or(readout.prompt_ids.len() as u32) as u64;
            logits = chunk.decision_logits.clone();
        }
    }
    let Some(logits) = logits.filter(|l| l.len() == readout.head.output_len()) else {
        return Err(engine_failure(
            "engine returned no (or mis-sized) decision logits",
        ));
    };
    // NaN/±inf (an fp16 overflow, a degenerate head) would softmax to NaN, serialise as
    // `null` and argmax to the first option: a silent wrong 200. Fail the request instead.
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(engine_failure("non-finite decision logits"));
    }
    Ok((logits, prefilled))
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
