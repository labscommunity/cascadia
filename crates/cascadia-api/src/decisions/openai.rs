//! OpenAI Decisions API wire format (developers.openai.com/api/docs/guides/decisions) at
//! POST /v1/decisions. A thin adapter over the shared core: the models see the same prompts as
//! through /v1/systemone.
use std::collections::HashSet;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::schema::{
    parse_ordered, Answer, DecisionRequest, OrderedValue, Question, QuestionType, RequestStrategy,
};
use super::DecisionError;
use crate::AppState;
use cascadia_types::MAX_DECISION_OUTPUTS;

#[derive(Clone, Debug, Deserialize)]
pub struct OaRequest {
    #[serde(default)]
    pub model: String,
    pub input: OaInput,
    pub questions: Vec<OaQuestion>,
    /// As `/v1/systemone`'s top-level `strategy` (JEV only).
    #[serde(default)]
    pub strategy: Option<RequestStrategy>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OaInput {
    Text(String),
    Messages(Vec<OaMessage>),
}

#[derive(Clone, Debug, Deserialize)]
pub struct OaMessage {
    #[serde(default = "user_role")]
    pub role: String,
    pub content: OaContent,
}

fn user_role() -> String {
    "user".into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OaContent {
    Text(String),
    Parts(Vec<OaPart>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OaPart {
    InputText { text: String },
    InputImage { image_url: String },
}

/// Tagged by `type`, but not with `#[serde(tag)]`: an internally tagged enum buffers the whole
/// question through serde's f64 number path, which refuses an integer of 310+ digits ("number
/// out of range") before any field is read. See the `Deserialize` impl below.
#[derive(Clone, Debug)]
pub enum OaQuestion {
    Predicate {
        name: String,
        instructions: String,
    },
    Choice {
        name: String,
        instructions: String,
        choices: Vec<OaChoice>,
    },
    Score {
        name: String,
        instructions: String,
        levels: Vec<OaLevel>,
    },
}

#[derive(Deserialize)]
struct OaQuestionTag<'a> {
    #[serde(rename = "type", borrow)]
    qtype: std::borrow::Cow<'a, str>,
}

#[derive(Deserialize)]
struct OaPredicateFields {
    name: String,
    instructions: String,
}

#[derive(Deserialize)]
struct OaChoiceFields {
    name: String,
    instructions: String,
    choices: Vec<OaChoice>,
}

#[derive(Deserialize)]
struct OaScoreFields {
    name: String,
    instructions: String,
    levels: Vec<OaLevel>,
}

/// Reads the question's raw text twice, never buffering it: once for `type` (other fields are
/// skipped unparsed), once into that type's fields (unknown fields, `type` included, ignored, as
/// the tagged derive did). Needs a serde_json deserializer, which [`parse_request`] is.
impl<'de> Deserialize<'de> for OaQuestion {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let raw = <&serde_json::value::RawValue>::deserialize(d)?;
        // The outer error adds its own position; keep only the message of the inner one.
        let inner = |e: serde_json::Error| {
            let m = e.to_string();
            D::Error::custom(match m.rfind(" at line ") {
                Some(i) => m[..i].to_string(),
                None => m,
            })
        };
        let tag: OaQuestionTag = serde_json::from_str(raw.get()).map_err(inner)?;
        Ok(match tag.qtype.as_ref() {
            "predicate" => {
                let f: OaPredicateFields = serde_json::from_str(raw.get()).map_err(inner)?;
                OaQuestion::Predicate {
                    name: f.name,
                    instructions: f.instructions,
                }
            }
            "choice" => {
                let f: OaChoiceFields = serde_json::from_str(raw.get()).map_err(inner)?;
                OaQuestion::Choice {
                    name: f.name,
                    instructions: f.instructions,
                    choices: f.choices,
                }
            }
            "score" => {
                let f: OaScoreFields = serde_json::from_str(raw.get()).map_err(inner)?;
                OaQuestion::Score {
                    name: f.name,
                    instructions: f.instructions,
                    levels: f.levels,
                }
            }
            other => {
                return Err(D::Error::unknown_variant(
                    &echo(other),
                    &["predicate", "choice", "score"],
                ))
            }
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OaChoice {
    /// A string or a number, exactly as `/v1/systemone` would parse the same literal. Serde
    /// would read a number through f64, so the derive only checks that the field is there (and
    /// leaves `Null`, skipping the literal unparsed); `parse_request` fills it from the body's
    /// raw text. `to_internal` rejects any other JSON type.
    #[serde(deserialize_with = "present")]
    pub value: OrderedValue,
    #[serde(default)]
    pub description: Option<String>,
}

fn present<'de, D: serde::Deserializer<'de>>(d: D) -> Result<OrderedValue, D::Error> {
    serde::de::IgnoredAny::deserialize(d).map(|_| OrderedValue::Null)
}

#[derive(Clone, Debug, Deserialize)]
pub struct OaLevel {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// What the reply needs back per question: its name, the caller's original choice values (their
/// JSON type is echoed, not the internal string key), and the level labels.
#[derive(Clone, Debug)]
pub struct OaMeta {
    pub name: String,
    pub choice_values: Vec<OrderedValue>,
    pub level_labels: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OaChoiceProb {
    pub value: OrderedValue,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OaLevelProb {
    pub value: usize,
    pub label: String,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OaAnswer {
    Predicate {
        name: String,
        probability: f64,
    },
    Choice {
        name: String,
        choice: OrderedValue,
        probabilities: Vec<OaChoiceProb>,
        confidence: f64,
    },
    Score {
        name: String,
        score: f64,
        probabilities: Vec<OaLevelProb>,
        confidence: f64,
    },
}

/// Token counts in the shape OpenAI's SDKs read. `output_tokens` is always 0: decision models do
/// not decode.
#[derive(Clone, Debug, Serialize)]
pub struct OaUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OaResponse {
    pub id: String,
    pub answers: Vec<OaAnswer>,
    pub usage: OaUsage,
}

fn bad(m: impl Into<String>) -> DecisionError {
    DecisionError::BadRequest(m.into())
}

fn input_text(input: &OaInput) -> Result<String, DecisionError> {
    match input {
        OaInput::Text(t) => Ok(t.clone()),
        OaInput::Messages(msgs) => {
            let mut out = Vec::with_capacity(msgs.len());
            for m in msgs {
                if m.role != "user" {
                    return Err(bad(format!(
                        "input messages must have role \"user\", got {:?}",
                        m.role
                    )));
                }
                out.push(match &m.content {
                    OaContent::Text(t) => t.clone(),
                    OaContent::Parts(parts) => {
                        let mut s = String::new();
                        for p in parts {
                            match p {
                                OaPart::InputText { text } => s.push_str(text),
                                OaPart::InputImage { .. } => {
                                    return Err(bad(
                                        "input_image is not supported (text-only server)",
                                    ))
                                }
                            }
                        }
                        s
                    }
                });
            }
            Ok(out.join("\n"))
        }
    }
}

/// A choice value as the internal model holds it: strings and numbers only. Numbers come from
/// `parse_ordered`, so `-0` is Python's int 0 and `0` and `-0` are one value.
fn choice_value(name: &str, v: &OrderedValue) -> Result<OrderedValue, DecisionError> {
    match v {
        OrderedValue::String(_) | OrderedValue::Number(_) | OrderedValue::BigInt(_) => {
            Ok(v.clone())
        }
        _ => Err(bad(format!(
            "{name}: choice values must be strings or numbers"
        ))),
    }
}

/// The internal criteria key: the string itself, or Python's `str()` of a number (`2` → `"2"`,
/// `1e5` → `"100000.0"`, every digit of a big int), as the reference implementations key a
/// parsed number.
fn choice_key(v: &OrderedValue) -> String {
    match v {
        OrderedValue::String(s) => s.clone(),
        other => other.py_json(false, true),
    }
}

/// OpenAI request → the internal model every codec consumes, plus what the reply needs.
pub fn to_internal(r: &OaRequest) -> Result<(DecisionRequest, Vec<OaMeta>), DecisionError> {
    let mut questions: IndexMap<String, Question> = IndexMap::new();
    let mut meta = Vec::with_capacity(r.questions.len());
    for q in &r.questions {
        let (name, instructions) = match q {
            OaQuestion::Predicate { name, instructions }
            | OaQuestion::Choice {
                name, instructions, ..
            }
            | OaQuestion::Score {
                name, instructions, ..
            } => (name, instructions),
        };
        if name.is_empty() {
            return Err(bad("every question needs a non-empty name"));
        }
        if questions.contains_key(name) {
            return Err(bad(format!("duplicate question name {name:?}")));
        }
        let mut m = OaMeta {
            name: name.clone(),
            choice_values: Vec::new(),
            level_labels: Vec::new(),
        };
        let (qtype, criteria) = match q {
            OaQuestion::Predicate { .. } => (QuestionType::Noul, None),
            OaQuestion::Choice { choices, .. } => {
                // Cap before any per-choice work: this runs on the async executor, before
                // `evaluate` applies the profile's `max_options` and before a permit is taken.
                if choices.len() > MAX_DECISION_OUTPUTS {
                    return Err(bad(format!(
                        "{name}: at most {MAX_DECISION_OUTPUTS} choices are allowed, got {}",
                        choices.len()
                    )));
                }
                let mut crit: Vec<(String, OrderedValue)> = Vec::with_capacity(choices.len());
                let mut seen: HashSet<String> = HashSet::with_capacity(choices.len());
                for c in choices {
                    let value = choice_value(name, &c.value)?;
                    let key = choice_key(&value);
                    if !seen.insert(key.clone()) {
                        return Err(bad(format!("{name}: duplicate choice value {key:?}")));
                    }
                    crit.push((
                        key,
                        c.description
                            .clone()
                            .map(OrderedValue::String)
                            .unwrap_or(OrderedValue::Null),
                    ));
                    m.choice_values.push(value);
                }
                (QuestionType::Choice, Some(OrderedValue::Object(crit)))
            }
            OaQuestion::Score { levels, .. } => {
                if levels.len() > MAX_DECISION_OUTPUTS {
                    return Err(bad(format!(
                        "{name}: at most {MAX_DECISION_OUTPUTS} levels are allowed, got {}",
                        levels.len()
                    )));
                }
                let texts = levels
                    .iter()
                    .map(|l| {
                        m.level_labels.push(l.label.clone());
                        // One text per level, as the models were trained; see docs/decisions.md.
                        OrderedValue::String(
                            match l.description.as_deref().filter(|d| !d.is_empty()) {
                                Some(d) => format!("{}: {d}", l.label),
                                None => l.label.clone(),
                            },
                        )
                    })
                    .collect();
                (QuestionType::Score, Some(OrderedValue::Array(texts)))
            }
        };
        questions.insert(
            name.clone(),
            Question {
                qtype,
                instructions: Some(OrderedValue::String(instructions.clone())),
                criteria,
            },
        );
        meta.push(m);
    }
    let req = DecisionRequest {
        model: r.model.clone(),
        state: OrderedValue::String(input_text(&r.input)?),
        questions,
        images: None,
        videos: None,
        strategy: r.strategy,
    };
    Ok((req, meta))
}

/// Internal answers (request order) → the OpenAI response, with a fresh id and the prompt tokens
/// the core prefilled. A question the core left unanswered is a server error, not a panic.
pub fn to_openai(
    answers: &IndexMap<String, Answer>,
    meta: &[OaMeta],
    input_tokens: u64,
) -> Result<OaResponse, DecisionError> {
    let mut out = Vec::with_capacity(meta.len());
    for m in meta {
        let answer = answers.get(&m.name).ok_or_else(|| {
            DecisionError::Engine(format!(
                "internal error: no answer for question {:?}",
                m.name
            ))
        })?;
        out.push(match answer {
            Answer::Noul { noul } => OaAnswer::Predicate {
                name: m.name.clone(),
                probability: *noul,
            },
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                let idx = probabilities.get_index_of(choice).unwrap_or(0);
                OaAnswer::Choice {
                    name: m.name.clone(),
                    choice: m.choice_values[idx].clone(),
                    probabilities: probabilities
                        .values()
                        .zip(&m.choice_values)
                        .map(|(p, v)| OaChoiceProb {
                            value: v.clone(),
                            probability: *p,
                        })
                        .collect(),
                    confidence: *confidence,
                }
            }
            Answer::Score {
                score,
                confidence,
                probabilities,
                ..
            } => OaAnswer::Score {
                name: m.name.clone(),
                score: *score,
                probabilities: probabilities
                    .values()
                    .enumerate()
                    .map(|(i, p)| OaLevelProb {
                        value: i,
                        label: m.level_labels.get(i).cloned().unwrap_or_default(),
                        probability: *p,
                    })
                    .collect(),
                confidence: *confidence,
            },
        });
    }
    Ok(OaResponse {
        id: format!("dec-{}", uuid::Uuid::new_v4().simple()),
        answers: out,
        usage: OaUsage {
            input_tokens,
            output_tokens: 0,
            total_tokens: input_tokens,
        },
    })
}

/// OpenAI's error envelope, which its SDKs parse into typed errors: `status` and
/// `{"error": {"message", "type", "code": null}}`. `kind` is `invalid_request_error` for the
/// caller's fault, `server_error` otherwise. For errors that are not a [`DecisionError`] (the
/// OSS route wraps the framework's body rejections in it); a `DecisionError` is [`openai_error`].
pub fn openai_envelope(
    status: StatusCode,
    kind: &str,
    message: String,
) -> (StatusCode, serde_json::Value) {
    (
        status,
        serde_json::json!({"error": {"message": message, "type": kind, "code": null}}),
    )
}

/// A `/v1/decisions` error as the OSS server returns it, in [`openai_envelope`]. Stable: these
/// are the route's documented statuses, types and messages.
pub fn openai_error(e: &DecisionError) -> (StatusCode, serde_json::Value) {
    let kind = match e {
        DecisionError::BadRequest(_) | DecisionError::TooLong(_) => "invalid_request_error",
        DecisionError::NotServed | DecisionError::Engine(_) => "server_error",
    };
    openai_envelope(e.status(), kind, e.to_string())
}

/// A body the OSS route's `Json` extractor rejected (content type, body size, JSON syntax), as
/// the route returns it: the rejection's own status and text in [`openai_envelope`].
pub fn openai_rejection(r: &JsonRejection) -> (StatusCode, serde_json::Value) {
    openai_envelope(r.status(), "invalid_request_error", r.body_text())
}

/// The `/v1/decisions` request from the raw body, exactly as the OSS route reads it once the
/// content type and body size have passed (the extractor's JSON read, then [`parse_request`]);
/// on failure, the status and envelope the route returns. The 415 and 413 the route sends before
/// reading the body are the embedder's own (see docs/decisions.md, "Embedding the decision core").
pub fn openai_parse(body: &[u8]) -> Result<OaRequest, (StatusCode, serde_json::Value)> {
    let raw = match Json::<Box<serde_json::value::RawValue>>::from_bytes(body) {
        Ok(Json(raw)) => raw,
        Err(r) => return Err(openai_rejection(&r)),
    };
    parse_request(raw.get()).map_err(|e| openai_error(&e))
}

fn oa_response((status, body): (StatusCode, serde_json::Value)) -> Response {
    (status, Json(body)).into_response()
}

fn oa_err(e: DecisionError) -> Response {
    oa_response(openai_error(&e))
}

/// How much of a caller's unknown content-part or question `type` an error message echoes, in
/// characters.
const ECHO_CHARS: usize = 64;

/// `s` capped at [`ECHO_CHARS`] characters, with `…` when cut.
fn echo(s: &str) -> String {
    let mut shown: String = s.chars().take(ECHO_CHARS).collect();
    if shown.len() < s.len() {
        shown.push('…');
    }
    shown
}

/// The first content part in `body` that `OaPart` cannot read, named for the caller. The untagged
/// `OaContent` reports only that no variant matched, which does not say which part is at fault.
fn bad_content_part(body: &serde_json::Value) -> Option<String> {
    let msgs = body.get("input")?.as_array()?;
    for part in msgs
        .iter()
        .filter_map(|m| m.get("content")?.as_array())
        .flatten()
    {
        let Some(kind) = part.get("type") else {
            return Some("every content part needs a \"type\"".into());
        };
        let Some(kind) = kind.as_str() else {
            return Some("content part \"type\" must be a string".into());
        };
        let field = match kind {
            "input_text" => "text",
            "input_image" => "image_url",
            other => {
                return Some(format!("unsupported content part type '{}'", echo(other)));
            }
        };
        if !part.get(field).is_some_and(serde_json::Value::is_string) {
            return Some(format!("{kind} parts need a string \"{field}\""));
        }
    }
    None
}

/// The request from the body's raw JSON text. Read straight from the text, not via a
/// `serde_json::Value`, so a repeated key is a 400 (a `Value` keeps the last one) and number
/// choice values are exact: the body is read again by `parse_ordered`, the reader `/v1/systemone`
/// uses (every digit of a big int, correctly rounded floats), and each choice value is taken
/// from there.
pub fn parse_request(text: &str) -> Result<OaRequest, DecisionError> {
    let mut r = serde_json::from_str::<OaRequest>(text).map_err(|e| {
        let v = serde_json::from_str::<serde_json::Value>(text).unwrap_or_default();
        bad(bad_content_part(&v).unwrap_or_else(|| format!("invalid request body: {e}")))
    })?;
    // Only the choice values are read exactly: other fields (unknown ones included) are skipped
    // unread, as the typed parse skips them, so an out-of-range float there is not refused.
    // The typed parse refused repeated keys at every level read here, so each path is unique.
    let raw = serde_json::from_str::<RawBody>(text)
        .map_err(|e| bad(format!("invalid request body: {e}")))?;
    for (q, node) in r.questions.iter_mut().zip(raw.questions) {
        // Read only under a choice question (the typed parse checked its `choices`); any other
        // question's unknown `choices` field stays unread.
        let (OaQuestion::Choice { choices, .. }, Some(cs)) = (q, node.choices) else {
            continue;
        };
        let cs = serde_json::from_str::<Vec<RawChoice>>(cs.get())
            .map_err(|e| bad(format!("invalid request body: {e}")))?;
        for (c, cn) in choices.iter_mut().zip(cs) {
            c.value = match cn.value {
                Some(v) => {
                    parse_ordered(v.get()).map_err(|e| bad(format!("invalid request body: {e}")))?
                }
                None => OrderedValue::Null,
            };
        }
    }
    Ok(r)
}

/// The body read only down to each choice's raw `value` text ([`parse_request`]).
#[derive(Deserialize)]
struct RawBody<'a> {
    #[serde(borrow, default)]
    questions: Vec<RawQuestion<'a>>,
}

#[derive(Deserialize)]
struct RawQuestion<'a> {
    #[serde(borrow, default)]
    choices: Option<&'a serde_json::value::RawValue>,
}

#[derive(Deserialize)]
struct RawChoice<'a> {
    #[serde(borrow, default)]
    value: Option<&'a serde_json::value::RawValue>,
}

/// POST /v1/decisions: the OpenAI Decisions API wire format. The body is taken as raw JSON text
/// and parsed by `parse_request` rather than by `Json<OaRequest>`, whose schema rejection is a
/// 422 text body the SDKs do not parse; the framework's other rejections (content type, syntax,
/// body limit) keep their status but get the envelope too.
pub async fn decide(
    State(state): State<AppState>,
    body: Result<Json<Box<serde_json::value::RawValue>>, JsonRejection>,
) -> Response {
    let raw = match body {
        Ok(Json(raw)) => raw,
        Err(r) => return oa_response(openai_rejection(&r)),
    };
    let r = match parse_request(raw.get()) {
        Ok(r) => r,
        Err(e) => return oa_err(e),
    };
    let (req, meta) = match to_internal(&r) {
        Ok(x) => x,
        Err(e) => return oa_err(e),
    };
    match super::handler::evaluate(&state, &req).await {
        Ok((answers, input_tokens)) => match to_openai(&answers, &meta, input_tokens) {
            Ok(r) => Json(r).into_response(),
            Err(e) => oa_err(e),
        },
        Err(e) => oa_err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oa(s: &str) -> OaRequest {
        parse_request(s).unwrap()
    }

    fn one_choice(values: &str) -> String {
        format!(
            r#"{{"input": "x", "questions": [{{"type": "choice", "name": "c", "instructions": "x", "choices": [{values}]}}]}}"#
        )
    }

    #[test]
    fn number_choice_values_key_exactly_like_systemone() {
        // H16/L9: the key is Python's str() of the EXACT parsed literal, as `/v1/systemone`
        // renders the same literal: every digit of a big int, the correctly rounded 17-digit float
        // (serde_json's default parser misrounds this one), Python repr for the rest.
        let lits = [
            "123456789012345678901234567890",
            "0.18466034385487662",
            "1e20",
            "1.0",
            "-0.0",
            "-0",
        ];
        let body = one_choice(
            &lits
                .iter()
                .map(|l| format!(r#"{{"value": {l}}}"#))
                .collect::<Vec<_>>()
                .join(","),
        );
        let (r, meta) = to_internal(&oa(&body)).unwrap();
        assert_eq!(
            r.questions["c"].option_keys(),
            [
                "123456789012345678901234567890",
                "0.18466034385487662",
                "1e+20",
                "1.0",
                "-0.0",
                "0"
            ]
        );
        for (i, lit) in lits.iter().enumerate() {
            let so: OrderedValue = serde_json::from_str(lit).unwrap();
            assert_eq!(meta[0].choice_values[i], so, "{lit}");
        }
        // echoed with every digit
        assert_eq!(
            serde_json::to_string(&meta[0].choice_values[0]).unwrap(),
            "123456789012345678901234567890"
        );
    }

    #[test]
    fn only_choice_values_are_range_checked() {
        // Final review N1: an out-of-range float in an unknown field is skipped unread (as the
        // typed parse skips it and as at e109a582); in a choice value it is still a 400.
        let ok = r#"{"input": "x", "metadata": {"x": 1e400}, "questions": [
            {"type": "predicate", "name": "p", "instructions": "q", "choices": "not read"},
            {"type": "choice", "name": "c", "instructions": "q", "extra": 1e400,
             "choices": [{"value": 1.5, "note": -1e400}, {"value": "b"}]}]}"#;
        let r = parse_request(ok).unwrap();
        let OaQuestion::Choice { choices, .. } = &r.questions[1] else {
            panic!("choice")
        };
        assert_eq!(serde_json::to_string(&choices[0].value).unwrap(), "1.5");
        let bad_value = r#"{"input": "x", "questions": [{"type": "choice", "name": "c",
            "instructions": "q", "choices": [{"value": 1e400}, {"value": "b"}]}]}"#;
        match parse_request(bad_value) {
            Err(DecisionError::BadRequest(m)) => {
                assert!(m.starts_with("invalid request body: "), "{m}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_integer_choice_value_of_any_size_keeps_every_digit() {
        // Task 26 M1: 310+ digits overflow f64, which an internally tagged enum's buffering used
        // to hit ("number out of range"); `/v1/systemone` reads the literal as a big int.
        let big = format!("1{}", "0".repeat(400));
        let (r, meta) = to_internal(&oa(&one_choice(&format!(
            r#"{{"value": {big}}}, {{"value": "b"}}"#
        ))))
        .unwrap();
        assert_eq!(r.questions["c"].option_keys(), [big.as_str(), "b"]);
        assert_eq!(
            serde_json::to_string(&meta[0].choice_values[0]).unwrap(),
            big
        );
        // The question-level checks are unchanged: a missing field, an unknown type, a
        // non-string type.
        for (q, want) in [
            (
                r#"{"type": "choice", "name": "c", "choices": []}"#,
                "missing field `instructions`",
            ),
            (
                r#"{"type": "maybe", "name": "c"}"#,
                "unknown variant `maybe`, expected one of `predicate`, `choice`, `score`",
            ),
            (r#"{"name": "c"}"#, "missing field `type`"),
            (
                r#"{"type": 5, "name": "c"}"#,
                "invalid type: integer `5`, expected a string",
            ),
        ] {
            let body = format!(r#"{{"input": "x", "questions": [{q}]}}"#);
            match parse_request(&body) {
                Err(DecisionError::BadRequest(m)) => {
                    assert!(
                        m.starts_with(&format!("invalid request body: {want}")),
                        "{m}"
                    );
                    assert_eq!(m.matches(" at line ").count(), 1, "one position: {m}");
                }
                other => panic!("{q}: {other:?}"),
            }
        }
    }

    #[test]
    fn duplicate_keys_in_the_body_are_rejected() {
        // Task 25 review (a): a repeated top-level key used to be a 400, not last-one-wins.
        let q = r#""questions": [{"type": "predicate", "name": "a", "instructions": "x"}]"#;
        for body in [
            format!(r#"{{"input": "x", "input": "y", {q}}}"#),
            format!(r#"{{"input": "x", {q}, {q}}}"#),
            one_choice(r#"{"value": 1, "value": 2}"#),
        ] {
            match parse_request(&body) {
                Err(DecisionError::BadRequest(m)) => assert!(m.contains("duplicate field"), "{m}"),
                other => panic!("{body}: {other:?}"),
            }
        }
    }

    #[test]
    fn an_unknown_part_type_is_echoed_truncated() {
        // Task 25 review (b): the echoed `type` is capped at 64 characters.
        let long = "é".repeat(1000);
        let body = format!(
            r#"{{"input": [{{"role": "user", "content": [{{"type": "{long}"}}]}}], "questions": []}}"#
        );
        match parse_request(&body) {
            Err(DecisionError::BadRequest(m)) => assert_eq!(
                m,
                format!("unsupported content part type '{}…'", "é".repeat(64))
            ),
            other => panic!("{other:?}"),
        }
        // An unknown question type is capped the same way.
        let body = format!(r#"{{"input": "x", "questions": [{{"type": "{long}"}}]}}"#);
        match parse_request(&body) {
            Err(DecisionError::BadRequest(m)) => assert!(
                m.starts_with(&format!(
                    "invalid request body: unknown variant `{}…`, expected one of",
                    "é".repeat(64)
                )),
                "{m}"
            ),
            other => panic!("{other:?}"),
        }
        let body = r#"{"input": [{"role": "user", "content": [{"type": "input_file"}]}], "questions": []}"#;
        match parse_request(body) {
            Err(DecisionError::BadRequest(m)) => {
                assert_eq!(m, "unsupported content part type 'input_file'")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn maps_every_question_type_and_input_shape() {
        let (r, meta) = to_internal(&oa(r#"{"model": "gpt-6-luna",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "charged "}, {"type": "input_text", "text": "twice"}]},
                      {"role": "user", "content": "order 7"}],
            "questions": [
              {"type": "choice", "name": "department", "instructions": "Which department?",
               "choices": [{"value": "billing", "description": "Payments."}, {"value": 2, "description": "Two."}, {"value": "other"}]},
              {"type": "predicate", "name": "urgent", "instructions": "Urgent?"},
              {"type": "score", "name": "severity", "instructions": "How bad?",
               "levels": [{"label": "Cosmetic", "description": "Appearance only."}, {"label": "Blocked"}]}]}"#)).unwrap();
        assert_eq!(
            r.state,
            OrderedValue::String("charged twice\norder 7".into())
        );
        let ids: Vec<&str> = r.questions.keys().map(String::as_str).collect();
        assert_eq!(ids, ["department", "urgent", "severity"]);
        assert_eq!(
            r.questions["department"].option_keys(),
            ["billing", "2", "other"]
        );
        assert_eq!(
            r.questions["department"].choice_criteria()[2].1,
            OrderedValue::Null
        );
        assert_eq!(r.questions["urgent"].qtype, QuestionType::Noul);
        assert_eq!(
            r.questions["severity"].score_levels(),
            vec![
                OrderedValue::String("Cosmetic: Appearance only.".into()),
                OrderedValue::String("Blocked".into())
            ]
        );
        assert_eq!(meta[0].choice_values[1], OrderedValue::Number(2.into()));
        // number keys are Python str() of the parsed number, not the request's JSON text
        let (r2, _) = to_internal(&oa(r#"{"input": "x", "questions": [{"type": "choice", "name": "c", "instructions": "x", "choices": [{"value": 1e5}, {"value": 2.50}]}]}"#)).unwrap();
        assert_eq!(r2.questions["c"].option_keys(), ["100000.0", "2.5"]);
        assert_eq!(meta[2].level_labels, ["Cosmetic", "Blocked"]);
        // EDGE-4: `-0` is Python's int 0 (key "0"); `0.0` and `-0.0` keep their float keys.
        let (r3, m3) = to_internal(&oa(r#"{"input": "x", "questions": [{"type": "choice", "name": "c", "instructions": "x", "choices": [{"value": -0}, {"value": 0.0}, {"value": -0.0}]}]}"#)).unwrap();
        assert_eq!(r3.questions["c"].option_keys(), ["0", "0.0", "-0.0"]);
        assert_eq!(m3[0].choice_values[0], OrderedValue::Number(0.into()));
    }

    #[test]
    fn rejects_images_other_roles_and_bad_names() {
        let bad = |s: &str| matches!(to_internal(&oa(s)), Err(DecisionError::BadRequest(_)));
        assert!(bad(
            r#"{"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}], "questions": [{"type": "predicate", "name": "a", "instructions": "x"}]}"#
        ));
        assert!(bad(
            r#"{"input": [{"role": "system", "content": "x"}], "questions": [{"type": "predicate", "name": "a", "instructions": "x"}]}"#
        ));
        assert!(bad(
            r#"{"input": "x", "questions": [{"type": "predicate", "name": "a", "instructions": "x"}, {"type": "predicate", "name": "a", "instructions": "y"}]}"#
        ));
        assert!(bad(
            r#"{"input": "x", "questions": [{"type": "predicate", "name": "", "instructions": "x"}]}"#
        ));
        assert!(bad(
            r#"{"input": "x", "questions": [{"type": "choice", "name": "c", "instructions": "x", "choices": [{"value": "a"}, {"value": "a"}]}]}"#
        ));
        assert!(bad(
            r#"{"input": "x", "questions": [{"type": "choice", "name": "c", "instructions": "x", "choices": [{"value": true}, {"value": "a"}]}]}"#
        )); // `0` and `-0` are the same value.
        assert!(bad(
            r#"{"input": "x", "questions": [{"type": "choice", "name": "c", "instructions": "x", "choices": [{"value": 0}, {"value": -0}]}]}"#
        ));
    }

    #[test]
    fn oversized_choice_and_level_lists_are_rejected_before_the_duplicate_check() {
        use cascadia_types::MAX_DECISION_OUTPUTS;
        // Every value is the same: were the duplicate scan to run first, the error would name a
        // duplicate. The count cap must fire before any per-choice work.
        let choices = vec![r#"{"value": "a"}"#; MAX_DECISION_OUTPUTS + 1].join(",");
        let r = oa(&format!(
            r#"{{"input": "x", "questions": [{{"type": "choice", "name": "c", "instructions": "x", "choices": [{choices}]}}]}}"#
        ));
        match to_internal(&r) {
            Err(DecisionError::BadRequest(m)) => assert_eq!(
                m,
                format!("c: at most {MAX_DECISION_OUTPUTS} choices are allowed, got 4097")
            ),
            other => panic!("{other:?}"),
        }
        let levels = vec![r#"{"label": "l"}"#; MAX_DECISION_OUTPUTS + 1].join(",");
        let r = oa(&format!(
            r#"{{"input": "x", "questions": [{{"type": "score", "name": "s", "instructions": "x", "levels": [{levels}]}}]}}"#
        ));
        match to_internal(&r) {
            Err(DecisionError::BadRequest(m)) => assert_eq!(
                m,
                format!("s: at most {MAX_DECISION_OUTPUTS} levels are allowed, got 4097")
            ),
            other => panic!("{other:?}"),
        }
        // At the cap, distinct values still convert (the profile's own max_options applies later).
        let choices = (0..MAX_DECISION_OUTPUTS)
            .map(|i| format!(r#"{{"value": {i}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let r = oa(&format!(
            r#"{{"input": "x", "questions": [{{"type": "choice", "name": "c", "instructions": "x", "choices": [{choices}]}}]}}"#
        ));
        assert_eq!(
            to_internal(&r).unwrap().1[0].choice_values.len(),
            MAX_DECISION_OUTPUTS
        );
    }

    #[test]
    fn answers_render_in_openai_shape() {
        let (r, meta) = to_internal(&oa(r#"{"input": "x", "questions": [
            {"type": "choice", "name": "department", "instructions": "q", "choices": [{"value": "billing"}, {"value": "technical"}, {"value": "shipping"}, {"value": "other"}]},
            {"type": "score", "name": "severity", "instructions": "q", "levels": [{"label": "Cosmetic"}, {"label": "Workaround available"}, {"label": "Fully blocked"}]},
            {"type": "predicate", "name": "urgent", "instructions": "q"}]}"#)).unwrap();
        let mut answers = indexmap::IndexMap::new();
        answers.insert(
            "department".to_string(),
            crate::decisions::schema::answer_from_probs(
                &r.questions["department"],
                &[0.95, 0.02, 0.01, 0.02],
            ),
        );
        answers.insert(
            "severity".to_string(),
            crate::decisions::schema::answer_from_probs(&r.questions["severity"], &[0.1, 0.7, 0.2]),
        );
        answers.insert(
            "urgent".to_string(),
            crate::decisions::schema::answer_from_probs(&r.questions["urgent"], &[0.2, 0.8]),
        );
        let v = serde_json::to_value(to_openai(&answers, &meta, 7).unwrap()).unwrap();
        // OpenAI's documented examples, reproduced exactly (confidence 0.93 and 0.55 included)
        assert_eq!(
            v["answers"][0],
            serde_json::json!({"type": "choice", "name": "department", "choice": "billing",
            "probabilities": [{"value": "billing", "probability": 0.95}, {"value": "technical", "probability": 0.02},
                              {"value": "shipping", "probability": 0.01}, {"value": "other", "probability": 0.02}],
            "confidence": 0.9333})
        );
        assert_eq!(
            v["answers"][1],
            serde_json::json!({"type": "score", "name": "severity", "score": 1.1,
            "probabilities": [{"value": 0, "label": "Cosmetic", "probability": 0.1}, {"value": 1, "label": "Workaround available", "probability": 0.7},
                              {"value": 2, "label": "Fully blocked", "probability": 0.2}],
            "confidence": 0.55})
        );
        assert_eq!(
            v["answers"][2],
            serde_json::json!({"type": "predicate", "name": "urgent", "probability": 0.8})
        );
        assert!(v["id"].as_str().unwrap().starts_with("dec-"));
        assert_eq!(
            v["usage"],
            serde_json::json!({"input_tokens": 7, "output_tokens": 0, "total_tokens": 7})
        );
        assert_eq!(
            v.as_object().unwrap().len(),
            3,
            "only `id`, `answers` and `usage`"
        );
    }

    #[test]
    fn a_question_the_core_left_unanswered_is_a_server_error_not_a_panic() {
        let (_, meta) = to_internal(&oa(
            r#"{"input": "x", "questions": [{"type": "predicate", "name": "urgent", "instructions": "q"}]}"#,
        ))
        .unwrap();
        let e = to_openai(&indexmap::IndexMap::new(), &meta, 1).unwrap_err();
        assert!(matches!(e, DecisionError::Engine(_)), "{e:?}");
        assert_eq!(
            e.to_string(),
            r#"internal error: no answer for question "urgent""#
        );
    }
}
