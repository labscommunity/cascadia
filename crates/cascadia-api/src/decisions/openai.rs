//! OpenAI Decisions API wire format (developers.openai.com/api/docs/guides/decisions) at
//! POST /v1/decisions. A thin adapter over the shared core: the models see the same prompts as
//! through /v1/systemone.
use std::collections::HashSet;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::schema::{Answer, DecisionRequest, OrderedValue, Question, QuestionType};
use super::DecisionError;
use crate::AppState;
use cascadia_types::MAX_DECISION_OUTPUTS;

#[derive(Clone, Debug, Deserialize)]
pub struct OaRequest {
    #[serde(default)]
    pub model: String,
    pub input: OaInput,
    pub questions: Vec<OaQuestion>,
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

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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

#[derive(Clone, Debug, Deserialize)]
pub struct OaChoice {
    /// A string or a number. `serde_json::Value`, not `OrderedValue`: `OrderedValue` parses from
    /// raw JSON text, which serde does not keep inside the internally tagged `OaQuestion`. A
    /// scalar has no key order to lose; `to_internal` converts it (and rejects other types).
    pub value: serde_json::Value,
    #[serde(default)]
    pub description: Option<String>,
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

#[derive(Clone, Debug, Serialize)]
pub struct OaResponse {
    pub answers: Vec<OaAnswer>,
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

/// A choice value as the internal model holds it: strings and numbers only.
fn choice_value(name: &str, v: &serde_json::Value) -> Result<OrderedValue, DecisionError> {
    match v {
        serde_json::Value::String(s) => Ok(OrderedValue::String(s.clone())),
        // serde_json parses the integer literal `-0` as the float -0.0, and keeps no trace of the
        // literal; Python's `json.loads("-0")` is the int 0, keyed "0" (as `/v1/systemone` keys
        // it). So a negative zero is the integer 0, and `0` and `-0` are one value. A `-0.0`
        // literal is folded too (Python would key it "-0.0"); `0.0` stays "0.0".
        serde_json::Value::Number(n)
            if n.is_f64() && n.as_f64().is_some_and(|x| x == 0.0 && x.is_sign_negative()) =>
        {
            Ok(OrderedValue::Number(0.into()))
        }
        serde_json::Value::Number(n) => Ok(OrderedValue::Number(n.clone())),
        _ => Err(bad(format!(
            "{name}: choice values must be strings or numbers"
        ))),
    }
}

/// The internal criteria key: the string itself, or Python's `str()` of a number (`2` → `"2"`,
/// `1e5` → `"100000.0"`), as the reference implementations key a parsed number.
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
    };
    Ok((req, meta))
}

/// Internal answers (request order) → OpenAI answers.
pub fn to_openai(answers: &IndexMap<String, Answer>, meta: &[OaMeta]) -> OaResponse {
    let out = meta
        .iter()
        .map(|m| match &answers[&m.name] {
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
        })
        .collect();
    OaResponse { answers: out }
}

/// OpenAI's error envelope, which its SDKs parse into typed errors.
fn oa_err(e: DecisionError) -> Response {
    let kind = match e {
        DecisionError::BadRequest(_) | DecisionError::TooLong(_) => "invalid_request_error",
        DecisionError::NotServed | DecisionError::Engine(_) => "server_error",
    };
    (
        e.status(),
        Json(serde_json::json!({"error": {"message": e.to_string(), "type": kind, "code": null}})),
    )
        .into_response()
}

/// POST /v1/decisions: the OpenAI Decisions API wire format.
pub async fn decide(State(state): State<AppState>, Json(r): Json<OaRequest>) -> Response {
    let (req, meta) = match to_internal(&r) {
        Ok(x) => x,
        Err(e) => return oa_err(e),
    };
    match super::handler::evaluate(&state, &req).await {
        Ok((answers, _input_tokens)) => Json(to_openai(&answers, &meta)).into_response(),
        Err(e) => oa_err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oa(s: &str) -> OaRequest {
        serde_json::from_str(s).unwrap()
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
        // EDGE-4: `-0` is Python's int 0 (key "0"), not "-0.0"; `0.0` keeps its float key.
        let (r3, m3) = to_internal(&oa(r#"{"input": "x", "questions": [{"type": "choice", "name": "c", "instructions": "x", "choices": [{"value": -0}, {"value": 0.0}]}]}"#)).unwrap();
        assert_eq!(r3.questions["c"].option_keys(), ["0", "0.0"]);
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
        let v = serde_json::to_value(to_openai(&answers, &meta)).unwrap();
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
        assert_eq!(
            v.as_object().unwrap().len(),
            1,
            "only `answers`, as in OpenAI's responses"
        );
    }
}
