//! OpenAI's ranges and enums for the sampling fields, and the fields OpenAI
//! defines that this server cannot honour.
//!
//! Each check returns a [`FieldError`] naming the request field at fault, so
//! the route answers a 400 whose `param` is that field. Another server that
//! embeds these routes can call the same checks on its own requests.

use serde_json::Value;

use crate::{ChatCompletionRequest, CompletionRequest, StopSpec};

/// A request field whose value OpenAI's API would refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldError {
    /// The field, as OpenAI names it in the error's `param`.
    pub param: String,
    /// Names the field, the value sent and what is allowed.
    pub message: String,
}

impl FieldError {
    fn new(param: &str, message: String) -> Self {
        Self {
            param: param.to_owned(),
            message,
        }
    }
}

/// Every `reasoning_effort` word OpenAI defines.
pub const REASONING_EFFORTS: [&str; 7] =
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// The most `stop` sequences OpenAI takes.
const MAX_STOP: usize = 4;
/// The most `top_logprobs` OpenAI returns per position.
const MAX_TOP_LOGPROBS: u32 = 20;

fn check_range(v: f32, lo: f32, hi: f32, param: &str) -> Result<(), FieldError> {
    if v.is_finite() && (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(FieldError::new(
            param,
            format!("{param} {v} is out of range; it must be between {lo} and {hi}"),
        ))
    }
}

/// `temperature`: 0 to 2.
pub fn check_temperature(v: f32, param: &str) -> Result<(), FieldError> {
    check_range(v, 0.0, 2.0, param)
}

/// `top_p`: 0 to 1.
pub fn check_top_p(v: f32, param: &str) -> Result<(), FieldError> {
    check_range(v, 0.0, 1.0, param)
}

/// `frequency_penalty` and `presence_penalty`: -2 to 2.
pub fn check_penalty(v: f32, param: &str) -> Result<(), FieldError> {
    check_range(v, -2.0, 2.0, param)
}

/// `reasoning_effort`: one of [`REASONING_EFFORTS`].
pub fn check_reasoning_effort(v: &str, param: &str) -> Result<(), FieldError> {
    if REASONING_EFFORTS.contains(&v) {
        Ok(())
    } else {
        Err(FieldError::new(
            param,
            format!(
                "{param} `{v}` is not one of {}",
                REASONING_EFFORTS.join(", ")
            ),
        ))
    }
}

/// `stop`: at most four sequences.
pub fn check_stop_count(n: usize) -> Result<(), FieldError> {
    if n <= MAX_STOP {
        Ok(())
    } else {
        Err(FieldError::new(
            "stop",
            format!("stop has {n} sequences; at most {MAX_STOP} are allowed"),
        ))
    }
}

/// `top_logprobs`: 0 to 20.
pub fn check_top_logprobs(v: u32) -> Result<(), FieldError> {
    if v <= MAX_TOP_LOGPROBS {
        Ok(())
    } else {
        Err(FieldError::new(
            "top_logprobs",
            format!(
                "top_logprobs {v} is out of range; it must be between 0 and {MAX_TOP_LOGPROBS}"
            ),
        ))
    }
}

/// `n`: absent or 1. Zero is outside OpenAI's range; more than one is a
/// valid request this server cannot serve, since it returns one choice.
pub fn check_n(n: Option<u32>) -> Result<(), FieldError> {
    match n {
        None | Some(1) => Ok(()),
        Some(0) => Err(FieldError::new(
            "n",
            "n 0 is out of range; it must be at least 1".to_owned(),
        )),
        Some(n) => Err(FieldError::new(
            "n",
            format!("n {n} is not supported; this server returns one choice (n must be 1)"),
        )),
    }
}

/// Fields OpenAI defines that this server can't honour: non-empty logit_bias object, modalities containing "audio",
/// non-null audio, non-null web_search_options. `null`, `{}`, `[]`, `["text"]` pass.
pub fn check_unsupported(
    logit_bias: Option<&Value>,
    modalities: Option<&Value>,
    audio: Option<&Value>,
    web_search_options: Option<&Value>,
) -> Result<(), FieldError> {
    check_logit_bias(logit_bias)?;
    let wants_audio = modalities
        .and_then(Value::as_array)
        .is_some_and(|m| m.iter().any(|x| x == "audio"));
    if wants_audio {
        return Err(FieldError::new(
            "modalities",
            "modalities `audio` is not supported; this server returns text only (modalities \
             must be [\"text\"])"
                .to_owned(),
        ));
    }
    if audio.is_some_and(|a| !a.is_null()) {
        return Err(FieldError::new(
            "audio",
            "audio output is not supported; this server returns text only (audio must be null)"
                .to_owned(),
        ));
    }
    if web_search_options.is_some_and(|w| !w.is_null()) {
        return Err(FieldError::new(
            "web_search_options",
            "web_search_options is not supported; this server has no web search \
             (web_search_options must be null)"
                .to_owned(),
        ));
    }
    Ok(())
}

/// `logit_bias`: only `null` or `{}`; the engines take no per-token bias.
fn check_logit_bias(logit_bias: Option<&Value>) -> Result<(), FieldError> {
    match logit_bias {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(m)) if m.is_empty() => Ok(()),
        Some(_) => Err(FieldError::new(
            "logit_bias",
            "logit_bias is not supported; this server cannot bias token logits (logit_bias \
             must be null or {})"
                .to_owned(),
        )),
    }
}

fn stop_count(stop: Option<&StopSpec>) -> usize {
    match stop {
        None => 0,
        Some(StopSpec::Single(_)) => 1,
        Some(StopSpec::Multiple(v)) => v.len(),
    }
}

/// Every check above that applies to a `/v1/chat/completions` request.
pub fn check_chat(req: &ChatCompletionRequest) -> Result<(), FieldError> {
    check_temperature(req.temperature, "temperature")?;
    if let Some(v) = req.top_p {
        check_top_p(v, "top_p")?;
    }
    if let Some(v) = req.frequency_penalty {
        check_penalty(v, "frequency_penalty")?;
    }
    if let Some(v) = req.presence_penalty {
        check_penalty(v, "presence_penalty")?;
    }
    check_n(req.n)?;
    if let Some(v) = req.top_logprobs {
        check_top_logprobs(v)?;
    }
    check_stop_count(stop_count(req.stop.as_ref()))?;
    if let Some(e) = req.reasoning_effort.as_deref() {
        check_reasoning_effort(e, "reasoning_effort")?;
    }
    check_unsupported(
        req.logit_bias.as_ref(),
        req.modalities.as_ref(),
        req.audio.as_ref(),
        req.web_search_options.as_ref(),
    )
}

/// Every check above that applies to a `/v1/completions` request.
pub fn check_completion(req: &CompletionRequest) -> Result<(), FieldError> {
    check_temperature(req.temperature, "temperature")?;
    if let Some(v) = req.top_p {
        check_top_p(v, "top_p")?;
    }
    if let Some(v) = req.frequency_penalty {
        check_penalty(v, "frequency_penalty")?;
    }
    if let Some(v) = req.presence_penalty {
        check_penalty(v, "presence_penalty")?;
    }
    check_n(req.n)?;
    check_stop_count(stop_count(req.stop.as_ref()))?;
    check_logit_bias(req.logit_bias.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ranges_are_inclusive_and_name_the_value() {
        assert!(check_temperature(0.0, "temperature").is_ok());
        assert!(check_temperature(2.0, "temperature").is_ok());
        assert!(check_temperature(f32::NAN, "temperature").is_err());
        assert_eq!(
            check_temperature(2.5, "temperature").unwrap_err().message,
            "temperature 2.5 is out of range; it must be between 0 and 2"
        );
        assert!(check_top_p(1.0, "top_p").is_ok());
        assert_eq!(check_top_p(1.5, "top_p").unwrap_err().param, "top_p");
        assert!(check_penalty(-2.0, "presence_penalty").is_ok());
        assert!(check_penalty(2.1, "presence_penalty").is_err());
        assert!(check_top_logprobs(20).is_ok());
        assert!(check_top_logprobs(21).is_err());
        assert!(check_stop_count(4).is_ok());
        assert_eq!(check_stop_count(5).unwrap_err().param, "stop");
    }

    #[test]
    fn reasoning_effort_and_n_messages() {
        for e in REASONING_EFFORTS {
            assert!(check_reasoning_effort(e, "reasoning_effort").is_ok());
        }
        assert_eq!(
            check_reasoning_effort("turbo", "reasoning_effort")
                .unwrap_err()
                .message,
            "reasoning_effort `turbo` is not one of none, minimal, low, medium, high, xhigh, max"
        );
        assert!(check_n(None).is_ok());
        assert!(check_n(Some(1)).is_ok());
        assert!(check_n(Some(0))
            .unwrap_err()
            .message
            .contains("out of range"));
        assert_eq!(
            check_n(Some(3)).unwrap_err().message,
            "n 3 is not supported; this server returns one choice (n must be 1)"
        );
    }

    #[test]
    fn unsupported_fields_pass_only_their_no_op_forms() {
        let ok = |lb: Value, m: Value, a: Value, w: Value| {
            check_unsupported(Some(&lb), Some(&m), Some(&a), Some(&w))
        };
        assert!(check_unsupported(None, None, None, None).is_ok());
        assert!(ok(json!(null), json!(null), json!(null), json!(null)).is_ok());
        assert!(ok(json!({}), json!([]), json!(null), json!(null)).is_ok());
        assert!(ok(json!({}), json!(["text"]), json!(null), json!(null)).is_ok());
        let param = |r: Result<(), FieldError>| r.unwrap_err().param;
        assert_eq!(
            param(ok(json!({"1": 5}), json!(null), json!(null), json!(null))),
            "logit_bias"
        );
        assert_eq!(
            param(ok(
                json!(null),
                json!(["text", "audio"]),
                json!(null),
                json!(null)
            )),
            "modalities"
        );
        assert_eq!(
            param(ok(
                json!(null),
                json!(null),
                json!({"voice": "alloy"}),
                json!(null)
            )),
            "audio"
        );
        assert_eq!(
            param(ok(json!(null), json!(null), json!(null), json!({}))),
            "web_search_options"
        );
    }
}
