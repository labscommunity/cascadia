//! Test helper: validate a response body against the pinned OpenAI spec.
//!
//! The fixture is the `$ref` closure of the OpenAI response schemas, cut from
//! upstream `openai/openai-openapi` by `scripts/openai_spec_subset.py` (its
//! `x-source` field records the commit and the command). Regenerate it with
//! that script rather than editing it by hand.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

const SUBSET: &str = include_str!("../tests/fixtures/openai-spec-subset.json");

fn validator(schema_name: &str) -> Arc<jsonschema::Validator> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<jsonschema::Validator>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache
        .entry(schema_name.to_owned())
        .or_insert_with(|| {
            let subset: Value = serde_json::from_str(SUBSET).expect("spec subset is valid JSON");
            assert!(
                subset["components"]["schemas"].get(schema_name).is_some(),
                "{schema_name} is not in the spec subset; add it to ROOTS in \
                 scripts/openai_spec_subset.py and regenerate the fixture"
            );
            let root = serde_json::json!({
                "$ref": format!("#/components/schemas/{schema_name}"),
                "components": subset["components"],
            });
            Arc::new(
                jsonschema::validator_for(&root)
                    .unwrap_or_else(|e| panic!("{schema_name} does not compile: {e}")),
            )
        })
        .clone()
}

/// Panics, listing every violation with its instance path, unless `value`
/// conforms to the OpenAI schema `components.schemas.<schema_name>`.
pub(crate) fn assert_spec(schema_name: &str, value: &Value) {
    let errors: Vec<String> = validator(schema_name)
        .iter_errors(value)
        .map(|e| format!("  at {:?}: {e}", e.instance_path().to_string()))
        .collect();
    assert!(
        errors.is_empty(),
        "response does not match OpenAI {schema_name}:\n{}\nbody: {value}",
        errors.join("\n")
    );
}

/// Checks one streamed chunk: the pinned schema for its `object`, plus what
/// the schema cannot say (no extension counters, a fingerprint, `logprobs`
/// on every choice). An error event is `id`, `object: "error"` and the
/// OpenAI error object (`message`, `type`, `param`, `code`), nothing else.
pub(crate) fn assert_stream_chunk_spec(event: &Value) {
    match event["object"].as_str() {
        Some("error") => {
            let keys = |v: &Value| {
                let mut k: Vec<String> = v
                    .as_object()
                    .map(|o| o.keys().cloned().collect())
                    .unwrap_or_default();
                k.sort();
                k
            };
            assert_eq!(keys(event), ["error", "id", "object"], "{event}");
            assert!(event["id"].is_string(), "error event without id: {event}");
            let error = &event["error"];
            assert_eq!(
                keys(error),
                ["code", "message", "param", "type"],
                "error event's error is not the OpenAI error object: {event}"
            );
            assert!(error["message"].is_string(), "{event}");
            assert!(error["type"].is_string(), "{event}");
            assert!(
                error["code"].is_string() || error["code"].is_null(),
                "{event}"
            );
            assert!(
                error["param"].is_string() || error["param"].is_null(),
                "{event}"
            );
            return;
        }
        Some("chat.completion.chunk") => {
            assert_spec("CreateChatCompletionStreamResponse", event);
        }
        Some("text_completion") => {
            // The pinned completions schema is the non-streamed object, whose
            // `usage` cannot be null; a streamed chunk carries `usage: null`
            // before the usage chunk, as chat chunks do.
            let mut event = event.clone();
            if event.get("usage").is_some_and(Value::is_null) {
                event.as_object_mut().map(|o| o.remove("usage"));
            }
            assert_spec("CreateCompletionResponse", &event);
        }
        other => panic!("unexpected stream object {other:?}: {event}"),
    }
    for key in ["n_tokens", "completion_tokens"] {
        assert!(event.get(key).is_none(), "chunk carries `{key}`: {event}");
    }
    assert!(
        event["system_fingerprint"].is_string(),
        "chunk without system_fingerprint: {event}"
    );
    for choice in event["choices"].as_array().into_iter().flatten() {
        assert!(
            choice.get("logprobs").is_some(),
            "choice without logprobs: {event}"
        );
    }
}

/// Checks a whole SSE body of a chat or completions stream: only `data:`
/// lines and `:` comments, every chunk spec-shaped, one `id` and `created`
/// throughout, `role` on the first chat delta only, and either exactly one
/// spec `finish_reason` or an error event as the last event (with no
/// `finish_reason` and no usage chunk before it); then `[DONE]`.
pub(crate) fn assert_stream_spec(body: &str) {
    let mut events = Vec::new();
    let mut done = false;
    // SSE joins consecutive `data:` lines of one event with `\n`, so two
    // chunks written into one event reach clients as one invalid payload:
    // every event (lines up to a blank line) holds exactly one `data:` line,
    // or only comments.
    let normalized = body.replace("\r\n", "\n");
    for event in normalized.split("\n\n").filter(|e| !e.trim().is_empty()) {
        assert!(!done, "event after data: [DONE]: {body}");
        let mut data_lines = Vec::new();
        for line in event.lines().filter(|l| !l.is_empty()) {
            if line.starts_with(':') {
                continue;
            }
            let data = line.strip_prefix("data: ").unwrap_or_else(|| {
                panic!("SSE line is neither data nor a comment: {line:?}\nbody: {body}")
            });
            data_lines.push(data);
        }
        match data_lines[..] {
            [] => continue,
            [data] => {
                if data == "[DONE]" {
                    done = true;
                    continue;
                }
                let event: Value = serde_json::from_str(data)
                    .unwrap_or_else(|e| panic!("data line is not JSON ({e}): {data}"));
                assert_stream_chunk_spec(&event);
                events.push(event);
            }
            _ => panic!("SSE event has more than one data line: {event:?}\nbody: {body}"),
        }
    }
    assert!(done, "stream does not end with data: [DONE]: {body}");
    let chunks: Vec<&Value> = events.iter().filter(|e| e["object"] != "error").collect();
    if let Some(first) = chunks.first() {
        for c in &chunks {
            assert_eq!(c["id"], first["id"], "id changes within a stream: {body}");
            assert_eq!(
                c["created"], first["created"],
                "created changes within a stream: {body}"
            );
            assert_eq!(
                c["object"], first["object"],
                "object changes within a stream: {body}"
            );
        }
    }
    let deltas: Vec<&Value> = chunks
        .iter()
        .filter_map(|c| c["choices"].as_array())
        .flatten()
        .filter_map(|choice| choice.get("delta"))
        .collect();
    for (i, delta) in deltas.iter().enumerate() {
        if i == 0 {
            assert_eq!(
                delta["role"], "assistant",
                "first delta without role: {body}"
            );
        } else {
            assert!(delta.get("role").is_none(), "role on a later delta: {body}");
        }
    }
    check_usage_placement(&events, body);
    let ends_in_error = events.last().is_some_and(|e| e["object"] == "error");
    assert!(
        events.len() == chunks.len() + usize::from(ends_in_error),
        "an error event that is not the last event: {body}"
    );
    let finishes: Vec<&Value> = chunks
        .iter()
        .filter_map(|c| c["choices"].as_array())
        .flatten()
        .map(|choice| &choice["finish_reason"])
        .filter(|f| !f.is_null())
        .collect();
    if ends_in_error {
        // Clients match the error to the stream it ends by `id`.
        if let (Some(first), Some(error)) = (chunks.first(), events.last()) {
            assert_eq!(
                error["id"], first["id"],
                "error event's id differs from the chunks' id: {body}"
            );
        }
        assert!(
            finishes.is_empty(),
            "a stream that ends in an error closed its choice: {body}"
        );
        assert!(
            !chunks
                .iter()
                .any(|c| c["choices"].as_array().is_some_and(Vec::is_empty)),
            "a usage chunk in a stream that ends in an error: {body}"
        );
        return;
    }
    assert_eq!(finishes.len(), 1, "want exactly one finish_reason: {body}");
    for f in finishes {
        assert!(
            [
                "stop",
                "length",
                "tool_calls",
                "content_filter",
                "function_call"
            ]
            .contains(&f.as_str().unwrap_or_default()),
            "finish_reason {f} is not a spec value: {body}"
        );
    }
}

/// `include_usage` placement: the usage chunk (`choices: []`) comes once,
/// last, with an object; every chunk with choices then has a `usage` key
/// (null, or the usage so far in continuous mode). Without it no chunk has
/// the key.
fn check_usage_placement(events: &[Value], body: &str) {
    let with_choices: Vec<&Value> = events
        .iter()
        .filter(|e| e["object"] != "error")
        .filter(|e| e["choices"].as_array().is_some_and(|c| !c.is_empty()))
        .collect();
    let keyed = with_choices
        .iter()
        .filter(|e| e.get("usage").is_some())
        .count();
    assert!(
        keyed == 0 || keyed == with_choices.len(),
        "usage key on some chunks but not others: {body}"
    );
    for e in &with_choices {
        let u = &e["usage"];
        assert!(
            u.is_null() || u.is_object(),
            "a chunk's usage is neither null nor an object: {body}"
        );
    }
    let usage_chunks: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| e["choices"].as_array().is_some_and(Vec::is_empty))
        .map(|(i, _)| i)
        .collect();
    assert!(
        usage_chunks.len() <= 1,
        "more than one choices-less usage chunk: {body}"
    );
    if let Some(&i) = usage_chunks.first() {
        assert_eq!(
            i,
            events.len() - 1,
            "the usage chunk is not the last chunk before [DONE]: {body}"
        );
        assert!(
            events[i]["usage"].is_object(),
            "the usage chunk has no usage object: {body}"
        );
        assert_eq!(
            keyed,
            with_choices.len(),
            "a chunk before the usage chunk has no usage key: {body}"
        );
    }
}

#[test]
fn assert_spec_rejects_a_response_without_choices() {
    // The helper must not be a no-op: a chat response missing the required
    // `choices` array has to fail.
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
    });
    let panic = std::panic::catch_unwind(|| assert_spec("CreateChatCompletionResponse", &body))
        .expect_err("a response without `choices` must fail the schema");
    let msg = panic.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("\"choices\" is a required property"), "{msg}");
}

#[test]
fn assert_stream_spec_rejects_extension_fields_and_repeated_roles() {
    let chunk = |extra: &str, delta: &str, finish: &str| {
        format!(
            "data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\
             \"model\":\"m\",\"system_fingerprint\":\"fp_0123456789\"{extra},\
             \"choices\":[{{\"index\":0,\"delta\":{delta},\"logprobs\":null,\
             \"finish_reason\":{finish}}}]}}\n\n"
        )
    };
    let first = chunk("", r#"{"role":"assistant","content":"hi"}"#, "null");
    let last = chunk("", "{}", "\"stop\"");
    let good = format!("{first}: keep-alive\n\n{last}data: [DONE]\n\n");
    assert_stream_spec(&good);
    let err = "data: {\"id\":\"c\",\"object\":\"error\",\"error\":{\"message\":\"x\",\
               \"type\":\"server_error\",\"param\":null,\"code\":\"engine_error\"}}\n\n";
    assert_stream_spec(&format!("{first}{err}data: [DONE]\n\n"));
    // Each bad body and the rule's own message it must fail with.
    let bad = [
        (
            format!(
                "{}{last}data: [DONE]\n\n",
                chunk(
                    ",\"n_tokens\":1",
                    r#"{"role":"assistant","content":"hi"}"#,
                    "null"
                )
            ),
            "chunk carries `n_tokens`",
        ),
        (
            format!(
                "{first}{}data: [DONE]\n\n",
                chunk("", r#"{"role":"assistant"}"#, "\"stop\"")
            ),
            "role on a later delta",
        ),
        (
            format!("{first}data: [DONE]\n\n"),
            "want exactly one finish_reason",
        ),
        (
            chunk("", r#"{"role":"assistant","content":"hi"}"#, "\"stop\""),
            "stream does not end with data: [DONE]",
        ),
        (
            format!("event: x\n{last}data: [DONE]\n\n"),
            "SSE line is neither data nor a comment",
        ),
        (
            format!("{}\n{last}data: [DONE]\n\n", first.trim_end_matches('\n')),
            "SSE event has more than one data line",
        ),
        (
            format!("{first}{last}data: [DONE]\n\n{last}"),
            "event after data: [DONE]",
        ),
        (
            format!(
                "{first}data: {{\"id\":\"c\",\"object\":\"error\",\
                 \"error\":{{\"message\":\"x\",\"type\":\"engine_error\"}}}}\n\ndata: [DONE]\n\n"
            ),
            "not the OpenAI error object",
        ),
        (
            format!("{first}{err}{last}data: [DONE]\n\n"),
            "an error event that is not the last event",
        ),
        (
            format!("{first}{last}{err}data: [DONE]\n\n"),
            "a stream that ends in an error closed its choice",
        ),
        (
            format!(
                "{first}{}data: [DONE]\n\n",
                err.replace("\"c\"", "\"other\"")
            ),
            "error event's id differs from the chunks' id",
        ),
    ];
    for (body, rule) in bad {
        let panic = std::panic::catch_unwind(|| assert_stream_spec(&body))
            .expect_err("a bad stream must fail");
        let msg = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default();
        assert!(msg.contains(rule), "want {rule:?}, got: {msg}");
    }
}

#[test]
fn assert_stream_spec_checks_usage_placement() {
    let chunk = |delta: &str, finish: &str, usage: &str| {
        format!(
            "data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\
             \"model\":\"m\",\"system_fingerprint\":\"fp_0123456789\"{usage},\
             \"choices\":[{{\"index\":0,\"delta\":{delta},\"logprobs\":null,\
             \"finish_reason\":{finish}}}]}}\n\n"
        )
    };
    let tail = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\
                \"model\":\"m\",\"system_fingerprint\":\"fp_0123456789\",\"choices\":[],\
                \"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n";
    let first = |usage: &str| chunk(r#"{"role":"assistant","content":"hi"}"#, "null", usage);
    let last = |usage: &str| chunk("{}", "\"stop\"", usage);
    let null = ",\"usage\":null";
    assert_stream_spec(&format!(
        "{}{}{tail}data: [DONE]\n\n",
        first(null),
        last(null)
    ));
    assert_stream_spec(&format!("{}{}data: [DONE]\n\n", first(""), last("")));
    let bad = [
        (
            format!("{}{}{tail}data: [DONE]\n\n", first(""), last(null)),
            "usage key on some chunks but not others",
        ),
        (
            format!("{}{}{tail}{tail}data: [DONE]\n\n", first(null), last(null)),
            "more than one choices-less usage chunk",
        ),
        (
            format!("{}{tail}{}data: [DONE]\n\n", first(null), last(null)),
            "the usage chunk is not the last chunk before [DONE]",
        ),
        (
            format!("{}{}{tail}data: [DONE]\n\n", first(""), last("")),
            "a chunk before the usage chunk has no usage key",
        ),
    ];
    for (body, rule) in bad {
        let panic = std::panic::catch_unwind(|| assert_stream_spec(&body))
            .expect_err("a bad stream must fail");
        let msg = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default();
        assert!(msg.contains(rule), "want {rule:?}, got: {msg}");
    }
}
