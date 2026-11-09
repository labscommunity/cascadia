//! The public mapping driven as a gateway would, through `responses_api`
//! items only, every event and response checked against the pinned spec.

use super::*;
use crate::spec_schema::{assert_event, assert_spec};
use serde_json::{json, Value};
use std::sync::Arc;

const TURN2: &str = include_str!("../../tests/fixtures/codex-0.166.0-turn2.json");
const MAX_EXPAND: usize = 1 << 20;

/// `(event, data)` pairs of an SSE body, each frame `event:` then `data:`.
fn frames(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter(|f| !f.trim().is_empty())
        .map(|f| {
            let mut lines = f.lines();
            let event = lines.next().unwrap().strip_prefix("event: ").unwrap();
            let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
            assert!(lines.next().is_none(), "{f}");
            (event.to_owned(), serde_json::from_str(data).unwrap())
        })
        .collect()
}

/// Codex's turn-2 request with its `function_call` sent as an
/// `item_reference` to a copy kept in `cache`.
fn turn2_by_reference(cache: &ItemCache) -> ResponsesRequest {
    let mut body: Value = serde_json::from_str(TURN2).unwrap();
    let input = body["input"].as_array_mut().unwrap();
    let at = input
        .iter()
        .position(|i| i["type"] == "function_call")
        .unwrap();
    cache.insert_all(std::slice::from_ref(&input[at]));
    input[at] = json!({"type": "item_reference", "id": input[at]["id"]});
    serde_json::from_value(body).unwrap()
}

/// `include` asks for encrypted reasoning, the tools, the checked
/// `tool_choice`, and the chat messages.
type Mapped = (bool, ToolMap, Option<&'static str>, Vec<ChatMessage>);

/// Validate and map `req` as `create_response` does.
fn map(req: &ResponsesRequest, cache: &ItemCache) -> Result<Mapped, Refusal> {
    let encrypted = refuse_unsupported(req)?;
    let tools = ToolMap::from_request(req.tools.as_ref())?;
    let choice = check_tool_choice(req.tool_choice.as_ref())?;
    let resolve = |id: &str, budget: usize| {
        cache
            .get_within(id, budget)
            .map_err(|_| Refusal::too_large("input", MAX_EXPAND))
    };
    let messages = input_messages(req, &resolve, MAX_EXPAND)?;
    assert!(!is_empty_conversation(&messages));
    Ok((encrypted, tools, choice, messages))
}

#[test]
fn codex_turn2_maps_onto_chat() {
    let cache = ItemCache::default();
    let req = turn2_by_reference(&cache);
    let (encrypted, tools, choice, messages) = map(&req, &cache).unwrap();
    assert!(encrypted, "Codex includes reasoning.encrypted_content");
    assert_eq!(choice, Some("auto"));
    let roles: Vec<_> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(
        roles,
        ["system", "user", "assistant", "tool"],
        "{messages:?}"
    );
    assert!(messages[0].content.contains("coding agent"));
    assert!(messages[1].content.contains("list the files here"));
    let assistant = &messages[2];
    assert!(assistant.content.contains("<think>I should list.</think>"));
    let calls = assistant.tool_calls.as_ref().unwrap();
    assert_eq!(
        (calls[0].id.as_str(), calls[0].function.name.as_str()),
        ("call_1", "exec_command")
    );
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_1"));
    assert!(messages[3].content.contains("a.txt"));

    let names: Vec<_> = tools
        .chat_tools
        .iter()
        .map(|t: &Tool| t.function["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "exec_command",
            "view_image",
            "multi_agent_v1__close_agent",
            "multi_agent_v1__resume_agent"
        ]
    );
    assert_eq!(tools.dropped, ["web_search"]);
    assert!(tools.tools_active(choice));
    let served = tools.get(&ToolMap::flat(Some("multi_agent_v1"), "close_agent"));
    assert_eq!(
        served.map(|t| (t.kind, t.namespace.as_deref())),
        Some((Served::Function, Some("multi_agent_v1")))
    );
}

/// A reference past the expansion budget is the server's 413 body.
#[test]
fn an_oversized_reference_is_a_413_refusal() {
    let cache = ItemCache::default();
    let req = turn2_by_reference(&cache);
    let resolve = |id: &str, budget: usize| {
        cache
            .get_within(id, budget)
            .map_err(|_| Refusal::too_large("input", 8))
    };
    let refusal = input_messages(&req, &resolve, 8).unwrap_err();
    assert_eq!(refusal.status().as_u16(), 413);
    assert_eq!(refusal.code(), Some("prompt_too_large"));
    let body = refusal.to_error_json();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], refusal.param());
    assert_eq!(body["error"]["message"], refusal.message());
}

/// A gateway's turn: the request mapped, the backend's streamed text
/// (reasoning folded in as `<think>`) and one structured call rendered as
/// `/v1/responses` streams them.
#[test]
fn a_backend_turn_renders_as_the_spec_stream() {
    let cache = ItemCache::default();
    let req = turn2_by_reference(&cache);
    let (encrypted, tools, choice, _) = map(&req, &cache).unwrap();
    let shell = ResponseShell::new(&req, new_response_id(), &tools, choice);
    let id = shell.id().to_owned();
    assert!(id.starts_with("resp_"));
    let mut builder = OutputBuilder::new(
        true,
        false,
        false,
        Arc::new(tools),
        req.parallel_tool_calls.unwrap_or(true),
        encrypted,
    );
    let mut events = ResponseEvents::new(shell);
    let mut sse = events.start();
    for delta in ["<think>", "r</think>", "answer"] {
        sse.push_str(&events.push(builder.feed(delta, 1)));
    }
    let call = ToolCall {
        id: "call_9".into(),
        r#type: "function".into(),
        function: FunctionCall {
            name: "multi_agent_v1__close_agent".into(),
            arguments: r#"{"id":"a1"}"#.into(),
        },
    };
    sse.push_str(&events.push(builder.push_calls(vec![call])));
    let end = End::from_finish_reason("tool_calls");
    let (closing, items, response) = events.finish(&mut builder, end, 120, 3, None);
    sse.push_str(&closing);
    cache.insert_all(&items);

    let ev = frames(&sse);
    for (i, (kind, data)) in ev.iter().enumerate() {
        assert_eq!(data["type"], kind.as_str());
        assert_eq!(data["sequence_number"], i, "{sse}");
        assert_event(data);
    }
    let kinds: Vec<_> = ev.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_spec("Response", &response);
    assert_eq!(ev.last().unwrap().1["response"], response);
    assert_eq!(response["id"], id.as_str());
    assert_eq!(response["status"], "completed");
    assert_eq!(response["output"], json!(items));
    assert_eq!(response["usage"]["input_tokens"], 120);
    assert_eq!(response["usage"]["output_tokens"], 3);
    let shape: Vec<_> = items.iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(shape, ["reasoning", "message", "function_call"]);
    assert_eq!(items[0]["content"][0]["text"], "r");
    assert!(items[0]["encrypted_content"].is_string());
    assert_eq!(items[1]["content"][0]["text"], "answer");
    assert_eq!(items[2]["call_id"], "call_9");
    assert_eq!(items[2]["name"], "close_agent");
    assert_eq!(items[2]["namespace"], "multi_agent_v1");
    let call_id = items[2]["id"].as_str().unwrap();
    assert!(cache.get_within(call_id, MAX_EXPAND).unwrap().is_some());
}

/// The other ends: `length` and `content_filter` are `incomplete` naming
/// why, an unknown reason is a cancelled turn, and a backend failure is
/// `response.failed`, each in the spec's shape.
#[test]
fn every_end_renders_in_the_spec_shape() {
    let req: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi"})).unwrap();
    let render = |end: End, error: Option<String>| {
        let tools = ToolMap::from_request(None).unwrap();
        let shell = ResponseShell::new(&req, new_response_id(), &tools, None);
        let mut builder = OutputBuilder::new(false, false, false, Arc::new(tools), true, false);
        let mut events = ResponseEvents::new(shell);
        let mut sse = events.start();
        sse.push_str(&events.push(builder.feed("partial", 1)));
        let (closing, _, response) = events.finish(&mut builder, end, 5, 1, error);
        sse.push_str(&closing);
        let ev = frames(&sse);
        ev.iter().for_each(|(_, data)| assert_event(data));
        assert_spec("Response", &response);
        (ev.last().unwrap().0.clone(), response)
    };
    let (kind, r) = render(End::from_finish_reason("length"), None);
    assert_eq!(
        (kind.as_str(), &r["status"]),
        ("response.incomplete", &json!("incomplete"))
    );
    assert_eq!(r["incomplete_details"]["reason"], "max_output_tokens");
    let (kind, r) = render(End::from_finish_reason("content_filter"), None);
    assert_eq!(kind, "response.incomplete");
    assert_eq!(r["incomplete_details"]["reason"], "content_filter");
    assert_eq!(r["output"][0]["status"], "incomplete");
    let (kind, r) = render(End::from_finish_reason("abort"), None);
    assert_eq!(
        (kind.as_str(), &r["status"]),
        ("response.incomplete", &json!("cancelled"))
    );
    let (kind, r) = render(End::Completed, Some("backend down".into()));
    assert_eq!(
        (kind.as_str(), &r["status"]),
        ("response.failed", &json!("failed"))
    );
    assert_eq!(r["error"]["message"], "backend down");
    assert_eq!(r["output"][0]["status"], "incomplete");
}

/// `End::from_finish_reason` over OpenAI's finish reasons.
#[test]
fn finish_reasons_map_to_ends() {
    for (reason, end) in [
        ("stop", End::Completed),
        ("tool_calls", End::Completed),
        ("function_call", End::Completed),
        ("length", End::TokenCap),
        ("content_filter", End::ContentFilter),
        ("", End::Cancelled),
    ] {
        assert_eq!(End::from_finish_reason(reason), end, "{reason}");
    }
}

/// The system-turn fold retry a gateway runs when its backend's template
/// refuses a later system turn.
#[test]
fn folds_are_reachable() {
    let msgs: Vec<ChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "system", "content": "c"},
    ]))
    .unwrap();
    assert_eq!(folds_to_try(&msgs), [Fold::Later, Fold::All]);
    let folded = fold_system_turns(msgs, Fold::Later);
    let roles: Vec<_> = folded.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user"]);
}
