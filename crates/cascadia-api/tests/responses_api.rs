//! `cascadia_api::responses_api` from outside the crate: a gateway maps a
//! Codex turn onto chat, then renders a backend's chat completion (text with
//! reasoning folded in as `<think>`, and one structured tool call) as the
//! `/v1/responses` stream. The events are checked against the pinned spec
//! inside the crate (`responses_api::tests`); here, that the public items
//! are enough to do it.

use cascadia_api::responses_api::{
    check_tool_choice, input_messages, is_empty_conversation, new_response_id, refuse_unsupported,
    End, FunctionCall, ItemCache, OutputBuilder, Refusal, ResponseEvents, ResponseShell,
    ResponsesRequest, ToolCall, ToolMap,
};
use serde_json::{json, Value};
use std::sync::Arc;

const TURN2: &str = include_str!("fixtures/codex-0.166.0-turn2.json");
const MAX_EXPAND: usize = 1 << 20;

#[test]
fn a_gateway_maps_and_renders_a_codex_turn() {
    // The previous turn's call is in the gateway's own cache; this turn
    // sends it back by reference.
    let cache = ItemCache::default();
    let mut body: Value = serde_json::from_str(TURN2).unwrap();
    let input = body["input"].as_array_mut().unwrap();
    let at = input
        .iter()
        .position(|i| i["type"] == "function_call")
        .unwrap();
    cache.insert_all(std::slice::from_ref(&input[at]));
    input[at] = json!({"type": "item_reference", "id": input[at]["id"]});
    let req: ResponsesRequest = serde_json::from_value(body).unwrap();

    let encrypted = refuse_unsupported(&req).unwrap();
    let tools = ToolMap::from_request(req.tools.as_ref()).unwrap();
    let choice = check_tool_choice(req.tool_choice.as_ref()).unwrap();
    let resolve = |id: &str, budget: usize| {
        cache
            .get_within(id, budget)
            .map_err(|_| Refusal::too_large("input", MAX_EXPAND))
    };
    let messages = input_messages(&req, &resolve, MAX_EXPAND).unwrap();
    assert!(!is_empty_conversation(&messages));
    let roles: Vec<_> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool"]);
    let replayed = messages[2].tool_calls.as_ref().unwrap();
    assert_eq!(replayed[0].function.name, "exec_command");
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_1"));
    let names: Vec<_> = tools
        .chat_tools
        .iter()
        .map(|t| t.function["name"].as_str().unwrap())
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
    assert_eq!(choice, Some("auto"));

    // The backend's answer: streamed text, then a parsed call.
    let shell = ResponseShell::new(&req, new_response_id(), &tools, choice);
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

    let frames: Vec<(String, Value)> = sse
        .split("\n\n")
        .filter(|f| !f.is_empty())
        .map(|f| {
            let (event, data) = f.split_once('\n').unwrap();
            let event = event.strip_prefix("event: ").unwrap();
            let data = data.strip_prefix("data: ").unwrap();
            (event.to_owned(), serde_json::from_str(data).unwrap())
        })
        .collect();
    assert_eq!(frames.first().unwrap().0, "response.created");
    assert_eq!(frames[1].0, "response.in_progress");
    for (i, (kind, data)) in frames.iter().enumerate() {
        assert_eq!(data["type"], kind.as_str());
        assert_eq!(data["sequence_number"], i);
    }
    let (last, data) = frames.last().unwrap();
    assert_eq!(last, "response.completed");
    assert_eq!(data["response"], response);
    assert_eq!(response["status"], "completed");
    assert_eq!(response["output"], json!(items));
    let kinds: Vec<_> = items.iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["reasoning", "message", "function_call"]);
    assert_eq!(items[1]["content"][0]["text"], "answer");
    assert_eq!(
        (&items[2]["name"], &items[2]["namespace"]),
        (&json!("close_agent"), &json!("multi_agent_v1"))
    );
    let call_id = items[2]["id"].as_str().unwrap();
    assert!(cache.get_within(call_id, MAX_EXPAND).unwrap().is_some());
}

/// A refusal is the OpenAI error body the server sends.
#[test]
fn a_refusal_renders_as_the_openai_error_body() {
    let req: ResponsesRequest = serde_json::from_value(
        json!({"model": "m", "input": "x", "previous_response_id": "resp_1"}),
    )
    .unwrap();
    let refusal = refuse_unsupported(&req).unwrap_err();
    assert_eq!(refusal.status().as_u16(), 400);
    assert_eq!(refusal.param(), "previous_response_id");
    assert_eq!(
        refusal.to_error_json(),
        json!({"error": {
            "message": refusal.message(),
            "type": "invalid_request_error",
            "param": "previous_response_id",
            "code": null,
        }})
    );
}
