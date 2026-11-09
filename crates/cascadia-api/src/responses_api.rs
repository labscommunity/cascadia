//! The `/v1/responses` mapping, without an engine, for a gateway that serves
//! OpenAI's Responses API over its own chat-completion backends. It parses
//! and validates a request as this server does ([`ResponsesRequest`],
//! [`refuse_unsupported`], [`ToolMap::from_request`], [`check_tool_choice`],
//! [`is_empty_conversation`]), maps its `input` to chat messages
//! ([`input_messages`], with the caller's own lookup for `item_reference`,
//! e.g. over an [`ItemCache`]), and turns a chat completion back into
//! Responses output: [`OutputBuilder`] splits the text (reasoning folded in
//! as `<think>…</think>`) into items and takes structured tool calls through
//! [`OutputBuilder::push_calls`]; [`ResponseEvents`] renders the SSE events
//! `/v1/responses` streams, in the same order and framing, and the final
//! response object. A [`Refusal`] renders as the OpenAI error body the
//! server sends ([`Refusal::to_error_json`]). See `docs/RESPONSES.md`,
//! "Using the mapping from another crate".

pub use crate::responses::{
    check_tool_choice, fold_system_turns, folds_to_try, input_messages, is_empty_conversation,
    new_response_id, refuse_unsupported, End, Fold, ItemCache, Outcome, OutputBuilder, OutputEvent,
    ReasoningParam, Refusal, Resolve, ResponseEvents, ResponseShell, ResponsesRequest, Served,
    ServedTool, TooLarge, ToolMap,
};
pub use crate::{ChatMessage, ContentPart, FunctionCall, Tool, ToolCall};

#[cfg(test)]
mod tests;
