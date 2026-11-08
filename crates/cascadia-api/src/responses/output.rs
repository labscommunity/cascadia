//! The Responses output path: splitting a generation into reasoning and answer.
//!
//! [`ReasoningSplit`] is stage 1 of the output builder: with thinking on it
//! routes the text inside `<think>…</think>` to reasoning and the rest to the
//! answer. It starts inside reasoning when the generation prompt already
//! opened `<think>` (see [`crate::opens_think`]); otherwise it holds leading
//! whitespace and any prefix of `<think>` until the first non-whitespace text
//! decides. A reply that never opens `<think>` is released unchanged, so its
//! text stays byte-identical to a plain chat reply.
//!
//! [`OutputBuilder`] drives it and the tool-call splitter, turning a
//! generation into Responses output items and the events that announce them,
//! so the streaming and non-streaming paths return the same items.

use super::input::encode_reasoning;
use super::tools::{custom_input, Served, ToolMap};
use crate::{ToolCall, ToolStreamEnd, ToolStreamSplitter};
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// A run of text routed to one side of the split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Seg {
    Reasoning(String),
    Answer(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Thinking off: everything is answer.
    Off,
    /// Thinking on, nothing but whitespace or a prefix of `<think>` seen yet.
    Undecided,
    /// Inside `<think>`.
    Reasoning,
    /// Past `</think>`, or the reply never opened `<think>`.
    Answer,
}

/// Splits generated text into reasoning and answer (spec "Output", stage 1).
#[derive(Debug)]
pub(super) struct ReasoningSplit {
    state: State,
    /// Undecided: the whitespace and `<think>` prefix seen so far.
    /// Reasoning: a trailing prefix of `</think>` split across feeds.
    held: String,
    /// The answer's leading whitespace is still being dropped (right after
    /// `</think>`).
    trim_answer: bool,
    consumed_close: bool,
}

impl ReasoningSplit {
    /// `thinking`: the request has thinking on. `opened`: the generation
    /// prompt already opened `<think>`, so the model starts inside reasoning.
    pub fn new(thinking: bool, opened: bool) -> Self {
        let state = match (thinking, opened) {
            (false, _) => State::Off,
            (true, true) => State::Reasoning,
            (true, false) => State::Undecided,
        };
        Self {
            state,
            held: String::new(),
            trim_answer: false,
            consumed_close: false,
        }
    }

    /// Route one chunk of generated text.
    pub fn feed(&mut self, text: &str) -> Vec<Seg> {
        let mut out = Vec::new();
        let mut rest = text.to_owned();
        loop {
            match self.state {
                State::Off => {
                    push(&mut out, Seg::Answer, rest);
                    return out;
                }
                State::Undecided => {
                    let buf = std::mem::take(&mut self.held) + &rest;
                    let lead = buf.len() - buf.trim_start().len();
                    let body = &buf[lead..];
                    if let Some(after) = body.strip_prefix(OPEN) {
                        // The model opened reasoning: the tag and the
                        // whitespace before it are not text.
                        self.state = State::Reasoning;
                        rest = after.to_owned();
                    } else if OPEN.starts_with(body) {
                        // Whitespace only, or a prefix of `<think>`: wait.
                        self.held = buf;
                        return out;
                    } else {
                        // No reasoning: everything held goes out unchanged.
                        self.state = State::Answer;
                        push(&mut out, Seg::Answer, buf);
                        return out;
                    }
                }
                State::Reasoning => {
                    let buf = std::mem::take(&mut self.held) + &rest;
                    if let Some(at) = buf.find(CLOSE) {
                        push(&mut out, Seg::Reasoning, buf[..at].to_owned());
                        self.consumed_close = true;
                        self.trim_answer = true;
                        self.state = State::Answer;
                        rest = buf[at + CLOSE.len()..].to_owned();
                    } else {
                        let keep = partial_suffix(&buf, CLOSE);
                        self.held = buf[buf.len() - keep..].to_owned();
                        push(&mut out, Seg::Reasoning, buf[..buf.len() - keep].to_owned());
                        return out;
                    }
                }
                State::Answer => {
                    if self.trim_answer {
                        let trimmed = rest.trim_start();
                        if trimmed.is_empty() {
                            return out;
                        }
                        self.trim_answer = false;
                        rest = trimmed.to_owned();
                    }
                    push(&mut out, Seg::Answer, rest);
                    return out;
                }
            }
        }
    }

    /// End of the generation: release whatever is held (an undecided prefix
    /// is answer text; a partial `</think>` inside reasoning is reasoning).
    pub fn finish(&mut self) -> Vec<Seg> {
        let held = std::mem::take(&mut self.held);
        let mut out = Vec::new();
        match self.state {
            State::Undecided => push(&mut out, Seg::Answer, held),
            State::Reasoning => push(&mut out, Seg::Reasoning, held),
            State::Off | State::Answer => {}
        }
        out
    }

    /// The split is inside `<think>`.
    pub fn in_reasoning(&self) -> bool {
        self.state == State::Reasoning
    }

    /// A `</think>` was consumed (even around empty reasoning).
    pub fn consumed_close(&self) -> bool {
        self.consumed_close
    }

    /// Bytes fed but not yet released or consumed.
    fn held_len(&self) -> usize {
        self.held.len()
    }
}

/// Append a non-empty run of text as `kind`.
fn push(out: &mut Vec<Seg>, kind: fn(String) -> Seg, text: String) {
    if !text.is_empty() {
        out.push(kind(text));
    }
}

/// Length of the longest proper prefix of `tag` that `buf` ends with.
fn partial_suffix(buf: &str, tag: &str) -> usize {
    (1..tag.len().min(buf.len() + 1))
        .rev()
        .find(|&k| buf.is_char_boundary(buf.len() - k) && tag.starts_with(&buf[buf.len() - k..]))
        .unwrap_or(0)
}

/// What the output builder announces, in order: enough to render the
/// stream (each item added, its deltas, done), and to rebuild its items.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum OutputEvent {
    /// An item opened at `index`, status `in_progress`, no text yet.
    ItemAdded { index: usize, item: Value },
    ReasoningDelta {
        index: usize,
        item_id: String,
        delta: String,
    },
    TextDelta {
        index: usize,
        item_id: String,
        delta: String,
    },
    /// A call's whole arguments (`custom`: its `input`), in one payload.
    CallArgs {
        index: usize,
        item_id: String,
        /// The call's `name`, as its item has it.
        name: String,
        custom: bool,
        full: String,
    },
    /// The item at `index` finished, as [`OutputBuilder::items`] has it.
    ItemDone { index: usize, item: Value },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Reasoning,
    Message,
}

/// The item taking text now.
struct Open {
    kind: Kind,
    id: String,
    text: String,
}

/// Turns translated engine text into Responses output items (spec
/// "Output"): reasoning, message, then the calls.
pub(super) struct OutputBuilder {
    split: ReasoningSplit,
    thinking: bool,
    tools_active: bool,
    /// Created on the first answer text (when tools are active).
    tools: Option<ToolStreamSplitter>,
    map: Arc<ToolMap>,
    parallel: bool,
    encrypted: bool,
    /// Finished items, in output order.
    items: Vec<Value>,
    open: Option<Open>,
    /// The reasoning is over (its item, if any, closed).
    reasoning_done: bool,
    /// Released text of the next item, whitespace only so far.
    held: String,
    /// Bytes the reasoning split is holding.
    split_held: usize,
    /// Tokens of feeds that released nothing yet.
    carried: u32,
    /// Reasoning tokens by the per-chunk counts.
    reasoning: f64,
    /// The per-chunk counts' sum.
    observed: u32,
    /// Bytes released or consumed, and the reasoning share of them
    /// (`<think>`/`</think>` included).
    out_bytes: usize,
    reasoning_bytes: usize,
}

impl OutputBuilder {
    /// `thinking`/`opened` as [`ReasoningSplit::new`]; `tools_active` as
    /// [`ToolMap::tools_active`]; `parallel: false` keeps the first call;
    /// `encrypted` fills reasoning items' `encrypted_content`.
    pub fn new(
        thinking: bool,
        opened: bool,
        tools_active: bool,
        map: Arc<ToolMap>,
        parallel: bool,
        encrypted: bool,
    ) -> Self {
        Self {
            split: ReasoningSplit::new(thinking, opened),
            thinking,
            tools_active,
            tools: None,
            map,
            parallel,
            encrypted,
            items: Vec::new(),
            open: None,
            reasoning_done: false,
            held: String::new(),
            split_held: 0,
            carried: 0,
            reasoning: 0.0,
            observed: 0,
            out_bytes: 0,
            reasoning_bytes: 0,
        }
    }

    /// One chunk of translated text and the tokens it carried.
    pub fn feed(&mut self, text: &str, tokens: u32) -> Vec<OutputEvent> {
        let segs = self.split.feed(text);
        self.count(text.len(), &segs, tokens);
        let mut ev = Vec::new();
        self.route(segs, &mut ev);
        if self.split.consumed_close() {
            self.close_reasoning(&mut ev);
        }
        ev
    }

    /// The end of the generation. `item_status` (`completed` or
    /// `incomplete`) closes the open item when no call was parsed.
    pub fn finish(&mut self, item_status: &str) -> Vec<OutputEvent> {
        let segs = self.split.finish();
        self.count(0, &segs, 0);
        let mut ev = Vec::new();
        self.route(segs, &mut ev);
        if self.split.in_reasoning() {
            // `</think>` never came: the reasoning is the open item (or,
            // whitespace only, none).
            self.held.clear();
            self.close(item_status, &mut ev);
        }
        let mut calls = Vec::new();
        if let Some(t) = self.tools.as_mut() {
            let rest = match t.finish("") {
                ToolStreamEnd::Content(c) => c.to_owned(),
                ToolStreamEnd::ToolCalls { content, calls: c } => {
                    calls = c;
                    content.to_owned()
                }
            };
            self.message_text(&rest, &mut ev);
        }
        if calls.is_empty() {
            if self.open.is_none() && !self.held.is_empty() {
                let held = std::mem::take(&mut self.held);
                self.open_item(Kind::Message, &mut ev);
                self.delta(&held, &mut ev);
            }
            if self.open.is_some() {
                self.close(item_status, &mut ev);
            }
            return ev;
        }
        self.held.clear();
        if self.open.is_some() {
            self.close("completed", &mut ev);
        }
        if !self.parallel && calls.len() > 1 {
            tracing::debug!(
                dropped = calls.len() - 1,
                "responses: parallel_tool_calls is false; keeping the first call"
            );
            calls.truncate(1);
        }
        for c in calls {
            self.emit_call(c, &mut ev);
        }
        ev
    }

    /// The output so far: the finished items, then the open one (status
    /// `in_progress`).
    pub fn items(&self) -> Vec<Value> {
        self.items_closing_open("in_progress")
    }

    /// The items announced so far, the open one with `status`, and nothing
    /// the builder still holds: a failed stream's `output`, which no
    /// closing event precedes.
    pub fn items_closing_open(&self, status: &str) -> Vec<Value> {
        let mut items = self.items.clone();
        if let Some(o) = &self.open {
            items.push(self.item(o, status));
        }
        items
    }

    /// `usage.output_tokens_details.reasoning_tokens` of an `output_tokens`
    /// total, clamped to it. A total other than the per-chunk counts' sum
    /// (an engine's authoritative final count) scales the counted reasoning
    /// to it; when the chunks counted nothing, the reasoning share of the
    /// bytes splits it.
    pub fn reasoning_tokens(&self, output_tokens: u32) -> u32 {
        let total = f64::from(output_tokens);
        let reasoning = if self.observed == output_tokens {
            self.reasoning
        } else if self.observed > 0 {
            self.reasoning * total / f64::from(self.observed)
        } else if self.out_bytes > 0 {
            total * self.reasoning_bytes as f64 / self.out_bytes as f64
        } else {
            0.0
        };
        (reasoning.round() as u32).min(output_tokens)
    }

    /// Split a feed's tokens by the share of the bytes it released or
    /// consumed that went to reasoning (`<think>`/`</think>` included); a
    /// feed that only held text carries its tokens to the next one.
    fn count(&mut self, fed: usize, segs: &[Seg], tokens: u32) {
        self.observed += tokens;
        let pool = self.split_held + fed;
        self.split_held = self.split.held_len();
        let out = pool - self.split_held;
        let tokens = self.carried + tokens;
        if out == 0 {
            self.carried = tokens;
            return;
        }
        self.carried = 0;
        let answer: usize = segs
            .iter()
            .map(|s| match s {
                Seg::Answer(t) => t.len(),
                Seg::Reasoning(_) => 0,
            })
            .sum();
        self.out_bytes += out;
        self.reasoning_bytes += out - answer;
        self.reasoning += f64::from(tokens) * (out - answer) as f64 / out as f64;
    }

    fn route(&mut self, segs: Vec<Seg>, ev: &mut Vec<OutputEvent>) {
        for seg in segs {
            match seg {
                Seg::Reasoning(t) => self.text(Kind::Reasoning, &t, ev),
                Seg::Answer(t) => {
                    self.close_reasoning(ev);
                    if !self.tools_active {
                        self.message_text(&t, ev);
                        continue;
                    }
                    let (thinking, closed) = (self.thinking, self.split.consumed_close());
                    let tools = self
                        .tools
                        .get_or_insert_with(|| ToolStreamSplitter::new(thinking && !closed));
                    let released = tools.feed(&t).to_owned();
                    self.message_text(&released, ev);
                }
            }
        }
    }

    /// The reasoning ended: close its item; whitespace-only reasoning made
    /// none and is dropped.
    fn close_reasoning(&mut self, ev: &mut Vec<OutputEvent>) {
        if std::mem::replace(&mut self.reasoning_done, true) {
            return;
        }
        // No answer text yet, so whatever is held or open is reasoning.
        self.held.clear();
        self.close("completed", ev);
    }

    fn message_text(&mut self, text: &str, ev: &mut Vec<OutputEvent>) {
        self.text(Kind::Message, text, ev);
    }

    /// Text for an item of `kind`: a delta of the open one, or held until
    /// it is not whitespace only, which opens the item.
    fn text(&mut self, kind: Kind, text: &str, ev: &mut Vec<OutputEvent>) {
        if text.is_empty() {
            return;
        }
        if self.open.is_some() {
            self.delta(text, ev);
            return;
        }
        self.held.push_str(text);
        if !self.held.trim().is_empty() {
            let held = std::mem::take(&mut self.held);
            self.open_item(kind, ev);
            self.delta(&held, ev);
        }
    }

    fn open_item(&mut self, kind: Kind, ev: &mut Vec<OutputEvent>) {
        let prefix = match kind {
            Kind::Reasoning => "rs",
            Kind::Message => "msg",
        };
        let open = Open {
            kind,
            id: new_id(prefix),
            text: String::new(),
        };
        let mut item = self.item(&open, "in_progress");
        item["content"] = json!([]);
        ev.push(OutputEvent::ItemAdded {
            index: self.items.len(),
            item,
        });
        self.open = Some(open);
    }

    fn delta(&mut self, text: &str, ev: &mut Vec<OutputEvent>) {
        let index = self.items.len();
        let Some(o) = self.open.as_mut() else {
            return;
        };
        o.text.push_str(text);
        let (item_id, delta) = (o.id.clone(), text.to_owned());
        ev.push(match o.kind {
            Kind::Reasoning => OutputEvent::ReasoningDelta {
                index,
                item_id,
                delta,
            },
            Kind::Message => OutputEvent::TextDelta {
                index,
                item_id,
                delta,
            },
        });
    }

    fn close(&mut self, status: &str, ev: &mut Vec<OutputEvent>) {
        let Some(o) = self.open.take() else {
            return;
        };
        let item = self.item(&o, status);
        ev.push(OutputEvent::ItemDone {
            index: self.items.len(),
            item: item.clone(),
        });
        self.items.push(item);
    }

    /// The reasoning or message item `o` with `status`.
    fn item(&self, o: &Open, status: &str) -> Value {
        match o.kind {
            Kind::Reasoning => {
                let mut item = json!({
                    "id": o.id,
                    "type": "reasoning",
                    "status": status,
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": o.text}],
                });
                // Only when `include` asks for it, as OpenAI returns it.
                if self.encrypted && status != "in_progress" {
                    item["encrypted_content"] = json!(encode_reasoning(&o.text));
                }
                item
            }
            Kind::Message => json!({
                "id": o.id,
                "type": "message",
                "status": status,
                "role": "assistant",
                "content": [super::output_text(&o.text)],
            }),
        }
    }

    /// A parsed call: added (empty arguments), its arguments, done. A name
    /// the map does not know is a plain `function_call` under that name.
    fn emit_call(&mut self, call: ToolCall, ev: &mut Vec<OutputEvent>) {
        let flat = call.function.name;
        let (kind, name, namespace) = match self.map.get(&flat) {
            Some(t) => (t.kind, t.name.clone(), t.namespace.clone()),
            None => (Served::Function, flat, None),
        };
        let custom = kind == Served::Custom;
        let (id, field, full) = if custom {
            let input = custom_input(&call.function.arguments);
            (new_id("ctc"), "input", input)
        } else {
            (new_id("fc"), "arguments", call.function.arguments)
        };
        let mut item = json!({
            "id": id,
            "type": if custom { "custom_tool_call" } else { "function_call" },
            "call_id": call.id,
            "name": name,
            field: "",
        });
        if let Some(ns) = namespace {
            item["namespace"] = json!(ns);
        }
        if !custom {
            item["status"] = json!("in_progress");
        }
        let index = self.items.len();
        ev.push(OutputEvent::ItemAdded {
            index,
            item: item.clone(),
        });
        ev.push(OutputEvent::CallArgs {
            index,
            item_id: id,
            name: item["name"].as_str().unwrap_or_default().to_owned(),
            custom,
            full: full.clone(),
        });
        item[field] = json!(full);
        if !custom {
            item["status"] = json!("completed");
        }
        ev.push(OutputEvent::ItemDone {
            index,
            item: item.clone(),
        });
        self.items.push(item);
    }
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `chunks` then finish; merge adjacent segments of one kind.
    fn run(thinking: bool, opened: bool, chunks: &[&str]) -> (Vec<Seg>, ReasoningSplit) {
        let mut s = ReasoningSplit::new(thinking, opened);
        let mut segs = Vec::new();
        for c in chunks {
            segs.extend(s.feed(c));
        }
        segs.extend(s.finish());
        (merge(segs), s)
    }

    fn merge(segs: Vec<Seg>) -> Vec<Seg> {
        let mut out: Vec<Seg> = Vec::new();
        for seg in segs {
            match (out.last_mut(), seg) {
                (Some(Seg::Reasoning(a)), Seg::Reasoning(b)) => a.push_str(&b),
                (Some(Seg::Answer(a)), Seg::Answer(b)) => a.push_str(&b),
                (_, seg) => out.push(seg),
            }
        }
        out
    }

    fn r(s: &str) -> Seg {
        Seg::Reasoning(s.into())
    }

    fn a(s: &str) -> Seg {
        Seg::Answer(s.into())
    }

    #[test]
    fn thinking_off_is_all_answer_with_think_verbatim() {
        let (segs, s) = run(false, false, &["<think>x", "</think>y"]);
        assert_eq!(segs, vec![a("<think>x</think>y")]);
        assert!(!s.consumed_close());
        // even when the prompt opened it: thinking off means no split
        let (segs, _) = run(false, true, &["r</think>y"]);
        assert_eq!(segs, vec![a("r</think>y")]);
    }

    #[test]
    fn prompt_opened_splits_at_a_close_split_across_feeds() {
        let mut s = ReasoningSplit::new(true, true);
        assert!(s.in_reasoning());
        let mut segs = Vec::new();
        for c in ["ab", "c</th", "ink>\n", "\nhi"] {
            segs.extend(s.feed(c));
        }
        assert!(!s.in_reasoning());
        assert!(s.consumed_close());
        segs.extend(s.finish());
        assert_eq!(merge(segs), vec![r("abc"), a("hi")]);
    }

    #[test]
    fn partial_close_is_held_not_released_as_reasoning() {
        let mut s = ReasoningSplit::new(true, true);
        assert_eq!(merge(s.feed("ab</th")), vec![r("ab")]);
        assert!(s.in_reasoning());
    }

    #[test]
    fn output_opened_think_after_whitespace() {
        let (segs, s) = run(true, false, &["  <th", "ink>r</think>x"]);
        assert_eq!(segs, vec![r("r"), a("x")]);
        assert!(s.consumed_close());
    }

    #[test]
    fn undecided_resolves_on_the_first_non_whitespace_text() {
        let mut s = ReasoningSplit::new(true, false);
        assert!(!s.in_reasoning());
        assert_eq!(s.feed("hello"), vec![a("hello")]);
        assert_eq!(s.feed(" <think>"), vec![a(" <think>")]);
        assert!(s.finish().is_empty());
        assert!(!s.consumed_close());
    }

    #[test]
    fn a_longer_tag_is_answer() {
        let (segs, s) = run(true, false, &["<thinking"]);
        assert_eq!(segs, vec![a("<thinking")]);
        assert!(!s.in_reasoning());
    }

    #[test]
    fn unclosed_reasoning_flushes_as_reasoning() {
        let mut s = ReasoningSplit::new(true, false);
        let mut segs = s.feed("<think>r");
        assert!(s.in_reasoning());
        segs.extend(s.finish());
        assert_eq!(merge(segs), vec![r("r")]);
        // a held partial close is reasoning at the end
        let (segs, _) = run(true, true, &["r</thi"]);
        assert_eq!(segs, vec![r("r</thi")]);
    }

    #[test]
    fn whitespace_only_flushes_as_answer() {
        let mut s = ReasoningSplit::new(true, false);
        assert!(s.feed("  ").is_empty());
        assert_eq!(s.finish(), vec![a("  ")]);
        // a held `<think>` prefix is answer at the end too
        let (segs, _) = run(true, false, &["\n<thi"]);
        assert_eq!(segs, vec![a("\n<thi")]);
    }

    #[test]
    fn non_thinking_reply_is_byte_identical() {
        let (segs, _) = run(true, false, &["\n", "\nhello"]);
        assert_eq!(segs, vec![a("\n\nhello")]);
    }

    #[test]
    fn empty_reasoning_still_consumes_the_close() {
        let (segs, s) = run(true, false, &["<think></think>"]);
        assert!(segs.is_empty(), "{segs:?}");
        assert!(s.consumed_close());
        let (segs, s) = run(true, true, &["</think>\n\nanswer"]);
        assert_eq!(segs, vec![a("answer")]);
        assert!(s.consumed_close());
    }

    #[test]
    fn answer_whitespace_dropped_only_at_its_start() {
        let (segs, _) = run(true, true, &["r</think>", " ", "\n", "a b", " c"]);
        assert_eq!(segs, vec![r("r"), a("a b c")]);
    }

    #[test]
    fn multibyte_text_around_the_tags() {
        let (segs, _) = run(true, false, &["<think>é</", "think>ü"]);
        assert_eq!(segs, vec![r("é"), a("ü")]);
    }

    // ---- OutputBuilder ----

    use super::super::input::decode_reasoning;
    use serde_json::json;

    fn tool_map(tools: Value) -> Arc<ToolMap> {
        Arc::new(ToolMap::from_request(Some(&tools)).unwrap())
    }

    fn weather() -> Arc<ToolMap> {
        tool_map(json!([{"type": "function", "name": "get_weather",
            "parameters": {"type": "object"}}]))
    }

    fn call(name: &str, args: &str) -> String {
        format!("<tool_call>{{\"name\": \"{name}\", \"arguments\": {args}}}</tool_call>")
    }

    /// Feed `chunks` (one token each), finish with `status`, and check the
    /// events tell the same story as `items()`: every item added once at its
    /// index, its deltas adding up to its text, then done as `items()` has it.
    fn drive(b: &mut OutputBuilder, chunks: &[&str], status: &str) -> Vec<OutputEvent> {
        let mut ev = Vec::new();
        for c in chunks {
            ev.extend(b.feed(c, 1));
        }
        ev.extend(b.finish(status));
        let items = b.items();
        let mut done = Vec::new();
        let mut added = 0;
        let mut text = String::new();
        for e in &ev {
            match e {
                OutputEvent::ItemAdded { index, item } => {
                    assert_eq!(*index, added, "{ev:?}");
                    assert_eq!(
                        item["status"].as_str().unwrap_or("in_progress"),
                        "in_progress"
                    );
                    added += 1;
                    text.clear();
                }
                OutputEvent::ReasoningDelta { index, delta, .. }
                | OutputEvent::TextDelta { index, delta, .. } => {
                    assert_eq!(*index + 1, added, "{ev:?}");
                    text.push_str(delta);
                }
                OutputEvent::CallArgs { index, .. } => assert_eq!(*index + 1, added),
                OutputEvent::ItemDone { index, item } => {
                    assert_eq!(*index, done.len(), "{ev:?}");
                    let full = item["content"][0]["text"].as_str();
                    if let Some(full) = full {
                        assert_eq!(full, text, "{ev:?}");
                    }
                    done.push(item.clone());
                }
            }
        }
        assert_eq!(added, done.len(), "{ev:?}");
        assert_eq!(done, items, "{ev:?}");
        ev
    }

    fn kinds(items: &[Value]) -> Vec<&str> {
        items.iter().map(|i| i["type"].as_str().unwrap()).collect()
    }

    fn text_of(item: &Value) -> &str {
        item["content"][0]["text"].as_str().unwrap()
    }

    fn plain(thinking: bool, opened: bool) -> OutputBuilder {
        OutputBuilder::new(thinking, opened, false, Arc::default(), true, false)
    }

    fn with_tools(thinking: bool, map: Arc<ToolMap>, parallel: bool) -> OutputBuilder {
        OutputBuilder::new(thinking, false, true, map, parallel, false)
    }

    #[test]
    fn text_only_is_one_message() {
        let mut b = plain(false, false);
        drive(&mut b, &["hel", "lo"], "completed");
        let items = b.items();
        assert_eq!(kinds(&items), ["message"]);
        assert_eq!(text_of(&items[0]), "hello");
        assert_eq!(items[0]["status"], "completed");
        assert_eq!(items[0]["role"], "assistant");
        assert!(items[0]["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(b.reasoning_tokens(2), 0);
    }

    #[test]
    fn prompt_opened_reasoning_then_text() {
        let mut b = plain(true, true);
        drive(&mut b, &["plan", "</think>", "\n\nhi"], "completed");
        let items = b.items();
        assert_eq!(kinds(&items), ["reasoning", "message"]);
        let r = &items[0];
        assert!(r["id"].as_str().unwrap().starts_with("rs_"));
        assert_eq!(r["summary"], json!([]));
        assert_eq!(
            r["content"],
            json!([{"type": "reasoning_text", "text": "plan"}])
        );
        assert!(r["encrypted_content"].is_null());
        assert_eq!(r["status"], "completed");
        assert_eq!(text_of(&items[1]), "hi");
    }

    #[test]
    fn reasoning_then_a_call_counts_reasoning_tokens() {
        let mut b = with_tools(true, weather(), true);
        let args = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
        let ev = drive(
            &mut b,
            &[
                "<think>",
                "plan",
                "</think>",
                "<tool_call>",
                args,
                "</tool_call>",
            ],
            "completed",
        );
        let items = b.items();
        assert_eq!(kinds(&items), ["reasoning", "function_call"]);
        let c = &items[1];
        assert!(c["id"].as_str().unwrap().starts_with("fc_"));
        assert!(c["call_id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(c["name"], "get_weather");
        assert_eq!(c["status"], "completed");
        assert!(c.get("namespace").is_none(), "{c}");
        let parsed: Value = serde_json::from_str(c["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(parsed, json!({"city": "Paris"}));
        assert_eq!(b.reasoning_tokens(6), 3);
        // the call is announced with empty arguments, then one payload
        let added = ev.iter().find_map(|e| match e {
            OutputEvent::ItemAdded { index: 1, item } => Some(item),
            _ => None,
        });
        assert_eq!(added.unwrap()["arguments"], "");
        assert!(ev.iter().any(|e| matches!(e,
            OutputEvent::CallArgs { index: 1, custom: false, full, .. } if *full == c["arguments"])));
    }

    #[test]
    fn one_chunk_is_split_by_its_bytes() {
        let mut b = plain(true, false);
        let mut ev = b.feed("<think>plan</think>answer", 10);
        ev.extend(b.finish("completed"));
        assert_eq!(kinds(&b.items()), ["reasoning", "message"]);
        // 19 of 25 bytes (tags included) are reasoning: 7.6 → 8
        assert_eq!(b.reasoning_tokens(10), 8);
        // an authoritative total of 5 scales it: 3.8 → 4
        assert_eq!(b.reasoning_tokens(5), 4);
    }

    /// Chunks that carried no tokens, then an authoritative total: the
    /// reasoning share of every released byte (tags included) splits it.
    #[test]
    fn uncounted_chunks_split_the_total_by_bytes() {
        let mut b = plain(true, false);
        b.feed("<think>plan</think>", 0);
        b.feed("answer", 0);
        b.finish("completed");
        // 19 of 25 bytes are reasoning: 7.6 → 8
        assert_eq!(b.reasoning_tokens(10), 8);
        assert_eq!(b.reasoning_tokens(0), 0);
        // nothing released at all: none
        let mut b = plain(true, false);
        b.finish("completed");
        assert_eq!(b.reasoning_tokens(3), 0);
    }

    /// Per-chunk counts that disagree with the authoritative total are
    /// scaled to it.
    #[test]
    fn counted_chunks_are_scaled_to_the_total() {
        let mut b = plain(true, false);
        // 1 token all reasoning, then 1 token 8 of 10 bytes reasoning: 1.8
        b.feed("<think>ab", 1);
        b.feed("</think>xy", 1);
        b.finish("completed");
        assert_eq!(b.reasoning_tokens(2), 2);
        // ×5: 9
        assert_eq!(b.reasoning_tokens(10), 9);
        // ×0.5: 0.9 → 1
        assert_eq!(b.reasoning_tokens(1), 1);
    }

    #[test]
    fn held_tokens_are_carried_to_the_next_release() {
        // the `<th` feed holds everything; its token goes with the next feed
        let mut b = plain(true, false);
        drive(&mut b, &["<th", "ink>ab", "</think>x"], "completed");
        // feed 2: 3 + 4 bytes all reasoning, 2 tokens; feed 3: 8 of 9 bytes
        // of 1 token: 2.89 → 3
        assert_eq!(b.reasoning_tokens(3), 3);
        // finish attributes what is still carried
        let mut b = plain(true, false);
        b.feed("\n", 4);
        b.finish("completed");
        assert_eq!(b.reasoning_tokens(4), 0);
        let mut b = plain(true, true);
        b.feed("ab</thi", 4);
        b.finish("incomplete");
        assert_eq!(b.reasoning_tokens(4), 4);
    }

    #[test]
    fn text_before_a_call_is_a_message() {
        let mut b = with_tools(false, weather(), true);
        let tc = call("get_weather", "{}");
        drive(&mut b, &["Let me check", &tc], "completed");
        let items = b.items();
        assert_eq!(kinds(&items), ["message", "function_call"]);
        assert_eq!(text_of(&items[0]), "Let me check");
        assert_eq!(items[0]["status"], "completed");
    }

    #[test]
    fn parallel_false_keeps_the_first_call() {
        let two = format!(
            "{}{}",
            call("get_weather", r#"{"city": "A"}"#),
            call("get_weather", r#"{"city": "B"}"#)
        );
        let mut b = with_tools(false, weather(), false);
        drive(&mut b, &[&two], "completed");
        let items = b.items();
        assert_eq!(kinds(&items), ["function_call"]);
        assert!(items[0]["arguments"].as_str().unwrap().contains("\"A\""));
        let mut b = with_tools(false, weather(), true);
        drive(&mut b, &[&two], "completed");
        assert_eq!(kinds(&b.items()), ["function_call", "function_call"]);
    }

    #[test]
    fn namespaced_call_names_its_namespace() {
        let map = tool_map(
            json!([{"type": "namespace", "name": "mcp__x", "description": "x",
            "tools": [{"type": "function", "name": "click"}]}]),
        );
        let mut b = with_tools(false, map, true);
        drive(&mut b, &[&call("mcp__x__click", "{}")], "completed");
        let items = b.items();
        assert_eq!(items[0]["name"], "click");
        assert_eq!(items[0]["namespace"], "mcp__x");
    }

    #[test]
    fn custom_call_recovers_its_input() {
        let map = tool_map(json!([{"type": "custom", "name": "apply_patch"}]));
        let mut b = with_tools(false, map, true);
        let ev = drive(
            &mut b,
            &[&call("apply_patch", r#"{"input": "*** Begin Patch"}"#)],
            "completed",
        );
        let items = b.items();
        assert_eq!(kinds(&items), ["custom_tool_call"]);
        let c = &items[0];
        assert!(c["id"].as_str().unwrap().starts_with("ctc_"));
        assert!(c["call_id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(c["name"], "apply_patch");
        assert_eq!(c["input"], "*** Begin Patch");
        assert!(ev.iter().any(|e| matches!(e,
            OutputEvent::CallArgs { custom: true, full, .. } if full == "*** Begin Patch")));
    }

    #[test]
    fn a_call_cut_at_the_cap_stays_text() {
        let raw = r#"<tool_call>{"name": "get_weather", "argu"#;
        let mut b = with_tools(false, weather(), true);
        drive(&mut b, &["ok ", raw], "incomplete");
        let items = b.items();
        assert_eq!(kinds(&items), ["message"]);
        assert_eq!(items[0]["status"], "incomplete");
        assert_eq!(text_of(&items[0]), format!("ok {raw}"));
    }

    #[test]
    fn encrypted_content_decodes_back() {
        let mut b = OutputBuilder::new(true, true, false, Arc::default(), true, true);
        drive(&mut b, &["my plan", "</think>a"], "completed");
        let r = &b.items()[0];
        let blob = r["encrypted_content"].as_str().unwrap();
        assert_eq!(decode_reasoning(blob).as_deref(), Some("my plan"));
    }

    #[test]
    fn inactive_tools_leave_calls_as_text() {
        let map = weather();
        assert!(!map.tools_active(Some("none")));
        let tc = call("get_weather", "{}");
        let active = map.tools_active(Some("none"));
        let mut b = OutputBuilder::new(false, false, active, map, true, false);
        drive(&mut b, &[&tc], "completed");
        let items = b.items();
        assert_eq!(kinds(&items), ["message"]);
        assert_eq!(text_of(&items[0]), tc);
    }

    #[test]
    fn empty_reasoning_makes_no_item() {
        let mut b = plain(true, false);
        let ev = drive(&mut b, &["<think>\n\n</think>answer"], "completed");
        assert_eq!(kinds(&b.items()), ["message"]);
        assert_eq!(text_of(&b.items()[0]), "answer");
        assert!(!ev
            .iter()
            .any(|e| matches!(e, OutputEvent::ReasoningDelta { .. })));
    }

    #[test]
    fn an_undecided_prefix_at_the_end_is_text() {
        let mut b = plain(true, false);
        drive(&mut b, &["<th"], "completed");
        assert_eq!(text_of(&b.items()[0]), "<th");
    }

    #[test]
    fn unknown_name_is_a_plain_function_call() {
        let mut b = with_tools(false, weather(), true);
        drive(&mut b, &[&call("nope", "{}")], "completed");
        let items = b.items();
        assert_eq!(kinds(&items), ["function_call"]);
        assert_eq!(items[0]["name"], "nope");
        assert!(items[0].get("namespace").is_none());
    }

    #[test]
    fn whitespace_before_a_call_opens_no_message() {
        for thinking in [false, true] {
            let mut b = with_tools(thinking, weather(), true);
            let ev = drive(&mut b, &["\n", &call("get_weather", "{}")], "completed");
            assert_eq!(kinds(&b.items()), ["function_call"]);
            assert!(!ev.iter().any(|e| matches!(e,
                OutputEvent::ItemAdded { item, .. } if item["type"] == "message")));
        }
    }

    #[test]
    fn leading_whitespace_of_a_reply_is_kept() {
        for thinking in [false, true] {
            let mut b = plain(thinking, false);
            drive(&mut b, &["\n", "\nhi"], "completed");
            assert_eq!(text_of(&b.items()[0]), "\n\nhi");
        }
    }

    #[test]
    fn a_reply_that_never_opened_think_scans_past_a_stray_close() {
        let draft = call("get_weather", r#"{"city": "Draft"}"#);
        let real = call("get_weather", r#"{"city": "Real"}"#);
        let mut b = with_tools(true, weather(), true);
        drive(&mut b, &[&format!("{draft}</think>{real}")], "completed");
        let items = b.items();
        assert_eq!(kinds(&items), ["message", "function_call"]);
        assert_eq!(text_of(&items[0]), format!("{draft}</think>"));
        assert!(items[1]["arguments"].as_str().unwrap().contains("Real"));
    }
}
