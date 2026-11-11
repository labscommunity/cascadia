//! Incremental tool-call deltas for a streamed chat reply with tools active.
//!
//! The emitter reads the text the tool splitter holds back and turns each
//! `<tool_call>` block into OpenAI tool-call deltas as soon as it can: a
//! Hermes/Qwen JSON block (`{"name": …, "arguments": {`) sends its header as
//! soon as the arguments object opens and then the object's raw text as it
//! is generated; any other block is parsed whole when its closing tag
//! arrives. A call is committed once its header is out: later text never
//! takes it back.

use crate::{parse_tool_call_block, ToolCall, TOOL_CALL_CLOSE, TOOL_CALL_OPEN};

/// One tool-call delta: a call's header, or the next piece of its arguments.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CallEvent {
    Header {
        index: usize,
        id: String,
        name: String,
    },
    Arguments {
        index: usize,
        fragment: String,
    },
}

impl CallEvent {
    /// The chat `delta` carrying this event: a `tool_calls` entry, or for a
    /// `functions` request, `function_call`.
    pub(crate) fn delta(&self, legacy: bool) -> serde_json::Map<String, serde_json::Value> {
        let mut delta = serde_json::Map::new();
        let (key, value) = match (self, legacy) {
            (Self::Header { name, .. }, true) => (
                "function_call",
                serde_json::json!({"name": name, "arguments": ""}),
            ),
            (Self::Arguments { fragment, .. }, true) => {
                ("function_call", serde_json::json!({"arguments": fragment}))
            }
            (Self::Header { index, id, name }, false) => (
                "tool_calls",
                serde_json::json!([{"index": index, "id": id, "type": "function",
                    "function": {"name": name, "arguments": ""}}]),
            ),
            (Self::Arguments { index, fragment }, false) => (
                "tool_calls",
                serde_json::json!([{"index": index, "function": {"arguments": fragment}}]),
            ),
        };
        delta.insert(key.into(), value);
        delta
    }
}

/// The events for calls parsed whole at the end of a reply, numbered from
/// 0 (nothing was streamed before them); a `functions` request takes only
/// the first.
pub(crate) fn whole_calls(calls: Vec<ToolCall>, legacy: bool) -> Vec<CallEvent> {
    let take = if legacy { 1 } else { calls.len() };
    calls
        .into_iter()
        .take(take)
        .enumerate()
        .flat_map(|(index, call)| call_events(index, call))
        .collect()
}

/// Header plus one arguments delta for a call parsed whole.
fn call_events(index: usize, call: ToolCall) -> Vec<CallEvent> {
    let mut out = vec![CallEvent::Header {
        index,
        id: call.id,
        name: call.function.name,
    }];
    if !call.function.arguments.is_empty() {
        out.push(CallEvent::Arguments {
            index,
            fragment: call.function.arguments,
        });
    }
    out
}

/// Turns the held text of a tools stream into call deltas. Driven with the
/// splitter's whole buffer and where tool-call scanning starts (`None` while
/// a reasoning block is still open: a call drafted there is not a call).
pub(crate) struct ToolCallEmitter {
    /// Only the first call goes out (`functions` request).
    legacy: bool,
    /// Calls sent so far; the next call's index.
    sent: usize,
    state: State,
}

enum State {
    /// Scanning has not been allowed yet.
    Waiting,
    /// Looking for the next `<tool_call>` from this byte.
    Search(usize),
    /// Inside a block whose form is not known yet, or that is parsed whole
    /// at its closing tag; the byte right after its opening tag.
    Block(usize),
    /// Streaming a Hermes call's arguments object.
    Arguments {
        index: usize,
        scan: JsonScan,
        /// Bytes before this one are sent.
        sent_to: usize,
    },
    /// The arguments object closed; the rest of the call object follows.
    Rest(JsonScan),
    /// The call object closed; waiting for the block's closing tag from
    /// this byte.
    Close(usize),
    /// Nothing more is sent.
    Done,
}

impl ToolCallEmitter {
    pub(crate) fn new(legacy: bool) -> Self {
        Self {
            legacy,
            sent: 0,
            state: State::Waiting,
        }
    }

    /// Whether any call has been sent: from then on the reply is a call.
    pub(crate) fn started(&self) -> bool {
        self.sent > 0
    }

    /// A streamed call's arguments object is still open.
    pub(crate) fn inside_arguments(&self) -> bool {
        matches!(self.state, State::Arguments { .. })
    }

    /// The events the text so far allows.
    pub(crate) fn advance(&mut self, buf: &str, scan_from: Option<usize>) -> Vec<CallEvent> {
        self.run(buf, scan_from, false)
    }

    /// The reply ended: the events for the rest of it, every block still
    /// open parsed as `parse_tool_calls` parses it. A streamed call's
    /// arguments end where the reply did.
    pub(crate) fn finish(&mut self, buf: &str, scan_from: Option<usize>) -> Vec<CallEvent> {
        self.run(buf, scan_from, true)
    }

    fn run(&mut self, buf: &str, scan_from: Option<usize>, end: bool) -> Vec<CallEvent> {
        let mut out = Vec::new();
        if matches!(self.state, State::Waiting) {
            match scan_from {
                Some(at) => self.state = State::Search(at),
                None => return out,
            }
        }
        while self.step(buf, end, &mut out) {}
        out
    }

    fn next_index(&mut self) -> usize {
        self.sent += 1;
        self.sent - 1
    }

    /// One transition; `false` when it needs more text (or nothing more is
    /// sent).
    fn step(&mut self, buf: &str, end: bool, out: &mut Vec<CallEvent>) -> bool {
        match self.state {
            State::Waiting | State::Done => false,
            State::Search(_) if self.legacy && self.sent > 0 => {
                self.state = State::Done;
                false
            }
            State::Search(from) => match buf.get(from..).and_then(|r| r.find(TOOL_CALL_OPEN)) {
                Some(i) => {
                    self.state = State::Block(from + i + TOOL_CALL_OPEN.len());
                    true
                }
                None => false,
            },
            State::Block(after) => {
                let Some(text) = buf.get(after..) else {
                    return false;
                };
                let inner = text.trim_start();
                if inner.starts_with('{') {
                    match hermes_prefix(inner) {
                        Prefix::Match { name, args_at } => {
                            let index = self.next_index();
                            out.push(CallEvent::Header {
                                index,
                                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                                name,
                            });
                            let start = after + (text.len() - inner.len()) + args_at;
                            self.state = State::Arguments {
                                index,
                                scan: JsonScan::at(start, 0),
                                sent_to: start,
                            };
                            return true;
                        }
                        Prefix::NeedMore if !end => return false,
                        Prefix::NeedMore | Prefix::No => {}
                    }
                }
                if !end && !block_closed(inner) {
                    return false;
                }
                match parse_tool_call_block(text) {
                    Some(block) => {
                        if let Some(call) = block.call {
                            let index = self.next_index();
                            out.extend(call_events(index, call));
                        }
                        self.state = State::Search(after + block.consumed);
                        true
                    }
                    None => {
                        self.state = State::Done;
                        false
                    }
                }
            }
            State::Arguments {
                index,
                mut scan,
                sent_to,
            } => {
                let scanned = scan.run(buf.as_bytes());
                let upto = match scanned {
                    Scanned::Closed(at) | Scanned::Tag(at) => at,
                    Scanned::Wait => scan.pos,
                };
                if upto > sent_to {
                    if let Some(fragment) = buf.get(sent_to..upto) {
                        out.push(CallEvent::Arguments {
                            index,
                            fragment: fragment.to_owned(),
                        });
                    }
                }
                match scanned {
                    // The rest of the call object (`}` and any other keys)
                    // is read, not sent.
                    Scanned::Closed(at) => self.state = State::Rest(JsonScan::at(at, 1)),
                    // A closing tag outside any string: the model gave up on
                    // the object; the call stays as sent.
                    Scanned::Tag(at) => self.state = State::Search(at + TOOL_CALL_CLOSE.len()),
                    Scanned::Wait => {
                        self.state = State::Arguments {
                            index,
                            scan,
                            sent_to: upto,
                        };
                        return false;
                    }
                }
                true
            }
            State::Rest(mut scan) => match scan.run(buf.as_bytes()) {
                Scanned::Closed(at) => {
                    self.state = State::Close(at);
                    true
                }
                Scanned::Tag(at) => {
                    self.state = State::Search(at + TOOL_CALL_CLOSE.len());
                    true
                }
                Scanned::Wait => {
                    self.state = State::Rest(scan);
                    false
                }
            },
            // As `parse_tool_call_block`: the block runs to the first
            // closing tag after the object, or, with none, just the object.
            State::Close(from) => match buf.get(from..).and_then(|r| r.find(TOOL_CALL_CLOSE)) {
                Some(i) => {
                    self.state = State::Search(from + i + TOOL_CALL_CLOSE.len());
                    true
                }
                None if end => {
                    self.state = State::Search(from);
                    true
                }
                None => false,
            },
        }
    }
}

/// Whether a block that is parsed whole has all its text: a JSON value and
/// a closing tag after it, or for any other form, a closing tag. `inner` is
/// the block's text from its first non-space byte.
fn block_closed(inner: &str) -> bool {
    if inner.starts_with(['{', '[']) {
        // A closing tag inside an unbalanced value may be argument text.
        return crate::balanced_json_end(inner)
            .and_then(|len| inner.get(len..))
            .is_some_and(|tail| tail.contains(TOOL_CALL_CLOSE));
    }
    inner.contains(TOOL_CALL_CLOSE)
}

/// The Hermes call prefix `{"name": "<name>", "arguments": {`.
enum Prefix {
    /// The name, JSON-decoded, and where the arguments object's `{` is.
    Match {
        name: String,
        args_at: usize,
    },
    /// The text so far is a prefix of it.
    NeedMore,
    No,
}

/// Matches `{ "name" : "<name>" , ("arguments"|"parameters") : {` with JSON
/// whitespace anywhere between the tokens.
fn hermes_prefix(s: &str) -> Prefix {
    match hermes_prefix_at(s) {
        Ok(p) | Err(p) => p,
    }
}

fn hermes_prefix_at(s: &str) -> Result<Prefix, Prefix> {
    let b = s.as_bytes();
    let mut i = lit(b, ws(b, 0), b"{")?;
    i = lit(b, ws(b, i), b"\"name\"")?;
    i = lit(b, ws(b, i), b":")?;
    let start = ws(b, i);
    lit(b, start, b"\"")?;
    let end = string_end(b, start + 1)?;
    let name: String = s
        .get(start..end)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .ok_or(Prefix::No)?;
    i = ws(b, lit(b, ws(b, end), b",")?);
    i = match lit(b, i, b"\"arguments\"") {
        Ok(j) => j,
        Err(a) => match lit(b, i, b"\"parameters\"") {
            Ok(j) => j,
            Err(p) if matches!(a, Prefix::NeedMore) || matches!(p, Prefix::NeedMore) => {
                return Err(Prefix::NeedMore)
            }
            Err(_) => return Err(Prefix::No),
        },
    };
    i = lit(b, ws(b, i), b":")?;
    let args_at = ws(b, i);
    lit(b, args_at, b"{")?;
    Ok(Prefix::Match { name, args_at })
}

/// First non-whitespace byte at/after `i`.
fn ws(b: &[u8], i: usize) -> usize {
    i + b
        .get(i..)
        .unwrap_or_default()
        .iter()
        .take_while(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r'))
        .count()
}

/// `token` at `i`: the byte after it, or whether more text could still make
/// it.
fn lit(b: &[u8], i: usize, token: &[u8]) -> Result<usize, Prefix> {
    let rest = b.get(i..).unwrap_or_default();
    if rest.starts_with(token) {
        Ok(i + token.len())
    } else if token.starts_with(rest) {
        Err(Prefix::NeedMore)
    } else {
        Err(Prefix::No)
    }
}

/// The byte after the closing quote of the JSON string whose body starts at
/// `i`.
fn string_end(b: &[u8], i: usize) -> Result<usize, Prefix> {
    let mut esc = false;
    for (j, &c) in b.iter().enumerate().skip(i) {
        if esc {
            esc = false;
        } else if c == b'\\' {
            esc = true;
        } else if c == b'"' {
            return Ok(j + 1);
        }
    }
    Err(Prefix::NeedMore)
}

/// String- and escape-aware brace scan over a JSON object's text, resumable
/// as more text arrives. Counts `{`/`}` only, as `balanced_json_end` does.
#[derive(Clone, Copy, Debug)]
struct JsonScan {
    /// Next byte to read.
    pos: usize,
    depth: u32,
    in_str: bool,
    esc: bool,
}

enum Scanned {
    /// The object closed; the byte after its `}`.
    Closed(usize),
    /// A closing tag outside any string starts here.
    Tag(usize),
    /// Out of text (or at a `<` that may start a closing tag).
    Wait,
}

impl JsonScan {
    fn at(pos: usize, depth: u32) -> Self {
        Self {
            pos,
            depth,
            in_str: false,
            esc: false,
        }
    }

    fn run(&mut self, buf: &[u8]) -> Scanned {
        while let Some(&b) = buf.get(self.pos) {
            if self.in_str {
                if self.esc {
                    self.esc = false;
                } else if b == b'\\' {
                    self.esc = true;
                } else if b == b'"' {
                    self.in_str = false;
                }
            } else {
                match b {
                    b'"' => self.in_str = true,
                    b'{' => self.depth += 1,
                    b'}' => {
                        self.depth = self.depth.saturating_sub(1);
                        if self.depth == 0 {
                            self.pos += 1;
                            return Scanned::Closed(self.pos);
                        }
                    }
                    // JSON has no `<` outside a string, so this is markup.
                    b'<' => {
                        let rest = buf.get(self.pos..).unwrap_or_default();
                        if rest.starts_with(TOOL_CALL_CLOSE.as_bytes()) {
                            return Scanned::Tag(self.pos);
                        }
                        if TOOL_CALL_CLOSE.as_bytes().starts_with(rest) {
                            return Scanned::Wait;
                        }
                    }
                    _ => {}
                }
            }
            self.pos += 1;
        }
        Scanned::Wait
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolStreamSplitter;
    use serde_json::Value;

    /// What a stream of `pieces` produced: released content, the events
    /// with the piece that produced each, and the events after the end.
    struct Run {
        content: String,
        events: Vec<(usize, CallEvent)>,
    }

    fn run(pieces: &[&str], thinking: bool, legacy: bool) -> Run {
        let mut splitter = ToolStreamSplitter::new(thinking);
        let mut emitter = ToolCallEmitter::new(legacy);
        let mut content = String::new();
        let mut events = Vec::new();
        for (i, piece) in pieces.iter().enumerate() {
            content.push_str(splitter.feed(piece));
            for e in emitter.advance(splitter.text(), splitter.scan_from()) {
                events.push((i, e));
            }
        }
        for e in emitter.finish(splitter.text(), splitter.scan_from()) {
            events.push((pieces.len(), e));
        }
        Run { content, events }
    }

    /// Calls assembled by index: (id, name, arguments).
    fn assemble(events: &[(usize, CallEvent)]) -> Vec<(String, String, String)> {
        let mut calls: Vec<(String, String, String)> = Vec::new();
        for (_, e) in events {
            match e {
                CallEvent::Header { index, id, name } => {
                    assert_eq!(*index, calls.len(), "indices in order: {events:?}");
                    calls.push((id.clone(), name.clone(), String::new()));
                }
                CallEvent::Arguments { index, fragment } => {
                    assert!(!fragment.is_empty(), "empty fragment: {events:?}");
                    calls[*index].2.push_str(fragment);
                }
            }
        }
        calls
    }

    /// `text` cut into pieces of `n` chars.
    fn pieces(text: &str, n: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chars.chunks(n).map(|c| c.iter().collect()).collect()
    }

    fn json(s: &str) -> Value {
        serde_json::from_str(s).unwrap_or_else(|e| panic!("not JSON ({e}): {s}"))
    }

    /// The arguments `parse_tool_calls` gives for the same text.
    fn parsed_arguments(text: &str) -> Vec<Value> {
        crate::parse_tool_calls(text)
            .unwrap_or_default()
            .iter()
            .map(|c| json(&c.function.arguments))
            .collect()
    }

    const HERMES: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
        {\"city\": \"Paris\", \"days\": [1, 2], \"q\": {\"unit\": \"c\"}}}\n</tool_call>";

    #[test]
    fn a_hermes_call_streams_its_arguments_in_every_split() {
        let args_open = HERMES.find("\"arguments\": {").unwrap() + "\"arguments\": {".len();
        let args_close = HERMES.rfind("}}").unwrap();
        for n in [1, 2, 3, 7, HERMES.len()] {
            let parts = pieces(HERMES, n);
            let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
            let r = run(&refs, false, false);
            assert_eq!(r.content, "", "n={n}");
            let calls = assemble(&r.events);
            assert_eq!(calls.len(), 1, "n={n}: {:?}", r.events);
            assert_eq!(calls[0].1, "get_weather");
            assert!(calls[0].0.starts_with("call_") && calls[0].0.len() == 37);
            assert_eq!(vec![json(&calls[0].2)], parsed_arguments(HERMES), "n={n}");
            if n == 1 {
                // The header goes out with the arguments object's `{`, long
                // before the object closes.
                let (at, first) = &r.events[0];
                assert!(matches!(first, CallEvent::Header { .. }));
                assert_eq!(*at, args_open - 1, "{:?}", r.events);
                assert!(*at < args_close);
                let fragments = r
                    .events
                    .iter()
                    .filter(|(_, e)| matches!(e, CallEvent::Arguments { .. }))
                    .count();
                assert!(fragments > 10, "{:?}", r.events);
            }
        }
    }

    #[test]
    fn two_calls_get_indices_zero_and_one() {
        let text = format!("{HERMES}\n<tool_call>{{\"name\":\"get_time\",\"arguments\":{{\"tz\":\"CET\"}}}}</tool_call>");
        for n in [1, 5, text.len()] {
            let parts = pieces(&text, n);
            let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
            let calls = assemble(&run(&refs, false, false).events);
            let names: Vec<&str> = calls.iter().map(|c| c.1.as_str()).collect();
            assert_eq!(names, ["get_weather", "get_time"], "n={n}");
            assert_ne!(calls[0].0, calls[1].0);
            let args: Vec<Value> = calls.iter().map(|c| json(&c.2)).collect();
            assert_eq!(args, parsed_arguments(&text));
        }
    }

    #[test]
    fn parameters_is_an_alias_of_arguments() {
        let text = r#"<tool_call>{ "name" : "f" , "parameters" : { "x" : 1 } }</tool_call>"#;
        let r = run(
            &pieces(text, 1)
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            false,
            false,
        );
        let calls = assemble(&r.events);
        assert_eq!(calls.len(), 1);
        assert_eq!(json(&calls[0].2), serde_json::json!({"x": 1}));
        // Streamed: more than one fragment.
        assert!(r.events.len() > 2, "{:?}", r.events);
    }

    #[test]
    fn string_arguments_go_out_whole_at_the_closing_tag() {
        let text = r#"<tool_call>{"name":"f","arguments":"{\"a\":1}"}</tool_call>"#;
        let parts = pieces(text, 1);
        let r = run(
            &parts.iter().map(String::as_str).collect::<Vec<_>>(),
            false,
            false,
        );
        assert_eq!(r.events.len(), 2, "{:?}", r.events);
        // Both at the last byte of the closing tag.
        assert!(
            r.events.iter().all(|(at, _)| *at == parts.len() - 1),
            "{:?}",
            r.events
        );
        let calls = assemble(&r.events);
        assert_eq!(calls[0].1, "f");
        assert_eq!(calls[0].2, r#"{"a":1}"#);
    }

    #[test]
    fn an_xml_block_goes_out_whole_at_the_closing_tag() {
        let text = "<tool_call><function=f><parameter=x>1</parameter></function></tool_call>";
        let parts = pieces(text, 1);
        let r = run(
            &parts.iter().map(String::as_str).collect::<Vec<_>>(),
            false,
            false,
        );
        assert!(
            r.events.iter().all(|(at, _)| *at == parts.len() - 1),
            "{:?}",
            r.events
        );
        let calls = assemble(&r.events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "f");
        assert_eq!(json(&calls[0].2), serde_json::json!({"x": "1"}));
    }

    #[test]
    fn braces_and_closing_tags_inside_strings_do_not_end_the_object() {
        let text = r#"<tool_call>{"name":"f","arguments":{"s":"a}</tool_call>{b","t":"\"}\\","u":"日本🎉é"}}</tool_call>"#;
        for n in [1, 2, 4] {
            let parts = pieces(text, n);
            let calls = assemble(
                &run(
                    &parts.iter().map(String::as_str).collect::<Vec<_>>(),
                    false,
                    false,
                )
                .events,
            );
            assert_eq!(calls.len(), 1, "n={n}");
            assert_eq!(vec![json(&calls[0].2)], parsed_arguments(text), "n={n}");
        }
    }

    #[test]
    fn a_call_drafted_while_thinking_is_not_sent() {
        let text = r#"<think>maybe <tool_call>{"name":"rm","arguments":{"path":"/"}}</tool_call> no</think>The answer is 4"#;
        for n in [1, 3, text.len()] {
            let parts = pieces(text, n);
            let r = run(
                &parts.iter().map(String::as_str).collect::<Vec<_>>(),
                true,
                false,
            );
            assert!(r.events.is_empty(), "n={n}: {:?}", r.events);
        }
        let after = format!("<think>hmm</think>{HERMES}");
        let parts = pieces(&after, 1);
        let r = run(
            &parts.iter().map(String::as_str).collect::<Vec<_>>(),
            true,
            false,
        );
        let calls = assemble(&r.events);
        assert_eq!(calls.len(), 1);
        // Streamed, not decided at the end.
        assert!(r.events[0].0 < parts.len() - 5, "{:?}", r.events);
    }

    #[test]
    fn malformed_arguments_after_the_header_stay_one_call() {
        for text in [
            r#"<tool_call>{"name":"f","arguments":{a:1, b}}</tool_call>"#,
            r#"<tool_call>{"name":"f","arguments":{"a":"x}</tool_call>"#,
            r#"<tool_call>{"name":"f","arguments":{"a":1</tool_call>"#,
        ] {
            let parts = pieces(text, 1);
            let r = run(
                &parts.iter().map(String::as_str).collect::<Vec<_>>(),
                false,
                false,
            );
            assert_eq!(r.content, "");
            let calls = assemble(&r.events);
            assert_eq!(calls.len(), 1, "{text}: {:?}", r.events);
            assert_eq!(calls[0].1, "f");
            assert!(
                !calls[0].2.contains("</tool_call>") || text.contains("\"x}"),
                "{text}: {calls:?}"
            );
        }
        let r = run(
            &[r#"<tool_call>{"name":"f","arguments":{"a":1</tool_call>"#],
            false,
            false,
        );
        assert_eq!(assemble(&r.events)[0].2, r#"{"a":1"#);
    }

    #[test]
    fn a_hermes_call_then_an_xml_call() {
        let text = format!(
            "{HERMES}<tool_call><function=g><parameter=y>2</parameter></function></tool_call>"
        );
        for n in [1, 4, text.len()] {
            let parts = pieces(&text, n);
            let calls = assemble(
                &run(
                    &parts.iter().map(String::as_str).collect::<Vec<_>>(),
                    false,
                    false,
                )
                .events,
            );
            let names: Vec<&str> = calls.iter().map(|c| c.1.as_str()).collect();
            assert_eq!(names, ["get_weather", "g"], "n={n}");
        }
    }

    #[test]
    fn an_escaped_name_is_decoded() {
        let text = r#"<tool_call>{"name":"get_weather","arguments":{}}</tool_call>"#;
        let calls = assemble(&run(&[text], false, false).events);
        assert_eq!(calls[0].1, "get_weather");
        assert_eq!(calls[0].2, "{}");
    }

    #[test]
    fn content_before_a_call_is_released_and_the_markup_is_not() {
        let text = format!("Let me check. {HERMES}");
        let parts = pieces(&text, 2);
        let r = run(
            &parts.iter().map(String::as_str).collect::<Vec<_>>(),
            false,
            false,
        );
        assert_eq!(r.content, "Let me check. ");
        assert_eq!(assemble(&r.events).len(), 1);
    }

    #[test]
    fn a_functions_request_sends_only_the_first_call() {
        let text = format!("{HERMES}<tool_call>{{\"name\":\"g\",\"arguments\":{{}}}}</tool_call>");
        let r = run(
            &pieces(&text, 3)
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            false,
            true,
        );
        let calls = assemble(&r.events);
        assert_eq!(calls.len(), 1, "{:?}", r.events);
        assert_eq!(calls[0].1, "get_weather");
        let header = CallEvent::Header {
            index: 0,
            id: "call_x".into(),
            name: "f".into(),
        };
        assert_eq!(
            Value::Object(header.delta(true)),
            serde_json::json!({"function_call": {"name": "f", "arguments": ""}})
        );
        let args = CallEvent::Arguments {
            index: 0,
            fragment: "{".into(),
        };
        assert_eq!(
            Value::Object(args.delta(true)),
            serde_json::json!({"function_call": {"arguments": "{"}})
        );
        assert_eq!(
            Value::Object(args.delta(false)),
            serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]})
        );
    }

    #[test]
    fn a_reply_cut_inside_the_arguments_keeps_what_was_streamed() {
        let text = r#"<tool_call>{"name":"f","arguments":{"city":"Pa"#;
        let mut splitter = ToolStreamSplitter::new(false);
        let mut emitter = ToolCallEmitter::new(false);
        splitter.feed(text);
        let mut events = emitter.advance(splitter.text(), splitter.scan_from());
        assert!(emitter.inside_arguments());
        events.extend(emitter.finish(splitter.text(), splitter.scan_from()));
        let calls = assemble(&events.into_iter().map(|e| (0, e)).collect::<Vec<_>>());
        assert_eq!(calls[0].2, r#"{"city":"Pa"#);
    }

    #[test]
    fn an_unclosed_fallback_block_is_parsed_at_the_end() {
        // Whole JSON with string arguments and no closing tag: parsed as
        // `parse_tool_calls` would, at the end.
        let text = r#"<tool_call>{"name":"f","arguments":"{}"}"#;
        let r = run(&[text], false, false);
        let calls = assemble(&r.events);
        assert_eq!(calls.len(), 1, "{:?}", r.events);
        assert!(r.events.iter().all(|(at, _)| *at == 1));
    }

    #[test]
    fn streamed_calls_match_the_whole_reply_parse_in_every_split() {
        let texts = [
            HERMES.to_owned(),
            format!("Sure: {HERMES}"),
            format!("{HERMES}{HERMES}"),
            format!("{HERMES}<tool_call><function=g><parameter=y>2</parameter></function></tool_call>"),
            r#"<tool_call>{"arguments":{"a":1},"name":"f"}</tool_call>"#.to_owned(),
            r#"<tool_call>{"name":"f","arguments":"{\"a\":1}"}</tool_call>"#.to_owned(),
            r#"<tool_call>{"name":"f","arguments":{"é":"日本🎉","n":[{"x":"}"}]},"extra":1}</tool_call>"#
                .to_owned(),
            format!("{{\"x\": 1}} {HERMES}"),
            "<tool_call>get_time\n</tool_call>".to_owned(),
        ];
        for text in &texts {
            let want: Vec<(String, Value)> = crate::parse_tool_calls(text)
                .unwrap_or_default()
                .into_iter()
                .map(|c| (c.function.name, json(&c.function.arguments)))
                .collect();
            assert!(!want.is_empty(), "{text}");
            for n in 1..=9 {
                let parts = pieces(text, n);
                let r = run(
                    &parts.iter().map(String::as_str).collect::<Vec<_>>(),
                    false,
                    false,
                );
                let got: Vec<(String, Value)> = assemble(&r.events)
                    .into_iter()
                    .map(|(_, name, args)| (name, json(&args)))
                    .collect();
                assert_eq!(got, want, "n={n}: {text}");
            }
        }
    }
}
