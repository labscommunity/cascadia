//! Incremental tool-call deltas for a streamed chat reply with tools active.
//!
//! The emitter reads the text the tool splitter holds back and turns each
//! `<tool_call>` block into OpenAI tool-call deltas as soon as it can: a
//! Hermes/Qwen JSON block (`{"name": …, "arguments": {`) sends its header as
//! soon as the arguments object opens and then the object's raw text as it
//! is generated; any other block is parsed whole when its closing tag
//! arrives. A call is committed once its header is out: later text never
//! takes it back.

use crate::{parse_tool_call_block, CloseTags, ToolCall, TOOL_CALL_CLOSE, TOOL_CALL_OPEN};

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

/// Whether the served model writes a `<think>…</think>` reasoning block
/// before its answer: its chat template mentions `</think>` (Qwen3, GLM,
/// R1-distill templates all do), or its output framing has a thinking
/// marker that is translated into one. With no template at all the API
/// cannot tell, so it assumes one may come and keeps guarding against
/// calls drafted while reasoning.
pub(crate) fn model_thinks(template: Option<&str>, marker_dialect: bool) -> bool {
    marker_dialect || template.is_none_or(|t| t.contains(crate::THINK_CLOSE))
}

/// Turns the held text of a tools stream into call deltas. Driven with the
/// splitter's whole buffer and where tool-call scanning starts (`None` while
/// a reasoning block is still open: a call drafted there is not a call).
///
/// Every state resumes where the previous feed stopped, so a reply of `n`
/// bytes costs O(n) however finely it is fed.
pub(crate) struct ToolCallEmitter {
    /// Only the first call goes out (`functions` request).
    legacy: bool,
    /// Calls sent so far; the next call's index.
    sent: usize,
    state: State,
    /// Closing-tag searches of the blocks parsed whole, shared between them.
    tags: CloseTags,
    /// A streamed call's arguments were ended by a closing tag before their
    /// object closed: that call is incomplete.
    cut: bool,
    /// Bytes examined so far, for the linear-time test.
    #[cfg(test)]
    work: usize,
}

enum State {
    /// Scanning has not been allowed yet.
    Waiting,
    /// Looking for the next `<tool_call>`.
    Search(Finder),
    /// Inside a block whose form is not known yet, or that is parsed whole
    /// at its closing tag.
    Block(Block),
    /// Streaming a Hermes call's arguments object.
    Arguments {
        index: usize,
        scan: JsonScan,
        /// Bytes before this one are sent.
        sent_to: usize,
    },
    /// The arguments object closed; the rest of the call object follows.
    Rest(JsonScan),
    /// The call object closed; waiting for the block's closing tag.
    Close(Finder),
    /// Nothing more is sent.
    Done,
}

/// A `<tool_call>` block being read, from the byte after its opening tag.
struct Block {
    after: usize,
    /// First non-whitespace byte of the block, once seen; before that, how
    /// far the whitespace run has been read.
    lead: Result<usize, usize>,
    /// The Hermes prefix match, while it may still match.
    prefix: Option<PrefixScan>,
    /// Form decided as parsed whole: how far its end has been looked for.
    whole: Option<Whole>,
}

enum Whole {
    /// A JSON value: its brace scan.
    Value(JsonScan),
    /// After the JSON value (or for a non-JSON block, from the block's
    /// start): the closing tag search.
    Tag(Finder),
}

impl ToolCallEmitter {
    pub(crate) fn new(legacy: bool) -> Self {
        Self {
            legacy,
            sent: 0,
            state: State::Waiting,
            tags: CloseTags::default(),
            cut: false,
            #[cfg(test)]
            work: 0,
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

    /// A streamed call is incomplete: its arguments object is still open, or
    /// a closing tag ended it before it closed.
    pub(crate) fn call_cut_short(&self) -> bool {
        self.cut || self.inside_arguments()
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
                Some(at) => self.state = State::Search(Finder::new(at)),
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

    #[cfg_attr(not(test), allow(unused_variables))]
    fn charge(&mut self, bytes: usize) {
        #[cfg(test)]
        {
            self.work += bytes;
        }
    }

    /// One transition; `false` when it needs more text (or nothing more is
    /// sent).
    fn step(&mut self, buf: &str, end: bool, out: &mut Vec<CallEvent>) -> bool {
        match std::mem::replace(&mut self.state, State::Done) {
            State::Waiting => {
                self.state = State::Waiting;
                false
            }
            State::Done => false,
            State::Search(_) if self.legacy && self.sent > 0 => false,
            State::Search(mut finder) => {
                let (found, work) = finder.find(buf, TOOL_CALL_OPEN);
                self.charge(work);
                match found {
                    Some(at) => {
                        self.state = State::Block(Block {
                            after: at + TOOL_CALL_OPEN.len(),
                            lead: Err(at + TOOL_CALL_OPEN.len()),
                            prefix: None,
                            whole: None,
                        });
                        true
                    }
                    None => {
                        self.state = State::Search(finder);
                        false
                    }
                }
            }
            State::Block(block) => self.step_block(block, buf, end, out),
            State::Arguments {
                index,
                mut scan,
                sent_to,
            } => {
                let from = scan.pos;
                let scanned = scan.run(buf.as_bytes());
                self.charge(scan.pos - from);
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
                    Scanned::Closed(at) => self.state = State::Rest(JsonScan::object(at, 1)),
                    // A closing tag outside any string: the model gave up on
                    // the object; the call stays as sent, incomplete.
                    Scanned::Tag(at) => {
                        self.cut = true;
                        self.state = State::Search(Finder::new(at + TOOL_CALL_CLOSE.len()))
                    }
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
            State::Rest(mut scan) => {
                let from = scan.pos;
                let scanned = scan.run(buf.as_bytes());
                self.charge(scan.pos - from);
                match scanned {
                    Scanned::Closed(at) => {
                        self.state = State::Close(Finder::new(at));
                        true
                    }
                    Scanned::Tag(at) => {
                        self.state = State::Search(Finder::new(at + TOOL_CALL_CLOSE.len()));
                        true
                    }
                    Scanned::Wait => {
                        self.state = State::Rest(scan);
                        false
                    }
                }
            }
            // As `parse_tool_call_block`: the block runs to the first
            // closing tag after the object, or, with none, just the object.
            State::Close(mut finder) => {
                let (found, work) = finder.find(buf, TOOL_CALL_CLOSE);
                self.charge(work);
                match found {
                    Some(at) => {
                        self.state = State::Search(Finder::new(at + TOOL_CALL_CLOSE.len()));
                        true
                    }
                    None if end => {
                        self.state = State::Search(Finder::new(finder.from));
                        true
                    }
                    None => {
                        self.state = State::Close(finder);
                        false
                    }
                }
            }
        }
    }

    fn step_block(
        &mut self,
        mut block: Block,
        buf: &str,
        end: bool,
        out: &mut Vec<CallEvent>,
    ) -> bool {
        let lead = match block.lead {
            Ok(lead) => lead,
            Err(from) => {
                let rest = buf.get(from..).unwrap_or_default();
                let skipped = rest.len() - rest.trim_start().len();
                self.charge(skipped);
                if skipped == rest.len() && !end {
                    block.lead = Err(from + skipped);
                    self.state = State::Block(block);
                    return false;
                }
                let lead = from + skipped;
                block.lead = Ok(lead);
                if buf.as_bytes().get(lead) == Some(&b'{') {
                    block.prefix = Some(PrefixScan::new(lead));
                }
                lead
            }
        };
        if let Some(mut prefix) = block.prefix.take() {
            let from = prefix.pos;
            let matched = prefix.run(buf);
            self.charge(prefix.pos - from);
            match matched {
                Prefix::Match { name, args_at } => {
                    let index = self.next_index();
                    out.push(CallEvent::Header {
                        index,
                        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                        name,
                    });
                    self.state = State::Arguments {
                        index,
                        scan: JsonScan::arguments(args_at),
                        sent_to: args_at,
                    };
                    return true;
                }
                Prefix::NeedMore if !end => {
                    block.prefix = Some(prefix);
                    self.state = State::Block(block);
                    return false;
                }
                Prefix::NeedMore | Prefix::No => {}
            }
        }
        if !end && !self.block_closed(&mut block, buf, lead) {
            self.state = State::Block(block);
            return false;
        }
        // Parsed once, whole, by the shared parser: it reads the block, and
        // searches for closing tags through `tags`, which counts its own
        // bytes (a search within the block reads the same bytes again, so
        // the larger of the two is charged); a block that cannot end may
        // have read the rest of the text.
        let after = block.after;
        #[cfg(test)]
        let searched = self.tags.work;
        let parsed = parse_tool_call_block(buf, after, &mut self.tags);
        #[cfg(test)]
        {
            let read = match &parsed {
                Some(p) => p.consumed,
                None => buf.len().saturating_sub(after),
            };
            self.charge(read.max(self.tags.work - searched));
        }
        match parsed {
            Some(parsed) => {
                if let Some(call) = parsed.call {
                    let index = self.next_index();
                    out.extend(call_events(index, call));
                }
                self.state = State::Search(Finder::new(after + parsed.consumed));
                true
            }
            None => false,
        }
    }

    /// Whether a block parsed whole has all its text: a JSON value and a
    /// closing tag after it, or for any other form (including a value that
    /// reaches a `<` outside a string before it balances), a closing tag. A
    /// closing tag inside a string of an unbalanced JSON value may be
    /// argument text, so it does not count.
    fn block_closed(&mut self, block: &mut Block, buf: &str, lead: usize) -> bool {
        let whole = block
            .whole
            .get_or_insert_with(|| match buf.as_bytes().get(lead) {
                Some(b'{') => Whole::Value(JsonScan::value(lead, b'{', b'}')),
                Some(b'[') => Whole::Value(JsonScan::value(lead, b'[', b']')),
                _ => Whole::Tag(Finder::new(block.after)),
            });
        let mut work = 0;
        let closed = loop {
            match whole {
                Whole::Value(scan) => {
                    let from = scan.pos;
                    let scanned = scan.run(buf.as_bytes());
                    work += scan.pos - from;
                    match scanned {
                        Scanned::Closed(at) => *whole = Whole::Tag(Finder::new(at)),
                        // Not JSON: as `parse_tool_call_block`, the block
                        // runs to its first closing tag.
                        Scanned::Tag(_) => *whole = Whole::Tag(Finder::new(block.after)),
                        Scanned::Wait => break false,
                    }
                }
                Whole::Tag(finder) => {
                    let (found, w) = finder.find(buf, TOOL_CALL_CLOSE);
                    work += w;
                    break found.is_some();
                }
            }
        };
        self.charge(work);
        closed
    }
}

/// A resumable search for an ASCII marker: bytes before `seen` hold no
/// start of it, so each search reads only what arrived since the last one
/// (plus a possible partial marker at the end).
struct Finder {
    from: usize,
    seen: usize,
}

impl Finder {
    fn new(from: usize) -> Self {
        Self { from, seen: from }
    }

    /// The marker's position, and how many bytes were read.
    fn find(&mut self, buf: &str, marker: &str) -> (Option<usize>, usize) {
        let start = self.from.max(self.seen);
        let Some(hay) = buf.get(start..) else {
            return (None, 0);
        };
        if let Some(i) = hay.find(marker) {
            return (Some(start + i), hay.len().min(i + marker.len()));
        }
        // Resume at the first tail byte that could start the marker: every
        // such byte is ASCII, so this is a char boundary.
        let tail_from = buf.len().saturating_sub(marker.len() - 1).max(start);
        self.seen = (tail_from..buf.len())
            .find(|&k| buf.get(k..).is_some_and(|tail| marker.starts_with(tail)))
            .unwrap_or(buf.len());
        (None, hay.len())
    }
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

/// Resumable match of `{ "name" : "<name>" , ("arguments"|"parameters") :
/// {` with JSON whitespace between the tokens, one byte at a time.
///
/// The header goes out as soon as this prefix is read, so the stream
/// commits to the FIRST `name` and the FIRST `arguments`/`parameters`
/// object. A call object that repeats a key later is decided differently by
/// the whole-reply parse (`serde_json` keeps the last `name`, and
/// `arguments` wins over `parameters`); matching that would mean holding
/// every call until its object closes, which is what streaming the
/// arguments exists to avoid. A model writes such a key twice only by
/// mistake.
struct PrefixScan {
    /// Next byte to read.
    pos: usize,
    /// Which token is being read (see `run`).
    step: u8,
    /// Bytes of the current token read.
    lit: usize,
    /// `"arguments"` or `"parameters"`, once its second byte chose it.
    key: &'static [u8],
    /// The name string's opening quote.
    name_start: usize,
    esc: bool,
    name: String,
}

impl PrefixScan {
    fn new(pos: usize) -> Self {
        Self {
            pos,
            step: 0,
            lit: 0,
            key: b"",
            name_start: pos,
            esc: false,
            name: String::new(),
        }
    }

    fn run(&mut self, buf: &str) -> Prefix {
        let b = buf.as_bytes();
        while let Some(&c) = b.get(self.pos) {
            let pos = self.pos;
            self.pos += 1;
            if self.step == 4 {
                // Inside the name string.
                if self.esc {
                    self.esc = false;
                } else if c == b'\\' {
                    self.esc = true;
                } else if c == b'"' {
                    match buf
                        .get(self.name_start..self.pos)
                        .and_then(|raw| serde_json::from_str::<String>(raw).ok())
                    {
                        Some(name) => self.name = name,
                        None => return Prefix::No,
                    }
                    self.step = 5;
                }
                continue;
            }
            if self.lit == 0 && matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                continue;
            }
            let token: &[u8] = match (self.step, self.lit) {
                (0, _) | (8, _) => b"{",
                (1, _) => b"\"name\"",
                (2, _) | (7, _) => b":",
                (3, _) => b"\"",
                (5, _) => b",",
                (6, 0) => b"\"",
                (6, 1) => {
                    self.key = match c {
                        b'a' => b"\"arguments\"",
                        b'p' => b"\"parameters\"",
                        _ => return Prefix::No,
                    };
                    self.key
                }
                (6, _) => self.key,
                _ => return Prefix::No,
            };
            if token.get(self.lit) != Some(&c) {
                return Prefix::No;
            }
            self.lit += 1;
            if self.step == 6 && self.lit == 1 {
                continue;
            }
            if self.lit == token.len() {
                self.lit = 0;
                match self.step {
                    3 => {
                        self.name_start = pos;
                        self.step = 4;
                    }
                    8 => {
                        return Prefix::Match {
                            name: std::mem::take(&mut self.name),
                            args_at: pos,
                        }
                    }
                    s => self.step = s + 1,
                }
            }
        }
        Prefix::NeedMore
    }
}

/// String- and escape-aware bracket scan over a JSON value's text,
/// resumable as more text arrives. Counts only the value's own bracket
/// kind, as `balanced_json_end` does.
#[derive(Clone, Copy, Debug)]
struct JsonScan {
    /// Next byte to read.
    pos: usize,
    depth: u32,
    in_str: bool,
    esc: bool,
    open: u8,
    close: u8,
    /// Stop at a closing tag outside any string (a streamed call); when
    /// unset, at any `<` outside a string (a value parsed whole, which then
    /// is not JSON, as `balanced_json_end` reads it).
    tags: bool,
}

enum Scanned {
    /// The value closed; the byte after its closing bracket.
    Closed(usize),
    /// A closing tag (or, for a value parsed whole, any `<`) outside any
    /// string starts here.
    Tag(usize),
    /// Out of text (or at a `<` that may start a closing tag).
    Wait,
}

impl JsonScan {
    /// A streamed call's object, `depth` levels in.
    fn object(pos: usize, depth: u32) -> Self {
        Self {
            pos,
            depth,
            in_str: false,
            esc: false,
            open: b'{',
            close: b'}',
            tags: true,
        }
    }

    /// A streamed call's arguments object, from its `{`.
    fn arguments(pos: usize) -> Self {
        Self::object(pos, 0)
    }

    /// A value parsed whole, from its opening bracket.
    fn value(pos: usize, open: u8, close: u8) -> Self {
        Self {
            open,
            close,
            tags: false,
            ..Self::object(pos, 0)
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
            } else if b == b'"' {
                self.in_str = true;
            } else if b == self.open {
                self.depth += 1;
            } else if b == self.close {
                self.depth = self.depth.saturating_sub(1);
                if self.depth == 0 {
                    self.pos += 1;
                    return Scanned::Closed(self.pos);
                }
            } else if b == b'<' && !self.tags {
                return Scanned::Tag(self.pos);
            } else if b == b'<' {
                // JSON has no `<` outside a string, so this is markup.
                let rest = buf.get(self.pos..).unwrap_or_default();
                if rest.starts_with(TOOL_CALL_CLOSE.as_bytes()) {
                    return Scanned::Tag(self.pos);
                }
                if TOOL_CALL_CLOSE.as_bytes().starts_with(rest) {
                    return Scanned::Wait;
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

    /// A closing tag ends a streamed call's arguments before their object
    /// closed: the call stays as sent and is marked cut short, and the next
    /// block still streams, in every split.
    #[test]
    fn a_closing_tag_inside_the_arguments_cuts_the_call_short() {
        let text = r#"<tool_call>{"name":"f","arguments":{"a":1</tool_call><tool_call>{"name":"g","arguments":{}}</tool_call>"#;
        for n in [1, 2, 3, 5, 7, text.len()] {
            let parts = pieces(text, n);
            let mut splitter = ToolStreamSplitter::new(false);
            let mut emitter = ToolCallEmitter::new(false);
            let mut events = Vec::new();
            for (i, piece) in parts.iter().enumerate() {
                splitter.feed(piece);
                for e in emitter.advance(splitter.text(), splitter.scan_from()) {
                    events.push((i, e));
                }
            }
            assert!(emitter.call_cut_short(), "n={n}");
            assert!(!emitter.inside_arguments(), "n={n}");
            let streamed: Vec<_> = assemble(&events)
                .into_iter()
                .map(|(_, name, args)| (name, args))
                .collect();
            assert_eq!(
                streamed,
                [
                    ("f".to_owned(), r#"{"a":1"#.to_owned()),
                    ("g".to_owned(), "{}".to_owned())
                ],
                "n={n}"
            );
            assert!(
                emitter
                    .finish(splitter.text(), splitter.scan_from())
                    .is_empty(),
                "n={n}"
            );
        }
    }

    /// A block parsed whole that starts with `{` but is not JSON (a `<`
    /// outside a string) is decided when its closing tag arrives, not when
    /// the reply ends.
    #[test]
    fn a_non_json_block_starting_with_a_brace_is_decided_at_its_tag() {
        let block = "<tool_call>{<function=f><parameter=a>1</parameter></function></tool_call>";
        let text = format!("{block} and more text after it");
        for n in [1, 2, 3, 7, block.len()] {
            let parts = pieces(&text, n);
            let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
            let mut fed = 0;
            let tag_piece = refs
                .iter()
                .position(|p| {
                    fed += p.len();
                    fed >= block.len()
                })
                .unwrap();
            let r = run(&refs, false, false);
            assert!(!r.events.is_empty(), "n={n}");
            assert!(
                r.events.iter().all(|(i, _)| *i == tag_piece),
                "n={n}: want piece {tag_piece}: {:?}",
                r.events
            );
            let calls = assemble(&r.events);
            assert_eq!(calls.len(), 1, "n={n}");
            assert_eq!(calls[0].1, "f");
            assert_eq!(json(&calls[0].2), json(r#"{"a":"1"}"#));
        }
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

    /// A long reply fed one byte at a time costs linear work in every
    /// state, counting both the splitter's scans and the emitter's: a
    /// streamed argument, a value parsed whole, an XML block, prose with no
    /// call, a long name inside the Hermes prefix, and a long reasoning
    /// block before a call.
    #[test]
    fn feeding_a_byte_at_a_time_costs_linear_work() {
        let big = "x".repeat(64 * 1024);
        let call = r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#;
        let texts = [
            (
                format!(r#"<tool_call>{{"name":"f","arguments":{{"v":"{big}"}}}}</tool_call>"#),
                false,
                1,
            ),
            (
                format!(r#"<tool_call>{{"name":"f","arguments":"{big}"}}</tool_call>"#),
                false,
                1,
            ),
            (
                format!(
                    "<tool_call><function=f><parameter=v>{big}</parameter></function></tool_call>"
                ),
                false,
                1,
            ),
            (format!("{big} no call here"), false, 0),
            (
                format!(r#"<tool_call>{{"name":"{big}","arguments":{{}}}}</tool_call>"#),
                false,
                1,
            ),
            (format!("<think>{big}</think>{call}"), true, 1),
            (
                format!("{call}{}", "<tool_call>{</tool_call>".repeat(4000)),
                false,
                1,
            ),
            (format!("{call}{}", "<tool_call>{}".repeat(8000)), false, 1),
        ];
        for (text, thinking, want_calls) in &texts {
            let started = std::time::Instant::now();
            let mut splitter = ToolStreamSplitter::new(*thinking);
            let mut emitter = ToolCallEmitter::new(false);
            let mut events = Vec::new();
            for (i, c) in text.char_indices() {
                splitter.feed(&text[i..i + c.len_utf8()]);
                events.extend(
                    emitter
                        .advance(splitter.text(), splitter.scan_from())
                        .into_iter()
                        .map(|e| (i, e)),
                );
            }
            events.extend(
                emitter
                    .finish(splitter.text(), splitter.scan_from())
                    .into_iter()
                    .map(|e| (text.len(), e)),
            );
            let elapsed = started.elapsed();
            let head = &text[..40];
            assert_eq!(assemble(&events).len(), *want_calls, "{head}");
            let work = emitter.work + splitter.work.get();
            assert!(
                work < 8 * text.len(),
                "work {work} for {} bytes: {head}",
                text.len()
            );
            assert!(
                elapsed < std::time::Duration::from_secs(2),
                "{elapsed:?}: {head}"
            );
        }
    }

    /// Blocks decided only when the reply ends, each parsed once: blocks
    /// that never balance read up to their own closing tag, and blocks with
    /// no closing tag after them share one search for it.
    #[test]
    fn blocks_decided_at_the_end_cost_linear_work() {
        let call = r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#;
        for text in [
            format!("{call}{}", "<tool_call>{</tool_call>".repeat(4000)),
            format!("{call}{}", "<tool_call>{}".repeat(8000)),
        ] {
            let started = std::time::Instant::now();
            let mut splitter = ToolStreamSplitter::new(false);
            splitter.feed(&text);
            let mut emitter = ToolCallEmitter::new(false);
            let events: Vec<(usize, CallEvent)> = emitter
                .finish(splitter.text(), splitter.scan_from())
                .into_iter()
                .map(|e| (text.len(), e))
                .collect();
            let elapsed = started.elapsed();
            let head = &text[text.len() - 30..];
            assert_eq!(assemble(&events).len(), 1, "{head}");
            assert!(
                emitter.work < 8 * text.len(),
                "work {} for {} bytes: {head}",
                emitter.work,
                text.len()
            );
            assert!(
                elapsed < std::time::Duration::from_secs(2),
                "{elapsed:?}: {head}"
            );
        }
    }

    /// A call object that repeats a key: the stream has already sent the
    /// first `name` and the first arguments object when the repeat arrives,
    /// so it keeps those, while the whole-reply parse keeps the last `name`
    /// and prefers `arguments` over `parameters`.
    #[test]
    fn a_repeated_key_keeps_what_was_streamed_first() {
        for (text, streamed, parsed) in [
            (
                r#"<tool_call>{"name":"f","parameters":{"a":1},"arguments":{"b":2}}</tool_call>"#,
                ("f", serde_json::json!({"a": 1})),
                ("f", serde_json::json!({"b": 2})),
            ),
            (
                r#"<tool_call>{"name":"f","arguments":{"a":1},"name":"g"}</tool_call>"#,
                ("f", serde_json::json!({"a": 1})),
                ("g", serde_json::json!({"a": 1})),
            ),
        ] {
            let parts = pieces(text, 1);
            let calls = assemble(
                &run(
                    &parts.iter().map(String::as_str).collect::<Vec<_>>(),
                    false,
                    false,
                )
                .events,
            );
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!((calls[0].1.as_str(), json(&calls[0].2)), streamed, "{text}");
            let whole = crate::parse_tool_calls(text).unwrap();
            assert_eq!(
                (
                    whole[0].function.name.as_str(),
                    json(&whole[0].function.arguments)
                ),
                parsed,
                "{text}"
            );
        }
    }
}
