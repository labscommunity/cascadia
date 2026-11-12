//! OpenAI `response_format` parsing: turns the request field into a
//! [`ResponseFormat`], maps it to a backend-neutral [`GrammarSpec`], and
//! lists JSON Schema keywords the decoder cannot enforce.

use cascadia_grammar::GrammarFactory;
use cascadia_types::{GrammarKind, GrammarSpec};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Grammar body used for `json_object` (and for `json_schema` whose schema is
/// missing, `null`, `true` or `{}`): any JSON object.
pub const JSON_OBJECT_SCHEMA: &str = r#"{"type":"object","additionalProperties":true}"#;

/// The output format a chat request asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: Option<String>,
        /// The schema, parsed, object keys in the client's order. Use it for
        /// checks; the decoder gets `raw`.
        schema: Value,
        /// The schema as the client sent it, whitespace removed, with each
        /// `allOf` of plain object schemas merged into one object (the
        /// decoder does not enforce such an `allOf`; `schema` matches this
        /// form). Key order is kept: the decoder emits object properties in
        /// schema order.
        raw: String,
        strict: bool,
    },
}

impl ResponseFormat {
    /// True when the output must be JSON.
    pub fn is_json(&self) -> bool {
        !matches!(self, ResponseFormat::Text)
    }

    /// The OpenAI `type` string for this format.
    pub fn type_name(&self) -> &'static str {
        match self {
            ResponseFormat::Text => "text",
            ResponseFormat::JsonObject => "json_object",
            ResponseFormat::JsonSchema { .. } => "json_schema",
        }
    }
}

// Limits on a `json_schema` schema. Large repetition counts and deep nesting
// make the backend's grammar compile take seconds inside the request
// (measured on OpenVINO GenAI 2026.4.1); the limits sit far above any client
// schema seen. They are fast 400s for the shapes known to be slow, not the
// bound: [`SchemaCompileCheck`]'s time budget is the hard limit on how long
// a schema may take to build.

/// Largest accepted schema, in bytes of its text with whitespace removed.
pub const MAX_SCHEMA_BYTES: usize = 65536;
/// Deepest accepted nesting of objects and arrays; the schema itself is level 1.
pub const MAX_SCHEMA_DEPTH: usize = 64;
/// Largest accepted value of `minItems`, `maxItems`, `minLength`,
/// `maxLength`, `minProperties` and `maxProperties`, and of any number in a
/// `pattern` repetition quantifier such as `{n}` or `{n,m}`.
pub const MAX_SCHEMA_COUNT: u64 = 10_000;
/// Largest accepted sum of all those counts and quantifier numbers in one
/// schema, since many counts under the per-value limit still add up. A
/// `minX`/`maxX` pair on one object counts once, at the larger value. Sized
/// so an extraction schema with 21 text fields of `maxLength: 5000` passes.
pub const MAX_SCHEMA_COUNT_TOTAL: u64 = 200_000;
/// Most `patternProperties` keys on one object. Building the grammar grows
/// about with the cube of the key count (500 keys: 20 s; 100 keys: 0.1 s).
pub const MAX_PATTERN_PROPERTIES: usize = 64;
/// Most `patternProperties` keys in one schema, all objects together, and
/// counted again each time the grammar builder visits or merges an object
/// anew (each entry of a `type` list, each `allOf` that merges it).
pub const MAX_PATTERN_PROPERTIES_TOTAL: usize = 128;
/// Largest accepted sum of the lengths, in bytes, of every `pattern` and
/// `patternProperties` key in one schema.
pub const MAX_PATTERN_BYTES: usize = 16384;

/// Most schemas built at once by [`SchemaCompileCheck`].
pub const SCHEMA_COMPILE_PERMITS: usize = 2;
/// How long a request waits for a free [`SchemaCompileCheck`] slot before a 503.
pub const SCHEMA_COMPILE_WAIT: Duration = Duration::from_millis(500);
/// Longest a schema may take to build in [`SchemaCompileCheck`].
pub const SCHEMA_COMPILE_BUDGET: Duration = Duration::from_secs(2);

/// Why [`SchemaCompileCheck::check`] refused a schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemaCheckError {
    /// Every build slot stayed taken for the whole wait: a 503.
    Busy,
    /// The build ran past the budget, or the grammar it built fails on its
    /// first token: a 400.
    TooComplex,
}

/// Most passed schemas [`SchemaCompileCheck`] remembers.
pub const SCHEMA_PASS_CACHE: usize = 256;

/// The schemas that built within the budget, by a keyed hash of the schema
/// text, at most `cap` of them; the least recently used leaves first. Only
/// passes are kept: a build that ran long on a busy machine may pass when
/// tried again.
#[derive(Debug)]
struct PassedSchemas {
    cap: usize,
    hasher: std::hash::RandomState,
    tick: u64,
    entries: HashMap<u64, u64>,
}

impl PassedSchemas {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            hasher: std::hash::RandomState::new(),
            tick: 0,
            entries: HashMap::new(),
        }
    }

    fn key(&self, text: &str) -> u64 {
        use std::hash::BuildHasher;
        self.hasher.hash_one(text)
    }

    /// Whether `text` passed before; marks it used.
    fn contains(&mut self, text: &str) -> bool {
        let key = self.key(text);
        self.tick += 1;
        let tick = self.tick;
        self.entries
            .get_mut(&key)
            .map(|used| *used = tick)
            .is_some()
    }

    fn insert(&mut self, text: &str) {
        if self.cap == 0 {
            return;
        }
        let key = self.key(text);
        if self.entries.len() >= self.cap && !self.entries.contains_key(&key) {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, used)| **used)
                .map(|(k, _)| *k);
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
        self.tick += 1;
        self.entries.insert(key, self.tick);
    }
}

/// The hard bound on how long a schema's grammar takes to build. The caps
/// above are fast 400s for the shapes known to be slow; this is what holds
/// for everything else. Each enforced schema is built once with llguidance
/// over a tokenizer-free byte vocab (the build cost of a schema doesn't
/// depend on the tokenizer), on the blocking pool and never under the
/// engine lock. A build past the budget is a 400; the request leaves, and
/// the build keeps its slot until it ends, so abandoned builds hold at most
/// `permits` cores. Whether the schema builds at all is left to the engine,
/// whose backend may accept what llguidance doesn't. A grammar that builds
/// must also take its first token (see
/// [`cascadia_grammar::CompiledGrammar::check_first_step`]) within the same
/// budget, or it is a 400 too: it would fail every request mid-generation.
///
/// `json_object`'s constant grammar is never built here, and a schema that
/// passed is remembered ([`SCHEMA_PASS_CACHE`] of them, by schema text):
/// agents resend the same schema every step, and it never waits for a slot
/// again.
#[derive(Clone)]
pub struct SchemaCompileCheck {
    permits: Arc<Semaphore>,
    wait: Duration,
    budget: Duration,
    passed: Arc<std::sync::Mutex<PassedSchemas>>,
    factory: GrammarFactory,
}

impl std::fmt::Debug for SchemaCompileCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaCompileCheck")
            .field("permits", &self.permits)
            .field("wait", &self.wait)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl Default for SchemaCompileCheck {
    fn default() -> Self {
        Self::new(
            SCHEMA_COMPILE_PERMITS,
            SCHEMA_COMPILE_WAIT,
            SCHEMA_COMPILE_BUDGET,
        )
    }
}

impl SchemaCompileCheck {
    /// At most `permits` builds at once; a request waits up to `wait` for
    /// one; a build may take up to `budget`.
    pub fn new(permits: usize, wait: Duration, budget: Duration) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(permits)),
            wait,
            budget,
            passed: Arc::new(std::sync::Mutex::new(PassedSchemas::new(SCHEMA_PASS_CACHE))),
            factory: GrammarFactory::byte_level(),
        }
    }

    /// The same check over `factory` in place of the byte-level one.
    #[cfg(test)]
    pub(crate) fn with_factory(self, factory: GrammarFactory) -> Self {
        Self { factory, ..self }
    }

    /// Builds `spec` within the budget, or says why not.
    pub async fn check(&self, spec: &GrammarSpec) -> Result<(), SchemaCheckError> {
        if spec.body == JSON_OBJECT_SCHEMA {
            return Ok(());
        }
        let passed = self
            .passed
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains(&spec.body);
        if passed {
            return Ok(());
        }
        let permit =
            match tokio::time::timeout(self.wait, self.permits.clone().acquire_owned()).await {
                Ok(Ok(permit)) => permit,
                _ => return Err(SchemaCheckError::Busy),
            };
        let spec = spec.clone();
        let body = spec.body.clone();
        let factory = self.factory.clone();
        // Set once the request stops waiting: the first-step check then ends
        // before its next mask and frees the slot (the build itself can't be
        // stopped).
        let abandoned = Arc::new(AtomicBool::new(false));
        let stop = abandoned.clone();
        let build = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            // A build error is the engine's to report (its backend may build
            // what llguidance doesn't): only the time counts for it here. A
            // grammar that builds but fails on its first token would fail
            // every request mid-generation instead.
            match factory.compile(&spec) {
                Ok(grammar) => grammar.check_first_step(&stop).map_err(|e| e.to_string()),
                Err(_) => Ok(()),
            }
        });
        let joined = tokio::time::timeout(self.budget, build).await;
        if joined.is_err() {
            abandoned.store(true, Ordering::Relaxed);
        }
        let verdict = build_verdict(joined);
        if verdict.is_ok() {
            self.passed
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(&body);
        }
        verdict
    }
}

/// The check's verdict on one timed build: a pass only when the build ran to
/// its end within the budget and the grammar took its first step. A build
/// that panicked proved nothing about the schema, so it is refused like one
/// that ran too long, and never remembered.
fn build_verdict(
    joined: Result<Result<Result<(), String>, tokio::task::JoinError>, tokio::time::error::Elapsed>,
) -> Result<(), SchemaCheckError> {
    match joined {
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(Ok(Err(e))) => {
            tracing::warn!(error = %e, "response_format schema check: the grammar fails on its first token");
            Err(SchemaCheckError::TooComplex)
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "response_format schema check: the grammar build failed");
            Err(SchemaCheckError::TooComplex)
        }
        Err(_) => Err(SchemaCheckError::TooComplex),
    }
}

const TYPE_ERR: &str = "response_format.type must be one of text, json_object, json_schema";

/// Parse the `response_format` field of a chat request from its JSON text.
/// Absent or `null` means plain text. `raw` is the field exactly as the
/// client sent it; the schema keeps the client's key order, both parsed and
/// as text (see [`ResponseFormat::JsonSchema`]).
pub fn parse_response_format(raw: Option<&str>) -> Result<ResponseFormat, String> {
    let Some(raw) = raw else {
        return Ok(ResponseFormat::Text);
    };
    let v: Value =
        serde_json::from_str(raw).map_err(|e| format!("response_format is not valid JSON: {e}"))?;
    let obj = match &v {
        Value::Null => return Ok(ResponseFormat::Text),
        Value::Object(o) => o,
        _ => return Err("response_format must be an object".into()),
    };
    match obj.get("type").and_then(Value::as_str) {
        None => Err(TYPE_ERR.into()),
        Some("text") => Ok(ResponseFormat::Text),
        Some("json_object") => Ok(ResponseFormat::JsonObject),
        Some("json_schema") => parse_json_schema(obj.get("json_schema"), raw),
        Some(other) => Err(format!("{TYPE_ERR}, got {}", Value::from(other))),
    }
}

/// `response_format` with the schema left as text.
#[derive(Deserialize)]
struct RawFormat {
    json_schema: Box<RawValue>,
}

/// `response_format.json_schema` with the schema left as text.
#[derive(Deserialize)]
struct RawJsonSchema {
    schema: Box<RawValue>,
}

/// The text of `response_format.json_schema.schema` inside `raw`, which
/// holds one.
fn raw_schema_text(raw: &str) -> Result<String, String> {
    let bad = |e: serde_json::Error| format!("response_format.json_schema: {e}");
    let format: RawFormat = serde_json::from_str(raw).map_err(bad)?;
    let js: RawJsonSchema = serde_json::from_str(format.json_schema.get()).map_err(bad)?;
    Ok(minify_json(js.schema.get()))
}

/// `text`, valid JSON, with the whitespace between tokens removed. Strings
/// and key order stay as they are.
fn minify_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let (mut in_string, mut escaped) = (false, false);
    for c in text.chars() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
            out.push(c);
        } else if !matches!(c, ' ' | '\t' | '\n' | '\r') {
            out.push(c);
        }
    }
    out
}

/// Parse `response_format.json_schema`. A missing, `null`, `true` or `{}`
/// schema asks for any JSON object; `false` and non-object schemas are
/// rejected. `raw` is the whole `response_format` text.
fn parse_json_schema(js: Option<&Value>, raw: &str) -> Result<ResponseFormat, String> {
    let Some(Value::Object(js)) = js else {
        return Err("response_format.json_schema must be an object".into());
    };
    match js.get("schema") {
        None | Some(Value::Null) | Some(Value::Bool(true)) => Ok(ResponseFormat::JsonObject),
        Some(Value::Object(o)) if o.is_empty() => Ok(ResponseFormat::JsonObject),
        Some(Value::Bool(false)) => {
            Err("response_format.json_schema.schema is false: no output can match it".into())
        }
        Some(schema @ Value::Object(_)) => {
            let text = raw_schema_text(raw)?;
            check_limits(schema, &text)?;
            check_bounds(schema)?;
            let (schema, text) = match merge_object_all_of(&text) {
                Some(merged) => (
                    serde_json::from_str(&merged).map_err(|e| format!("{FIELD}: {e}"))?,
                    merged,
                ),
                None => (schema.clone(), text),
            };
            let raw: Box<RawValue> =
                serde_json::from_str(&text).map_err(|e| format!("{FIELD}: {e}"))?;
            check_converter_depth(&Node::parse(raw).map_err(|e| format!("{FIELD}: {e}"))?)?;
            check_ref_depth_bound(&schema)?;
            // A ref to a document outside the schema: no backend fetches
            // one, refused for every engine.
            if let Some(why) = cascadia_grammar::ref_outside_the_schema(&schema) {
                return Err(format!("{FIELD}: {why}"));
            }
            Ok(ResponseFormat::JsonSchema {
                name: js.get("name").and_then(Value::as_str).map(str::to_owned),
                schema,
                raw: text,
                strict: js.get("strict") == Some(&Value::Bool(true)),
            })
        }
        Some(_) => Err("response_format.json_schema.schema must be a JSON Schema object".into()),
    }
}

/// Keywords whose value is a repetition count, capped at [`MAX_SCHEMA_COUNT`],
/// as `[min, max]` pairs. Toward [`MAX_SCHEMA_COUNT_TOTAL`] each pair on one
/// object counts once, at its larger value: the bounds describe one range.
const COUNT_KEYWORDS: [[&str; 2]; 3] = [
    ["minItems", "maxItems"],
    ["minLength", "maxLength"],
    ["minProperties", "maxProperties"],
];

const FIELD: &str = "response_format.json_schema.schema";

/// Reject a schema over [`MAX_SCHEMA_BYTES`], with a repeated key in any
/// object, nested deeper than [`MAX_SCHEMA_DEPTH`], with more
/// `patternProperties` keys than [`MAX_PATTERN_PROPERTIES`] on one object or
/// [`MAX_PATTERN_PROPERTIES_TOTAL`] in all, with a count keyword or
/// pattern quantifier above [`MAX_SCHEMA_COUNT`], or whose counts sum above
/// [`MAX_SCHEMA_COUNT_TOTAL`]. `text` is the schema's compact text. Depth
/// through `$ref` is checked later, on the text the backend receives
/// ([`check_converter_depth`]).
fn check_limits(schema: &Value, text: &str) -> Result<(), String> {
    let n = text.len();
    if n > MAX_SCHEMA_BYTES {
        return Err(format!(
            "{FIELD} is {n} bytes; the limit is {MAX_SCHEMA_BYTES}"
        ));
    }
    reject_duplicate_keys(text)?;
    if nests_deeper_than(schema, MAX_SCHEMA_DEPTH) {
        return Err(format!(
            "{FIELD} nests deeper than {MAX_SCHEMA_DEPTH} levels"
        ));
    }
    let (mut keys, mut bytes) = (0, 0);
    count_patterns(schema, &mut keys, &mut bytes)?;
    if keys > MAX_PATTERN_PROPERTIES_TOTAL {
        return Err(format!(
            "{FIELD} has {keys} patternProperties keys; the limit is {MAX_PATTERN_PROPERTIES_TOTAL}"
        ));
    }
    if bytes > MAX_PATTERN_BYTES {
        return Err(format!(
            "{FIELD}: its patterns add up to {bytes} bytes; the limit is {MAX_PATTERN_BYTES}"
        ));
    }
    let mut total = 0;
    sum_counts(schema, &mut total)?;
    if total > MAX_SCHEMA_COUNT_TOTAL {
        return Err(format!(
            "{FIELD}: repetition counts add up to {total}; the limit is {MAX_SCHEMA_COUNT_TOTAL}"
        ));
    }
    Ok(())
}

/// Bound pairs no value can satisfy when `lower` is above `upper`
/// (`strict`: not below it), as `(lower, upper, strict)`.
const BOUND_PAIRS: [(&str, &str, bool); 7] = [
    ("minLength", "maxLength", false),
    ("minItems", "maxItems", false),
    ("minProperties", "maxProperties", false),
    ("minimum", "maximum", false),
    ("exclusiveMinimum", "exclusiveMaximum", true),
    ("minimum", "exclusiveMaximum", true),
    ("exclusiveMinimum", "maximum", true),
];

/// The first bound pair in `schema` that no value can satisfy, such as
/// `minLength` above `maxLength` or `exclusiveMinimum` not below
/// `exclusiveMaximum`, described with the JSON pointer (`#`, `#/items`, ...)
/// of the object holding it; `None` when there is none.
///
/// No value meets such a pair, and a grammar backend given it may build a
/// grammar no output can finish (llama-server then decodes until the
/// request is killed), so [`parse_response_format`] refuses it up front,
/// wherever it sits: a pair no value meets is a mistake in the schema even
/// in a branch a value need not take. This only reports, for callers that
/// would rather drop the constraint.
///
/// It visits every schema in the document, `$defs` included, and skips
/// `enum`, `const`, `default` and `examples`, whose values are data: a
/// bound-named key in a default object is not a bound. Whatever a local
/// `$ref` points to is a schema and is checked, inside a data keyword too. A property map's
/// entries are schemas whatever they are called. Only numeric values are
/// compared, so a property map whose names are keywords is never read as
/// bounds.
/// For a schema whose every `type` is `integer`, a range holding no integer
/// (`exclusiveMinimum` 2 with `exclusiveMaximum` 3) is reported too.
/// `minContains`/`maxContains` are left alone: no backend enforces
/// `contains`, so they never empty the grammar. The boolean (draft 4) form
/// of `exclusiveMinimum`/`exclusiveMaximum` is not a number and is skipped.
pub fn unsatisfiable_bound(schema: &Value) -> Option<String> {
    let mut path = Vec::new();
    find_unsatisfiable_bound(schema, &mut path, 0).or_else(|| {
        // A local `$ref` makes whatever it points to a schema, inside a data
        // keyword too, which the walk above skips. Each such target gets a
        // walk of its own, once, and a target inside another one only when
        // that one's walk skips it: refs to every level of one subtree cost
        // one walk of it.
        uncovered_ref_targets(schema)
            .into_iter()
            .find_map(|(target, mut path)| find_unsatisfiable_bound(target, &mut path, 0))
    })
}

/// The targets of the local `$ref`s in `root` that no walk from the root
/// visits (it skips data keywords), outermost first, each with the decoded
/// path of a ref naming it. A target is left out when the walk from the root
/// or from another target returned here visits it, the way
/// [`find_unsatisfiable_bound`] walks: data keywords skipped, the entries of a
/// map of named schemas walked whatever their names.
fn uncovered_ref_targets(root: &Value) -> Vec<(&Value, Vec<String>)> {
    let mut targets: HashMap<*const Value, Vec<String>> = HashMap::new();
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(o) => {
                if let Some(r) = o.get("$ref").and_then(Value::as_str) {
                    for target in backstop_targets(root, r) {
                        targets
                            .entry(target as *const Value)
                            .or_insert_with(|| decode_pointer(r));
                    }
                }
                stack.extend(o.values());
            }
            Value::Array(a) => stack.extend(a.iter()),
            _ => {}
        }
    }
    let mut uncovered = Vec::new();
    // Each value with whether a walk visits it as a schema, and whether it is
    // a map of named schemas.
    let mut stack = vec![(root, true, false)];
    while let Some((v, mut walked, schema_map)) = stack.pop() {
        if targets.is_empty() {
            break;
        }
        if !walked {
            if let Some(path) = targets.remove(&(v as *const Value)) {
                uncovered.push((v, path));
                walked = true;
            }
        }
        match v {
            Value::Object(o) => stack.extend(o.iter().map(|(key, child)| {
                if schema_map {
                    (child, walked, false)
                } else if DATA_KEYS.contains(&key.as_str()) {
                    (child, false, false)
                } else {
                    let map = SCHEMA_MAPS.contains(&key.as_str()) && child.is_object();
                    (child, walked, map)
                }
            })),
            Value::Array(a) => stack.extend(a.iter().map(|child| (child, walked, false))),
            _ => {}
        }
    }
    uncovered
}

/// The keys of a local `$ref` such as `#/$defs/a~1b`, decoded: `["$defs", "a/b"]`.
fn decode_pointer(r: &str) -> Vec<String> {
    r.trim_start_matches('#')
        .split('/')
        .filter(|k| !k.is_empty())
        .map(|k| k.replace("~1", "/").replace("~0", "~"))
        .collect()
}

fn find_unsatisfiable_bound(v: &Value, path: &mut Vec<String>, depth: usize) -> Option<String> {
    // Past the nesting cap [`check_limits`] enforces; a bare call may get here.
    if depth > MAX_SCHEMA_DEPTH {
        return None;
    }
    let descend = |path: &mut Vec<String>, key: String, child: &Value| {
        path.push(key);
        let found = find_unsatisfiable_bound(child, path, depth + 1);
        path.pop();
        found
    };
    match v {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .find_map(|(i, c)| descend(path, i.to_string(), c)),
        Value::Object(obj) => {
            if let Some(found) = bound_violation(obj) {
                return Some(locate(found, path));
            }
            for (key, child) in obj {
                if DATA_KEYS.contains(&key.as_str()) {
                    continue;
                }
                let found = match child {
                    // A map of named schemas: each entry is a schema, a
                    // `default` property too.
                    Value::Object(map) if SCHEMA_MAPS.contains(&key.as_str()) => {
                        path.push(key.clone());
                        let found = map
                            .iter()
                            .find_map(|(name, sub)| descend(path, name.clone(), sub));
                        path.pop();
                        found
                    }
                    _ => descend(path, key.clone(), child),
                };
                if found.is_some() {
                    return found;
                }
            }
            // The members of an `allOf` all apply to the same value: bounds
            // from different members (and the holder) can leave nothing.
            if let Some(Value::Array(_)) = obj.get("allOf") {
                let (mut arena, mut sources) = (Vec::new(), Vec::new());
                add_source(&mut arena, &mut sources, obj, None, Step::Holder, depth);
                return all_of_violation(&mut arena, &sources, path, depth);
            }
            None
        }
        _ => None,
    }
}

/// A bound pair in `obj` no value meets, or an integer-only range with no
/// integer: its description and whether it is a pair.
fn bound_violation(obj: &Map<String, Value>) -> Option<(String, bool)> {
    Some(match violated_pair(obj)? {
        (lower, upper, Some(strict)) => (
            format!(
                "{lower} ({}) {} {upper} ({})",
                obj[lower],
                relation(strict),
                obj[upper]
            ),
            true,
        ),
        (lower, upper, None) => (
            format!(
                "no integer meets {lower} ({}) and {upper} ({})",
                obj[lower], obj[upper]
            ),
            false,
        ),
    })
}

fn relation(strict: bool) -> &'static str {
    if strict {
        "is not less than"
    } else {
        "is greater than"
    }
}

/// The lower and upper keywords of a bound pair in `obj` no value meets, with
/// `Some(strict)`, or of an integer-only range with no integer, with `None`.
fn violated_pair(obj: &Map<String, Value>) -> Option<(&'static str, &'static str, Option<bool>)> {
    for (lower, upper, strict) in BOUND_PAIRS {
        let (Some(Value::Number(lo)), Some(Value::Number(hi))) = (obj.get(lower), obj.get(upper))
        else {
            continue;
        };
        let Some(order) = compare_numbers(lo, hi) else {
            continue;
        };
        let empty = if strict { order.is_ge() } else { order.is_gt() };
        if empty {
            return Some((lower, upper, Some(strict)));
        }
    }
    let (lower, upper) = empty_integer_range(obj)?;
    Some((lower, upper, None))
}

/// The 400 detail for `found` at `path`.
fn locate((detail, pair): (String, bool), path: &[String]) -> String {
    let at = json_pointer(path);
    if pair {
        format!("{detail} at {at}; no value can satisfy both")
    } else {
        format!("{detail} at {at}")
    }
}

/// One schema of those that apply to the value an `allOf` holder describes:
/// the holder, a member, a member's member, and so on; or, one level down, a
/// property or `items` schema several of those define. `parent` and `step`
/// give its pointer.
struct Source<'a> {
    map: &'a Map<String, Value>,
    parent: Option<usize>,
    step: Step<'a>,
}

#[derive(Clone, Copy)]
enum Step<'a> {
    Holder,
    Member(usize),
    Property(&'a str),
    Items,
}

/// Add `map` to `arena` and its index to `out`, then the members of its
/// `allOf`, and theirs. A `$ref` schema is left out with its members: draft 7
/// ignores what sits beside a `$ref`, and the target is not followed here.
fn add_source<'a>(
    arena: &mut Vec<Source<'a>>,
    out: &mut Vec<usize>,
    map: &'a Map<String, Value>,
    parent: Option<usize>,
    step: Step<'a>,
    depth: usize,
) {
    if depth > MAX_SCHEMA_DEPTH || map.contains_key("$ref") {
        return;
    }
    let index = arena.len();
    arena.push(Source { map, parent, step });
    out.push(index);
    if let Some(Value::Array(members)) = map.get("allOf") {
        for (i, member) in members.iter().enumerate() {
            if let Some(member) = member.as_object() {
                add_source(arena, out, member, Some(index), Step::Member(i), depth + 1);
            }
        }
    }
}

/// Schemas several sources define for one value (a property, or `items`),
/// each with the index of the source defining it.
type Defs<'a> = Vec<(usize, &'a Map<String, Value>)>;

/// The pointer of `arena[index]`, under `base` (the holder's).
fn source_path(arena: &[Source], mut index: usize, base: &[String]) -> Vec<String> {
    let mut keys = Vec::new();
    loop {
        let source = &arena[index];
        match source.step {
            Step::Holder => {}
            Step::Member(i) => keys.extend([i.to_string(), "allOf".into()]),
            Step::Property(name) => keys.extend([name.to_string(), "properties".into()]),
            Step::Items => keys.push("items".into()),
        }
        match source.parent {
            Some(parent) => index = parent,
            None => break,
        }
    }
    base.iter().cloned().chain(keys.into_iter().rev()).collect()
}

/// A bound pair no value meets across `sources`, schemas that all apply to
/// one value, each bound named with the pointer of the schema holding it;
/// then the same for each property, and for `items`, that more than one of
/// them defines. One pass over their property maps groups the definitions,
/// so the cost stays linear in the schema.
fn all_of_violation<'a>(
    arena: &mut Vec<Source<'a>>,
    sources: &[usize],
    base: &[String],
    depth: usize,
) -> Option<String> {
    if depth > MAX_SCHEMA_DEPTH {
        return None;
    }
    if let Some(found) = merged_violation(arena, sources, base) {
        return Some(found);
    }
    let mut names: Vec<&'a str> = Vec::new();
    let mut defs: HashMap<&'a str, Defs<'a>> = HashMap::new();
    let mut items = Vec::new();
    for &index in sources {
        let map = arena[index].map;
        if let Some(Value::Object(props)) = map.get("properties") {
            for (name, def) in props {
                if let Some(def) = def.as_object() {
                    let entry = defs.entry(name.as_str()).or_default();
                    if entry.is_empty() {
                        names.push(name.as_str());
                    }
                    entry.push((index, def));
                }
            }
        }
        if let Some(Value::Object(def)) = map.get("items") {
            items.push((index, def));
        }
    }
    let mut groups: Vec<(Step<'a>, Defs<'a>)> = names
        .into_iter()
        .filter_map(|name| {
            let group = defs.remove(name)?;
            (group.len() > 1).then_some((Step::Property(name), group))
        })
        .collect();
    if items.len() > 1 {
        groups.push((Step::Items, items));
    }
    for (step, group) in groups {
        let mut next = Vec::new();
        for (parent, def) in group {
            add_source(arena, &mut next, def, Some(parent), step, depth + 1);
        }
        if let Some(found) = all_of_violation(arena, &next, base, depth + 1) {
            return Some(found);
        }
    }
    None
}

/// The bounds `sources` put on one value together (the tightest lower and
/// upper of each keyword, and `type: "integer"` when one of them allows only
/// integers), checked like one schema's.
fn merged_violation(arena: &[Source], sources: &[usize], base: &[String]) -> Option<String> {
    const LOWER: [&str; 5] = [
        "minLength",
        "minItems",
        "minProperties",
        "minimum",
        "exclusiveMinimum",
    ];
    const UPPER: [&str; 5] = [
        "maxLength",
        "maxItems",
        "maxProperties",
        "maximum",
        "exclusiveMaximum",
    ];
    let mut merged = Map::new();
    let mut origin: HashMap<&str, usize> = HashMap::new();
    for (keys, keep) in [
        (LOWER, std::cmp::Ordering::Greater),
        (UPPER, std::cmp::Ordering::Less),
    ] {
        for key in keys {
            for &index in sources {
                let Some(Value::Number(n)) = arena[index].map.get(key) else {
                    continue;
                };
                let tighter = match merged.get(key) {
                    Some(Value::Number(m)) => compare_numbers(n, m) == Some(keep),
                    _ => true,
                };
                if tighter {
                    merged.insert(key.to_string(), Value::Number(n.clone()));
                    origin.insert(key, index);
                }
            }
        }
    }
    let integer = |t: &Value| t.as_str() == Some("integer");
    let only_integer = |m: &Map<String, Value>| match m.get("type") {
        Some(Value::Array(types)) => !types.is_empty() && types.iter().all(integer),
        Some(t) => integer(t),
        None => false,
    };
    if sources.iter().any(|&index| only_integer(arena[index].map)) {
        merged.insert("type".into(), Value::String("integer".into()));
    }
    let (lower, upper, pair) = violated_pair(&merged)?;
    let at = |key: &str| json_pointer(&source_path(arena, origin[key], base));
    let (lo, hi) = (&merged[lower], &merged[upper]);
    let (lo_at, hi_at) = (at(lower), at(upper));
    Some(match pair {
        Some(strict) => format!(
            "{lower} ({lo}) at {lo_at} {} {upper} ({hi}) at {hi_at}; no value can satisfy both",
            relation(strict)
        ),
        None => format!("no integer meets {lower} ({lo}) at {lo_at} and {upper} ({hi}) at {hi_at}"),
    })
}

/// The RFC 6901 pointer of `path`, as a URI fragment: `#`, `#/items`, ...
fn json_pointer(path: &[String]) -> String {
    let mut pointer = String::from("#");
    for key in path {
        pointer.push('/');
        pointer.push_str(&key.replace('~', "~0").replace('/', "~1"));
    }
    pointer
}

/// For a schema whose every `type` is `integer`, the lower and upper bound
/// keywords that leave no integer between them, or `None`. The smallest
/// integer allowed is the largest of `ceil(minimum)` and
/// `floor(exclusiveMinimum) + 1`, the largest the smallest of
/// `floor(maximum)` and `ceil(exclusiveMaximum) - 1`; the keywords named
/// are the ones that set them. A pair that is empty for any number is
/// already reported by [`BOUND_PAIRS`]; this catches ranges like
/// `(2, 3)` or `[2.5, 2.9]` that hold numbers but no integer.
fn empty_integer_range(obj: &Map<String, Value>) -> Option<(&'static str, &'static str)> {
    let integer = |t: &Value| t.as_str() == Some("integer");
    let only_integer = match obj.get("type")? {
        Value::Array(types) => !types.is_empty() && types.iter().all(integer),
        t => integer(t),
    };
    if !only_integer {
        return None;
    }
    // An integer bound as is; a fraction rounded the given way. A float
    // past what an i128 holds is no bound here (a cast would saturate and
    // make a wide range look empty).
    let bound = |key: &str, round: fn(f64) -> f64| -> Option<i128> {
        let n = obj.get(key)?.as_number()?;
        match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Some(i128::from(i)),
            (_, Some(u)) => Some(i128::from(u)),
            _ => {
                let r = round(n.as_f64()?);
                (r.is_finite() && r.abs() < 1e37).then_some(r as i128)
            }
        }
    };
    let lowers = [
        ("minimum", bound("minimum", f64::ceil)),
        (
            "exclusiveMinimum",
            bound("exclusiveMinimum", f64::floor).map(|b| b.saturating_add(1)),
        ),
    ];
    let uppers = [
        ("maximum", bound("maximum", f64::floor)),
        (
            "exclusiveMaximum",
            bound("exclusiveMaximum", f64::ceil).map(|b| b.saturating_sub(1)),
        ),
    ];
    let (lower, lo) = lowers
        .into_iter()
        .filter_map(|(k, b)| Some((k, b?)))
        .max_by_key(|(_, b)| *b)?;
    let (upper, hi) = uppers
        .into_iter()
        .filter_map(|(k, b)| Some((k, b?)))
        .min_by_key(|(_, b)| *b)?;
    (lo > hi).then_some((lower, upper))
}

/// `a` against `b`: exactly when both are integers, else as doubles.
fn compare_numbers(a: &serde_json::Number, b: &serde_json::Number) -> Option<std::cmp::Ordering> {
    let int = |n: &serde_json::Number| {
        n.as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
    };
    match (int(a), int(b)) {
        (Some(a), Some(b)) => Some(a.cmp(&b)),
        _ => a.as_f64()?.partial_cmp(&b.as_f64()?),
    }
}

/// Reject a schema holding a bound pair no value can satisfy
/// ([`unsatisfiable_bound`]).
fn check_bounds(schema: &Value) -> Result<(), String> {
    match unsatisfiable_bound(schema) {
        Some(found) => Err(format!("{FIELD}: {found}")),
        None => Ok(()),
    }
}

/// Add the number of `patternProperties` keys of every object in `v` to
/// `keys`, and the length of each of those keys and of every string
/// `pattern` to `bytes` (data objects too, as [`sum_counts`]), failing on
/// an object with more than [`MAX_PATTERN_PROPERTIES`] keys.
fn count_patterns(v: &Value, keys: &mut usize, bytes: &mut usize) -> Result<(), String> {
    match v {
        Value::Array(items) => items
            .iter()
            .try_for_each(|c| count_patterns(c, keys, bytes)),
        Value::Object(obj) => {
            if let Some(Value::Object(map)) = obj.get("patternProperties") {
                let n = map.len();
                if n > MAX_PATTERN_PROPERTIES {
                    return Err(format!(
                        "{FIELD}: an object has {n} patternProperties keys; \
                         the limit is {MAX_PATTERN_PROPERTIES}"
                    ));
                }
                *keys += n;
                *bytes += map.keys().map(String::len).sum::<usize>();
            }
            if let Some(Value::String(p)) = obj.get("pattern") {
                *bytes += p.len();
            }
            obj.values()
                .try_for_each(|c| count_patterns(c, keys, bytes))
        }
        _ => Ok(()),
    }
}

/// Add every count keyword value and pattern quantifier number in `v` to
/// `total`, failing on the first one above [`MAX_SCHEMA_COUNT`]. This visits
/// every object in the document, data keys included: a `$ref` may point
/// anywhere, so any object may end up compiled as a schema. Counts inside
/// data are therefore charged too.
fn sum_counts(v: &Value, total: &mut u64) -> Result<(), String> {
    match v {
        Value::Array(items) => items.iter().try_for_each(|c| sum_counts(c, total)),
        Value::Object(obj) => {
            for pair in COUNT_KEYWORDS {
                let mut largest = 0;
                for key in pair {
                    let Some(value) = obj.get(key) else { continue };
                    if let Some(c) = value.as_f64().filter(|c| *c >= 0.0) {
                        if c > MAX_SCHEMA_COUNT as f64 {
                            return Err(format!(
                                "{FIELD}: {key} = {value} exceeds {MAX_SCHEMA_COUNT}"
                            ));
                        }
                        largest = largest.max(c as u64);
                    }
                }
                *total += largest;
            }
            for key in INTEGER_KEYWORDS {
                if obj.get(key).is_some_and(|v| !is_plain_count(v)) {
                    return Err(format!("{FIELD}: {key} must be a non-negative integer"));
                }
            }
            for (key, value) in obj {
                let key = key.as_str();
                match value {
                    Value::String(re) if key == "pattern" => sum_quantifiers(re, total)?,
                    Value::Object(map) if key == "patternProperties" => {
                        for re in map.keys() {
                            sum_quantifiers(re, total)?;
                        }
                    }
                    _ => {}
                }
                sum_counts(value, total)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Keywords JSON Schema defines as non-negative integers: the count
/// keywords and `minContains`/`maxContains`.
const INTEGER_KEYWORDS: [&str; 8] = [
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minProperties",
    "maxProperties",
    "minContains",
    "maxContains",
];

/// True when `v` may stand where a count keyword is: a number written as a
/// plain non-negative integer (`0`, `10`; not `0.0`, `0e0`, `-0` or `1.5`,
/// which the converter reads as a double), or an object, array or boolean,
/// which is a property map entry (a property named `maxItems`) or data,
/// not a count. serde_json keeps an integer literal as an integer and reads
/// any number with a fraction, an exponent or a minus sign on zero as a
/// float, so `is_u64` holds exactly for the plain literals.
fn is_plain_count(v: &Value) -> bool {
    match v {
        Value::Number(n) => n.is_u64(),
        Value::Object(_) | Value::Array(_) | Value::Bool(_) => true,
        Value::String(_) | Value::Null => false,
    }
}

/// Add the numbers of every `{n}`, `{n,}`, `{n,m}` and `{,m}` quantifier in the
/// regex `re` to `total`, failing on one above [`MAX_SCHEMA_COUNT`]. Escaped
/// characters such as `\{` are skipped. Braces inside a character class are
/// read as quantifiers too, and nested quantifiers (`(a{100}){100}`) are
/// summed, not multiplied: both are out of scope for this check.
fn sum_quantifiers(re: &str, total: &mut u64) -> Result<(), String> {
    let b = re.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'{' => {
                let (numbers, end) = quantifier_at(b, i);
                for digits in numbers {
                    let n = digits.parse::<u64>().unwrap_or(u64::MAX);
                    if n > MAX_SCHEMA_COUNT {
                        return Err(format!(
                            "{FIELD}: pattern repetition {{{digits}}} exceeds {MAX_SCHEMA_COUNT}"
                        ));
                    }
                    *total += n;
                }
                i = end;
            }
            _ => i += 1,
        }
    }
    Ok(())
}

/// The numbers of the quantifier opening at `b[start] == b'{'` (none when
/// it is not one: digits, optional `,` and digits, then `}`; or `,`, digits,
/// then `}`), and the index to resume scanning at.
fn quantifier_at(b: &[u8], start: usize) -> (Vec<&str>, usize) {
    let digits_from = |from: usize| {
        let len = b[from..].iter().take_while(|c| c.is_ascii_digit()).count();
        (
            from + len,
            std::str::from_utf8(&b[from..from + len]).unwrap_or(""),
        )
    };
    let not_one = (Vec::new(), start + 1);
    let (mut i, n) = digits_from(start + 1);
    let mut numbers = Vec::new();
    if !n.is_empty() {
        numbers.push(n);
    }
    if b.get(i) == Some(&b',') {
        let (j, m) = digits_from(i + 1);
        if !m.is_empty() {
            numbers.push(m);
        }
        i = j;
    }
    // `{}` and `{,}` are not quantifiers.
    if numbers.is_empty() {
        return not_one;
    }
    if b.get(i) != Some(&b'}') {
        return not_one;
    }
    (numbers, i + 1)
}

/// Deepest nesting [`reject_duplicate_keys`] walks before giving up.
const DUPLICATE_CHECK_DEPTH: usize = 128;

/// Reject a schema text in which any object repeats a key. The limits are
/// checked on a parsed `Value`, which keeps the last of repeated keys, while
/// the backend compiles the text and may keep the first: without this check
/// `{"maxLength":1000000,"maxLength":5}` would pass the limits and compile
/// the large one.
fn reject_duplicate_keys(text: &str) -> Result<(), String> {
    use serde::de::{DeserializeSeed, Deserializer, Error as _, MapAccess, SeqAccess, Visitor};
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::fmt;

    /// Walks one JSON value; records the first repeated key in `found`.
    #[derive(Clone, Copy)]
    struct Walk<'a> {
        depth: usize,
        found: &'a RefCell<Option<String>>,
    }

    impl<'de> DeserializeSeed<'de> for Walk<'_> {
        type Value = ();
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
            if self.depth > DUPLICATE_CHECK_DEPTH {
                return Err(D::Error::custom("nested too deep"));
            }
            d.deserialize_any(self)
        }
    }

    impl<'de> Visitor<'de> for Walk<'_> {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a JSON value")
        }
        fn visit_bool<E>(self, _: bool) -> Result<(), E> {
            Ok(())
        }
        fn visit_i64<E>(self, _: i64) -> Result<(), E> {
            Ok(())
        }
        fn visit_u64<E>(self, _: u64) -> Result<(), E> {
            Ok(())
        }
        fn visit_f64<E>(self, _: f64) -> Result<(), E> {
            Ok(())
        }
        fn visit_str<E>(self, _: &str) -> Result<(), E> {
            Ok(())
        }
        fn visit_unit<E>(self) -> Result<(), E> {
            Ok(())
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
            let inner = Walk {
                depth: self.depth + 1,
                ..self
            };
            while seq.next_element_seed(inner)?.is_some() {}
            Ok(())
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            let inner = Walk {
                depth: self.depth + 1,
                ..self
            };
            let mut keys = HashSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if keys.contains(&key) {
                    *self.found.borrow_mut() = Some(key);
                    return Err(A::Error::custom("duplicate key"));
                }
                map.next_value_seed(inner)?;
                keys.insert(key);
            }
            Ok(())
        }
    }

    let found = RefCell::new(None);
    let mut de = serde_json::Deserializer::from_str(text);
    let walked = Walk {
        depth: 1,
        found: &found,
    }
    .deserialize(&mut de);
    if let Some(key) = found.into_inner() {
        return Err(format!(
            "{FIELD} has a duplicate key {}",
            Value::String(key)
        ));
    }
    walked.map_err(|e| format!("{FIELD}: {e}"))
}

/// Most work [`check_converter_depth`] does before it calls the schema too
/// complex: parsing the schema's scalars costs their length, each visit one
/// plus the visited object's key count, and resolving a `$ref` its length
/// plus the keys scanned, so the budget bounds the walk's time. A real
/// schema costs about its size.
const MAX_SCHEMA_VISITS: usize = 1_000_000;

/// A scalar of the schema as [`check_converter_depth`] reads it.
#[derive(Clone, Copy, PartialEq)]
enum Scalar {
    True,
    False,
    /// A string, by its index in [`Scalars::strings`].
    Str(usize),
    /// Anything else, or not a scalar.
    Other,
}

/// Every scalar of a schema, parsed once before the walk, its strings
/// interned: a visit reads `type`, `$ref` or `additionalProperties` in
/// constant time whatever the value's size.
struct Scalars {
    by_node: HashMap<*const Node, Scalar>,
    strings: Vec<String>,
    /// The work of parsing them: their total text length.
    cost: usize,
}

impl Scalars {
    fn new(root: &Node) -> Scalars {
        let mut ids: HashMap<String, usize> = HashMap::new();
        let mut out = Scalars {
            by_node: HashMap::new(),
            strings: Vec::new(),
            cost: 0,
        };
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let raw = match node {
                Node::Obj(entries) => {
                    stack.extend(entries.iter().map(|(_, v)| v));
                    continue;
                }
                Node::Arr(items) => {
                    stack.extend(items.iter());
                    continue;
                }
                Node::Scalar(raw) => raw,
            };
            out.cost += raw.get().len();
            let parsed = match serde_json::from_str(raw.get()) {
                Ok(Value::Bool(true)) => Scalar::True,
                Ok(Value::Bool(false)) => Scalar::False,
                Ok(Value::String(s)) => Scalar::Str(match ids.get(&s) {
                    Some(&id) => id,
                    None => {
                        let id = out.strings.len();
                        ids.insert(s.clone(), id);
                        out.strings.push(s);
                        id
                    }
                }),
                _ => Scalar::Other,
            };
            out.by_node.insert(node as *const Node, parsed);
        }
        out
    }

    fn get(&self, node: &Node) -> Scalar {
        self.by_node
            .get(&(node as *const Node))
            .copied()
            .unwrap_or(Scalar::Other)
    }

    /// `node` as a `type` name: one of [`SCHEMA_TYPES`], `"unknown"` for any
    /// other string (the converter rejects it), `None` when not a string.
    /// Constant time: comparing strings checks their lengths first.
    fn type_name(&self, node: &Node) -> Option<&'static str> {
        let Scalar::Str(id) = self.get(node) else {
            return None;
        };
        let t = self.strings[id].as_str();
        Some(
            SCHEMA_TYPES
                .into_iter()
                .find(|k| *k == t)
                .unwrap_or("unknown"),
        )
    }
}

/// The value of `type` as the grammar converter sees a schema: its own, or
/// replaced by one entry of a `type` list, or removed (an empty list).
#[derive(Clone, Copy)]
enum TypeView {
    Own,
    Is(&'static str),
    Removed,
}

/// One schema the walk in [`check_converter_depth`] has still to visit.
struct Visit<'a> {
    node: &'a Node,
    ty: TypeView,
    level: usize,
    /// Reached through the converter's `CreateRuleFromSchema`, which reuses
    /// its built-in rules for basic schemas without visiting them.
    via_create: bool,
}

/// Walk `root`, the schema text the backend receives, the way OpenVINO
/// GenAI's grammar converter (xgrammar 0.1.31, `JSONSchemaConverter`)
/// recurses through it, and reject it when that recursion would nest more
/// than [`MAX_SCHEMA_DEPTH`] levels, when a local `$ref` is not a plain JSON
/// pointer (see [`local_ref_target`]), or once the walk has cost more than
/// [`MAX_SCHEMA_VISITS`].
///
/// Each nested `VisitSchema` call is one level, the schema itself level 1.
/// The converter descends into a schema's subschemas in the schema's own
/// key order and caches only what it entered through a `$ref`, by the ref's
/// text: a ref already entered (or being entered) stops there, and a target
/// reached by plain nesting is converted again each time. So the walk is a
/// depth-first visit in the converter's order with that same cache. It
/// keeps its own stack, so it never recurses whatever the schema.
fn check_converter_depth(root: &Node) -> Result<(), String> {
    converter_walk(root, None)
}

/// Refuse a `$ref` the engine's grammar backend would turn into "any"
/// rather than fail on (named `engine` in the message):
///
/// - [`crate::RefResolver::Xgrammar`]: any ref but `#` and `#/...` that
///   xgrammar 0.1.31's converter reaches: its `URIToRule` makes it "any
///   JSON" (`json_schema_converter.cc`, "URI should either be '#' or start
///   with '#/'"). Walks what [`check_converter_depth`] walks: the schemas
///   the converter visits, ref targets included (inside data keywords too).
/// - [`crate::RefResolver::LlamaServer`]: any ref but `#/...` anywhere in
///   the document, data and `$ref` siblings included. llama-server's
///   `resolve_refs` visits every value (data keywords, tuple `items`, every
///   `allOf` member) and records the refs it meets, except beside another
///   `$ref`; a `#/...` pointer can still reach a ref there, and a ref it
///   never recorded, or one to an `https://` document (it never fetches),
///   becomes "any object". It fails on the other forms (`#` alone
///   included). A ref in data that no pointer reaches is refused too: a
///   conservative 400, since telling which data a pointer reaches would
///   mean mirroring the converter.
pub fn check_pointer_refs_only(
    raw: &str,
    schema: &Value,
    engine: &str,
    resolver: crate::RefResolver,
) -> Result<(), String> {
    match resolver {
        crate::RefResolver::Full => Ok(()),
        crate::RefResolver::Xgrammar => {
            let raw: Box<RawValue> =
                serde_json::from_str(raw).map_err(|e| format!("{FIELD}: {e}"))?;
            let root = Node::parse(raw).map_err(|e| format!("{FIELD}: {e}"))?;
            converter_walk(&root, Some(engine))
        }
        crate::RefResolver::LlamaServer => {
            let mut stack = vec![schema];
            while let Some(v) = stack.pop() {
                match v {
                    Value::Array(items) => stack.extend(items),
                    Value::Object(obj) => {
                        if let Some(Value::String(r)) = obj.get("$ref") {
                            if !r.starts_with("#/") {
                                return Err(unenforced_ref(r, engine, "`#/...`"));
                            }
                        }
                        // A `$ref`'s siblings too: llama-server records
                        // none there, but a `#/...` pointer can reach one,
                        // and an unrecorded ref becomes "any object".
                        stack.extend(obj.values());
                    }
                    _ => {}
                }
            }
            Ok(())
        }
    }
}

/// The 400 for a `$ref` `engine` would leave unconstrained.
fn unenforced_ref(r: &str, engine: &str, forms: &str) -> String {
    let shown: String = r.chars().take(80).collect();
    format!(
        "{FIELD}: $ref {} is not supported by this server's engine ({engine}): it \
         enforces only references inside the schema written as {forms} pointers, and \
         would leave this one unconstrained",
        Value::String(shown)
    )
}

/// The walk of [`check_converter_depth`]; with `pointer_refs_only` (the
/// engine's name) it also refuses a ref that is not `#` or `#/...`.
fn converter_walk(root: &Node, pointer_refs_only: Option<&str>) -> Result<(), String> {
    let scalars = Scalars::new(root);
    let mut walk = Walk {
        root,
        cost: scalars.cost,
        entered: vec![false; scalars.strings.len()],
        scalars: &scalars,
        pattern_keys: 0,
        pointer_refs_only,
    };
    let mut stack = vec![Visit {
        node: root,
        ty: TypeView::Own,
        level: 1,
        via_create: true,
    }];
    let mut children = Vec::new();
    while let Some(v) = stack.pop() {
        walk.cost += 1 + match v.node {
            Node::Obj(entries) => entries.len(),
            _ => 0,
        };
        if walk.cost > MAX_SCHEMA_VISITS {
            return Err(format!("{FIELD} is too complex"));
        }
        if v.via_create && walk.is_basic_schema(v.node, v.ty) {
            continue;
        }
        if v.level > MAX_SCHEMA_DEPTH {
            return Err(format!(
                "{FIELD} nests deeper than {MAX_SCHEMA_DEPTH} levels through $ref"
            ));
        }
        walk.children(v.node, v.ty, &mut children)?;
        if walk.pattern_keys > MAX_PATTERN_PROPERTIES_TOTAL {
            return Err(format!(
                "{FIELD} has {} patternProperties keys counting each visit and merge; \
                 the limit is {MAX_PATTERN_PROPERTIES_TOTAL}",
                walk.pattern_keys
            ));
        }
        // Last pushed is visited first: push in reverse to keep the order.
        stack.extend(
            children
                .drain(..)
                .rev()
                .map(|(node, ty, via_create)| Visit {
                    node,
                    ty,
                    level: v.level + 1,
                    via_create,
                }),
        );
    }
    Ok(())
}

/// A subschema the converter visits next: the node, how it sees its
/// `type`, and whether it goes through `CreateRuleFromSchema`.
type Child<'a> = (&'a Node, TypeView, bool);

/// The fused schema the converter visits for an `allOf` of two or more.
static EMPTY_SCHEMA: Node = Node::Obj(Vec::new());

/// The state of [`check_converter_depth`]'s walk.
struct Walk<'a, 's> {
    /// The whole schema, which local refs point into.
    root: &'a Node,
    scalars: &'s Scalars,
    /// By interned `$ref` text: entered, or seen needing no visit (`#`, a
    /// remote ref). Each ref text is resolved once.
    entered: Vec<bool>,
    /// Work done so far, against [`MAX_SCHEMA_VISITS`].
    cost: usize,
    /// `patternProperties` keys charged so far, against
    /// [`MAX_PATTERN_PROPERTIES_TOTAL`]: once per visit of an object, and
    /// once per `allOf` member, which llguidance merges anew for each
    /// `allOf` (its ref cache doesn't cover merges).
    pattern_keys: usize,
    /// See [`converter_walk`].
    pointer_refs_only: Option<&'s str>,
}

impl<'a> Walk<'a, '_> {
    /// Append to `out` the subschemas the converter's `VisitSchema` visits
    /// for `node`, in its order, entering local `$ref`s on the way. Mirrors
    /// `VisitSchema`'s dispatch: `$ref`, then `const`, `enum`,
    /// `anyOf`/`oneOf`, `allOf`, `type`, then object or array keywords.
    /// Keywords the converter would reject are walked anyway when they hold
    /// schemas. Such an over-count is not harmless: with the converter's
    /// order-dependent ref cache, a `$ref` entered too early is marked
    /// entered at a shallow level and hides a deeper real first entry. So
    /// the walk must not enter what the converter never visits.
    fn children(
        &mut self,
        node: &'a Node,
        ty: TypeView,
        out: &mut Vec<Child<'a>>,
    ) -> Result<(), String> {
        let Node::Obj(_) = node else {
            return Ok(());
        };
        if let TypeView::Own = ty {
            if let Some(r) = node.get("$ref") {
                let Scalar::Str(id) = self.scalars.get(r) else {
                    return Ok(());
                };
                if std::mem::replace(&mut self.entered[id], true) {
                    return Ok(());
                }
                let r = &self.scalars.strings[id];
                if let Some(engine) = self.pointer_refs_only {
                    if !r.starts_with('#') {
                        return Err(unenforced_ref(r, engine, "`#` or `#/...`"));
                    }
                }
                let (target, work) = local_ref_target(self.root, r)?;
                self.cost += work;
                if let Some(target) = target {
                    out.push((target, TypeView::Own, false));
                }
                return Ok(());
            }
            if node.get("const").is_some() || node.get("enum").is_some() {
                return Ok(());
            }
            if let Some(cases) = node.get("anyOf").or_else(|| node.get("oneOf")) {
                if let Node::Arr(cases) = cases {
                    out.extend(cases.iter().map(|c| (c, TypeView::Own, true)));
                }
                return Ok(());
            }
            if let Some(members) = node.get("allOf") {
                if let Node::Arr(m) = members {
                    for member in m {
                        self.charge_merged_pattern_keys(member);
                    }
                }
                match members {
                    Node::Arr(m) if m.len() == 1 => out.push((&m[0], TypeView::Own, false)),
                    Node::Arr(_) => out.push((&EMPTY_SCHEMA, TypeView::Own, false)),
                    _ => {}
                }
                return Ok(());
            }
        }
        let ty = match ty {
            TypeView::Own => match node.get("type") {
                Some(Node::Arr(types)) if types.is_empty() => {
                    out.push((node, TypeView::Removed, false));
                    return Ok(());
                }
                Some(Node::Arr(types)) => {
                    for t in types {
                        let Some(t) = self.scalars.type_name(t) else {
                            break;
                        };
                        out.push((node, TypeView::Is(t), true));
                    }
                    return Ok(());
                }
                Some(t) => match self.scalars.type_name(t) {
                    Some(t) => Some(t),
                    None => return Ok(()),
                },
                None => None,
            },
            TypeView::Is(t) => Some(t),
            TypeView::Removed => None,
        };
        let has = |k: &str| node.get(k).is_some();
        let object = match ty {
            Some("object") => true,
            Some("array") => false,
            Some(_) => return Ok(()),
            None if has("properties")
                || has("additionalProperties")
                || has("unevaluatedProperties") =>
            {
                true
            }
            None if has("items") || has("prefixItems") || has("unevaluatedItems") => false,
            None => return Ok(()),
        };
        if object {
            self.object_children(node, out);
        } else {
            array_children(node, out);
        }
        Ok(())
    }

    /// The subschemas `VisitObject` visits: none when `maxProperties` is
    /// 0 (the object holds no property); with a non-empty
    /// `patternProperties` its values (else a `propertyNames`, which the
    /// converter visits with `type: "string"` added to its other keys);
    /// otherwise the `properties` values in order, then the schema for
    /// other properties.
    fn object_children(&mut self, node: &'a Node, out: &mut Vec<Child<'a>>) {
        // [`check_limits`] has made it a plain non-negative integer literal.
        if let Some(Node::Scalar(max)) = node.get("maxProperties") {
            if max.get().parse::<u64>() == Ok(0) {
                return;
            }
        }
        match (node.get("patternProperties"), node.get("propertyNames")) {
            (Some(Node::Obj(patterns)), _) if !patterns.is_empty() => {
                self.pattern_keys += patterns.len();
                out.extend(patterns.iter().map(|(_, s)| (s, TypeView::Own, true)));
                return;
            }
            (_, Some(names @ Node::Obj(_))) => {
                // `VisitSchema` dispatches on these before `type`.
                let dispatched = ["$ref", "const", "enum", "anyOf", "oneOf", "allOf"]
                    .iter()
                    .any(|k| names.get(k).is_some());
                let ty = if dispatched {
                    TypeView::Own
                } else {
                    TypeView::Is("string")
                };
                out.push((names, ty, true));
                return;
            }
            _ => {}
        }
        if let Some(Node::Obj(props)) = node.get("properties") {
            out.extend(props.iter().map(|(_, s)| (s, TypeView::Own, true)));
        }
        let other = match node.get("additionalProperties") {
            Some(a) => Some(a),
            None => node.get("unevaluatedProperties"),
        };
        if let Some(other) = other.filter(|o| self.scalars.get(o) != Scalar::False) {
            out.push((other, TypeView::Own, true));
        }
    }

    /// Charge the `patternProperties` keys an `allOf` member brings to a
    /// merge: its own, and those of the schema its local `$ref` points to.
    fn charge_merged_pattern_keys(&mut self, member: &'a Node) {
        let keys = |n: &Node| match n.get("patternProperties") {
            Some(Node::Obj(p)) => p.len(),
            _ => 0,
        };
        if !matches!(member, Node::Obj(_)) {
            return;
        }
        self.pattern_keys += keys(member);
        if let Some(r) = member.get("$ref") {
            if let Scalar::Str(id) = self.scalars.get(r) {
                // A ref this walk can't read is left to the converter,
                // as before; only the count is skipped.
                if let Ok((target, work)) = local_ref_target(self.root, &self.scalars.strings[id]) {
                    self.cost += work;
                    if let Some(target @ Node::Obj(_)) = target {
                        self.pattern_keys += keys(target);
                    }
                }
            }
        }
    }

    /// True when the converter answers `node` (seen with `ty`) from its
    /// built-in rules without visiting it: `true`, `{}` or `{"type": T}` for
    /// a basic type `T`, once keys its cache ignores are dropped.
    fn is_basic_schema(&self, node: &Node, ty: TypeView) -> bool {
        let entries = match node {
            Node::Obj(entries) => entries,
            Node::Scalar(_) => return self.scalars.get(node) == Scalar::True,
            Node::Arr(_) => return false,
        };
        let mut own = None;
        for (k, v) in entries {
            let k = k.as_str();
            if CACHE_IGNORED_KEYS.contains(&k) {
                continue;
            }
            if k != "type" {
                return false;
            }
            if let TypeView::Own = ty {
                match self.scalars.type_name(v) {
                    Some(t) => own = Some(t),
                    None => return false,
                }
            }
        }
        let ty = match ty {
            TypeView::Own => own,
            TypeView::Is(t) => Some(t),
            TypeView::Removed => None,
        };
        ty.is_none_or(|t| t != "unknown")
    }
}

/// The subschemas `VisitArray` visits: each `prefixItems` entry, then
/// `items` (or, without it, `unevaluatedItems`).
fn array_children<'a>(node: &'a Node, out: &mut Vec<Child<'a>>) {
    if let Some(Node::Arr(prefix)) = node.get("prefixItems") {
        out.extend(prefix.iter().map(|s| (s, TypeView::Own, true)));
    }
    let rest = match node.get("items") {
        Some(i) => Some(i),
        None => node.get("unevaluatedItems"),
    };
    if let Some(rest @ Node::Obj(_)) = rest {
        out.push((rest, TypeView::Own, true));
    }
}

/// Keys the converter's rule cache ignores (`GetSchemaCacheIndex`).
const CACHE_IGNORED_KEYS: [&str; 9] = [
    "title",
    "default",
    "description",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
    "$schema",
];

/// The `type` names the converter knows; the basic ones of its built-in
/// rules are the same seven.
const SCHEMA_TYPES: [&str; 7] = [
    "integer", "number", "string", "boolean", "null", "array", "object",
];

/// The schema a `$ref` points to, `None` for a ref that is not local (it
/// does not start with `#`; the converter accepts any value there, and
/// [`parse_json_schema`] refuses one that names a document outside the
/// schema afterwards, see `cascadia_grammar::ref_outside_the_schema`) or is `#` itself (the converter
/// refers back to the root without visiting it), and the work resolving it
/// took: the ref's length plus the keys scanned.
///
/// A local ref must be a plain JSON pointer: `#/` and keys joined by `/`,
/// with no empty key and no `~` or `%`. Where it is, it is read here the way
/// both RFC 6901 and the converter (which skips empty keys, does not
/// unescape, and walks into objects only) read it, and must reach the same
/// value both ways, a JSON object or boolean (a schema). Any other local
/// ref is an error, so a ref cannot point somewhere other than where it
/// seems to, nor at a value that is not a schema.
fn local_ref_target<'a>(root: &'a Node, r: &str) -> Result<(Option<&'a Node>, usize), String> {
    let Some(pointer) = r.strip_prefix('#') else {
        return Ok((None, 1));
    };
    if pointer.is_empty() {
        return Ok((None, 1));
    }
    let unsupported = || {
        let shown: String = r.chars().take(80).collect();
        format!("{FIELD}: unsupported $ref {}", Value::String(shown))
    };
    let plain = pointer.strip_prefix('/').is_some_and(|keys| {
        keys.split('/')
            .all(|k| !k.is_empty() && !k.contains(['~', '%']))
    });
    if !plain {
        return Err(unsupported());
    }
    let mut scanned = r.len();
    let rfc = rfc6901_target(root, pointer, &mut scanned);
    let converter = converter_target(root, pointer, &mut scanned);
    match (rfc, converter) {
        (Some(a), Some(b)) if std::ptr::eq(a, b) && is_schema_value(a) => Ok((Some(a), scanned)),
        _ => Err(unsupported()),
    }
}

/// True for a JSON object or boolean, the values that can be a schema.
fn is_schema_value(node: &Node) -> bool {
    match node {
        Node::Obj(_) => true,
        Node::Scalar(v) => matches!(v.get(), "true" | "false"),
        Node::Arr(_) => false,
    }
}

/// The keys [`Node::get`] scans to look one up in `node`.
fn keys_of(node: &Node) -> usize {
    match node {
        Node::Obj(entries) => entries.len(),
        _ => 1,
    }
}

/// The node the RFC 6901 JSON pointer `pointer` (`""` or `/...`) names.
/// Adds the keys it scans to `scanned`.
fn rfc6901_target<'a>(root: &'a Node, pointer: &str, scanned: &mut usize) -> Option<&'a Node> {
    if pointer.is_empty() {
        return Some(root);
    }
    let mut current = root;
    for token in pointer.strip_prefix('/')?.split('/') {
        *scanned += keys_of(current);
        let token = token.replace("~1", "/").replace("~0", "~");
        current = match current {
            Node::Obj(_) => current.get(&token)?,
            Node::Arr(items) => {
                let canonical = token == "0" || (!token.starts_with('0') && !token.is_empty());
                if !canonical || !token.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                items.get(token.parse::<usize>().ok()?)?
            }
            Node::Scalar(_) => return None,
        };
    }
    Some(current)
}

/// The node the converter's `URIToRule` reaches for `#` + `pointer`: split
/// on `/`, empty parts skipped, no unescaping, each part a key of an object.
/// Adds the keys it scans to `scanned`.
fn converter_target<'a>(root: &'a Node, pointer: &str, scanned: &mut usize) -> Option<&'a Node> {
    let mut current = root;
    for part in pointer.split('/').filter(|p| !p.is_empty()) {
        *scanned += keys_of(current);
        match current {
            Node::Obj(_) => current = current.get(part)?,
            _ => return None,
        }
    }
    Some(current)
}

/// Highest [`ref_depth_bound`] accepted. A coarse backstop to
/// [`check_converter_depth`] in case that walk ever drifts from the real
/// converter (another version, a visitor mirrored wrongly): far below the
/// ~2000-link depth that crashed the converter, far above realistic
/// recursive unions.
const MAX_REF_DEPTH_BOUND: usize = 512;

/// Reject `root` when [`ref_depth_bound`] is above [`MAX_REF_DEPTH_BOUND`].
/// Linear in the schema's size and iterative.
fn check_ref_depth_bound(root: &Value) -> Result<(), String> {
    if ref_depth_bound(root) > MAX_REF_DEPTH_BOUND {
        return Err(format!(
            "{FIELD} may nest deeper than {MAX_REF_DEPTH_BOUND} levels through $ref"
        ));
    }
    Ok(())
}

/// An upper bound on how deep `root` nests with its local refs followed,
/// whatever order a backend walks it in.
///
/// The graph's nodes are the root and every ref target reached from it. A
/// node's subtree stops at any other target inside it, and has an edge to
/// it, as it has to the target of each `$ref` in it. Each node's height is
/// its own structural nesting (objects and arrays, the node itself
/// included). Entering a node from another puts it at most the other's
/// height below the other's top, so a path's depth is at most the sum of
/// its nodes' heights. A path visits a node once (a ref back to one on the
/// path stops), so within one strongly connected set of nodes it is at most
/// the sum of their heights. The bound is the heaviest path through the
/// condensed graph from the root.
fn ref_depth_bound(root: &Value) -> usize {
    let targets = ref_targets(root);
    let mut ids: HashMap<*const Value, usize> = HashMap::from([(root as *const Value, 0)]);
    let mut nodes: Vec<&Value> = vec![root];
    let mut heights: Vec<usize> = Vec::new();
    let mut edges: Vec<Vec<usize>> = Vec::new();
    let mut next = 0;
    while next < nodes.len() {
        let node = nodes[next];
        let mut out = Vec::new();
        let mut height = 0;
        let mut stack = vec![(node, 1)];
        while let Some((v, level)) = stack.pop() {
            let children: Box<dyn Iterator<Item = &Value>> = match v {
                Value::Object(o) => Box::new(o.values()),
                Value::Array(a) => Box::new(a.iter()),
                _ => continue,
            };
            height = height.max(level);
            if let Some(r) = v.get("$ref").and_then(Value::as_str) {
                for t in backstop_targets(root, r) {
                    out.push(node_id(t, &mut ids, &mut nodes));
                }
            }
            for c in children {
                if targets.contains(&(c as *const Value)) {
                    out.push(node_id(c, &mut ids, &mut nodes));
                } else {
                    stack.push((c, level + 1));
                }
            }
        }
        heights.push(height);
        edges.push(out);
        next += 1;
    }
    heaviest_path(&heights, &edges)
}

/// The graph node for `t`, added when new.
fn node_id<'a>(
    t: &'a Value,
    ids: &mut HashMap<*const Value, usize>,
    nodes: &mut Vec<&'a Value>,
) -> usize {
    *ids.entry(t as *const Value).or_insert_with(|| {
        nodes.push(t);
        nodes.len() - 1
    })
}

/// Every value some local `$ref` in `root` points to, by address.
fn ref_targets(root: &Value) -> HashSet<*const Value> {
    let mut targets = HashSet::new();
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(o) => {
                if let Some(r) = o.get("$ref").and_then(Value::as_str) {
                    targets.extend(backstop_targets(root, r).map(|t| t as *const Value));
                }
                stack.extend(o.values());
            }
            Value::Array(a) => stack.extend(a.iter()),
            _ => {}
        }
    }
    targets
}

/// The heaviest path from node 0 through the graph `edges` once each strongly
/// connected set is condensed into one node weighing its members' `weights`
/// summed. Iterative Tarjan: it emits a set only after every set reachable
/// from it, so each set's heaviest path is final when it is emitted.
fn heaviest_path(weights: &[usize], edges: &[Vec<usize>]) -> usize {
    let n = weights.len();
    let mut index: Vec<Option<usize>> = vec![None; n];
    let mut low = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut set_of: Vec<Option<usize>> = vec![None; n];
    let mut best: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut counter = 0;
    let mut calls: Vec<(usize, usize)> = vec![(0, 0)];
    index[0] = Some(0);
    low[0] = 0;
    counter += 1;
    stack.push(0);
    on_stack[0] = true;
    while let Some(&mut (v, ref mut e)) = calls.last_mut() {
        if let Some(&w) = edges[v].get(*e) {
            *e += 1;
            match index[w] {
                None => {
                    index[w] = Some(counter);
                    low[w] = counter;
                    counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    calls.push((w, 0));
                }
                Some(iw) if on_stack[w] => low[v] = low[v].min(iw),
                Some(_) => {}
            }
            continue;
        }
        calls.pop();
        if let Some(&(parent, _)) = calls.last() {
            low[parent] = low[parent].min(low[v]);
        }
        if Some(low[v]) != index[v] {
            continue;
        }
        let set = best.len();
        let mut members = Vec::new();
        while let Some(w) = stack.pop() {
            on_stack[w] = false;
            set_of[w] = Some(set);
            members.push(w);
            if w == v {
                break;
            }
        }
        let weight: usize = members.iter().map(|&m| weights[m]).sum();
        let after = members
            .iter()
            .flat_map(|&m| edges[m].iter())
            .filter_map(|&w| set_of[w].filter(|&s| s != set))
            .map(|s| best[s])
            .max()
            .unwrap_or(0);
        best.push(weight.saturating_add(after));
    }
    set_of[0].map_or(0, |s| best[s])
}

/// Every value the `$ref` text `r` may point to in `root`, for
/// [`ref_depth_bound`]: read as an RFC 6901 pointer and as the converter
/// reads it (empty parts skipped, no unescaping, objects only). Remote refs
/// point nowhere.
fn backstop_targets<'a>(root: &'a Value, r: &str) -> impl Iterator<Item = &'a Value> {
    let pointer = r.strip_prefix('#');
    let rfc = pointer.and_then(|p| root.pointer(p));
    let converter = pointer.and_then(|p| {
        p.split('/')
            .filter(|k| !k.is_empty())
            .try_fold(root, |v, k| v.as_object()?.get(k))
    });
    let converter = converter.filter(|c| !rfc.is_some_and(|r| std::ptr::eq(r, *c)));
    rfc.into_iter().chain(converter)
}

/// True when objects and arrays in `v` nest more than `levels` deep, counting
/// `v` itself as one level. Stops descending once past the limit.
fn nests_deeper_than(v: &Value, levels: usize) -> bool {
    let deeper = |c: &Value| nests_deeper_than(c, levels - 1);
    match v {
        Value::Object(o) => levels == 0 || o.values().any(deeper),
        Value::Array(a) => levels == 0 || a.iter().any(deeper),
        _ => false,
    }
}

/// A JSON value that keeps its object key order and the exact text of its
/// scalars. The workspace's `serde_json::Value` sorts keys, and the grammar
/// emits object properties in schema order.
#[derive(Clone)]
enum Node {
    Obj(Vec<(String, Node)>),
    Arr(Vec<Node>),
    /// Any other value, as its JSON text.
    Scalar(Box<RawValue>),
}

/// An object's entries in order, values left as text.
struct RawEntries(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for RawEntries {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = RawEntries;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<RawEntries, A::Error> {
                let mut out = Vec::new();
                while let Some(entry) = map.next_entry::<String, Box<RawValue>>()? {
                    out.push(entry);
                }
                Ok(RawEntries(out))
            }
        }
        d.deserialize_map(V)
    }
}

impl Node {
    fn parse(raw: Box<RawValue>) -> serde_json::Result<Node> {
        match raw.get().trim_start().as_bytes().first() {
            Some(b'{') => {
                let RawEntries(entries) = serde_json::from_str(raw.get())?;
                let entries = entries
                    .into_iter()
                    .map(|(k, v)| Ok((k, Node::parse(v)?)))
                    .collect::<serde_json::Result<_>>()?;
                Ok(Node::Obj(entries))
            }
            Some(b'[') => {
                let items: Vec<Box<RawValue>> = serde_json::from_str(raw.get())?;
                let items = items
                    .into_iter()
                    .map(Node::parse)
                    .collect::<serde_json::Result<_>>()?;
                Ok(Node::Arr(items))
            }
            _ => Ok(Node::Scalar(raw)),
        }
    }

    fn write(&self, out: &mut String) {
        match self {
            Node::Obj(entries) => {
                out.push('{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(k.clone()).to_string());
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
            Node::Arr(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Node::Scalar(raw) => out.push_str(raw.get()),
        }
    }

    fn get(&self, key: &str) -> Option<&Node> {
        match self {
            Node::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The value as a parsed scalar, `None` for an object or array.
    fn scalar(&self) -> Option<Value> {
        match self {
            Node::Scalar(raw) => serde_json::from_str(raw.get()).ok(),
            _ => None,
        }
    }

    fn scalar_node(v: &Value) -> Node {
        Node::Scalar(RawValue::from_string(v.to_string()).expect("a serialized value is JSON"))
    }
}

/// Keys an `allOf` member may have and still be merged.
const MERGEABLE_MEMBER_KEYS: [&str; 6] = [
    "type",
    "properties",
    "required",
    "additionalProperties",
    "title",
    "description",
];

/// `text`, a compact schema, with every `allOf` of two or more plain object
/// schemas merged into one object schema, or `None` when it has none. The
/// grammar backend `ov-genai` uses accepts any JSON value for such an
/// `allOf`, and zod intersections (`.and()`) produce exactly that shape.
///
/// A member is plain when it holds only [`MERGEABLE_MEMBER_KEYS`], with
/// `type` (if any) `"object"`, `properties` an object, `required` a list of
/// names and `additionalProperties` a boolean. The merge keeps the
/// properties in member order, the union of `required` in order, and
/// `additionalProperties: false` when any member has it. A property defined
/// by more than one member must have the same schema text in each, or the
/// `allOf` is left alone. The node holding the `allOf` keeps its other keys; it is
/// left alone if it has its own `properties`, `required`,
/// `additionalProperties`, another dispatch key or a `type` other than
/// `"object"`.
fn merge_object_all_of(text: &str) -> Option<String> {
    let raw: Box<RawValue> = serde_json::from_str(text).ok()?;
    let mut root = Node::parse(raw).ok()?;
    let mut merged = false;
    merge_in_schema(&mut root, 0, &mut merged);
    if !merged {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    root.write(&mut out);
    Some(out)
}

/// Merge mergeable `allOf`s in `node` and below, deepest first. Walks the
/// same places as [`for_each_schema`]: never data keys, and schema maps by
/// their values only.
fn merge_in_schema(node: &mut Node, depth: usize, merged: &mut bool) {
    if depth > MAX_SCHEMA_DEPTH {
        return;
    }
    match node {
        Node::Arr(items) => {
            for item in items {
                merge_in_schema(item, depth + 1, merged);
            }
        }
        Node::Obj(entries) => {
            for (key, value) in entries.iter_mut() {
                let key = key.as_str();
                if DATA_KEYS.contains(&key) {
                    continue;
                }
                match value {
                    Node::Obj(map) if SCHEMA_MAPS.contains(&key) => {
                        for (_, sub) in map.iter_mut() {
                            merge_in_schema(sub, depth + 1, merged);
                        }
                    }
                    _ => merge_in_schema(value, depth + 1, merged),
                }
            }
            if let Some(replacement) = merged_all_of(entries) {
                *entries = replacement;
                *merged = true;
            }
        }
        Node::Scalar(_) => {}
    }
}

/// The entries of an object schema with its `allOf` merged, or `None` when
/// it has no mergeable `allOf` (see [`merge_object_all_of`]).
fn merged_all_of(entries: &[(String, Node)]) -> Option<Vec<(String, Node)>> {
    let object = Value::from("object");
    let find = |key: &str| entries.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let Some(Node::Arr(members)) = find("allOf") else {
        return None;
    };
    if members.len() < 2 {
        return None;
    }
    let blocked = ["properties", "required", "additionalProperties"]
        .iter()
        .chain(DISPATCH_KEYS.iter().filter(|k| **k != "allOf"))
        .any(|k| find(k).is_some());
    let holder_type = find("type");
    if blocked || holder_type.is_some_and(|t| t.scalar().as_ref() != Some(&object)) {
        return None;
    }
    let mut properties: Vec<(String, Node)> = Vec::new();
    let mut has_properties = false;
    let mut required: Vec<String> = Vec::new();
    let mut closed = false;
    for member in members {
        let Node::Obj(fields) = member else {
            return None;
        };
        if fields
            .iter()
            .any(|(k, _)| !MERGEABLE_MEMBER_KEYS.contains(&k.as_str()))
        {
            return None;
        }
        if member
            .get("type")
            .is_some_and(|t| t.scalar().as_ref() != Some(&object))
        {
            return None;
        }
        match member.get("additionalProperties").map(Node::scalar) {
            None => {}
            Some(Some(Value::Bool(b))) => closed |= !b,
            Some(_) => return None,
        }
        match member.get("required") {
            None => {}
            Some(Node::Arr(names)) => {
                for name in names {
                    let Some(Value::String(name)) = name.scalar() else {
                        return None;
                    };
                    if !required.contains(&name) {
                        required.push(name);
                    }
                }
            }
            Some(_) => return None,
        }
        match member.get("properties") {
            None => {}
            Some(Node::Obj(props)) => {
                has_properties = true;
                for (name, schema) in props {
                    match properties.iter().find(|(n, _)| n == name) {
                        None => properties.push((name.clone(), schema.clone())),
                        // Both definitions apply to the value; keeping one
                        // would drop the other's constraint.
                        Some((_, kept)) if node_text(kept) != node_text(schema) => return None,
                        Some(_) => {}
                    }
                }
            }
            Some(_) => return None,
        }
    }
    let mut out = Vec::with_capacity(entries.len() + 3);
    for (key, value) in entries {
        if key != "allOf" {
            out.push((key.clone(), value.clone()));
            continue;
        }
        if holder_type.is_none() {
            out.push(("type".into(), Node::scalar_node(&object)));
        }
        if has_properties {
            out.push((
                "properties".into(),
                Node::Obj(std::mem::take(&mut properties)),
            ));
        }
        if !required.is_empty() {
            let names = required
                .iter()
                .map(|n| Node::scalar_node(&Value::from(n.as_str())));
            out.push(("required".into(), Node::Arr(names.collect())));
        }
        if closed {
            out.push((
                "additionalProperties".into(),
                Node::scalar_node(&Value::Bool(false)),
            ));
        }
    }
    Some(out)
}

/// The compact text of `node`.
fn node_text(node: &Node) -> String {
    let mut out = String::new();
    node.write(&mut out);
    out
}

/// The grammar that enforces `rf`, or `None` for plain text.
pub fn grammar_for(rf: &ResponseFormat) -> Option<GrammarSpec> {
    let body = match rf {
        ResponseFormat::Text => return None,
        ResponseFormat::JsonObject => JSON_OBJECT_SCHEMA.to_owned(),
        ResponseFormat::JsonSchema { raw, .. } => raw.clone(),
    };
    Some(GrammarSpec {
        kind: GrammarKind::JsonSchema,
        body,
    })
}

const SIBLINGS: &str = "siblings of $ref/const/enum/anyOf/oneOf/allOf";

/// Keys whose presence beside another keyword changes how the schema is
/// resolved; the decoder handles each only on its own.
const DISPATCH_KEYS: [&str; 6] = ["$ref", "const", "enum", "anyOf", "oneOf", "allOf"];

/// Keys that never constrain output, so they may sit beside a dispatch key.
const ANNOTATIONS: [&str; 12] = [
    "description",
    "title",
    "default",
    "examples",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$defs",
    "definitions",
    "$schema",
    "$id",
];

/// Keywords flagged whenever they appear.
const ALWAYS_UNENFORCED: [&str; 10] = [
    "not",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "contains",
    "minContains",
    "maxContains",
    "multipleOf",
];

/// Keys whose value is a map from names to subschemas: walk the values,
/// never treat the names as keywords.
const SCHEMA_MAPS: [&str; 6] = [
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
];

/// Keys whose value is instance data, not a schema.
const DATA_KEYS: [&str; 4] = ["enum", "const", "default", "examples"];

/// JSON Schema keywords in `schema` the output grammar does not enforce,
/// sorted and de-duplicated. Output may still be valid JSON that violates them.
pub fn unenforced_keywords(schema: &Value) -> Vec<&'static str> {
    let mut found = Vec::new();
    for_each_schema(schema, 0, &mut |obj| flag_unenforced(obj, &mut found));
    found.sort_unstable();
    found.dedup();
    found
}

/// Call `f` on every schema object reachable from `v`: values of schema maps
/// (never their names) and every other key's value, except data keys. Stops
/// below [`MAX_SCHEMA_DEPTH`].
fn for_each_schema(v: &Value, depth: usize, f: &mut impl FnMut(&Map<String, Value>)) {
    if depth > MAX_SCHEMA_DEPTH {
        return;
    }
    match v {
        Value::Array(items) => {
            for item in items {
                for_each_schema(item, depth + 1, f);
            }
        }
        Value::Object(obj) => {
            f(obj);
            for (key, value) in obj {
                let key = key.as_str();
                if DATA_KEYS.contains(&key) {
                    continue;
                }
                match value {
                    Value::Object(map) if SCHEMA_MAPS.contains(&key) => {
                        for sub in map.values() {
                            for_each_schema(sub, depth + 1, f);
                        }
                    }
                    _ => for_each_schema(value, depth + 1, f),
                }
            }
        }
        _ => {}
    }
}

fn flag_unenforced(obj: &Map<String, Value>, found: &mut Vec<&'static str>) {
    if has_constrained_siblings(obj) {
        found.push(SIBLINGS);
    }
    for (key, value) in obj {
        let key = key.as_str();
        if let Some(k) = ALWAYS_UNENFORCED.iter().find(|k| **k == key) {
            found.push(k);
        }
        match key {
            "uniqueItems" if value == &Value::Bool(true) => found.push("uniqueItems"),
            "allOf" if value.as_array().is_some_and(|a| a.len() > 1) => found.push("allOf"),
            _ => {}
        }
    }
}

/// True when the object has a dispatch key plus any other constraining key.
/// `type` beside `enum` or `const` is allowed: the values already pin the type.
fn has_constrained_siblings(obj: &Map<String, Value>) -> bool {
    let dispatch: Vec<&str> = DISPATCH_KEYS
        .iter()
        .copied()
        .filter(|k| obj.contains_key(*k))
        .collect();
    if dispatch.is_empty() {
        return false;
    }
    if dispatch.len() > 1 {
        return true;
    }
    let pins_values = dispatch[0] == "enum" || dispatch[0] == "const";
    obj.keys().any(|k| {
        let k = k.as_str();
        k != dispatch[0] && !ANNOTATIONS.contains(&k) && !(pins_values && k == "type")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::GrammarKind;
    use serde_json::json;

    fn parse(v: serde_json::Value) -> Result<ResponseFormat, String> {
        parse_response_format(Some(&v.to_string()))
    }

    /// The `JsonSchema` variant [`parse`] gives for `schema`.
    fn schema_variant(
        name: Option<&str>,
        schema: serde_json::Value,
        strict: bool,
    ) -> ResponseFormat {
        ResponseFormat::JsonSchema {
            name: name.map(str::to_owned),
            raw: schema.to_string(),
            schema,
            strict,
        }
    }

    #[test]
    fn schema_keeps_its_property_order() {
        let text =
            include_str!("../../cascadia-engine-mock/tests/fixtures/browser_use_schema.json");
        let format = format!(
            r#"{{"type": "json_schema", "json_schema": {{"name": "AgentOutput", "schema": {text}}}}}"#
        );
        let g = grammar_for(&parse_response_format(Some(&format)).unwrap()).unwrap();
        let thinking = g.body.find(r#""thinking":"#).unwrap();
        let action = g.body.find(r#""action":"#).unwrap();
        assert!(thinking < action, "{}", g.body);
        assert_eq!(g.body, text.trim());
    }

    #[test]
    fn schema_text_is_compacted_without_touching_strings() {
        let format = r#"{"type":"json_schema","json_schema":{"schema":
            { "type" : "object",
              "properties" : { "z b" : { "const" : "a \" b\\" , "description":"x\ty  " },
                               "a" : { "type" : [ "string" , "null" ] } } } }}"#;
        let g = grammar_for(&parse_response_format(Some(format)).unwrap()).unwrap();
        assert_eq!(
            g.body,
            r#"{"type":"object","properties":{"z b":{"const":"a \" b\\","description":"x\ty  "},"a":{"type":["string","null"]}}}"#
        );
        assert_eq!(minify_json(" [ 1 ,\n\t2 ] "), "[1,2]");
        assert_eq!(minify_json(r#"" \\" , " x ""#), r#"" \\"," x ""#);
    }

    #[test]
    fn duplicate_keys_in_the_schema_are_rejected() {
        let parse_text = |schema: &str| {
            parse_response_format(Some(&format!(
                r#"{{"type":"json_schema","json_schema":{{"schema":{schema}}}}}"#
            )))
        };
        let dup = |k: &str| {
            Err(format!(
                "response_format.json_schema.schema has a duplicate key \"{k}\""
            ))
        };
        assert_eq!(
            parse_text(r#"{"type":"string","maxLength":5,"maxLength":1000000}"#),
            dup("maxLength")
        );
        assert_eq!(
            parse_text(r#"{"type":"string","maxLength":1000000,"maxLength":5}"#),
            dup("maxLength")
        );
        assert_eq!(
            parse_text(
                r#"{"type":"object","properties":{"a":{"type":"string"},"a":{"type":"integer"}}}"#
            ),
            dup("a")
        );
        assert_eq!(
            parse_text(
                r##"{"$ref":"#/$defs/A","$defs":{"A":{"type":"array","items":[{"minItems":1,"minItems":2}]}}}"##
            ),
            dup("minItems")
        );
        // An escaped spelling of the same key is the same key.
        assert_eq!(
            parse_text(r#"{"type":"string","\u0074ype":"integer"}"#),
            dup("type")
        );
        // The same key in sibling or nested objects is fine.
        assert!(parse_text(
            r#"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string","properties":{"type":{"type":"string"}}}}}"#
        )
        .is_ok());
    }

    #[test]
    fn duplicate_key_check_is_depth_limited() {
        let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        assert!(reject_duplicate_keys(&format!(r#"{{"enum":{deep}}}"#)).is_err());
        let ok = format!("{}1{}", "[".repeat(100), "]".repeat(100));
        assert!(reject_duplicate_keys(&ok).is_ok());
    }

    /// Every spelling of "any JSON object" reaches the engine as the one
    /// fixed text, which ov-genai's schema budget never counts or refuses
    /// (its `JSON_OBJECT_PROBE` is this same text).
    #[test]
    fn every_any_object_spelling_reaches_the_engine_as_the_fixed_text() {
        assert_eq!(
            JSON_OBJECT_SCHEMA,
            r#"{"type":"object","additionalProperties":true}"#
        );
        for rf in [
            json!({"type": "json_object"}),
            json!({"type": "json_schema", "json_schema": {"name": "x"}}),
            json!({"type": "json_schema", "json_schema": {"name": "x", "schema": null}}),
            json!({"type": "json_schema", "json_schema": {"name": "x", "schema": true}}),
            json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}),
        ] {
            let g = grammar_for(&parse(rf.clone()).unwrap()).unwrap();
            assert_eq!(g.body, JSON_OBJECT_SCHEMA, "{rf}");
        }
    }

    #[test]
    fn empty_schema_is_json_object() {
        assert_eq!(
            parse(json!({"type": "json_schema", "json_schema": {"schema": {}}})),
            Ok(ResponseFormat::JsonObject)
        );
    }

    #[test]
    fn absent_null_and_text_are_text() {
        assert_eq!(parse_response_format(None), Ok(ResponseFormat::Text));
        assert_eq!(parse(json!(null)), Ok(ResponseFormat::Text));
        assert_eq!(parse(json!({"type": "text"})), Ok(ResponseFormat::Text));
    }

    #[test]
    fn json_object() {
        let rf = parse(json!({"type": "json_object"})).unwrap();
        assert_eq!(rf, ResponseFormat::JsonObject);
        assert!(rf.is_json());
        assert_eq!(rf.type_name(), "json_object");
        assert!(!ResponseFormat::Text.is_json());
        assert_eq!(ResponseFormat::Text.type_name(), "text");
    }

    #[test]
    fn json_schema_with_object_schema() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        let rf = parse(json!({
            "type": "json_schema",
            "json_schema": {"name": "thing", "schema": schema.clone(), "strict": true}
        }))
        .unwrap();
        assert_eq!(rf, schema_variant(Some("thing"), schema.clone(), true));
        assert!(rf.is_json());
        assert_eq!(rf.type_name(), "json_schema");

        // Non-string name is dropped; strict is only true when literally `true`.
        let rf = parse(json!({
            "type": "json_schema",
            "json_schema": {"name": 7, "schema": schema.clone(), "strict": "yes"}
        }))
        .unwrap();
        assert_eq!(rf, schema_variant(None, schema, false));
    }

    #[test]
    fn json_schema_without_schema_null_or_true_is_json_object() {
        assert_eq!(
            parse(json!({"type": "json_schema", "json_schema": {"name": "x"}})),
            Ok(ResponseFormat::JsonObject)
        );
        assert_eq!(
            parse(json!({"type": "json_schema", "json_schema": {"schema": null}})),
            Ok(ResponseFormat::JsonObject)
        );
        assert_eq!(
            parse(json!({"type": "json_schema", "json_schema": {"schema": true}})),
            Ok(ResponseFormat::JsonObject)
        );
    }

    #[test]
    fn json_schema_false_is_rejected() {
        assert_eq!(
            parse(json!({"type": "json_schema", "json_schema": {"schema": false}})),
            Err("response_format.json_schema.schema is false: no output can match it".into())
        );
    }

    #[test]
    fn non_object_is_rejected() {
        for v in [json!("json_object"), json!(1), json!([]), json!(true)] {
            assert_eq!(parse(v), Err("response_format must be an object".into()));
        }
    }

    #[test]
    fn unknown_or_missing_type_is_rejected() {
        let msg = "response_format.type must be one of text, json_object, json_schema";
        assert_eq!(parse(json!({})), Err(msg.into()));
        assert_eq!(parse(json!({"type": 3})), Err(msg.into()));
        let err = parse(json!({"type": "xml"})).unwrap_err();
        assert!(err.starts_with(msg), "{err}");
        assert!(err.contains("\"xml\""), "{err}");
    }

    #[test]
    fn json_schema_not_object_is_rejected() {
        let msg = "response_format.json_schema must be an object";
        assert_eq!(parse(json!({"type": "json_schema"})), Err(msg.into()));
        assert_eq!(
            parse(json!({"type": "json_schema", "json_schema": "x"})),
            Err(msg.into())
        );
    }

    #[test]
    fn schema_not_object_is_rejected() {
        let msg = "response_format.json_schema.schema must be a JSON Schema object";
        for s in [json!("object"), json!(1), json!([])] {
            assert_eq!(
                parse(json!({"type": "json_schema", "json_schema": {"schema": s}})),
                Err(msg.into())
            );
        }
    }

    fn spec_of(body: &str) -> GrammarSpec {
        GrammarSpec {
            kind: GrammarKind::JsonSchema,
            body: body.to_owned(),
        }
    }

    /// `json_object`'s grammar is the same constant every time: no build, no
    /// permit, even when none is free.
    #[tokio::test]
    async fn json_object_never_takes_a_compile_permit() {
        let check = SchemaCompileCheck::new(0, Duration::from_millis(10), Duration::from_secs(2));
        assert_eq!(check.check(&spec_of(JSON_OBJECT_SCHEMA)).await, Ok(()));
    }

    /// Agents resend the same schema every step: it is built once, and a
    /// passed schema never waits for a permit again. A schema not seen
    /// before still does.
    #[tokio::test]
    async fn a_repeated_schema_is_built_once() {
        let check = SchemaCompileCheck::new(1, Duration::from_millis(10), Duration::from_secs(2));
        let schema = spec_of(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#);
        assert_eq!(check.check(&schema).await, Ok(()));
        let _all = check.permits.clone().acquire_owned().await.unwrap();
        assert_eq!(check.check(&schema).await, Ok(()), "cached verdict");
        let other = spec_of(r#"{"type":"object","properties":{"b":{"type":"integer"}}}"#);
        assert_eq!(
            check.check(&other).await,
            Err(SchemaCheckError::Busy),
            "a new schema still needs a permit"
        );
    }

    /// The pass cache is bounded: the least recently used schema leaves
    /// first.
    #[test]
    fn the_pass_cache_evicts_the_least_recently_used() {
        let mut cache = PassedSchemas::new(2);
        cache.insert("a");
        cache.insert("b");
        assert!(cache.contains("a"));
        cache.insert("c");
        assert!(!cache.contains("b"), "b was the least recently used");
        assert!(cache.contains("a") && cache.contains("c"));
    }

    /// A schema that ran past the budget is tried again next time (the
    /// machine may just have been busy), not refused from memory.
    #[tokio::test]
    async fn a_schema_over_the_budget_is_not_remembered() {
        let check = SchemaCompileCheck::new(2, Duration::from_millis(500), Duration::from_nanos(1));
        let schema = spec_of(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#);
        assert_eq!(
            check.check(&schema).await,
            Err(SchemaCheckError::TooComplex)
        );
        assert!(!check.passed.lock().unwrap().contains(&schema.body));
    }

    /// 600 optional properties: past a row cap of 2000 (llguidance's
    /// default) on the first token, though the schema builds and gives its
    /// first mask.
    fn wide_optional_object() -> GrammarSpec {
        let props: Vec<String> = (0..600).map(|i| format!("\"p{i}\":{{}}")).collect();
        spec_of(&format!(
            r#"{{"type":"object","properties":{{{}}}}}"#,
            props.join(",")
        ))
    }

    /// A grammar that builds but fails on its first token is refused before
    /// any engine sees it, and not remembered as a pass.
    #[tokio::test]
    async fn a_grammar_that_fails_on_its_first_token_is_refused() {
        let check = SchemaCompileCheck::default()
            .with_factory(GrammarFactory::byte_level_with_row_cap_for_tests(2000));
        let schema = wide_optional_object();
        assert_eq!(
            check.check(&schema).await,
            Err(SchemaCheckError::TooComplex)
        );
        assert!(!check.passed.lock().unwrap().contains(&schema.body));
        // Under the limits engines use, it passes.
        assert_eq!(SchemaCompileCheck::default().check(&schema).await, Ok(()));
    }

    /// A check the request stopped waiting for stops its first-step masks
    /// and gives its slot back: the next schema gets it without waiting out
    /// the abandoned one.
    #[tokio::test]
    async fn an_abandoned_check_gives_its_slot_back() {
        let check = SchemaCompileCheck::new(1, Duration::from_secs(10), Duration::from_nanos(1));
        let schema = wide_optional_object();
        assert_eq!(
            check.check(&schema).await,
            Err(SchemaCheckError::TooComplex)
        );
        let freed = tokio::time::timeout(
            Duration::from_secs(10),
            check.permits.clone().acquire_owned(),
        )
        .await;
        assert!(freed.is_ok(), "the abandoned check still holds its slot");
    }

    /// A schema llguidance can't build is still left to the engine, whose
    /// refusal names the reason: the check passes it, as before, and the
    /// first-token step never runs on it.
    #[tokio::test]
    async fn a_schema_that_does_not_build_is_left_to_the_engine() {
        let body = r#"{"type":"string","enum":[1,2]}"#;
        assert!(GrammarFactory::byte_level()
            .compile(&spec_of(body))
            .is_err());
        let check = SchemaCompileCheck::default()
            .with_factory(GrammarFactory::byte_level_with_row_cap_for_tests(2000));
        assert_eq!(check.check(&spec_of(body)).await, Ok(()));
    }

    /// A build that panicked proved nothing about the schema: it is neither
    /// a pass nor remembered as one, so the engine never compiles it on the
    /// strength of this check.
    #[tokio::test]
    async fn a_build_that_panicked_is_not_a_pass() {
        let build = tokio::task::spawn_blocking(|| panic!("grammar build blew up"));
        let joined = tokio::time::timeout(Duration::from_secs(5), build).await;
        assert_eq!(build_verdict(joined), Err(SchemaCheckError::TooComplex));
        let done = tokio::time::timeout(
            Duration::from_secs(5),
            tokio::task::spawn_blocking(|| Ok(())),
        )
        .await;
        assert_eq!(build_verdict(done), Ok(()));
    }

    #[test]
    fn grammar_for_maps_each_variant() {
        assert_eq!(grammar_for(&ResponseFormat::Text), None);

        let g = grammar_for(&ResponseFormat::JsonObject).unwrap();
        assert_eq!(g.kind, GrammarKind::JsonSchema);
        assert_eq!(g.body, JSON_OBJECT_SCHEMA);

        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let g = grammar_for(&schema_variant(None, schema.clone(), false)).unwrap();
        assert_eq!(g.kind, GrammarKind::JsonSchema);
        assert!(
            !g.body.contains(' ') && !g.body.contains('\n'),
            "{}",
            g.body
        );
        let back: serde_json::Value = serde_json::from_str(&g.body).unwrap();
        assert_eq!(back, schema);
    }

    #[test]
    fn unenforced_keywords_finds_nested_and_ignores_property_names() {
        let schema = json!({
            "type": "object",
            "properties": {"not": {"type": "string"}, "b": {"$ref": "#/$defs/X"}},
            "$defs": {"X": {"allOf": [{"type": "object"}, {"required": ["a"]}]}}
        });
        assert_eq!(unenforced_keywords(&schema), vec!["allOf"]);

        // Draft-07 `dependencies` maps property names to schemas or name lists.
        let schema = json!({
            "type": "object",
            "dependencies": {"not": {"required": ["x"]}, "if": ["y"]}
        });
        assert!(unenforced_keywords(&schema).is_empty());

        let schema = json!({"type": "array", "uniqueItems": false});
        assert!(unenforced_keywords(&schema).is_empty());

        let schema = json!({"allOf": [{"type": "object"}]});
        assert!(unenforced_keywords(&schema).is_empty());

        // Sorted and de-duplicated across nesting.
        let schema = json!({
            "type": "array",
            "uniqueItems": true,
            "items": {"not": {"type": "null"}, "type": "object"},
            "contains": {"not": {"type": "string"}}
        });
        assert_eq!(
            unenforced_keywords(&schema),
            vec!["contains", "not", "uniqueItems"]
        );
    }

    #[test]
    fn unenforced_keywords_flags_siblings_but_not_annotations() {
        const SIB: &str = "siblings of $ref/const/enum/anyOf/oneOf/allOf";
        assert!(unenforced_keywords(
            &json!({"$ref": "#/$defs/A", "description": "d", "title": "t"})
        )
        .is_empty());
        assert_eq!(
            unenforced_keywords(&json!({"$ref": "#/$defs/A", "type": "object"})),
            vec![SIB]
        );
        assert_eq!(
            unenforced_keywords(
                &json!({"anyOf": [{"type": "string"}, {"type": "null"}], "type": "string"})
            ),
            vec![SIB]
        );
        assert_eq!(
            unenforced_keywords(&json!({"type": "number", "multipleOf": 2})),
            vec!["multipleOf"]
        );
        assert!(unenforced_keywords(&json!({"enum": ["a"], "type": "string"})).is_empty());
        assert!(unenforced_keywords(&json!({"const": 1, "type": "integer"})).is_empty());
        // Two of the dispatch keys together count.
        assert_eq!(
            unenforced_keywords(&json!({"enum": ["a"], "const": "a"})),
            vec![SIB]
        );
    }

    fn parse_schema(schema: serde_json::Value) -> Result<ResponseFormat, String> {
        parse(json!({"type": "json_schema", "json_schema": {"schema": schema}}))
    }

    #[test]
    fn schema_over_the_byte_limit_is_rejected() {
        let big = json!({"type": "string", "description": "x".repeat(70_000)});
        let n = serde_json::to_string(&big).unwrap().len();
        assert!(n > MAX_SCHEMA_BYTES);
        assert_eq!(
            parse_schema(big),
            Err(format!(
                "response_format.json_schema.schema is {n} bytes; the limit is 65536"
            ))
        );

        let fits = json!({"type": "string", "description": "x".repeat(60_000)});
        assert!(matches!(
            parse_schema(fits),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    /// `levels` objects nested inside each other through `items`.
    fn nested(levels: usize) -> serde_json::Value {
        let mut v = json!({"type": "string"});
        for _ in 1..levels {
            v = json!({"type": "array", "items": v});
        }
        v
    }

    #[test]
    fn schema_nested_too_deep_is_rejected() {
        assert_eq!(
            parse_schema(nested(65)),
            Err("response_format.json_schema.schema nests deeper than 64 levels".into())
        );
        assert!(matches!(
            parse_schema(nested(64)),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
        // Arrays count as levels too.
        let mut v = json!([1]);
        for _ in 1..64 {
            v = json!([v]);
        }
        assert!(parse_schema(json!({"enum": v})).is_err());
    }

    #[test]
    fn large_repetition_counts_are_rejected_at_any_depth() {
        for k in [
            "minItems",
            "maxItems",
            "minLength",
            "maxLength",
            "minProperties",
            "maxProperties",
        ] {
            let mut top = json!({"type": "object"});
            top[k] = json!(10_001);
            assert_eq!(
                parse_schema(top),
                Err(format!(
                    "response_format.json_schema.schema: {k} = 10001 exceeds 10000"
                ))
            );
            let mut deep = json!({"type": "string"});
            deep[k] = json!(100_000);
            let deep = json!({
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {"anyOf": [{"type": "null"}, deep]}}
            });
            let err = parse_schema(deep).unwrap_err();
            assert!(err.contains(k) && err.contains("100000"), "{err}");
        }
        assert_eq!(
            parse_schema(json!({"type": "string", "maxLength": 1e6})),
            Err("response_format.json_schema.schema: maxLength = 1000000.0 exceeds 10000".into())
        );
        assert!(matches!(
            parse_schema(json!({"type": "string", "maxLength": 10_000})),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    #[test]
    fn count_keywords_must_be_plain_non_negative_integers() {
        let keys = [
            "minItems",
            "maxItems",
            "minLength",
            "maxLength",
            "minProperties",
            "maxProperties",
            "minContains",
            "maxContains",
        ];
        for k in keys {
            for v in ["0.0", "0e0", "0E+0", "-0", "1.5", "-1", r#""0""#, "null"] {
                let text = format!(
                    r#"{{"type": "json_schema", "json_schema": {{"schema": {{"type": "object", "properties": {{"a": {{"type": "array", "{k}": {v}}}}}}}}}}}"#
                );
                assert_eq!(
                    parse_response_format(Some(&text)),
                    Err(format!(
                        "response_format.json_schema.schema: {k} must be a non-negative integer"
                    )),
                    "{k}: {v}"
                );
            }
            for v in ["0", "10"] {
                let text = format!(
                    r#"{{"type": "json_schema", "json_schema": {{"schema": {{"type": "array", "{k}": {v}}}}}}}"#
                );
                assert!(parse_response_format(Some(&text)).is_ok(), "{k}: {v}");
            }
        }
    }

    #[test]
    fn count_keyword_names_in_property_maps_are_not_counts() {
        let schema = json!({
            "type": "object",
            "properties": {
                "maxItems": {"type": "integer"},
                "minLength": {"type": "string"},
                "pattern": {"type": "string"}
            },
            "required": ["maxItems"],
            "dependencies": {"maxItems": ["minLength"]}
        });
        assert!(matches!(
            parse_schema(schema),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    #[test]
    fn counts_inside_data_are_checked_because_ref_can_reach_them() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"default": {"type": "array", "minItems": 1_000_000}},
                "b": {"$ref": "#/properties/a/default"}
            }
        });
        assert_eq!(
            parse_schema(schema),
            Err("response_format.json_schema.schema: minItems = 1000000 exceeds 10000".into())
        );
        for data in [
            json!({"const": {"maxLength": 99_999}}),
            json!({"enum": [{"maxItems": 99_999}]}),
            json!({"examples": [{"minLength": 99_999}]}),
        ] {
            assert!(parse_schema(data).is_err());
        }
    }

    #[test]
    fn large_pattern_quantifiers_are_rejected() {
        let err = |n: &str| {
            Err(format!(
                "response_format.json_schema.schema: pattern repetition {{{n}}} exceeds 10000"
            ))
        };
        assert_eq!(
            parse_schema(json!({"type": "string", "pattern": "^a{100000}$"})),
            err("100000")
        );
        assert_eq!(
            parse_schema(json!({"type": "string", "pattern": "^a{3,20000}$"})),
            err("20000")
        );
        assert_eq!(
            parse_schema(json!({"type": "string", "pattern": "^a{10001,}$"})),
            err("10001")
        );
        assert_eq!(
            parse_schema(json!({"type": "string", "pattern": "^a{,100000}$"})),
            err("100000")
        );
        assert_eq!(
            parse_schema(json!({
                "type": "object",
                "patternProperties": {"^x{20000}$": {"type": "string"}}
            })),
            err("20000")
        );
        for ok in [
            "^[a-z]{2,64}$",
            "\\{99999\\}",
            "a{,10000}",
            "{x}",
            "a{5",
            "{",
        ] {
            assert!(
                matches!(
                    parse_schema(json!({"type": "string", "pattern": ok})),
                    Ok(ResponseFormat::JsonSchema { .. })
                ),
                "{ok}"
            );
        }
    }

    /// An object with `n` string properties, each with `field` as its schema.
    fn object_of(n: usize, field: serde_json::Value) -> serde_json::Value {
        let props: serde_json::Map<String, serde_json::Value> =
            (0..n).map(|i| (format!("p{i}"), field.clone())).collect();
        json!({"type": "object", "properties": props})
    }

    #[test]
    fn repetition_counts_are_capped_in_total() {
        let ten_k = json!({"type": "string", "maxLength": 10_000});
        assert_eq!(
            parse_schema(object_of(21, ten_k.clone())),
            Err("response_format.json_schema.schema: repetition counts add up to 210000; the limit is 200000".into())
        );
        assert!(parse_schema(object_of(20, ten_k)).is_ok());
        // An extraction schema with 21 long text fields fits.
        assert!(parse_schema(object_of(21, json!({"type": "string", "maxLength": 5000}))).is_ok());
        // Pattern quantifiers count toward the total too, and a min/max pair
        // counts once, at the larger of the two.
        // 20000 + 19001 + 3 * 10000 + 13 * 10000 = 199001.
        let mut schema = object_of(13, json!({"type": "string", "maxLength": 10_000}));
        let props = schema["properties"].as_object_mut().unwrap();
        for (k, v) in [
            (
                "a",
                json!({"type": "string", "maxLength": 10_000, "pattern": "^.{0,10000}$"}),
            ),
            (
                "b",
                json!({"type": "array", "maxItems": 10_000, "items": {"pattern": "^(x{9000}){1}$"}}),
            ),
            (
                "c",
                json!({"type": "string", "minLength": 10_000, "maxLength": 10_000}),
            ),
            (
                "d",
                json!({"type": "array", "minItems": 10_000, "maxItems": 10_000}),
            ),
            (
                "e",
                json!({"type": "object", "minProperties": 10_000, "maxProperties": 10_000}),
            ),
        ] {
            props.insert(k.into(), v);
        }
        assert!(parse_schema(schema.clone()).is_ok());
        schema["properties"]["f"] = json!({"type": "string", "pattern": "^a{1000}$"});
        assert_eq!(
            parse_schema(schema),
            Err("response_format.json_schema.schema: repetition counts add up to 200001; the limit is 200000".into())
        );
    }

    #[test]
    fn a_min_max_pair_counts_its_larger_value() {
        // 33 * 6000 = 198000 fits only when each pair counts 6000, not 11000.
        let pair = json!({"type": "string", "minLength": 5000, "maxLength": 6000});
        assert!(parse_schema(object_of(33, pair.clone())).is_ok());
        assert_eq!(
            parse_schema(object_of(34, pair)),
            Err("response_format.json_schema.schema: repetition counts add up to 204000; the limit is 200000".into())
        );
        // Different families on one object still add up.
        let mixed = json!({"type": "array", "minItems": 6000, "maxLength": 6000});
        assert_eq!(
            parse_schema(object_of(17, mixed)),
            Err("response_format.json_schema.schema: repetition counts add up to 204000; the limit is 200000".into())
        );
    }

    /// A chain of `$ref` aliases, `d0` to `d{last}`, entered from the root.
    fn alias_chain(last: usize) -> serde_json::Value {
        let mut defs = serde_json::Map::new();
        for i in 0..last {
            defs.insert(
                format!("d{i}"),
                json!({"$ref": format!("#/$defs/d{}", i + 1)}),
            );
        }
        defs.insert(format!("d{last}"), json!({"type": "string"}));
        json!({"$defs": defs, "$ref": "#/$defs/d0"})
    }

    const REF_DEPTH_ERR: &str =
        "response_format.json_schema.schema nests deeper than 64 levels through $ref";

    #[test]
    fn a_long_ref_alias_chain_is_rejected() {
        let schema = alias_chain(2000);
        assert!(schema.to_string().len() <= MAX_SCHEMA_BYTES);
        assert_eq!(parse_schema(schema), Err(REF_DEPTH_ERR.into()));
    }

    #[test]
    fn ref_depth_counts_each_link_as_a_level() {
        // The root is level 1 and `d{i}` level 2 + i, so `d62` is at 64.
        assert!(matches!(
            parse_schema(alias_chain(62)),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
        assert_eq!(parse_schema(alias_chain(63)), Err(REF_DEPTH_ERR.into()));
    }

    #[test]
    fn a_huge_ref_alias_chain_is_rejected_without_deep_recursion() {
        // Far over the byte limit, so checked directly.
        let text = alias_chain(100_000).to_string();
        // Run on a small stack: the walk must not recurse per link.
        let result = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let root = Node::parse(serde_json::from_str(&text).unwrap()).unwrap();
                check_converter_depth(&root)
            })
            .unwrap()
            .join()
            .unwrap();
        // So large that the walk's budget may run out before the depth
        // limit is reached; refused either way.
        let complex = "response_format.json_schema.schema is too complex".to_string();
        assert!(
            result == Err(REF_DEPTH_ERR.into()) || result == Err(complex),
            "{result:?}"
        );
    }

    #[test]
    fn a_recursive_schema_passes_the_ref_depth_check() {
        let tree = json!({
            "$defs": {"node": {"type": "object", "properties": {
                "name": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            }}},
            "$ref": "#/$defs/node",
        });
        assert!(matches!(
            parse_schema(tree),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
        let self_ref = json!({"type": "object", "properties": {"next": {"$ref": "#"}}});
        assert!(matches!(
            parse_schema(self_ref),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    /// `levels` definitions, each an `anyOf` of `fan` refs to the next.
    fn ref_diamond(levels: usize, fan: usize) -> serde_json::Value {
        let mut defs = serde_json::Map::new();
        for i in 0..levels {
            let next = json!({"$ref": format!("#/$defs/d{}", i + 1)});
            defs.insert(format!("d{i}"), json!({"anyOf": vec![next; fan]}));
        }
        defs.insert(format!("d{levels}"), json!({"type": "string"}));
        json!({"$defs": defs, "$ref": "#/$defs/d0"})
    }

    #[test]
    fn a_ref_diamond_is_checked_in_linear_time() {
        let start = std::time::Instant::now();
        // Each level is two deep (the `anyOf`, then the ref in it), and a
        // ref entered once is not entered again: `d{i}` is at level 2 + 2i.
        assert!(matches!(
            parse_schema(ref_diamond(31, 50)),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
        assert_eq!(parse_schema(ref_diamond(32, 2)), Err(REF_DEPTH_ERR.into()));
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
    }

    /// `n` alias definitions `{prefix}0` .. `{prefix}{n-1}`, each a `$ref` to
    /// the next, the last one a `$ref` to `next_of_last`.
    fn aliases(defs: &mut serde_json::Map<String, Value>, prefix: &str, n: usize, last: &str) {
        for i in 0..n {
            let next = if i + 1 < n {
                format!("#/$defs/{prefix}{}", i + 1)
            } else {
                last.to_string()
            };
            defs.insert(format!("{prefix}{i}"), json!({"$ref": next}));
        }
    }

    #[test]
    fn a_long_ref_cycle_entered_from_two_properties_is_rejected() {
        let mut defs = serde_json::Map::new();
        aliases(&mut defs, "d", 101, "#/$defs/d0");
        let defs = Value::Object(defs).to_string();
        // Raw order "b" before "a"; sorted order is the other way round.
        let schema = format!(
            r##"{{"type":"object","properties":{{"b":{{"$ref":"#/$defs/d50"}},"a":{{"$ref":"#/$defs/d0"}}}},"$defs":{defs}}}"##
        );
        let format = format!(r#"{{"type":"json_schema","json_schema":{{"schema":{schema}}}}}"#);
        assert_eq!(
            parse_response_format(Some(&format)),
            Err(REF_DEPTH_ERR.into())
        );
    }

    #[test]
    fn a_ref_cycle_does_not_hide_a_deep_branch_behind_it() {
        // A and B refer to each other; A also leads into a chain of 58
        // aliases. Entered at B, the path B -> A -> chain is two levels
        // deeper than the path entered at A, and over the limit.
        let mut defs = serde_json::Map::new();
        defs.insert(
            "A".into(),
            json!({"anyOf": [{"$ref": "#/$defs/B"}, {"$ref": "#/$defs/D0"}]}),
        );
        defs.insert(
            "B".into(),
            json!({"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}),
        );
        aliases(&mut defs, "D", 58, "#/$defs/end");
        defs.insert("end".into(), json!({"type": "string"}));
        let entered_at = |def: &str| {
            json!({
                "$defs": defs,
                "type": "object",
                "properties": {"x": {"$ref": format!("#/$defs/{def}")}},
            })
        };
        assert_eq!(parse_schema(entered_at("B")), Err(REF_DEPTH_ERR.into()));
        assert!(parse_schema(entered_at("A")).is_ok());
    }

    #[test]
    fn a_mutually_recursive_pair_passes() {
        let schema = json!({
            "$defs": {
                "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
                "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
            },
            "$ref": "#/$defs/A",
        });
        assert!(matches!(
            parse_schema(schema),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    /// `levels` objects, each listing `object` ten times in `type` (so the
    /// converter visits the next level ten times per visit of this one),
    /// with `extra` added to each, around `inner`.
    fn revisited(levels: usize, extra: &Value, inner: Value) -> Value {
        let mut schema = inner;
        for _ in 0..levels {
            let mut level = json!({"type": vec!["object"; 10], "properties": {"a": schema}});
            for (k, v) in extra.as_object().unwrap() {
                level[k] = v.clone();
            }
            schema = level;
        }
        schema
    }

    #[test]
    fn a_long_ref_visited_many_times_is_resolved_once() {
        let name = "n".repeat(20_000);
        let r = format!("#/$defs/{name}");
        let mut schema = revisited(5, &json!({}), json!({"$ref": r}));
        schema["$defs"] = json!({name: {"type": "string"}});
        assert!(schema.to_string().len() <= MAX_SCHEMA_BYTES);
        let start = std::time::Instant::now();
        assert!(parse_schema(schema).is_ok());
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn long_scalars_visited_many_times_are_read_once() {
        let long = "x".repeat(9_000);
        for extra in [
            json!({"additionalProperties": long}),
            json!({"unevaluatedProperties": long}),
        ] {
            let schema = revisited(5, &extra, json!({"type": long}));
            assert!(schema.to_string().len() <= MAX_SCHEMA_BYTES);
            let start = std::time::Instant::now();
            let _ = parse_schema(schema);
            assert!(
                start.elapsed() < std::time::Duration::from_secs(2),
                "{extra:.20}"
            );
        }
    }

    #[test]
    fn the_converter_reads_some_pointers_differently_from_rfc_6901() {
        let root = Node::parse(
            serde_json::from_str(r#"{"$defs":{"a":{"type":"string"},"a~1b":1,"a/b":2},"l":[3]}"#)
                .unwrap(),
        )
        .unwrap();
        let text = |n: Option<&Node>| n.map(node_text);
        let both = |p: &str| {
            let mut scanned = 0;
            (
                text(rfc6901_target(&root, p, &mut scanned)),
                text(converter_target(&root, p, &mut scanned)),
            )
        };
        let s = Some(r#"{"type":"string"}"#.to_string());
        assert_eq!(both("/$defs/a"), (s.clone(), s.clone()));
        assert_eq!(both("/$defs//a"), (None, s.clone()));
        assert_eq!(both("/$defs/a/"), (None, s));
        assert_eq!(both("/$defs/a~1b"), (Some("2".into()), Some("1".into())));
        assert_eq!(both("/l/0"), (Some("3".into()), None));
        assert_eq!(both(""), (text(Some(&root)), text(Some(&root))));
    }

    /// `"response_format.json_schema.schema: unsupported $ref \"<r>\""`.
    fn unsupported_ref(r: &str) -> Result<ResponseFormat, String> {
        Err(format!(
            "response_format.json_schema.schema: unsupported $ref {}",
            Value::from(r)
        ))
    }

    /// A chain of `n` aliases whose names and refs `spell` writes, entered
    /// from the root.
    fn spelled_chain(
        n: usize,
        name: impl Fn(usize) -> String,
        r: impl Fn(usize) -> String,
    ) -> Value {
        let mut defs = serde_json::Map::new();
        for i in 0..n {
            defs.insert(name(i), json!({"$ref": r(i + 1)}));
        }
        defs.insert(name(n), json!({"type": "string"}));
        json!({"$defs": defs, "$ref": r(0)})
    }

    #[test]
    fn local_refs_that_are_not_plain_pointers_are_rejected() {
        // The grammar converter skips empty segments and does not unescape
        // `~1`, so each of these chains resolves there and is as deep as the
        // plain one.
        let empty_segment = spelled_chain(1900, |i| format!("d{i}"), |i| format!("#/$defs//d{i}"));
        assert_eq!(parse_schema(empty_segment), unsupported_ref("#/$defs//d0"));
        let trailing = spelled_chain(1900, |i| format!("d{i}"), |i| format!("#/$defs/d{i}/"));
        assert_eq!(parse_schema(trailing), unsupported_ref("#/$defs/d0/"));
        let tilde = spelled_chain(1700, |i| format!("d~1{i}"), |i| format!("#/$defs/d~1{i}"));
        assert_eq!(parse_schema(tilde), unsupported_ref("#/$defs/d~10"));
        let defs =
            json!({"a b": {"type": "string"}, "a/b": {"type": "string"}, "x": {"type": "string"}});
        for r in [
            "#/$defs/a%20b",
            "#/$defs/a~1b",
            "#/",
            "#anchor",
            "#/$defs/missing",
            "#/$defs/x/type/nope",
            "#/properties/a/anyOf/0",
        ] {
            let schema = json!({
                "$defs": defs,
                "type": "object",
                "properties": {
                    "a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "b": {"$ref": r}
                }
            });
            assert_eq!(parse_schema(schema), unsupported_ref(r), "{r}");
        }
        // The message quotes at most 80 characters of the ref.
        let long = format!("#/$defs/{}", "y".repeat(200));
        let schema = json!({"type": "object", "properties": {"a": {"$ref": long}}});
        assert_eq!(parse_schema(schema), unsupported_ref(&long[..80]));
    }

    #[test]
    fn plain_local_refs_pass() {
        for (r, target) in [
            ("#/$defs/X", "$defs"),
            ("#/definitions/X", "definitions"),
            ("#/properties/a", "properties"),
        ] {
            let mut schema = json!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"$ref": r}}
            });
            if target != "properties" {
                schema[target] = json!({"X": {"type": "integer"}});
            }
            assert!(parse_schema(schema).is_ok(), "{r}");
        }
        let schema = json!({"type": "object", "properties": {"a": {"$ref": "#"}}});
        assert!(parse_schema(schema).is_ok());
    }

    /// A ref to a document outside the schema is refused for every engine
    /// (no backend fetches one); refs the schema itself answers pass,
    /// through an embedded `$id` (TypeBox) or a fragment `$id`
    /// (jsonschema.net) too. What a passing ref reaches is checked for an
    /// integer whose build aborts in the grammar crate, whatever the ref.
    #[test]
    fn refs_to_documents_outside_the_schema_are_rejected() {
        for (schema, named) in [
            (
                json!({"type": "object", "properties": {"a": {"$ref": "https://example.com/s.json"}}}),
                "https://example.com/s.json",
            ),
            (
                json!({"type": "object", "properties": {"a": {"$ref": "s.json#/$defs/x~0y"}}}),
                "json-schema:///s.json",
            ),
            (
                json!({"type": "object", "properties": {"a": {"$ref": "urn:z#/default/x"}}}),
                "\"urn:z\"",
            ),
        ] {
            let err = parse_schema(schema.clone()).expect_err(&schema.to_string());
            assert!(
                err.starts_with("response_format.json_schema.schema: $ref ")
                    && err.contains(named)
                    && err.contains("outside the schema"),
                "{schema}: {err}"
            );
        }
        for schema in [
            json!({
                "$id": "https://example.com/owner.json",
                "type": "object",
                "properties": {"id": {"type": "string"}, "$id": {"type": "string"}, "pet": {"$ref": "#/$defs/Pet"}},
                "$defs": {"Pet": {"type": "integer"}},
            }),
            json!({"$id": "urn:x", "type": "object", "properties": {"n": {"$ref": "urn:x#/$defs/N"}}, "$defs": {"N": {"type": "integer"}}}),
            json!({"$schema": "http://json-schema.org/draft-07/schema#", "$id": "http://example.com/example.json", "type": "object", "properties": {"id": {"$id": "#/properties/id", "type": "integer"}}}),
            json!({"$defs": {"T": {"$id": "T", "type": "string"}}, "type": "object", "properties": {"t": {"$ref": "T"}}}),
            json!({"type": "object", "x-meta": {"$id": "abc"}, "properties": {"a": {"type": "string"}}}),
        ] {
            assert!(parse_schema(schema.clone()).is_ok(), "{schema}");
        }
    }

    /// A `$ref` into `default` passes the checks here, and llguidance builds
    /// its target as a schema: the grammar crate clamps, bounds and checks
    /// that target too, so the build is a result or a clean error, never an
    /// abort. A `$ref` into `examples` is refused here.
    #[test]
    fn a_ref_into_a_data_keyword_reaches_the_build_and_ends_cleanly() {
        let factory = GrammarFactory::byte_level();
        for (kw, r) in [("default", "#/default/x"), ("examples", "#/examples/0")] {
            for (target, builds) in [
                (json!({"type": "integer", "minimum": -1e300}), true),
                (json!({"type": "integer"}), true),
                (
                    json!({"type": "integer", "minimum": -1e20, "maximum": -1e19}),
                    false,
                ),
                (json!({"type": "integer", "const": i64::MIN}), false),
            ] {
                let schema = json!({
                    "type": "object",
                    "properties": {"n": {"$ref": r}},
                    "required": ["n"],
                    kw: if kw == "default" { json!({"x": target}) } else { json!([target]) },
                });
                if kw == "examples" {
                    assert_eq!(parse_schema(schema), unsupported_ref(r), "{target}");
                    continue;
                }
                let rf = parse_schema(schema).unwrap_or_else(|e| panic!("{r} {target}: {e}"));
                let built = factory.compile(&grammar_for(&rf).unwrap());
                assert_eq!(built.is_ok(), builds, "{r} {target}: {:?}", built.err());
            }
        }
    }

    #[test]
    fn refs_to_values_that_are_not_schemas_are_rejected() {
        for r in ["#/required", "#/properties/a/type"] {
            let schema = json!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"$ref": r}},
                "required": ["a"]
            });
            assert_eq!(parse_schema(schema), unsupported_ref(r), "{r}");
        }
        // A boolean schema is a schema.
        let schema = json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/T"}},
            "$defs": {"T": true}
        });
        assert!(parse_schema(schema).is_ok());
    }

    /// `$defs` entries `{p}0` .. `{p}{n}`: each an array whose items refer
    /// to the next, the last a string. Each link is two converter levels.
    fn array_ref_chain(p: &str, n: usize) -> serde_json::Map<String, Value> {
        let mut defs = serde_json::Map::new();
        for i in 0..n {
            defs.insert(
                format!("{p}{i}"),
                json!({"type": "array", "items": {"$ref": format!("#/$defs/{p}{}", i + 1)}}),
            );
        }
        defs.insert(format!("{p}{n}"), json!({"type": "string"}));
        defs
    }

    #[test]
    fn property_names_are_walked_as_the_converter_visits_them() {
        // The converter adds `type: "string"` to `propertyNames` and keeps
        // its other keys, so a `$ref`, `anyOf` or `allOf` there is followed.
        let defs = array_ref_chain("D", 40);
        for names in [
            json!({"$ref": "#/$defs/D0"}),
            json!({"anyOf": [{"$ref": "#/$defs/D0"}]}),
            json!({"allOf": [{"$ref": "#/$defs/D0"}]}),
        ] {
            let schema = json!({"type": "object", "propertyNames": names, "$defs": defs});
            assert_eq!(parse_schema(schema), Err(REF_DEPTH_ERR.into()), "{names}");
        }
        let schema = json!({"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}});
        assert!(parse_schema(schema).is_ok());
    }

    #[test]
    fn an_object_with_max_properties_zero_enters_no_refs() {
        // The converter visits nothing in an object that holds no property,
        // so `x` must not enter R0 at a shallow level before `y` reaches it
        // 21 levels down.
        let defs = array_ref_chain("R", 25);
        let mut y = json!({"$ref": "#/$defs/R0"});
        for _ in 0..20 {
            y = json!({"type": "array", "items": y});
        }
        let control = json!({"type": "object", "properties": {"y": y}, "$defs": defs});
        assert_eq!(parse_schema(control), Err(REF_DEPTH_ERR.into()));
        for x in [
            json!({"type": "object", "maxProperties": 0, "properties": {"p": {"$ref": "#/$defs/R0"}}}),
            json!({"type": "object", "maxProperties": 0, "additionalProperties": {"$ref": "#/$defs/R0"}}),
            json!({"type": "object", "maxProperties": 0, "patternProperties": {"^a": {"$ref": "#/$defs/R0"}}}),
        ] {
            let schema = json!({
                "type": "object",
                "properties": {"x": x, "y": y},
                "$defs": defs
            });
            assert_eq!(parse_schema(schema), Err(REF_DEPTH_ERR.into()), "{x}");
        }
    }

    #[test]
    fn a_target_reached_by_nesting_is_counted_each_time_it_is_entered() {
        // A1 items A2 ... items A59 items N, N an anyOf of refs back to
        // A59 .. A1. The converter caches only what it entered through a
        // `$ref`, so each ref converts the nesting below its target again.
        let k = 59;
        let refs: Vec<Value> = (1..=k)
            .rev()
            .map(|i| json!({"$ref": format!("#{}", "/items".repeat(i - 1))}))
            .collect();
        let mut schema = json!({"anyOf": refs});
        for _ in 0..k {
            schema = json!({"type": "array", "items": schema});
        }
        assert_eq!(parse_schema(schema), Err(REF_DEPTH_ERR.into()));
    }

    /// A UI tree: `Node` is a `oneOf` of `types` component types, each with
    /// `children`, an array of `Node`.
    fn ui_tree(types: usize) -> Value {
        let mut defs = serde_json::Map::new();
        let mut refs = Vec::new();
        for i in 0..types {
            let name = format!("Comp{i}");
            defs.insert(
                name.clone(),
                json!({"type": "object", "title": name, "properties": {
                    "type": {"const": name},
                    "label": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
                }, "required": ["type", "children"]}),
            );
            refs.push(json!({"$ref": format!("#/$defs/{name}")}));
        }
        defs.insert("Node".into(), json!({"oneOf": refs}));
        json!({
            "$defs": defs,
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
            "required": ["root"]
        })
    }

    #[test]
    fn a_recursive_union_of_many_types_passes() {
        for types in [12, 16, 20, 30] {
            assert!(parse_schema(ui_tree(types)).is_ok(), "{types} types");
        }
    }

    #[test]
    fn recursive_unions_stay_far_below_the_backstop() {
        for types in [16, 20, 30] {
            let bound = ref_depth_bound(&ui_tree(types));
            eprintln!("ui tree with {types} types: backstop bound {bound}");
            assert!(bound <= MAX_REF_DEPTH_BOUND, "{types} types: {bound}");
        }
    }

    #[test]
    fn the_backstop_rejects_a_long_chain_in_either_spelling() {
        let err = Err(format!(
            "response_format.json_schema.schema may nest deeper than {MAX_REF_DEPTH_BOUND} levels through $ref"
        ));
        assert_eq!(check_ref_depth_bound(&alias_chain(600)), err);
        assert_eq!(check_ref_depth_bound(&alias_chain(400)), Ok(()));
        // Spellings only the converter resolves count as links too.
        let empty_segment = spelled_chain(600, |i| format!("d{i}"), |i| format!("#/$defs//d{i}"));
        assert_eq!(check_ref_depth_bound(&empty_segment), err);
        let tilde = spelled_chain(600, |i| format!("d~1{i}"), |i| format!("#/$defs/d~1{i}"));
        assert_eq!(check_ref_depth_bound(&tilde), err);
    }

    #[test]
    fn one_definition_used_many_times_passes() {
        let mut inner = json!({"type": "string"});
        for _ in 0..20 {
            inner = json!({"type": "object", "properties": {"x": inner}});
        }
        let mut schema = object_of(200, json!({"$ref": "#/$defs/X"}));
        schema["$defs"] = json!({"X": inner});
        assert!(parse_schema(schema).is_ok());
    }

    #[test]
    fn a_schema_the_converter_would_expand_without_end_hits_the_visit_budget() {
        // Each level lists `object` ten times in `type`: the converter
        // converts the level below once per entry, 10^8 times in all.
        let mut schema = json!({"type": "string"});
        for _ in 0..8 {
            schema = json!({"type": vec!["object"; 10], "properties": {"a": schema}});
        }
        // The same with a wide object at the bottom: each visit of it
        // costs its keys.
        let mut wide: serde_json::Map<String, Value> =
            (0..3000).map(|i| (format!("k{i}"), json!(0))).collect();
        wide.insert("type".into(), json!("object"));
        let mut wide = Value::Object(wide);
        for _ in 0..6 {
            wide = json!({"type": vec!["object"; 10], "properties": {"a": wide}});
        }
        assert!(wide.to_string().len() <= MAX_SCHEMA_BYTES);
        for schema in [schema, wide] {
            let start = std::time::Instant::now();
            assert_eq!(
                parse_schema(schema),
                Err("response_format.json_schema.schema is too complex".into())
            );
            assert!(start.elapsed() < std::time::Duration::from_secs(2));
        }
    }

    #[test]
    fn real_client_schemas_fit_the_limits() {
        for text in [
            include_str!("../../cascadia-engine-mock/tests/fixtures/browser_use_schema.json"),
            include_str!("../../cascadia-engine-mock/tests/fixtures/pydantic_output_schema.json"),
        ] {
            let schema: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(
                parse_schema(schema.clone()),
                Ok(schema_variant(None, schema, false))
            );
        }
    }

    /// The grammar body and checked schema `parse_response_format` gives for
    /// the schema text `schema`.
    fn parsed_schema_text(schema: &str) -> (String, serde_json::Value) {
        let format = format!(r#"{{"type":"json_schema","json_schema":{{"schema":{schema}}}}}"#);
        match parse_response_format(Some(&format)).unwrap() {
            ResponseFormat::JsonSchema { raw, schema, .. } => (raw, schema),
            other => panic!("expected a json_schema format, got {other:?}"),
        }
    }

    /// A zod-style intersection: `z.object({...}).and(z.object({...}))`.
    const INTERSECTION: &str = r#"{"description":"agent","allOf":[{"type":"object","properties":{"thinking":{"type":"string"},"goal":{"type":"string"}},"required":["thinking","goal"],"additionalProperties":false},{"type":"object","properties":{"action":{"type":"string"},"goal":{"type": "string"}},"required":["action","goal"]}]}"#;

    #[test]
    fn object_all_of_is_merged_into_one_object() {
        let (raw, schema) = parsed_schema_text(INTERSECTION);
        assert_eq!(
            raw,
            r#"{"description":"agent","type":"object","properties":{"thinking":{"type":"string"},"goal":{"type":"string"},"action":{"type":"string"}},"required":["thinking","goal","action"],"additionalProperties":false}"#
        );
        assert!(!raw.contains("allOf"));
        assert!(unenforced_keywords(&schema).is_empty(), "{schema}");
        let g = grammar_for(
            &parse_response_format(Some(&format!(
                r#"{{"type":"json_schema","json_schema":{{"schema":{INTERSECTION}}}}}"#
            )))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(g.body, raw);
    }

    #[test]
    fn all_of_defining_a_property_two_ways_is_left_and_flagged() {
        for schema in [
            r#"{"allOf":[{"type":"object","properties":{"goal":{"type":"string"}}},{"type":"object","properties":{"goal":{"type":"integer"}}}]}"#,
            r#"{"allOf":[{"properties":{"a":{"type":"string"}}},{"properties":{"b":{"type":"string"}}},{"properties":{"a":{"type":"string","maxLength":3}}}]}"#,
        ] {
            let (raw, parsed) = parsed_schema_text(schema);
            assert_eq!(raw, schema);
            assert!(unenforced_keywords(&parsed).contains(&"allOf"), "{schema}");
        }
    }

    #[test]
    fn all_of_repeating_a_property_the_same_way_is_merged() {
        let schema = r#"{"allOf":[{"properties":{"a":{"type":"string"}}},{"properties":{"a":{ "type" : "string" },"b":{"type":"integer"}}}]}"#;
        let (raw, parsed) = parsed_schema_text(schema);
        assert_eq!(
            raw,
            r#"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}}}"#
        );
        assert!(unenforced_keywords(&parsed).is_empty(), "{parsed}");
    }

    #[test]
    fn merged_all_of_accepts_what_the_original_accepts() {
        let original = r#"{"allOf":[{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},{"properties":{"b":{"type":"integer"}},"required":["b"]}]}"#;
        let (raw, _) = parsed_schema_text(original);
        assert!(!raw.contains("allOf"), "{raw}");
        let before = jsonschema::validator_for(&serde_json::from_str(original).unwrap()).unwrap();
        let after = jsonschema::validator_for(&serde_json::from_str(&raw).unwrap()).unwrap();
        for instance in [
            json!({"a": "x", "b": 1}),
            json!({"a": "x", "b": 1, "c": true}),
            json!({"a": "x"}),
            json!({"b": 1}),
            json!({"a": 1, "b": 1}),
            json!({"a": "x", "b": "y"}),
            json!([]),
            json!("a"),
        ] {
            assert_eq!(
                before.is_valid(&instance),
                after.is_valid(&instance),
                "{instance}"
            );
        }
    }

    #[test]
    fn all_of_with_a_ref_or_other_keywords_is_left_and_flagged() {
        for schema in [
            r##"{"allOf":[{"$ref":"#/$defs/A"},{"type":"object","properties":{"b":{"type":"string"}}}],"$defs":{"A":{"type":"object"}}}"##,
            r#"{"allOf":[{"type":"object","minProperties":1},{"type":"object"}]}"#,
            r#"{"allOf":[{"type":"string"},{"type":"object"}]}"#,
            r#"{"allOf":[{"type":"object","additionalProperties":{"type":"string"}},{"type":"object"}]}"#,
            r#"{"type":"object","properties":{"z":{"type":"string"}},"allOf":[{"type":"object"},{"type":"object"}]}"#,
        ] {
            let (raw, parsed) = parsed_schema_text(schema);
            assert_eq!(raw, schema);
            assert!(unenforced_keywords(&parsed).contains(&"allOf"), "{schema}");
        }
    }

    #[test]
    fn nested_all_of_in_properties_and_defs_is_merged() {
        let schema = r##"{"type":"object","properties":{"p":{"allOf":[{"properties":{"x":{"type":"string"}}},{"properties":{"y":{"type":"string"}}}]},"q":{"$ref":"#/$defs/D"}},"$defs":{"D":{"allOf":[{"required":["m"]},{"type":"object","required":["n","m"]}]}}}"##;
        let (raw, parsed) = parsed_schema_text(schema);
        assert_eq!(
            raw,
            r##"{"type":"object","properties":{"p":{"type":"object","properties":{"x":{"type":"string"},"y":{"type":"string"}}},"q":{"$ref":"#/$defs/D"}},"$defs":{"D":{"type":"object","required":["m","n"]}}}"##
        );
        assert!(unenforced_keywords(&parsed).is_empty(), "{parsed}");
    }

    #[test]
    fn schema_without_a_mergeable_all_of_keeps_its_exact_text() {
        for schema in [
            r#"{"type":"object","properties":{"z":{"type":"number","default":1.50},"a":{"type":"string","description":"café"}},"required":["z"]}"#,
            r#"{"allOf":[{"type":"object","properties":{"z":{"type":"string"}}}]}"#,
            r#"{"type":"object","properties":{"allOf":{"type":"string"}},"default":{"allOf":[{},{}]}}"#,
        ] {
            assert_eq!(parsed_schema_text(schema).0, schema);
        }
    }

    /// An object schema with `n` literal `patternProperties` keys.
    fn pattern_props(n: usize) -> serde_json::Value {
        let map: serde_json::Map<String, serde_json::Value> = (0..n)
            .map(|i| (format!("^k{i}$"), json!({"type": "integer"})))
            .collect();
        json!({"type": "object", "patternProperties": map})
    }

    #[test]
    fn many_pattern_properties_keys_are_rejected() {
        assert_eq!(
            parse_schema(pattern_props(500)),
            Err(
                "response_format.json_schema.schema: an object has 500 patternProperties keys; \
                 the limit is 64"
                    .into()
            )
        );
        assert!(matches!(
            parse_schema(pattern_props(64)),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    #[test]
    fn pattern_properties_keys_are_capped_in_total() {
        let three = json!({
            "type": "object",
            "properties": {"a": pattern_props(60), "b": pattern_props(60), "c": pattern_props(60)}
        });
        assert_eq!(
            parse_schema(three),
            Err(
                "response_format.json_schema.schema has 180 patternProperties keys; \
                 the limit is 128"
                    .into()
            )
        );
        let two = json!({
            "type": "object",
            "properties": {"a": pattern_props(64), "b": pattern_props(64)}
        });
        assert!(matches!(
            parse_schema(two),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    #[test]
    fn pattern_properties_merged_through_all_of_refs_count_each_time() {
        // Each allOf of two refs is merged anew by the grammar builder:
        // 40 merges of 64 + 64 keys took 2.6 s to build.
        let props: serde_json::Map<String, serde_json::Value> = (0..40)
            .map(|i| {
                (
                    format!("p{i}"),
                    json!({"allOf": [{"$ref": "#/$defs/P"}, {"$ref": "#/$defs/Q"}]}),
                )
            })
            .collect();
        let schema = json!({
            "type": "object",
            "properties": props,
            "$defs": {"P": pattern_props(64), "Q": pattern_props(64)}
        });
        let err = parse_schema(schema).unwrap_err();
        assert!(
            err.contains("patternProperties") && err.contains("128"),
            "{err}"
        );
    }

    #[test]
    fn pattern_properties_visited_once_per_type_list_entry_count_each_time() {
        let mut obj = pattern_props(64);
        obj["type"] = json!(["object", "object", "object"]);
        let err = parse_schema(json!({"type": "object", "properties": {"a": obj}})).unwrap_err();
        assert!(
            err.contains("patternProperties") && err.contains("128"),
            "{err}"
        );
    }

    #[test]
    fn a_ref_reused_many_times_counts_its_pattern_properties_once() {
        // The builder caches a ref's grammar by its text: one build.
        let props: serde_json::Map<String, serde_json::Value> = (0..40)
            .map(|i| (format!("p{i}"), json!({"$ref": "#/$defs/P"})))
            .collect();
        let schema =
            json!({"type": "object", "properties": props, "$defs": {"P": pattern_props(64)}});
        assert!(matches!(
            parse_schema(schema),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    #[test]
    fn pattern_text_is_capped_in_total() {
        let long = "a|".repeat(4100) + "b";
        let schema = json!({
            "type": "object",
            "properties": {
                "x": {"type": "string", "pattern": long},
                "y": {"type": "object", "patternProperties": {(long.clone()): {"type": "integer"}}}
            }
        });
        assert_eq!(
            parse_schema(schema),
            Err(
                "response_format.json_schema.schema: its patterns add up to 16402 bytes; \
                 the limit is 16384"
                    .into()
            )
        );
        let fits = json!({"type": "string", "pattern": "a|".repeat(4000) + "b"});
        assert!(matches!(
            parse_schema(fits),
            Ok(ResponseFormat::JsonSchema { .. })
        ));
    }

    /// The 400 message for an unsatisfiable bound pair.
    fn unsat(detail: &str) -> Result<ResponseFormat, String> {
        Err(format!("response_format.json_schema.schema: {detail}"))
    }

    #[test]
    fn min_length_above_max_length_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "string", "minLength": 5, "maxLength": 2})),
            unsat("minLength (5) is greater than maxLength (2) at #; no value can satisfy both")
        );
    }

    #[test]
    fn min_items_above_max_items_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "array", "minItems": 5, "maxItems": 2})),
            unsat("minItems (5) is greater than maxItems (2) at #; no value can satisfy both")
        );
    }

    #[test]
    fn min_properties_above_max_properties_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "object", "minProperties": 3, "maxProperties": 1})),
            unsat("minProperties (3) is greater than maxProperties (1) at #; no value can satisfy both")
        );
    }

    #[test]
    fn minimum_above_maximum_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "number", "minimum": 5, "maximum": 2.5})),
            unsat("minimum (5) is greater than maximum (2.5) at #; no value can satisfy both")
        );
        assert_eq!(
            parse_schema(json!({"type": "integer", "minimum": -1, "maximum": -2})),
            unsat("minimum (-1) is greater than maximum (-2) at #; no value can satisfy both")
        );
    }

    #[test]
    fn exclusive_minimum_not_below_exclusive_maximum_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "number", "exclusiveMinimum": 2, "exclusiveMaximum": 2})),
            unsat("exclusiveMinimum (2) is not less than exclusiveMaximum (2) at #; no value can satisfy both")
        );
        assert_eq!(
            parse_schema(json!({"type": "number", "exclusiveMinimum": 3, "exclusiveMaximum": 2})),
            unsat("exclusiveMinimum (3) is not less than exclusiveMaximum (2) at #; no value can satisfy both")
        );
    }

    #[test]
    fn minimum_not_below_exclusive_maximum_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "number", "minimum": 2, "exclusiveMaximum": 2})),
            unsat(
                "minimum (2) is not less than exclusiveMaximum (2) at #; no value can satisfy both"
            )
        );
    }

    #[test]
    fn exclusive_minimum_not_below_maximum_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "number", "exclusiveMinimum": 2, "maximum": 2})),
            unsat(
                "exclusiveMinimum (2) is not less than maximum (2) at #; no value can satisfy both"
            )
        );
    }

    #[test]
    fn unsatisfiable_bounds_are_found_nested() {
        // Inside `properties`, and the pointer escapes `/` and `~` per RFC 6901.
        assert_eq!(
            parse_schema(json!({
                "type": "object",
                "properties": {
                    "ok": {"type": "string"},
                    "a/b~c": {"type": "string", "minLength": 5, "maxLength": 2}
                }
            })),
            unsat("minLength (5) is greater than maxLength (2) at #/properties/a~1b~0c; no value can satisfy both")
        );
        // Inside `items`.
        assert_eq!(
            parse_schema(json!({
                "type": "array",
                "items": {"type": "array", "minItems": 4, "maxItems": 3}
            })),
            unsat(
                "minItems (4) is greater than maxItems (3) at #/items; no value can satisfy both"
            )
        );
        // Inside an `anyOf` branch and `prefixItems`.
        assert_eq!(
            parse_schema(json!({
                "anyOf": [
                    {"type": "null"},
                    {"type": "array", "prefixItems": [{"type": "number", "minimum": 1, "maximum": 0}]}
                ]
            })),
            unsat("minimum (1) is greater than maximum (0) at #/anyOf/1/prefixItems/0; no value can satisfy both")
        );
        // An `additionalProperties` schema.
        assert_eq!(
            parse_schema(json!({
                "type": "object",
                "additionalProperties": {"type": "object", "minProperties": 2, "maxProperties": 1}
            })),
            unsat("minProperties (2) is greater than maxProperties (1) at #/additionalProperties; no value can satisfy both")
        );
    }

    #[test]
    fn unsatisfiable_bounds_reached_through_ref_are_rejected() {
        assert_eq!(
            parse_schema(json!({
                "type": "object",
                "properties": {"name": {"$ref": "#/$defs/Name"}},
                "required": ["name"],
                "$defs": {"Name": {"type": "string", "minLength": 5, "maxLength": 2}}
            })),
            unsat("minLength (5) is greater than maxLength (2) at #/$defs/Name; no value can satisfy both")
        );
        assert_eq!(
            parse_schema(json!({
                "$ref": "#/definitions/N",
                "definitions": {"N": {"type": "number", "exclusiveMinimum": 1, "maximum": 1}}
            })),
            unsat("exclusiveMinimum (1) is not less than maximum (1) at #/definitions/N; no value can satisfy both")
        );
    }

    #[test]
    fn equal_and_open_bounds_are_accepted() {
        for schema in [
            json!({"type": "string", "minLength": 3, "maxLength": 3}),
            json!({"type": "array", "minItems": 2, "maxItems": 2}),
            json!({"type": "object", "minProperties": 1, "maxProperties": 1}),
            json!({"type": "number", "minimum": 2, "maximum": 2}),
            json!({"type": "number", "minimum": 1.5, "exclusiveMaximum": 2}),
            json!({"type": "number", "exclusiveMinimum": 1, "maximum": 1.5}),
            json!({"type": "number", "exclusiveMinimum": 1, "exclusiveMaximum": 1.5}),
            json!({"type": "string", "minLength": 5}),
            json!({"type": "array", "maxItems": 0}),
            json!({"type": "number", "minimum": 100}),
            json!({"type": "number", "exclusiveMaximum": -100}),
            // A property map whose names are keywords holds schemas, not bounds.
            json!({
                "type": "object",
                "properties": {"minLength": {"type": "integer"}, "maxLength": {"type": "integer"}}
            }),
        ] {
            assert!(
                matches!(
                    parse_schema(schema.clone()),
                    Ok(ResponseFormat::JsonSchema { .. })
                ),
                "{schema}"
            );
            assert_eq!(unsatisfiable_bound(&schema), None, "{schema}");
        }
    }

    /// `enum`, `const`, `default` and `examples` hold values, not schemas:
    /// bound-named keys inside them are data. A property map's entries are
    /// schemas whatever they are called, `default` included.
    #[test]
    fn bounds_inside_data_keywords_are_data() {
        for schema in [
            json!({"type": "object", "properties": {"range": {
                "type": "object",
                "properties": {"minimum": {"type": "integer"}, "maximum": {"type": "integer"}},
                "default": {"minimum": 10, "maximum": 0}
            }}}),
            json!({"enum": [{"minLength": 5, "maxLength": 2}]}),
            json!({"properties": {"s": {"const": {"type": "integer", "minimum": 2.5, "maximum": 2.9}}}}),
            json!({"type": "string", "examples": [{"minItems": 3, "maxItems": 1}]}),
        ] {
            assert_eq!(unsatisfiable_bound(&schema), None, "{schema}");
            assert!(parse_schema(schema.clone()).is_ok(), "{schema}");
        }
        assert_eq!(
            parse_schema(json!({"type": "object", "properties": {
                "default": {"type": "string", "minLength": 5, "maxLength": 2}
            }})),
            unsat("minLength (5) is greater than maxLength (2) at #/properties/default; no value can satisfy both")
        );
    }

    /// The members of an `allOf` all apply to one value: a lower bound in one
    /// and an upper bound in another can leave nothing. A pair inside one
    /// member keeps that member's pointer.
    #[test]
    fn a_pair_split_across_all_of_members_is_refused() {
        assert_eq!(
            parse_schema(json!({"allOf": [{"type": "string", "minLength": 5}, {"maxLength": 2}]})),
            unsat("minLength (5) at #/allOf/0 is greater than maxLength (2) at #/allOf/1; no value can satisfy both")
        );
        assert_eq!(
            parse_schema(
                json!({"type": "integer", "exclusiveMinimum": 2, "allOf": [{"exclusiveMaximum": 3}]})
            ),
            unsat(
                "no integer meets exclusiveMinimum (2) at # and exclusiveMaximum (3) at #/allOf/0"
            )
        );
        assert_eq!(
            parse_schema(json!({"type": "object", "allOf": [
                {"properties": {"a": {"type": "string", "minLength": 5}}},
                {"properties": {"a": {"maxLength": 2}}}
            ]})),
            unsat("minLength (5) at #/allOf/0/properties/a is greater than maxLength (2) at #/allOf/1/properties/a; no value can satisfy both")
        );
        assert_eq!(
            parse_schema(json!({"allOf": [{"minItems": 1}, {"minItems": 4, "maxItems": 3}]})),
            unsat(
                "minItems (4) is greater than maxItems (3) at #/allOf/1; no value can satisfy both"
            )
        );
        // Bounds that leave room across the members, and members that are
        // references, are not refused.
        for schema in [
            json!({"allOf": [{"minLength": 2}, {"maxLength": 5}]}),
            json!({"$defs": {"N": {"maxLength": 2}}, "allOf": [{"minLength": 5}, {"$ref": "#/$defs/N"}]}),
            // Beside a `$ref`, draft 7 ignores the siblings: its properties
            // are left out as well as its bounds.
            json!({"$defs": {"A": {}}, "allOf": [
                {"$ref": "#/$defs/A", "properties": {"a": {"minLength": 5}}},
                {"properties": {"a": {"maxLength": 2}}}
            ]}),
        ] {
            assert_eq!(unsatisfiable_bound(&schema), None, "{schema}");
        }
    }

    /// The members of a member's `allOf` apply to the value too, and so does
    /// `items` when several members define it.
    #[test]
    fn a_split_in_a_nested_all_of_or_in_shared_items_is_refused() {
        assert_eq!(
            parse_schema(json!({"allOf": [{"allOf": [{"minLength": 5}]}, {"maxLength": 2}]})),
            unsat("minLength (5) at #/allOf/0/allOf/0 is greater than maxLength (2) at #/allOf/1; no value can satisfy both")
        );
        assert_eq!(
            parse_schema(json!({"type": "array", "allOf": [
                {"items": {"type": "string", "minLength": 5}},
                {"items": {"maxLength": 2}}
            ]})),
            unsat("minLength (5) at #/allOf/0/items is greater than maxLength (2) at #/allOf/1/items; no value can satisfy both")
        );
        assert_eq!(
            parse_schema(json!({"allOf": [
                {"properties": {"n": {"type": "integer", "exclusiveMinimum": 2}}},
                {"properties": {"n": {"allOf": [{"exclusiveMaximum": 3}]}}}
            ]})),
            unsat("no integer meets exclusiveMinimum (2) at #/allOf/0/properties/n and exclusiveMaximum (3) at #/allOf/1/properties/n/allOf/0")
        );
    }

    /// Each `$ref` target the main walk skips is walked once, and a target
    /// inside another one is left to that one's walk, unless it sits under a
    /// data keyword there (which that walk skips too).
    #[test]
    fn ref_targets_are_walked_once_each_and_none_is_missed() {
        // Two refs to one target, and a ref to a target the main walk visits.
        let schema = json!({
            "properties": {
                "a": {"$ref": "#/default/x"},
                "b": {"$ref": "#/default/x"},
                "c": {"$ref": "#/properties/a"}
            },
            "default": {"x": {"minLength": 5, "maxLength": 2}}
        });
        assert_eq!(uncovered_ref_targets(&schema).len(), 1);
        assert_eq!(
            unsatisfiable_bound(&schema).as_deref(),
            Some("minLength (5) is greater than maxLength (2) at #/default/x; no value can satisfy both")
        );
        // A ref to each ancestor of one subtree: only the outermost is walked.
        let schema = json!({
            "properties": {
                "a": {"$ref": "#/default"},
                "b": {"$ref": "#/default/0"},
                "c": {"$ref": "#/default/0/0"}
            },
            "default": [[{"minItems": 4, "maxItems": 3}]]
        });
        let targets = uncovered_ref_targets(&schema);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].1, ["default"]);
        assert_eq!(
            unsatisfiable_bound(&schema).as_deref(),
            Some("minItems (4) is greater than maxItems (3) at #/default/0/0; no value can satisfy both")
        );
        // Under a data keyword inside another target: walked on its own.
        let schema = json!({
            "properties": {
                "a": {"$ref": "#/default"},
                "b": {"$ref": "#/default/enum/0"}
            },
            "default": {"enum": [{"minLength": 5, "maxLength": 2}]}
        });
        assert_eq!(uncovered_ref_targets(&schema).len(), 2);
        assert_eq!(
            unsatisfiable_bound(&schema).as_deref(),
            Some("minLength (5) is greater than maxLength (2) at #/default/enum/0; no value can satisfy both")
        );
    }

    /// The largest schema the limits let through, built to make the bounds
    /// check slow: an `allOf` of thousands of members, one of them defining
    /// thousands of properties, inside a `default` nested 50 arrays deep, with
    /// a `$ref` to every level of it. Each level's walk used to merge the
    /// whole `allOf` again, at a cost of names times members (seconds per
    /// request); it is now linear and walks the subtree once.
    #[test]
    fn a_wide_all_of_under_a_ref_chain_is_checked_in_linear_time() {
        const LEVELS: usize = 50;
        let names: Map<String, Value> = (0..3400).map(|i| (format!("p{i}"), json!(0))).collect();
        let mut members = vec![json!({"properties": names})];
        members.extend((0..8000).map(|_| json!({})));
        let mut nested = json!({"allOf": members});
        for _ in 0..LEVELS {
            nested = json!([nested]);
        }
        let refs: Map<String, Value> = (0..=LEVELS)
            .map(|level| {
                let pointer = std::iter::once("#/default")
                    .chain(std::iter::repeat_n("0", level))
                    .collect::<Vec<_>>()
                    .join("/");
                (format!("r{level}"), json!({"$ref": pointer}))
            })
            .collect();
        let schema = json!({"type": "object", "properties": refs, "default": nested});
        let text = schema.to_string();
        assert!(text.len() <= MAX_SCHEMA_BYTES, "{} bytes", text.len());
        check_limits(&schema, &text).expect("the limits let this schema through");
        let start = std::time::Instant::now();
        assert_eq!(unsatisfiable_bound(&schema), None);
        let took = start.elapsed();
        // Linear: a few milliseconds in release and well under this bound in
        // a debug build under load; the old cost was billions of steps.
        assert!(took < Duration::from_secs(2), "took {took:?}");
    }

    /// A local `$ref` into a data keyword makes that value a schema: its
    /// bounds are checked like any other.
    #[test]
    fn a_ref_into_a_data_keyword_is_checked() {
        assert_eq!(
            parse_schema(json!({
                "type": "object",
                "properties": {"a": {"$ref": "#/default/x"}},
                "default": {"x": {"type": "string", "minLength": 5, "maxLength": 2}}
            })),
            unsat("minLength (5) is greater than maxLength (2) at #/default/x; no value can satisfy both")
        );
    }

    /// The pair is refused wherever it sits, also in a branch a value need not
    /// take; the message says what is true of the pair itself.
    #[test]
    fn a_pair_in_an_optional_branch_is_named_for_what_it_is() {
        assert_eq!(
            parse_schema(json!({"anyOf": [
                {"type": "integer", "minimum": 5, "maximum": 2},
                {"type": "null"}
            ]})),
            unsat(
                "minimum (5) is greater than maximum (2) at #/anyOf/0; no value can satisfy both"
            )
        );
    }

    /// Float bounds past the range an integer comparison can hold are not
    /// read as an empty integer range.
    #[test]
    fn huge_float_bounds_on_an_integer_are_not_refused() {
        for schema in [
            json!({"type": "integer", "minimum": 1e39, "exclusiveMaximum": 1e40}),
            json!({"type": "integer", "exclusiveMinimum": -1e40, "maximum": -1e39}),
        ] {
            assert_eq!(unsatisfiable_bound(&schema), None, "{schema}");
        }
    }

    #[test]
    fn unsatisfiable_bound_reports_without_failing() {
        assert_eq!(
            unsatisfiable_bound(&json!({
                "type": "object",
                "properties": {"a": {"type": "string", "minLength": 5, "maxLength": 2}}
            }))
            .as_deref(),
            Some("minLength (5) is greater than maxLength (2) at #/properties/a; no value can satisfy both")
        );
    }

    #[test]
    fn integer_range_with_no_integer_is_rejected() {
        assert_eq!(
            parse_schema(json!({"type": "integer", "exclusiveMinimum": 2, "exclusiveMaximum": 3})),
            unsat("no integer meets exclusiveMinimum (2) and exclusiveMaximum (3) at #")
        );
        assert_eq!(
            parse_schema(json!({"type": "integer", "minimum": 2.5, "maximum": 2.9})),
            unsat("no integer meets minimum (2.5) and maximum (2.9) at #")
        );
        assert_eq!(
            parse_schema(json!({"type": ["integer"], "minimum": 2.1, "exclusiveMaximum": 3})),
            unsat("no integer meets minimum (2.1) and exclusiveMaximum (3) at #")
        );
        // The tighter of two lower bounds is the one named.
        assert_eq!(
            parse_schema(json!({
                "type": "integer", "minimum": 1, "exclusiveMinimum": 4.5, "maximum": 4.9
            })),
            unsat("no integer meets exclusiveMinimum (4.5) and maximum (4.9) at #")
        );
    }

    #[test]
    fn integer_range_with_no_integer_is_found_nested_and_through_ref() {
        assert_eq!(
            parse_schema(json!({
                "type": "object",
                "properties": {"n": {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 1}}
            })),
            unsat(
                "no integer meets exclusiveMinimum (0) and exclusiveMaximum (1) at #/properties/n"
            )
        );
        assert_eq!(
            parse_schema(json!({
                "type": "array",
                "items": {"type": "integer", "minimum": -0.5, "maximum": -0.25}
            })),
            unsat("no integer meets minimum (-0.5) and maximum (-0.25) at #/items")
        );
        assert_eq!(
            parse_schema(json!({
                "type": "object",
                "properties": {"n": {"$ref": "#/$defs/N"}},
                "$defs": {"N": {"type": "integer", "minimum": 7.1, "exclusiveMaximum": 8}}
            })),
            unsat("no integer meets minimum (7.1) and exclusiveMaximum (8) at #/$defs/N")
        );
    }

    #[test]
    fn integer_range_holding_an_integer_is_accepted() {
        for schema in [
            json!({"type": "integer", "exclusiveMinimum": 2, "exclusiveMaximum": 4}),
            json!({"type": "integer", "minimum": 2.5, "maximum": 3}),
            json!({"type": "integer", "minimum": 3, "maximum": 3}),
            json!({"type": "integer", "exclusiveMinimum": 2.5, "exclusiveMaximum": 3.5}),
            json!({"type": "integer", "minimum": 2.5}),
            // A type list allowing other numbers keeps the range.
            json!({"type": ["integer", "number"], "exclusiveMinimum": 2, "exclusiveMaximum": 3}),
            json!({"type": "number", "minimum": 2.5, "maximum": 2.9}),
            json!({"minimum": 2.5, "maximum": 2.9}),
        ] {
            assert!(
                matches!(
                    parse_schema(schema.clone()),
                    Ok(ResponseFormat::JsonSchema { .. })
                ),
                "{schema}"
            );
            assert_eq!(unsatisfiable_bound(&schema), None, "{schema}");
        }
    }
}
