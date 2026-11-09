//! OpenAI `response_format` parsing: turns the request field into a
//! [`ResponseFormat`], maps it to a backend-neutral [`GrammarSpec`], and
//! lists JSON Schema keywords the decoder cannot enforce.

use cascadia_types::{GrammarKind, GrammarSpec};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

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
        /// The schema, parsed. Its object keys are sorted, so use it for
        /// checks only.
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
// schema seen.

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

const TYPE_ERR: &str = "response_format.type must be one of text, json_object, json_schema";

/// Parse the `response_format` field of a chat request from its JSON text.
/// Absent or `null` means plain text. The text is the field exactly as the
/// client sent it, so a schema keeps its key order (see
/// [`ResponseFormat::JsonSchema`]).
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
            let (schema, text) = match merge_object_all_of(&text) {
                Some(merged) => (
                    serde_json::from_str(&merged).map_err(|e| format!("{FIELD}: {e}"))?,
                    merged,
                ),
                None => (schema.clone(), text),
            };
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
/// object, nested deeper than
/// [`MAX_SCHEMA_DEPTH`], with a count keyword or pattern quantifier above
/// [`MAX_SCHEMA_COUNT`], or whose counts sum above [`MAX_SCHEMA_COUNT_TOTAL`].
/// `text` is the schema's compact text.
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
    let mut total = 0;
    sum_counts(schema, &mut total)?;
    if total > MAX_SCHEMA_COUNT_TOTAL {
        return Err(format!(
            "{FIELD}: repetition counts add up to {total}; the limit is {MAX_SCHEMA_COUNT_TOTAL}"
        ));
    }
    Ok(())
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
}
