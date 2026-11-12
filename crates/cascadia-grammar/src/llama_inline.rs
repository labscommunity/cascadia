//! Inlining a schema's `$ref`s before it is sent to llama-server.
//!
//! llama-server's JSON-schema-to-grammar converter resolves `$ref`s itself,
//! and has several ways to give a ref the wrong shape without failing: a ref
//! it never recorded (one beside another `$ref`) becomes "any object", a ref
//! whose rule name another rule already holds reuses that rule, and tools
//! and `response_format` share one table of refs. Inlining every `#/...` ref
//! before forwarding leaves it no client ref to resolve.
//!
//! A ref on a cycle can't be inlined: each schema on a cycle is moved
//! ("hoisted") to a `$defs` entry with a reserved name, and the refs on the
//! cycle point there. On llama-server's chat endpoint every other rule is
//! named under the `response_format` or tool prefix, so those names can't
//! collide; with `--no-jinja` the schema is visited as the root, where a
//! property's rule can take any name, so hoisting is refused there.

use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

/// Keywords whose value, in a schema, is data: never a schema llama-server
/// visits, though its ref scan still reads `$ref`s there (and fails on a
/// form it doesn't support, such as `#`).
const DATA_KEYWORDS: &[&str] = &["const", "enum", "default", "examples"];

/// Data keywords the converter only ignores (annotations): their values
/// are sent as `null`, so no `$ref` in them reaches its ref scan (the key
/// stays: an object with only an annotation is "any value" to the
/// converter, an empty one "any object"). `const` and `enum` are kept as
/// they are: they are the grammar.
const ANNOTATIONS: &[&str] = &["default", "examples"];

/// Keywords whose value is an object of named schemas.
const NAME_MAPS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];

/// Keywords holding schemas that are only reachable through a `$ref`:
/// dropped from the inlined schema (their targets are copied where used).
const DEFINITIONS: &[&str] = &["$defs", "definitions"];

/// Largest inlined schema, in bytes of its JSON text (estimated while it is
/// built): a schema reusing `$defs` many times grows with each use.
pub const MAX_INLINED_BYTES: usize = 1 << 20;

/// Deepest nesting of inlined refs (a ref inside a target inside a target...).
const MAX_INLINE_DEPTH: usize = 256;

/// Why a schema can't be sent to llama-server inlined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineError {
    /// A `$ref` in a schema position that is not a `#` pointer into the
    /// schema, or one that names nothing.
    Unresolvable(String),
    /// The schema is recursive and hoisting is not allowed (`--no-jinja`).
    Recursive(String),
    /// Over [`MAX_INLINED_BYTES`] or the nesting cap.
    TooBig,
}

impl std::fmt::Display for InlineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shown = |r: &str| Value::String(r.chars().take(80).collect());
        match self {
            InlineError::Unresolvable(r) => write!(
                f,
                "$ref {} is not supported: only a reference inside the schema \
                 (`#` or `#/...`) to an existing schema is",
                shown(r)
            ),
            InlineError::Recursive(r) => write!(
                f,
                "$ref {} makes the schema recursive, which this engine can't enforce \
                 when llama-server runs without --jinja",
                shown(r)
            ),
            InlineError::TooBig => write!(
                f,
                "the schema with every $ref replaced by its target is over {} bytes",
                MAX_INLINED_BYTES
            ),
        }
    }
}

/// The value `#`-pointer `r` names in `doc`: RFC 6901 (`~1` is `/`, `~0` is
/// `~`, an array index in decimal).
pub fn pointer_target<'a>(doc: &'a Value, r: &str) -> Option<&'a Value> {
    let pointer = r.strip_prefix('#')?;
    if pointer.is_empty() {
        return Some(doc);
    }
    let mut cur = doc;
    for tok in pointer.strip_prefix('/')?.split('/') {
        let tok = tok.replace("~1", "/").replace("~0", "~");
        cur = match cur {
            Value::Object(o) => o.get(&tok)?,
            Value::Array(a) => {
                if tok.is_empty() || (tok.len() > 1 && tok.starts_with('0')) {
                    return None;
                }
                a.get(tok.parse::<usize>().ok()?)?
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// The `$ref`s llama-server's converter can visit in `v`: through schema
/// positions only (not data, not `$defs`), stopping at an object holding a
/// `$ref` (its siblings are ignored).
fn schema_refs<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(o) => {
                if let Some(r) = o.get("$ref") {
                    if let Some(r) = r.as_str() {
                        out.push(r);
                    }
                    continue;
                }
                for (k, c) in o {
                    let k = k.as_str();
                    if DATA_KEYWORDS.contains(&k) || DEFINITIONS.contains(&k) {
                        continue;
                    }
                    match c {
                        Value::Object(m) if NAME_MAPS.contains(&k) => stack.extend(m.values()),
                        _ => stack.push(c),
                    }
                }
            }
            Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
}

/// Whether a client's `$ref` (not one to a hoisted copy, named in `ours`)
/// is left in `v` outside the root's `$defs`: only in data, since every
/// other one was inlined.
fn has_client_ref(v: &Value, is_root: bool, ours: &HashSet<String>) -> bool {
    match v {
        Value::Object(o) => o.iter().any(|(k, c)| {
            (k == "$ref" && !c.as_str().is_some_and(|r| ours.contains(r)))
                || (!(is_root && DEFINITIONS.contains(&k.as_str()))
                    && has_client_ref(c, false, ours))
        }),
        Value::Array(items) => items.iter().any(|c| has_client_ref(c, false, ours)),
        _ => false,
    }
}

/// What [`inline_refs`] did, for the caller to log or measure.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Inlined {
    pub schema: Value,
    /// Schemas hoisted to reserved `$defs` entries (recursive ones).
    pub hoisted: usize,
}

struct Inliner<'a> {
    doc: &'a Value,
    /// Reserved `$defs` name of each ref on a cycle.
    hoisted: HashMap<&'a str, String>,
    bytes: usize,
    depth: usize,
}

impl<'a> Inliner<'a> {
    fn charge(&mut self, n: usize) -> Result<(), InlineError> {
        self.bytes += n;
        if self.bytes > MAX_INLINED_BYTES {
            return Err(InlineError::TooBig);
        }
        Ok(())
    }

    /// `v` (a schema) with every `$ref` in a schema position replaced by its
    /// target, or by a ref to its hoisted copy.
    fn schema(&mut self, v: &'a Value) -> Result<Value, InlineError> {
        self.depth += 1;
        if self.depth > MAX_INLINE_DEPTH {
            return Err(InlineError::TooBig);
        }
        let out = self.schema_inner(v);
        self.depth -= 1;
        out
    }

    // The recursion is split so each frame stays small: it nests once per
    // JSON level and per inlined ref, on a server worker's stack.
    fn schema_inner(&mut self, v: &'a Value) -> Result<Value, InlineError> {
        match v {
            Value::Object(o) => match o.get("$ref") {
                Some(r) => self.reference(r),
                None => self.object(o),
            },
            Value::Array(items) => self.array(items),
            other => {
                self.charge(other.to_string().len())?;
                Ok(other.clone())
            }
        }
    }

    #[inline(never)]
    fn reference(&mut self, r: &'a Value) -> Result<Value, InlineError> {
        let Some(r) = r.as_str() else {
            return Err(InlineError::Unresolvable(r.to_string()));
        };
        if let Some(name) = self.hoisted.get(r).cloned() {
            self.charge(name.len() + 20)?;
            let mut out = Map::new();
            out.insert("$ref".into(), Value::String(format!("#/$defs/{name}")));
            return Ok(Value::Object(out));
        }
        let target =
            pointer_target(self.doc, r).ok_or_else(|| InlineError::Unresolvable(r.to_owned()))?;
        self.schema(target)
    }

    #[inline(never)]
    fn object(&mut self, o: &'a Map<String, Value>) -> Result<Value, InlineError> {
        let mut out = Map::new();
        for (k, c) in o {
            let key = k.as_str();
            if DEFINITIONS.contains(&key) {
                continue;
            }
            if ANNOTATIONS.contains(&key) {
                self.charge(k.len() + 8)?;
                out.insert(k.clone(), Value::Null);
                continue;
            }
            self.charge(k.len() + 4)?;
            let value = if DATA_KEYWORDS.contains(&key) {
                self.data(c)?
            } else {
                match c {
                    Value::Object(m) if NAME_MAPS.contains(&key) => self.named(m)?,
                    _ => self.schema(c)?,
                }
            };
            out.insert(k.clone(), value);
        }
        Ok(Value::Object(out))
    }

    #[inline(never)]
    fn named(&mut self, m: &'a Map<String, Value>) -> Result<Value, InlineError> {
        let mut named = Map::new();
        for (name, s) in m {
            self.charge(name.len() + 4)?;
            named.insert(name.clone(), self.schema(s)?);
        }
        Ok(Value::Object(named))
    }

    #[inline(never)]
    fn array(&mut self, items: &'a [Value]) -> Result<Value, InlineError> {
        let mut out = Vec::with_capacity(items.len());
        for c in items {
            out.push(self.schema(c)?);
        }
        Ok(Value::Array(out))
    }

    #[inline(never)]
    fn data(&mut self, c: &Value) -> Result<Value, InlineError> {
        self.charge(c.to_string().len())?;
        Ok(c.clone())
    }
}

/// `schema` with every `#` / `#/...` `$ref` in a schema position replaced
/// by a copy of its target, siblings of a `$ref` dropped (llama-server
/// ignores them too), `default` and `examples` values made `null`
/// (annotations it ignores), `const` and `enum` values left as they are. A ref on a
/// cycle points instead at a copy of its target under a reserved `$defs`
/// name, `__cascadia_<prefix>r<N>x` (refused unless `allow_hoisting`). The
/// root `$defs`/`definitions` are kept only while a `$ref` is left in data.
pub fn inline_refs(
    schema: &Value,
    prefix: &str,
    allow_hoisting: bool,
) -> Result<Inlined, InlineError> {
    // The refs each ref's target (and the root) can reach directly.
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut root_refs = Vec::new();
    schema_refs(schema, &mut root_refs);
    let mut todo: Vec<&str> = root_refs.clone();
    while let Some(r) = todo.pop() {
        if edges.contains_key(r) {
            continue;
        }
        if !r.starts_with('#') {
            return Err(InlineError::Unresolvable(r.to_owned()));
        }
        let target =
            pointer_target(schema, r).ok_or_else(|| InlineError::Unresolvable(r.to_owned()))?;
        let mut next = Vec::new();
        schema_refs(target, &mut next);
        todo.extend(next.iter().copied());
        edges.insert(r, next);
    }
    // Refs on a cycle: Tarjan's strongly connected components.
    let on_cycle = refs_on_cycles(&edges);
    if let Some(r) = on_cycle.iter().min() {
        if !allow_hoisting {
            return Err(InlineError::Recursive((*r).to_owned()));
        }
    }
    let existing: HashSet<&str> = schema
        .get("$defs")
        .and_then(Value::as_object)
        .map(|d| d.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let mut sorted: Vec<&str> = on_cycle.into_iter().collect();
    sorted.sort_unstable();
    let mut hoisted = HashMap::new();
    let mut n = 0;
    for r in &sorted {
        let name = loop {
            let name = format!("__cascadia_{prefix}r{n}x");
            n += 1;
            if !existing.contains(name.as_str()) {
                break name;
            }
        };
        hoisted.insert(*r, name);
    }
    let mut inliner = Inliner {
        doc: schema,
        hoisted,
        bytes: 0,
        depth: 0,
    };
    let body = inliner.schema(schema)?;
    let mut defs = Map::new();
    for r in &sorted {
        let target = pointer_target(schema, r).expect("resolved above");
        let copy = inliner.schema(target)?;
        defs.insert(inliner.hoisted[r].clone(), copy);
    }
    // The root itself on a cycle (`#`): it becomes a ref to its copy, in an
    // `anyOf` so llama-server's ref scan still reads the `$defs` beside it.
    let mut out = match inliner.hoisted.get("#") {
        Some(name) => serde_json::json!({ "anyOf": [{ "$ref": format!("#/$defs/{name}") }] }),
        None => body,
    };
    let Value::Object(root) = &mut out else {
        return if sorted.is_empty() {
            Ok(Inlined {
                schema: out,
                hoisted: 0,
            })
        } else {
            Err(InlineError::Unresolvable("#".into()))
        };
    };
    // Keep the client's definitions while data still holds a ref into them.
    let ours: HashSet<String> = inliner
        .hoisted
        .values()
        .map(|n| format!("#/$defs/{n}"))
        .collect();
    let data_refs = has_client_ref(&Value::Object(root.clone()), true, &ours)
        || defs.values().any(|d| has_client_ref(d, false, &ours));
    if data_refs {
        for key in DEFINITIONS {
            if let Some(d) = schema.get(*key) {
                root.insert((*key).to_owned(), d.clone());
            }
        }
    }
    if !defs.is_empty() {
        let all = root
            .entry("$defs")
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(all) = all else {
            return Err(InlineError::Unresolvable("#/$defs".into()));
        };
        all.extend(defs);
    }
    Ok(Inlined {
        schema: out,
        hoisted: sorted.len(),
    })
}

/// The refs in `edges` that are on a cycle (a component of more than one,
/// or one with an edge to itself).
fn refs_on_cycles<'a>(edges: &HashMap<&'a str, Vec<&'a str>>) -> HashSet<&'a str> {
    // Iterative Tarjan.
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut low: HashMap<&str, usize> = HashMap::new();
    let mut on_stack: HashSet<&str> = HashSet::new();
    let mut stack: Vec<&str> = Vec::new();
    let mut out = HashSet::new();
    let mut next = 0;
    let mut nodes: Vec<&str> = edges.keys().copied().collect();
    nodes.sort_unstable();
    for start in nodes {
        if index.contains_key(start) {
            continue;
        }
        // (node, next edge to look at)
        let mut work: Vec<(&str, usize)> = vec![(start, 0)];
        while let Some(&(v, i)) = work.last() {
            if i == 0 && !index.contains_key(v) {
                index.insert(v, next);
                low.insert(v, next);
                next += 1;
                stack.push(v);
                on_stack.insert(v);
            }
            let succ = edges.get(v).map(Vec::as_slice).unwrap_or(&[]);
            if i < succ.len() {
                work.last_mut().expect("not empty").1 += 1;
                let w = succ[i];
                if !index.contains_key(w) {
                    work.push((w, 0));
                } else if on_stack.contains(w) {
                    let m = low[v].min(index[w]);
                    low.insert(v, m);
                }
                continue;
            }
            work.pop();
            if let Some(&(parent, _)) = work.last() {
                let m = low[parent].min(low[v]);
                low.insert(parent, m);
            }
            if low[v] == index[v] {
                let mut comp = Vec::new();
                loop {
                    let w = stack.pop().expect("on the stack");
                    on_stack.remove(w);
                    comp.push(w);
                    if w == v {
                        break;
                    }
                }
                let self_loop = edges.get(v).is_some_and(|s| s.contains(&v));
                if comp.len() > 1 || self_loop {
                    out.extend(comp);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn refs_left(v: &Value) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![v];
        while let Some(v) = stack.pop() {
            match v {
                Value::Object(o) => {
                    if let Some(Value::String(r)) = o.get("$ref") {
                        out.push(r.clone());
                    }
                    stack.extend(o.values());
                }
                Value::Array(a) => stack.extend(a),
                _ => {}
            }
        }
        out
    }

    #[test]
    fn every_ref_is_replaced_by_its_target() {
        let schema = json!({
            "$defs": {"A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}}, "B": {"type": "integer"}},
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A", "description": "dropped"}, "c": {"items": {"$ref": "#/$defs/B"}}},
        });
        let got = inline_refs(&schema, "", true).unwrap();
        assert_eq!(got.hoisted, 0);
        assert_eq!(
            got.schema,
            json!({
                "type": "object",
                "properties": {
                    "a": {"type": "object", "properties": {"b": {"type": "integer"}}},
                    "c": {"items": {"type": "integer"}},
                },
            })
        );
    }

    #[test]
    fn data_is_left_alone_by_position_not_by_name() {
        let schema = json!({
            "$defs": {"B": {"type": "integer"}},
            "type": "object",
            "properties": {
                // A property named like a data keyword holds a schema.
                "default": {"$ref": "#/$defs/B"},
                "k": {"const": {"$ref": "#/$defs/B"}, "examples": [{"$ref": "x"}], "default": {"$ref": "#"}},
            },
        });
        let got = inline_refs(&schema, "", true).unwrap().schema;
        assert_eq!(got["properties"]["default"], json!({"type": "integer"}));
        // `const` is the grammar: kept as it is. Annotations become `null`.
        assert_eq!(
            got["properties"]["k"],
            json!({"const": {"$ref": "#/$defs/B"}, "examples": null, "default": null})
        );
        // A ref left in data keeps the definitions it may point into.
        assert_eq!(got["$defs"], schema["$defs"]);
    }

    #[test]
    fn pointers_follow_rfc_6901() {
        let schema = json!({
            "$defs": {"a/b": {"type": "integer"}, "a~b": {"type": "string"}, "a": {"b": {"type": "boolean"}}},
            "properties": {"x": {"$ref": "#/$defs/a~1b"}, "y": {"$ref": "#/$defs/a~0b"}, "z": {"$ref": "#/$defs/a/b"}},
        });
        let got = inline_refs(&schema, "", true).unwrap().schema;
        assert_eq!(got["properties"]["x"], json!({"type": "integer"}));
        assert_eq!(got["properties"]["y"], json!({"type": "string"}));
        assert_eq!(got["properties"]["z"], json!({"type": "boolean"}));
    }

    #[test]
    fn a_ref_that_cannot_be_inlined_is_named() {
        for (schema, r) in [
            (
                json!({"properties": {"a": {"$ref": "https://example.com/a.json"}}}),
                "https://example.com/a.json",
            ),
            (json!({"properties": {"a": {"$ref": "T"}}}), "T"),
            (
                json!({"properties": {"a": {"$ref": "#/$defs/missing"}}}),
                "#/$defs/missing",
            ),
            (json!({"items": [{"$ref": "urn:z"}]}), "urn:z"),
        ] {
            assert_eq!(
                inline_refs(&schema, "", true),
                Err(InlineError::Unresolvable(r.into())),
                "{schema}"
            );
        }
    }

    #[test]
    fn a_cycle_is_hoisted_under_a_reserved_name() {
        let list = json!({
            "type": "object",
            "properties": {"head": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {"type": "object", "properties": {"next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}}}},
        });
        let got = inline_refs(&list, "", true).unwrap();
        assert_eq!(got.hoisted, 1);
        assert_eq!(
            got.schema,
            json!({
                "type": "object",
                "properties": {"head": {"$ref": "#/$defs/__cascadia_r0x"}},
                "$defs": {"__cascadia_r0x": {"type": "object", "properties": {"next": {"anyOf": [{"$ref": "#/$defs/__cascadia_r0x"}, {"type": "null"}]}}}},
            })
        );
        assert_eq!(
            inline_refs(&list, "", false),
            Err(InlineError::Recursive("#/$defs/Node".into()))
        );
        // A tool's copies get a prefix of their own.
        let got = inline_refs(&list, "t2", true).unwrap().schema;
        assert!(got["$defs"].get("__cascadia_t2r0x").is_some(), "{got}");
    }

    #[test]
    fn the_root_on_a_cycle_is_hoisted_behind_an_any_of() {
        let tree = json!({"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#"}}}});
        let got = inline_refs(&tree, "", true).unwrap().schema;
        assert_eq!(
            got,
            json!({
                "anyOf": [{"$ref": "#/$defs/__cascadia_r0x"}],
                "$defs": {"__cascadia_r0x": {"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/__cascadia_r0x"}}}}},
            })
        );
        assert_eq!(refs_left(&got).len(), 2);
    }

    #[test]
    fn a_schema_that_grows_past_the_budget_is_refused() {
        // Each definition uses the previous one twice: 2^30 copies.
        let mut defs = Map::new();
        defs.insert("d0".into(), json!({"type": "string"}));
        for i in 1..30 {
            let prev = format!("#/$defs/d{}", i - 1);
            defs.insert(
                format!("d{i}"),
                json!({"type": "object", "properties": {"a": {"$ref": prev}, "b": {"$ref": prev}}}),
            );
        }
        let schema = json!({"$defs": defs, "properties": {"x": {"$ref": "#/$defs/d29"}}});
        let start = std::time::Instant::now();
        assert_eq!(inline_refs(&schema, "", true), Err(InlineError::TooBig));
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
    }
}
