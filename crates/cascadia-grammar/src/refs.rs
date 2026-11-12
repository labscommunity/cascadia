//! Where llguidance resolves a schema's `$ref`s, so the integer guards
//! (`int_bounds`) can follow them to the same place.
//!
//! llguidance resolves a `$ref` with the `referencing` crate: against the
//! base URI of the schema resource it is in (the root's `$id`, or any
//! `$id` below it), through anchors (`$anchor`, a draft-6/7 `$id: "#name"`),
//! and to any URI the document registers. The guards only read the
//! document itself, so a schema is refused when one of its refs could land
//! somewhere they don't look (see [`unfollowable_ref`]), and the refs left,
//! `#` and `#/...`, are resolved here the way llguidance resolves them
//! (see [`local_ref_pointers`]).

use crate::NAME_MAPS;
use serde_json::Value;

/// Keywords whose values are data: llguidance never builds them as schemas
/// unless a `$ref` points into them.
const DATA_KEYWORDS: &[&str] = &["const", "enum", "default", "examples", "x-guidance"];

/// Why llguidance could resolve a `$ref` in `schema` somewhere other than
/// where [`local_ref_pointers`] looks, if it could. Read in every schema
/// llguidance can build: the root, its subschemas, and the target of each
/// `$ref` (which may sit in a data keyword such as `default`). Refused:
///
/// - a `$ref` that does not start with `#` (`urn:x#/default/x`,
///   `json-schema:///#/...`): it resolves against the registry, which holds
///   the root under its `$id` or a default base URI;
/// - a `#` ref that is not a JSON pointer (`#name`): an anchor;
/// - a `$id` below the root, or a draft-4 `id` (when the document names
///   draft 4 anywhere): it changes the base URI that every `#/...` ref under
///   it resolves against;
/// - a root `$id` (or draft-4 `id`) with a non-empty fragment: llguidance
///   looks the root up by it, so `"$id": "#/default/x"` builds `default.x`
///   as the whole schema.
///
/// `$dynamicRef` and `$recursiveRef` are not resolved at all (llguidance
/// does not implement them and ignores them), so they are left alone.
pub fn unfollowable_ref(schema: &Value) -> Option<String> {
    let id_keys: &[&str] = if names_draft_4(schema) {
        &["$id", "id"]
    } else {
        &["$id"]
    };
    let mut entered = std::collections::HashSet::new();
    let mut stack = vec![(schema, true)];
    while let Some((v, is_root)) = stack.pop() {
        match v {
            Value::Object(obj) => {
                if is_root {
                    // llguidance looks the root up by its id: a fragment
                    // there (`#/default/x`) makes the value it points to,
                    // data or not, the schema it builds.
                    if let Some(key) = id_keys.iter().find(|k| {
                        obj.get(**k)
                            .and_then(Value::as_str)
                            .and_then(|id| id.split_once('#'))
                            .is_some_and(|(_, fragment)| !fragment.is_empty())
                    }) {
                        return Some(format!(
                            "`{key}` with a fragment is not supported (`#` followed by \
                             anything changes what the schema is)"
                        ));
                    }
                } else if let Some(key) = id_keys
                    .iter()
                    .find(|k| obj.get(**k).is_some_and(Value::is_string))
                {
                    return Some(format!(
                        "`{key}` below the root is not supported (it changes what \
                             a `$ref` under it refers to)"
                    ));
                }
                if let Some(r) = obj.get("$ref").and_then(Value::as_str) {
                    let shown = Value::String(r.chars().take(80).collect());
                    let Some(fragment) = r.strip_prefix('#') else {
                        return Some(format!(
                            "$ref {shown} is not supported: only a reference inside \
                             the schema (`#` or `#/...`) is"
                        ));
                    };
                    if !fragment.is_empty() && !fragment.starts_with('/') {
                        return Some(format!(
                            "$ref {shown} is not supported: an anchor; use a JSON \
                             pointer (`#/...`)"
                        ));
                    }
                    if let Some(p) = resolve_pointer(schema, fragment) {
                        if !p.is_empty() && entered.insert(p.clone()) {
                            if let Some(target) = schema.pointer(&p) {
                                stack.push((target, false));
                            }
                        }
                    }
                }
                for (k, child) in obj {
                    let k = k.as_str();
                    if DATA_KEYWORDS.contains(&k) {
                        continue;
                    }
                    match child {
                        Value::Object(map) if NAME_MAPS.contains(&k) => {
                            stack.extend(map.values().map(|s| (s, false)));
                        }
                        _ => stack.push((child, false)),
                    }
                }
            }
            Value::Array(items) => stack.extend(items.iter().map(|s| (s, false))),
            _ => {}
        }
    }
    None
}

/// Whether a `$schema` anywhere in `v` names draft 4, the draft whose
/// resource id is `id` rather than `$id`.
fn names_draft_4(v: &Value) -> bool {
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(obj) => {
                if obj.get("$schema").and_then(Value::as_str).is_some_and(|s| {
                    s.trim_end_matches('#') == "http://json-schema.org/draft-04/schema"
                }) {
                    return true;
                }
                stack.extend(obj.values());
            }
            Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
    false
}

/// The value the `#/...` refs anywhere in `doc` (data keywords included)
/// reach, as RFC 6901 pointers `serde_json` reads to the same value, each
/// once. A ref that reaches nothing (llguidance fails to build it) and any
/// other ref is left out.
pub(crate) fn local_ref_pointers(doc: &Value) -> Vec<String> {
    let mut pointers = std::collections::BTreeSet::new();
    let mut stack = vec![doc];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(obj) => {
                if let Some(p) = obj
                    .get("$ref")
                    .and_then(Value::as_str)
                    .and_then(|r| r.strip_prefix('#'))
                    .filter(|p| p.starts_with('/'))
                    .and_then(|p| resolve_pointer(doc, p))
                {
                    pointers.insert(p);
                }
                stack.extend(obj.values());
            }
            Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
    pointers.into_iter().collect()
}

/// Resolves the fragment `pointer` (`""` or `/...`, as written after `#`)
/// in `doc` the way llguidance's resolver (`referencing`) does, and returns
/// it as a plain RFC 6901 pointer to the same value, or `None` when it
/// reaches nothing. That resolver percent-decodes the whole fragment first
/// (so `%2F` splits a key and `%25` is `%`), then splits it on `/`,
/// unescapes `~1` and `~0` in each key, and reads an array index as any
/// `usize` text (`01`, `+1`), where RFC 6901 alone (and `serde_json`) would
/// read the fragment as written.
pub(crate) fn resolve_pointer(doc: &Value, pointer: &str) -> Option<String> {
    let Some(rest) = pointer.strip_prefix('/') else {
        return pointer.is_empty().then(String::new);
    };
    let decoded = percent_decode(rest)?;
    let mut current = doc;
    let mut canonical = String::new();
    for segment in decoded.split('/') {
        canonical.push('/');
        current = match current {
            Value::Array(items) => {
                let i = segment.parse::<usize>().ok()?;
                canonical.push_str(&i.to_string());
                items.get(i)?
            }
            Value::Object(obj) => {
                let key = segment.replace("~1", "/").replace("~0", "~");
                canonical.push_str(&key.replace('~', "~0").replace('/', "~1"));
                obj.get(&key)?
            }
            _ => return None,
        };
    }
    Some(canonical)
}

/// `%XX` (two hex digits) decoded to its byte, anything else kept as is;
/// `None` when the result is not UTF-8 (the resolver refuses it).
fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_pointer_is_read_the_way_llguidance_reads_it() {
        let doc = json!({
            "d": {"a%b": 1, "a%25b": 2, "a/b": 3, "a~b": 4, "a": {"b": 5}, "": 6},
            "e": [7, 8],
        });
        for (fragment, want) in [
            ("", Some("")),
            ("/d/a%25b", Some("/d/a%b")),
            ("/d/a%b", Some("/d/a%b")),
            ("/d/a%2Fb", Some("/d/a/b")),
            ("/d/a%2fb", Some("/d/a/b")),
            ("/d/a~1b", Some("/d/a~1b")),
            ("/d/a%7E1b", Some("/d/a~1b")),
            ("/d/a~0b", Some("/d/a~0b")),
            ("/d/", Some("/d/")),
            ("/e/01", Some("/e/1")),
            ("/e/+1", Some("/e/1")),
            ("/e/2", None),
            ("/e/-1", None),
            ("/d/a%FFb", None),
            ("/nope", None),
        ] {
            let got = resolve_pointer(&doc, fragment);
            assert_eq!(got.as_deref(), want, "{fragment}");
            if let Some(p) = got {
                assert!(doc.pointer(&p).is_some(), "{fragment} -> {p}");
            }
        }
        assert_eq!(doc.pointer("/d/a%b"), Some(&json!(1)));
        assert_eq!(doc.pointer("/d/a~1b"), Some(&json!(3)));
    }

    #[test]
    fn a_ref_the_guards_cannot_follow_is_named() {
        let int = json!({"type": "integer"});
        for (schema, want) in [
            (
                json!({"properties": {"n": {"$ref": "urn:x#/default/x"}}}),
                "$ref \"urn:x#/default/x\"",
            ),
            (
                json!({"properties": {"n": {"$ref": "json-schema:///#/default/x"}}}),
                "$ref \"json-schema:///",
            ),
            (
                json!({"properties": {"n": {"$ref": "other.json"}}}),
                "$ref \"other.json\"",
            ),
            (json!({"properties": {"n": {"$ref": "#foo"}}}), "an anchor"),
            (
                json!({"properties": {"n": {"$ref": "#%2Fdefault"}}}),
                "an anchor",
            ),
            (
                json!({"properties": {"a": {"$id": "urn:y"}}}),
                "`$id` below the root",
            ),
            (json!({"not": {"$id": "urn:y"}}), "`$id` below the root"),
            (json!({"$id": "#/default/x"}), "`$id` with a fragment"),
            (json!({"$id": "urn:x#foo"}), "`$id` with a fragment"),
            (
                json!({"$defs": {"a": {"$id": "#foo"}}}),
                "`$id` below the root",
            ),
            // A ref target in data is built as a schema: its `$id` counts.
            (
                json!({"properties": {"n": {"$ref": "#/default/t"}}, "default": {"t": {"$id": "urn:z"}}}),
                "`$id` below the root",
            ),
            (
                json!({"$schema": "http://json-schema.org/draft-04/schema#", "properties": {"a": {"id": "urn:y"}}}),
                "`id` below the root",
            ),
        ] {
            let got = unfollowable_ref(&schema).unwrap_or_else(|| panic!("{schema}"));
            assert!(got.contains(want), "{schema}: {got}");
        }
        for schema in [
            json!({"$id": "urn:x", "properties": {"n": {"$ref": "#/$defs/N"}}, "$defs": {"N": int}}),
            json!({"properties": {"n": {"$ref": "#"}, "m": {"$ref": "#/definitions/N"}}, "definitions": {"N": int}}),
            // A property, a `$defs` entry or data named like the keyword.
            json!({"properties": {"$id": int, "id": int, "$ref": int}, "$defs": {"$id": int}}),
            json!({"default": {"$id": "urn:y", "$ref": "urn:y"}, "examples": [{"id": "abc"}]}),
            // `id` is not an id outside draft 4.
            json!({"properties": {"a": {"id": "urn:y"}}}),
            // Not resolved by llguidance.
            json!({"properties": {"a": {"$dynamicRef": "urn:y#/x", "$recursiveRef": "#"}}}),
        ] {
            assert_eq!(unfollowable_ref(&schema), None, "{schema}");
        }
    }
}
