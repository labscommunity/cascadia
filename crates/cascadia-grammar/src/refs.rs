//! Which `$ref`s llguidance can resolve, and where the integer guards
//! (`int_bounds`) look for their targets.
//!
//! llguidance resolves a `$ref` with the `referencing` crate: against the
//! base URI of the schema resource it is in (the root's `$id`, or any
//! `$id` below it), through anchors, and to any URI the document
//! registers. It never fetches: a ref to a document the schema doesn't hold
//! is refused here with a 400 ([`ref_outside_the_schema`]), for every
//! engine, by building the same `referencing` registry llguidance builds
//! first (same crate version, same root, same base URI).
//!
//! The guards read `#/...` refs against the document root
//! ([`local_ref_pointers`]), the way llguidance reads them when no `$id`
//! below the root changes the base. Their abort check does not depend on
//! it: it reads every object in the document. A target reached only
//! through an embedded `$id` (a `#/...` ref under a non-root `$id`, or
//! one named by that id) inside a data keyword (`default`, `examples`) is
//! not bounded or clamped: no abort, but its integers may take more than
//! 19 digits, or llguidance refuses a lower bound below what it builds
//! with its own error. Targets in schema positions (`$defs`, properties)
//! are bounded where they sit.

use referencing::{Draft, Error, Registry, Resource};
use serde_json::Value;

/// The base URI llguidance gives a root without `$id`.
const DEFAULT_ROOT_URI: &str = "json-schema:///";

/// Why a `$ref` in `schema` names a document outside it, if one does: a
/// URI no resource in the schema has (`https://example.com/s.json`,
/// `other.json`, `urn:z` under a non-`urn` base), which llguidance would
/// have to fetch and never does.
///
/// This is the first thing llguidance does with a schema: build the
/// `referencing` registry from the root (its draft, its `$id` or the
/// default base URI) without a retriever. That build resolves every ref in
/// every subresource, and in the target of each `#` ref, against its own
/// scope's base URI, and fails when one names a document it doesn't hold.
/// Asking the same question with the same crate refuses exactly the schemas
/// llguidance fails on there, never one it builds. Refs that only fail
/// later, at a build-time lookup (in a scope whose base is a `urn:`, which
/// the registry doesn't check), get llguidance's own error.
pub fn ref_outside_the_schema(schema: &Value) -> Option<String> {
    // An unknown `$schema` (`http://json-schema.org/schema#`, an OpenAPI
    // dialect) is no ref, but the registry treats it as one: llguidance
    // refuses the schema with its own error, other backends ignore it.
    // Read as the latest draft, which is how llguidance reads it, so only
    // refs are judged here.
    let mut schema = schema.clone();
    known_dollar_schema_only(&mut schema);
    // As llguidance's `PreContext::new`.
    let draft = Draft::Draft202012
        .detect(&schema)
        .unwrap_or(Draft::Draft202012);
    let resource: Resource = draft.create_resource(schema);
    let base = resource.id().unwrap_or(DEFAULT_ROOT_URI).to_string();
    let empty = Registry::try_from_resources(std::iter::empty::<(String, Resource)>()).ok()?;
    match empty.try_with_resources_and_retriever(
        vec![(base.as_str(), resource)],
        &referencing::DefaultRetriever,
        draft,
    ) {
        Err(Error::Unretrievable { uri, .. }) => {
            let shown: String = uri.chars().take(80).collect();
            Some(format!(
                "$ref {} is not supported: it names a document outside the schema \
                 (only references to the schema itself, such as `#/$defs/...`, are)",
                Value::String(shown)
            ))
        }
        _ => None,
    }
}

/// Every `$schema` in `v` that names no draft `referencing` knows, set to
/// the latest draft's URI: the draft llguidance reads it as, and one the
/// registry never looks up.
fn known_dollar_schema_only(v: &mut Value) {
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(obj) => {
                if let Some(Value::String(meta)) = obj.get_mut("$schema") {
                    let probe = serde_json::json!({ "$schema": meta.as_str() });
                    if Draft::Draft202012.detect(&probe).is_err() {
                        *meta = "https://json-schema.org/draft/2020-12/schema".into();
                    }
                }
                stack.extend(obj.values_mut());
            }
            Value::Array(items) => stack.extend(items.iter_mut()),
            _ => {}
        }
    }
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
    fn a_ref_to_a_document_outside_the_schema_is_named() {
        for (schema, want) in [
            (
                json!({"properties": {"n": {"$ref": "https://example.com/s.json"}}}),
                "https://example.com/s.json",
            ),
            (
                json!({"properties": {"n": {"$ref": "other.json#/x"}}}),
                "json-schema:///other.json",
            ),
            (json!({"properties": {"n": {"$ref": "urn:z"}}}), "urn:z"),
        ] {
            let got = ref_outside_the_schema(&schema).unwrap_or_else(|| panic!("{schema}"));
            assert!(
                got.contains(want) && got.contains("outside the schema"),
                "{schema}: {got}"
            );
        }
        let int = json!({"type": "integer"});
        for schema in [
            json!({"$id": "urn:x", "properties": {"n": {"$ref": "urn:x#/$defs/N"}}, "$defs": {"N": int}}),
            json!({"properties": {"n": {"$ref": "json-schema:///#/$defs/N"}}, "$defs": {"N": int}}),
            json!({"properties": {"n": {"$ref": "#"}, "m": {"$ref": "#/definitions/N"}}, "definitions": {"N": int}}),
            json!({"$defs": {"T": {"$id": "T", "type": "string"}}, "properties": {"t": {"$ref": "T"}}}),
            json!({"properties": {"a": {"$id": "urn:y", "properties": {"n": {"$ref": "#/default/x"}}, "default": {"x": int}}}}),
            json!({"properties": {"n": {"$ref": "#foo"}}, "$defs": {"a": {"$anchor": "foo", "type": "integer"}}}),
            json!({"$schema": "http://json-schema.org/draft-07/schema#", "$id": "#root", "properties": {"a": {"$id": "#/properties/a"}}}),
            // Data and unknown keywords are not schemas.
            json!({"x-meta": {"$ref": "https://example.com/x.json"}, "default": {"$ref": "urn:q"}}),
            // Under a `urn:` base the registry doesn't check refs: left to
            // llguidance's own lookup error.
            json!({"$id": "urn:x", "properties": {"n": {"$ref": "urn:z"}}}),
            // Inside a ref target in data: the registry doesn't look there,
            // llguidance's lookup fails when it builds the target.
            json!({"properties": {"n": {"$ref": "#/default/t"}}, "default": {"t": {"$ref": "https://example.com/t.json"}}}),
            // An unknown `$schema` is not a ref.
            json!({"$schema": "http://json-schema.org/schema#", "type": "object"}),
            json!({"$schema": "https://spec.openapis.org/oas/3.1/dialect/base", "properties": {"a": {"$schema": "x:y"}}}),
            // A ref to nothing is llguidance's own error.
            json!({"properties": {"n": {"$ref": "#/nope"}}}),
        ] {
            assert_eq!(ref_outside_the_schema(&schema), None, "{schema}");
        }
    }
}
