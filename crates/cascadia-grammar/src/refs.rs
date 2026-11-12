//! Which `$ref`s llguidance can resolve, and where the integer guards
//! (`int_bounds`) look for their targets.
//!
//! llguidance resolves a `$ref` with the `referencing` crate: against the
//! base URI of the schema resource it is in (the root's `$id`, or any
//! `$id` below it), through anchors, and to any URI the document
//! registers. It never fetches: a ref to a document the schema doesn't hold
//! is refused here with a 400 ([`ref_outside_the_schema`]), for every
//! engine, by asking `referencing` (the version llguidance builds with)
//! the same question llguidance asks.
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

use referencing::{Draft, Error, Registry, Resolver, Resource, ResourceRef};
use serde_json::Value;

/// The base URI llguidance gives a root without `$id`.
const DEFAULT_ROOT_URI: &str = "json-schema:///";

/// Why a `$ref` in `schema` names a document outside it, if one does: a
/// URI no resource in the schema has (`https://example.com/s.json`,
/// `other.json`, `urn:z`), which llguidance would have to fetch and never
/// does. Every ref llguidance resolves while building is checked, from the
/// same base URI: the refs in the root and its subschemas, and in the
/// target of each ref (which may sit in a data keyword such as `default`),
/// each target once. Refs llguidance can't resolve for another reason (a
/// pointer to nothing, a missing anchor) are left to its own error.
pub fn ref_outside_the_schema(schema: &Value) -> Option<String> {
    let outside = |uri: &str| {
        let shown: String = uri.chars().take(80).collect();
        format!(
            "$ref {} is not supported: it names a document outside the schema \
             (only references to the schema itself, such as `#/$defs/...`, are)",
            Value::String(shown)
        )
    };
    // As llguidance's `PreContext`: the draft from the root's `$schema`,
    // the root registered under its `$id` or the default base URI.
    let draft = Draft::Draft202012
        .detect(schema)
        .unwrap_or(Draft::Draft202012);
    let resource: Resource = draft.create_resource(schema.clone());
    let base = resource.id().unwrap_or(DEFAULT_ROOT_URI).to_string();
    let empty = Registry::try_from_resources(std::iter::empty::<(String, Resource)>()).ok()?;
    let registry = match empty.try_with_resources_and_retriever(
        vec![(base.as_str(), resource)],
        &referencing::DefaultRetriever,
        draft,
    ) {
        Ok(r) => r,
        Err(Error::Unretrievable { uri, .. }) => return Some(outside(&uri)),
        Err(_) => return None,
    };
    let root_resolver = registry.try_resolver(&base).ok()?;
    let root = root_resolver.lookup(&base).ok()?.contents();
    // (schema, the resolver of the schema it sits in or is referenced from)
    let mut stack: Vec<(ResourceRef<'_>, Resolver<'_>)> =
        vec![(resource_ref(root, draft), root_resolver)];
    let mut entered = std::collections::HashSet::new();
    while let Some((res, parent)) = stack.pop() {
        let Ok(resolver) = parent.in_subresource(res) else {
            continue;
        };
        let contents = res.contents();
        if let Some(r) = contents.get("$ref").and_then(Value::as_str) {
            match resolver.lookup(r) {
                Ok(target) => {
                    let target = target.contents();
                    if entered.insert(std::ptr::from_ref(target)) {
                        stack.push((resource_ref(target, res.draft()), resolver.clone()));
                    }
                }
                Err(Error::Unretrievable { .. }) => return Some(outside(r)),
                Err(_) => {}
            }
        }
        for sub in res.draft().subresources_of(contents) {
            stack.push((resource_ref(sub, res.draft()), resolver.clone()));
        }
    }
    None
}

/// `contents` as a resource, its draft read as llguidance reads it: its own
/// `$schema`, else `draft`, an unknown one the latest.
fn resource_ref(contents: &Value, draft: Draft) -> ResourceRef<'_> {
    draft
        .detect(contents)
        .unwrap_or(Draft::Draft202012)
        .create_resource_ref(contents)
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
            (
                json!({"$id": "urn:x", "properties": {"n": {"$ref": "urn:z"}}}),
                "urn:z",
            ),
            // Inside a ref target in data, which llguidance builds.
            (
                json!({"properties": {"n": {"$ref": "#/default/t"}}, "default": {"t": {"$ref": "https://example.com/t.json"}}}),
                "https://example.com/t.json",
            ),
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
            // A ref to nothing is llguidance's own error.
            json!({"properties": {"n": {"$ref": "#/nope"}}}),
        ] {
            assert_eq!(ref_outside_the_schema(&schema), None, "{schema}");
        }
    }
}
