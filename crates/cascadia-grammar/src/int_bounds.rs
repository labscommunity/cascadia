//! Integer bounds added to a schema before a grammar backend sees it.

use crate::refs::local_ref_pointers;
use crate::NAME_MAPS;
use serde_json::{Map, Value};

/// The `maximum` given to an `integer` schema that sets no upper bound:
/// `i64::MAX`, 19 digits. Neither backend builds an integer bound past it
/// (llguidance reads bounds as f64 and saturates them to `i64`; xgrammar
/// refuses a bound past `int64` with "Integer exceeds maximum limit"), so a
/// non-negative integer is capped here too, not at `u64::MAX`.
pub const ADDED_INTEGER_MAXIMUM: i64 = i64::MAX;

/// The `minimum` given to an `integer` schema that sets no lower bound:
/// `-i64::MAX`, one above `i64::MIN`, whose negation overflows in xgrammar's
/// range builder. The llguidance build raises it further, to
/// [`LLGUIDANCE_MIN_INTEGER`].
pub const ADDED_INTEGER_MINIMUM: i64 = -i64::MAX;

/// The lowest integer lower bound llguidance builds: it reads bounds as f64,
/// and fails to build the range for a lower bound at or below -2^63.
const LLGUIDANCE_MIN_INTEGER: i64 = -(i64::MAX - 1023);

/// Keywords whose values are data or whose subschema must not be narrowed:
/// narrowing what `not` or `if` matches widens or changes what the schema
/// accepts.
const SKIP_KEYWORDS: &[&str] = &[
    "const",
    "enum",
    "default",
    "examples",
    "x-guidance",
    "not",
    "if",
];

/// Gives every `integer` schema in `schema` without an upper bound
/// `maximum:` [`ADDED_INTEGER_MAXIMUM`], and without a lower bound
/// `minimum:` [`ADDED_INTEGER_MINIMUM`]: the `i64` range, 19 digits.
/// Both JSON grammars (xgrammar, llguidance) allow an integer of any length,
/// and a small model in JSON mode can repeat digits until it runs out of
/// tokens; the `i64` range still admits every `i64` id (a Twitter/X or
/// Discord snowflake, `i64::MAX`).
///
/// The target of a local `$ref` is bounded too, inside a data keyword
/// (`default`, `examples`) as well: llguidance builds it as a schema.
///
/// Where a schema is used directly, this only narrows it. A `$defs` entry
/// is bounded once for every place that names it, so a `$ref` reached from
/// under `not`, `if`, `contains` with `maxContains`, or one member of a
/// `oneOf`, lands on the bounded copy, and what those keywords match can
/// widen. Neither backend enforces those keywords (the API flags `not`,
/// `if`, `contains` and `maxContains` as not enforced, and `oneOf` is
/// matched as `anyOf`), so the outputs either backend allows are unchanged.
///
/// Left as they are: `number` (llguidance can't bound a fraction's length
/// and xgrammar already caps it at 6 digits; a bound on a type list that
/// holds `number` would bound its numbers too), a schema with `enum`,
/// `const` or `multipleOf` (a bound can leave no multiple in range), one
/// with a range keyword that is not a number (left for the backend to
/// refuse), a side whose other bound is past the added one (it would leave no
/// value), and anything under `not` or `if`.
///
/// Returns how many schemas it bounded.
pub fn bound_integers(schema: &mut Value) -> usize {
    let mut bounded = 0;
    visit_schemas_and_ref_targets(schema, &mut |obj| {
        if !is_integer_only(obj)
            || ["enum", "const", "multipleOf"]
                .iter()
                .any(|k| obj.contains_key(*k))
            || BOUND_KEYWORDS
                .iter()
                .any(|k| obj.get(*k).is_some_and(|b| !b.is_number()))
        {
            return;
        }
        let (max, min) = (ADDED_INTEGER_MAXIMUM as f64, ADDED_INTEGER_MINIMUM as f64);
        let (lower, upper) = (lower_bound(obj), upper_bound(obj));
        let mut changed = false;
        if upper.is_none() && lower.is_none_or(|l| l < max) {
            obj.insert("maximum".into(), ADDED_INTEGER_MAXIMUM.into());
            changed = true;
        }
        if lower.is_none() && upper.is_none_or(|u| u > min) {
            obj.insert("minimum".into(), ADDED_INTEGER_MINIMUM.into());
            changed = true;
        }
        bounded += usize::from(changed);
    });
    bounded
}

/// Raises an `integer` schema's `minimum`/`exclusiveMinimum` below what
/// llguidance builds (about -2^63, e.g. C# `long.MinValue`) to the lowest it
/// builds, when that leaves the range non-empty. Only narrows. A type list
/// holding `number` too is raised as well: llguidance builds each type on
/// its own, and fails to build a `number` range that low either, so no
/// number it could admit is lost.
pub(crate) fn clamp_for_llguidance(schema: &mut Value) {
    visit_schemas_and_ref_targets(schema, &mut |obj| {
        if !has_integer(obj) {
            return;
        }
        let floor = LLGUIDANCE_MIN_INTEGER as f64;
        if upper_bound(obj).is_some_and(|u| u <= floor) {
            return;
        }
        for key in ["minimum", "exclusiveMinimum"] {
            if obj
                .get(key)
                .and_then(Value::as_f64)
                .is_some_and(|m| m < floor)
            {
                obj.insert(key.into(), LLGUIDANCE_MIN_INTEGER.into());
            }
        }
    });
}

/// Whether `schema` has a number llguidance builds as an integer and can't
/// build: a `const` or `enum` number, or a `maximum`/`exclusiveMaximum`, at
/// or below -2^63 as an f64 (so -9223372036854775807 counts: it rounds to
/// -2^63). llguidance saturates integer bounds to `i64`, and building a
/// range whose ends are both `i64::MIN` negates them and recurses until the
/// build thread's stack overflows, aborting the process.
///
/// llguidance builds such a number as an integer when it is an integer
/// literal that fits `i64`, or when it meets an `integer` type: its own, a
/// type list holding `integer`, or one it is intersected with through
/// `allOf`, a `$ref` or properties merged from two schemas. That last can
/// come from anywhere in the document, so a `number` this low (`-1e20`) is
/// refused only when the document names the `integer` type somewhere; a
/// `number` schema in a document without one builds as before.
///
/// Every object in the document is read, data keywords (`default`,
/// `examples`, objects inside `const` and `enum`, unknown keywords)
/// included, not only the schemas the ref and id resolution says
/// llguidance builds: a backstop for a way to make llguidance build data as
/// a schema that the resolution mirror (`refs`) misses. The cost is a false
/// 400 for a document with an `integer` type (or an `i64` literal this low)
/// and a data object shaped like such a schema (`{"const": ...}`,
/// `{"maximum": ...}`), which no real client sends. A bare low number in
/// data is not read.
pub(crate) fn has_integer_llguidance_cannot_build(schema: &Value) -> bool {
    let any_integer = names_integer_type(schema);
    let at_or_below_min = |v: &Value| -> bool {
        let mut stack = vec![v];
        while let Some(v) = stack.pop() {
            match v {
                Value::Number(n) => {
                    if n.as_f64().is_some_and(|f| f <= I64_MIN_F64) && (any_integer || n.is_i64()) {
                        return true;
                    }
                }
                Value::Array(items) => stack.extend(items),
                Value::Object(obj) => stack.extend(obj.values()),
                _ => {}
            }
        }
        false
    };
    let mut stack = vec![schema];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(obj) => {
                for (k, child) in obj {
                    let low = match k.as_str() {
                        "const" | "enum" => at_or_below_min(child),
                        // An upper bound at or below the lowest bound
                        // llguidance builds leaves nothing above the
                        // raised minimum (`clamp_for_llguidance` skips
                        // it): refused here, not with llguidance's text.
                        "maximum" | "exclusiveMaximum" => {
                            any_integer
                                && child
                                    .as_f64()
                                    .is_some_and(|m| m <= LLGUIDANCE_MIN_INTEGER as f64)
                        }
                        _ => false,
                    };
                    if low {
                        return true;
                    }
                    stack.push(child);
                }
            }
            Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
    false
}

/// Whether any `type` in `v` is `"integer"` or a list holding it, data
/// keywords included (a `$ref` can make one a schema).
fn names_integer_type(v: &Value) -> bool {
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        match v {
            Value::Object(obj) => {
                if obj.get("type").is_some_and(|t| match t {
                    Value::String(t) => t == "integer",
                    Value::Array(ts) => ts.iter().any(|t| t == "integer"),
                    _ => false,
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

/// `i64::MIN` as an f64: -2^63.
const I64_MIN_F64: f64 = i64::MIN as f64;

/// The tighter of `minimum` and `exclusiveMinimum`, if either is a number.
fn lower_bound(obj: &Map<String, Value>) -> Option<f64> {
    let get = |k: &str| obj.get(k).and_then(Value::as_f64);
    match (get("minimum"), get("exclusiveMinimum")) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// The tighter of `maximum` and `exclusiveMaximum`, if either is a number.
fn upper_bound(obj: &Map<String, Value>) -> Option<f64> {
    let get = |k: &str| obj.get(k).and_then(Value::as_f64);
    match (get("maximum"), get("exclusiveMaximum")) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// `type` is `"integer"`, or a list holding `"integer"` and not `"number"`.
fn is_integer_only(obj: &Map<String, Value>) -> bool {
    match obj.get("type") {
        Some(Value::String(t)) => t == "integer",
        Some(Value::Array(ts)) => {
            ts.iter().any(|t| t == "integer") && !ts.iter().any(|t| t == "number")
        }
        _ => false,
    }
}

/// `type` is or holds `"integer"`.
fn has_integer(obj: &Map<String, Value>) -> bool {
    match obj.get("type") {
        Some(Value::String(t)) => t == "integer",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "integer"),
        _ => false,
    }
}

/// The numeric range keywords. One that is not a number is the backend's
/// to refuse: a bound added beside it would hide it.
const BOUND_KEYWORDS: [&str; 4] = ["minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"];

/// [`visit_schemas`] on `schema`, then on the target of each `#/...` `$ref`
/// in it, resolved the way llguidance resolves it ([`local_ref_pointers`]).
/// llguidance builds whatever a `$ref` points to as a schema, inside a data
/// keyword (`default`, `examples`) too, which the walk skips. Refs that could
/// resolve elsewhere (another base URI, an anchor) are refused before this
/// runs (`refs::unfollowable_ref`). A
/// target the walk already visited is visited again: `f` must leave a
/// schema it already changed as it is.
fn visit_schemas_and_ref_targets(schema: &mut Value, f: &mut impl FnMut(&mut Map<String, Value>)) {
    visit_schemas(schema, f);
    for pointer in local_ref_pointers(schema) {
        if let Some(target) = schema.pointer_mut(&pointer) {
            visit_schemas(target, f);
        }
    }
}

/// Calls `f` on every schema object in `v`: through subschema keywords,
/// lists (`anyOf`, `prefixItems`, ...) and the schemas of name maps
/// (`properties`, `$defs`, ...), never into [`SKIP_KEYWORDS`].
fn visit_schemas(v: &mut Value, f: &mut impl FnMut(&mut Map<String, Value>)) {
    match v {
        Value::Object(obj) => {
            f(obj);
            for (k, child) in obj.iter_mut() {
                let k = k.as_str();
                if SKIP_KEYWORDS.contains(&k) {
                    continue;
                }
                match child {
                    Value::Object(map) if NAME_MAPS.contains(&k) => {
                        for schema in map.values_mut() {
                            visit_schemas(schema, f);
                        }
                    }
                    _ => visit_schemas(child, f),
                }
            }
        }
        Value::Array(items) => {
            for child in items {
                visit_schemas(child, f);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MAX: i64 = ADDED_INTEGER_MAXIMUM;
    const MIN: i64 = ADDED_INTEGER_MINIMUM;

    fn bounded(mut v: Value) -> Value {
        bound_integers(&mut v);
        v
    }

    #[test]
    fn it_counts_the_schemas_it_bounded() {
        let int = json!({"type": "integer"});
        let mut v = json!({"properties": {"a": int, "b": int, "c": {"type": "integer", "minimum": 0, "maximum": 1}}});
        assert_eq!(bound_integers(&mut v), 2);
    }

    #[test]
    fn every_unbounded_integer_gets_both_bounds_wherever_it_sits() {
        let int = json!({"type": "integer"});
        let both = json!({"type": "integer", "maximum": MAX, "minimum": MIN});
        let schema = json!({
            "type": "object",
            "properties": {"a": int, "b": {"type": "array", "items": int}},
            "additionalProperties": int,
            "patternProperties": {"^x": int},
            "anyOf": [int, {"oneOf": [int]}, {"allOf": [int]}],
            "prefixItems": [int],
            "$defs": {"N": int},
        });
        let want = json!({
            "type": "object",
            "properties": {"a": both, "b": {"type": "array", "items": both}},
            "additionalProperties": both,
            "patternProperties": {"^x": both},
            "anyOf": [both, {"oneOf": [both]}, {"allOf": [both]}],
            "prefixItems": [both],
            "$defs": {"N": both},
        });
        assert_eq!(bounded(schema), want);
        assert_eq!(
            bounded(json!({"type": ["integer", "null"]})),
            json!({"type": ["integer", "null"], "maximum": MAX, "minimum": MIN})
        );
    }

    #[test]
    fn a_bound_the_client_set_is_kept() {
        for (schema, want) in [
            (
                json!({"type": "integer", "minimum": 0}),
                json!({"type": "integer", "minimum": 0, "maximum": MAX}),
            ),
            (
                json!({"type": "integer", "exclusiveMaximum": 10}),
                json!({"type": "integer", "exclusiveMaximum": 10, "minimum": MIN}),
            ),
            (
                json!({"type": "integer", "minimum": -5, "maximum": 1e30}),
                json!({"type": "integer", "minimum": -5, "maximum": 1e30}),
            ),
        ] {
            assert_eq!(bounded(schema), want);
        }
    }

    #[test]
    fn what_a_bound_would_change_is_left_alone() {
        for schema in [
            // number: and an integer-or-number list, whose numbers it would bound.
            json!({"type": "number"}),
            json!({"type": ["integer", "number"]}),
            json!({"type": "integer", "enum": [10_000_000_000_000_000_000u64]}),
            json!({"type": "integer", "const": 3}),
            // Narrowing under `not` or `if` widens or changes what matches.
            json!({"not": {"type": "integer"}}),
            json!({"if": {"type": "integer"}, "then": {"minimum": 0, "maximum": 1}}),
        ] {
            assert_eq!(bounded(schema.clone()), schema);
        }
        // A lower bound past i64::MAX: an added maximum would leave no value.
        assert_eq!(
            bounded(json!({"type": "integer", "minimum": 1e19})),
            json!({"type": "integer", "minimum": 1e19})
        );
        assert_eq!(
            bounded(json!({"type": "integer", "exclusiveMaximum": -1e19})),
            json!({"type": "integer", "exclusiveMaximum": -1e19})
        );
    }

    #[test]
    fn an_integer_with_multiple_of_is_left_alone() {
        // An added bound can leave no multiple in range: this accepts
        // 9223372036854776625 and nothing at or below i64::MAX.
        let schema =
            json!({"type": "integer", "minimum": 9223372036854775700u64, "multipleOf": 1009});
        assert_eq!(bounded(schema.clone()), schema);
        let schema = json!({"type": "integer", "multipleOf": 2});
        assert_eq!(bounded(schema.clone()), schema);
    }

    #[test]
    fn a_property_named_like_a_skipped_keyword_is_still_bounded() {
        let int = json!({"type": "integer"});
        let both = json!({"type": "integer", "maximum": MAX, "minimum": MIN});
        let names = ["default", "enum", "const", "not", "if", "examples"];
        let props = |v: &Value| -> Value {
            Value::Object(names.iter().map(|n| (n.to_string(), v.clone())).collect())
        };
        let schema = json!({"type": "object", "properties": props(&int), "$defs": props(&int)});
        let want = json!({"type": "object", "properties": props(&both), "$defs": props(&both)});
        assert_eq!(bounded(schema), want);
        // The keywords themselves are still skipped.
        let schema = json!({"not": int, "if": int, "default": int, "examples": [int]});
        assert_eq!(bounded(schema.clone()), schema);
    }

    #[test]
    fn a_bound_that_is_not_a_number_is_left_for_the_backend_to_refuse() {
        for schema in [
            json!({"type": "integer", "maximum": null}),
            json!({"type": "integer", "minimum": "0"}),
            json!({"type": "integer", "exclusiveMaximum": true}),
            json!({"type": "integer", "exclusiveMinimum": [1]}),
        ] {
            assert_eq!(bounded(schema.clone()), schema);
        }
    }

    #[test]
    fn an_integer_at_or_below_i64_min_is_found_where_llguidance_builds_it() {
        let low = |v: Value| has_integer_llguidance_cannot_build(&v);
        assert!(low(
            json!({"anyOf": [{"type": "integer", "const": i64::MIN}]})
        ));
        assert!(low(
            json!({"properties": {"a": {"enum": [[{"x": i64::MIN}]]}}})
        ));
        assert!(low(
            json!({"type": "integer", "items": {"exclusiveMaximum": -1e19}})
        ));
        // A number this low that no integer type meets builds as a float.
        assert!(!low(json!({"items": {"exclusiveMaximum": -1e19}})));
        assert!(!low(json!({"type": "number", "const": -1e20})));
        // An i64 literal is an integer wherever it sits.
        assert!(low(
            json!({"type": "number", "const": -9223372036854775807i64})
        ));
        // -2^63 + 1024 is the first i64 above -2^63 as an f64.
        assert!(!low(json!({"const": -9223372036854774784i64})));
        assert!(!low(json!({"examples": [i64::MIN], "default": i64::MIN})));
        // The backstop: an object in data shaped like a low integer schema.
        assert!(low(json!({"default": {"x": {"const": i64::MIN}}})));
        assert!(low(
            json!({"type": "integer", "examples": [{"maximum": -1e20}]})
        ));
        assert!(!low(
            json!({"properties": {"const": {"type": "integer"}, "maximum": {}}})
        ));
        assert!(low(json!({"type": "integer", "allOf": [{"const": -1e19}]})));
        assert!(!low(json!({"allOf": [{"const": -1e19}]})));
        assert!(!low(json!({"const": -1e18})));
    }

    #[test]
    fn a_lower_bound_llguidance_cannot_build_is_raised() {
        let floor = LLGUIDANCE_MIN_INTEGER;
        let mut v = json!({"type": "integer", "minimum": i64::MIN, "maximum": i64::MAX});
        clamp_for_llguidance(&mut v);
        assert_eq!(
            v,
            json!({"type": "integer", "minimum": floor, "maximum": i64::MAX})
        );
        let mut v = json!({"type": ["integer", "null"], "exclusiveMinimum": -1e19});
        clamp_for_llguidance(&mut v);
        assert_eq!(
            v,
            json!({"type": ["integer", "null"], "exclusiveMinimum": floor})
        );
        // Nothing would be left: not raised (the build refuses an upper
        // bound this low; see `has_integer_llguidance_cannot_build`).
        let mut v = json!({"type": "integer", "minimum": -1e20, "maximum": -1e19});
        clamp_for_llguidance(&mut v);
        assert_eq!(
            v,
            json!({"type": "integer", "minimum": -1e20, "maximum": -1e19})
        );
    }
}
