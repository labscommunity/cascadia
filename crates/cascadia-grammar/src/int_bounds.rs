//! Integer bounds added to a schema before a grammar backend sees it.

use serde_json::{Map, Value};

/// The largest magnitude an unbounded `integer` may take: 18 digits.
pub const INTEGER_DIGITS_CAP: i64 = 999_999_999_999_999_999;

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
/// `maximum: 10^18 - 1`, and without a lower bound `minimum: -(10^18 - 1)`.
/// Both JSON grammars (xgrammar, llguidance) allow an integer of any length,
/// and a small model in JSON mode can repeat digits until it runs out of
/// tokens; 18 digits are more than an `i64` id or a count needs. Only
/// narrows, so every output still matches the client's schema.
///
/// Left as they are: `number` (llguidance can't bound a fraction's length
/// and xgrammar already caps it at 6 digits; a bound on a type list that
/// holds `number` would bound its numbers too), a schema with `enum`,
/// `const` or `multipleOf` (a bound can leave no multiple in range), a side whose other bound is past the cap (it would leave no
/// value), and anything under `not` or `if`.
///
/// Returns how many schemas it bounded.
pub fn bound_integers(schema: &mut Value) -> usize {
    let mut bounded = 0;
    visit_schemas(schema, &mut |obj| {
        if !is_integer_only(obj)
            || ["enum", "const", "multipleOf"]
                .iter()
                .any(|k| obj.contains_key(*k))
        {
            return;
        }
        let cap = INTEGER_DIGITS_CAP as f64;
        let (lower, upper) = (lower_bound(obj), upper_bound(obj));
        let mut changed = false;
        if upper.is_none() && lower.is_none_or(|l| l < cap) {
            obj.insert("maximum".into(), INTEGER_DIGITS_CAP.into());
            changed = true;
        }
        if lower.is_none() && upper.is_none_or(|u| u > -cap) {
            obj.insert("minimum".into(), (-INTEGER_DIGITS_CAP).into());
            changed = true;
        }
        bounded += usize::from(changed);
    });
    bounded
}

/// Raises an `integer` schema's `minimum`/`exclusiveMinimum` below what
/// llguidance builds (about -2^63, e.g. C# `long.MinValue`) to the lowest it
/// builds, when that leaves the range non-empty. Only narrows.
pub(crate) fn clamp_for_llguidance(schema: &mut Value) {
    visit_schemas(schema, &mut |obj| {
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

/// Calls `f` on every object in `v` reachable through schema keywords, maps
/// (`properties`, `$defs`, ...) and lists (`anyOf`, `prefixItems`, ...).
/// A map of schemas is visited like a schema; it holds no `type` string or
/// list, so `f` leaves it alone.
fn visit_schemas(v: &mut Value, f: &mut impl FnMut(&mut Map<String, Value>)) {
    match v {
        Value::Object(obj) => {
            f(obj);
            for (k, child) in obj.iter_mut() {
                if !SKIP_KEYWORDS.contains(&k.as_str()) {
                    visit_schemas(child, f);
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

    const CAP: i64 = INTEGER_DIGITS_CAP;

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
        let both = json!({"type": "integer", "maximum": CAP, "minimum": -CAP});
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
            json!({"type": ["integer", "null"], "maximum": CAP, "minimum": -CAP})
        );
    }

    #[test]
    fn a_bound_the_client_set_is_kept() {
        for (schema, want) in [
            (
                json!({"type": "integer", "minimum": 0}),
                json!({"type": "integer", "minimum": 0, "maximum": CAP}),
            ),
            (
                json!({"type": "integer", "exclusiveMaximum": 10}),
                json!({"type": "integer", "exclusiveMaximum": 10, "minimum": -CAP}),
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
        // A lower bound past the cap: an upper bound at the cap would leave no value.
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
        // A bound at the cap can leave no multiple in range: this accepts
        // 1000000000000000302 and nothing at or below the cap.
        let schema =
            json!({"type": "integer", "minimum": 999999999999999360u64, "multipleOf": 1009});
        assert_eq!(bounded(schema.clone()), schema);
        let schema = json!({"type": "integer", "multipleOf": 2});
        assert_eq!(bounded(schema.clone()), schema);
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
        // Nothing would be left: left for llguidance to refuse.
        let mut v = json!({"type": "integer", "minimum": -1e20, "maximum": -1e19});
        clamp_for_llguidance(&mut v);
        assert_eq!(
            v,
            json!({"type": "integer", "minimum": -1e20, "maximum": -1e19})
        );
    }
}
