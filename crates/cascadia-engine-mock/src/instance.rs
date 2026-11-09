//! Deterministic minimal instance of a JSON Schema.

use std::collections::BTreeSet;

use serde_json::{Map, Number, Value};

/// Nesting beyond this many levels (each `$ref` hop counts as one) yields `null`.
const MAX_DEPTH: usize = 16;
/// Every item count and string length is clamped to this.
const MAX_COUNT: u64 = 64;
/// Total size budget for one instance: every emitted value costs one unit and
/// every string or key character costs one more. Running out yields `{}`.
const SIZE_BUDGET: usize = 64 * 1024;

/// Build a small, deterministic JSON value that matches `schema`.
///
/// Rules, applied in this order:
/// - `const` returns its value; `enum` returns its first member.
/// - A local `$ref` (`#`, `#/$defs/X`, `#/definitions/X`, any JSON pointer
///   into the root document) is resolved; anything else is treated as `{}`.
/// - `anyOf` / `oneOf` use the first branch that is not `{"type":"null"}`,
///   or `null` when every branch is.
/// - A single-member `allOf` is that member; a larger one merges the
///   members' `properties` and `required` into one object schema.
/// - `type` (the first non-`null` entry of a type array):
///   object emits every `required` property and nothing else; array emits
///   `minItems` copies of `items`; string emits `minLength` times `"a"`;
///   integer and number emit `minimum`, else `exclusiveMinimum` + 1, else 0
///   (integers are emitted as integers); boolean emits `false`; null emits
///   `null`. A schema without a type emits `{}`.
///
/// Limits: nesting deeper than 16 yields `null`; counts and lengths are
/// clamped to 64; when the whole instance would exceed about 64 KiB the
/// result is `{}`. The budget is checked while generating, so a schema whose
/// minimal instance grows exponentially still returns quickly.
///
/// Not covered (the instance may not validate when these constrain it):
/// `pattern`, `format`, `multipleOf`, `uniqueItems`, `prefixItems`,
/// `patternProperties`, `dependentRequired`, `if`/`then`/`else`, `not`.
pub fn minimal_instance(schema: &Value) -> Value {
    let mut generator = Generator {
        root: schema,
        budget: SIZE_BUDGET,
    };
    generator
        .instance(schema, 0)
        .unwrap_or_else(|OutOfBudget| Value::Object(Map::new()))
}

/// The size budget ran out; the whole instance is abandoned.
struct OutOfBudget;

struct Generator<'a> {
    root: &'a Value,
    budget: usize,
}

impl<'a> Generator<'a> {
    fn spend(&mut self, units: usize) -> Result<(), OutOfBudget> {
        self.budget = self.budget.checked_sub(units).ok_or(OutOfBudget)?;
        Ok(())
    }

    fn instance(&mut self, schema: &Value, depth: usize) -> Result<Value, OutOfBudget> {
        self.spend(1)?;
        if depth > MAX_DEPTH {
            return Ok(Value::Null);
        }
        let Some(obj) = schema.as_object() else {
            // Boolean schemas (and anything malformed) carry no shape.
            return Ok(Value::Object(Map::new()));
        };
        if let Some(c) = obj.get("const") {
            return self.copy(c);
        }
        if let Some(first) = obj
            .get("enum")
            .and_then(Value::as_array)
            .and_then(|e| e.first())
        {
            return self.copy(first);
        }
        if let Some(r) = obj.get("$ref").and_then(Value::as_str) {
            return match self.resolve(r) {
                Some(target) => self.instance(target, depth + 1),
                None => Ok(Value::Object(Map::new())),
            };
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(branches) = obj.get(key).and_then(Value::as_array) {
                return match branches.iter().find(|b| !is_null_schema(b)) {
                    Some(b) => self.instance(b, depth + 1),
                    None => Ok(Value::Null),
                };
            }
        }
        if let Some(members) = obj.get("allOf").and_then(Value::as_array) {
            if members.len() == 1 {
                return self.instance(&members[0], depth + 1);
            }
            return self.merged_object(members, depth);
        }
        match first_type(obj) {
            Some("object") => self.object(obj, depth),
            Some("array") => self.array(obj, depth),
            Some("string") => {
                let n = clamped_count(obj.get("minLength"));
                self.spend(n)?;
                Ok(Value::String("a".repeat(n)))
            }
            Some("integer") => Ok(integer_minimum(obj)),
            Some("number") => Ok(number_minimum(obj)),
            Some("boolean") => Ok(Value::Bool(false)),
            Some("null") => Ok(Value::Null),
            _ => Ok(Value::Object(Map::new())),
        }
    }

    fn object(&mut self, obj: &Map<String, Value>, depth: usize) -> Result<Value, OutOfBudget> {
        let props = obj.get("properties").and_then(Value::as_object);
        let required = obj.get("required").and_then(Value::as_array);
        let names = required.into_iter().flatten().filter_map(Value::as_str);
        let mut out = Map::new();
        for name in names {
            // Every entry costs at least one unit, so a long `required` of
            // empty or repeated names is not free.
            self.spend(1 + name.len())?;
            if out.contains_key(name) {
                continue;
            }
            let value = match props.and_then(|p| p.get(name)) {
                Some(s) => self.instance(s, depth + 1)?,
                None => Value::Object(Map::new()),
            };
            out.insert(name.to_string(), value);
        }
        Ok(Value::Object(out))
    }

    fn array(&mut self, obj: &Map<String, Value>, depth: usize) -> Result<Value, OutOfBudget> {
        let n = clamped_count(obj.get("minItems"));
        let items = obj.get("items");
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let item = match items {
                Some(Value::Array(tuple)) => tuple.get(i),
                other => other,
            };
            out.push(match item {
                Some(s) => self.instance(s, depth + 1)?,
                None => {
                    self.spend(1)?;
                    Value::Object(Map::new())
                }
            });
        }
        Ok(Value::Array(out))
    }

    /// Multi-member `allOf`: merge the object members' `properties` and `required`.
    ///
    /// Members are borrowed, never copied, and every property entry and
    /// `required` entry a visit merges is charged to the budget, so a wide
    /// `allOf` reached many times through a recursive `$ref` runs out of
    /// budget instead of re-walking all its properties on every visit.
    fn merged_object(&mut self, members: &[Value], depth: usize) -> Result<Value, OutOfBudget> {
        let mut objects: Vec<&Map<String, Value>> = Vec::with_capacity(members.len());
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut required: Vec<&str> = Vec::new();
        for member in members {
            self.spend(1)?;
            let Some(m) = self.follow_refs(member).as_object() else {
                continue;
            };
            if let Some(p) = m.get("properties").and_then(Value::as_object) {
                self.spend(p.len())?;
            }
            for name in m
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                self.spend(1)?;
                if let Some(name) = name.as_str() {
                    if seen.insert(name) {
                        required.push(name);
                    }
                }
            }
            objects.push(m);
        }
        let mut out = Map::new();
        for name in required {
            self.spend(name.len())?;
            // The first member that defines the property wins.
            let schema = objects.iter().find_map(|m| {
                m.get("properties")
                    .and_then(Value::as_object)
                    .and_then(|p| p.get(name))
            });
            let value = match schema {
                Some(s) => self.instance(s, depth + 1)?,
                None => Value::Object(Map::new()),
            };
            out.insert(name.to_string(), value);
        }
        Ok(Value::Object(out))
    }

    /// Follow a chain of `$ref`s (bounded) to the schema it names.
    fn follow_refs<'s>(&self, mut schema: &'s Value) -> &'s Value
    where
        'a: 's,
    {
        for _ in 0..MAX_DEPTH {
            match schema
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|r| self.resolve(r))
            {
                Some(target) => schema = target,
                None => break,
            }
        }
        schema
    }

    fn resolve(&self, reference: &str) -> Option<&'a Value> {
        let pointer = reference.strip_prefix('#')?;
        if pointer.is_empty() {
            return Some(self.root);
        }
        self.root.pointer(pointer)
    }

    /// Copy a literal (`const` / `enum`) into the output, charging the budget.
    fn copy(&mut self, v: &Value) -> Result<Value, OutOfBudget> {
        self.spend(literal_size(v))?;
        Ok(v.clone())
    }
}

fn literal_size(v: &Value) -> usize {
    match v {
        Value::String(s) => s.len(),
        Value::Array(a) => a.iter().map(|x| 1 + literal_size(x)).sum(),
        Value::Object(o) => o.iter().map(|(k, x)| 1 + k.len() + literal_size(x)).sum(),
        _ => 0,
    }
}

fn is_null_schema(schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(t)) => t == "null",
        Some(Value::Array(ts)) => !ts.is_empty() && ts.iter().all(|t| t == "null"),
        _ => false,
    }
}

/// The schema's type, taking the first non-`null` entry of a type array
/// (or `null` when that is the only entry).
fn first_type(obj: &Map<String, Value>) -> Option<&str> {
    match obj.get("type")? {
        Value::String(t) => Some(t),
        Value::Array(ts) => {
            let mut names = ts.iter().filter_map(Value::as_str);
            let first = names.clone().next();
            names.find(|t| *t != "null").or(first)
        }
        _ => None,
    }
}

fn clamped_count(v: Option<&Value>) -> usize {
    v.and_then(Value::as_u64).unwrap_or(0).min(MAX_COUNT) as usize
}

/// The lower bound for a numeric schema: `minimum`, else `exclusiveMinimum` + 1
/// (also the draft-04 boolean form next to `minimum`).
fn lower_bound(obj: &Map<String, Value>) -> Option<(&Number, bool)> {
    let exclusive_flag = obj.get("exclusiveMinimum").and_then(Value::as_bool) == Some(true);
    if let Some(Value::Number(n)) = obj.get("minimum") {
        return Some((n, exclusive_flag));
    }
    match obj.get("exclusiveMinimum") {
        Some(Value::Number(n)) => Some((n, true)),
        _ => None,
    }
}

fn integer_minimum(obj: &Map<String, Value>) -> Value {
    let v = match lower_bound(obj) {
        None => 0,
        Some((n, exclusive)) => {
            let base = if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                // A fractional bound: the smallest integer that meets it. An
                // exclusive bound that is not whole is already met by its
                // ceiling, so only a whole one needs the + 1 below.
                let c = f.ceil();
                if exclusive && c > f {
                    return Value::from(c as i64);
                }
                c as i64
            } else {
                // A u64 beyond i64: emit it unchanged.
                return Value::Number(n.clone());
            };
            if exclusive {
                base.saturating_add(1)
            } else {
                base
            }
        }
    };
    Value::from(v)
}

fn number_minimum(obj: &Map<String, Value>) -> Value {
    match lower_bound(obj) {
        None => Value::from(0),
        Some((n, false)) => Value::Number(n.clone()),
        Some((n, true)) => {
            if let Some(i) = n.as_i64() {
                Value::from(i.saturating_add(1))
            } else {
                n.as_f64()
                    .and_then(|f| Number::from_f64(f + 1.0))
                    .map(Value::Number)
                    .unwrap_or_else(|| Value::Number(n.clone()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{Duration, Instant};

    const BROWSER_USE_SCHEMA: &str = include_str!("../tests/fixtures/browser_use_schema.json");
    const PYDANTIC_OUTPUT_SCHEMA: &str =
        include_str!("../tests/fixtures/pydantic_output_schema.json");

    fn assert_valid(schema: &Value) {
        let instance = minimal_instance(schema);
        let validator = jsonschema::validator_for(schema).unwrap();
        assert!(
            validator.is_valid(&instance),
            "instance {instance} does not validate against {schema}"
        );
    }

    #[test]
    fn const_is_returned_as_is() {
        let s = json!({"type": "string", "const": "fixed"});
        assert_eq!(minimal_instance(&s), json!("fixed"));
    }

    #[test]
    fn enum_returns_first_member() {
        let s = json!({"type": "string", "enum": ["NP", "IN", "US"]});
        assert_eq!(minimal_instance(&s), json!("NP"));
    }

    #[test]
    fn ref_into_defs_is_resolved() {
        let s = json!({
            "$defs": {"Id": {"type": "integer", "minimum": 7}},
            "$ref": "#/$defs/Id"
        });
        assert_eq!(minimal_instance(&s), json!(7));
    }

    #[test]
    fn ref_into_definitions_is_resolved() {
        let s = json!({
            "definitions": {"Name": {"type": "string", "minLength": 2}},
            "type": "object",
            "properties": {"n": {"$ref": "#/definitions/Name"}},
            "required": ["n"]
        });
        assert_eq!(minimal_instance(&s), json!({"n": "aa"}));
    }

    #[test]
    fn any_of_skips_a_leading_null_branch() {
        let s = json!({"anyOf": [{"type": "null"}, {"type": "boolean"}]});
        assert_eq!(minimal_instance(&s), json!(false));
        let only_null = json!({"oneOf": [{"type": "null"}]});
        assert_eq!(minimal_instance(&only_null), Value::Null);
    }

    #[test]
    fn all_of_merges_object_members() {
        let s = json!({"allOf": [
            {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}
        ]});
        assert_eq!(minimal_instance(&s), json!({"a": 0, "b": ""}));
        let single = json!({"allOf": [{"type": "integer", "minimum": 3}]});
        assert_eq!(minimal_instance(&single), json!(3));
    }

    #[test]
    fn type_array_takes_first_non_null_type() {
        let s = json!({"type": ["null", "string"]});
        assert_eq!(minimal_instance(&s), json!(""));
    }

    #[test]
    fn object_emits_only_required_properties() {
        let s = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a"]
        });
        assert_eq!(minimal_instance(&s), json!({"a": 0}));
    }

    #[test]
    fn array_emits_min_items_copies() {
        let s = json!({"type": "array", "minItems": 2, "items": {"type": "integer"}});
        assert_eq!(minimal_instance(&s), json!([0, 0]));
        let empty = json!({"type": "array", "items": {"type": "integer"}});
        assert_eq!(minimal_instance(&empty), json!([]));
    }

    #[test]
    fn string_emits_min_length_letters() {
        let s = json!({"type": "string", "minLength": 3});
        assert_eq!(minimal_instance(&s), json!("aaa"));
    }

    #[test]
    fn numbers_honour_minimum_and_exclusive_minimum() {
        assert_eq!(
            minimal_instance(&json!({"type": "integer", "minimum": 5})),
            json!(5)
        );
        assert_eq!(
            minimal_instance(&json!({"type": "integer", "exclusiveMinimum": 5})),
            json!(6)
        );
        assert_eq!(
            minimal_instance(&json!({"type": "number", "minimum": 1.5})),
            json!(1.5)
        );
        assert_eq!(
            minimal_instance(&json!({"type": "number", "exclusiveMinimum": 1.5})),
            json!(2.5)
        );
        assert_eq!(minimal_instance(&json!({"type": "integer"})), json!(0));
        let int = minimal_instance(&json!({"type": "integer"}));
        assert!(int.is_i64(), "integers are emitted as integers: {int}");
    }

    #[test]
    fn boolean_null_and_untyped() {
        assert_eq!(minimal_instance(&json!({"type": "boolean"})), json!(false));
        assert_eq!(minimal_instance(&json!({"type": "null"})), Value::Null);
        assert_eq!(minimal_instance(&json!({})), json!({}));
    }

    #[test]
    fn self_recursive_ref_stops_at_depth_limit() {
        let s = json!({
            "type": "object",
            "properties": {"next": {"$ref": "#"}},
            "required": ["next"]
        });
        let mut v = &minimal_instance(&s);
        let mut objects = 0;
        while let Some(next) = v.get("next") {
            objects += 1;
            v = next;
        }
        assert!(v.is_null(), "the innermost value is null: {v}");
        assert!(
            (1..=16).contains(&objects),
            "nesting is bounded by the depth limit: {objects}"
        );
        // A bare reference cycle terminates too.
        let cycle = json!({"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"});
        assert_eq!(minimal_instance(&cycle), Value::Null);
    }

    #[test]
    fn huge_min_items_is_clamped_to_64() {
        let s =
            json!({"type": "array", "minItems": 1_000_000_000u64, "items": {"type": "integer"}});
        assert_eq!(minimal_instance(&s).as_array().unwrap().len(), 64);
    }

    #[test]
    fn huge_min_length_is_clamped_to_64() {
        let s = json!({"type": "string", "minLength": 4_000_000_000u64});
        assert_eq!(minimal_instance(&s).as_str().unwrap().len(), 64);
    }

    #[test]
    fn exponential_schema_exhausts_the_size_budget_quickly() {
        let s = json!({
            "$defs": {"A": {"type": "array", "minItems": 64, "items": {"$ref": "#/$defs/A"}}},
            "$ref": "#/$defs/A"
        });
        let start = Instant::now();
        let v = minimal_instance(&s);
        let took = start.elapsed();
        assert_eq!(v, json!({}));
        assert!(
            took < Duration::from_millis(100),
            "budget is checked while generating: {took:?}"
        );
    }

    #[test]
    fn wide_all_of_inside_a_recursive_array_exhausts_the_budget_quickly() {
        // A multi-member allOf with many properties, reached through a
        // self-referencing array: each visit must pay for the properties it
        // merges, or the walk crawls through every visit before giving up.
        let mut props = serde_json::Map::new();
        props.insert(
            "x".into(),
            json!({"type": "array", "minItems": 64, "items": {"$ref": "#/$defs/A"}}),
        );
        for i in 0..2500 {
            props.insert(format!("p{i}"), json!({"type": "string"}));
        }
        let s = json!({
            "$defs": {"A": {"allOf": [
                {"properties": props, "required": ["x"]},
                {}
            ]}},
            "$ref": "#/$defs/A"
        });
        let start = Instant::now();
        let v = minimal_instance(&s);
        let took = start.elapsed();
        assert_eq!(v, json!({}));
        assert!(
            took < Duration::from_millis(250),
            "merged allOf members are charged to the budget: {took:?}"
        );
    }

    #[test]
    fn empty_required_names_are_charged_to_the_budget() {
        let mut required = vec![json!("x"), json!("y")];
        required.extend(std::iter::repeat_n(json!(""), 21_000));
        let s = json!({
            "$defs": {"N": {
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/N"}, "y": {"$ref": "#/$defs/N"}},
                "required": required,
            }},
            "$ref": "#/$defs/N"
        });
        let start = Instant::now();
        let v = minimal_instance(&s);
        let took = start.elapsed();
        assert_eq!(v, json!({}), "the budget runs out");
        assert!(
            took < Duration::from_millis(250),
            "every required name costs budget: {took:?}"
        );
    }

    #[test]
    fn repeated_required_names_are_emitted_once() {
        let s = json!({
            "type": "object",
            "properties": {"a": {"type": "string", "minLength": 2}},
            "required": ["a", "a", "a"],
        });
        assert_eq!(minimal_instance(&s), json!({"a": "aa"}));
    }

    #[test]
    fn browser_use_style_schema_instance_validates() {
        let schema: Value = serde_json::from_str(BROWSER_USE_SCHEMA).unwrap();
        assert_valid(&schema);
    }

    #[test]
    fn pydantic_output_style_schema_instance_validates() {
        let schema: Value = serde_json::from_str(PYDANTIC_OUTPUT_SCHEMA).unwrap();
        assert_valid(&schema);
    }
}
