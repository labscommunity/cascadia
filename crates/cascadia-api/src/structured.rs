//! OpenAI `response_format` parsing: turns the request field into a
//! [`ResponseFormat`], maps it to a backend-neutral [`GrammarSpec`], and
//! lists JSON Schema keywords the decoder cannot enforce.

use cascadia_types::{GrammarKind, GrammarSpec};
use serde_json::{Map, Value};

/// Grammar body used for `json_object` (and for `json_schema` whose schema is
/// missing, `null` or `true`): any JSON object.
pub const JSON_OBJECT_SCHEMA: &str = r#"{"type":"object","additionalProperties":true}"#;

/// The output format a chat request asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: Option<String>,
        schema: Value,
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

const TYPE_ERR: &str = "response_format.type must be one of text, json_object, json_schema";

/// Parse the `response_format` field of a chat request. Absent or `null`
/// means plain text.
pub fn parse_response_format(v: Option<&Value>) -> Result<ResponseFormat, String> {
    let obj = match v {
        None | Some(Value::Null) => return Ok(ResponseFormat::Text),
        Some(Value::Object(o)) => o,
        Some(_) => return Err("response_format must be an object".into()),
    };
    match obj.get("type").and_then(Value::as_str) {
        None => Err(TYPE_ERR.into()),
        Some("text") => Ok(ResponseFormat::Text),
        Some("json_object") => Ok(ResponseFormat::JsonObject),
        Some("json_schema") => parse_json_schema(obj.get("json_schema")),
        Some(other) => Err(format!("{TYPE_ERR}, got {}", Value::from(other))),
    }
}

/// Parse `response_format.json_schema`. A missing, `null` or `true` schema
/// asks for any JSON object; `false` and non-object schemas are rejected.
fn parse_json_schema(js: Option<&Value>) -> Result<ResponseFormat, String> {
    let Some(Value::Object(js)) = js else {
        return Err("response_format.json_schema must be an object".into());
    };
    match js.get("schema") {
        None | Some(Value::Null) | Some(Value::Bool(true)) => Ok(ResponseFormat::JsonObject),
        Some(Value::Bool(false)) => {
            Err("response_format.json_schema.schema is false: no output can match it".into())
        }
        Some(schema @ Value::Object(_)) => Ok(ResponseFormat::JsonSchema {
            name: js.get("name").and_then(Value::as_str).map(str::to_owned),
            schema: schema.clone(),
            strict: js.get("strict") == Some(&Value::Bool(true)),
        }),
        Some(_) => Err("response_format.json_schema.schema must be a JSON Schema object".into()),
    }
}

/// The grammar that enforces `rf`, or `None` for plain text.
pub fn grammar_for(rf: &ResponseFormat) -> Option<GrammarSpec> {
    let body = match rf {
        ResponseFormat::Text => return None,
        ResponseFormat::JsonObject => JSON_OBJECT_SCHEMA.to_owned(),
        ResponseFormat::JsonSchema { schema, .. } => {
            serde_json::to_string(schema).expect("a serde_json::Value always serializes")
        }
    };
    Some(GrammarSpec {
        kind: GrammarKind::JsonSchema,
        body,
    })
}

const MAX_DEPTH: usize = 64;

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
const SCHEMA_MAPS: [&str; 5] = [
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];

/// Keys whose value is instance data, not a schema.
const DATA_KEYS: [&str; 4] = ["enum", "const", "default", "examples"];

/// JSON Schema keywords in `schema` the output grammar does not enforce,
/// sorted and de-duplicated. Output may still be valid JSON that violates them.
pub fn unenforced_keywords(schema: &Value) -> Vec<&'static str> {
    let mut found = Vec::new();
    walk(schema, 0, &mut found);
    found.sort_unstable();
    found.dedup();
    found
}

fn walk(v: &Value, depth: usize, found: &mut Vec<&'static str>) {
    if depth > MAX_DEPTH {
        return;
    }
    match v {
        Value::Array(items) => {
            for item in items {
                walk(item, depth + 1, found);
            }
        }
        Value::Object(obj) => walk_schema(obj, depth, found),
        _ => {}
    }
}

fn walk_schema(obj: &Map<String, Value>, depth: usize, found: &mut Vec<&'static str>) {
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
        if DATA_KEYS.contains(&key) {
            continue;
        }
        if SCHEMA_MAPS.contains(&key) {
            if let Value::Object(map) = value {
                for sub in map.values() {
                    walk(sub, depth + 1, found);
                }
                continue;
            }
        }
        walk(value, depth + 1, found);
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
        parse_response_format(Some(&v))
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
        assert_eq!(
            rf,
            ResponseFormat::JsonSchema {
                name: Some("thing".into()),
                schema: schema.clone(),
                strict: true
            }
        );
        assert!(rf.is_json());
        assert_eq!(rf.type_name(), "json_schema");

        // Non-string name is dropped; strict is only true when literally `true`.
        let rf = parse(json!({
            "type": "json_schema",
            "json_schema": {"name": 7, "schema": schema.clone(), "strict": "yes"}
        }))
        .unwrap();
        assert_eq!(
            rf,
            ResponseFormat::JsonSchema {
                name: None,
                schema,
                strict: false
            }
        );
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
        let g = grammar_for(&ResponseFormat::JsonSchema {
            name: None,
            schema: schema.clone(),
            strict: false,
        })
        .unwrap();
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
}
