//! Test helper: validate a response body against the pinned OpenAI spec.
//!
//! The fixture is the `$ref` closure of the OpenAI response schemas, cut from
//! upstream `openai/openai-openapi` by `scripts/openai_spec_subset.py` (its
//! `x-source` field records the commit and the command). Regenerate it with
//! that script rather than editing it by hand.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

const SUBSET: &str = include_str!("../tests/fixtures/openai-spec-subset.json");

fn validator(schema_name: &str) -> Arc<jsonschema::Validator> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<jsonschema::Validator>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache
        .entry(schema_name.to_owned())
        .or_insert_with(|| {
            let subset: Value = serde_json::from_str(SUBSET).expect("spec subset is valid JSON");
            assert!(
                subset["components"]["schemas"].get(schema_name).is_some(),
                "{schema_name} is not in the spec subset; add it to ROOTS in \
                 scripts/openai_spec_subset.py and regenerate the fixture"
            );
            let root = serde_json::json!({
                "$ref": format!("#/components/schemas/{schema_name}"),
                "components": subset["components"],
            });
            Arc::new(
                jsonschema::validator_for(&root)
                    .unwrap_or_else(|e| panic!("{schema_name} does not compile: {e}")),
            )
        })
        .clone()
}

/// Panics, listing every violation with its instance path, unless `value`
/// conforms to the OpenAI schema `components.schemas.<schema_name>`.
pub(crate) fn assert_spec(schema_name: &str, value: &Value) {
    let errors: Vec<String> = validator(schema_name)
        .iter_errors(value)
        .map(|e| format!("  at {:?}: {e}", e.instance_path().to_string()))
        .collect();
    assert!(
        errors.is_empty(),
        "response does not match OpenAI {schema_name}:\n{}\nbody: {value}",
        errors.join("\n")
    );
}

/// Map each stream event `type` to its own schema name: the `$ref` members of
/// `ResponseStreamEvent`'s `anyOf`, keyed by their `properties.type.enum[0]`.
fn event_schemas() -> &'static HashMap<String, String> {
    static MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        let subset: Value = serde_json::from_str(SUBSET).expect("spec subset is valid JSON");
        let schemas = &subset["components"]["schemas"];
        let members = schemas["ResponseStreamEvent"]["anyOf"]
            .as_array()
            .expect("ResponseStreamEvent has an anyOf");
        members
            .iter()
            .map(|member| {
                let name = member["$ref"]
                    .as_str()
                    .and_then(|r| r.strip_prefix("#/components/schemas/"))
                    .unwrap_or_else(|| {
                        panic!("ResponseStreamEvent member is not a $ref: {member}")
                    });
                let ty = schemas[name]["properties"]["type"]["enum"][0]
                    .as_str()
                    .unwrap_or_else(|| panic!("{name} has no properties.type.enum[0]"));
                (ty.to_owned(), name.to_owned())
            })
            .collect()
    })
}

/// Validate one stream event against its own schema, chosen by `v["type"]`:
/// the `ResponseStreamEvent` anyOf member whose `properties.type.enum[0]`
/// equals it. Panics naming the type when no member matches.
pub(crate) fn assert_event(v: &Value) {
    let ty = v["type"]
        .as_str()
        .unwrap_or_else(|| panic!("stream event has no string `type`: {v}"));
    let name = event_schemas().get(ty).unwrap_or_else(|| {
        panic!("unknown stream event type {ty:?}: not a ResponseStreamEvent member")
    });
    assert_spec(name, v);
}

#[test]
fn assert_spec_rejects_a_response_without_choices() {
    // The helper must not be a no-op: a chat response missing the required
    // `choices` array has to fail.
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
    });
    let panic = std::panic::catch_unwind(|| assert_spec("CreateChatCompletionResponse", &body))
        .expect_err("a response without `choices` must fail the schema");
    let msg = panic.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("\"choices\" is a required property"), "{msg}");
}

fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let panic = std::panic::catch_unwind(f).expect_err("expected a panic");
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

#[test]
fn assert_event_checks_the_event_against_its_own_schema() {
    let msg =
        panic_message(|| assert_event(&serde_json::json!({"type": "response.output_text.delta"})));
    assert!(msg.contains("ResponseTextDeltaEvent"), "{msg}");
    assert!(msg.contains("\"delta\" is a required property"), "{msg}");
}

#[test]
fn assert_event_rejects_an_unknown_event_type() {
    let msg = panic_message(|| assert_event(&serde_json::json!({"type": "response.bogus"})));
    assert!(msg.contains("response.bogus"), "{msg}");
}
