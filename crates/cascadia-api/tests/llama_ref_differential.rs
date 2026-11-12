//! The sycl-llama `$ref` check against llama.cpp's own converter, from the
//! fixture `scripts/llama-ref-differential/run.sh` writes (it runs the C++
//! side; this test needs none). Each entry carries llama.cpp's verdict as a
//! tag, and our check must agree:
//!
//! - `degraded` (llama.cpp turns a `$ref` into "any"): refused, never a
//!   silent 200;
//! - `faithful`: passed;
//! - `conservative:<class>` (faithful, but we refuse): refused, and the
//!   schema is in that documented class;
//! - `error` (llama.cpp refuses it, a 4xx): either.

use cascadia_api::{check_pointer_refs_only, RefResolver};
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/llama_ref_differential.jsonl");

/// A `$ref` under a `$ref` object's siblings.
fn ref_beside_ref(v: &Value, beside: bool) -> bool {
    match v {
        Value::Array(items) => items.iter().any(|x| ref_beside_ref(x, beside)),
        Value::Object(obj) => {
            if beside && obj.get("$ref").is_some_and(Value::is_string) {
                return true;
            }
            let b = beside || obj.contains_key("$ref");
            obj.iter()
                .filter(|(k, _)| *k != "$ref")
                .any(|(_, c)| ref_beside_ref(c, b))
        }
        _ => false,
    }
}

/// A `$ref` to an `https://` document with no (or an empty) fragment.
fn no_fragment_https_ref(v: &Value) -> bool {
    match v {
        Value::Array(items) => items.iter().any(no_fragment_https_ref),
        Value::Object(obj) => {
            let here = obj.get("$ref").and_then(Value::as_str).is_some_and(|r| {
                r.starts_with("https://") && r.split_once('#').is_none_or(|(_, f)| f.is_empty())
            });
            here || obj.values().any(no_fragment_https_ref)
        }
        _ => false,
    }
}

#[test]
fn the_sycl_llama_ref_check_matches_llama_cpp() {
    let mut entries = 0;
    for line in FIXTURE.lines().filter(|l| !l.trim().is_empty()) {
        let entry: Value = serde_json::from_str(line).unwrap();
        let schema = &entry["schema"];
        let tag = entry["tag"].as_str().unwrap();
        let verdict = check_pointer_refs_only("", schema, "sycl-llama", RefResolver::LlamaServer);
        let reason = verdict.as_ref().err().map(String::as_str).unwrap_or("");
        match tag {
            "degraded" => assert!(verdict.is_err(), "a silent 200: {schema}"),
            "faithful" => assert!(
                verdict.is_ok(),
                "refused a faithful schema: {schema}: {reason}"
            ),
            "error" => {}
            // Each conservative class is refused for its own reason.
            "conservative:ref-beside-ref" => assert!(
                ref_beside_ref(schema, false) && reason.contains("beside another `$ref`"),
                "{schema}: {reason}"
            ),
            "conservative:https-ref-unreached" => assert!(
                no_fragment_https_ref(schema)
                    && reason.contains("$ref \"https://")
                    && reason.contains("written as `#/...` pointers"),
                "{schema}: {reason}"
            ),
            other => panic!("unknown tag {other}: {schema}"),
        }
        entries += 1;
    }
    assert!(entries > 300, "{entries} entries");
}
