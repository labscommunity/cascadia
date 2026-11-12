//! The sycl-llama engine's `$ref` inlining, pinned against llama.cpp's own
//! converter: `scripts/llama-ref-differential/run.sh` ran every schema of
//! the fixture through the inlining and then through the converter, and
//! checked that what it accepts reaches llama-server with no client ref and
//! converts to the grammar the schema means (and that a recursive schema is
//! refused without `--jinja`). This test needs no C++: it checks the
//! inlining still gives the verdicts that run recorded.

use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/llama_ref_differential.jsonl");

/// What the engine does with `schema`: `inlined` (with the number of
/// hoisted copies) or `refused`, as `run.sh`'s `llama_inline` example
/// reports it (inlining, then the llama-server cost guard).
fn verdict(schema: &Value, no_jinja: bool) -> (&'static str, usize) {
    if !cascadia_grammar::has_ref(schema) {
        return match cascadia_grammar::llama_grammar_cost_error(schema) {
            Some(_) => ("refused", 0),
            None => ("inlined", 0),
        };
    }
    match cascadia_grammar::inline_refs(schema, "", !no_jinja) {
        Ok(i) if cascadia_grammar::llama_grammar_cost_error(&i.schema).is_none() => {
            ("inlined", i.hoisted)
        }
        _ => ("refused", 0),
    }
}

#[test]
fn the_sycl_llama_inlining_gives_the_verdicts_checked_against_llama_cpp() {
    let mut entries = 0;
    for line in FIXTURE.lines().filter(|l| !l.trim().is_empty()) {
        let entry: Value = serde_json::from_str(line).unwrap();
        let schema = &entry["schema"];
        let (got, hoisted) = verdict(schema, false);
        assert_eq!(got, entry["verdict"], "{schema}");
        assert_eq!(
            hoisted as u64,
            entry["hoisted"].as_u64().unwrap(),
            "{schema}"
        );
        let (got_no_jinja, _) = verdict(schema, true);
        assert_eq!(got_no_jinja, entry["no_jinja"], "without --jinja: {schema}");
        entries += 1;
    }
    assert!(entries > 400, "{entries} entries");
}
