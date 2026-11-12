use super::*;
use serde_json::json;

// Times in comments are xgrammar 0.1.31's `compile_json_schema` (one
// thread, cache off) with Qwen2.5-1.5B-Instruct's tokenizer (151,665 tokens)
// on an Apple M-series Mac.

/// An object with `n` properties `p0…` of any value, those whose index is in
/// `required` required.
fn object(n: usize, required: impl IntoIterator<Item = usize>) -> Value {
    let props: Map<String, Value> = (0..n).map(|i| (format!("p{i}"), json!({}))).collect();
    let req: Vec<String> = required.into_iter().map(|i| format!("p{i}")).collect();
    json!({"type": "object", "properties": props, "required": req})
}

fn refused(schema: &Value) -> bool {
    xgrammar_cost_error(schema).is_some()
}

#[test]
fn measured_wide_objects_fall_on_the_right_side() {
    // 300 optional properties 1.7 s, 1000 take 18 s.
    assert!(!refused(&object(300, [])));
    let e = xgrammar_cost_error(&object(1000, [])).expect("1000 optional properties");
    assert!(
        e.starts_with("xgrammar would take too long to compile a grammar for this schema"),
        "{e}"
    );
    assert!(
        e.contains("a run of 1000 properties not in `required`"),
        "{e}"
    );
    // The 6000-property schema that held a worker for over 15 minutes.
    assert!(refused(&object(6000, [])));
}

#[test]
fn a_required_property_ends_a_run_of_optional_ones() {
    // Only the first required: 600 take 5.7 s, 1000 take 16.9 s.
    assert!(!refused(&object(600, [0])));
    assert!(refused(&object(1000, [0])));
    // Every 100th required, 1000 in all: 1.34 s.
    assert!(!refused(&object(1000, (0..1000).step_by(100))));
    // 5000 required properties with one- and two-letter names: 0.44 s.
    let letters: Vec<char> = ('a'..='z').chain('A'..='Z').collect();
    let names: Vec<String> = letters
        .iter()
        .map(|c| c.to_string())
        .chain(
            letters
                .iter()
                .flat_map(|a| letters.iter().map(move |b| format!("{a}{b}"))),
        )
        .take(5000)
        .collect();
    let props: Map<String, Value> = names.iter().map(|n| (n.clone(), json!({}))).collect();
    let short = json!({"type": "object", "properties": props, "required": names});
    assert!(!refused(&short));
}

#[test]
fn min_and_max_properties_cost_the_cube_of_the_optional_run() {
    // minProperties over 100 optional properties 4.8 s, over 200 45 s;
    // maxProperties over 200 46 s; over 300 required 0.05 s.
    let with = |mut o: Value, k: &str, v: usize| {
        o[k] = json!(v);
        o
    };
    assert!(!refused(&with(object(100, []), "minProperties", 1)));
    let e = xgrammar_cost_error(&with(object(200, []), "minProperties", 1)).unwrap();
    assert!(
        e.contains("`minProperties`/`maxProperties` over 200 properties"),
        "{e}"
    );
    assert!(refused(&with(object(200, []), "maxProperties", 199)));
    assert!(!refused(&with(object(300, 0..300), "minProperties", 1)));
    assert!(!refused(&with(object(300, 0..300), "maxProperties", 300)));
}

/// `{"type": types, "properties": {"a": …}}` nested `depth` deep.
fn type_nest(depth: usize, types: Value) -> Value {
    let mut s = json!({"type": types.clone(), "properties": {}});
    for _ in 1..depth {
        s = json!({"type": types.clone(), "properties": {"a": s}});
    }
    s
}

#[test]
fn nested_type_lists_multiply() {
    // Here: `["object","object"]` nested 10 deep 0.20 s, 12 deep 1.9 s, 14
    // deep 26 s; `["array","array"]` 10 deep 0.85 s, 11 deep 2.7 s, 12
    // deep 10 s; `["object","null"]` 12 deep 0.002 s. OpenVINO GenAI took
    // 3.2 times as long as here on these (12.8 s on a 12-deep
    // `["object","object"]`), so a list 11 or 12 deep is refused.
    assert!(!refused(&type_nest(10, json!(["object", "object"]))));
    assert!(refused(&type_nest(12, json!(["object", "object"]))));
    assert!(!refused(&type_nest(12, json!(["object", "null"]))));
    let arrays = |depth: usize| {
        let mut a = json!({"type": ["array", "array"], "items": {"type": "integer"}});
        for _ in 1..depth {
            a = json!({"type": ["array", "array"], "items": a});
        }
        a
    };
    assert!(!refused(&arrays(10)));
    assert!(refused(&arrays(11)));
    assert!(refused(&arrays(12)));
}

#[test]
fn a_type_list_sixty_deep_is_refused_in_linear_time() {
    // 2^60 visits for the converter; one per schema for the walk.
    let e = xgrammar_cost_error(&type_nest(60, json!(["object", "object"]))).expect("2^60");
    // Counted, not timed: tens of visits for 2^60 converter visits
    // (measured 21).
    let visits = xgrammar_cost_visits(&type_nest(60, json!(["object", "object"])));
    assert!(visits < 200, "{visits}");
    assert!(e.contains("xgrammar would take too long"), "{e}");
}

/// `inner` under `depth` objects that each require their one property.
fn deep(depth: usize, inner: Value) -> Value {
    let mut s = inner;
    for _ in 0..depth {
        s = json!({"type": "object", "properties": {"a": s}, "required": ["a"]});
    }
    s
}

#[test]
fn deep_nesting_costs_the_cube_of_its_depth() {
    // Objects 100 deep 0.13 s, 200 deep 0.96 s (7.2 times as long).
    let (d100, d200) = (
        xgrammar_cost(&deep(100, json!({"type": "integer"}))),
        xgrammar_cost(&deep(200, json!({"type": "integer"}))),
    );
    assert!(d200 < MAX_XGRAMMAR_COST && d200 > 6 * d100, "{d100} {d200}");
    // 300 deep (3.5 s) is past the walk's depth bound: refused unpriced.
    // (JSON that deep never reaches the engine: the API's parser stops at
    // 128 levels.)
    let e = xgrammar_cost_error(&deep(300, json!({"type": "integer"}))).unwrap();
    assert!(e.contains("can't be estimated"), "{e}");
}

#[test]
fn the_walk_bounds_its_depth() {
    // A chain of 5000 `$ref`s, each under an `anyOf`: the walk stops at its
    // depth limit (256) and refuses the schema as one it can't price rather
    // than follow it. (The converter names a `$ref` target's rule after its
    // pointer, so such a chain is cheap to compile, 1000 of them 0.04 s; no
    // real schema nests 256 deep.)
    let mut defs = Map::new();
    for i in 0..5000 {
        defs.insert(
            format!("d{i}"),
            json!({"anyOf": [{"$ref": format!("#/$defs/d{}", i + 1)}]}),
        );
    }
    defs.insert("d5000".into(), json!({"type": "string"}));
    let chain = json!({"$defs": defs, "$ref": "#/$defs/d0"});
    assert_eq!(xgrammar_cost(&chain), u64::MAX);
    // Counted, not timed: the walk stops at the depth limit (measured 257
    // visits for the 5001-link chain).
    let visits = xgrammar_cost_visits(&chain);
    assert!(visits <= 300, "{visits}");
    let e = xgrammar_cost_error(&chain).unwrap();
    assert!(
        e.contains("can't be estimated (schemas nested more than 256 deep)"),
        "{e}"
    );
}

#[test]
fn a_chain_of_aliases_is_followed_in_a_loop() {
    // 1000 `$ref`s that each name the next: 0.023 s.
    let mut defs = Map::new();
    for i in 0..1000 {
        defs.insert(
            format!("d{i}"),
            json!({"$ref": format!("#/$defs/d{}", i + 1)}),
        );
    }
    defs.insert("d1000".into(), json!({"type": "integer"}));
    let chain = json!({"$defs": defs, "$ref": "#/$defs/d0"});
    assert!(!refused(&chain));
}

#[test]
fn a_ref_target_is_compiled_once() {
    let target = object(120, []);
    let props: Map<String, Value> = (0..10)
        .map(|i| (format!("r{i}"), json!({"$ref": "#/$defs/T"})))
        .collect();
    let by_ref = json!({"$defs": {"T": target.clone()}, "type": "object", "properties": props});
    let once = json!({"$defs": {"T": target.clone()}, "type": "object", "properties": {"r": {"$ref": "#/$defs/T"}}});
    assert!(xgrammar_cost(&by_ref) < xgrammar_cost(&once) + xgrammar_cost(&once) / 10);
    // The same object written out ten times is compiled ten times.
    let copies: Map<String, Value> = (0..10).map(|i| (format!("r{i}"), target.clone())).collect();
    let written = json!({"type": "object", "properties": copies});
    assert!(xgrammar_cost(&written) > 5 * xgrammar_cost(&once));
}

#[test]
fn refs_resolve_as_xgrammar_resolves_them() {
    // `URIToRule` splits the pointer on `/` and looks each part up as
    // written: no `~0`/`~1` or percent decoding.
    let wide = object(1000, []);
    let tilde = json!({"$defs": {"a~1b": wide.clone()}, "$ref": "#/$defs/a~1b"});
    assert!(xgrammar_cost_error(&tilde)
        .unwrap()
        .contains("a run of 1000 properties"));
    let slash = json!({"$defs": {"a/b": wide.clone()}, "$ref": "#/$defs/a~1b"});
    let e = xgrammar_cost_error(&slash).unwrap();
    assert!(
        e.contains("would reject this schema: a `$ref` that does not resolve"),
        "{e}"
    );
    let percent = json!({"$defs": {"a%20b": wide.clone()}, "$ref": "#/$defs/a%20b"});
    assert!(xgrammar_cost_error(&percent)
        .unwrap()
        .contains("a run of 1000 properties"));
    let decoded = json!({"$defs": {"a b": wide}, "$ref": "#/$defs/a%20b"});
    assert!(xgrammar_cost_error(&decoded)
        .unwrap()
        .contains("does not resolve"));
}

#[test]
fn an_all_of_of_several_members_is_any_value() {
    // The converter fuses two or more `allOf` members into `{}`; one member
    // is compiled as itself.
    assert!(!refused(
        &json!({"allOf": [object(1000, []), object(1000, [])]})
    ));
    assert!(refused(&json!({"allOf": [object(1000, [])]})));
}

/// `{"type": "object", "properties": {"s": {"type": "string", "maxLength": n}}}`
fn max_length(n: u64) -> Value {
    json!({"type": "object", "properties": {"s": {"type": "string", "maxLength": n}}, "required": ["s"]})
}

#[test]
fn string_length_bounds_cost_per_unrolled_copy() {
    // maxLength 32 0.49 s, 64 1.08 s, 255, 1000 and 5000 3.45 s each (the
    // compiler unrolls up to 128 characters, then 128 more and a repeat).
    for n in [32, 64, 255, 1000, 5000, 100_000] {
        assert!(!refused(&max_length(n)), "{n}");
    }
    assert!(2 * xgrammar_cost(&max_length(64)) > 3 * xgrammar_cost(&max_length(32)));
    assert_eq!(
        xgrammar_cost(&max_length(255)),
        xgrammar_cost(&max_length(5000))
    );
    // 21 fields with the same bounds share one rule: 3.45 s.
    let same: Map<String, Value> = (0..21)
        .map(|i| {
            (
                format!("f{i}"),
                json!({"type": "string", "maxLength": 5000}),
            )
        })
        .collect();
    assert!(!refused(&json!({"type": "object", "properties": same})));
    // maxLength 100, 254 and 1000 on three fields: 9.62 s.
    let three = json!({"type": "object", "properties": {
        "name": {"type": "string", "maxLength": 100},
        "email": {"type": "string", "maxLength": 254},
        "bio": {"type": "string", "maxLength": 1000}}, "required": ["name", "email"]});
    assert!(!refused(&three), "{}", xgrammar_cost(&three));
    // 50 distinct maxLength 100..149: over 60 s.
    let distinct: Map<String, Value> = (0..50)
        .map(|i| {
            (
                format!("p{i}"),
                json!({"type": "string", "maxLength": 100 + i}),
            )
        })
        .collect();
    let e = xgrammar_cost_error(&json!({"type": "object", "properties": distinct})).unwrap();
    assert!(e.contains("maxLength"), "{e}");
}

#[test]
fn numeric_bounds_are_read_as_xgrammar_reads_them() {
    // picojson keeps `5000.0` and `5e3` as doubles, and the converter
    // reads string lengths and item and property counts as int64 only.
    for bad in [
        json!({"type": "string", "maxLength": 5000.0}),
        json!({"type": "string", "minLength": 1e3}),
        json!({"type": "array", "items": {}, "minItems": 2.0}),
        json!({"type": "array", "items": {}, "maxItems": 10.0}),
        json!({"type": "object", "properties": {"a": {}}, "maxProperties": 1.0}),
        json!({"type": "integer", "minimum": 0.5}),
    ] {
        let e = xgrammar_cost_error(&bad).expect("rejected");
        assert!(e.starts_with("xgrammar would reject this schema"), "{e}");
    }
    // Integer bounds may be whole doubles.
    assert!(!refused(
        &json!({"type": "integer", "minimum": 0.0, "maximum": 1000.0})
    ));
    // Bounds past what the converter keeps in an `int`, or the grammar
    // parser reads: refused, not ignored.
    let e = xgrammar_cost_error(
        &json!({"type": "object", "properties": {"a": {}}, "minProperties": 1u64 << 40}),
    )
    .unwrap();
    assert!(e.contains("can't be estimated"), "{e}");
    let e = xgrammar_cost_error(&json!({"type": "string", "maxLength": 1u64 << 60})).unwrap();
    assert!(e.contains("would reject"), "{e}");
}

#[test]
fn a_schema_the_walk_cannot_follow_is_refused() {
    // The converter rejects these; the walk stops and refuses them too
    // rather than pass them with the cost so far.
    for bad in [
        json!({"type": "strin"}),
        json!({"type": "object", "properties": {"a": {"$ref": "#/nowhere"}}}),
        json!({"type": "object", "properties": {"a": false}}),
        json!({"enum": "x"}),
    ] {
        assert_eq!(xgrammar_cost(&bad), u64::MAX, "{bad}");
        assert!(refused(&bad), "{bad}");
    }
}

#[test]
fn regex_repetitions_cost_per_unrolled_copy() {
    // `[a-z]{1,1000}` 0.35 s; `[^"]{1,100}` 1.9 s;
    // `(([a-z]{1,30}){1,30}){1,5}` 2.0 s.
    let pattern = |p: &str| json!({"type": "string", "pattern": p});
    let az = xgrammar_cost(&pattern("^[a-z]{1,1000}$"));
    let wide = xgrammar_cost(&pattern("^[^\"]{1,100}$"));
    assert!(az < wide && wide < MAX_XGRAMMAR_COST, "{az} {wide}");
    assert!(!refused(&pattern("^(([a-z]{1,30}){1,30}){1,5}$")));
    // A hundred copies of the 1.9 s repetition.
    assert!(refused(&pattern("^([^\"]{1,100}){100}$")));
}

#[test]
fn regex_bounds_are_parsed_as_xgrammar_parses_them() {
    let pattern = |p: &str| json!({"type": "string", "pattern": p});
    // Leading zeros are read through, however many.
    assert_eq!(
        regex_cost("^[^\"]{000000000000000000000000000100}$"),
        regex_cost("^[^\"]{100}$")
    );
    assert_eq!(
        regex_cost("^[^\"]{0,0000000000000000000000000100}$"),
        regex_cost("^[^\"]{0,100}$")
    );
    assert!(regex_cost("^[^\"]{0,100}$").unwrap() > 1_000_000);
    // What the converter or its grammar parser rejects is refused.
    for bad in [
        "^a{,5}$",
        "^a{5$",
        "^a{1,99999999999999999}$",
        "^a{5,2}$",
        "^a{x}$",
        "^{5}$",
        "^a**$",
        "^(a$",
    ] {
        let e = xgrammar_cost_error(&pattern(bad)).unwrap_or_default();
        assert!(
            e.starts_with("xgrammar would reject this schema"),
            "{bad}: {e}"
        );
    }
    // A `{` inside a class is a character.
    assert!(!refused(&pattern("^[{]{1,5}$")));
}

#[test]
fn property_names_are_compiled_like_any_schema() {
    // The converter sets `propertyNames`' type to "string", but a `$ref`,
    // `anyOf` or `allOf` in it is still followed: with 300 optional
    // properties behind each, 1.52 s, as long as the object alone.
    let opt300 = object(300, []);
    let alone = xgrammar_cost(&opt300);
    for names in [
        json!({"anyOf": [opt300.clone()]}),
        json!({"allOf": [opt300.clone()]}),
        json!({"$ref": "#/$defs/O"}),
    ] {
        let schema =
            json!({"$defs": {"O": opt300.clone()}, "type": "object", "propertyNames": names});
        let cost = xgrammar_cost(&schema);
        assert!(
            cost >= alone && cost < MAX_XGRAMMAR_COST,
            "{names}: {cost} {alone}"
        );
    }
    // 6000 behind it: refused.
    let wide = object(6000, []);
    for names in [
        json!({"anyOf": [wide.clone()]}),
        json!({"allOf": [wide.clone()]}),
        json!({"$ref": "#/$defs/O"}),
    ] {
        let schema =
            json!({"$defs": {"O": wide.clone()}, "type": "object", "propertyNames": names});
        let e = xgrammar_cost_error(&schema).expect("6000 optional properties");
        assert!(e.contains("a run of 6000 properties"), "{e}");
    }
    // Without them it is a string: `maxLength` 20, 0.28 s.
    let plain = json!({"type": "object", "propertyNames": {"maxLength": 20}});
    assert!(xgrammar_cost(&plain) > 200_000 && !refused(&plain));
}

#[test]
fn a_unicode_escape_with_braces_is_one_character() {
    // xgrammar reads `\u{1F600}` as one code point (0.001 s).
    let pattern = |p: &str| json!({"type": "string", "pattern": p});
    assert!(!refused(&pattern(r"^[a-z]+\u{1F600}$")));
    assert!(!refused(&pattern(r"^[\u{1F600}-\u{1F64F}]{1,3}$")));
    assert_eq!(regex_cost(r"\u{1F600}"), regex_cost("a"));
    for bad in [r"^\u{}$", r"^\u{1234567}$", r"^\u{12$"] {
        let e = xgrammar_cost_error(&pattern(bad)).unwrap_or_default();
        assert!(
            e.starts_with("xgrammar would reject this schema"),
            "{bad}: {e}"
        );
    }
}

fn fixture(f: &str) -> Value {
    let path = format!(
        "{}/../cascadia-engine-mock/tests/fixtures/{f}",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

#[test]
fn real_client_schemas_pass() {
    // 0.012 s and 0.011 s.
    for f in ["browser_use_schema.json", "pydantic_output_schema.json"] {
        let cost = xgrammar_cost(&fixture(f));
        assert!(cost < MAX_XGRAMMAR_COST / 10, "{f}: {cost}");
    }
}

/// A 68 KB extraction schema: 40 records of 25 fields (10 required:
/// nullable strings, bounded integers, enums, string lists of up to 10, an
/// address `$ref`, booleans) and a 40-field summary. 0.59 s.
fn realistic_big() -> Value {
    let mut records = Map::new();
    for r in 0..40 {
        let mut fields = Map::new();
        for f in 0..25 {
            let v = match f % 8 {
                0 | 5 => json!({"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}),
                1 | 3 => json!({"type": "string", "description": format!("Field {f} text")}),
                2 => json!({"enum": ["low", "medium", "high", "critical"]}),
                4 => json!({"type": "integer", "minimum": 0, "maximum": 1000}),
                6 => json!({"type": "array", "items": {"type": "string"}, "maxItems": 10}),
                _ => {
                    if f == 7 || f == 23 {
                        json!({"$ref": "#/$defs/Address"})
                    } else {
                        json!({"type": "boolean"})
                    }
                }
            };
            fields.insert(format!("field_{r}_{f}"), v);
        }
        let req: Vec<String> = (0..10).map(|f| format!("field_{r}_{f}")).collect();
        records.insert(
            format!("record_{r}"),
            json!({"type": "object", "properties": fields, "required": req, "additionalProperties": false}),
        );
    }
    let summary: Map<String, Value> = (0..40)
        .map(|i| (format!("summary_{i}"), json!({"type": "string"})))
        .collect();
    let all: Vec<String> = summary.keys().cloned().collect();
    records.insert(
        "summary".into(),
        json!({"type": "object", "properties": summary, "required": all}),
    );
    json!({
        "$defs": {"Address": {"type": "object", "properties": {
            "street": {"type": "string"}, "city": {"type": "string"},
            "zip": {"type": ["string", "null"]}}, "required": ["street", "city"]}},
        "type": "object", "properties": records
    })
}

#[test]
fn a_realistic_extraction_schema_passes() {
    let schema = realistic_big();
    assert!(serde_json::to_string(&schema).unwrap().len() > 50_000);
    assert!(!refused(&schema), "{}", xgrammar_cost(&schema));
}

#[test]
fn the_walk_stops_once_over_the_limit() {
    // 200,000 optional properties: the root's count alone passes the limit,
    // and the walk stops before visiting them.
    let wide = object(200_000, []);
    assert!(refused(&wide));
    // Counted, not timed: the root's optional properties price it past the
    // limit, so the walk visits none of the 200,000 property schemas.
    let visits = xgrammar_cost_visits(&wide);
    assert!(visits <= 1, "{visits}");
}

#[test]
fn long_patterns_and_type_lists_are_walked_in_linear_time() {
    // 60,000 unclosed `{`s, and 20,000 `type` entries.
    // Timed as a ratio, so a loaded machine slows both sizes alike: 10x the
    // input must cost well under the 100x a quadratic scan would (best of 3
    // runs each, to drop scheduling noise).
    fn best_of_3(n: usize) -> std::time::Duration {
        let braces = "a{".repeat(n / 2);
        let types: Vec<Value> = (0..n / 3).map(|i| json!(format!("t{i}"))).collect();
        (0..3)
            .map(|_| {
                let t0 = std::time::Instant::now();
                let _ = xgrammar_cost(&json!({"type": "string", "pattern": braces.clone()}));
                let _ = xgrammar_cost(&json!({"type": types.clone()}));
                t0.elapsed()
            })
            .min()
            .unwrap()
    }
    let small = best_of_3(6_000).max(std::time::Duration::from_micros(200));
    let large = best_of_3(60_000);
    assert!(large < small * 40, "{large:?} vs {small:?}");
}

/// Prints the cost and verdict of each `*.json` schema in the directory
/// `XGRAMMAR_COST_CORPUS`, to compare with xgrammar timed on the same files.
/// Run by hand: `XGRAMMAR_COST_CORPUS=dir cargo test -p cascadia-grammar
/// --lib xgrammar_cost_tests::corpus -- --ignored --nocapture`.
#[test]
#[ignore]
fn corpus() {
    let dir = std::env::var("XGRAMMAR_COST_CORPUS").expect("XGRAMMAR_COST_CORPUS");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .collect();
    files.sort();
    for f in files
        .into_iter()
        .filter(|f| f.extension().is_some_and(|e| e == "json"))
    {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        let schema: Value = match serde_json::from_str(&std::fs::read_to_string(&f).unwrap()) {
            Ok(s) => s,
            Err(e) => {
                println!("XGCORPUS {name} unparsed ({e})");
                continue;
            }
        };
        let t0 = std::time::Instant::now();
        // Uncapped, so a schema over the limit shows how far over.
        let mut walk = Walk::capped(&schema, u64::MAX);
        let cost = walk.run();
        let why = match &walk.stop {
            Some(Stop::Rejected(w)) => format!(" rejected: {w}"),
            Some(Stop::Unpriced(w)) => format!(" unpriced: {w}"),
            None => String::new(),
        };
        let verdict = if xgrammar_cost_error(&schema).is_some() {
            "REFUSE"
        } else {
            "accept"
        };
        println!(
            "XGCORPUS {name} cost={cost} {verdict} check={:?}{why}",
            t0.elapsed()
        );
    }
}

#[test]
fn the_added_integer_bounds_pass_and_a_u64_bound_is_refused() {
    // `bound_integers` adds the i64 range; xgrammar's converter reads an
    // integer bound as int64 and refuses one past it ("Integer exceeds
    // maximum limit"), which is why a non-negative integer is not capped at
    // u64::MAX.
    let mut v = json!({"type": "object", "properties": {"id": {"type": "integer"}, "n": {"type": "integer", "minimum": 0}}});
    crate::bound_integers(&mut v);
    assert_eq!(xgrammar_cost_error(&v), None);
    let e = xgrammar_cost_error(&json!({"type": "integer", "minimum": 0, "maximum": u64::MAX}))
        .expect("a bound past int64");
    assert!(e.contains("not a whole 64-bit number"), "{e}");
}
