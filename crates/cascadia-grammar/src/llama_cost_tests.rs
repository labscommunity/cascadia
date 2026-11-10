use super::*;
use serde_json::json;

/// An object with `n` properties `p0…`, the first `required` of them required.
fn object(n: usize, required: usize) -> Value {
    let props: Map<String, Value> = (0..n).map(|i| (format!("p{i}"), json!({}))).collect();
    let req: Vec<String> = (0..required).map(|i| format!("p{i}")).collect();
    json!({"type": "object", "properties": props, "required": req})
}

#[test]
fn an_object_with_too_many_optional_properties_is_refused() {
    let wide = json!({"type": "object", "properties": {"a": object(600, 0)}});
    let e = llama_grammar_cost_error(&wide).expect("600 optional properties");
    assert!(e.contains("600 at #/properties/a"), "{e}");
    assert!(e.contains("llama-server would take too long"), "{e}");
    // The 6000-property schema that took 176 s in the converter.
    assert!(llama_grammar_cost_error(&object(6000, 0)).is_some());
    assert_eq!(llama_grammar_cost_error(&object(400, 0)), None);
}

#[test]
fn required_properties_cost_only_their_visit() {
    assert_eq!(llama_grammar_cost_error(&object(6000, 6000)), None);
    // Two of three properties made optional: 2³, plus their rule names
    // (`p1`, `p2`, each with its 5-byte `-rest` suffix) once per optional one.
    let diff = llama_grammar_cost(&object(3, 1)) - llama_grammar_cost(&object(3, 3));
    assert_eq!(diff, 8 + 2 * (2 * (2 + 5)) * SUFFIX_BYTE_COST);
}

#[test]
fn additional_properties_count_as_one_more_optional_property() {
    let n: u64 = 300;
    let with = |a: Value| {
        let mut o = object(n as usize, 0);
        o["additionalProperties"] = a;
        llama_grammar_cost(&o)
    };
    let (open, typed, closed) = (
        with(json!(true)),
        with(json!({"type": "string"})),
        with(json!(false)),
    );
    assert!(
        open - closed >= (n + 1).pow(3) - n.pow(3),
        "{open} {closed}"
    );
    assert!(typed > open);
}

#[test]
fn several_objects_add_up() {
    let props = |n: usize| -> Value {
        let m: Map<String, Value> = (0..n).map(|i| (format!("o{i}"), object(200, 0))).collect();
        json!({"type": "object", "properties": m, "required": (0..n).map(|i| format!("o{i}")).collect::<Vec<_>>()})
    };
    let one = llama_grammar_cost(&props(1));
    let fits = (MAX_LLAMA_GRAMMAR_COST / one) as usize;
    assert!(fits >= 2, "{one}");
    assert_eq!(llama_grammar_cost_error(&props(fits - 1)), None);
    assert!(llama_grammar_cost_error(&props(fits + 1)).is_some());
}

#[test]
fn an_all_of_merge_counts_optional_members_and_visits_ref_targets_again() {
    // Members under `anyOf` add optional properties; plain members required ones.
    let optional = json!({"allOf": [{"anyOf": [object(300, 0), object(300, 0)]}]});
    let e = llama_grammar_cost_error(&optional).expect("600 merged optional properties");
    assert!(e.contains("600 at #"), "{e}");
    let required = json!({"allOf": [object(300, 0), object(300, 0)]});
    assert_eq!(llama_grammar_cost_error(&required), None);

    // A `$ref` target is visited once however often it is referenced ...
    let refs: Map<String, Value> = (0..10)
        .map(|i| (format!("r{i}"), json!({"$ref": "#/$defs/D"})))
        .collect();
    let by_ref = json!({
        "$defs": {"D": {"type": "object", "properties": {"x": object(300, 0)}, "required": ["x"]}},
        "type": "object", "properties": refs
    });
    assert_eq!(llama_grammar_cost_error(&by_ref), None);
    // ... but each `allOf` that merges it visits its properties again.
    let merges: Map<String, Value> = (0..10)
        .map(|i| (format!("r{i}"), json!({"allOf": [{"$ref": "#/$defs/D"}]})))
        .collect();
    let by_merge = json!({
        "$defs": {"D": {"type": "object", "properties": {"x": object(300, 0)}, "required": ["x"]}},
        "type": "object", "properties": merges
    });
    let e = llama_grammar_cost_error(&by_merge).expect("ten merges of a 300-property object");
    assert!(e.contains("300 at #/$defs/D/properties/x"), "{e}");
}

#[test]
fn unions_and_type_lists_visit_each_branch() {
    let alts = json!({"anyOf": [object(300, 0), object(300, 0), object(300, 0), object(300, 0), object(300, 0)]});
    assert!(llama_grammar_cost_error(&alts).is_some());
    let mut listed = object(300, 0);
    listed["type"] = json!(["object", "object"]);
    assert!(llama_grammar_cost_error(&listed).is_some());
    let mut once = object(300, 0);
    once["type"] = json!(["object", "null"]);
    assert_eq!(llama_grammar_cost_error(&once), None);
    let items = json!({"type": "array", "prefixItems": [object(300, 0), object(300, 0), object(300, 0), object(300, 0), object(300, 0)]});
    assert!(llama_grammar_cost_error(&items).is_some());
}

#[test]
fn data_and_unknown_refs_cost_nothing_and_ref_cycles_end() {
    // `enum`/`const` values are only printed: linear in their size.
    let data = json!({"enum": [object(6000, 0)]});
    assert_eq!(llama_grammar_cost_error(&data), None);
    let remote = json!({"$ref": "https://example.com/schema.json"});
    assert!(llama_grammar_cost(&remote) < 2 * LLAMA_VISIT_COST);
    let cycle = json!({
        "$defs": {"N": {"type": "object", "properties": {"next": {"$ref": "#/$defs/N"}}}},
        "$ref": "#/$defs/N"
    });
    assert!(llama_grammar_cost(&cycle) < 10 * LLAMA_VISIT_COST);
    let self_member =
        json!({"$defs": {"S": {"allOf": [{"$ref": "#/$defs/S"}]}}, "$ref": "#/$defs/S"});
    assert!(llama_grammar_cost(&self_member) < 10 * LLAMA_VISIT_COST);
}

#[test]
fn a_doubling_merge_chain_is_refused_in_linear_time() {
    // D{i} has two properties that each merge D{i-1}: the converter visits
    // D0's properties 2^40 times; the walk visits each schema once.
    let mut defs = Map::new();
    defs.insert(
        "D0".into(),
        json!({"type": "object", "properties": {"leaf": {"type": "string"}}}),
    );
    for i in 1..=40 {
        let prev = format!("#/$defs/D{}", i - 1);
        defs.insert(
            format!("D{i}"),
            json!({"type": "object", "properties": {
                "a": {"allOf": [{"$ref": prev}]},
                "b": {"allOf": [{"$ref": prev}]}
            }, "required": ["a", "b"]}),
        );
    }
    // The entry `$ref` sits under a property: the converter registers no
    // `$ref` inside an object that has one, so `$defs` beside a root `$ref`
    // would be unresolved (null) for it.
    let schema =
        json!({"$defs": defs, "type": "object", "properties": {"r": {"$ref": "#/$defs/D40"}}});
    let t0 = std::time::Instant::now();
    let e = llama_grammar_cost_error(&schema).expect("2^40 visits");
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(1),
        "{:?}",
        t0.elapsed()
    );
    assert!(e.contains("its cost is"), "{e}");
}

#[test]
fn real_client_schemas_pass() {
    for f in ["browser_use_schema.json", "pydantic_output_schema.json"] {
        let path = format!(
            "{}/../cascadia-engine-mock/tests/fixtures/{f}",
            env!("CARGO_MANIFEST_DIR")
        );
        let schema: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let cost = llama_grammar_cost(&schema);
        assert!(cost < MAX_LLAMA_GRAMMAR_COST / 10, "{f}: {cost}");
        eprintln!("{f}: {cost}");
    }
}

#[test]
fn measured_wide_objects_fall_on_the_right_side() {
    // Converter times at the pin: 400 optional 0.081 s, 1000 take 0.68 s.
    assert_eq!(llama_grammar_cost_error(&object(400, 0)), None);
    assert!(llama_grammar_cost_error(&object(1000, 0)).is_some());
}

#[test]
fn long_rule_names_cost_per_byte() {
    // Measured: a 58 KB property name over 500 optional properties 418 s;
    // a 16 KB one over 3,000 required properties 5.1 s.
    let long = "a".repeat(58_000);
    let opt = json!({"type": "object", "properties": {long: object(500, 0)}});
    assert!(llama_grammar_cost_error(&opt).is_some());
    let long = "a".repeat(16_000);
    let req = json!({"type": "object", "properties": {long: object(3000, 3000)}});
    assert!(llama_grammar_cost_error(&req).is_some());
    // Short names under the same objects pass.
    let short = json!({"type": "object", "properties": {"a": object(3000, 3000)}});
    assert_eq!(llama_grammar_cost_error(&short), None);
}

#[test]
fn printed_values_cost_per_byte() {
    // Measured: 1,600 merges of a property whose `const` is 30 KB of `"`, 5.6 s.
    let members: Vec<Value> = (0..1600).map(|_| json!({"$ref": "#/$defs/a"})).collect();
    let quotes = "\"".repeat(30_000);
    let schema =
        json!({"$defs": {"a": {"properties": {"x": {"const": quotes}}}}, "allOf": members});
    assert!(llama_grammar_cost_error(&schema).is_some());
}

#[test]
fn same_named_merged_properties_cost_per_pair() {
    // Measured: 2,500 members each defining `x`, 2.2 s.
    let members: Vec<Value> = (0..2500)
        .map(|i| json!({"properties": {"x": {"const": i}}}))
        .collect();
    assert!(llama_grammar_cost_error(&json!({"allOf": members})).is_some());
    let few: Vec<Value> = (0..20)
        .map(|i| json!({"properties": {"x": {"const": i}}}))
        .collect();
    assert_eq!(llama_grammar_cost_error(&json!({"allOf": few})), None);
}

#[test]
fn containers_of_another_json_type_are_iterated_as_the_converter_does() {
    // An array of properties: its indexes are the names, all optional.
    let listed = json!({"type": "object", "properties": vec![json!({}); 600]});
    let e = llama_grammar_cost_error(&listed).expect("600 properties named 0..599");
    assert!(e.contains("600 at #"), "{e}");
    // An `allOf` (and an `anyOf` in it) given as an object: its values are the members.
    let members = json!({"allOf": {"x": {"anyOf": {"y": object(600, 0)}}}});
    assert!(llama_grammar_cost_error(&members).is_some());
}

#[test]
fn an_all_of_member_follows_any_ref_chain_and_a_cycle_is_refused() {
    let mut defs = Map::new();
    for i in 0..100 {
        defs.insert(
            format!("d{i}"),
            json!({"$ref": format!("#/$defs/d{}", i + 1)}),
        );
    }
    defs.insert("d100".into(), object(600, 0));
    let chain = json!({"$defs": defs, "allOf": [{"anyOf": [{"$ref": "#/$defs/d0"}]}, {}]});
    assert!(llama_grammar_cost_error(&chain).is_some());

    let cycle =
        json!({"$defs": {"a": {"$ref": "#/$defs/a"}}, "allOf": [{"$ref": "#/$defs/a"}, {}]});
    let e = llama_grammar_cost_error(&cycle).expect("the converter loops on it");
    assert!(e.contains("form a cycle"), "{e}");
}

#[test]
fn refs_resolve_as_the_converter_resolves_them() {
    // Refs inside an object that has a `$ref` are never registered.
    let hidden = json!({"$defs": {"w": object(600, 0)}, "$ref": "#/$defs/v", "properties": {"x": {"$ref": "#/$defs/w"}}});
    assert_eq!(llama_grammar_cost_error(&hidden), None);
    // No `~1` decoding: the key is looked up as written.
    let tilde = json!({"$defs": {"x~1y": object(600, 0)}, "$ref": "#/$defs/x~1y"});
    assert!(llama_grammar_cost_error(&tilde).is_some());
    // `stoull` indexes: "0x" reads as 0.
    let index = json!({"$defs": [object(600, 0)], "$ref": "#/$defs/0x"});
    assert!(llama_grammar_cost_error(&index).is_some());
    // A bare `#` is not resolved (the converter reports it unsupported).
    let bare = json!({"type": "object", "properties": {"me": {"$ref": "#"}}});
    assert!(llama_grammar_cost(&bare) < 10 * LLAMA_VISIT_COST);
}

#[test]
fn each_distinct_ref_costs_a_copy_of_the_schema() {
    // 2000 refs that fail to resolve, in a schema of about 6000 values:
    // measured 1.39 s for 2500 of them.
    let props: Map<String, Value> = (0..2000)
        .map(|i| (format!("p{i}"), json!({"$ref": format!("#/$defs/x{i}")})))
        .collect();
    let req: Vec<String> = (0..2000).map(|i| format!("p{i}")).collect();
    let many = json!({"type": "object", "properties": props, "required": req});
    assert!(llama_grammar_cost_error(&many).is_some());
}

#[test]
fn deep_visits_are_refused() {
    let mut defs = Map::new();
    for i in 0..300 {
        defs.insert(
            format!("d{i}"),
            json!({"type": "object", "properties": {"n": {"$ref": format!("#/$defs/d{}", i + 1)}}, "required": ["n"]}),
        );
    }
    defs.insert("d300".into(), json!({"type": "string"}));
    let deep =
        json!({"$defs": defs, "type": "object", "properties": {"r": {"$ref": "#/$defs/d0"}}});
    let e = llama_grammar_cost_error(&deep).expect("300 levels of refs");
    assert!(e.contains("deeper than 256"), "{e}");
}

#[test]
fn many_merges_of_a_wide_target_stop_early() {
    // 1600 members each merging 3000 properties: 4.8M merged entries if
    // collected in full.
    let members: Vec<Value> = (0..1600).map(|_| json!({"$ref": "#/$defs/a"})).collect();
    let schema = json!({"$defs": {"a": object(3000, 3000)}, "allOf": members});
    let t0 = std::time::Instant::now();
    assert!(llama_grammar_cost_error(&schema).is_some());
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(1),
        "{:?}",
        t0.elapsed()
    );
}

#[test]
fn a_type_list_copies_the_whole_node_per_entry() {
    // Measured: 30,000 entries 5.8 s, 9,000 `"null"` entries 1.57 s.
    assert!(llama_grammar_cost_error(&json!({"type": vec![json!(1); 30000]})).is_some());
    assert!(llama_grammar_cost_error(&json!({"type": vec![json!("null"); 9000]})).is_some());
    let nullable = json!({"type": ["string", "null"], "maxLength": 20});
    assert!(llama_grammar_cost(&nullable) < 10 * LLAMA_VISIT_COST);
}

#[test]
fn merged_properties_are_copied_each_time() {
    // Measured: 1,600 merges of a property with a 1,000-value enum, 0.37 s.
    let members: Vec<Value> = (0..1600).map(|_| json!({"$ref": "#/$defs/a"})).collect();
    let enums: Vec<u32> = (0..1000).collect();
    let schema = json!({"$defs": {"a": {"properties": {"x": {"enum": enums}}}}, "allOf": members});
    assert!(llama_grammar_cost_error(&schema).is_some());
}

#[test]
fn ref_steps_in_merges_cost_little() {
    // Measured: 3,000 members through a 50-ref alias chain, 0.013 s.
    let mut defs: Map<String, Value> = (0..49)
        .map(|i| {
            (
                format!("d{i}"),
                json!({"$ref": format!("#/$defs/d{}", i + 1)}),
            )
        })
        .collect();
    defs.insert("d49".into(), json!({"type": "string"}));
    let members: Vec<Value> = (0..3000).map(|_| json!({"$ref": "#/$defs/d0"})).collect();
    let schema = json!({"$defs": defs, "allOf": members});
    assert_eq!(llama_grammar_cost_error(&schema), None);
}

#[test]
fn a_long_ref_token_is_formatted_only_when_reported() {
    // 1,300 short members reach a 50-property target through one alias
    // with a 30 KB `$ref` token (about 80 KB in all): building a pointer
    // with that token per property visited would be about 2 GB of text.
    let long = "k".repeat(30_000);
    let members: Vec<Value> = (0..1300).map(|_| json!({"$ref": "#/s"})).collect();
    let mut schema = json!({"allOf": members, "s": {"$ref": format!("#/{long}")}});
    schema[&long] = object(50, 50);
    let t0 = std::time::Instant::now();
    let _ = llama_grammar_cost(&schema);
    let took = t0.elapsed();
    assert!(took < std::time::Duration::from_millis(500), "{took:?}");
}

/// Prints the cost and verdict of each `*.json` schema in the directory
/// `LLAMA_COST_CORPUS`, to compare with llama.cpp's converter timed on the
/// same files. Run by hand: `LLAMA_COST_CORPUS=dir cargo test -p
/// cascadia-grammar --lib corpus -- --ignored --nocapture`.
#[test]
#[ignore]
fn corpus() {
    let dir = std::env::var("LLAMA_COST_CORPUS").expect("LLAMA_COST_CORPUS");
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
        let schema: Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        let t0 = std::time::Instant::now();
        let cost = llama_grammar_cost(&schema);
        let verdict = if cost > MAX_LLAMA_GRAMMAR_COST {
            "REFUSE"
        } else {
            "accept"
        };
        println!(
            "CORPUS {} cost={} {verdict} check={:?}",
            f.file_name().unwrap().to_string_lossy(),
            cost,
            t0.elapsed()
        );
    }
}
