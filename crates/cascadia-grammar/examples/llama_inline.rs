//! Reads one JSON schema per stdin line and prints, per line, what the
//! sycl-llama engine sends llama-server for it: `{"inlined": <schema>,
//! "hoisted": <n>}`, or `{"refused": "<reason>"}` (a ref it can't inline,
//! or a schema llama-server's converter would take too long on). `--no-jinja` for a
//! llama-server without its jinja template, `--prefix P` for a tool's
//! hoisted names. Used by `scripts/llama-ref-differential/run.sh`.

use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let no_jinja = args.iter().any(|a| a == "--no-jinja");
    let prefix = args
        .iter()
        .position(|a| a == "--prefix")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let schema: serde_json::Value = serde_json::from_str(&line).expect("a JSON schema");
        // As the engine: a schema without a `$ref` (decided on the parsed
        // value) is sent as written.
        if !cascadia_grammar::has_ref(&schema) {
            match cascadia_grammar::llama_grammar_cost_error(&schema) {
                Some(e) => println!("{}", serde_json::json!({"refused": e})),
                None => println!(r#"{{"inlined":{line},"hoisted":0}}"#),
            }
            continue;
        }
        // The engine also refuses an inlined schema llama-server would take
        // too long (or recurse without end) to convert.
        let out = match cascadia_grammar::inline_refs(&schema, &prefix, !no_jinja) {
            Ok(i) => match cascadia_grammar::llama_grammar_cost_error(&i.schema) {
                Some(e) => serde_json::json!({"refused": e}),
                None => serde_json::json!({"inlined": i.schema, "hoisted": i.hoisted}),
            },
            Err(e) => serde_json::json!({"refused": e.to_string()}),
        };
        println!("{out}");
    }
}
