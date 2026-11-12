//! Reads one JSON schema per stdin line and prints, per line, `refused` or
//! `passed`: the API's `$ref` check for the sycl-llama engine
//! (llama-server). Used by `scripts/llama-ref-differential/run.sh` to compare
//! it with llama.cpp's own converter.

use std::io::BufRead;

use cascadia_api::{check_pointer_refs_only, RefResolver};

fn main() {
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let schema: serde_json::Value = serde_json::from_str(&line).expect("a JSON schema");
        let verdict =
            match check_pointer_refs_only(&line, &schema, "sycl-llama", RefResolver::LlamaServer) {
                Ok(()) => "passed",
                Err(_) => "refused",
            };
        println!("{verdict}");
    }
}
