//! Greedy-parity regression gate for the qwen36-moe staged engine.
//!
//! Needs real OpenVINO + the exported shards, so it is `#[ignore]`d and
//! skips unless `QWEN36_SHARDS` points at a shard dir. Run on a node:
//!
//! ```text
//! QWEN36_SHARDS=C:\cascadia\models\qwen36-shards-2stage \
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!   --test qwen36_parity -- --ignored
//! ```
//!
//! Golden provenance: `tools/qwen36_surgery/golden/qwen36_parity_64.json`,
//! blessed from a run whose output equaled the Python chain reference
//! (engine ≡ chain, see probe_chain_vs_full_prompt.py). Regenerate with
//! `QWEN36_WRITE_GOLDEN=1` after intentional numeric changes (e.g. an
//! OpenVINO upgrade shifts f16 fusion order) and re-verify against the
//! chain probe before committing the new golden.

use cascadia_engine::Builder;
use cascadia_engine_openvino::Qwen36Builder;
use cascadia_types::{GenerationTask, PeerLayout, ShardSpec};

/// Rendered exactly as the API's legacy formatter renders
/// `[{role: "user", content: "Explain how rainbows form."}]`.
const PROMPT: &str = "user: Explain how rainbows form.";
const MAX_TOKENS: u32 = 64;

#[derive(serde::Serialize, serde::Deserialize)]
struct Golden {
    prompt: String,
    max_tokens: u32,
    ids: Vec<i64>,
    text: String,
}

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/qwen36_surgery/golden/qwen36_parity_64.json")
}

#[tokio::test]
#[ignore = "needs real OpenVINO + exported shards (QWEN36_SHARDS)"]
async fn qwen36_greedy_parity() {
    let Ok(shards) = std::env::var("QWEN36_SHARDS") else {
        eprintln!("QWEN36_SHARDS not set; skipping");
        return;
    };

    let mut builder = Qwen36Builder::new(&shards, "CPU");
    builder
        .connect(PeerLayout::single_stage())
        .await
        .expect("connect");
    let mut load = builder.load(ShardSpec::single_stage(&shards, "CPU")).await.expect("load");
    use futures::StreamExt;
    while load.next().await.is_some() {}
    let mut engine = Box::new(builder).build().expect("build");

    let task = GenerationTask::new("parity-test", PROMPT).with_max_tokens(MAX_TOKENS);
    engine.submit(task).expect("submit");

    let mut ids = Vec::new();
    let mut text = String::new();
    loop {
        let chunks = engine.step();
        let mut done = false;
        for (_, c) in chunks {
            if c.is_final {
                text.push_str(&c.text);
                done = true;
            } else {
                ids.push(c.token_id);
                text.push_str(&c.text);
            }
        }
        if done {
            break;
        }
    }
    assert!(!ids.is_empty(), "engine generated no tokens");

    let path = golden_path();
    if std::env::var("QWEN36_WRITE_GOLDEN").is_ok() {
        let g = Golden {
            prompt: PROMPT.to_string(),
            max_tokens: MAX_TOKENS,
            ids,
            text,
        };
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(&g).unwrap()).unwrap();
        eprintln!("golden written: {} ({} tokens)", path.display(), g.ids.len());
        return;
    }

    let g: Golden = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("golden missing ({e}); run once with QWEN36_WRITE_GOLDEN=1")),
    )
    .expect("golden parse");
    assert_eq!(g.prompt, PROMPT, "golden was blessed for a different prompt");
    assert_eq!(g.max_tokens, MAX_TOKENS);
    let match_len = ids.iter().zip(&g.ids).take_while(|(a, b)| a == b).count();
    assert_eq!(
        ids, g.ids,
        "greedy divergence at token {match_len}/{} (engine text: {text:?})",
        g.ids.len()
    );
    assert_eq!(text, g.text);
}
