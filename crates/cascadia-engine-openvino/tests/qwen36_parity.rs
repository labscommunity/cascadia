//! Greedy-parity regression gates for the qwen35 staged engine (Qwen3.6
//! MoE and dense Qwen3.8 share the engine; each family has its own golden).
//!
//! Needs real OpenVINO + the exported shards, so the tests are `#[ignore]`d
//! and skip unless the shard-dir env var is set. Run on a node:
//!
//! ```text
//! QWEN36_SHARDS=C:\cascadia\models\qwen36-shards-2stage \
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!   --test qwen36_parity -- --ignored qwen36_greedy_parity
//! QWEN38_SHARDS=C:\cascadia\models\qwen38-shards-2stage \
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!   --test qwen36_parity -- --ignored qwen38_greedy_parity
//! ```
//!
//! Golden provenance: `tools/qwen36_surgery/golden/qwen36_parity_64.json`
//! and `qwen38_parity_64.json`, blessed from runs whose output equaled the
//! Python chain reference (engine ≡ chain, see
//! probe_chain_vs_full_prompt.py / probe_engine_parity.py). Regenerate with
//! `QWEN36_WRITE_GOLDEN=1` / `QWEN38_WRITE_GOLDEN=1` after intentional
//! numeric changes (e.g. an OpenVINO upgrade shifts f16 fusion order) and
//! re-verify against the chain probe before committing the new golden.

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

fn golden_path(file: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/qwen36_surgery/golden")
        .join(file)
}

/// Single-box greedy run of the staged chain at `shards`, compared against
/// (or, with `write_env` set, written to) the golden `file`. The Python
/// probes that bless a golden do not inject the empty `<think>` block, so
/// thinking stays ON here to keep engine ≡ chain comparable.
async fn run_parity(shards: &str, write_env: &str, file: &str) {
    let mut builder = Qwen36Builder::new(shards, "CPU");
    builder
        .connect(PeerLayout::single_stage())
        .await
        .expect("connect");
    let mut load = builder
        .load(ShardSpec::single_stage(shards, "CPU"))
        .await
        .expect("load");
    use futures::StreamExt;
    while load.next().await.is_some() {}
    let mut engine = Box::new(builder).build().expect("build");

    let mut task = GenerationTask::new("parity-test", PROMPT).with_max_tokens(MAX_TOKENS);
    task.enable_thinking = true;
    engine.submit(task).expect("submit");

    let mut ids = Vec::new();
    let mut text = String::new();
    loop {
        let chunks = engine.step().expect("step");
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

    let path = golden_path(file);
    if std::env::var(write_env).is_ok() {
        let g = Golden {
            prompt: PROMPT.to_string(),
            max_tokens: MAX_TOKENS,
            ids,
            text,
        };
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(&g).unwrap()).unwrap();
        eprintln!(
            "golden written: {} ({} tokens)",
            path.display(),
            g.ids.len()
        );
        return;
    }

    let g: Golden = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("golden missing ({e}); run once with {write_env}=1")),
    )
    .expect("golden parse");
    assert_eq!(
        g.prompt, PROMPT,
        "golden was blessed for a different prompt"
    );
    assert_eq!(g.max_tokens, MAX_TOKENS);
    let match_len = ids.iter().zip(&g.ids).take_while(|(a, b)| a == b).count();
    assert_eq!(
        ids,
        g.ids,
        "greedy divergence at token {match_len}/{} (engine text: {text:?})",
        g.ids.len()
    );
    assert_eq!(text, g.text);
}

#[tokio::test]
#[ignore = "needs real OpenVINO + exported shards (QWEN36_SHARDS)"]
async fn qwen36_greedy_parity() {
    let Ok(shards) = std::env::var("QWEN36_SHARDS") else {
        eprintln!("QWEN36_SHARDS not set; skipping");
        return;
    };
    run_parity(&shards, "QWEN36_WRITE_GOLDEN", "qwen36_parity_64.json").await;
}

#[tokio::test]
#[ignore = "needs real OpenVINO + exported Qwen3.8 shards (QWEN38_SHARDS)"]
async fn qwen38_greedy_parity() {
    let Ok(shards) = std::env::var("QWEN38_SHARDS") else {
        eprintln!("QWEN38_SHARDS not set; skipping");
        return;
    };
    run_parity(&shards, "QWEN38_WRITE_GOLDEN", "qwen38_parity_64.json").await;
}
