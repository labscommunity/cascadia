//! A prompt the API rendered through the model's chat template reaches the
//! model with exactly one BOS, on both ov-genai pipelines.
//!
//! A rendered template already starts with the model's BOS (Llama-3.x
//! `<|begin_of_text|>`, Gemma `<bos>`, Mistral `<s>`). Handing that text to
//! ov-genai's string entry points encodes it with special tokens added, which
//! prepends a second BOS: the output stays fluent and just gets quietly worse.
//! The reference is the pipeline templating the raw user message itself
//! (`apply_chat_template` on), which encodes the rendered text without adding
//! special tokens. Greedy decoding makes the text comparison exact.
//!
//! Hardware + a real IR whose template starts with BOS are required:
//!
//!   CASCADIA_BOS_PARITY_MODEL=<ir-dir> \
//!     cargo test -p cascadia-engine-openvino --features openvino \
//!     --test bos_parity -- --nocapture
//!
//! Designed for OpenVINO/Llama-3.2-1B-Instruct-int4-ov, whose template starts
//! with `<|begin_of_text|>`. Any IR whose tokenizer adds special tokens works
//! (TinyLlama-1.1B-Chat adds a BOS its template lacks); the test fails loudly
//! on a tokenizer that adds none, so it cannot pass vacuously. Optional: CASCADIA_BOS_PARITY_DEVICE (default CPU) and
//! CASCADIA_BOS_PARITY_VLM_MODEL=<vlm-ir-dir> to check the VLM path too.
//! Skips when CASCADIA_BOS_PARITY_MODEL is unset.
#![cfg(feature = "openvino")]

use cascadia_ov_genai_shim::{
    CbPipeline, CbSchedulerConfig, CbStatus, GenConfig, LlmPipeline, PluginConfig,
};

const USER: &str = "What is the capital of France?";

/// Greedy. `templated`: the prompt is the rendered template (skip
/// ov-genai's apply, encode without special tokens); otherwise the pipeline
/// templates the raw user message itself.
fn cfg(templated: bool) -> GenConfig {
    GenConfig {
        max_new_tokens: 24,
        do_sample: false,
        temperature: 0.0,
        skip_chat_template: templated,
        prompt_templated: templated,
        top_p: 1.0,
        ..GenConfig::default()
    }
}

fn cb_generate(pipe: &CbPipeline, id: u64, prompt: &str, cfg: &GenConfig) -> String {
    let mut handle = pipe.add_request(id, prompt, cfg).expect("cb add_request");
    let mut text = String::new();
    for _ in 0..1024 {
        pipe.step().expect("cb step");
        let read = handle.read().expect("cb read");
        text.push_str(&read.text_delta);
        if read.status != CbStatus::Running {
            break;
        }
    }
    text
}

/// The rendered prompt and its token counts without and with special tokens,
/// after checking the tokenizer does add special tokens on top of it (else a
/// doubled BOS cannot happen and the check proves nothing).
fn rendered_prompt(pipe: &LlmPipeline, model: &str) -> (String, u32, u32) {
    let rendered = pipe.apply_chat_template(USER).expect("render template");
    let with = pipe.count_tokens(&rendered, true).expect("count with");
    let without = pipe.count_tokens(&rendered, false).expect("count without");
    eprintln!("{model}: rendered {rendered:?}, {without} tokens (+special: {with})");
    assert!(
        with > without,
        "{model}'s tokenizer adds no special tokens to its rendered template; \
         pick a BOS model (e.g. Llama-3.2-1B-Instruct) or the check proves nothing"
    );
    (rendered, without, with)
}

#[test]
fn a_templated_prompt_carries_one_bos_on_both_pipelines() {
    let Ok(model) = std::env::var("CASCADIA_BOS_PARITY_MODEL") else {
        eprintln!("CASCADIA_BOS_PARITY_MODEL not set; skipping");
        return;
    };
    let device = std::env::var("CASCADIA_BOS_PARITY_DEVICE").unwrap_or_else(|_| "CPU".into());

    let llm = LlmPipeline::new(&model, &device, &PluginConfig::new()).expect("llm pipeline");
    let (rendered, templated_tokens, with_special) = rendered_prompt(&llm, &model);

    let reference = llm.generate(USER, &cfg(false)).expect("reference generate");
    let ours = llm
        .generate(&rendered, &cfg(true))
        .expect("templated generate");
    let mut raw_cfg = cfg(true);
    raw_cfg.prompt_templated = false;
    let raw = llm.generate(&rendered, &raw_cfg).expect("raw generate");
    eprintln!(
        "llm reference: {:?} ({:?} prompt tokens)",
        reference.text, reference.prompt_tokens
    );
    eprintln!(
        "llm templated: {:?} ({:?} prompt tokens)",
        ours.text, ours.prompt_tokens
    );
    eprintln!(
        "llm raw:       {:?} ({:?} prompt tokens)",
        raw.text, raw.prompt_tokens
    );

    assert_eq!(
        reference.prompt_tokens,
        Some(templated_tokens),
        "reference prompt tokens"
    );
    assert_eq!(
        ours.prompt_tokens,
        Some(templated_tokens),
        "a templated prompt reached LLMPipeline with a doubled BOS"
    );
    assert_eq!(
        ours.text, reference.text,
        "LLMPipeline templated vs reference"
    );
    // A prompt not flagged templated keeps ov-genai's own encoding, special
    // tokens added: the doubled BOS this check exists to catch.
    assert_eq!(
        raw.prompt_tokens,
        Some(with_special),
        "raw string path changed"
    );
    drop(llm);

    let cb = CbPipeline::new(
        &model,
        &device,
        &CbSchedulerConfig::default(),
        &PluginConfig::new(),
    )
    .expect("cb pipeline");
    assert_eq!(
        cb.count_tokens(&rendered, false),
        Some(templated_tokens),
        "CB tokenizer disagrees"
    );
    let cb_ours = cb_generate(&cb, 1, &rendered, &cfg(true));
    let cb_reference = cb_generate(&cb, 2, USER, &cfg(false));
    eprintln!("cb  templated: {cb_ours:?}");
    eprintln!("cb  reference: {cb_reference:?}");
    assert_eq!(
        cb_ours, reference.text,
        "CB templated vs LLMPipeline reference"
    );
    assert_eq!(
        cb_reference, reference.text,
        "CB own template vs LLMPipeline reference"
    );
}

#[test]
fn a_templated_prompt_carries_one_bos_on_the_vlm_pipeline() {
    let Ok(model) = std::env::var("CASCADIA_BOS_PARITY_VLM_MODEL") else {
        eprintln!("CASCADIA_BOS_PARITY_VLM_MODEL not set; skipping");
        return;
    };
    let device = std::env::var("CASCADIA_BOS_PARITY_DEVICE").unwrap_or_else(|_| "CPU".into());
    let vlm = LlmPipeline::vlm(&model, &device, false, &PluginConfig::new()).expect("vlm pipeline");
    let (rendered, templated_tokens, _) = rendered_prompt(&vlm, &model);
    let reference = vlm.generate(USER, &cfg(false)).expect("reference generate");
    let ours = vlm
        .generate(&rendered, &cfg(true))
        .expect("templated generate");
    eprintln!(
        "vlm reference: {:?} ({:?} prompt tokens)",
        reference.text, reference.prompt_tokens
    );
    eprintln!(
        "vlm templated: {:?} ({:?} prompt tokens)",
        ours.text, ours.prompt_tokens
    );
    assert_eq!(
        reference.prompt_tokens,
        Some(templated_tokens),
        "reference prompt tokens"
    );
    assert_eq!(
        ours.prompt_tokens,
        Some(templated_tokens),
        "VLM templated prompt tokens"
    );
    assert_eq!(ours.text, reference.text, "VLM templated vs reference");
}
