//! Chat-template parity between the continuous-batching and monolithic
//! ov-genai paths.
//!
//! `ContinuousBatchingPipeline`'s string `add_request()` overload ignores
//! `GenerationConfig::apply_chat_template` (OV GenAI 2026.2), so the shim
//! renders the template itself and enrolls the encoded ids. Two things can go
//! wrong there and neither shows up as an error:
//!
//! * not templating at all — what upstream does, and what this test caught.
//!   The engine relies on that flag whenever the API could not pre-render the
//!   template, and for thinking-on requests; those got raw prompts.
//! * templating and then encoding with `add_special_tokens=true`, which
//!   prepends a second BOS because the rendered template already carries one.
//!   The output stays fluent, just quietly worse.
//!
//! Both are invisible in isolation, so this compares against `LLMPipeline`,
//! where the flag is known to work. Greedy decoding makes the comparison exact:
//! on a matching prompt the two pipelines agree byte for byte (verified on
//! Phi-3.5-mini/CPU), so any prompt-level difference shows up as a text diff.
//!
//! Hardware + a real IR are required:
//!
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!     --test cb_chat_template_parity -- --nocapture
//!
//! with CASCADIA_CB_TEMPLATE_MODEL=<ir-dir>. Skips when unset.
#![cfg(feature = "openvino")]

use cascadia_ov_genai_shim::{
    CbPipeline, CbSchedulerConfig, CbStatus, GenConfig, LlmPipeline, PluginConfig,
};

/// Greedy, so both pipelines are deterministic and a diff means the prompt
/// differed — not that sampling wandered.
fn cfg(skip_chat_template: bool) -> GenConfig {
    GenConfig {
        max_new_tokens: 32,
        do_sample: false,
        temperature: 0.0,
        num_assistant_tokens: 0,
        max_ngram_size: 0,
        skip_chat_template,
        top_p: 1.0,
        top_k: 0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: None,
    }
}

fn cb_generate(pipe: &CbPipeline, id: u64, prompt: &str, skip: bool) -> String {
    let mut handle = pipe
        .add_request(id, prompt, &cfg(skip))
        .expect("cb add_request");
    let mut text = String::new();
    // Bounded: 32 max_new_tokens plus prefill iterations, with slack.
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

#[test]
fn cb_matches_llm_pipeline_with_and_without_the_chat_template() {
    let Ok(model) = std::env::var("CASCADIA_CB_TEMPLATE_MODEL") else {
        eprintln!("CASCADIA_CB_TEMPLATE_MODEL not set; skipping");
        return;
    };
    let device = std::env::var("CASCADIA_CB_TEMPLATE_DEVICE").unwrap_or_else(|_| "CPU".into());
    let prompt = "What is the capital of France?";

    let cb = CbPipeline::new(
        &model,
        &device,
        &CbSchedulerConfig::default(),
        &PluginConfig::new(),
    )
    .expect("cb pipeline");
    let cb_templated = cb_generate(&cb, 1, prompt, /*skip=*/ false);
    let cb_raw = cb_generate(&cb, 2, prompt, /*skip=*/ true);
    drop(cb);

    let llm = LlmPipeline::new(&model, &device, &PluginConfig::new()).expect("llm pipeline");
    let llm_templated = llm
        .generate(prompt, &cfg(false))
        .expect("llm generate")
        .text;
    let llm_raw = llm.generate(prompt, &cfg(true)).expect("llm generate").text;

    eprintln!("cb  templated: {cb_templated:?}");
    eprintln!("llm templated: {llm_templated:?}");
    eprintln!("cb  raw:       {cb_raw:?}");
    eprintln!("llm raw:       {llm_raw:?}");

    // The bug itself: asking for the template has to change something.
    assert_ne!(
        cb_templated, cb_raw,
        "apply_chat_template had no effect on the CB path — the template is \
         not being applied (upstream's string add_request ignores the flag)"
    );
    // Catches templating that produces the wrong ids — a doubled BOS from
    // encoding the rendered template with add_special_tokens=true is the
    // likely way to get here, and it is otherwise invisible.
    assert_eq!(
        cb_templated, llm_templated,
        "CB and LLMPipeline disagree WITH the chat template; the CB prompt is \
         not tokenizing the same way"
    );
    // The branch the shim deliberately leaves on upstream's string overload.
    assert_eq!(
        cb_raw, llm_raw,
        "CB and LLMPipeline disagree WITHOUT the chat template"
    );
}
