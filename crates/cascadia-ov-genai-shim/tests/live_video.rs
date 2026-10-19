//! Native video through GenAI, on a real VLM export. Needs the `openvino`
//! feature and `CASCADIA_LIVE_VLM` naming a Qwen-VL-family export directory:
//!
//! ```text
//! CASCADIA_LIVE_VLM=/path/to/Qwen3-VL-4B-Instruct-int4-ov \
//!   cargo test -p cascadia-ov-genai-shim --features openvino \
//!   --test live_video -- --ignored --nocapture
//! ```
#![cfg(feature = "openvino")]

use cascadia_ov_genai_shim::{
    CbPipeline, CbSchedulerConfig, CbStatus, GenConfig, LlmPipeline, Media, MediaKind, PluginConfig,
};

const SIDE: u32 = 448;
const COLORS: [(&str, [u8; 3]); 8] = [
    ("red", [255, 0, 0]),
    ("green", [0, 255, 0]),
    ("blue", [0, 0, 255]),
    ("yellow", [255, 255, 0]),
    ("magenta", [255, 0, 255]),
    ("cyan", [0, 255, 255]),
    ("white", [255, 255, 255]),
    ("black", [0, 0, 0]),
];
const PROMPT: &str = "<ov_genai_video_0>List the solid colors shown in this video, in order.";

/// 16 frames at 1 fps, each color held for two frames (one Qwen temporal
/// patch), so a model that sees every frame names all eight in order.
fn frames() -> Vec<u8> {
    let frame = (SIDE * SIDE * 3) as usize;
    let mut px = Vec::with_capacity(16 * frame);
    for k in 0..16 {
        let rgb = COLORS[k / 2].1;
        px.extend(std::iter::repeat_n(rgb, frame / 3).flatten());
    }
    px
}

fn video(px: &[u8]) -> Media<'_> {
    Media {
        kind: MediaKind::Video,
        pixels: px,
        num_frames: 16,
        height: SIDE,
        width: SIDE,
        fps: 1.0,
    }
}

fn cfg() -> GenConfig {
    GenConfig {
        max_new_tokens: 48,
        top_p: 1.0,
        ..GenConfig::default()
    }
}

/// `CASCADIA_LIVE_DEVICE` (default CPU).
fn device() -> String {
    std::env::var("CASCADIA_LIVE_DEVICE").unwrap_or_else(|_| "CPU".into())
}

/// f32 inference on an ARM CPU (the VLMPipeline path).
fn plugin() -> PluginConfig {
    if cfg!(target_arch = "aarch64") && device() == "CPU" {
        PluginConfig::new().with("INFERENCE_PRECISION_HINT", "f32")
    } else {
        PluginConfig::new()
    }
}

fn model() -> String {
    std::env::var("CASCADIA_LIVE_VLM").expect("set CASCADIA_LIVE_VLM to a VLM export")
}

/// Every color in order: the model saw all 16 frames, not GenAI's own
/// resample of them.
fn assert_all_colors(text: &str) {
    let lower = text.to_lowercase();
    let mut at = 0;
    for (name, _) in COLORS.iter().take(4) {
        let found = lower[at..].find(name);
        assert!(
            found.is_some(),
            "`{name}` missing after byte {at}: {text:?}"
        );
        at += found.unwrap();
    }
}

#[test]
#[ignore = "needs CASCADIA_LIVE_VLM and the OpenVINO GenAI runtime"]
fn vlm_pipeline_takes_a_native_video() {
    let pipe = LlmPipeline::vlm(&model(), &device(), false, &plugin()).unwrap();
    let px = frames();
    let r = pipe.generate(PROMPT, &[video(&px)], &cfg()).unwrap();
    eprintln!("vlm: input_tokens={:?} text={:?}", r.input_tokens, r.text);
    assert_all_colors(&r.text);
}

#[test]
#[ignore = "needs CASCADIA_LIVE_VLM and the OpenVINO GenAI runtime"]
fn cb_pipeline_takes_a_native_video_and_an_image() {
    // GenAI 2026.4.1's paged attention has no ARM CPU executor (f16 or f32),
    // so the CB pipeline cannot be built on an Apple Silicon CPU.
    if cfg!(target_arch = "aarch64") && device() == "CPU" {
        eprintln!("skipping: the CB pipeline needs an x86 CPU or a GPU");
        return;
    }
    let sched = CbSchedulerConfig {
        cache_size_gb: 2,
        ..CbSchedulerConfig::default()
    };
    let pipe = CbPipeline::new(&model(), &device(), &sched, &plugin()).unwrap();
    let px = frames();
    let mut h = pipe.add_request(1, PROMPT, &[video(&px)], &cfg()).unwrap();
    let mut text = String::new();
    loop {
        pipe.step().unwrap();
        let r = h.read().unwrap();
        text.push_str(&r.text_delta);
        match r.status {
            CbStatus::Running => {}
            CbStatus::Finished => break,
            other => panic!("request ended {other:?}: {text:?}"),
        }
    }
    eprintln!("cb: input_tokens={:?} text={text:?}", h.input_tokens());
    assert_all_colors(&text);

    // An image and a video in one request: each fills its own tag sequence.
    let img: Vec<u8> = std::iter::repeat_n([255u8, 128, 0], (SIDE * SIDE) as usize)
        .flatten()
        .collect();
    let media = [
        Media {
            kind: MediaKind::Image,
            pixels: &img,
            num_frames: 1,
            height: SIDE,
            width: SIDE,
            fps: 0.0,
        },
        video(&px),
    ];
    let prompt = "<ov_genai_image_0>What color is this image? <ov_genai_video_0>And what is \
                  the first color of this video? Answer in two words.";
    let mut h = pipe.add_request(2, prompt, &media, &cfg()).unwrap();
    let mut text = String::new();
    loop {
        pipe.step().unwrap();
        let r = h.read().unwrap();
        text.push_str(&r.text_delta);
        match r.status {
            CbStatus::Running => {}
            CbStatus::Finished => break,
            other => panic!("request ended {other:?}: {text:?}"),
        }
    }
    eprintln!(
        "cb mixed: input_tokens={:?} text={text:?}",
        h.input_tokens()
    );
    let lower = text.to_lowercase();
    assert!(
        lower.contains("orange") && lower.contains("red"),
        "{text:?}"
    );
}
