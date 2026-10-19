//! Native video through GenAI, on a real VLM export. Needs the `openvino`
//! feature and `CASCADIA_LIVE_VLM` naming a Qwen-VL-family export directory:
//!
//! ```text
//! CASCADIA_LIVE_VLM=/path/to/Qwen3-VL-4B-Instruct-int4-ov \
//!   cargo test -p cascadia-ov-genai-shim --features openvino \
//!   --test live_video -- --ignored --nocapture --test-threads 1
//! ```
#![cfg(feature = "openvino")]

use cascadia_ov_genai_shim::{
    CbHandle, CbPipeline, CbSchedulerConfig, CbStatus, GenConfig, LlmPipeline, Media, MediaKind,
    PluginConfig,
};

const SIDE: u32 = 448;
/// Each color with the names a model may give it.
const COLORS: [(&[&str], [u8; 3]); 8] = [
    (&["red"], [255, 0, 0]),
    (&["green"], [0, 160, 0]),
    (&["blue"], [0, 0, 255]),
    (&["yellow"], [255, 220, 0]),
    (&["magenta", "pink", "purple", "violet"], [255, 0, 255]),
    (
        &["cyan", "teal", "turquoise", "aqua", "light blue"],
        [0, 200, 200],
    ),
    (&["orange"], [255, 128, 0]),
    (&["black"], [0, 0, 0]),
];
const PROMPT: &str =
    "<ov_genai_video_0>A colored square is shown on a white background. List its colors \
     in the order they appear in this video.";

/// 16 frames at 1 fps: a square of one color on white, each color held for
/// two frames (one Qwen temporal patch), so a model that sees every frame
/// names all eight in order. A shape on white, not a solid frame: some
/// families describe a full-frame color as one tint.
fn frames() -> Vec<u8> {
    let side = SIDE as usize;
    let mut px = Vec::with_capacity(16 * side * side * 3);
    for k in 0..16 {
        let rgb = COLORS[k / 2].1;
        for y in 0..side {
            for x in 0..side {
                let inside =
                    (side / 4..3 * side / 4).contains(&x) && (side / 4..3 * side / 4).contains(&y);
                px.extend_from_slice(if inside { &rgb } else { &[255, 255, 255] });
            }
        }
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
        max_new_tokens: 64,
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

/// Whether the export is a Qwen3-VL / Qwen3.5 model (its `config.json`
/// names a `qwen3...` model type, at the top level or in a sub-config).
fn is_qwen3() -> bool {
    std::fs::read_to_string(std::path::Path::new(&model()).join("config.json"))
        .expect("config.json in CASCADIA_LIVE_VLM")
        .contains("\"model_type\": \"qwen3")
}

/// All eight colors, in order: the model saw all 16 frames, not GenAI's own
/// resample of them. Qwen3-VL and Qwen3.5 name every one; Qwen2.5-VL 7B
/// reads two-frame color changes at 1 fps less finely (on GenAI 2026.4.1 it
/// names six of the eight, the same with or without frame metadata), so for
/// other families it is enough that four colors show the frames arrived.
fn assert_all_colors(text: &str) {
    let lower = text.to_lowercase();
    if !is_qwen3() {
        let named = COLORS
            .iter()
            .filter(|(names, _)| names.iter().any(|n| lower.contains(n)))
            .count();
        assert!(named >= 4, "only {named} colors named: {text:?}");
        return;
    }
    let mut at = 0;
    for (names, _) in COLORS {
        let found = names
            .iter()
            .filter_map(|n| lower[at..].find(n).map(|i| (i, n.len())))
            .min();
        let Some((i, len)) = found else {
            panic!("{names:?} missing after byte {at}: {text:?}");
        };
        at += i + len;
    }
}

/// Step the pipeline until `h` finishes; its text and prompt tokens.
fn run_to_end(pipe: &CbPipeline, h: &mut CbHandle) -> (String, Option<u32>) {
    let mut text = String::new();
    loop {
        pipe.step().unwrap();
        let r = h.read().unwrap();
        text.push_str(&r.text_delta);
        match r.status {
            CbStatus::Running => {}
            CbStatus::Finished => return (text, h.input_tokens()),
            other => panic!("request ended {other:?}: {text:?}"),
        }
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
    let (text, video_tokens) = run_to_end(&pipe, &mut h);
    eprintln!("cb: input_tokens={video_tokens:?} text={text:?}");
    assert_all_colors(&text);

    // An image and a video in one request: each fills its own tag sequence,
    // so the prompt holds both (more tokens than the video alone). One
    // question, about the image: a small model need not answer two.
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
    let prompt = "<ov_genai_image_0><ov_genai_video_0>What color is the image? One word.";
    let mut h = pipe.add_request(2, prompt, &media, &cfg()).unwrap();
    let (text, mixed_tokens) = run_to_end(&pipe, &mut h);
    eprintln!("cb mixed: input_tokens={mixed_tokens:?} text={text:?}");
    let (video_tokens, mixed_tokens) = (video_tokens.unwrap(), mixed_tokens.unwrap());
    assert!(
        mixed_tokens > video_tokens + 100,
        "the image was not placed: {mixed_tokens} vs {video_tokens} for the video alone"
    );
    assert!(!text.trim().is_empty());
}
