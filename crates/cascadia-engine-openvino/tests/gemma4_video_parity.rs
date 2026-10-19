//! Native video parity of the Rust Gemma 4 front end with Hugging Face's
//! `Gemma4Processor` (transformers 5.20.1), against
//! `tests/fixtures/gemma4_vision/video_reference.json` (made by
//! `tools/vlm_fixtures/gen_gemma4_video_fixtures.py` from synthetic frames
//! this test rebuilds, [`frame_pixels`]):
//!
//! - frames: the frames HF keeps (all of a pre-sampled video, 32 of a
//!   40-frame one), their `mm:ss` times, soft tokens per frame, resized size;
//! - pixel values: per frame within two 8-bit levels of the (torchvision-only)
//!   video processor, position ids exact;
//! - prompts: the text rank 0 tokenizes equals the text HF tokenized, and the
//!   id expansion gives HF's `input_ids`; `mm_token_type_ids` and the
//!   attention blocks (one per frame) match `get_block_sequence_ids_for_mask`,
//!   for one video, two videos, minutes past the hour, a sampled video, and
//!   videos mixed with images across turns.

use std::path::PathBuf;
use std::sync::Arc;

use cascadia_engine_openvino::gemma4_vision::video::{self, Item, VideoConfig, VideoMarkup};
use cascadia_engine_openvino::gemma4_vision::{
    preprocess, soft_tokens_for_size, ImageTokens, PreprocessConfig,
};
use cascadia_types::{MediaInput, MediaKind};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4_vision")
}

fn reference() -> Value {
    let p = fixtures().join("video_reference.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// The generator's `frame_pixels`: `[frames, height, width, 3]` RGB8.
fn frame_pixels(frames: u32, height: u32, width: u32, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity((frames * height * width * 3) as usize);
    for f in 0..u64::from(frames) {
        for y in 0..u64::from(height) {
            for x in 0..u64::from(width) {
                for c in 0..3u64 {
                    let v = x * 3 + y * 5 + f * (17 + seed) + c * 71 + ((x * y + seed) % 23) * 2;
                    out.push((v % 256) as u8);
                }
            }
        }
    }
    out
}

/// Seeds by video name, as the generator's `VIDEOS`.
fn seed(name: &str) -> u64 {
    match name {
        "api4" => 1,
        "odd3" => 2,
        "one1" => 3,
        "minutes3" => 4,
        "tall2" => 5,
        "long40" => 6,
        other => panic!("unknown video {other}"),
    }
}

/// A video as the API sends it: every decoded frame and its time.
fn video_input(v: &Value, name: &str) -> MediaInput {
    let (n, w, h) = (
        v["frames"].as_u64().unwrap() as u32,
        v["width"].as_u64().unwrap() as u32,
        v["height"].as_u64().unwrap() as u32,
    );
    MediaInput {
        kind: MediaKind::Video,
        pixels: Arc::from(frame_pixels(n, h, w, seed(name))),
        num_frames: n,
        width: w,
        height: h,
        timestamps_ms: Some(
            ints(&v["timestamps_ms"])
                .iter()
                .map(|&t| t as u32)
                .collect(),
        ),
        sha256: [0; 32],
    }
}

fn image_input(name: &str) -> MediaInput {
    let png = image::open(fixtures().join(format!("{name}.png")))
        .unwrap()
        .to_rgb8();
    let (w, h) = png.dimensions();
    MediaInput {
        kind: MediaKind::Image,
        pixels: Arc::from(png.into_raw()),
        num_frames: 1,
        width: w,
        height: h,
        timestamps_ms: None,
        sha256: [0; 32],
    }
}

/// The fixture's video processor settings, read back as the engine reads
/// `processor_config.json`'s `video_processor`.
fn config(r: &Value) -> VideoConfig {
    let p = &r["video_processor"];
    let mut obj = p.as_object().unwrap().clone();
    obj.retain(|k, _| {
        [
            "patch_size",
            "max_soft_tokens",
            "pooling_kernel_size",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "num_frames",
            "do_sample_frames",
        ]
        .contains(&k.as_str())
    });
    let dir = std::env::temp_dir().join(format!("cascadia-g4v-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("processor_config.json"),
        serde_json::json!({ "video_processor": obj }).to_string(),
    )
    .unwrap();
    let c = VideoConfig::from_model_dir(&dir).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(p["resample"], 3, "fixtures were made with bicubic");
    assert_eq!(
        (c.frame.max_soft_tokens, c.num_frames, c.do_sample_frames),
        (70, 32, true)
    );
    c
}

fn tokens(r: &Value) -> (ImageTokens, i64) {
    let t = &r["tokens"];
    let id = |k: &str| t[k].as_i64().unwrap();
    (
        ImageTokens {
            image: id("image"),
            boi: id("boi"),
            eoi: id("eoi"),
            pad: id("pad"),
        },
        id("video"),
    )
}

/// Sampled values within `tol` of HF's summary, shapes equal.
fn check_summary(what: &str, got: &[f64], shape: &[usize], s: &Value, tol: f64) {
    let want: Vec<usize> = ints(&s["shape"]).iter().map(|&x| x as usize).collect();
    assert_eq!(want, shape, "{what}: shape");
    let close = |label: &str, want: &[f64], have: &mut dyn Iterator<Item = f64>| {
        for (i, (&w, h)) in want.iter().zip(have).enumerate() {
            assert!(
                (w - h).abs() <= tol,
                "{what} {label}[{i}]: want {w}, got {h}"
            );
        }
    };
    let tail = floats(&s["tail"]);
    close("head", &floats(&s["head"]), &mut got.iter().copied());
    close(
        "tail",
        &tail,
        &mut got[got.len() - tail.len()..].iter().copied(),
    );
    let stride = floats(&s["stride_1009"]);
    close("stride", &stride, &mut got.iter().step_by(1009).copied());
    assert_eq!(
        stride.len(),
        got.len().div_ceil(1009),
        "{what}: stride count"
    );
    let n = got.len() as f64;
    let sum: f64 = got.iter().sum();
    assert!(
        (sum - s["sum"].as_f64().unwrap()).abs() <= tol * n + 1e-6 * n,
        "{what}: sum {sum} vs {}",
        s["sum"]
    );
}

#[test]
fn frames_times_and_pixels_match_the_video_processor() {
    let r = reference();
    let cfg = config(&r);
    let videos = r["videos"].as_object().unwrap();
    assert_eq!(videos.len(), 6);
    for (name, v) in videos {
        let m = video_input(v, name);
        let l = video::layout(&m, &cfg).unwrap();
        let kept: Vec<usize> = ints(&v["kept_frames"])
            .iter()
            .map(|&x| x as usize)
            .collect();
        assert_eq!(l.frames, kept, "{name}: frames kept");
        let kept_ms: Vec<u32> = ints(&v["kept_ms"]).iter().map(|&x| x as u32).collect();
        assert_eq!(l.timestamps_ms, kept_ms, "{name}: frame times");
        let labels: Vec<String> = l
            .timestamps_ms
            .iter()
            .map(|&t| video::timestamp(t))
            .collect();
        let want: Vec<String> = v["timestamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(labels, want, "{name}: mm:ss");
        assert_eq!(
            l.soft_tokens as u64,
            v["soft_tokens_per_frame"].as_u64().unwrap(),
            "{name}: soft tokens per frame"
        );

        // Every kept frame through the image preprocessing at the video
        // budget, stacked as HF's `[frames, max_patches, patch_dim]`.
        let mut pixels = Vec::new();
        let mut positions = Vec::new();
        let mut shape = [0usize; 3];
        for &f in &l.frames {
            let p = preprocess(video::frame(&m, f), m.width, m.height, &cfg.frame).unwrap();
            let hw = ints(&v["resized_hw"]);
            assert_eq!(
                [i64::from(p.resized_hw.0), i64::from(p.resized_hw.1)],
                [hw[0], hw[1]],
                "{name}: resized size"
            );
            assert_eq!(p.num_patches as u64, v["num_patches"].as_u64().unwrap());
            pixels.extend(p.pixel_values.iter().map(|&x| f64::from(x)));
            positions.extend(p.position_ids.iter().map(|&x| x as f64));
            shape = [l.frames.len(), p.rows, p.cols];
        }
        check_summary(
            &format!("{name} pixel_values_videos"),
            &pixels,
            &shape,
            &v["pixel_values_videos"],
            2.0 / 255.0 + 1e-6,
        );
        check_summary(
            &format!("{name} video_position_ids"),
            &positions,
            &[shape[0], shape[1], 2],
            &v["video_position_ids"],
            0.0,
        );
    }
}

/// HF's ids with each image's framed run collapsed back to the one
/// `<|image|>` the tokenizer sees in the template (videos stay expanded:
/// rank 0 expands them in the text).
fn collapse_images(ids: &[i64], t: &ImageTokens) -> Vec<i64> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == t.boi && ids.get(i + 1) == Some(&t.image) {
            let n = ids[i + 1..].iter().take_while(|&&x| x == t.image).count();
            assert_eq!(ids[i + 1 + n], t.eoi);
            out.push(t.image);
            i += n + 2;
        } else {
            assert_ne!(ids[i], t.image, "an image token outside an image");
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

/// HF's expanded text with each image's `<|image>`, soft tokens and
/// `<image|>` collapsed back to the template's one `<|image|>`.
fn collapse_image_text(text: &str) -> String {
    const OPEN: &str = "<|image><|image|>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(OPEN) {
        out.push_str(&rest[..at]);
        out.push_str("<|image|>");
        let after = &rest[at + "<|image>".len()..];
        let end = after.find("<image|>").expect("an image run ends");
        assert!(after[..end].split("<|image|>").all(str::is_empty));
        rest = &after[end + "<image|>".len()..];
    }
    out.push_str(rest);
    out
}

#[test]
fn prompts_match_the_processor() {
    let r = reference();
    let cfg = config(&r);
    let (t, video_token) = tokens(&r);
    // E2B / E4B attend causally: the engine needs no per-frame block mask.
    assert!(r["tokens"]["use_bidirectional_attention"].is_null());
    let markup = VideoMarkup::gemma4();
    let prompts = r["prompts"].as_object().unwrap();
    assert_eq!(prompts.len(), 8);
    for (name, p) in prompts {
        let media: Vec<MediaInput> = p["media"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let n = m["name"].as_str().unwrap();
                match m["kind"].as_str().unwrap() {
                    "video" => video_input(&r["videos"][n], n),
                    _ => image_input(n),
                }
            })
            .collect();

        // The text rank 0 tokenizes is the text HF tokenized, but for the
        // images, which rank 0 expands on ids.
        let template = p["template_text"].as_str().unwrap();
        let text = video::prompt_text(template, &media, &cfg, &markup);
        assert_eq!(
            text,
            collapse_image_text(p["expanded_text"].as_str().unwrap()),
            "{name}: text"
        );

        // From the tokenized text (images still one placeholder each), the
        // id expansion gives HF's input_ids.
        let want = ints(&p["input_ids"]);
        let items: Vec<Item> = media
            .iter()
            .map(|m| match m.kind {
                MediaKind::Video => {
                    let l = video::layout(m, &cfg).unwrap();
                    Item::Video {
                        frames: l.frames.len(),
                        soft_tokens: l.soft_tokens,
                    }
                }
                _ => Item::Image(
                    soft_tokens_for_size(m.width, m.height, &PreprocessConfig::gemma4()).unwrap(),
                ),
            })
            .collect();
        let tokenized = collapse_images(&want, &t);
        let got = video::expand_media_tokens(&tokenized, &t, video_token, &items).unwrap();
        assert_eq!(got, want, "{name}: input_ids");

        // Modality types and attention blocks: one block per frame (and per
        // image), as HF numbers them.
        let types: Vec<i64> = got
            .iter()
            .map(|&x| {
                if x == t.image {
                    1
                } else if x == video_token {
                    2
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(types, ints(&p["mm_token_type_ids"]), "{name}: mm types");
        let blocks = video::block_ids(&got, t.image, video_token);
        assert_eq!(blocks, ints(&p["block_ids"]), "{name}: blocks");
        let frames: usize = items
            .iter()
            .map(|i| match *i {
                Item::Video { frames, .. } => frames,
                Item::Image(_) => 1,
            })
            .sum();
        assert_eq!(
            blocks.iter().max().map_or(0, |&b| b + 1) as usize,
            frames,
            "{name}: one block per frame and image"
        );

        // Stage 0 sees every soft token as `<|image|>`, the same count.
        let staged = video::stage_ids(&got, t.image, video_token);
        assert!(!staged.contains(&video_token));
        assert_eq!(
            staged.iter().filter(|&&x| x == t.image).count(),
            types.iter().filter(|&&x| x != 0).count()
        );
    }
}

/// The API's 4 s clip: 4 frames of 63 soft tokens, 299 tokens with the
/// template, against E4B's 1109 for the same frames sent as labelled images.
#[test]
fn the_api_clip_costs_what_hf_says() {
    let r = reference();
    let p = &r["prompts"]["api_like"];
    assert_eq!(p["input_ids"].as_array().unwrap().len(), 299);
    assert_eq!(r["videos"]["api4"]["soft_tokens_per_frame"], 63);
}

/// With `GEMMA4_TOKENIZER` pointing at a Gemma 4 `tokenizer.json`, the
/// `tokenizers` crate rank 0 uses turns the expanded text into HF's ids
/// (images still one placeholder each). Skipped without it: the tokenizer
/// is 32 MB and not checked in.
#[test]
fn rank0_tokenizer_reads_the_markup_as_hf() {
    let Ok(path) = std::env::var("GEMMA4_TOKENIZER") else {
        eprintln!("skipping: set GEMMA4_TOKENIZER to a Gemma 4 tokenizer.json");
        return;
    };
    let tok = tokenizers::Tokenizer::from_file(path).unwrap();
    let r = reference();
    let cfg = config(&r);
    let (t, _) = tokens(&r);
    for (name, p) in r["prompts"].as_object().unwrap() {
        let media: Vec<MediaInput> = p["media"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let n = m["name"].as_str().unwrap();
                match m["kind"].as_str().unwrap() {
                    "video" => video_input(&r["videos"][n], n),
                    _ => image_input(n),
                }
            })
            .collect();
        let text = video::prompt_text(
            p["template_text"].as_str().unwrap(),
            &media,
            &cfg,
            &VideoMarkup::gemma4(),
        );
        let ids: Vec<i64> = tok
            .encode(text.as_ref(), false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&u| i64::from(u))
            .collect();
        assert_eq!(
            ids,
            collapse_images(&ints(&p["input_ids"]), &t),
            "{name}: tokenized"
        );
    }
}
