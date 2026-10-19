//! Native video parity of the Rust Gemma 4 front end with Hugging Face's
//! `Gemma4Processor` (transformers 5.20.1), against
//! `tests/fixtures/gemma4_vision/video_reference.json` (E4B) and
//! `video_reference_26b_31b.json` (26B-A4B and 31B, which share it), made by
//! `tools/vlm_fixtures/gen_gemma4_video_fixtures.py` from synthetic frames
//! this test rebuilds, [`frame_pixels`]:
//!
//! - frames: the frames HF keeps (all of a pre-sampled video, 32 of a
//!   40-frame one), their `mm:ss` times, soft tokens per frame, resized size;
//! - pixel values: per frame within two 8-bit levels of the (torchvision-only)
//!   video processor, position ids exact;
//! - prompts: the text rank 0 tokenizes equals the text HF tokenized, and the
//!   id expansion gives HF's `input_ids`; `mm_token_type_ids` and the
//!   attention blocks (one per frame) match `get_block_sequence_ids_for_mask`,
//!   for one video, two videos, minutes past the hour, a sampled video, and
//!   videos mixed with images across turns;
//! - `ov-runtime` on 26B / 31B (bidirectional inside each image and frame):
//!   the `token_type_ids` every stage takes, the per-row block ids packed
//!   slots carry, and the packed host masks (sliding: each frame open both
//!   ways; full: causal) equal HF's `create_masks_for_vision_model`.

use std::path::PathBuf;
use std::sync::Arc;

use cascadia_engine_openvino::gemma4_vision::video::{self, Item, VideoConfig, VideoMarkup};
use cascadia_engine_openvino::gemma4_vision::{
    preprocess, soft_tokens_for_size, token_type_ids, ImageTokens, PreprocessConfig,
};
use cascadia_engine_openvino::packed::{prompt_image_ids, PackedKv, PackedPlan};
use cascadia_ov_genai_shim::DType;
use cascadia_types::{MediaInput, MediaKind};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4_vision")
}

/// E2B / E4B (causal), and 26B-A4B / 31B (bidirectional inside images).
const E4B: &str = "video_reference.json";
const BIG: &str = "video_reference_26b_31b.json";

fn reference(file: &str) -> Value {
    let p = fixtures().join(file);
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
    for file in [E4B, BIG] {
        frames_match(&reference(file));
    }
}

fn frames_match(r: &Value) {
    let cfg = config(r);
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

/// A prompt's media in prompt order, rebuilt from the fixture.
fn media(r: &Value, p: &Value) -> Vec<MediaInput> {
    p["media"]
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
        .collect()
}

/// Each media item's soft tokens, as rank 0 lays them out from sizes.
fn items(media: &[MediaInput], cfg: &VideoConfig) -> Vec<Item> {
    media
        .iter()
        .map(|m| match m.kind {
            MediaKind::Video => {
                let l = video::layout(m, cfg).unwrap();
                Item::Video {
                    frames: l.frames.len(),
                    soft_tokens: l.soft_tokens,
                }
            }
            _ => Item::Image(
                soft_tokens_for_size(m.width, m.height, &PreprocessConfig::gemma4()).unwrap(),
            ),
        })
        .collect()
}

#[test]
fn prompts_match_the_processor() {
    // E2B / E4B attend causally: the gemma4 engine needs no per-frame block
    // mask. 26B / 31B open each image and frame both ways (ov-runtime).
    let r = reference(E4B);
    assert!(r["tokens"]["use_bidirectional_attention"].is_null());
    prompts_match(&r);
    let r = reference(BIG);
    assert_eq!(r["tokens"]["use_bidirectional_attention"], "vision");
    prompts_match(&r);
}

fn prompts_match(r: &Value) {
    let cfg = config(r);
    let (t, video_token) = tokens(r);
    let markup = VideoMarkup::gemma4();
    let prompts = r["prompts"].as_object().unwrap();
    assert_eq!(prompts.len(), 8);
    for (name, p) in prompts {
        let media = media(r, p);

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
        let items = items(&media, &cfg);
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

/// The API's 4 s clip: 4 frames of 63 soft tokens, 299 tokens with E4B's
/// template (against 1109 for the same frames sent as labelled images) and
/// 303 with 26B / 31B's (which opens an empty thought channel).
#[test]
fn the_api_clip_costs_what_hf_says() {
    for (file, want) in [(E4B, 299), (BIG, 303)] {
        let r = reference(file);
        let p = &r["prompts"]["api_like"];
        assert_eq!(p["input_ids"].as_array().unwrap().len(), want, "{file}");
        assert_eq!(r["videos"]["api4"]["soft_tokens_per_frame"], 63, "{file}");
    }
}

/// The keys row `r` of a one-slot packed mask may attend, as one range
/// (`past` columns of empty KV precede the chunk's own columns).
fn allowed(buf: &[u8], ctx: usize, past: usize, r: usize) -> [i64; 2] {
    const BLOCKED: u16 = 0xFBFF; // the most negative finite f16
    let cols: Vec<usize> = (0..ctx)
        .filter(|&c| {
            let off = (r * ctx + c) * 2;
            u16::from_le_bytes([buf[off], buf[off + 1]]) != BLOCKED
        })
        .collect();
    let (lo, hi) = (cols[0], *cols.last().unwrap());
    assert!(lo >= past, "row {r} sees an empty past column");
    assert_eq!(cols.len(), hi - lo + 1, "row {r}: not one range");
    [(lo - past) as i64, (hi - past) as i64]
}

/// ov-runtime on 26B / 31B: rank 0 hands stage 0 the video soft tokens as
/// `<|image|>`, and from those ids every stage's `token_type_ids` (1 on
/// image and video soft tokens, as GenAI 2026.4.1 feeds the same IR), the
/// packed rows' block ids and the packed host masks follow; each equals HF.
#[test]
fn ov_runtime_types_blocks_and_masks_match_hf() {
    let r = reference(BIG);
    let cfg = config(&r);
    let (t, video_token) = tokens(&r);
    let window = r["tokens"]["sliding_window"].as_u64().unwrap() as usize;
    assert_eq!(window, 1024);
    for (name, p) in r["prompts"].as_object().unwrap() {
        let media = media(&r, p);
        let hf = ints(&p["input_ids"]);
        let tokenized = collapse_images(&hf, &t);
        let got =
            video::expand_media_tokens(&tokenized, &t, video_token, &items(&media, &cfg)).unwrap();
        let staged = video::stage_ids(&got, t.image, video_token);

        let mm = ints(&p["mm_token_type_ids"]);
        let types = token_type_ids(&staged, t.image);
        let want: Vec<i64> = mm.iter().map(|&m| i64::from(m != 0)).collect();
        assert_eq!(types, want, "{name}: token_type_ids");

        let blocks: Vec<i64> = prompt_image_ids(&staged, t.image)
            .iter()
            .map(|b| b.map_or(-1, i64::from))
            .collect();
        assert_eq!(blocks, ints(&p["block_ids"]), "{name}: packed block ids");

        // The whole prompt as one packed prefill chunk of one slot.
        let n = staged.len();
        let kv = PackedKv::new(1, n, n, 1, 1, 2, DType::F16, 0);
        let plan = PackedPlan::chunk_with_images(n, 0, &prompt_image_ids(&staged, t.image));
        for (kind, w, open) in [
            ("sliding_attention", Some(window), true),
            ("full_attention", None, false),
        ] {
            let mut buf = Vec::new();
            kv.fill_mask_blocks(&plan, &mut buf, DType::F16, w, open);
            let want = p["masks"][kind].as_array().unwrap();
            assert_eq!(want.len(), n, "{name} {kind}: rows");
            for (q, w) in want.iter().enumerate() {
                let w = ints(w);
                assert_eq!(
                    allowed(&buf, kv.context, n, q),
                    [w[0], w[1]],
                    "{name} {kind}: row {q}"
                );
            }
        }
    }
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
    // E4B, 26B-A4B and 31B ship the same tokenizer.
    for file in [E4B, BIG] {
        rank0_tokenizes(&tok, &reference(file));
    }
}

fn rank0_tokenizes(tok: &tokenizers::Tokenizer, r: &Value) {
    let cfg = config(r);
    let (t, _) = tokens(r);
    for (name, p) in r["prompts"].as_object().unwrap() {
        let media = media(r, p);
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
