//! Native video parity of the Rust Qwen VL front ends with Hugging Face
//! transformers 5.14.1, against `tests/fixtures/qwen_vl_vision/video_reference.json`
//! (made by `tools/vlm_fixtures/gen_qwen_vl_video_fixtures.py` from synthetic
//! frames this test rebuilds, [`frame_pixels`]):
//!
//! - grids: each video processor's `video_grid_thw` for odd and even frame
//!   counts, one frame, and an API-sized 8-frame clip;
//! - pixel values: within two 8-bit levels of the (torchvision-only) video
//!   processors;
//! - prompts (Qwen2.5-VL, Qwen3-VL on `ov-runtime`): the text rank 0
//!   tokenizes (Qwen3-VL's per-group timestamps), the expanded ids, HF's
//!   `get_rope_index` positions and decode delta, for one video, two videos,
//!   videos mixed with images across turns, and Qwen2.5-VL's
//!   `second_per_grid_ts` spacing at 0.5, 1, 1.5 and 2 frames a second.

use std::path::PathBuf;
use std::sync::Arc;

use cascadia_engine_openvino::qwen35_vision::video::{self, VideoResize};
use cascadia_engine_openvino::qwen35_vision::PreprocessConfig;
use cascadia_engine_openvino::qwen_vl_vision::{self as qvl, ImageTokens, Kind, Layout};
use cascadia_types::{MediaInput, MediaKind};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(name: &str) -> Value {
    let p = fixtures().join("qwen_vl_vision").join(name);
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

fn grid(v: &Value) -> [u32; 3] {
    let a = ints(v);
    [a[0] as u32, a[1] as u32, a[2] as u32]
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
        "odd5" => 1,
        "even4" => 2,
        "one1" => 3,
        "api8" => 4,
        "slow4" => 5,
        "frac4" => 6,
        other => panic!("unknown video {other}"),
    }
}

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
    let png = image::open(fixtures().join(format!("qwen35_vision/{name}.png")))
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

fn resize(model_type: &str) -> VideoResize {
    if model_type == "qwen2_5_vl" {
        VideoResize::PerFrame
    } else {
        VideoResize::Temporal
    }
}

/// Sampled values within `tol` of HF's summary, and the shape equal.
fn check_summary(what: &str, got: &[f32], shape: &[usize], s: &Value, tol: f64) {
    let want: Vec<usize> = ints(&s["shape"]).iter().map(|&x| x as usize).collect();
    assert_eq!(want, shape, "{what}: shape");
    let head = floats(&s["head"]);
    let tail = floats(&s["tail"]);
    let stride = floats(&s["stride_1009"]);
    let pairs = head
        .iter()
        .zip(got.iter())
        .chain(tail.iter().zip(got[got.len() - tail.len()..].iter()))
        .chain(stride.iter().zip(got.iter().step_by(1009)));
    for (i, (&w, &g)) in pairs.enumerate() {
        assert!(
            (w - f64::from(g)).abs() <= tol,
            "{what} sample {i}: want {w}, got {g}"
        );
    }
    let mean = |x: f64| x / got.len() as f64;
    let sum: f64 = got.iter().map(|&v| f64::from(v)).sum();
    assert!(
        (mean(sum) - mean(s["sum"].as_f64().unwrap())).abs() <= tol,
        "{what}: mean {} vs {}",
        mean(sum),
        mean(s["sum"].as_f64().unwrap())
    );
}

#[test]
fn video_grids_and_pixels_match_hf_video_processors() {
    let r = load("video_reference.json");
    let fams = r["families"].as_object().unwrap();
    assert_eq!(fams.len(), 3);
    for (fam, f) in fams {
        let cfg = PreprocessConfig::from_value(&f["video_processor"]);
        let min_std = cfg.image_std.iter().copied().fold(f32::MAX, f32::min);
        let tol = 2.0 / 255.0 / f64::from(min_std) + 1e-6;
        for (name, v) in f["videos"].as_object().unwrap() {
            let what = format!("{fam} {name}");
            let m = video_input(v, name);
            let p = video::preprocess_video(
                &m.pixels,
                m.num_frames,
                m.width,
                m.height,
                &cfg,
                resize(fam),
            )
            .unwrap();
            let want = grid(&v["video_grid_thw"]);
            assert_eq!(p.grid_thw, want, "{what}: grid");
            assert_eq!(
                video::video_grid(m.num_frames, m.width, m.height, &cfg, resize(fam)).unwrap(),
                want,
                "{what}: size-only grid"
            );
            check_summary(
                &what,
                &p.pixel_values,
                &[p.rows, p.cols],
                &v["pixel_values_videos"],
                tol,
            );
        }
    }
}

/// The `ov-runtime` layout of a family: its image processor (from the image
/// fixtures), its video processor and token ids.
fn layout(kind: Kind, image: &Value, f: &Value) -> Layout {
    let tok = |k: &str| f["token_ids"][k].as_i64().unwrap();
    let preprocess = PreprocessConfig::from_value(&image["processor"]);
    Layout {
        kind,
        merge: preprocess.merge_size,
        preprocess,
        video: PreprocessConfig::from_value(&f["video_processor"]),
        tokens: ImageTokens {
            image_pad: tok("image"),
            video_pad: tok("video"),
            vision_start: tok("vision_start"),
            vision_end: tok("vision_end"),
        },
        tokens_per_second: f["tokens_per_second"].as_u64().map(|t| t as u32),
    }
}

#[test]
fn video_prompts_and_positions_match_hf() {
    let r = load("video_reference.json");
    let images = load("reference.json");
    for (fam, kind) in [("qwen2_5_vl", Kind::Qwen25Vl), ("qwen3_vl", Kind::Qwen3Vl)] {
        let f = &r["families"][fam];
        let l = layout(kind, &images["families"][fam], f);
        let (image, video) = (l.tokens.image_pad, l.tokens.video_pad);
        let prompts = f["prompts"].as_object().unwrap();
        assert_eq!(prompts.len(), 8, "{fam}");
        for (name, p) in prompts {
            let what = format!("{fam} {name}");
            let media: Vec<MediaInput> = p["media"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    let n = m["name"].as_str().unwrap();
                    match m["kind"].as_str().unwrap() {
                        "video" => video_input(&f["videos"][n], n),
                        _ => image_input(n),
                    }
                })
                .collect();
            // The text rank 0 tokenizes: Qwen3-VL's timestamps written in.
            let text = l.prompt_text(p["template_text"].as_str().unwrap(), &media);
            assert_eq!(text, p["prompt_text"].as_str().unwrap(), "{what}: text");
            // Its ids (one pad per run, as HF's collapsed) expand to HF's,
            // with video pads fed as image pads.
            let hf_ids = ints(&p["input_ids"]);
            let (ids, m, grids) = l.prepare_ids(&media, &ints(&p["prompt_ids"])).unwrap();
            let fed: Vec<i64> = hf_ids
                .iter()
                .map(|&t| if t == video { image } else { t })
                .collect();
            assert_eq!(ids, fed, "{what}: expanded ids");
            assert_eq!(
                l.expand_ids(&media, &ints(&p["prompt_ids"])).unwrap(),
                ids,
                "{what}: size-only expansion"
            );
            let (mut vg, mut ig) = (Vec::new(), Vec::new());
            for (g, it) in grids.iter().zip(&media) {
                if it.kind == MediaKind::Video {
                    vg.push(*g)
                } else {
                    ig.push(*g)
                }
            }
            let want_v: Vec<[u32; 3]> = p["video_grid_thw"]
                .as_array()
                .unwrap()
                .iter()
                .map(grid)
                .collect();
            let want_i: Vec<[u32; 3]> = p["image_grid_thw"]
                .as_array()
                .unwrap()
                .iter()
                .map(grid)
                .collect();
            assert_eq!((vg, ig), (want_v, want_i), "{what}: grids");
            let want: Vec<Vec<i64>> = p["position_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(ints)
                .collect();
            assert_eq!(&m.thw[..], &want[..], "{what}: positions");
            assert_eq!(
                m.rope_delta,
                p["rope_delta"].as_i64().unwrap(),
                "{what}: rope_delta"
            );
            assert_eq!(
                qvl::positions(Some(&m), 0, ids.len()),
                want.concat(),
                "{what}: fed positions"
            );
            if let Some(spg) = p.get("second_per_grid_ts") {
                let ours: Vec<Option<f64>> = media
                    .iter()
                    .filter(|m| m.kind == MediaKind::Video)
                    .map(|m| video::second_per_grid(m.timestamps_ms.as_deref().unwrap(), 2))
                    .collect();
                // A one-frame video has no spacing to read (and one group).
                for (o, h) in ours.iter().zip(floats(spg)) {
                    if let Some(o) = o {
                        // HF keeps it in f32.
                        assert!((o - h).abs() < 1e-3, "{what}: second_per_grid {o} vs {h}");
                    }
                }
            }
        }
    }
}

/// The token cost a video saves: Qwen3-VL's 8-frame 448x252 clip is 4
/// groups of 8x14 tokens (plus timestamps), half its frames as images.
#[test]
fn a_video_costs_one_frame_pair_per_group() {
    let r = load("video_reference.json");
    let f = &r["families"]["qwen3_vl"];
    let p = &f["prompts"]["api_like"];
    let video = f["token_ids"]["video"].as_i64().unwrap();
    let pads = ints(&p["input_ids"])
        .iter()
        .filter(|&&t| t == video)
        .count();
    assert_eq!(pads, 4 * 8 * 14);
    let image = load("reference.json");
    let l = layout(Kind::Qwen3Vl, &image["families"]["qwen3_vl"], f);
    let [_, h, w] = l.grid_for_size(448, 252).unwrap();
    assert_eq!(8 * (h / 2) * (w / 2), 2 * pads as u32);
}
