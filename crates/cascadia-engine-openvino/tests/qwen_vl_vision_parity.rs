//! Parity of the Rust Qwen2.5-VL / Qwen3-VL front end with Hugging Face
//! transformers 5.14.1, against `tests/fixtures/qwen_vl_vision/` (made by
//! `tools/vlm_fixtures/gen_qwen_vl_fixtures.py`; the input PNGs are the
//! Qwen3.5 fixtures').
//!
//! - grids: equal to both HF processors';
//! - pixel values: bit-identical to the PIL processor, within two 8-bit
//!   levels of the torchvision one;
//! - prompts: the size-only expansion equals HF's `input_ids`, the
//!   positions equal `get_rope_index` and the decode delta equals HF's;
//! - vision host inputs: Qwen2.5-VL's window index and window boundaries
//!   exactly, Qwen3-VL's bilinear corner indices exactly, weights and rotary
//!   tables to float rounding.

use std::path::PathBuf;

use cascadia_engine_openvino::qwen35_vision::{encoder, preprocess, rope, PreprocessConfig};
use cascadia_engine_openvino::qwen_vl_vision as qvl;
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn reference() -> Value {
    let p = fixtures().join("qwen_vl_vision/reference.json");
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

/// The processor settings recorded with the fixture.
fn config(f: &Value) -> PreprocessConfig {
    let p = &f["processor"];
    let mean = floats(&p["image_mean"]);
    let std = floats(&p["image_std"]);
    assert_eq!(p["resample"], 3, "fixtures were made with bicubic");
    PreprocessConfig {
        patch_size: p["patch_size"].as_u64().unwrap() as u32,
        temporal_patch_size: p["temporal_patch_size"].as_u64().unwrap() as u32,
        merge_size: p["merge_size"].as_u64().unwrap() as u32,
        min_pixels: p["size"]["shortest_edge"].as_u64().unwrap(),
        max_pixels: p["size"]["longest_edge"].as_u64().unwrap(),
        image_mean: [mean[0] as f32, mean[1] as f32, mean[2] as f32],
        image_std: [std[0] as f32, std[1] as f32, std[2] as f32],
        rescale_factor: p["rescale_factor"].as_f64().unwrap(),
    }
}

/// Compare values with an HF summary: sampled values within `tol` (0 =
/// bit-identical in f32), sums consistent with that.
fn check_summary(what: &str, got: &[f32], shape: &[usize], s: &Value, tol: f64) {
    let want_shape: Vec<usize> = ints(&s["shape"]).iter().map(|&x| x as usize).collect();
    assert_eq!(want_shape, shape, "{what}: shape");
    let cmp = |label: &str, want: &[f64], have: &mut dyn Iterator<Item = f32>| {
        for (i, (&w, h)) in want.iter().zip(have).enumerate() {
            let ok = if tol == 0.0 {
                w as f32 == h
            } else {
                (w - f64::from(h)).abs() <= tol
            };
            assert!(ok, "{what} {label}[{i}]: want {w}, got {h}");
        }
    };
    let tail = floats(&s["tail"]);
    cmp("head", &floats(&s["head"]), &mut got.iter().copied());
    cmp(
        "tail",
        &tail,
        &mut got[got.len() - tail.len()..].iter().copied(),
    );
    let stride = floats(&s["stride_1009"]);
    cmp("stride", &stride, &mut got.iter().step_by(1009).copied());
    assert_eq!(
        stride.len(),
        got.len().div_ceil(1009),
        "{what}: stride count"
    );
    let n = got.len() as f64;
    let slack = tol * n + 1e-6 * n.max(1.0);
    let sum: f64 = got.iter().map(|&v| f64::from(v)).sum();
    let abs: f64 = got.iter().map(|&v| f64::from(v).abs()).sum();
    assert!(
        (sum - s["sum"].as_f64().unwrap()).abs() <= slack,
        "{what}: sum {sum} vs {}",
        s["sum"]
    );
    assert!(
        (abs - s["abs_sum"].as_f64().unwrap()).abs() <= slack,
        "{what}: abs_sum"
    );
}

fn families(r: &Value) -> Vec<(&str, &Value)> {
    let f = r["families"].as_object().unwrap();
    assert_eq!(f.len(), 2);
    vec![
        ("qwen2_5_vl", &f["qwen2_5_vl"]),
        ("qwen3_vl", &f["qwen3_vl"]),
    ]
}

#[test]
fn preprocessing_matches_hf_processors() {
    let r = reference();
    for (fam, f) in families(&r) {
        let base = config(f);
        // Two 8-bit levels after normalisation by the smallest std.
        let min_std = base.image_std.iter().copied().fold(f32::MAX, f32::min);
        let tol_fast = 2.0 / 255.0 / f64::from(min_std) + 1e-6;
        let images = f["images"].as_object().unwrap();
        assert_eq!(images.len(), 7, "{fam}");
        for (name, e) in images {
            let source = e.get("source").and_then(Value::as_str).unwrap_or(name);
            let mut cfg = base.clone();
            if let Some(max) = e.get("max_pixels").and_then(Value::as_u64) {
                cfg.max_pixels = max;
            }
            let png = image::open(fixtures().join(format!("qwen35_vision/{source}.png")))
                .unwrap()
                .to_rgb8();
            let (w, h) = png.dimensions();
            let p = preprocess(&[png.as_raw()], w, h, &cfg).unwrap();
            let what = format!("{fam} {name}");
            assert_eq!(
                p.grid_thw,
                grid(&e["slow"]["grid_thw"]),
                "{what}: grid vs PIL"
            );
            assert_eq!(
                p.grid_thw,
                grid(&e["fast"]["grid_thw"]),
                "{what}: grid vs torchvision"
            );
            assert_eq!(
                qvl::grid_for_size(w, h, &cfg).unwrap(),
                p.grid_thw,
                "{what}: size-only grid"
            );
            let shape = [p.rows, p.cols];
            check_summary(
                &format!("{what} pil"),
                &p.pixel_values,
                &shape,
                &e["slow"]["pixel_values"],
                0.0,
            );
            check_summary(
                &format!("{what} torchvision"),
                &p.pixel_values,
                &shape,
                &e["fast"]["pixel_values"],
                tol_fast,
            );
        }
    }
}

/// The size-only expansion reproduces HF's prompt ids and the positions and
/// decode delta are HF's.
#[test]
fn prompts_and_positions_match_hf() {
    let r = reference();
    for (fam, f) in families(&r) {
        let cfg = config(f);
        let image = f["token_ids"]["image"].as_i64().unwrap();
        let video = f["token_ids"]["video"].as_i64().unwrap();
        let merge = f["spatial_merge_size"].as_u64().unwrap() as u32;
        for name in ["single", "two_images", "multi_turn", "text_only"] {
            let p = &f["prompts"][name];
            let what = format!("{fam} {name}");
            let counts: Vec<usize> = p["image_sizes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let s = ints(s);
                    let [t, h, w] = qvl::grid_for_size(s[0] as u32, s[1] as u32, &cfg).unwrap();
                    (t * (h / merge) * (w / merge)) as usize
                })
                .collect();
            let ids = qvl::expand_pads(&ints(&p["prompt_ids"]), image, &counts).unwrap();
            assert_eq!(ids, ints(&p["input_ids"]), "{what}: expanded ids");
            let grids: Vec<[u32; 3]> = p["image_grid_thw"]
                .as_array()
                .unwrap()
                .iter()
                .map(grid)
                .collect();
            let m = rope::mrope_positions(&ids, image, video, &grids, &[], merge).unwrap();
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
            // The [3, T] block every stage is fed is HF's position_ids.
            let fed = qvl::positions(Some(&m), 0, ids.len());
            assert_eq!(fed, want.concat(), "{what}: fed positions");
        }
        // The SDK check's one 448x448 image: 256 tokens on Qwen2.5-VL (14 px
        // patches), 196 on Qwen3-VL (16 px), as GenAI's 294 / 223 prompts.
        let single = ints(&f["prompts"]["single"]["input_ids"]);
        let pads = single.iter().filter(|&&t| t == image).count();
        assert_eq!(pads, if fam == "qwen2_5_vl" { 256 } else { 196 }, "{fam}");
    }
}

#[test]
fn encoder_inputs_match_vision_utils() {
    let r = reference();
    for (fam, f) in families(&r) {
        let enc = &f["encoder"];
        let merge = f["spatial_merge_size"].as_u64().unwrap() as usize;
        let rot_dim = enc["rotary_dim"].as_u64().unwrap() as usize;
        for (name, e) in enc["grids"].as_object().unwrap() {
            let what = format!("{fam} {name}");
            let g = grid(&e["grid_thw"]);
            let n = (g[0] * g[1] * g[2]) as usize;
            let rot = encoder::rotary_pos_emb(&[g], merge, rot_dim, 10000.0);
            check_summary(
                &format!("{what} rotary"),
                &rot,
                &[n, rot_dim],
                &e["rotary_pos_emb"],
                1e-4,
            );
            let cu: Vec<i64> = encoder::cu_seqlens(&[g])
                .iter()
                .map(|&v| i64::from(v))
                .collect();
            assert_eq!(cu, ints(&e["cu_seqlens"]), "{what}: cu_seqlens");
            if fam == "qwen2_5_vl" {
                let vw = enc["window_size"].as_u64().unwrap() as usize
                    / merge
                    / enc["patch_size"].as_u64().unwrap() as usize;
                let (idx, cu_w) = qvl::window_index(&[g], merge, vw);
                assert_eq!(idx, ints(&e["window_index"]), "{what}: window_index");
                let cu_w: Vec<i64> = cu_w.iter().map(|&v| i64::from(v)).collect();
                assert_eq!(
                    cu_w,
                    ints(&e["cu_window_seqlens"]),
                    "{what}: cu_window_seqlens"
                );
            } else {
                let side = enc["num_grid_per_side"].as_u64().unwrap() as usize;
                let b = encoder::bilinear_indices_and_weights(&[g], side, merge);
                match &e["indices"] {
                    Value::Array(_) => {
                        assert_eq!(b.indices, ints(&e["indices"]), "{what}: indices")
                    }
                    s => {
                        let as_f32: Vec<f32> = b.indices.iter().map(|&v| v as f32).collect();
                        check_summary(&format!("{what} indices"), &as_f32, &[4, n], s, 0.0);
                    }
                }
                // 1e-5: torch's own f32 weights move by ~1e-6 between builds
                // (this fixture's macOS arm64 torch vs the Qwen3.5 one's x86).
                check_summary(
                    &format!("{what} weights"),
                    &b.weights,
                    &[4, n],
                    &e["weights"],
                    1e-5,
                );
            }
        }
    }
}
