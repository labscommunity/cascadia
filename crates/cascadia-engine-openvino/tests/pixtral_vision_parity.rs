//! Parity of the Rust Pixtral image preprocessing and prompt layout with
//! Hugging Face transformers 5.14.1, against `tests/fixtures/pixtral_vision/`
//! (`tools/vlm_fixtures/gen_pixtral_fixtures.py` on OpenVINO/pixtral-12b-int4-ov)
//! and `tests/fixtures/mistral3_vision/` (the same processor with
//! `spatial_merge_size` 2, on an optimum-intel export of
//! mistralai/Ministral-3-3B-Instruct-2512-BF16; the same eight images).
//!
//! - resized size and patch grid: equal to both HF processors';
//! - pixel values: bit-identical to the PIL processor (the torchvision one
//!   resamples differently, by up to 0.3 on extreme aspect ratios, so it is
//!   not a bit-level reference);
//! - prompts: expanding the template's one `[IMG]` per image reproduces the
//!   processor's input_ids.

use std::path::PathBuf;

use cascadia_engine_openvino::pixtral_vision::{
    expand_image_tokens, preprocess, ImageTokens, PreprocessConfig,
};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pixtral_vision")
}

fn reference_in(dir: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
        .join("reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn reference() -> Value {
    reference_in("pixtral_vision")
}

/// The processor settings a fixture set was made with.
fn config_of(r: &Value) -> PreprocessConfig {
    let p = &r["processor"];
    let patch = p["patch_size"]
        .get("height")
        .unwrap_or(&p["patch_size"])
        .as_u64()
        .unwrap() as u32;
    PreprocessConfig {
        patch_size: patch,
        spatial_merge: r["spatial_merge_size"].as_u64().unwrap_or(1) as u32,
        longest_edge: p["size"]["longest_edge"].as_u64().unwrap() as u32,
        ..PreprocessConfig::pixtral()
    }
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

fn tokens(r: &Value) -> ImageTokens {
    let t = &r["tokens"];
    ImageTokens {
        image: t["image"].as_i64().unwrap(),
        image_break: t["image_break"].as_i64().unwrap(),
        image_end: t["image_end"].as_i64().unwrap(),
    }
}

#[test]
fn preprocessing_is_bit_identical_to_the_pil_processor() {
    let r = reference();
    let cfg = PreprocessConfig::pixtral();
    assert_eq!(
        config_of(&r),
        cfg,
        "Pixtral fixtures match the shipped defaults"
    );
    check_preprocessing(&r, &cfg);
}

/// Mistral 3: 14-px patches merged 2x2, so sides are whole 28-px cells and
/// the token grid is half the patch grid; the pixels are still the PIL
/// processor's, bit for bit.
#[test]
fn mistral3_preprocessing_is_bit_identical_to_the_pil_processor() {
    let r = reference_in("mistral3_vision");
    let cfg = config_of(&r);
    assert_eq!(
        (cfg.patch_size, cfg.spatial_merge, cfg.longest_edge),
        (14, 2, 1540)
    );
    check_preprocessing(&r, &cfg);
}

#[test]
fn mistral3_prompt_expansion_reproduces_the_processor_ids() {
    check_expansion(&reference_in("mistral3_vision"));
}

fn check_preprocessing(r: &Value, cfg: &PreprocessConfig) {
    let p = &r["processor"];
    assert_eq!(p["resample"], 3, "fixtures were made with bicubic");
    for (name, e) in r["images"].as_object().unwrap() {
        let png = image::open(fixtures().join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let px = preprocess(png.as_raw(), w, h, cfg).unwrap();
        for which in ["slow", "fast"] {
            let hw = ints(&e[which]["resized_hw"]);
            assert_eq!(
                [i64::from(px.height), i64::from(px.width)],
                [hw[0], hw[1]],
                "{name} {which}: resized size"
            );
            let grid = ints(&e[which]["grid"]);
            assert_eq!([i64::from(px.rows), i64::from(px.cols)], [grid[0], grid[1]]);
        }
        let s = &e["slow"]["pixel_values"];
        let shape: Vec<usize> = ints(&s["shape"]).iter().map(|&v| v as usize).collect();
        assert_eq!(
            shape,
            vec![3, px.height as usize, px.width as usize],
            "{name}: shape"
        );
        let got = &px.pixel_values;
        let check = |label: &str, want: Vec<f64>, have: &mut dyn Iterator<Item = &f32>| {
            for (i, (w, h)) in want.iter().zip(have).enumerate() {
                assert_eq!(*w as f32, *h, "{name} {label}[{i}]");
            }
        };
        check("head", floats(&s["head"]), &mut got.iter());
        let tail = floats(&s["tail"]);
        let n = tail.len();
        check("tail", tail, &mut got[got.len() - n..].iter());
        check(
            "stride",
            floats(&s["stride_1009"]),
            &mut got.iter().step_by(1009),
        );
        let sum: f64 = got.iter().map(|&v| f64::from(v)).sum();
        let want = s["sum"].as_f64().unwrap();
        assert!(
            (sum - want).abs() <= 1e-6 * got.len() as f64,
            "{name}: sum {sum} vs {want}"
        );
    }
}

#[test]
fn prompt_expansion_reproduces_the_processor_ids() {
    check_expansion(&reference());
}

fn check_expansion(r: &Value) {
    let t = tokens(r);
    for (name, e) in r["prompts"].as_object().unwrap() {
        let ids = ints(&e["input_ids"]);
        let grids: Vec<(u32, u32)> = e["grids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                let g = ints(g);
                (g[0] as u32, g[1] as u32)
            })
            .collect();
        // Collapse every image back to the template's single [IMG]: drop the
        // break/end tokens and all but the first [IMG] of each image.
        let mut collapsed = Vec::new();
        let mut in_image = false;
        for &x in &ids {
            if x == t.image {
                if !in_image {
                    collapsed.push(x);
                    in_image = true;
                }
            } else if x == t.image_break {
                continue;
            } else if x == t.image_end {
                in_image = false;
            } else {
                collapsed.push(x);
            }
        }
        assert_eq!(
            expand_image_tokens(&collapsed, &t, &grids).unwrap(),
            ids,
            "{name}: expansion"
        );
    }
}
