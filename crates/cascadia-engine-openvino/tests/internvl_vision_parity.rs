//! Parity of the Rust InternVL2 image path with the model card's
//! `load_image` / `dynamic_preprocess` and `chat`'s prompt expansion, against
//! `tests/fixtures/internvl_vision/` (`tools/vlm_fixtures/gen_internvl_fixtures.py`
//! on OpenVINO/InternVL2-4B-int4-ov; the eight images of
//! `tests/fixtures/pixtral_vision/`).
//!
//! - tile grids and view counts (thumbnail included): equal, from size alone;
//! - pixel values: bit-identical (Pillow bicubic, ToTensor + Normalize);
//! - Python's target-ratio order is the table the Rust side uses;
//! - prompts: widening each `<img><IMG_CONTEXT></img>` of the template's ids
//!   reproduces the ids of the string `chat` builds.

use std::path::PathBuf;

use cascadia_engine_openvino::internvl_vision::{
    expand_image_tokens, num_image_tokens, preprocess, target_ratios, tiles_for_size, ImageTokens,
    PreprocessConfig,
};
use serde_json::Value;

fn fixtures(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
}

fn reference() -> Value {
    let path = fixtures("internvl_vision").join("reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
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

fn config_of(r: &Value) -> PreprocessConfig {
    let c = &r["config"];
    let triple = |k: &str| {
        let v = floats(&c[k]);
        [v[0] as f32, v[1] as f32, v[2] as f32]
    };
    PreprocessConfig {
        image_size: c["image_size"].as_u64().unwrap() as u32,
        min_tiles: c["min_tiles"].as_u64().unwrap() as u32,
        max_tiles: c["max_tiles"].as_u64().unwrap() as u32,
        use_thumbnail: c["use_thumbnail"].as_bool().unwrap(),
        tokens_per_tile: c["tokens_per_tile"].as_u64().unwrap() as u32,
        image_mean: triple("image_mean"),
        image_std: triple("image_std"),
    }
}

#[test]
fn settings_are_the_shipped_defaults_and_ratio_order_is_pythons() {
    let r = reference();
    assert_eq!(config_of(&r), PreprocessConfig::internvl2());
    let pairs = |v: &Value| -> Vec<(u32, u32)> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let p = ints(p);
                (p[0] as u32, p[1] as u32)
            })
            .collect()
    };
    assert_eq!(target_ratios(1, 12), pairs(&r["ratios"]["12"]));
    // Other settings: the same set in area order (ties may be ordered
    // differently, which only matters for an exact tie in closeness).
    let mut six = target_ratios(1, 6);
    let mut want = pairs(&r["ratios"]["6"]);
    assert!(six.windows(2).all(|w| w[0].0 * w[0].1 <= w[1].0 * w[1].1));
    six.sort();
    want.sort();
    assert_eq!(six, want);
}

#[test]
fn tiles_and_pixels_match_load_image() {
    let r = reference();
    let cfg = config_of(&r);
    for (name, e) in r["images"].as_object().unwrap() {
        let png = image::open(fixtures("pixtral_vision").join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let (grid, views) = tiles_for_size(w, h, &cfg).unwrap();
        let g = ints(&e["grid"]);
        assert_eq!(
            [i64::from(grid.0), i64::from(grid.1)],
            [g[0], g[1]],
            "{name}: grid"
        );
        assert_eq!(views as u64, e["views"].as_u64().unwrap(), "{name}: views");
        assert_eq!(
            num_image_tokens(w, h, &cfg).unwrap() as u64,
            e["num_tokens"].as_u64().unwrap()
        );
        let px = preprocess(png.as_raw(), w, h, &cfg).unwrap();
        let s = &e["pixel_values"];
        let n: i64 = ints(&s["shape"]).iter().product();
        let got = &px.pixel_values;
        assert_eq!(got.len(), n as usize, "{name}: length");
        let check = |label: &str, want: Vec<f64>, have: Vec<f32>| {
            for (i, (w, h)) in want.iter().zip(have).enumerate() {
                assert_eq!(*w as f32, h, "{name} {label}[{i}]");
            }
        };
        check("head", floats(&s["head"]), got[..64].to_vec());
        let tail = floats(&s["tail"]);
        let k = tail.len();
        check("tail", tail, got[got.len() - k..].to_vec());
        check(
            "stride",
            floats(&s["stride_1009"]),
            got.iter().step_by(1009).copied().collect(),
        );
        let sum: f64 = got.iter().map(|&v| f64::from(v)).sum();
        let want = s["sum"].as_f64().unwrap();
        assert!((sum - want).abs() <= 1e-6 * got.len() as f64, "{name}: sum");
    }
}

#[test]
fn prompt_expansion_reproduces_chat_ids() {
    let r = reference();
    let t = &r["tokens"];
    let tokens = ImageTokens {
        context: t["context"].as_i64().unwrap(),
        start: t["start"].as_i64().unwrap(),
        end: t["end"].as_i64().unwrap(),
    };
    for (name, p) in r["prompts"].as_object().unwrap() {
        let counts: Vec<usize> = ints(&p["num_tokens"]).iter().map(|&n| n as usize).collect();
        let got = expand_image_tokens(&ints(&p["template_ids"]), &tokens, &counts).unwrap();
        assert_eq!(got, ints(&p["input_ids"]), "{name}");
    }
}
