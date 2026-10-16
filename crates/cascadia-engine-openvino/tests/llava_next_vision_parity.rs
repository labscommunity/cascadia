//! Parity of the Rust LLaVA-NeXT image input with Hugging Face transformers
//! 5.14.1, against `tests/fixtures/llava_next_vision/reference.json`
//! (`tools/vlm_fixtures/gen_llava_next_fixtures.py` on the configs and
//! tokenizer of OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov; the images are the
//! Pixtral fixture PNGs).
//!
//! - pinpoint, tile grid and view count: equal to the PIL processor's;
//! - pixel values of every view: bit-identical to the PIL processor (the
//!   torchvision one differs by up to 0.24 on the 3x1200 image);
//! - token count per image and over a sweep of sizes: equal to
//!   `LlavaNextProcessor`'s;
//! - packed rows: the same rows in the same order as the model's
//!   `pack_image_features` (base view, unpadded tile grid, `image_newline`
//!   closing each row);
//! - prompts: expanding the template's one `<image>` per image reproduces the
//!   processor's input_ids (the generator also checks that the prompt
//!   Cascadia renders tokenizes to them).

use std::path::PathBuf;

use cascadia_engine_openvino::llava_next_vision::{
    expand_image_tokens, image_tokens, layout, pack_features, preprocess, PreprocessConfig,
};
use serde_json::Value;

fn fixture(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
}

fn reference() -> Value {
    let path = fixture("llava_next_vision").join("reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// The processor settings the fixtures were made with.
fn config_of(r: &Value) -> PreprocessConfig {
    let p = &r["processor"];
    let pins = p["image_grid_pinpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| {
            let x = ints(x);
            (x[0] as u32, x[1] as u32)
        })
        .collect();
    PreprocessConfig {
        pinpoints: pins,
        tile: p["crop_size"]["height"].as_u64().unwrap() as u32,
        patch_size: r["patch_size"].as_u64().unwrap() as u32,
        ..PreprocessConfig::llava_v16()
    }
}

#[test]
fn fixtures_match_the_shipped_defaults() {
    let r = reference();
    assert_eq!(config_of(&r), PreprocessConfig::llava_v16());
    assert_eq!(r["processor"]["resample"], 3, "bicubic");
    assert_eq!(r["processor"]["size"]["shortest_edge"], 336);
}

#[test]
fn layout_and_pixels_are_bit_identical_to_the_pil_processor() {
    let r = reference();
    let cfg = config_of(&r);
    for (name, e) in r["images"].as_object().unwrap() {
        let png = image::open(fixture("pixtral_vision").join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        assert_eq!(ints(&e["size"]), [i64::from(w), i64::from(h)]);
        let v = preprocess(png.as_raw(), w, h, &cfg).unwrap();
        let best = ints(&e["best_resolution"]);
        assert_eq!(
            [i64::from(v.layout.best.0), i64::from(v.layout.best.1)],
            [best[0], best[1]],
            "{name}: pinpoint"
        );
        let grid = ints(&e["grid"]);
        assert_eq!(
            [i64::from(v.layout.grid.0), i64::from(v.layout.grid.1)],
            [grid[0], grid[1]],
            "{name}: tile grid"
        );
        assert_eq!(v.layout.views() as i64, e["views"].as_i64().unwrap());
        let s = &e["pixel_values"];
        let shape: Vec<usize> = ints(&s["shape"]).iter().map(|&x| x as usize).collect();
        assert_eq!(shape, vec![v.layout.views(), 3, 336, 336], "{name}: shape");
        let got = &v.pixel_values;
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
        let sum: f64 = got.iter().map(|&x| f64::from(x)).sum();
        let want = s["sum"].as_f64().unwrap();
        assert!(
            (sum - want).abs() <= 1e-6 * got.len() as f64,
            "{name}: sum {sum} vs {want}"
        );
    }
}

#[test]
fn token_counts_equal_the_processor() {
    let r = reference();
    let cfg = config_of(&r);
    for (name, e) in r["images"].as_object().unwrap() {
        let s = ints(&e["size"]);
        assert_eq!(
            image_tokens(s[1] as u32, s[0] as u32, &cfg).unwrap() as i64,
            e["tokens"].as_i64().unwrap(),
            "{name}"
        );
    }
    let sweep = r["counts"].as_array().unwrap();
    assert!(sweep.len() > 1000);
    for c in sweep {
        let c = ints(c);
        assert_eq!(
            image_tokens(c[0] as u32, c[1] as u32, &cfg).unwrap() as i64,
            c[2],
            "{}x{} (h x w)",
            c[0],
            c[1]
        );
    }
}

/// Feed `pack_features` rows that name their source (`view * 576 + pos` in
/// channel 0, its negation minus one in channel 1; newline `[-1, -2]`), as
/// the generator fed HF's `pack_image_features`, and compare the layouts.
#[test]
fn packing_matches_pack_image_features() {
    let r = reference();
    let cfg = config_of(&r);
    for (name, e) in r["images"].as_object().unwrap() {
        let s = ints(&e["size"]);
        let lay = layout(s[1] as u32, s[0] as u32, &cfg).unwrap();
        let rows = lay.views() * lay.per_view as usize;
        let feat: Vec<f32> = (0..rows)
            .flat_map(|i| [i as f32, -(i as f32) - 1.0])
            .collect();
        let packed = pack_features(&feat, 2, &lay, &[-1.0, -2.0]).unwrap();
        let mut want = Vec::new();
        for run in e["packed_runs"].as_array().unwrap() {
            let run = ints(run);
            for k in 0..run[1] {
                want.push(if run[0] == -1 { -1 } else { run[0] + k });
            }
        }
        assert_eq!(packed.len(), want.len() * 2, "{name}: rows");
        assert_eq!(want.len(), e["tokens"].as_u64().unwrap() as usize);
        for (i, (row, &w)) in packed.chunks_exact(2).zip(&want).enumerate() {
            let second = if w == -1 { -2.0 } else { -(w as f32) - 1.0 };
            assert_eq!(row, [w as f32, second], "{name}: packed row {i}");
        }
    }
}

#[test]
fn prompt_expansion_reproduces_the_processor_ids() {
    let r = reference();
    let image = r["image_token"].as_i64().unwrap();
    for (name, p) in r["prompts"].as_object().unwrap() {
        let ids = ints(&p["input_ids"]);
        let counts: Vec<usize> = p["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| r["images"][n.as_str().unwrap()]["tokens"].as_u64().unwrap() as usize)
            .collect();
        let mut collapsed = Vec::new();
        for (i, &x) in ids.iter().enumerate() {
            if x != image || i == 0 || ids[i - 1] != image {
                collapsed.push(x);
            }
        }
        assert_eq!(
            expand_image_tokens(&collapsed, image, &counts).unwrap(),
            ids,
            "{name}: expansion"
        );
    }
    // The SDK check's one 448x448 image: HF counts 2945 prompt tokens.
    assert_eq!(ints(&r["prompts"]["single"]["input_ids"]).len(), 2945);
    assert_eq!(ints(&r["prompts"]["two_images"]["input_ids"]).len(), 4199);
}
