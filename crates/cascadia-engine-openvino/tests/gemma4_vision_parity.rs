//! Parity of the Rust Gemma 4 image preprocessing with Hugging Face
//! transformers 5.14.1, against `tests/fixtures/gemma4_vision/` (see its
//! README for how the fixtures were made).
//!
//! - resized size, patch count, soft tokens and position ids: equal to both
//!   HF processors';
//! - pixel values: bit-identical to the PIL processor, and within two 8-bit
//!   levels (2/255, Gemma 4 does not normalise) of the torchvision one;
//! - prompts: the image soft-token runs and `mm_token_type_ids` the Rust
//!   helpers produce match the processor's.

use std::path::PathBuf;

use cascadia_engine_openvino::gemma4_vision::{
    expand_image_tokens, preprocess, token_type_ids, ImageTokens, PreprocessConfig,
};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4_vision")
}

fn reference() -> Value {
    let text = std::fs::read_to_string(fixtures().join("reference.json")).unwrap();
    serde_json::from_str(&text).unwrap()
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

/// The fixture's processor settings, read back so a regenerated fixture with
/// different constants fails here rather than comparing apples to oranges.
fn config(r: &Value) -> PreprocessConfig {
    let p = &r["processor"];
    let c = PreprocessConfig::gemma4();
    assert_eq!(p["patch_size"], c.patch_size);
    assert_eq!(p["max_soft_tokens"], c.max_soft_tokens);
    assert_eq!(p["pooling_kernel_size"], c.pooling_kernel_size);
    assert_eq!(p["do_normalize"], false);
    assert_eq!(p["resample"], 3, "fixtures were made with bicubic");
    c
}

/// Compare a tensor with an HF summary: shape exact, every sampled value
/// within `tol` (0 = bit-identical in f32), and the sums consistent with that.
fn check_summary(name: &str, which: &str, got: &[f64], shape: &[usize], s: &Value, tol: f64) {
    let want_shape: Vec<usize> = s["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    // HF's tensors carry a batch dimension of 1.
    let want_shape: Vec<usize> = match want_shape.split_first() {
        Some((1, rest)) if rest.len() == shape.len() => rest.to_vec(),
        _ => want_shape,
    };
    assert_eq!(want_shape, shape, "{name} {which}: shape");
    let check = |label: &str, want: &[f64], have: &mut dyn Iterator<Item = f64>| {
        for (i, (&w, h)) in want.iter().zip(have).enumerate() {
            let ok = if tol == 0.0 {
                w as f32 == h as f32
            } else {
                (w - h).abs() <= tol
            };
            assert!(ok, "{name} {which} {label}[{i}]: want {w}, got {h}");
        }
    };
    let head = floats(&s["head"]);
    let tail = floats(&s["tail"]);
    let stride = floats(&s["stride_1009"]);
    check("head", &head, &mut got.iter().copied());
    check(
        "tail",
        &tail,
        &mut got[got.len() - tail.len()..].iter().copied(),
    );
    check("stride", &stride, &mut got.iter().step_by(1009).copied());
    assert_eq!(
        stride.len(),
        got.len().div_ceil(1009),
        "{name} {which}: stride count"
    );
    let n = got.len() as f64;
    let slack = tol * n + 1e-6 * n.max(1.0);
    let sum: f64 = got.iter().sum();
    let abs: f64 = got.iter().map(|v| v.abs()).sum();
    assert!(
        (sum - s["sum"].as_f64().unwrap()).abs() <= slack,
        "{name} {which}: sum {sum} vs {}",
        s["sum"]
    );
    assert!(
        (abs - s["abs_sum"].as_f64().unwrap()).abs() <= slack,
        "{name} {which}: abs_sum"
    );
}

#[test]
fn preprocessing_matches_both_hf_processors() {
    let r = reference();
    let base = config(&r);
    let images = r["images"].as_object().unwrap();
    assert!(images.len() >= 8, "fixture has {} images", images.len());
    for (name, e) in images {
        let src = e.get("source").and_then(Value::as_str).unwrap_or(name);
        let cfg = match e.get("max_soft_tokens").and_then(Value::as_u64) {
            Some(n) => PreprocessConfig {
                max_soft_tokens: n as u32,
                ..base.clone()
            },
            None => base.clone(),
        };
        let png = image::open(fixtures().join(format!("{src}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let p = preprocess(png.as_raw(), w, h, &cfg).unwrap();
        let pixels: Vec<f64> = p.pixel_values.iter().map(|&v| f64::from(v)).collect();
        let positions: Vec<f64> = p.position_ids.iter().map(|&v| v as f64).collect();
        for (which, tol) in [("slow", 0.0), ("fast", 2.0 / 255.0 + 1e-6)] {
            let want = &e[which];
            let hw = ints(&want["resized_hw"]);
            assert_eq!(
                [i64::from(p.resized_hw.0), i64::from(p.resized_hw.1)],
                [hw[0], hw[1]],
                "{name} {which}: resized size"
            );
            assert_eq!(
                p.num_patches as u64,
                want["num_patches"].as_u64().unwrap(),
                "{name} {which}: patches"
            );
            assert_eq!(
                p.soft_tokens(cfg.pooling_kernel_size) as u64,
                want["num_soft_tokens"].as_u64().unwrap(),
                "{name} {which}: soft tokens"
            );
            check_summary(
                name,
                which,
                &positions,
                &[p.rows, 2],
                &want["position_ids"],
                0.0,
            );
            check_summary(
                name,
                which,
                &pixels,
                &[p.rows, p.cols],
                &want["pixel_values"],
                tol,
            );
        }
    }
}

#[test]
fn prompt_image_runs_match_the_processor() {
    let r = reference();
    let t = &r["tokens"];
    let (image, boi, eoi) = (
        t["image_id"].as_i64().unwrap(),
        t["boi_id"].as_i64().unwrap(),
        t["eoi_id"].as_i64().unwrap(),
    );
    for (name, e) in r["prompts"].as_object().unwrap() {
        let ids = ints(&e["input_ids"]);
        let soft = ints(&e["num_soft_tokens"]);
        // Every image is boi, `soft` image tokens, eoi, in prompt order.
        let mut runs = Vec::new();
        let mut i = 0;
        while i < ids.len() {
            if ids[i] == image {
                let start = i;
                while i < ids.len() && ids[i] == image {
                    i += 1;
                }
                assert_eq!(
                    ids[start - 1],
                    boi,
                    "{name}: image run at {start} lacks boi"
                );
                assert_eq!(
                    ids.get(i),
                    Some(&eoi),
                    "{name}: image run at {start} lacks eoi"
                );
                runs.push((i - start) as i64);
            } else {
                i += 1;
            }
        }
        assert_eq!(runs, soft, "{name}: image runs");
        // Collapsing each framed run back to the template's one placeholder
        // and expanding it again reproduces the processor's ids.
        let tokens = ImageTokens {
            image,
            boi,
            eoi,
            pad: t["pad_id"].as_i64().unwrap(),
        };
        let collapsed: Vec<i64> = ids
            .iter()
            .enumerate()
            .filter(|&(i, &x)| x != boi && x != eoi && !(x == image && ids[i - 1] == image))
            .map(|(_, &x)| x)
            .collect();
        let counts: Vec<usize> = soft.iter().map(|&n| n as usize).collect();
        assert_eq!(
            expand_image_tokens(&collapsed, &tokens, &counts).unwrap(),
            ids,
            "{name}: expansion"
        );
        if let Some(tt) = e.get("mm_token_type_ids") {
            assert_eq!(token_type_ids(&ids, image), ints(tt), "{name}: token types");
        }
    }
}
