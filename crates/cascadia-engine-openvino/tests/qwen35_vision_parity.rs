//! Parity of the Rust Qwen3.5 vision preprocessing and M-RoPE with Hugging
//! Face transformers 5.14.1, against `tests/fixtures/qwen35_vision/` (see its
//! README for how the fixtures were made).
//!
//! - grids: equal to both HF processors';
//! - pixel values: bit-identical to the PIL processor, and within one 8-bit
//!   level (2/255 after normalisation) of the torchvision one;
//! - positions and rope delta: equal to `get_rope_index`, video included.

use std::path::PathBuf;

use cascadia_engine_openvino::qwen35_vision::{encoder, preprocess, rope, PreprocessConfig};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_vision")
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

fn grid(v: &Value) -> [u32; 3] {
    let a: Vec<u32> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    [a[0], a[1], a[2]]
}

/// The fixture's processor settings, read back so a regenerated fixture with
/// different constants fails here rather than comparing apples to oranges.
fn config(r: &Value) -> PreprocessConfig {
    let p = &r["processor"];
    let c = PreprocessConfig::qwen35();
    assert_eq!(p["patch_size"], c.patch_size);
    assert_eq!(p["temporal_patch_size"], c.temporal_patch_size);
    assert_eq!(p["merge_size"], c.merge_size);
    assert_eq!(p["size"]["shortest_edge"], c.min_pixels);
    assert_eq!(p["size"]["longest_edge"], c.max_pixels);
    assert_eq!(floats(&p["image_mean"]), vec![0.5; 3]);
    assert_eq!(floats(&p["image_std"]), vec![0.5; 3]);
    assert_eq!(p["resample"], 3, "fixtures were made with bicubic");
    c
}

/// Compare a pixel tensor with an HF summary: shape exact, every sampled
/// value within `tol`, and the sums consistent with that.
fn check_summary(
    name: &str,
    which: &str,
    got: &[f32],
    rows: usize,
    cols: usize,
    s: &Value,
    tol: f64,
) {
    let shape: Vec<usize> = s["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(shape, vec![rows, cols], "{name} {which}: shape");
    let worst = |label: &str, want: &[f64], have: &mut dyn Iterator<Item = f32>| {
        let mut max = 0f64;
        for (i, (&w, h)) in want.iter().zip(have).enumerate() {
            // tol 0 means bit-identical in f32: the reference passes through
            // f64 when summarised, so compare it rounded back to f32.
            let ok = if tol == 0.0 {
                w as f32 == h
            } else {
                (w - f64::from(h)).abs() <= tol
            };
            assert!(ok, "{name} {which} {label}[{i}]: want {w}, got {h}");
            max = max.max((w - f64::from(h)).abs());
        }
        max
    };
    let head = floats(&s["head"]);
    let tail = floats(&s["tail"]);
    let stride = floats(&s["stride_1009"]);
    worst("head", &head, &mut got.iter().copied());
    worst(
        "tail",
        &tail,
        &mut got[got.len() - tail.len()..].iter().copied(),
    );
    worst("stride", &stride, &mut got.iter().step_by(1009).copied());
    assert_eq!(
        stride.len(),
        got.len().div_ceil(1009),
        "{name} {which}: stride sample count"
    );
    let sum: f64 = got.iter().map(|&v| f64::from(v)).sum();
    let abs: f64 = got.iter().map(|&v| f64::from(v).abs()).sum();
    let n = got.len() as f64;
    // Per-value error <= tol bounds the sums; f64 accumulation adds ~1e-9.
    let slack = tol * n + 1e-6 * n.max(1.0);
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
fn preprocessing_matches_hf_processors() {
    let r = reference();
    let cfg = config(&r);
    let images = r["images"].as_object().unwrap();
    assert_eq!(images.len(), 7);
    let mut downscaled = false;
    for (name, e) in images {
        // Low-budget entries re-run a source image with a lower max_pixels to
        // force a strong (antialiased) downscale.
        let source = e.get("source").and_then(Value::as_str).unwrap_or(name);
        let mut cfg = cfg.clone();
        if let Some(max) = e.get("max_pixels").and_then(Value::as_u64) {
            cfg.max_pixels = max;
        }
        let png = image::open(fixtures().join(format!("{source}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        assert_eq!(
            vec![w as u64, h as u64],
            e["size"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect::<Vec<_>>()
        );
        let p = preprocess(&[png.as_raw()], w, h, &cfg).unwrap();
        let [_, gh, gw] = p.grid_thw;
        downscaled |= w >= 2 * gw * cfg.patch_size && h >= 2 * gh * cfg.patch_size;
        assert_eq!(
            p.grid_thw,
            grid(&e["slow"]["grid_thw"]),
            "{name}: grid vs PIL"
        );
        assert_eq!(
            p.grid_thw,
            grid(&e["fast"]["grid_thw"]),
            "{name}: grid vs torchvision"
        );
        // PIL backend: same algorithm, so bit-identical.
        check_summary(
            name,
            "pil",
            &p.pixel_values,
            p.rows,
            p.cols,
            &e["slow"]["pixel_values"],
            0.0,
        );
        // torchvision backend: one 8-bit level apart at most (2/255 normalised).
        check_summary(
            name,
            "torchvision",
            &p.pixel_values,
            p.rows,
            p.cols,
            &e["fast"]["pixel_values"],
            2.0 / 255.0 + 1e-6,
        );
    }
    assert!(
        downscaled,
        "no fixture shrinks an image 2x or more; the antialiased kernel is untested"
    );
}

#[test]
fn positions_match_get_rope_index() {
    let r = reference();
    let image_token = r["token_ids"]["image"].as_i64().unwrap();
    let video_token = r["token_ids"]["video"].as_i64().unwrap();
    let merge = r["spatial_merge_size"].as_u64().unwrap() as u32;
    let prompts = r["prompts"].as_object().unwrap();
    for name in ["single", "two_images", "multi_turn", "video", "text_only"] {
        let p = &prompts[name];
        assert!(
            p.get("error").is_none(),
            "{name}: fixture recorded an error: {}",
            p["error"]
        );
        let ids: Vec<i64> = p["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        let grids =
            |k: &str| -> Vec<[u32; 3]> { p[k].as_array().unwrap().iter().map(grid).collect() };
        let got = rope::mrope_positions(
            &ids,
            image_token,
            video_token,
            &grids("image_grid_thw"),
            &grids("video_grid_thw"),
            merge,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want: Vec<Vec<i64>> = p["position_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect()
            })
            .collect();
        for (axis, (g, w)) in got.thw.iter().zip(&want).enumerate() {
            if let Some(i) = g.iter().zip(w).position(|(a, b)| a != b) {
                panic!(
                    "{name}: axis {axis} differs at token {i}: got {}, want {}",
                    g[i], w[i]
                );
            }
            assert_eq!(g.len(), w.len(), "{name}: axis {axis} length");
        }
        assert_eq!(
            got.rope_delta,
            p["rope_delta"].as_i64().unwrap(),
            "{name}: rope_delta"
        );
        // The 4-row layout prepends the plain token index.
        let four = got.four_rows();
        assert_eq!(four[0], (0..ids.len() as i64).collect::<Vec<_>>());
        assert_eq!(&four[1..], &got.thw[..]);
    }
}

/// An image's prompt span is exactly its grid's merged token count: the
/// count the API must expand `<|image_pad|>` to.
#[test]
fn llm_tokens_match_the_prompt_image_runs() {
    let r = reference();
    let cfg = config(&r);
    let image_token = r["token_ids"]["image"].as_i64().unwrap();
    let single = &r["prompts"]["single"];
    let ids = single["input_ids"].as_array().unwrap();
    let pads = ids
        .iter()
        .filter(|x| x.as_i64() == Some(image_token))
        .count();
    let png = image::open(fixtures().join("square_448.png"))
        .unwrap()
        .to_rgb8();
    let p = preprocess(&[png.as_raw()], 448, 448, &cfg).unwrap();
    assert_eq!(p.llm_tokens(cfg.merge_size), pads);
}

/// The vision IRs' host inputs equal transformers' `vision_utils`: corner
/// indices and cu_seqlens exactly, weights and rotary to float rounding.
#[test]
fn encoder_inputs_match_vision_utils() {
    let r = reference();
    let enc = &r["encoder"];
    let side = enc["num_grid_per_side"].as_u64().unwrap() as usize;
    let rot_dim = enc["rotary_dim"].as_u64().unwrap() as usize;
    let merge = r["spatial_merge_size"].as_u64().unwrap() as usize;
    let grids = enc["grids"].as_object().unwrap();
    assert!(
        grids.contains_key("video_4f_128x96"),
        "video grid must be covered"
    );
    for (name, e) in grids {
        let g = grid(&e["grid_thw"]);
        let b = encoder::bilinear_indices_and_weights(&[g], side, merge);
        let n = b.positions;
        match &e["indices"] {
            Value::Array(all) => {
                let want: Vec<i64> = all.iter().map(|x| x.as_i64().unwrap()).collect();
                if let Some(i) = want.iter().zip(&b.indices).position(|(a, b)| a != b) {
                    panic!(
                        "{name}: index {i} (corner {}, pos {}) got {} want {}",
                        i / n,
                        i % n,
                        b.indices[i],
                        want[i]
                    );
                }
                assert_eq!(want.len(), b.indices.len(), "{name}: index count");
            }
            summary => {
                let as_f32: Vec<f32> = b.indices.iter().map(|&v| v as f32).collect();
                check_summary(name, "indices", &as_f32, 4, n, summary, 0.0);
            }
        }
        check_summary(name, "weights", &b.weights, 4, n, &e["weights"], 1e-6);
        let rot = encoder::rotary_pos_emb(&[g], merge, rot_dim, 10000.0);
        // Rust's powf and torch's pow may differ in the last bit of a
        // frequency; scaled by positions up to ~100 that stays far below 1e-4.
        check_summary(name, "rotary", &rot, n, rot_dim, &e["rotary_pos_emb"], 1e-4);
        let cu: Vec<i64> = e["cu_seqlens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(
            encoder::cu_seqlens(&[g])
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<_>>(),
            cu,
            "{name}: cu_seqlens"
        );
    }
}
