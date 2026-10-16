//! Parity of the Rust Phi-3.5-vision image path with the export's own
//! remote-code processor and `modeling_phi3_v.py` assembly, against
//! `tests/fixtures/phi3v_vision/` (`tools/vlm_fixtures/gen_phi3v_fixtures.py`
//! on OpenVINO/Phi-3.5-vision-instruct-int4-ov; the eight images of
//! `tests/fixtures/pixtral_vision/`).
//!
//! - HD geometry and soft-token counts: equal to the processor's, from the
//!   image size alone;
//! - crops: bit-identical (Pillow bilinear, white padding, f32 normalise);
//!   the global view (torch bicubic on the normalised tensor) within 1e-5;
//! - feature assembly (2x2 merge, crop tiling, `sub_GN` / `glb_GN`):
//!   bit-identical to `hd_feature_transform` on synthetic features;
//! - prompts: the per-chunk layout with each image expanded reproduces the
//!   processor's input_ids (its negative image ids standing as `<|image|>`).

use std::path::PathBuf;

use cascadia_engine_openvino::phi3v_vision::{
    assemble, encode_prompt, expand_image_tokens, hd_geometry, num_image_tokens, preprocess,
    split_image_tags, PreprocessConfig, CROP,
};
use serde_json::Value;

fn fixtures(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
}

fn reference() -> Value {
    let path = fixtures("phi3v_vision").join("reference.json");
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

/// `got` against a fixture summary: head, tail and every 1009th value
/// within `tol` (0 = bit-identical), and the sum.
fn check_summary(label: &str, s: &Value, got: &[f32], tol: f32) {
    let shape: usize = ints(&s["shape"]).iter().product::<i64>() as usize;
    assert_eq!(shape, got.len(), "{label}: length");
    let cmp = |what: &str, want: Vec<f64>, have: Vec<f32>| {
        assert_eq!(want.len(), have.len(), "{label} {what}: count");
        for (i, (w, h)) in want.iter().zip(have).enumerate() {
            let w = *w as f32;
            assert!(
                (w - h).abs() <= tol,
                "{label} {what}[{i}]: want {w}, got {h}"
            );
        }
    };
    cmp(
        "head",
        floats(&s["head"]),
        got.iter().take(64).copied().collect(),
    );
    let n = floats(&s["tail"]).len();
    cmp("tail", floats(&s["tail"]), got[got.len() - n..].to_vec());
    cmp(
        "stride",
        floats(&s["stride_1009"]),
        got.iter().step_by(1009).copied().collect(),
    );
    let sum: f64 = got.iter().map(|&v| f64::from(v)).sum();
    let want = s["sum"].as_f64().unwrap();
    let slack = if tol == 0.0 { 1e-6 } else { f64::from(tol) };
    assert!(
        (sum - want).abs() <= slack * got.len() as f64,
        "{label}: sum {sum} vs {want}"
    );
}

#[test]
fn processor_settings_are_the_shipped_defaults() {
    let r = reference();
    let p = &r["processor"];
    let cfg = PreprocessConfig::phi35();
    assert_eq!(p["num_crops"].as_u64(), Some(u64::from(cfg.num_crops)));
    assert_eq!(
        p["num_img_tokens"].as_u64(),
        Some(u64::from(cfg.num_img_tokens))
    );
    let mean = floats(&p["image_mean"]);
    let std = floats(&p["image_std"]);
    for c in 0..3 {
        assert_eq!(mean[c] as f32, cfg.image_mean[c]);
        assert_eq!(std[c] as f32, cfg.image_std[c]);
    }
}

#[test]
fn geometry_and_token_counts_match_from_size_alone() {
    let r = reference();
    let cfg = PreprocessConfig::phi35();
    for (name, e) in r["images"].as_object().unwrap() {
        let size = ints(&e["size"]);
        let (w, h) = (size[0] as u32, size[1] as u32);
        let g = hd_geometry(w, h, cfg.num_crops).unwrap();
        let hw = ints(&e["image_size_hw"]);
        assert_eq!(
            [i64::from(g.image_size().0), i64::from(g.image_size().1)],
            [hw[0], hw[1]],
            "{name}: HD size"
        );
        assert_eq!(
            num_image_tokens(w, h, &cfg).unwrap() as u64,
            e["num_img_tokens"].as_u64().unwrap(),
            "{name}: soft tokens"
        );
    }
}

#[test]
fn pixels_match_the_processor() {
    let r = reference();
    let cfg = PreprocessConfig::phi35();
    let view = 3 * (CROP * CROP) as usize;
    for (name, e) in r["images"].as_object().unwrap() {
        let png = image::open(fixtures("pixtral_vision").join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let px = preprocess(png.as_raw(), w, h, &cfg).unwrap();
        check_summary(
            &format!("{name} crops"),
            &e["crops_pixels"],
            &px.pixel_values[view..],
            0.0,
        );
        check_summary(
            &format!("{name} global"),
            &e["global"],
            &px.pixel_values[..view],
            1e-5,
        );
    }
}

fn synthetic_features(views: usize) -> Vec<f32> {
    let (l, d) = (576usize, 1024usize);
    let mut out = Vec::with_capacity(views * l * d);
    for v in 0..views {
        for p in 0..l {
            for c in 0..d {
                out.push(((v * 7919 + p * 31 + c * 7) % 1000) as f32 / 8.0 - 60.0);
            }
        }
    }
    out
}

#[test]
fn assembly_matches_hd_feature_transform() {
    let r = reference();
    let sep = |k: &str| -> Vec<f32> {
        floats(&r["separators"][k])
            .into_iter()
            .map(|v| v as f32)
            .collect()
    };
    let (sub, glb) = (sep("sub_GN"), sep("glb_GN"));
    for (name, a) in r["assembly"].as_object().unwrap() {
        let hw = ints(&a["image_size_hw"]);
        let crops = (hw[0] as usize / 336, hw[1] as usize / 336);
        let views = a["views"].as_u64().unwrap() as usize;
        assert_eq!(views, 1 + crops.0 * crops.1);
        let rows = assemble(&synthetic_features(views), 1024, crops, &sub, &glb).unwrap();
        check_summary(&format!("{name} assembly"), &a["rows"], &rows, 0.0);
    }
}

#[test]
fn prompt_layout_reproduces_the_processor_ids() {
    let r = reference();
    let image = r["tokens"]["image"].as_i64().unwrap();
    for (name, p) in r["prompts"].as_object().unwrap() {
        let template = p["template"].as_str().unwrap();
        let (chunks, _) = split_image_tags(template).unwrap();
        let want_chunks: Vec<&str> = p["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap())
            .collect();
        assert_eq!(chunks, want_chunks, "{name}: chunks");
        // The tokenizer's per-chunk ids (with BOS), as HF gives them.
        let mut chunk_ids = p["chunk_ids"].as_array().unwrap().iter().map(ints);
        let ids = encode_prompt(template, image, |_| Ok(chunk_ids.next().unwrap())).unwrap();
        let counts: Vec<usize> = ints(&p["num_img_tokens"])
            .iter()
            .map(|&n| n as usize)
            .collect();
        let got = expand_image_tokens(&ids, image, &counts).unwrap();
        let want: Vec<i64> = ints(&p["input_ids"])
            .into_iter()
            .map(|x| if x < 0 { image } else { x })
            .collect();
        assert_eq!(got, want, "{name}: ids");
    }
}

/// With `CASCADIA_PHI3V_TOKENIZER` pointing at the export's
/// `tokenizer.json`, the real tokenizer gives the fixture's chunk ids (the
/// tokenizers crate is the library HF's fast tokenizer runs). Skipped
/// otherwise: the file is 3.6 MB.
#[test]
fn real_tokenizer_gives_the_processor_chunks() {
    let Ok(path) = std::env::var("CASCADIA_PHI3V_TOKENIZER") else {
        return;
    };
    let tok = tokenizers::Tokenizer::from_file(path).unwrap();
    let r = reference();
    for (name, p) in r["prompts"].as_object().unwrap() {
        for (chunk, want) in p["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .zip(p["chunk_ids"].as_array().unwrap())
        {
            let got: Vec<i64> = tok
                .encode(chunk.as_str().unwrap(), true)
                .unwrap()
                .get_ids()
                .iter()
                .map(|&u| i64::from(u))
                .collect();
            assert_eq!(got, ints(want), "{name}: chunk {chunk}");
        }
    }
}
