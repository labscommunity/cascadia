//! Parity of the Rust Gemma 3 image preprocessing and prompt layout with
//! Hugging Face transformers 5.14.1 (Pillow 12.3), against
//! `tests/fixtures/gemma3_vision/` (`tools/vlm_fixtures/gen_gemma3_fixtures.py`
//! on OpenVINO/gemma-3-12b-it-int4-ov's configs and tokenizer; the images are
//! the Gemma 4 fixture PNGs).
//!
//! - pixel values: bit-identical to the PIL processor (896x896 bilinear,
//!   rescale, mean/std 0.5); the torchvision one differs by up to 2/255;
//! - prompts: the text expansion reproduces the processor's expanded text,
//!   its input_ids pass `check_expanded` with one 256-token run per image,
//!   and the token types rank 0 sends are the processor's `token_type_ids`;
//! - with `CASCADIA_GEMMA3_TOKENIZER` naming a `tokenizer.json` (the
//!   export's), the Rust tokenizer on the expanded text gives the
//!   processor's input_ids exactly.

use std::path::PathBuf;

use cascadia_engine_openvino::gemma3_vision::{
    check_expanded, expand_prompt_text, preprocess, ImageTokens, PreprocessConfig,
};
use cascadia_engine_openvino::gemma4_vision::token_type_ids;
use serde_json::Value;

fn fixtures(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
}

fn reference() -> Value {
    let text = std::fs::read_to_string(fixtures("gemma3_vision").join("reference.json")).unwrap();
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

fn tokens(r: &Value) -> ImageTokens {
    let t = &r["tokens"];
    let s = |k: &str| t[k].as_str().unwrap().to_string();
    ImageTokens {
        image: t["image_id"].as_i64().unwrap(),
        boi: t["boi_id"].as_i64().unwrap(),
        eoi: t["eoi_id"].as_i64().unwrap(),
        image_text: s("image"),
        boi_text: s("boi"),
        eoi_text: s("eoi"),
    }
}

#[test]
fn config_is_the_shipped_default() {
    let r = reference();
    let p = &r["processor"];
    assert_eq!(p["resample"], 2, "fixtures were made with bilinear");
    let v: Value = serde_json::json!({
        "size": p["size"], "resample": p["resample"], "rescale_factor": p["rescale_factor"],
        "do_normalize": p["do_normalize"], "image_mean": p["image_mean"],
        "image_std": p["image_std"], "image_seq_length": r["image_seq_length"],
    });
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("preprocessor_config.json"), v.to_string()).unwrap();
    assert_eq!(
        PreprocessConfig::from_model_dir(dir.path()).unwrap(),
        PreprocessConfig::gemma3()
    );
}

#[test]
fn preprocessing_is_bit_identical_to_the_pil_processor() {
    let r = reference();
    let cfg = PreprocessConfig::gemma3();
    for (name, e) in r["images"].as_object().unwrap() {
        let png = image::open(fixtures("gemma4_vision").join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let px = preprocess(png.as_raw(), w, h, &cfg).unwrap();
        assert_eq!(
            ints(&e["slow"]["num_crops"]),
            vec![0],
            "{name}: no pan-and-scan"
        );
        let s = &e["slow"]["pixel_values"];
        assert_eq!(ints(&s["shape"]), vec![3, 896, 896], "{name}: shape");
        let got = &px.pixel_values;
        assert_eq!(got.len(), 3 * 896 * 896);
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
        for (key, f) in [
            ("sum", (|v: f64| v) as fn(f64) -> f64),
            ("abs_sum", f64::abs),
            ("sq_sum", |v: f64| v * v),
        ] {
            let have: f64 = got.iter().map(|&v| f(f64::from(v))).sum();
            let want = s[key].as_f64().unwrap();
            assert!(
                (have - want).abs() <= 1e-6 * got.len() as f64,
                "{name}: {key} {have} vs {want}"
            );
        }
        // The torchvision processor is within two 8-bit levels.
        let fs = e["fast_vs_slow"]["max_abs"].as_f64().unwrap();
        assert!(fs <= 2.0 / 255.0 * 2.0 + 1e-6, "{name}: fast vs slow {fs}");
    }
}

#[test]
fn prompt_expansion_reproduces_the_processor() {
    let r = reference();
    let t = tokens(&r);
    let seq = r["image_seq_length"].as_u64().unwrap() as usize;
    assert_eq!(seq, 256);
    assert_eq!(
        r["full_image_sequence"].as_str().unwrap(),
        cascadia_engine_openvino::gemma3_vision::full_image_sequence(&t, seq)
    );
    let mut counts = Vec::new();
    for (name, e) in r["prompts"].as_object().unwrap() {
        let template = e["template"].as_str().unwrap();
        let images = e["images"].as_array().unwrap().len();
        let expanded = if images > 0 {
            expand_prompt_text(template, &t, seq)
        } else {
            template.to_string()
        };
        assert_eq!(expanded, e["expanded"].as_str().unwrap(), "{name}: text");
        let ids = ints(&e["input_ids"]);
        check_expanded(&ids, &t, images, seq).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(
            token_type_ids(&ids, t.image),
            ints(&e["token_type_ids"]),
            "{name}: token types"
        );
        counts.push((name.clone(), images, ids.len()));
    }
    // HF's prompt token counts: one image 270, two images 535.
    let len = |n: &str| counts.iter().find(|c| c.0 == n).unwrap().2;
    assert_eq!((len("single"), len("two_images")), (270, 535));
}

/// Opt-in: the Rust tokenizer on the expanded text gives HF's ids (needs the
/// export's 33 MB tokenizer.json, not checked in).
#[test]
fn rust_tokenizer_reproduces_the_processor_ids() {
    let Ok(path) = std::env::var("CASCADIA_GEMMA3_TOKENIZER") else {
        eprintln!("CASCADIA_GEMMA3_TOKENIZER not set; skipped");
        return;
    };
    let tok = tokenizers::Tokenizer::from_file(&path).unwrap();
    let r = reference();
    for (name, e) in r["prompts"].as_object().unwrap() {
        let enc = tok.encode(e["expanded"].as_str().unwrap(), false).unwrap();
        let got: Vec<i64> = enc.get_ids().iter().map(|&u| i64::from(u)).collect();
        assert_eq!(got, ints(&e["input_ids"]), "{name}");
    }
}
