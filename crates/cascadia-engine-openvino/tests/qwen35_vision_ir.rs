//! The Rust vision path end to end on a real export: preprocessing, host
//! inputs and the three vision IRs, against the `vision_ir` reference in
//! `tests/fixtures/qwen35_vision/reference.json` (HF's processor and
//! `vision_utils` driving the same IRs).
//!
//! Needs an OpenVINO build and a Qwen3.5/3.6 VLM export, so it is ignored by
//! default:
//!
//! ```text
//! CASCADIA_QWEN35_VLM_DIR=/models/Qwen3.6-35B-A3B-int4-ov \
//!   cargo test --release -p cascadia-engine-openvino --features openvino \
//!   --test qwen35_vision_ir -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use cascadia_engine_openvino::qwen35_vision::encoder::{VisionConfig, VisionEncoder};
use cascadia_engine_openvino::qwen35_vision::{preprocess, PreprocessConfig};
use cascadia_ov_genai_shim::PluginConfig;
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_vision")
}

/// Largest gap between the sampled reference values and ours, relative to
/// the reference's magnitude (head, tail, every 1009th value).
fn worst_gap(got: &[f32], s: &Value) -> (f64, f64) {
    let f = |k: &str| -> Vec<f64> {
        s[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let (head, tail, stride) = (f("head"), f("tail"), f("stride_1009"));
    let pairs = head
        .iter()
        .zip(got.iter())
        .chain(tail.iter().zip(got[got.len() - tail.len()..].iter()))
        .chain(stride.iter().zip(got.iter().step_by(1009)));
    let mut abs = 0f64;
    let mut scale = 0f64;
    for (&w, &g) in pairs {
        abs = abs.max((w - f64::from(g)).abs());
        scale = scale.max(w.abs());
    }
    (abs, abs / scale.max(1e-12))
}

#[test]
#[ignore = "needs an OpenVINO build and a Qwen3.5 VLM export (CASCADIA_QWEN35_VLM_DIR)"]
fn vision_embeddings_match_the_hf_driven_reference() {
    let dir = std::env::var("CASCADIA_QWEN35_VLM_DIR").expect("set CASCADIA_QWEN35_VLM_DIR");
    let dir = Path::new(&dir);
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("reference.json")).unwrap())
            .unwrap();
    let pp = PreprocessConfig::from_json_file(&dir.join("preprocessor_config.json")).unwrap();
    let vc = VisionConfig::from_config_json(&dir.join("config.json")).unwrap();
    let mut enc = VisionEncoder::load(dir, "CPU", &PluginConfig::new(), vc).unwrap();
    let cases = r["vision_ir"]
        .as_object()
        .expect("fixtures lack vision_ir; regenerate with --run-ir");
    for (name, want) in cases {
        let png = image::open(fixtures().join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let p = preprocess(&[png.as_raw()], w, h, &pp).unwrap();
        let t = enc.encode_traced(&p).unwrap();
        for (stage, got) in [
            ("patch_embeds", &t.patch_embeds),
            ("hidden_states", &t.hidden_states),
            ("image_embeds", &t.image_embeds),
        ] {
            let (abs, rel) = worst_gap(got, &want[stage]);
            println!("{name} {stage}: max abs gap {abs:.3e}, relative {rel:.3e}");
            // Same IRs and bit-identical pixels: what remains is float
            // summation order in the corner sum and rotary rounding.
            assert!(rel < 1e-3, "{name} {stage}: relative gap {rel:.3e}");
        }
    }
}
