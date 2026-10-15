//! The Rust Gemma 4 vision path end to end on a real export: preprocessing
//! and the vision IR, against the `vision_ir` reference in
//! `tests/fixtures/gemma4_vision/reference.json` (HF's PIL processor driving
//! the same IR).
//!
//! Needs an OpenVINO build and a Gemma 4 export, so it is ignored by default:
//!
//! ```text
//! CASCADIA_GEMMA4_VLM_DIR=/models/gemma-4-26b-a4b-int4-ov \
//!   cargo test --release -p cascadia-engine-openvino --features openvino \
//!   --test gemma4_vision_ir -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use cascadia_engine_openvino::gemma4_vision::encoder::VisionEncoder;
use cascadia_engine_openvino::gemma4_vision::{preprocess, PreprocessConfig};
use cascadia_ov_genai_shim::PluginConfig;
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4_vision")
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
#[ignore = "needs an OpenVINO build and a Gemma 4 export (CASCADIA_GEMMA4_VLM_DIR)"]
fn vision_embeddings_match_the_hf_driven_reference() {
    let dir = std::env::var("CASCADIA_GEMMA4_VLM_DIR").expect("set CASCADIA_GEMMA4_VLM_DIR");
    let dir = Path::new(&dir);
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("reference.json")).unwrap())
            .unwrap();
    let cfg = PreprocessConfig::from_model_dir(dir).unwrap();
    let device = std::env::var("CASCADIA_GEMMA4_VISION_DEVICE").unwrap_or_else(|_| "CPU".into());
    let bound = if device == "CPU" { 1e-3 } else { 5e-2 };
    let t_load = std::time::Instant::now();
    let mut enc =
        VisionEncoder::load(dir, &device, &PluginConfig::new(), cfg.pooling_kernel_size).unwrap();
    println!(
        "{device}: compiled the vision IR in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );
    let cases = r["vision_ir"]
        .as_object()
        .expect("fixtures lack vision_ir; regenerate with --run-ir");
    for (name, want) in cases {
        let png = image::open(fixtures().join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let p = preprocess(png.as_raw(), w, h, &cfg).unwrap();
        // First call warms the kernels for this shape; time the second.
        enc.encode(&p).unwrap();
        let t_enc = std::time::Instant::now();
        let (emb, hidden) = enc.encode(&p).unwrap();
        let n = want["num_soft_tokens"].as_u64().unwrap() as usize;
        println!(
            "{device} {name}: {n} soft tokens x {hidden}, warm encode {:.0} ms",
            t_enc.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(emb.len(), n * hidden, "{name}: embedding size");
        let (abs, rel) = worst_gap(&emb, &want["image_embeds"]);
        println!("{name} image_embeds: max abs gap {abs:.3e}, relative {rel:.3e}");
        // Same IR and bit-identical pixels: what remains is kernel
        // nondeterminism across runs.
        assert!(rel < bound, "{device} {name}: relative gap {rel:.3e}");
    }
}
