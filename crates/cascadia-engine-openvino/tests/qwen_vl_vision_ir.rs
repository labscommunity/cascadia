//! The Rust Qwen2.5-VL / Qwen3-VL image chain end to end on a real export:
//! preprocessing, host inputs and the vision IRs, against the reference
//! `tools/vlm_fixtures/gen_qwen_vl_fixtures.py --run-ir FAMILY` writes (HF's
//! processor and `vision_utils` driving the same IRs).
//!
//! Needs an OpenVINO build and the export with its weights, so it is ignored
//! by default:
//!
//! ```text
//! python tools/vlm_fixtures/gen_qwen_vl_fixtures.py --qwen2-5-vl <export> \
//!     --out /tmp/qvl --run-ir qwen2_5_vl
//! CASCADIA_QWEN_VL_DIR=<export> CASCADIA_QWEN_VL_IR_REF=/tmp/qvl/vision_ir_qwen2_5_vl.json \
//!   cargo test --release -p cascadia-engine-openvino --features openvino \
//!   --test qwen_vl_vision_ir -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use cascadia_engine_openvino::qwen35_vision::preprocess;
use cascadia_engine_openvino::qwen_vl_vision::{Frontend, Kind};
use serde_json::Value;

fn images() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_vision")
}

/// Largest gap between the sampled reference values and ours, and that gap
/// relative to the reference's largest sampled magnitude.
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
    let (mut abs, mut scale) = (0f64, 0f64);
    for (&w, &g) in pairs {
        abs = abs.max((w - f64::from(g)).abs());
        scale = scale.max(w.abs());
    }
    (abs, abs / scale.max(1e-12))
}

#[test]
#[ignore = "needs an OpenVINO build, a Qwen VL export and its --run-ir reference"]
fn vision_rows_match_the_hf_driven_reference() {
    let dir = std::env::var("CASCADIA_QWEN_VL_DIR").expect("set CASCADIA_QWEN_VL_DIR");
    let dir = Path::new(&dir);
    let refp = std::env::var("CASCADIA_QWEN_VL_IR_REF").expect("set CASCADIA_QWEN_VL_IR_REF");
    let r: Value = serde_json::from_str(&std::fs::read_to_string(refp).unwrap()).unwrap();
    let cfg: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let kind = match cfg["model_type"].as_str() {
        Some("qwen2_5_vl") => Kind::Qwen25Vl,
        Some("qwen3_vl") => Kind::Qwen3Vl,
        other => panic!("not a Qwen VL export: {other:?}"),
    };
    let mut f = Frontend::load(dir, kind, &cfg).unwrap();
    for (name, e) in r.as_object().unwrap() {
        let png = image::open(images().join(format!("{name}.png")))
            .unwrap()
            .to_rgb8();
        let (w, h) = png.dimensions();
        let p = preprocess(&[png.as_raw()], w, h, &f.preprocess).unwrap();
        let want: Vec<u32> = e["grid_thw"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(p.grid_thw.to_vec(), want, "{name}: grid");
        let (rows, ds) = f.encode_patches(&p).unwrap();
        let (abs, rel) = worst_gap(&rows, &e["image_embeds"]);
        println!("{name}: image rows max |gap| {abs:.3e} (relative {rel:.3e})");
        assert!(
            rel < 1e-3,
            "{name}: image rows differ by {rel:.3e} relative"
        );
        if let Some(s) = e.get("deepstack") {
            let (abs, rel) = worst_gap(&ds, s);
            println!("{name}: DeepStack rows max |gap| {abs:.3e} (relative {rel:.3e})");
            assert!(
                rel < 1e-3,
                "{name}: DeepStack rows differ by {rel:.3e} relative"
            );
        }
    }
}
