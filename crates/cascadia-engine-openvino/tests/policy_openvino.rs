//! `ov-policy` on real OpenVINO, against our `input`-mode export of
//! `HuggingFaceVLA/smolvla_libero` (physical-ai-studio `8f132416`, f32 with
//! fp16 weights). Each test prints a skip reason and returns when
//! `CASCADIA_POLICY_PACKAGE` (the package directory) is unset.
#![cfg(feature = "openvino")]

use std::path::PathBuf;

use cascadia_engine::{EngineError, PolicyEngine, WARMUP_CALLS};
use cascadia_engine_openvino::policy::ov::OvTokenizer;
use cascadia_engine_openvino::policy::pipeline::{NoiseRng, PromptTokenizer};
use cascadia_engine_openvino::policy::{OvPolicyEngine, OvPolicyOptions, Precision};
use cascadia_types::{Image, NoiseMode, Observation, ResizeConvention};

fn package(test: &str) -> Option<PathBuf> {
    match std::env::var_os("CASCADIA_POLICY_PACKAGE").filter(|v| !v.is_empty()) {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            eprintln!("skip {test}: CASCADIA_POLICY_PACKAGE is unset");
            None
        }
    }
}

fn cpu() -> OvPolicyOptions {
    OvPolicyOptions {
        device: "CPU".into(),
        ..OvPolicyOptions::default()
    }
}

fn observation(engine: &OvPolicyEngine, prompt: &str) -> Observation {
    let spec = engine.spec();
    let side = (spec.height * spec.width * 3) as usize;
    Observation {
        images: spec
            .cameras
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let data = (0..side).map(|p| ((p + 37 * i) % 251) as u8).collect();
                (
                    c.clone(),
                    Image {
                        height: spec.height,
                        width: spec.width,
                        data,
                    },
                )
            })
            .collect(),
        state: vec![-0.05, 0.03, 0.76, 2.97, -0.22, -0.13, 0.027, -0.027],
        prompt: prompt.into(),
        seed: None,
        noise: Some(NoiseRng::seeded(42).normal(50 * 32)),
    }
}

const PICK_BOWL: &str = "pick up the black bowl and place it on the plate";

/// The fixture is for exactly `observation`'s inputs with [`PICK_BOWL`],
/// computed by an independent numpy + OpenVINO Python pipeline (see the
/// fixtures README).
fn reference_actions() -> Vec<f32> {
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/policy/smolvla-libero-noise-reference-actions.json"
    ))
    .unwrap();
    serde_json::from_value(reference["actions"].clone()).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn smolvla_package_spec() {
    let Some(dir) = package("smolvla_package_spec") else {
        return;
    };
    let engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec();
    assert_eq!(spec.noise, NoiseMode::Input);
    assert_eq!(spec.noise_shape, Some(vec![50, 32]));
    assert_eq!(spec.cameras, vec!["image", "image2"]);
    assert_eq!((spec.height, spec.width), (512, 512));
    assert_eq!(spec.resize, ResizeConvention::TopLeft);
    assert_eq!(spec.state_len, 8);
    assert_eq!(spec.chunk, [1, 7]);
    assert_eq!(spec.device, "CPU");
}

#[test]
fn smolvla_tokenizer_ids_for_a_known_prompt() {
    let Some(dir) = package("smolvla_tokenizer_ids_for_a_known_prompt") else {
        return;
    };
    let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
    // Our exporter's tokenizer carries the `truncation` variable.
    assert!(tok.has_truncation_state());
    // Reference: the same tokenizer.xml through openvino + openvino_tokenizers
    // 2026.4.1 in Python, on CPU. SmolVLM2 pads right with id 2.
    let t = tok
        .tokenize("pick up the black bowl and place it on the plate\n")
        .unwrap();
    let want = [
        18188, 614, 260, 2632, 12505, 284, 1379, 357, 335, 260, 6496, 198,
    ];
    assert_eq!(t.ids.len(), 48);
    assert_eq!(&t.ids[..12], &want);
    assert!(t.ids[12..].iter().all(|&id| id == 2));
    assert_eq!(t.mask.iter().filter(|&&m| m).count(), 12);
    assert!(t.mask[..12].iter().all(|&m| m) && t.mask[12..].iter().all(|&m| !m));
    // With truncation held off, an overlong prompt comes out wider than the port.
    let long = tok.tokenize(&"word ".repeat(60)).unwrap();
    assert_eq!(long.ids.len(), 61);
    assert!(long.mask.iter().all(|&m| m));
}

#[test]
fn smolvla_prompt_length_is_refused_past_the_port() {
    let Some(dir) = package("smolvla_prompt_length_is_refused_past_the_port") else {
        return;
    };
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let e = engine
        .infer(&observation(&engine, &"word ".repeat(60)))
        .unwrap_err();
    assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
    assert!(e.to_string().contains("limit is 48"), "{e}");
}

#[test]
fn smolvla_cpu_infer_with_fixed_noise_matches_the_reference() {
    let Some(dir) = package("smolvla_cpu_infer_with_fixed_noise_matches_the_reference") else {
        return;
    };
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let warm = engine.warmup().unwrap();
    assert_eq!(warm.len(), 3);
    eprintln!("warmup ms {warm:?}");
    let obs = observation(&engine, PICK_BOWL);
    let a = engine.infer(&obs).unwrap();
    assert_eq!((a.steps, a.dim, a.actions.len()), (1, 7, 7));
    eprintln!("actions {:?} in {:.1} ms", a.actions, a.infer_ms);
    let want = reference_actions();
    assert_eq!(want.len(), a.actions.len());
    let diff = max_abs_diff(&a.actions, &want);
    assert!(
        diff <= 1e-4,
        "max abs diff {diff}: {:?} vs {want:?}",
        a.actions
    );
    // CPU runs f32, so the same inputs give the same actions.
    let b = engine.infer(&obs).unwrap();
    for (x, y) in a.actions.iter().zip(&b.actions) {
        assert!((x - y).abs() <= 1e-5, "{:?} vs {:?}", a.actions, b.actions);
    }
}

/// Needs an Intel GPU, so it also waits for `CASCADIA_POLICY_GPU=1`. The
/// boolean mask ports are not reachable from here; `bind_ports` refuses a
/// non-Bool one at load, so a GPU load passing covers them.
#[test]
fn smolvla_gpu_infer_f16_and_f32() {
    let Some(dir) = package("smolvla_gpu_infer_f16_and_f32") else {
        return;
    };
    if std::env::var("CASCADIA_POLICY_GPU").as_deref() != Ok("1") {
        eprintln!("skip smolvla_gpu_infer_f16_and_f32: CASCADIA_POLICY_GPU is not 1");
        return;
    }
    let want = reference_actions();
    for (precision, label) in [(None, "f16"), (Some(Precision::F32), "f32")] {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            precision,
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        let a = engine.infer(&observation(&engine, PICK_BOWL)).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (1, 7, want.len()));
        assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
        let diff = max_abs_diff(&a.actions, &want);
        eprintln!(
            "gpu {label} actions {:?} diff={diff} infer_ms={:.1} warmup={warm:?}",
            a.actions, a.infer_ms
        );
        // f16 is loose on purpose: the f16-vs-f32 GPU default waits on LIBERO.
        let bound = if precision.is_some() { 1e-3 } else { 0.5 };
        assert!(diff <= bound, "gpu {label} max abs diff {diff} > {bound}");
    }
}
