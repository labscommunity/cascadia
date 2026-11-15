//! `ov-policy` on real OpenVINO, against our `input`-mode exports (f32 with
//! fp16 weights). Each test prints a skip reason and returns when its package
//! directory is unset:
//! - `CASCADIA_POLICY_PACKAGE`: `HuggingFaceVLA/smolvla_libero` (physical-ai-studio `8f132416`);
//! - `CASCADIA_POLICY_PACKAGE_PI0`: `lerobot/pi0_libero_finetuned_v044`;
//! - `CASCADIA_POLICY_PACKAGE_XVLA`: `lerobot/xvla-libero`, cameras flipped on the host.
//!
//! The π0 and X-VLA references are the exporters' PyTorch f32 outputs, in the
//! package's `ref.npz` (or `<dir>.ref.npz` beside it).
#![cfg(feature = "openvino")]

use std::path::{Path, PathBuf};

use cascadia_engine::{EngineError, PolicyEngine, WARMUP_CALLS};
use cascadia_engine_openvino::policy::ov::OvTokenizer;
use cascadia_engine_openvino::policy::pipeline::{NoiseRng, PromptTokenizer};
use cascadia_engine_openvino::policy::{
    compile_config, OvPolicyEngine, OvPolicyOptions, Precision,
};
use cascadia_ov_genai_shim::{DType, Runtime};
use cascadia_types::{Image, NoiseMode, Observation, ResizeConvention};

fn package(test: &str) -> Option<PathBuf> {
    package_from("CASCADIA_POLICY_PACKAGE", test)
}

fn package_from(var: &str, test: &str) -> Option<PathBuf> {
    match std::env::var_os(var).filter(|v| !v.is_empty()) {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            eprintln!("skip {test}: {var} is unset");
            None
        }
    }
}

/// One array of an uncompressed `.npz` (what `np.savez` writes).
struct Npy {
    descr: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Npy {
    fn f32s(&self) -> Vec<f32> {
        assert_eq!(self.descr, "<f4");
        self.data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn i64s(&self) -> Vec<i64> {
        assert_eq!(self.descr, "<i8");
        self.data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn bools(&self) -> Vec<bool> {
        assert_eq!(self.descr, "|b1");
        self.data.iter().map(|&b| b != 0).collect()
    }

    /// A numpy unicode scalar (`<U<n>`, UTF-32LE).
    fn string(&self) -> String {
        assert!(self.descr.starts_with("<U"), "{}", self.descr);
        self.data
            .chunks_exact(4)
            .map(|c| char::from_u32(u32::from_le_bytes(c.try_into().unwrap())).unwrap())
            .filter(|&c| c != '\0')
            .collect()
    }
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| i + from)
}

/// Read `name` from an `np.savez` archive: entries are stored, so each
/// local header's file name is followed by the raw `.npy` bytes.
fn npz(bytes: &[u8], name: &str) -> Npy {
    let file = format!("{name}.npy");
    let mut from = 0;
    let at = loop {
        let at = find(bytes, file.as_bytes(), from).unwrap_or_else(|| panic!("no {file}"));
        // A local header is 30 bytes before its name; this skips `ref_noise2.npy` for `noise2.npy`.
        if at >= 30 && bytes[at - 30..at - 26] == *b"PK\x03\x04" {
            break at;
        }
        from = at + 1;
    };
    let npy = find(bytes, b"\x93NUMPY", at).unwrap();
    let (len, start) = if bytes[npy + 6] == 1 {
        (
            u16::from_le_bytes([bytes[npy + 8], bytes[npy + 9]]) as usize,
            npy + 10,
        )
    } else {
        (
            u32::from_le_bytes(bytes[npy + 8..npy + 12].try_into().unwrap()) as usize,
            npy + 12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + len]).unwrap();
    assert!(header.contains("'fortran_order': False"), "{header}");
    let field = |key: &str, close: char| {
        let at = header.find(key).unwrap() + key.len();
        header[at..].split(close).next().unwrap().to_string()
    };
    let descr = field("'descr': '", '\'');
    let shape: Vec<usize> = field("'shape': (", ')')
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse().unwrap())
        .collect();
    let item = match &descr[1..] {
        "f4" => 4,
        "i8" => 8,
        "b1" => 1,
        u if u.starts_with('U') => 4 * u[1..].parse::<usize>().unwrap(),
        other => panic!("dtype {other}"),
    };
    let n = shape.iter().product::<usize>() * item;
    let data = bytes[start + len..start + len + n].to_vec();
    Npy { descr, shape, data }
}

/// The package's `ref.npz`, or `<dir>.ref.npz` beside it.
fn reference(dir: &Path) -> Vec<u8> {
    let inside = dir.join("ref.npz");
    let beside = dir.with_extension("ref.npz");
    let path = if inside.exists() { inside } else { beside };
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Run the policy IR alone on CPU, compiled with the engine's CPU config, on
/// already-processed inputs named as the IR's ports. Returns `action`.
fn ir_on_cpu(xml: &Path, feeds: &[(&str, DType, Vec<usize>, Vec<u8>)]) -> (Vec<usize>, Vec<f32>) {
    ir_on(xml, "CPU", None, feeds)
}

/// [`ir_on_cpu`] on any device, with the engine's config for `precision`.
fn ir_on(
    xml: &Path,
    device: &str,
    precision: Option<Precision>,
    feeds: &[(&str, DType, Vec<usize>, Vec<u8>)],
) -> (Vec<usize>, Vec<f32>) {
    let (device, config) = compile_config(device, precision).unwrap();
    let mut rt = Runtime::compile_inspected(xml.to_str().unwrap(), &device, None, &config).unwrap();
    for (name, dtype, shape, data) in feeds {
        rt.set_input(name, *dtype, shape, data).unwrap();
    }
    rt.infer().unwrap();
    let idx = (0..rt.output_count())
        .find(|&i| {
            rt.output_port(i)
                .unwrap()
                .aliases
                .iter()
                .any(|a| a == "action")
        })
        .unwrap();
    let (dtype, shape, bytes) = rt.output(idx).unwrap();
    assert_eq!(dtype, DType::F32);
    let values = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (shape, values)
}

/// `(name, dtype, shape, bytes)` for an IR input stored in the npz as `key`.
fn feed<'a>(z: &[u8], name: &'a str, key: &str) -> (&'a str, DType, Vec<usize>, Vec<u8>) {
    let a = npz(z, key);
    let dtype = match a.descr.as_str() {
        "<f4" => DType::F32,
        "<i8" => DType::I64,
        "|b1" => DType::Bool,
        other => panic!("{key}: {other}"),
    };
    (name, dtype, a.shape, a.data)
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

/// Ports of the π0 IR in the order `ref.npz` stores them (`keys`, `in_<i>`).
const PI0_PORTS: [&str; 6] = [
    "state",
    "noise",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "images",
    "image_masks",
];

const LIBERO_SOUP: &str = "put both the alphabet soup and the tomato sauce in the basket";

/// π0 on CPU: one load, the advertised spec, the tokenizer and one chunk.
#[test]
fn pi0_package_loads_and_serves_a_cpu_chunk() {
    let Some(dir) = package_from(
        "CASCADIA_POLICY_PACKAGE_PI0",
        "pi0_package_loads_and_serves_a_cpu_chunk",
    ) else {
        return;
    };
    let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
    assert!(tok.has_truncation_state());
    // PaliGemma pads right to 48 after BOS 2; the task plus new_line's "\n" is 14 tokens.
    let t = tok.tokenize(&format!("{LIBERO_SOUP}\n")).unwrap();
    assert_eq!(t.ids.len(), 48);
    assert_eq!(t.ids[0], 2);
    assert_eq!(t.mask.iter().filter(|&&m| m).count(), 14);
    assert!(t.mask[..14].iter().all(|&m| m) && t.mask[14..].iter().all(|&m| !m));

    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec().clone();
    assert_eq!(spec.cameras, vec!["image", "image2"]);
    assert_eq!((spec.height, spec.width), (224, 224));
    assert_eq!(spec.resize, ResizeConvention::Center);
    assert_eq!(spec.state_len, 8);
    assert_eq!(spec.chunk, [50, 7]);
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![50, 32]))
    );
    let a = engine.infer(&observation(&engine, LIBERO_SOUP)).unwrap();
    assert_eq!((a.steps, a.dim, a.actions.len()), (50, 7, 350));
    assert!(a.actions.iter().all(|x| x.is_finite()));
    eprintln!(
        "pi0 cpu infer_ms={:.0} first={:?}",
        a.infer_ms,
        &a.actions[..7]
    );
}

/// The π0 reference inputs are already processed (random float images, a
/// normalized state, synthetic token ids), so they cannot go through the
/// engine's uint8 path: they are fed to the IR as compiled for the engine's CPU.
#[test]
fn pi0_ir_on_cpu_matches_the_pytorch_reference() {
    let Some(dir) = package_from(
        "CASCADIA_POLICY_PACKAGE_PI0",
        "pi0_ir_on_cpu_matches_the_pytorch_reference",
    ) else {
        return;
    };
    let z = reference(&dir);
    let keys = npz(&z, "keys");
    let width = keys.data.len() / keys.shape[0];
    let keys: Vec<String> = keys
        .data
        .chunks(width)
        .map(|k| {
            Npy {
                descr: "<U".into(),
                shape: vec![],
                data: k.to_vec(),
            }
            .string()
        })
        .collect();
    assert_eq!(keys, PI0_PORTS);
    let feeds: Vec<_> = PI0_PORTS
        .iter()
        .enumerate()
        .map(|(i, name)| feed(&z, name, &format!("in_{i}")))
        .collect();
    let xml = dir.join("pi0.xml");
    let (shape, got) = ir_on_cpu(&xml, &feeds);
    assert_eq!(shape, vec![1, 50, 7]);
    let diff = max_abs_diff(&got, &npz(&z, "ref").f32s());
    eprintln!("pi0 cpu f32 vs pytorch f32 (traced noise): max abs {diff}");
    assert!(diff <= 2e-3, "max abs diff {diff}");

    let mut second = feeds.clone();
    second[1] = feed(&z, "noise", "noise2");
    let (_, got2) = ir_on_cpu(&xml, &second);
    let diff2 = max_abs_diff(&got2, &npz(&z, "ref_noise2").f32s());
    eprintln!("pi0 cpu f32 vs pytorch f32 (second noise): max abs {diff2}");
    assert!(diff2 <= 2e-3, "max abs diff {diff2}");
    assert!(
        max_abs_diff(&got, &got2) > 1e-2,
        "the noise input must move the output"
    );
}

/// The GPU tests also wait for `CASCADIA_POLICY_GPU=1`.
fn gpu_enabled(test: &str) -> bool {
    let on = std::env::var("CASCADIA_POLICY_GPU").as_deref() == Ok("1");
    if !on {
        eprintln!("skip {test}: CASCADIA_POLICY_GPU is not 1");
    }
    on
}

/// π0 on the GPU: f16 only. f32 weights (~12.5 GB) do not compile on a 16 GB
/// Arc 140V, and that has to come back as a load error, not a panic.
#[test]
fn pi0_gpu_infer_f16_and_f32_is_a_load_error() {
    let test = "pi0_gpu_infer_f16_and_f32_is_a_load_error";
    let Some(dir) = package_from("CASCADIA_POLICY_PACKAGE_PI0", test) else {
        return;
    };
    if !gpu_enabled(test) {
        return;
    }
    // The IR alone on the reference's processed inputs, compiled and dropped
    // before the engine's own compile: two 7.5 GB compiles do not fit at once.
    let z = reference(&dir);
    let feeds: Vec<_> = PI0_PORTS
        .iter()
        .enumerate()
        .map(|(i, name)| feed(&z, name, &format!("in_{i}")))
        .collect();
    let xml = dir.join("pi0.xml");
    let (shape, got) = ir_on(&xml, "GPU", None, &feeds);
    assert_eq!(shape, vec![1, 50, 7]);
    assert!(got.iter().all(|x| x.is_finite()));
    let diff = max_abs_diff(&got, &npz(&z, "ref").f32s());
    let mut second = feeds.clone();
    second[1] = feed(&z, "noise", "noise2");
    let (_, got2) = ir_on(&xml, "GPU", None, &second);
    assert!(got2.iter().all(|x| x.is_finite()));
    let diff = diff.max(max_abs_diff(&got2, &npz(&z, "ref_noise2").f32s()));

    let opts = |precision| OvPolicyOptions {
        device: "GPU".into(),
        precision,
        ..OvPolicyOptions::default()
    };
    {
        let mut engine = OvPolicyEngine::load(&dir, &opts(None)).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        let a = engine.infer(&observation(&engine, LIBERO_SOUP)).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (50, 7, 350));
        assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
        eprintln!(
            "gpu pi0 f16 actions {:?} diff={diff} infer_ms={:.1} warmup={warm:?}",
            &a.actions[..7],
            a.infer_ms
        );
    }
    // GPU f16 vs PyTorch f32 was 3.2e-3 at most over both noise draws.
    assert!(diff <= 1e-2, "gpu pi0 f16 max abs diff {diff} > 1e-2");

    match OvPolicyEngine::load(&dir, &opts(Some(Precision::F32))) {
        Ok(_) => panic!("gpu pi0 f32 compiled; expected it to fail on a 16 GB GPU"),
        Err(e) => {
            eprintln!("gpu pi0 f32 load error: {e}");
            assert!(matches!(e, EngineError::Backend(_)), "{e}");
            let m = e.to_string();
            assert!(m.contains("pi0.xml on GPU failed"), "{m}");
            assert!(m.contains("try without --precision f32"), "{m}");
        }
    }
}

fn xvla(test: &str) -> Option<PathBuf> {
    package_from("CASCADIA_POLICY_PACKAGE_XVLA", test)
}

#[test]
fn xvla_package_spec() {
    let Some(dir) = xvla("xvla_package_spec") else {
        return;
    };
    let engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec();
    assert_eq!(spec.cameras, vec!["image", "image2"]);
    assert_eq!((spec.height, spec.width), (224, 224));
    assert_eq!(spec.resize, ResizeConvention::TopLeft);
    assert_eq!(spec.state_len, 8);
    assert_eq!(spec.chunk, [30, 7], "served width, not the IR's 20");
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![30, 20]))
    );
}

/// The BART tokenizer IR has the `truncation` variable (on by default in the
/// IR), so the engine holds it off and refuses only past the port: an exact
/// fit of 50 tokens is served, 51 is refused.
#[test]
fn xvla_tokenizer_matches_the_reference_and_the_limit_is_the_port() {
    let Some(dir) = xvla("xvla_tokenizer_matches_the_reference_and_the_limit_is_the_port") else {
        return;
    };
    let z = reference(&dir);
    let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
    assert!(tok.has_truncation_state());
    let t = tok.tokenize(&npz(&z, "task").string()).unwrap();
    assert_eq!(t.ids, npz(&z, "in_tokenized_prompt").i64s());
    assert_eq!(t.mask, npz(&z, "in_tokenized_prompt_mask").bools());
    // <s> + one token per word + </s>.
    let words = |n: usize| vec!["word"; n].join(" ");
    let fit = tok.tokenize(&words(48)).unwrap();
    assert_eq!(
        (fit.ids.len(), fit.mask.iter().filter(|&&m| m).count()),
        (50, 50)
    );
    let over = tok.tokenize(&words(49)).unwrap();
    assert_eq!(
        over.ids.len(),
        51,
        "held-off truncation leaves it wider than the port"
    );

    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let mut obs = xvla_observation(&z);
    obs.prompt = words(48);
    engine.infer(&obs).unwrap();
    obs.prompt = words(49);
    let e = engine.infer(&obs).unwrap_err();
    assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
    let m = e.to_string();
    assert!(
        m.contains("needs 51 tokens") && m.contains("limit is 50"),
        "{m}"
    );
}

/// The reference's raw inputs, recovered: its images are uint8 frames scaled
/// to [-1, 1] with `image2` already flipped for the IR, so they are scaled
/// back and `image2` is rotated back for the engine to flip again.
fn xvla_observation(z: &[u8]) -> Observation {
    let images = npz(z, "in_images");
    assert_eq!(images.shape, vec![2, 1, 3, 224, 224]);
    let px = images.f32s();
    let n = 224 * 224;
    let frame = |slot: usize, rot180: bool| {
        let chw = &px[slot * 3 * n..(slot + 1) * 3 * n];
        let mut hwc = vec![0u8; 3 * n];
        for c in 0..3 {
            for p in 0..n {
                let v = ((chw[c * n + p] + 1.0) / 2.0 * 255.0).round() as u8;
                let at = if rot180 { n - 1 - p } else { p };
                hwc[at * 3 + c] = v;
            }
        }
        Image {
            height: 224,
            width: 224,
            data: hwc,
        }
    };
    Observation {
        images: vec![
            ("image".into(), frame(0, false)),
            ("image2".into(), frame(1, true)),
        ],
        state: npz(z, "in_state").f32s(),
        prompt: npz(z, "task").string(),
        seed: None,
        noise: Some(npz(z, "in_noise").f32s()),
    }
}

#[test]
fn xvla_cpu_infer_matches_the_pytorch_reference() {
    let Some(dir) = xvla("xvla_cpu_infer_matches_the_pytorch_reference") else {
        return;
    };
    let z = reference(&dir);
    // The IR alone on the reference's processed inputs, against LeRobot's 20-wide output.
    let feeds: Vec<_> = [
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "images",
        "image_masks",
        "noise",
    ]
    .iter()
    .map(|k| feed(&z, k, &format!("in_{k}")))
    .collect();
    let (shape, raw) = ir_on_cpu(&dir.join("xvla.xml"), &feeds);
    assert_eq!(shape, vec![1, 30, 20]);
    let raw_diff = max_abs_diff(&raw, &npz(&z, "ref20").f32s());
    eprintln!("xvla ir cpu f32 vs pytorch f32 (20-wide): max abs {raw_diff}");
    assert!(raw_diff <= 5e-4, "max abs diff {raw_diff}");

    // The engine end to end on the raw inputs, against LeRobot's served 7-wide actions.
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let a = engine.infer(&xvla_observation(&z)).unwrap();
    assert_eq!((a.steps, a.dim, a.actions.len()), (30, 7, 210));
    let want = npz(&z, "ref7").f32s();
    let diff = max_abs_diff(&a.actions, &want);
    eprintln!(
        "xvla engine cpu f32 vs pytorch f32 (served 7-wide): max abs {diff} infer_ms={:.0}",
        a.infer_ms
    );
    assert!(
        diff <= 5e-4,
        "max abs diff {diff}: {:?} vs {:?}",
        &a.actions[..7],
        &want[..7]
    );
}

/// Axis-angle to unit quaternion `(w, x, y, z)`.
fn quaternion(v: &[f32]) -> [f64; 4] {
    let v: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    let theta = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if theta < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let s = (theta / 2.0).sin() / theta;
    [(theta / 2.0).cos(), v[0] * s, v[1] * s, v[2] * s]
}

/// Angle in degrees between two axis-angle rotations. LIBERO's rotations sit
/// near π, where `v` and its antipode are the same rotation but differ by ~2π
/// per component, so rotations are compared as rotations.
fn rotation_deg(a: &[f32], b: &[f32]) -> f64 {
    let (p, q) = (quaternion(a), quaternion(b));
    let dot: f64 = p.iter().zip(&q).map(|(x, y)| x * y).sum();
    2.0 * dot.abs().min(1.0).acos().to_degrees()
}

/// X-VLA on the GPU, end to end on the reference's raw inputs, against
/// LeRobot's served 7-wide actions.
///
/// f32 is held to the CPU bar's order (spike: 2.86e-4). f16 is held per
/// channel, since a plain max on the served row would fail on two things
/// that are not errors: an axis-angle flip near π, and a gripper whose raw
/// value sits near the 0.5 threshold (the reference has one at 0.484). The
/// spike's f16 effect was position <= 6.3e-3, rotation <= 1.34 deg, raw gripper
/// <= 0.21; the bounds are about 3x that, and a gripper flip only counts where
/// the reference's raw value is more than 0.25 from the threshold.
#[test]
fn xvla_gpu_infer_f16_and_f32() {
    let test = "xvla_gpu_infer_f16_and_f32";
    let Some(dir) = xvla(test) else {
        return;
    };
    if !gpu_enabled(test) {
        return;
    }
    let z = reference(&dir);
    let want = npz(&z, "ref7").f32s();
    let raw_gripper: Vec<f32> = npz(&z, "ref20").f32s().chunks(20).map(|r| r[9]).collect();
    for (precision, label) in [(None, "f16"), (Some(Precision::F32), "f32")] {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            precision,
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        let a = engine.infer(&xvla_observation(&z)).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (30, 7, 210));
        assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
        let diff = max_abs_diff(&a.actions, &want);
        let rows = || a.actions.chunks(7).zip(want.chunks(7));
        let pos = rows()
            .map(|(g, w)| max_abs_diff(&g[..3], &w[..3]))
            .fold(0.0f32, f32::max);
        let rot = rows()
            .map(|(g, w)| rotation_deg(&g[3..6], &w[3..6]))
            .fold(0.0f64, f64::max);
        let flips: Vec<f32> = rows()
            .zip(&raw_gripper)
            .filter(|((g, w), _)| g[6] != w[6])
            .map(|(_, &r)| r)
            .collect();
        eprintln!(
            "gpu xvla {label} actions {:?} diff={diff} pos={pos} rot_deg={rot:.3} \
             gripper_flips_at_raw={flips:?} infer_ms={:.1} warmup={warm:?}",
            &a.actions[..7],
            a.infer_ms
        );
        if precision.is_some() {
            assert!(diff <= 1e-3, "gpu xvla f32 max abs diff {diff} > 1e-3");
        } else {
            assert!(pos <= 0.02, "gpu xvla f16 position diff {pos} > 0.02");
            assert!(rot <= 4.0, "gpu xvla f16 rotation diff {rot} deg > 4");
            assert!(
                flips.iter().all(|r| (r - 0.5).abs() <= 0.25),
                "gpu xvla f16 gripper flipped away from the threshold: raw {flips:?}"
            );
        }
    }
}

/// A compile that fails (here: a device the host lacks) is a load error, not a panic.
#[cfg(target_os = "macos")]
#[test]
fn a_failed_compile_is_a_load_error() {
    let Some(dir) = xvla("a_failed_compile_is_a_load_error") else {
        return;
    };
    let opts = OvPolicyOptions {
        device: "GPU".into(),
        precision: Some(Precision::F32),
        ..OvPolicyOptions::default()
    };
    let e = OvPolicyEngine::load(&dir, &opts)
        .err()
        .expect("macOS has no OpenVINO GPU");
    assert!(matches!(e, EngineError::Backend(_)), "{e}");
    let m = e.to_string();
    eprintln!("{m}");
    assert!(m.contains("xvla.xml on GPU failed"), "{m}");
}
