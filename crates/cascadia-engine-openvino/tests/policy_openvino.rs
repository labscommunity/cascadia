//! `ov-policy` on real OpenVINO, against our `input`-mode exports (f32 with
//! fp16 weights). Each test prints a skip reason and returns when its package
//! directory is unset:
//! - `CASCADIA_POLICY_PACKAGE`: `HuggingFaceVLA/smolvla_libero` (physical-ai-studio `8f132416`);
//! - `CASCADIA_POLICY_PACKAGE_PI0`: `lerobot/pi0_libero_finetuned_v044`;
//! - `CASCADIA_POLICY_PACKAGE_XVLA`: `lerobot/xvla-libero`, cameras flipped on the host;
//! - `CASCADIA_POLICY_PACKAGE_VLAJEPA`: `lerobot/VLA-JEPA-LIBERO`, prompt padded to 288;
//! - `CASCADIA_POLICY_PACKAGE_EVO1`: `zuoxingdong/evo1_libero`, cameras flipped on the host;
//! - `CASCADIA_POLICY_PACKAGE_XR0`: `XiaomiRobotics/Xiaomi-Robotics-0-LIBERO` with Xiaomi's
//!   inference action mask, a 10-action chunk, fp16 weights;
//! - `CASCADIA_POLICY_PACKAGE_XR1`: `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` (Intel's XR0
//!   exporter, patched), fp16 weights, `manifest.json` replaced by the
//!   `xr1-robocasa365-manifest.json` fixture;
//! - `CASCADIA_POLICY_PACKAGE_GROOT`: `nvidia/GR00T-N1.7-LIBERO` (libero_spatial), Intel's
//!   edge-ai-suites single-IR export, fp16 weights;
//! - `CASCADIA_POLICY_PACKAGE_MOLMOACT2`: `allenai/MolmoAct2-LIBERO`, Intel's exporter, fp16 weights;
//!   `CASCADIA_POLICY_PACKAGE_MOLMOACT2_DROID` and `CASCADIA_POLICY_PACKAGE_MOLMOACT2_SO100`:
//!   the DROID and SO-100/101 checkpoints, exported the same way;
//! - `CASCADIA_POLICY_PACKAGE_PI05_DROID`: `lerobot/pi05_droid`, `--stats` openpi's DROID
//!   `norm_stats.json`, `--empty-camera right_wrist_0_rgb`.
//!
//! The π0 and X-VLA references are the exporters' PyTorch f32 outputs, in the
//! package's `ref.npz` (or `<dir>.ref.npz` beside it). VLA-JEPA's are LeRobot's
//! f32 inputs and actions, `torch_ref.npz`, for the raw `inputs.npz` (each in
//! the package, or `<dir>.<name>` beside it). EVO1's are LeRobot's served
//! actions in `ref.npz`, for the raw LIBERO frames in `frames.npz` (same rule).
//! XR0's are Intel's runtime stages in `xr0-stages-f32.npz` (the exporter's
//! `fixtures-f32.npz`, saved uncompressed) and PyTorch f32 of Intel's port with
//! Xiaomi's action mask in `<dir>.ref.npz` (same rule). XR1's are Xiaomi's own processor's
//! graph inputs and model (PyTorch f32) in `xr1.ref.npz`, with the client's
//! cropped frames added by `tools/xr1_processor_fixture.py crop` (same rule). GR00T's are
//! NVIDIA's `Gr00tPolicy` (PyTorch f32) graph inputs, raw output and decoded actions in
//! `ref.npz`, for
//! the frames and states in `gr00t-frames.npz` (regenerated from the
//! exporter's seeds; same rule). Each MolmoAct2 package's are Intel's runtime stages
//! in `molmoact2-stages-f32.npz` (the exporter's `fixtures-f32.npz`, saved
//! uncompressed) and the PyTorch f32 pipeline on a real frame set in
//! `molmoact2-e2e.npz` (same rule). π0.5 DROID's are LeRobot's PI05Policy (PyTorch f32, openpi's
//! stats) served actions in `ref.npz`, with its raw frames, states, task and noise (same rule).
//! The RTC packages are listed above their tests.
#![cfg(feature = "openvino")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cascadia_engine::{EngineError, EngineResult, PolicyEngine, WARMUP_CALLS};
use cascadia_engine_openvino::policy::manifest::{MolmoAct2Step, PostStep, PreStep};
use cascadia_engine_openvino::policy::ov::OvTokenizer;
use cascadia_engine_openvino::policy::pipeline::{
    HostTensor, NoiseRng, PolicyModel, PromptTokenizer, TensorData,
};
use cascadia_engine_openvino::policy::{
    compile_config, LoadOptions, OvPolicyEngine, OvPolicyOptions, PolicyManifest, PolicyPipeline,
    Precision, ServerFacts,
};
use cascadia_ov_genai_shim::{DType, PortInfo, Runtime};
use cascadia_types::{Image, NoiseMode, Observation, ResizeConvention, RtcInputs};

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

    fn f64s(&self) -> Vec<f64> {
        assert_eq!(self.descr, "<f8");
        self.data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
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
        "i8" | "f8" => 8,
        "b1" | "u1" => 1,
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
        rtc: None,
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

#[test]
fn smolvla_model_id_replaces_the_directory_name() {
    let Some(dir) = package("smolvla_model_id_replaces_the_directory_name") else {
        return;
    };
    let dir_name = dir.file_name().unwrap().to_string_lossy().into_owned();
    let engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    assert_eq!(engine.spec().model, dir_name);
    let opts = OvPolicyOptions {
        model_id: Some("smolvla-libero.v1".into()),
        ..cpu()
    };
    let engine = OvPolicyEngine::load(&dir, &opts).unwrap();
    assert_eq!(engine.spec().model, "smolvla-libero.v1");
    let json = serde_json::to_value(engine.spec()).unwrap();
    assert_eq!(json["model"], "smolvla-libero.v1");
    for bad in ["", "a/b", "a b", "-x"] {
        let opts = OvPolicyOptions {
            model_id: Some(bad.into()),
            ..cpu()
        };
        let e = OvPolicyEngine::load(&dir, &opts).err().expect(bad);
        assert!(matches!(e, EngineError::InvalidConfig(_)), "{bad:?}: {e}");
        assert!(e.to_string().contains("model id"), "{e}");
    }
}

/// A second load from the same `cache_dir` compiles from the cache and must
/// give the same bits as an uncached load.
#[test]
fn smolvla_cpu_cache_dir_gives_bit_identical_actions() {
    let Some(dir) = package("smolvla_cpu_cache_dir_gives_bit_identical_actions") else {
        return;
    };
    let cache = tempfile::tempdir().unwrap();
    let cached = OvPolicyOptions {
        cache_dir: Some(cache.path().to_path_buf()),
        ..cpu()
    };
    let run = |opts: &OvPolicyOptions| {
        let started = std::time::Instant::now();
        let mut engine = OvPolicyEngine::load(&dir, opts).unwrap();
        let load_ms = started.elapsed().as_millis();
        let obs = observation(&engine, PICK_BOWL);
        let bits: Vec<u32> = engine
            .infer(&obs)
            .unwrap()
            .actions
            .iter()
            .map(|x| x.to_bits())
            .collect();
        (bits, load_ms)
    };
    let (plain, plain_ms) = run(&cpu());
    let (cold, cold_ms) = run(&cached);
    let blobs = std::fs::read_dir(cache.path()).unwrap().count();
    assert!(
        blobs > 0,
        "the first cached load wrote nothing to the cache"
    );
    let (warm, warm_ms) = run(&cached);
    eprintln!("load ms: uncached {plain_ms}, cold cache {cold_ms}, warm cache {warm_ms}; {blobs} cache files");
    assert_eq!(cold, plain, "a cold cache changed the actions");
    assert_eq!(warm, plain, "a load from the cache changed the actions");
}

#[test]
fn smolvla_gpu_memory_bytes_is_none_on_cpu() {
    let Some(dir) = package("smolvla_gpu_memory_bytes_is_none_on_cpu") else {
        return;
    };
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    engine.warmup().unwrap();
    assert_eq!(engine.gpu_memory_bytes(), None);
}

/// The measured GPU claim, next to the package's weight file. Needs an Intel
/// GPU and `CASCADIA_POLICY_GPU=1`.
#[test]
fn smolvla_gpu_memory_bytes_is_measured_on_gpu() {
    let Some(dir) = package("smolvla_gpu_memory_bytes_is_measured_on_gpu") else {
        return;
    };
    if !gpu_enabled("smolvla_gpu_memory_bytes_is_measured_on_gpu") {
        return;
    }
    let bin: u64 = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin") && !p.ends_with("tokenizer.bin"))
        .map(|p| std::fs::metadata(p).unwrap().len())
        .sum();
    for (precision, label) in [(None, "f16"), (Some(Precision::F32), "f32")] {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            precision,
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let loaded = engine
            .gpu_memory_bytes()
            .expect("a GPU load reports its memory");
        engine.warmup().unwrap();
        let warm = engine
            .gpu_memory_bytes()
            .expect("a GPU load reports its memory");
        eprintln!(
            "gpu {label} memory: {loaded} bytes after load, {warm} after warmup; policy weights {bin} bytes"
        );
        assert!(
            warm >= bin / 2,
            "{warm} bytes is less than half the {bin}-byte weights"
        );
        engine.close();
        assert_eq!(engine.gpu_memory_bytes(), None);
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
/// Arc 140V, and that has to come back as a load error, not a panic. That
/// check runs in a child process: after the failed compile, every later GPU
/// compile in the same process fails too (`CL_INVALID_EVENT`).
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

    let opts = OvPolicyOptions {
        device: "GPU".into(),
        ..OvPolicyOptions::default()
    };
    {
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
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

    let child = "pi0_gpu_f32_load_is_an_error_in_its_own_process";
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", child, "--ignored", "--nocapture"])
        .output()
        .unwrap();
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{child}: {}\n{stdout}",
        out.status
    );
}

#[test]
#[ignore = "run by pi0_gpu_infer_f16_and_f32_is_a_load_error, in its own process"]
fn pi0_gpu_f32_load_is_an_error_in_its_own_process() {
    let dir = PathBuf::from(std::env::var_os("CASCADIA_POLICY_PACKAGE_PI0").unwrap());
    let opts = OvPolicyOptions {
        device: "GPU".into(),
        precision: Some(Precision::F32),
        ..OvPolicyOptions::default()
    };
    match OvPolicyEngine::load(&dir, &opts) {
        Ok(_) => panic!("gpu pi0 f32 compiled; expected it to fail on a 16 GB GPU"),
        Err(e) => {
            eprintln!("gpu pi0 f32 load error: {e}");
            assert!(matches!(e, EngineError::Backend(_)), "{e}");
            let m = e.to_string();
            assert!(m.contains("pi0.xml on GPU failed"), "{m}");
            assert!(
                m.contains("try without --ov-inference-precision f32"),
                "{m}"
            );
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
        rtc: None,
    }
}

/// The served 7-wide rows against the reference's, per channel group.
struct ServedDiff {
    /// Plain max abs over the row, for the log only.
    max: f32,
    position: f32,
    rotation_deg: f64,
    /// The reference's raw (pre-threshold) gripper at each flipped row.
    gripper_flips: Vec<f32>,
}

/// LIBERO's rotations sit near π, where an axis-angle and its antipode are the
/// same rotation but differ by up to ~2π per component, so a tiny error can
/// move the plain max by several units: rotations are compared as rotations.
fn served_diff(got: &[f32], z: &[u8]) -> ServedDiff {
    let want = npz(z, "ref7").f32s();
    let raw_gripper: Vec<f32> = npz(z, "ref20").f32s().chunks(20).map(|r| r[9]).collect();
    let rows = || got.chunks(7).zip(want.chunks(7));
    ServedDiff {
        max: max_abs_diff(got, &want),
        position: rows()
            .map(|(g, w)| max_abs_diff(&g[..3], &w[..3]))
            .fold(0.0f32, f32::max),
        rotation_deg: rows()
            .map(|(g, w)| rotation_deg(&g[3..6], &w[3..6]))
            .fold(0.0f64, f64::max),
        gripper_flips: rows()
            .zip(&raw_gripper)
            .filter(|((g, w), _)| g[6] != w[6])
            .map(|(_, &r)| r)
            .collect(),
    }
}

impl ServedDiff {
    /// Position and rotation (in radians) within `bound`, no gripper flip.
    fn assert_within(&self, label: &str, bound: f32) {
        assert!(
            self.position <= bound,
            "{label} position diff {} > {bound}",
            self.position
        );
        let rad = self.rotation_deg.to_radians();
        assert!(
            rad <= bound as f64,
            "{label} rotation diff {rad} rad > {bound}"
        );
        assert!(
            self.gripper_flips.is_empty(),
            "{label} gripper flipped at raw {:?}",
            self.gripper_flips
        );
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
    let d = served_diff(&a.actions, &z);
    eprintln!(
        "xvla engine cpu f32 vs pytorch f32 (served 7-wide): max abs {} position {} \
         rotation {:.4} deg gripper flips {:?} infer_ms={:.0}",
        d.max, d.position, d.rotation_deg, d.gripper_flips, a.infer_ms
    );
    d.assert_within("xvla cpu f32", 5e-4);
}

/// Records what the pipeline binds and answers with the reference's raw output.
struct Recorder {
    inputs: Arc<Mutex<Vec<(String, HostTensor)>>>,
    output: Vec<f32>,
}

impl PolicyModel for Recorder {
    fn set_input(&mut self, name: &str, t: &HostTensor) -> EngineResult<()> {
        self.inputs.lock().unwrap().push((name.into(), t.clone()));
        Ok(())
    }
    fn infer(&mut self) -> EngineResult<()> {
        Ok(())
    }
    fn output_f32(&mut self, _: &str) -> EngineResult<(Vec<usize>, Vec<f32>)> {
        Ok((vec![1, 30, 20], self.output.clone()))
    }
}

/// The engine's preprocessing binds every port exactly as the reference
/// prepared it, and its postprocessing turns the reference's raw output into
/// the reference's served actions. Needs no device and no model run, so a
/// package whose manifest disagrees with its reference (a missing
/// `image_flip`) fails here, naming the port, on any host.
#[test]
fn xvla_engine_binds_the_reference_inputs() {
    let Some(dir) = xvla("xvla_engine_binds_the_reference_inputs") else {
        return;
    };
    let z = reference(&dir);
    let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
    let (device, config) = compile_config("CPU", None).unwrap();
    let rt = Runtime::compile_inspected(
        dir.join(&manifest.model_artifact).to_str().unwrap(),
        &device,
        None,
        &config,
    )
    .unwrap();
    let inputs: Vec<PortInfo> = (0..rt.input_count())
        .map(|i| rt.input_port(i).unwrap())
        .collect();
    let outputs: Vec<PortInfo> = (0..rt.output_count())
        .map(|i| rt.output_port(i).unwrap())
        .collect();
    drop(rt);
    let tokenizer = OvTokenizer::load(&dir.join(manifest.tokenizer_artifact())).unwrap();
    let bound = Arc::new(Mutex::new(Vec::new()));
    let model = Recorder {
        inputs: bound.clone(),
        output: npz(&z, "ref20").f32s(),
    };
    let mut pipeline = PolicyPipeline::new(
        manifest,
        ServerFacts {
            model: "xvla".into(),
            device,
            prompt_max_bytes: 1024,
        },
        Box::new(model),
        &inputs,
        &outputs,
        false,
        Box::new(tokenizer),
    )
    .unwrap();
    let a = pipeline.infer(&xvla_observation(&z)).unwrap();

    let bound = bound.lock().unwrap();
    for (name, got) in bound.iter() {
        let want = npz(&z, &format!("in_{name}"));
        assert_eq!(got.shape, want.shape, "{name} shape");
        match &got.data {
            TensorData::F32(v) if name == "images" => {
                let w = want.f32s();
                let n = 3 * 224 * 224;
                for (slot, (g, w)) in v.chunks(n).zip(w.chunks(n)).enumerate() {
                    let d = max_abs_diff(g, w);
                    // A slot that matches only when turned around is an `image_flip` disagreement.
                    let per_channel = |x: &[f32]| -> Vec<f32> {
                        x.chunks(224 * 224)
                            .flat_map(|c| c.iter().rev().copied())
                            .collect()
                    };
                    let turned = max_abs_diff(&per_channel(g), w);
                    eprintln!("xvla bound images slot {slot}: max abs {d}, rotated 180 {turned}");
                    assert!(
                        d <= 1e-6,
                        "images slot {slot} differs from the reference by {d} \
                         (by {turned} when rotated 180 degrees)"
                    );
                }
            }
            TensorData::F32(v) => assert_eq!(*v, want.f32s(), "{name}"),
            TensorData::I64(v) => assert_eq!(*v, want.i64s(), "{name}"),
            TensorData::Bool(v) => assert_eq!(*v, want.bools(), "{name}"),
        }
    }
    let mut names: Vec<&str> = bound.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "image_masks",
            "images",
            "noise",
            "state",
            "tokenized_prompt",
            "tokenized_prompt_mask"
        ]
    );

    let d = served_diff(&a.actions, &z);
    eprintln!(
        "xvla postprocess vs pytorch (served 7-wide): position {} rotation {:.6} deg gripper flips {:?}",
        d.position, d.rotation_deg, d.gripper_flips
    );
    d.assert_within("xvla postprocess", 1e-4);
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
/// LeRobot's served 7-wide actions ([`served_diff`]).
///
/// f32 is held to 1e-3 (spike: 2.86e-4). The spike's f16 effect was position
/// <= 6.3e-3, rotation <= 1.34 deg and raw gripper <= 0.21, so f16 is held to
/// about 3x that, and a gripper flip counts only where the reference's raw
/// value is more than 0.25 from the 0.5 threshold (one of its rows is at 0.484).
/// The reference call comes before warmup so a warmup failure still shows it.
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
    let obs = xvla_observation(&z);
    for (precision, label) in [(None, "f16"), (Some(Precision::F32), "f32")] {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            precision,
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let a = engine.infer(&obs).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (30, 7, 210));
        let warm = engine.warmup();
        let infer_ms = engine.infer(&obs).map(|b| b.infer_ms);
        let d = served_diff(&a.actions, &z);
        eprintln!(
            "gpu xvla {label} actions {:?} diff={} position={} rotation_deg={:.3} \
             gripper_flips_at_raw={:?} infer_ms={infer_ms:.1?} warmup={warm:?}",
            &a.actions[..7],
            d.max,
            d.position,
            d.rotation_deg,
            d.gripper_flips
        );
        if precision.is_some() {
            d.assert_within("gpu xvla f32", 1e-3);
        } else {
            assert!(
                d.position <= 0.02,
                "gpu xvla f16 position diff {} > 0.02",
                d.position
            );
            assert!(
                d.rotation_deg <= 4.0,
                "gpu xvla f16 rotation diff {} deg > 4",
                d.rotation_deg
            );
            assert!(
                d.gripper_flips.iter().all(|r| (r - 0.5).abs() <= 0.25),
                "gpu xvla f16 gripper flipped away from the threshold: raw {:?}",
                d.gripper_flips
            );
        }
        let warm = warm.unwrap_or_else(|e| panic!("gpu xvla {label} warmup: {e}"));
        assert_eq!(warm.len(), WARMUP_CALLS);
        infer_ms.unwrap();
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

fn vla_jepa(test: &str) -> Option<PathBuf> {
    package_from("CASCADIA_POLICY_PACKAGE_VLAJEPA", test)
}

/// `name` in the package, or `<dir>.<name>` beside it.
fn beside(dir: &Path, name: &str) -> Vec<u8> {
    let inside = dir.join(name);
    let path = if inside.exists() {
        inside
    } else {
        dir.with_extension(name)
    };
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The six reference cases: three LIBERO tasks, two seeded noise draws each, one frame pair.
const VLA_JEPA_CASES: usize = 6;

/// Qwen's `<|endoftext|>`, the tokenizer IR's right padding.
const QWEN_PAD: i64 = 151643;

struct JepaRef {
    inputs: Vec<u8>,
    torch: Vec<u8>,
}

impl JepaRef {
    fn load(dir: &Path) -> Self {
        Self {
            inputs: beside(dir, "inputs.npz"),
            torch: beside(dir, "torch_ref.npz"),
        }
    }

    fn observation(&self, case: usize) -> Observation {
        let frame = |key: &str| {
            let a = npz(&self.inputs, &format!("frame/images.{key}"));
            assert_eq!(
                (a.descr.as_str(), a.shape.as_slice()),
                ("|u1", &[224, 224, 3][..])
            );
            Image {
                height: 224,
                width: 224,
                data: a.data,
            }
        };
        Observation {
            images: vec![
                ("image".into(), frame("image")),
                ("image2".into(), frame("image2")),
            ],
            state: npz(&self.inputs, &format!("{case}/state")).f32s(),
            prompt: npz(&self.inputs, &format!("{case}/task")).string(),
            seed: None,
            noise: Some(npz(&self.inputs, &format!("{case}/noise")).f32s()),
            rtc: None,
        }
    }

    fn ids(&self, case: usize) -> Vec<i64> {
        npz(&self.torch, &format!("{case}/input_ids")).i64s()
    }

    fn actions(&self, case: usize) -> Vec<f32> {
        npz(&self.torch, &format!("{case}/actions")).f32s()
    }
}

fn template(m: &PolicyManifest) -> String {
    m.preprocessors
        .iter()
        .find_map(|s| match s {
            PreStep::VlaJepaPrompt { template } => Some(template.clone()),
            _ => None,
        })
        .unwrap()
}

/// Max abs diff and the rows whose gripper (dim 6, ±1) disagrees.
fn jepa_diff(got: &[f32], want: &[f32]) -> (f32, usize) {
    let flips = got
        .chunks(7)
        .zip(want.chunks(7))
        .filter(|(g, w)| g[6].signum() != w[6].signum())
        .count();
    (max_abs_diff(got, want), flips)
}

/// The IR's `truncation` variable defaults on; the engine holds it off, so
/// the count is exact: 288 tokens are served and 289 refused.
#[test]
fn vlajepa_spec_tokenizer_and_cpu_parity() {
    let Some(dir) = vla_jepa("vlajepa_spec_tokenizer_and_cpu_parity") else {
        return;
    };
    let r = JepaRef::load(&dir);
    let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
    let tpl = template(&manifest);
    let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
    assert!(tok.has_truncation_state());
    for case in 0..VLA_JEPA_CASES {
        let want = r.ids(case);
        let task = r.observation(case).prompt;
        let t = tok.tokenize(&tpl.replace("{instruction}", &task)).unwrap();
        let n = want.len();
        assert_eq!(t.ids.len(), 288);
        assert_eq!(&t.ids[..n], &want[..], "case {case} ids");
        assert!(
            t.ids[n..].iter().all(|&id| id == QWEN_PAD),
            "case {case} padding"
        );
        assert!(t.mask[..n].iter().all(|&m| m) && t.mask[n..].iter().all(|&m| !m));
    }
    // " word" is one token: find the count that fills the port exactly.
    let words = |k: usize| vec!["word"; k].join(" ");
    let count = |tok: &mut OvTokenizer, k: usize| {
        let t = tok
            .tokenize(&tpl.replace("{instruction}", &words(k)))
            .unwrap();
        (t.ids.len(), t.mask.iter().filter(|&&m| m).count())
    };
    let fit = (1..200).find(|&k| count(&mut tok, k).1 == 288).unwrap();
    assert_eq!(
        count(&mut tok, fit + 1),
        (289, 289),
        "held-off truncation leaves it wider"
    );

    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec().clone();
    assert_eq!(spec.cameras, vec!["image", "image2"]);
    assert_eq!((spec.height, spec.width), (224, 224));
    assert_eq!(spec.resize, ResizeConvention::Stretch);
    assert_eq!(spec.state_len, 8);
    assert_eq!(spec.chunk, [7, 7]);
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![7, 7]))
    );
    let mut worst = 0f32;
    for case in 0..VLA_JEPA_CASES {
        let a = engine.infer(&r.observation(case)).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (7, 7, 49));
        let (d, flips) = jepa_diff(&a.actions, &r.actions(case));
        eprintln!(
            "vlajepa engine cpu f32 vs pytorch f32 case {case}: max abs {d} gripper flips {flips} infer_ms={:.0}",
            a.infer_ms
        );
        assert_eq!(flips, 0, "case {case}");
        worst = worst.max(d);
    }
    eprintln!(
        "vlajepa engine cpu f32 vs pytorch f32: worst max abs {worst} over {VLA_JEPA_CASES} cases"
    );
    assert!(worst <= 2e-4, "max abs diff {worst}");

    let mut obs = r.observation(0);
    obs.prompt = words(fit);
    engine.infer(&obs).unwrap();
    obs.prompt = words(fit + 1);
    let e = engine.infer(&obs).unwrap_err();
    assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
    assert!(
        e.to_string().contains("needs 289 tokens; the limit is 288"),
        "{e}"
    );
}

/// Records what the pipeline binds and answers with `output` in `shape`.
struct JepaRecorder {
    inputs: Arc<Mutex<Vec<(String, HostTensor)>>>,
    output: Arc<Mutex<Vec<f32>>>,
    shape: Vec<usize>,
}

impl PolicyModel for JepaRecorder {
    fn set_input(&mut self, name: &str, t: &HostTensor) -> EngineResult<()> {
        self.inputs.lock().unwrap().push((name.into(), t.clone()));
        Ok(())
    }
    fn infer(&mut self) -> EngineResult<()> {
        Ok(())
    }
    fn output_f32(&mut self, _: &str) -> EngineResult<(Vec<usize>, Vec<f32>)> {
        Ok((self.shape.clone(), self.output.lock().unwrap().clone()))
    }
}

/// The engine's host steps bind every port as LeRobot prepared it: ids and
/// embodied positions exact, pixel values within 1e-6, the state and noise
/// as sent; the graph's actions are served untouched. No model run.
#[test]
fn vlajepa_engine_binds_the_reference_inputs() {
    let Some(dir) = vla_jepa("vlajepa_engine_binds_the_reference_inputs") else {
        return;
    };
    let r = JepaRef::load(&dir);
    let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
    let (device, config) = compile_config("CPU", None).unwrap();
    let rt = Runtime::compile_inspected(
        dir.join(&manifest.model_artifact).to_str().unwrap(),
        &device,
        None,
        &config,
    )
    .unwrap();
    let inputs: Vec<PortInfo> = (0..rt.input_count())
        .map(|i| rt.input_port(i).unwrap())
        .collect();
    let outputs: Vec<PortInfo> = (0..rt.output_count())
        .map(|i| rt.output_port(i).unwrap())
        .collect();
    drop(rt);
    let tokenizer = OvTokenizer::load(&dir.join(manifest.tokenizer_artifact())).unwrap();
    let specials = tokenizer.special_tokens();
    let bound = Arc::new(Mutex::new(Vec::new()));
    let output = Arc::new(Mutex::new(vec![0.0; 49]));
    let mut pipeline = PolicyPipeline::new(
        manifest,
        ServerFacts {
            model: "vlajepa".into(),
            device,
            prompt_max_bytes: 1024,
        },
        Box::new(JepaRecorder {
            inputs: bound.clone(),
            output: output.clone(),
            shape: vec![1, 7, 7],
        }),
        &inputs,
        &outputs,
        false,
        Box::new(tokenizer),
    )
    .unwrap();
    for case in 0..VLA_JEPA_CASES {
        let obs = r.observation(case);
        *output.lock().unwrap() = r.actions(case);
        bound.lock().unwrap().clear();
        let a = pipeline.infer(&obs).unwrap();
        assert_eq!(
            a.actions,
            r.actions(case),
            "case {case}: actions served untouched"
        );
        let bound = bound.lock().unwrap();
        let get = |name: &str| {
            bound
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.clone())
                .unwrap_or_else(|| panic!("{name} not bound"))
        };
        let ids = r.ids(case);
        let n = ids.len();
        let TensorData::I64(got) = get("input_ids").data else {
            panic!("input_ids is i64")
        };
        assert_eq!(&got[..n], &ids[..], "case {case} input_ids");
        assert!(got[n..].iter().all(|&id| id == QWEN_PAD));
        let TensorData::I64(mask) = get("attention_mask").data else {
            panic!("attention_mask is i64")
        };
        let want_mask = npz(&r.torch, &format!("{case}/attention_mask")).i64s();
        assert_eq!(&mask[..n], &want_mask[..]);
        assert!(mask[n..].iter().all(|&m| m == 0));
        assert_eq!(
            get("embodied_positions").data,
            TensorData::I64(npz(&r.torch, &format!("{case}/embodied_positions")).i64s()),
            "case {case} embodied_positions"
        );
        let px = get("pixel_values");
        assert_eq!(px.shape, vec![512, 1536]);
        let TensorData::F32(px) = px.data else {
            panic!("pixel_values is f32")
        };
        let d = max_abs_diff(&px, &npz(&r.torch, &format!("{case}/pixel_values")).f32s());
        eprintln!("vlajepa bound pixel_values case {case}: max abs {d}");
        assert!(d <= 1e-6, "case {case} pixel_values differ by {d}");
        assert_eq!(
            get("state").data,
            TensorData::F32(obs.state.clone()),
            "raw state"
        );
        assert_eq!(
            get("noise").data,
            TensorData::F32(obs.noise.clone().unwrap())
        );
        let mut names: Vec<&str> = bound.iter().map(|(n, _)| n.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "attention_mask",
                "embodied_positions",
                "input_ids",
                "noise",
                "pixel_values",
                "state"
            ]
        );
    }
    assert_special_tasks_refused(
        &mut pipeline,
        &specials,
        &[
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_pad|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|embodied_action|>",
            "<|action_0|>",
            "<|endoftext|>",
        ],
        &r.observation(0),
    );
}

/// GPU f16 (the default) within 2e-3 of PyTorch f32 with no gripper flip
/// (spike: 8.2e-4); GPU f32 within 1e-3 (spike: 1.65e-4).
#[test]
fn vlajepa_gpu_infer_f16_and_f32() {
    let test = "vlajepa_gpu_infer_f16_and_f32";
    let Some(dir) = vla_jepa(test) else {
        return;
    };
    if !gpu_enabled(test) {
        return;
    }
    let r = JepaRef::load(&dir);
    for (precision, label, bound) in [(None, "f16", 2e-3f32), (Some(Precision::F32), "f32", 1e-3)] {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            precision,
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        let (mut worst, mut flips, mut ms) = (0f32, 0, 0.0);
        for case in 0..VLA_JEPA_CASES {
            let a = engine.infer(&r.observation(case)).unwrap();
            assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
            let (d, f) = jepa_diff(&a.actions, &r.actions(case));
            worst = worst.max(d);
            flips += f;
            ms = a.infer_ms;
        }
        eprintln!(
            "gpu vlajepa {label}: worst max abs {worst} gripper flips {flips} over \
             {VLA_JEPA_CASES} cases infer_ms={ms:.1} warmup={warm:?}"
        );
        assert_eq!(flips, 0, "gpu vlajepa {label} gripper flips");
        assert!(
            worst <= bound,
            "gpu vlajepa {label} max abs diff {worst} > {bound}"
        );
    }
}

fn evo1(test: &str) -> Option<PathBuf> {
    package_from("CASCADIA_POLICY_PACKAGE_EVO1", test)
}

/// EVO1's binarized gripper column.
const EVO1_GRIPPER: usize = 6;

/// The two real LIBERO cases of `ref.npz` and their raw frames, as a client sends them.
fn evo1_cases(dir: &Path) -> Vec<(Observation, Vec<f32>)> {
    let z = beside(dir, "ref.npz");
    let fr = beside(dir, "frames.npz");
    let task = npz(&fr, "task").string();
    let states = npz(&fr, "state").f32s();
    let n = 448 * 448 * 3;
    ["real0", "real1"]
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let frame = |cam: &str| {
                let a = npz(&fr, &format!("env_{cam}"));
                assert_eq!(a.shape, vec![2, 448, 448, 3]);
                Image {
                    height: 448,
                    width: 448,
                    data: a.data[i * n..(i + 1) * n].to_vec(),
                }
            };
            let obs = Observation {
                images: vec![
                    ("image".into(), frame("image")),
                    ("image2".into(), frame("image2")),
                ],
                state: states[i * 8..(i + 1) * 8].to_vec(),
                prompt: task.clone(),
                seed: None,
                noise: Some(npz(&z, &format!("{case}_in_noise")).f32s()),
                rtc: None,
            };
            // LeRobot's 50 postprocessed rows; the engine serves the first 14.
            let want = npz(&z, &format!("{case}_final")).f32s()[..14 * 7].to_vec();
            (obs, want)
        })
        .collect()
}

/// Max abs over the non-gripper columns, and the rows whose gripper differs.
fn evo1_diff(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut max = 0f32;
    let mut flips = 0;
    for (g, w) in got.chunks(7).zip(want.chunks(7)) {
        for c in (0..7).filter(|&c| c != EVO1_GRIPPER) {
            max = max.max((g[c] - w[c]).abs());
        }
        flips += usize::from(g[EVO1_GRIPPER] != w[EVO1_GRIPPER]);
    }
    (max, flips)
}

#[test]
fn evo1_spec_tokenizer_and_cpu_parity() {
    let Some(dir) = evo1("evo1_spec_tokenizer_and_cpu_parity") else {
        return;
    };
    let z = beside(&dir, "ref.npz");
    let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
    assert!(tok.has_truncation_state());
    let task = npz(&z, "real0_task").string();
    let t = tok.tokenize(task.trim()).unwrap();
    assert_eq!(t.ids, npz(&z, "real0_in_tokenized_prompt").i64s());
    assert_eq!(t.mask, npz(&z, "real0_in_tokenized_prompt_mask").bools());

    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec().clone();
    assert_eq!(spec.cameras, vec!["image", "image2"]);
    assert_eq!((spec.height, spec.width), (448, 448));
    assert_eq!(spec.resize, ResizeConvention::Stretch);
    assert_eq!(spec.state_len, 8);
    assert_eq!(spec.chunk, [14, 7]);
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![50, 7]))
    );
    for (i, (mut obs, want)) in evo1_cases(&dir).into_iter().enumerate() {
        // LeRobot strips the task; so does the engine.
        obs.prompt = format!("  {}\n", obs.prompt);
        let a = engine.infer(&obs).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (14, 7, 98));
        let (d, flips) = evo1_diff(&a.actions, &want);
        eprintln!(
            "evo1 engine cpu f32 vs lerobot served case real{i}: max abs excl gripper {d} \
             gripper flips {flips} infer_ms={:.0}",
            a.infer_ms
        );
        assert_eq!(flips, 0, "real{i} gripper flips");
        assert!(d <= 2e-3, "real{i} max abs diff {d}");
    }
}

/// f16 is the GPU default only because the exporter splits the 44,800-input
/// pooling FC: unsplit, every flow step after the first is wrong at f16.
#[test]
fn evo1_gpu_infer_f16_and_f32() {
    let test = "evo1_gpu_infer_f16_and_f32";
    let Some(dir) = evo1(test) else {
        return;
    };
    if !gpu_enabled(test) {
        return;
    }
    let cases = evo1_cases(&dir);
    for (precision, label) in [(None, "f16"), (Some(Precision::F32), "f32")] {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            precision,
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        for (i, (obs, want)) in cases.iter().enumerate() {
            let a = engine.infer(obs).unwrap();
            assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
            let (d, flips) = evo1_diff(&a.actions, want);
            eprintln!(
                "gpu evo1 {label} case real{i}: max abs excl gripper {d} gripper flips {flips} \
                 infer_ms={:.1} warmup={warm:?}",
                a.infer_ms
            );
            assert_eq!(flips, 0, "gpu evo1 {label} real{i} gripper flips");
            assert!(d <= 2e-3, "gpu evo1 {label} real{i} max abs diff {d}");
        }
    }
}

fn xr0(test: &str) -> Option<PathBuf> {
    package_from("CASCADIA_POLICY_PACKAGE_XR0", test)
}

/// A float `[1, 3, H, W]` frame in [0, 1] as the uint8 HWC a client sends:
/// Intel's `_to_rgb_frame`, `round(clip(x, 0, 1) * 255)`.
fn xr0_frame(a: &Npy) -> Image {
    let (h, w) = (a.shape[2], a.shape[3]);
    let px = a.f32s();
    let mut hwc = vec![0u8; 3 * h * w];
    for c in 0..3 {
        for p in 0..h * w {
            hwc[p * 3 + c] = (px[c * h * w + p].clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
        }
    }
    Image {
        height: h as u32,
        width: w as u32,
        data: hwc,
    }
}

fn xr0_observation(z: &[u8], prefix: &str, task: String, noise: Vec<f32>) -> Observation {
    Observation {
        images: ["base", "wrist_left"]
            .iter()
            .map(|c| {
                (
                    c.to_string(),
                    xr0_frame(&npz(z, &format!("{prefix}images.{c}"))),
                )
            })
            .collect(),
        state: npz(z, &format!("{prefix}state")).f32s(),
        prompt: task,
        seed: None,
        noise: Some(noise),
        rtc: None,
    }
}

/// The ports of our XR0 export, read off `xr0.xml`.
fn xr0_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
    let port = |name: &str, dtype, dims: &[usize]| PortInfo {
        aliases: vec![name.into()],
        dtype: Some(dtype),
        dims: dims.iter().map(|&d| Some(d)).collect(),
    };
    (
        vec![
            port("tokenized_prompt", DType::I64, &[1, 256]),
            port("tokenized_prompt_mask", DType::I64, &[1, 256]),
            port("pixel_values", DType::F32, &[512, 1536]),
            port("state", DType::F32, &[1, 1, 32]),
            port("noise", DType::F32, &[1, 10, 32]),
        ],
        vec![port("action", DType::F32, &[1, 10, 32])],
    )
}

/// Every stage of Intel's runtime, bit for bit: the prompt text, ids and mask,
/// pixel values, the padded state and the noise; the denormalized actions
/// within 1e-6. No model run, so it needs no device.
#[test]
fn xr0_engine_binds_intels_stages() {
    let Some(dir) = xr0("xr0_engine_binds_intels_stages") else {
        return;
    };
    let z = beside(&dir, "xr0-stages-f32.npz");
    let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
    let task = npz(&z, "00_obs/task").string();
    let want_prompt = npz(&z, "01_XR0Preprocessor/task").string();
    assert_eq!(
        cascadia_engine_openvino::policy::xr0::render_prompt(
            &["base".into(), "wrist_left".into()],
            64,
            &task
        ),
        want_prompt
    );
    let tokenizer = OvTokenizer::load(&dir.join(manifest.tokenizer_artifact())).unwrap();
    let specials = tokenizer.special_tokens();
    let bound = Arc::new(Mutex::new(Vec::new()));
    let (inputs, outputs) = xr0_ports();
    let mut pipeline = PolicyPipeline::new(
        manifest,
        ServerFacts {
            model: "xr0".into(),
            device: "CPU".into(),
            prompt_max_bytes: 1024,
        },
        Box::new(JepaRecorder {
            inputs: bound.clone(),
            output: Arc::new(Mutex::new(npz(&z, "90_runner/action").f32s())),
            shape: vec![1, 10, 32],
        }),
        &inputs,
        &outputs,
        false,
        Box::new(tokenizer),
    )
    .unwrap();
    let noise = npz(&z, "01_XR0Preprocessor/noise").f32s();
    let obs = xr0_observation(&z, "00_obs/", task, noise.clone());
    let a = pipeline.infer(&obs).unwrap();
    let recorded = bound.lock().unwrap().clone();
    assert_special_tasks_refused(
        &mut pipeline,
        &specials,
        &[
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_pad|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<cot>",
            "</cot>",
        ],
        &obs,
    );
    let bound = recorded;
    let get = |name: &str| {
        bound
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.data.clone())
            .unwrap_or_else(|| panic!("{name} not bound"))
    };
    assert_eq!(
        get("tokenized_prompt"),
        TensorData::I64(npz(&z, "03_OVTokenizer/tokenized_prompt").i64s())
    );
    let mask: Vec<i64> = npz(&z, "03_OVTokenizer/tokenized_prompt_mask")
        .bools()
        .into_iter()
        .map(i64::from)
        .collect();
    eprintln!(
        "xr0 prompt: {} tokens of 256",
        mask.iter().filter(|&&m| m == 1).count()
    );
    assert_eq!(get("tokenized_prompt_mask"), TensorData::I64(mask));
    assert_eq!(
        get("pixel_values"),
        TensorData::F32(npz(&z, "01_XR0Preprocessor/pixel_values").f32s()),
        "pixel_values bit for bit"
    );
    assert_eq!(
        get("state"),
        TensorData::F32(npz(&z, "01_XR0Preprocessor/state").f32s())
    );
    assert_eq!(get("noise"), TensorData::F32(noise));
    assert_eq!((a.steps, a.dim), (10, 7));
    let d = max_abs_diff(&a.actions, &npz(&z, "91_XR0Postprocessor/action").f32s());
    eprintln!("xr0 denormalize vs intel: max abs {d}");
    assert!(d <= 1e-6, "xr0_denormalize differs by {d}");
}

/// The PyTorch f32 reference, `<dir>.ref.npz` in the package or beside it.
fn xr0_reference(dir: &Path) -> Vec<u8> {
    let name = dir.file_name().unwrap().to_string_lossy();
    let inside = dir.join(format!("{name}.ref.npz"));
    let path = if inside.exists() {
        inside
    } else {
        dir.with_extension("ref.npz")
    };
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// PyTorch f32's second draw, denormalized with the package's stats.
fn xr0_second_reference(dir: &Path, z: &[u8]) -> Vec<f32> {
    let manifest = PolicyManifest::load(dir, LoadOptions::default()).unwrap();
    let Some(PostStep::Xr0Denormalize(s)) = manifest.postprocessors.first() else {
        panic!("xr0_denormalize")
    };
    cascadia_engine_openvino::policy::xr0::denormalize(s, &npz(z, "ref2").f32s(), 32)
}

/// CPU f32 with fp16 weights against PyTorch f32 on both noise draws (measured:
/// 5.5e-4 / 4.5e-4 normalized, 2.1e-4 / 1.7e-4 served), and the prompt limit: the tokenizer
/// IR's `truncation` variable is held off, so 256 tokens are served and 257 refused.
#[test]
fn xr0_spec_prompt_limit_and_cpu_parity() {
    let Some(dir) = xr0("xr0_spec_prompt_limit_and_cpu_parity") else {
        return;
    };
    let z = xr0_reference(&dir);
    let task = npz(&z, "task").string();
    let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
    assert!(tok.has_truncation_state());
    let prompt = |t: &str| {
        cascadia_engine_openvino::policy::xr0::render_prompt(
            &["base".into(), "wrist_left".into()],
            64,
            t,
        )
    };
    let words = |k: usize| vec!["word"; k].join(" ");
    let mut count = |k: usize| {
        let t = tok.tokenize(&prompt(&words(k))).unwrap();
        (t.ids.len(), t.mask.iter().filter(|&&m| m).count())
    };
    let fit = (1..200).find(|&k| count(k).1 == 256).unwrap();
    assert_eq!(
        count(fit + 1),
        (257, 257),
        "held-off truncation leaves it wider"
    );

    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec().clone();
    assert_eq!(spec.cameras, vec!["base", "wrist_left"]);
    assert_eq!((spec.height, spec.width), (256, 256));
    assert_eq!(spec.resize, ResizeConvention::Stretch);
    assert_eq!((spec.state_len, spec.chunk), (8, [10, 7]));
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![10, 32]))
    );
    let second = xr0_second_reference(&dir, &z);
    for (noise, want, label) in [
        ("noise1", npz(&z, "act1").f32s(), "draw 1"),
        ("noise2", second, "draw 2"),
    ] {
        let obs = xr0_observation(&z, "obs_", task.clone(), npz(&z, noise).f32s());
        let a = engine.infer(&obs).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (10, 7, 70));
        let d = max_abs_diff(&a.actions, &want);
        eprintln!(
            "xr0 engine cpu f32 vs pytorch f32 ({label}): max abs {d} infer_ms={:.0}",
            a.infer_ms
        );
        assert!(d <= 1e-3, "{label}: max abs diff {d}");
    }
    let mut obs = xr0_observation(&z, "obs_", words(fit + 1), npz(&z, "noise1").f32s());
    let e = engine.infer(&obs).unwrap_err();
    assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
    assert!(
        e.to_string().contains("needs 257 tokens; the limit is 256"),
        "{e}"
    );
    obs.prompt = words(fit);
    engine.infer(&obs).unwrap();
}

/// XR0 on the GPU: f16 within 1e-2 of PyTorch f32, measured on the Arc 140V
/// (1.2e-3 / 9.4e-4 served).
/// At `--ov-inference-precision f32` the GPU compile fails (`CL_INVALID_EVENT`), which must
/// be a load error; that runs in a child process, as π0's does.
#[test]
fn xr0_gpu_infer_f16_and_f32_is_a_load_error() {
    let test = "xr0_gpu_infer_f16_and_f32_is_a_load_error";
    let Some(dir) = xr0(test) else {
        return;
    };
    if !gpu_enabled(test) {
        return;
    }
    let z = xr0_reference(&dir);
    let task = npz(&z, "task").string();
    let second = xr0_second_reference(&dir, &z);
    {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        for (noise, want, label) in [
            ("noise1", npz(&z, "act1").f32s(), "draw 1"),
            ("noise2", second, "draw 2"),
        ] {
            let obs = xr0_observation(&z, "obs_", task.clone(), npz(&z, noise).f32s());
            let a = engine.infer(&obs).unwrap();
            assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
            let d = max_abs_diff(&a.actions, &want);
            eprintln!(
                "gpu xr0 f16 ({label}): max abs {d} infer_ms={:.1} warmup={warm:?}",
                a.infer_ms
            );
            assert!(d <= 1e-2, "gpu xr0 f16 {label} max abs diff {d} > 1e-2");
        }
    }
    let child = "xr0_gpu_f32_load_is_an_error_in_its_own_process";
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", child, "--ignored", "--nocapture"])
        .output()
        .unwrap();
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{child}: {}\n{stdout}",
        out.status
    );
}

#[test]
#[ignore = "run by xr0_gpu_infer_f16_and_f32_is_a_load_error, in its own process"]
fn xr0_gpu_f32_load_is_an_error_in_its_own_process() {
    let dir = PathBuf::from(std::env::var_os("CASCADIA_POLICY_PACKAGE_XR0").unwrap());
    let opts = OvPolicyOptions {
        device: "GPU".into(),
        precision: Some(Precision::F32),
        ..OvPolicyOptions::default()
    };
    match OvPolicyEngine::load(&dir, &opts) {
        Ok(_) => panic!("gpu xr0 f32 compiled; expected it to fail on the Arc 140V"),
        Err(e) => {
            eprintln!("gpu xr0 f32 load error: {e}");
            assert!(matches!(e, EngineError::Backend(_)), "{e}");
            let m = e.to_string();
            assert!(m.contains("xr0.xml on GPU failed"), "{m}");
            assert!(
                m.contains("try without --ov-inference-precision f32"),
                "{m}"
            );
        }
    }
}

fn xr1(test: &str) -> Option<PathBuf> {
    package_from("CASCADIA_POLICY_PACKAGE_XR1", test)
}

/// The cameras XR1 serves, and the RoboCasa365 keys their frames have in `xr1.ref.npz`.
const XR1_CAMERAS: [(&str, &str); 3] = [
    ("agentview_left", "robot0_agentview_left"),
    ("agentview_right", "robot0_agentview_right"),
    ("eye_in_hand", "robot0_eye_in_hand"),
];

/// The request Xiaomi's client sends: each camera's four cropped frames, oldest first.
fn xr1_observation(z: &[u8], task: String, noise: Vec<f32>) -> Observation {
    let mut images = Vec::new();
    for (camera, key) in XR1_CAMERAS {
        let a = npz(z, &format!("crop_{key}"));
        assert_eq!(a.shape, [4, 256, 256, 3], "{key}");
        for f in a.data.chunks_exact(256 * 256 * 3) {
            let img = Image {
                height: 256,
                width: 256,
                data: f.to_vec(),
            };
            images.push((camera.to_string(), img));
        }
    }
    Observation {
        images,
        state: npz(z, "state14").f32s(),
        prompt: task,
        seed: None,
        noise: Some(noise),
        rtc: None,
    }
}

/// The ports of our XR1 export.
fn xr1_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
    let port = |name: &str, dtype, dims: &[usize]| PortInfo {
        aliases: vec![name.into()],
        dtype: Some(dtype),
        dims: dims.iter().map(|&d| Some(d)).collect(),
    };
    (
        vec![
            port("tokenized_prompt", DType::I64, &[1, 512]),
            port("tokenized_prompt_mask", DType::I64, &[1, 512]),
            port("pixel_values", DType::F32, &[1536, 1536]),
            port("state", DType::F32, &[1, 4, 60]),
            port("noise", DType::F32, &[1, 16, 60]),
        ],
        vec![port("action", DType::F32, &[1, 16, 60])],
    )
}

/// Xiaomi's processor, bit for bit: prompt ids and mask (video pads as image
/// pads), pixel values and the bf16 state; `xr0_denormalize` within 1e-5 of
/// Xiaomi's decode (it adds no eps). And the prompt limit: 512 tokens served,
/// 513 refused. No model run, so it needs no device.
#[test]
fn xr1_engine_binds_xiaomis_processor() {
    let Some(dir) = xr1("xr1_engine_binds_xiaomis_processor") else {
        return;
    };
    let z = beside(&dir, "xr1.ref.npz");
    let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
    let task = npz(&z, "task").string();
    let mut tok = OvTokenizer::load(&dir.join(manifest.tokenizer_artifact())).unwrap();
    assert!(tok.has_truncation_state());
    let specials = tok.special_tokens();
    let prompt = |t: &str| {
        cascadia_engine_openvino::policy::xr0::render_xr1_prompt(
            &[
                "Left camera".into(),
                "Right camera".into(),
                "Wrist camera".into(),
            ],
            2,
            2,
            64,
            t,
        )
    };
    assert_eq!(
        prompt(&task).replace("<|image_pad|>", "<|video_pad|>"),
        npz(&z, "prompt").string()
    );
    let words = |k: usize| vec!["word"; k].join(" ");
    let mut count = |k: usize| {
        let t = tok.tokenize(&prompt(&words(k))).unwrap();
        (t.ids.len(), t.mask.iter().filter(|&&m| m).count())
    };
    let fit = (1..100).find(|&k| count(k).1 == 512).unwrap();
    assert_eq!(count(fit + 1), (513, 513));

    let bound = Arc::new(Mutex::new(Vec::new()));
    let (inputs, outputs) = xr1_ports();
    let mut pipeline = PolicyPipeline::new(
        manifest,
        ServerFacts {
            model: "xr1".into(),
            device: "CPU".into(),
            prompt_max_bytes: 1024,
        },
        Box::new(JepaRecorder {
            inputs: bound.clone(),
            output: Arc::new(Mutex::new(npz(&z, "ref1").f32s())),
            shape: vec![1, 16, 60],
        }),
        &inputs,
        &outputs,
        false,
        Box::new(OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap()),
    )
    .unwrap();
    let spec = pipeline.spec().clone();
    assert_eq!(
        spec.cameras,
        XR1_CAMERAS.map(|(c, _)| c.to_string()).to_vec()
    );
    assert_eq!(
        (spec.history, spec.state_len, spec.chunk),
        (4, 14, [16, 12])
    );
    let noise = npz(&z, "noise1").f32s();
    let obs = xr1_observation(&z, task, noise.clone());
    let a = pipeline.infer(&obs).unwrap();
    let recorded = bound.lock().unwrap().clone();
    assert_special_tasks_refused(
        &mut pipeline,
        &specials,
        &[
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_pad|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<cot>",
            "</cot>",
        ],
        &obs,
    );
    let get = |name: &str| {
        recorded
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.data.clone())
            .unwrap_or_else(|| panic!("{name} not bound"))
    };
    let mask = npz(&z, "in_tokenized_prompt_mask").i64s();
    eprintln!(
        "xr1 prompt: {} tokens of 512",
        mask.iter().filter(|&&m| m == 1).count()
    );
    assert_eq!(
        get("tokenized_prompt"),
        TensorData::I64(npz(&z, "in_tokenized_prompt").i64s())
    );
    assert_eq!(get("tokenized_prompt_mask"), TensorData::I64(mask));
    assert_eq!(
        get("pixel_values"),
        TensorData::F32(npz(&z, "in_pixel_values").f32s()),
        "pixel_values bit for bit"
    );
    assert_eq!(get("state"), TensorData::F32(npz(&z, "in_state").f32s()));
    assert_eq!(get("noise"), TensorData::F32(noise));
    assert_eq!((a.steps, a.dim), (16, 12));
    let d = max_abs_diff(&a.actions, &npz(&z, "act1").f32s());
    eprintln!("xr1 denormalize vs xiaomi: max abs {d}");
    assert!(d <= 1e-5, "xr0_denormalize differs by {d}");
    let mut long = obs.clone();
    long.prompt = words(fit + 1);
    let e = pipeline.infer(&long).unwrap_err();
    assert!(
        e.to_string().contains("needs 513 tokens; the limit is 512"),
        "{e}"
    );
    long.prompt = words(fit);
    pipeline.infer(&long).unwrap();
}

/// The first `keep` values of each `width`-wide row.
fn rows_cut(v: &[f32], width: usize, keep: usize) -> Vec<f32> {
    v.chunks(width).flat_map(|r| r[..keep].to_vec()).collect()
}

/// CPU f32 with fp16 weights against Xiaomi's model in f32 on both noise
/// draws, on the 12 served dims (spike: 2.8e-4 / 2.8e-4).
#[test]
fn xr1_spec_and_cpu_parity() {
    let Some(dir) = xr1("xr1_spec_and_cpu_parity") else {
        return;
    };
    let z = beside(&dir, "xr1.ref.npz");
    let task = npz(&z, "task").string();
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec().clone();
    assert_eq!((spec.height, spec.width, spec.history), (256, 256, 4));
    assert_eq!(spec.resize, ResizeConvention::Stretch);
    assert_eq!(spec.server_resize, None);
    assert_eq!((spec.state_len, spec.chunk), (14, [16, 12]));
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![16, 60]))
    );
    for (noise, reference, label) in [("noise1", "ref1", "draw 1"), ("noise2", "ref2", "draw 2")] {
        let want = rows_cut(&npz(&z, reference).f32s(), 60, 12);
        let obs = xr1_observation(&z, task.clone(), npz(&z, noise).f32s());
        let a = engine.infer(&obs).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (16, 12, 192));
        let d = max_abs_diff(&a.actions, &want);
        eprintln!(
            "xr1 engine cpu f32 vs xiaomi f32 ({label}): max abs {d} infer_ms={:.0}",
            a.infer_ms
        );
        assert!(d <= 1e-3, "{label}: max abs diff {d}");
    }
}

/// XR1 on the GPU: f16 within 1e-2 of Xiaomi's model in f32, measured on the Arc
/// 140V at 3.4e-3 (about 660 ms a 16-step chunk).
#[test]
fn xr1_gpu_infer_f16() {
    let test = "xr1_gpu_infer_f16";
    let Some(dir) = xr1(test) else {
        return;
    };
    if !gpu_enabled(test) {
        return;
    }
    let z = beside(&dir, "xr1.ref.npz");
    let task = npz(&z, "task").string();
    let opts = OvPolicyOptions {
        device: "GPU".into(),
        ..OvPolicyOptions::default()
    };
    let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
    let warm = engine.warmup().unwrap();
    assert_eq!(warm.len(), WARMUP_CALLS);
    for (noise, reference, label) in [("noise1", "ref1", "draw 1"), ("noise2", "ref2", "draw 2")] {
        let want = rows_cut(&npz(&z, reference).f32s(), 60, 12);
        let obs = xr1_observation(&z, task.clone(), npz(&z, noise).f32s());
        let a = engine.infer(&obs).unwrap();
        assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
        let d = max_abs_diff(&a.actions, &want);
        eprintln!(
            "gpu xr1 f16 ({label}): max abs {d} infer_ms={:.1} warmup={warm:?}",
            a.infer_ms
        );
        assert!(d <= 1e-2, "gpu xr1 f16 {label} max abs diff {d} > 1e-2");
    }
}

/// Every special token of the package's tokenizer, and the named structure
/// strings, refused in a task before it is templated; `base` stays served.
fn assert_special_tasks_refused(
    pipeline: &mut PolicyPipeline,
    specials: &[String],
    named: &[&str],
    base: &Observation,
) {
    for n in named {
        assert!(
            specials.iter().any(|s| s == n) || n.starts_with("<cot") || n.starts_with("</cot"),
            "{n} not read from the tokenizer"
        );
    }
    let all: Vec<&str> = specials
        .iter()
        .map(String::as_str)
        .chain(named.iter().copied())
        .collect();
    for sp in &all {
        let mut obs = base.clone();
        obs.prompt = format!("{} {sp} now", base.prompt);
        let e = pipeline.infer(&obs).unwrap_err();
        assert!(
            matches!(e, EngineError::InvalidConfig(_))
                && e.to_string().contains("the task contains"),
            "{sp}: {e}"
        );
    }
    eprintln!(
        "refused {} special-token tasks ({} from the tokenizer)",
        all.len(),
        specials.len()
    );
    pipeline.infer(base).unwrap();
}

fn gr00t(test: &str) -> Option<PathBuf> {
    package_from("CASCADIA_POLICY_PACKAGE_GROOT", test)
}

/// `ref.npz`'s cases: frames and state seeds 0, 0, 1 with three noise draws.
const GR00T_CASES: [(&str, &str); 3] = [("A", "A"), ("B_same_obs_new_noise", "A"), ("C", "C")];

/// The reference fed NVIDIA's processor the frames as stored; the package's
/// `image_flip` rotates both cameras, so the client sends them rotated.
fn gr00t_observation(frames: &[u8], z: &[u8], case: &str, obs: &str) -> Observation {
    let frame = |cam: &str| {
        let a = npz(frames, &format!("{obs}_{cam}"));
        assert_eq!(a.shape, [256, 256, 3]);
        Image {
            height: 256,
            width: 256,
            data: a.data.chunks_exact(3).rev().flatten().copied().collect(),
        }
    };
    Observation {
        images: vec![
            ("image".into(), frame("image")),
            ("wrist_image".into(), frame("wrist_image")),
        ],
        state: npz(frames, &format!("{obs}_state")).f32s(),
        prompt: npz(frames, "task").string(),
        seed: None,
        noise: Some(npz(z, &format!("{case}__actions")).f32s()),
        rtc: None,
    }
}

/// Our GR00T export's ports, read off `gr00t.xml`.
fn gr00t_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
    let port = |name: &str, dtype, dims: &[usize]| PortInfo {
        aliases: vec![name.into()],
        dtype: Some(dtype),
        dims: dims.iter().map(|&d| Some(d)).collect(),
    };
    (
        vec![
            port("input_ids", DType::I64, &[1, 192]),
            port("attention_mask", DType::I64, &[1, 192]),
            port("pixel_values", DType::F32, &[512, 1536]),
            port("position_ids", DType::I64, &[3, 1, 192]),
            port("state", DType::F32, &[1, 1, 132]),
            port("embodiment_id", DType::I64, &[1]),
            port("actions", DType::F32, &[1, 40, 132]),
        ],
        vec![port("predicted_action", DType::F32, &[1, 40, 132])],
    )
}

fn f64_diff(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max)
}

/// Every graph input NVIDIA's processor made, from the frames and state a
/// client sends: ids, mask and mRoPE positions exact, pixel values and the
/// padded state within 1e-6, the embodiment id and noise as given; NVIDIA's
/// raw output through the postprocessors within 1e-6 of its decoded actions.
/// No model run, so it needs no device.
#[test]
fn gr00t_engine_binds_the_reference_inputs() {
    let Some(dir) = gr00t("gr00t_engine_binds_the_reference_inputs") else {
        return;
    };
    let z = reference(&dir);
    let frames = beside(&dir, "gr00t-frames.npz");
    let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
    let tokenizer = OvTokenizer::load(&dir.join(manifest.tokenizer_artifact())).unwrap();
    let specials = tokenizer.special_tokens();
    let bound = Arc::new(Mutex::new(Vec::new()));
    let output = Arc::new(Mutex::new(Vec::new()));
    let (inputs, outputs) = gr00t_ports();
    let mut pipeline = PolicyPipeline::new(
        manifest,
        ServerFacts {
            model: "gr00t".into(),
            device: "CPU".into(),
            prompt_max_bytes: 1024,
        },
        Box::new(JepaRecorder {
            inputs: bound.clone(),
            output: output.clone(),
            shape: vec![1, 40, 132],
        }),
        &inputs,
        &outputs,
        false,
        Box::new(tokenizer),
    )
    .unwrap();
    for (case, obs) in GR00T_CASES {
        bound.lock().unwrap().clear();
        *output.lock().unwrap() = npz(&z, &format!("{case}__pred")).f32s();
        let o = gr00t_observation(&frames, &z, case, obs);
        let a = pipeline.infer(&o).unwrap();
        let got = bound.lock().unwrap().clone();
        let get = |name: &str| {
            got.iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.data.clone())
                .unwrap_or_else(|| panic!("{name} not bound"))
        };
        for name in [
            "input_ids",
            "attention_mask",
            "position_ids",
            "embodiment_id",
        ] {
            assert_eq!(
                get(name),
                TensorData::I64(npz(&z, &format!("{case}__{name}")).i64s()),
                "{case} {name}"
            );
        }
        for name in ["pixel_values", "state"] {
            let TensorData::F32(v) = get(name) else {
                panic!("{name} is not f32")
            };
            let d = max_abs_diff(&v, &npz(&z, &format!("{case}__{name}")).f32s());
            eprintln!("gr00t {case} {name} vs NVIDIA's processor: max abs {d:e}");
            assert!(d <= 1e-6, "{case} {name} differs by {d}");
        }
        assert_eq!(get("actions"), TensorData::F32(o.noise.clone().unwrap()));
        assert_eq!((a.steps, a.dim), (16, 7));
        let d = f64_diff(&a.actions, &npz(&z, &format!("{case}__decoded")).f64s());
        eprintln!("gr00t {case} postprocessors vs decode_action: max abs {d:e}");
        assert!(d <= 1e-6, "{case} decode differs by {d}");
    }
    let base = gr00t_observation(&frames, &z, "A", "A");
    assert_special_tasks_refused(
        &mut pipeline,
        &specials,
        &[
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ],
        &base,
    );
}

/// CPU f32 with fp16 weights against NVIDIA's PyTorch f32 on all three
/// cases, as served (decoded, [16, 7]); the spike measured 3.3e-3. The
/// prompt limit: 64 language tokens are served, 65 refused.
#[test]
fn gr00t_spec_prompt_limit_and_cpu_parity() {
    let Some(dir) = gr00t("gr00t_spec_prompt_limit_and_cpu_parity") else {
        return;
    };
    let z = reference(&dir);
    let frames = beside(&dir, "gr00t-frames.npz");
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec().clone();
    assert_eq!(spec.cameras, ["image", "wrist_image"]);
    assert_eq!((spec.height, spec.width), (256, 256));
    assert_eq!(spec.resize, ResizeConvention::Center);
    assert_eq!((spec.state_len, spec.chunk), (8, [16, 7]));
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![40, 132]))
    );
    let notice = spec.attribution.clone().unwrap();
    assert!(
        notice.contains("NVIDIA Open Model License") && notice.contains("Built on NVIDIA Cosmos"),
        "{notice}"
    );
    for (case, obs) in GR00T_CASES {
        let a = engine
            .infer(&gr00t_observation(&frames, &z, case, obs))
            .unwrap();
        let d = f64_diff(&a.actions, &npz(&z, &format!("{case}__decoded")).f64s());
        eprintln!(
            "gr00t engine cpu f32 vs pytorch f32 ({case}): max abs {d:.3e} infer_ms={:.0}",
            a.infer_ms
        );
        assert!(d <= 8e-3, "{case}: max abs diff {d}");
    }
    // Language tokens of a k-word task: the ids less the two image tokens.
    let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
    let step = manifest
        .preprocessors
        .iter()
        .find_map(|s| match s {
            PreStep::Gr00tPrompt(p) => Some(p.clone()),
            _ => None,
        })
        .unwrap();
    let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
    let words = |k: usize| vec!["word"; k].join(" ");
    let mut count = |k: usize| {
        let text = cascadia_engine_openvino::policy::gr00t::render_prompt(&step, 2, &words(k));
        tok.tokenize(&text).unwrap().ids.len() - 2
    };
    let fit = (1..80).find(|&k| count(k) == 64).unwrap();
    assert_eq!(count(fit + 1), 65);
    let mut o = gr00t_observation(&frames, &z, "A", "A");
    o.prompt = words(fit + 1);
    let e = engine.infer(&o).unwrap_err();
    assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
    assert!(
        e.to_string()
            .contains("needs 65 language tokens; the limit is 64"),
        "{e}"
    );
    o.prompt = words(fit);
    engine.infer(&o).unwrap();
}

/// GR00T on the GPU: f16 within 1e-2 of PyTorch f32 as served (spike: 7.4e-3
/// normalized). At `--ov-inference-precision f32` the GPU compile fails
/// (`CL_INVALID_EVENT`), which must be a load error; that runs in a child
/// process, as π0's does.
#[test]
fn gr00t_gpu_infer_f16_and_f32_is_a_load_error() {
    let test = "gr00t_gpu_infer_f16_and_f32_is_a_load_error";
    let Some(dir) = gr00t(test) else {
        return;
    };
    if !gpu_enabled(test) {
        return;
    }
    let z = reference(&dir);
    let frames = beside(&dir, "gr00t-frames.npz");
    {
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(&dir, &opts).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        for (case, obs) in GR00T_CASES {
            let a = engine
                .infer(&gr00t_observation(&frames, &z, case, obs))
                .unwrap();
            assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
            let d = f64_diff(&a.actions, &npz(&z, &format!("{case}__decoded")).f64s());
            eprintln!(
                "gpu gr00t f16 ({case}): max abs {d:.3e} infer_ms={:.1} warmup={warm:?}",
                a.infer_ms
            );
            assert!(d <= 1e-2, "gpu gr00t f16 {case} max abs diff {d} > 1e-2");
        }
    }
    let child = "gr00t_gpu_f32_load_is_an_error_in_its_own_process";
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", child, "--ignored", "--nocapture"])
        .output()
        .unwrap();
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{child}: {}\n{stdout}",
        out.status
    );
}

#[test]
#[ignore = "run by gr00t_gpu_infer_f16_and_f32_is_a_load_error, in its own process"]
fn gr00t_gpu_f32_load_is_an_error_in_its_own_process() {
    let dir = PathBuf::from(std::env::var_os("CASCADIA_POLICY_PACKAGE_GROOT").unwrap());
    let opts = OvPolicyOptions {
        device: "GPU".into(),
        precision: Some(Precision::F32),
        ..OvPolicyOptions::default()
    };
    match OvPolicyEngine::load(&dir, &opts) {
        Ok(_) => panic!("gpu gr00t f32 compiled; expected it to fail on the Arc 140V"),
        Err(e) => {
            eprintln!("gpu gr00t f32 load error: {e}");
            assert!(matches!(e, EngineError::Backend(_)), "{e}");
            let m = e.to_string();
            assert!(m.contains("gr00t.xml on GPU failed"), "{m}");
            assert!(
                m.contains("try without --ov-inference-precision f32"),
                "{m}"
            );
        }
    }
}

/// A MolmoAct2 checkpoint we export: its package variable and what its spec
/// must state.
struct Molmo {
    var: &'static str,
    cameras: &'static [&'static str],
    state_len: usize,
    chunk: [usize; 2],
    /// Bounds on the served actions, in the checkpoint's own units: engine CPU f32 and
    /// GPU f16 against PyTorch f32.
    cpu_bound: f32,
    gpu_bound: f32,
}

const MOLMOACT2: [Molmo; 3] = [
    Molmo {
        var: "CASCADIA_POLICY_PACKAGE_MOLMOACT2",
        cameras: &["image", "wrist_image"],
        state_len: 8,
        chunk: [10, 7],
        cpu_bound: 1e-3,
        gpu_bound: 1e-2,
    },
    Molmo {
        var: "CASCADIA_POLICY_PACKAGE_MOLMOACT2_DROID",
        cameras: &["exterior_1_left", "exterior_2_left", "wrist_left"],
        state_len: 8,
        chunk: [15, 8],
        cpu_bound: 1e-3,
        // GPU f16 measured 8.8e-4 on the Arc 140V.
        gpu_bound: 3e-3,
    },
    Molmo {
        var: "CASCADIA_POLICY_PACKAGE_MOLMOACT2_SO100",
        cameras: &["top", "side"],
        state_len: 6,
        chunk: [30, 6],
        // Degrees, about 71 on average: the fp16 weights alone put Intel's own runtime
        // 3.8e-3 away from PyTorch here; GPU f16 measured 1.05e-2 on the Arc 140V.
        cpu_bound: 1e-2,
        gpu_bound: 5e-2,
    },
];

/// Each MolmoAct2 package whose variable is set.
fn molmoact2s(test: &str) -> Vec<(&'static Molmo, PathBuf)> {
    MOLMOACT2
        .iter()
        .filter_map(|m| package_from(m.var, test).map(|d| (m, d)))
        .collect()
}

fn molmo_step(manifest: &PolicyManifest) -> MolmoAct2Step {
    manifest
        .preprocessors
        .iter()
        .find_map(|s| match s {
            PreStep::MolmoAct2(p) => Some(p.clone()),
            _ => None,
        })
        .unwrap()
}

/// An IR's ports as its `.xml` declares them, without compiling it.
fn ir_ports(xml: &Path) -> (Vec<PortInfo>, Vec<PortInfo>) {
    let text = std::fs::read_to_string(xml).unwrap();
    let attr = |tag: &str, key: &str| {
        let at = tag
            .find(&format!(" {key}=\""))
            .unwrap_or_else(|| panic!("no {key} in {tag}"))
            + key.len()
            + 3;
        tag[at..].split('"').next().unwrap().to_string()
    };
    let dtype = |t: &str| match t {
        "i64" | "I64" => DType::I64,
        "f32" | "FP32" => DType::F32,
        "boolean" | "BOOL" => DType::Bool,
        other => panic!("{other}"),
    };
    let tag = |s: &str, open: &str| {
        s[s.find(open).unwrap()..]
            .split('>')
            .next()
            .unwrap()
            .to_string()
    };
    let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
    for layer in text.split("<layer ").skip(1) {
        let head = format!(" {}", layer.split('>').next().unwrap());
        if head.contains(" type=\"Parameter\"") {
            let data = tag(layer, "<data ");
            inputs.push(PortInfo {
                aliases: vec![attr(&head, "name")],
                dtype: Some(dtype(&attr(&data, "element_type"))),
                dims: attr(&data, "shape")
                    .split(',')
                    .map(|d| d.parse().ok())
                    .collect(),
            });
        } else if head.contains(" type=\"Result\"") {
            let port = &layer[layer.find("<port ").unwrap()..layer.find("</port>").unwrap()];
            outputs.push(PortInfo {
                aliases: attr(&head, "output_names")
                    .split(',')
                    .map(Into::into)
                    .collect(),
                dtype: Some(dtype(&attr(&tag(port, "<port "), "precision"))),
                dims: port
                    .split("<dim>")
                    .skip(1)
                    .map(|d| d.split('<').next().unwrap().parse().ok())
                    .collect(),
            });
        }
    }
    (inputs, outputs)
}

/// A float `[1, 3, H, W]` frame as the uint8 HWC a client sends: Intel's
/// `_resize` conversion, `clip(x · 255, 0, 255)` truncated.
fn molmo_frame(a: &Npy) -> Image {
    let (h, w) = (a.shape[2], a.shape[3]);
    let px = a.f32s();
    let scale = if px.iter().fold(f32::MIN, |m, &x| m.max(x)) <= 1.0 {
        255.0
    } else {
        1.0
    };
    let mut hwc = vec![0u8; 3 * h * w];
    for c in 0..3 {
        for p in 0..h * w {
            hwc[p * 3 + c] = (px[c * h * w + p] * scale).clamp(0.0, 255.0) as u8;
        }
    }
    Image {
        height: h as u32,
        width: w as u32,
        data: hwc,
    }
}

fn molmo_observation(
    z: &[u8],
    prefix: &str,
    task: String,
    cameras: &[impl AsRef<str>],
) -> Observation {
    Observation {
        images: cameras
            .iter()
            .map(|c| {
                let c = c.as_ref();
                (
                    c.to_string(),
                    molmo_frame(&npz(z, &format!("{prefix}images.{c}"))),
                )
            })
            .collect(),
        state: npz(z, &format!("{prefix}state")).f32s(),
        prompt: task,
        seed: None,
        noise: None,
        rtc: None,
    }
}

/// Every stage of Intel's runtime, bit for bit: the prompt text, the
/// tokenizer's ids, all six graph inputs; its postprocessor within 1e-6. The
/// ports and cameras are the package's own. No model run, so it needs no
/// device.
#[test]
fn molmoact2_engine_binds_intels_stages() {
    for (m, dir) in molmoact2s("molmoact2_engine_binds_intels_stages") {
        eprintln!("== {}", m.var);
        let z = beside(&dir, "molmoact2-stages-f32.npz");
        let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
        let step = molmo_step(&manifest);
        assert_eq!(step.image_keys, m.cameras);
        let task = npz(&z, "00_obs/task").string();
        let obs = molmo_observation(&z, "00_obs/", task.clone(), &step.image_keys);
        assert_eq!(obs.state.len(), m.state_len);
        assert_eq!(
            cascadia_engine_openvino::policy::molmoact2::render_prompt(&step, &task, &obs.state),
            npz(&z, "01_MolmoAct2Preprocessor/task").string()
        );
        let mut tok = OvTokenizer::load(&dir.join(manifest.tokenizer_artifact())).unwrap();
        let t = tok
            .tokenize(&npz(&z, "01_MolmoAct2Preprocessor/task").string())
            .unwrap();
        assert_eq!(t.ids, npz(&z, "02_OVTokenizer/tokenized_prompt").i64s());
        assert_eq!(
            t.mask,
            npz(&z, "02_OVTokenizer/tokenized_prompt_mask").bools()
        );
        let specials = tok.special_tokens();
        let bound = Arc::new(Mutex::new(Vec::new()));
        let (inputs, outputs) = ir_ports(&dir.join(&manifest.model_artifact));
        let action = outputs
            .iter()
            .find(|p| p.aliases.iter().any(|a| a == "action"));
        let shape: Vec<usize> = action.unwrap().dims.iter().map(|d| d.unwrap()).collect();
        assert_eq!(shape, [1, m.chunk[0], m.chunk[1]]);
        let mut pipeline = PolicyPipeline::new(
            manifest,
            ServerFacts {
                model: "molmoact2".into(),
                // No model runs; on CPU an ARM host would refuse the package.
                device: "GPU".into(),
                prompt_max_bytes: 1024,
            },
            Box::new(JepaRecorder {
                inputs: bound.clone(),
                output: Arc::new(Mutex::new(npz(&z, "90_runner/action").f32s())),
                shape,
            }),
            &inputs,
            &outputs,
            false,
            Box::new(tok),
        )
        .unwrap();
        let a = pipeline.infer(&obs).unwrap();
        let got = bound.lock().unwrap().clone();
        let get = |name: &str| {
            got.iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.data.clone())
                .unwrap_or_else(|| panic!("{name} not bound"))
        };
        let stage = |name: &str| format!("03_MolmoAct2ModelInputs/{name}");
        for name in [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "token_pooling",
        ] {
            assert_eq!(
                get(name),
                TensorData::I64(npz(&z, &stage(name)).i64s()),
                "{name}"
            );
        }
        assert_eq!(
            get("images"),
            TensorData::F32(npz(&z, &stage("images")).f32s()),
            "images bit for bit"
        );
        assert_eq!(
            get("action_dim_is_pad"),
            TensorData::Bool(npz(&z, &stage("action_dim_is_pad")).bools())
        );
        assert_eq!([a.steps, a.dim], m.chunk);
        let d = max_abs_diff(
            &a.actions,
            &npz(&z, "91_MolmoAct2Postprocessor/action").f32s(),
        );
        eprintln!("molmoact2 postprocess vs intel: max abs {d:e}");
        assert!(d <= 1e-6, "molmoact2_postprocess differs by {d}");
        assert_special_tasks_refused(
            &mut pipeline,
            &specials,
            &[
                "<|image|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
                "<action_output>",
                "<state_start>",
                "<state_end>",
                "<state_7>",
                "<setup_start>",
                "<control_end>",
            ],
            &obs,
        );
        // Lowercased by the task normalization, so still structure.
        let mut upper = obs.clone();
        upper.prompt = "pick <STATE_7> up".into();
        let e = pipeline.infer(&upper).unwrap_err();
        assert!(
            e.to_string().contains("the task contains \"<state_7>\""),
            "{e}"
        );
    }
}

/// CPU f32 with fp16 weights against the PyTorch f32 pipeline on a real
/// frame set, as served; and the prompt limit: the tokenizer IR's
/// `truncation` variable is held off, so 255 tokens are served and 256 refused.
#[test]
fn molmoact2_spec_prompt_limit_and_cpu_parity() {
    for (m, dir) in molmoact2s("molmoact2_spec_prompt_limit_and_cpu_parity") {
        eprintln!("== {}", m.var);
        let z = beside(&dir, "molmoact2-e2e.npz");
        let task = npz(&z, "task").string();
        let manifest = PolicyManifest::load(&dir, LoadOptions::default()).unwrap();
        let step = molmo_step(&manifest);
        let mut tok = OvTokenizer::load(&dir.join("tokenizer.xml")).unwrap();
        assert!(tok.has_truncation_state());
        let words = |k: usize| vec!["word"; k].join(" ");
        let state = npz(&z, "obs_state").f32s();
        let mut count = |k: usize| {
            let text = cascadia_engine_openvino::policy::molmoact2::render_prompt(
                &step,
                &words(k),
                &state,
            );
            let t = tok.tokenize(&text).unwrap();
            (t.ids.len(), t.mask.iter().filter(|&&m| m).count())
        };
        let fit = (1..300).find(|&k| count(k).1 == 255).unwrap();
        assert_eq!(
            count(fit + 1),
            (256, 256),
            "held-off truncation leaves it wider"
        );

        // OpenVINO 2026.4.1's ARM CPU plugin computes this graph wrong (the
        // LIBERO IR alone is 0.61 off on macOS, x86 within 1e-3), so the engine
        // refuses it there and the parity needs an x86 CPU.
        if cfg!(target_arch = "aarch64") {
            let e = OvPolicyEngine::load(&dir, &cpu())
                .err()
                .expect("an ARM CPU load is refused");
            assert!(e.to_string().contains("use an x86 CPU or the GPU"), "{e}");
            eprintln!(
                "skip molmoact2 cpu parity ({}): the engine refuses an ARM CPU",
                m.var
            );
            continue;
        }

        // The IR alone on PyTorch's own graph inputs, compiled and dropped first.
        let feeds: Vec<_> = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "images",
            "token_pooling",
            "action_dim_is_pad",
        ]
        .iter()
        .map(|n| feed(&z, n, &format!("pre_{n}")))
        .collect();
        let (shape, raw) = ir_on_cpu(&dir.join(&manifest.model_artifact), &feeds);
        assert_eq!(shape, [1, m.chunk[0], m.chunk[1]]);
        let ir_diff = max_abs_diff(&raw, &npz(&z, "norm_actions").f32s());
        eprintln!("molmoact2 IR cpu f32 vs pytorch f32 (normalized): max abs {ir_diff:.3e}");
        assert!(ir_diff <= 1e-3, "IR max abs diff {ir_diff}");
        let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
        let spec = engine.spec().clone();
        assert_eq!(spec.cameras, m.cameras);
        assert_eq!((spec.height, spec.width), (378, 378));
        assert_eq!(spec.resize, ResizeConvention::Stretch);
        assert_eq!((spec.state_len, spec.chunk), (m.state_len, m.chunk));
        assert_eq!(
            (spec.noise, spec.noise_shape.clone()),
            (NoiseMode::Constant, None)
        );
        let obs = molmo_observation(&z, "obs_", task, &spec.cameras);
        let a = engine.infer(&obs).unwrap();
        assert_eq!(
            (a.steps, a.dim, a.actions.len()),
            (m.chunk[0], m.chunk[1], m.chunk[0] * m.chunk[1])
        );
        let d = max_abs_diff(&a.actions, &npz(&z, "actions").f32s());
        eprintln!(
            "molmoact2 engine cpu f32 vs pytorch f32 {}: max abs {d:.3e} infer_ms={:.0}",
            m.var, a.infer_ms
        );
        assert!(d <= m.cpu_bound, "max abs diff {d} > {}", m.cpu_bound);
        let mut long = obs.clone();
        long.prompt = words(fit + 1);
        let e = engine.infer(&long).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
        assert!(
            e.to_string().contains("needs 256 tokens; the limit is 255"),
            "{e}"
        );
        long.prompt = words(fit);
        engine.infer(&long).unwrap();
    }
}

/// MolmoAct2 on the GPU: f16 against PyTorch f32 as served, per checkpoint (LIBERO
/// spike: 9.9e-4 normalized). At `--ov-inference-precision f32` the GPU compile fails
/// (`CL_INVALID_EVENT`, seen on LIBERO and DROID), which must be a load error; that runs
/// in a child process, as π0's does.
#[test]
fn molmoact2_gpu_infer_f16_and_f32_is_a_load_error() {
    let test = "molmoact2_gpu_infer_f16_and_f32_is_a_load_error";
    let packages = molmoact2s(test);
    if packages.is_empty() || !gpu_enabled(test) {
        return;
    }
    let mut over = Vec::new();
    for (m, dir) in &packages {
        let z = beside(dir, "molmoact2-e2e.npz");
        let opts = OvPolicyOptions {
            device: "GPU".into(),
            ..OvPolicyOptions::default()
        };
        let mut engine = OvPolicyEngine::load(dir, &opts).unwrap();
        let warm = engine.warmup().unwrap();
        assert_eq!(warm.len(), WARMUP_CALLS);
        let obs = molmo_observation(&z, "obs_", npz(&z, "task").string(), m.cameras);
        let a = engine.infer(&obs).unwrap();
        assert!(a.actions.iter().all(|x| x.is_finite()), "{:?}", a.actions);
        let d = max_abs_diff(&a.actions, &npz(&z, "actions").f32s());
        eprintln!(
            "gpu molmoact2 f16 {}: max abs {d:.3e} infer_ms={:.1} warmup={warm:?}",
            m.var, a.infer_ms
        );
        if d > m.gpu_bound {
            over.push(format!("{}: {d} > {}", m.var, m.gpu_bound));
        }
    }
    assert!(over.is_empty(), "gpu molmoact2 f16 max abs diff: {over:?}");
    let child = "molmoact2_gpu_f32_load_is_an_error_in_its_own_process";
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", child, "--ignored", "--nocapture"])
        .env("CASCADIA_POLICY_MOLMOACT2_F32_CHILD", &packages[0].1)
        .output()
        .unwrap();
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{child}: {}\n{stdout}",
        out.status
    );
}

#[test]
#[ignore = "run by molmoact2_gpu_infer_f16_and_f32_is_a_load_error, in its own process"]
fn molmoact2_gpu_f32_load_is_an_error_in_its_own_process() {
    // The parent passes its first package; run by hand, any MolmoAct2 package variable works.
    let dir = std::env::var_os("CASCADIA_POLICY_MOLMOACT2_F32_CHILD")
        .or_else(|| MOLMOACT2.iter().find_map(|m| std::env::var_os(m.var)))
        .map(PathBuf::from)
        .expect("set a CASCADIA_POLICY_PACKAGE_MOLMOACT2* package variable");
    let opts = OvPolicyOptions {
        device: "GPU".into(),
        precision: Some(Precision::F32),
        ..OvPolicyOptions::default()
    };
    match OvPolicyEngine::load(&dir, &opts) {
        Ok(_) => panic!("gpu molmoact2 f32 compiled; expected it to fail on the Arc 140V"),
        Err(e) => {
            eprintln!("gpu molmoact2 f32 load error: {e}");
            assert!(matches!(e, EngineError::Backend(_)), "{e}");
            let m = e.to_string();
            assert!(m.contains("molmoact2.xml on GPU failed"), "{m}");
            assert!(
                m.contains("try without --ov-inference-precision f32"),
                "{m}"
            );
        }
    }
}

// ---- real-time chunking (RTC) ----
//
// SmolVLA and π0.5 LIBERO exported with Intel's exporter with RTC on and off
// (the full chunk, no trimmer). Per model `<M>` (`SMOLVLA`, `PI05`):
// - `CASCADIA_POLICY_PACKAGE_<M>_RTC`, `..._<M>_NORTC`: the shipped packages
//   (fp16 weights) from the same export run;
// - `CASCADIA_POLICY_PACKAGE_<M>_RTC_F32`: the RTC export's float32 twin;
// - `CASCADIA_POLICY_RTC_REF_<M>`: raw inputs and Intel's PyTorch f32 outputs
//   (`actions_rtc`, `actions_plain`) for them.

/// The reference's raw inputs, as a robot sends them.
fn rtc_case(z: &[u8], spec: &cascadia_types::PolicySpec) -> (Observation, RtcInputs) {
    let images = spec
        .cameras
        .iter()
        .map(|c| {
            let a = npz(z, &format!("image_{c}"));
            assert_eq!(a.shape, [spec.height as usize, spec.width as usize, 3]);
            (
                c.clone(),
                Image {
                    height: spec.height,
                    width: spec.width,
                    data: a.data,
                },
            )
        })
        .collect();
    let obs = Observation {
        images,
        state: npz(z, "state").f32s(),
        prompt: npz(z, "prompt").string(),
        seed: None,
        noise: Some(npz(z, "noise").f32s()),
        rtc: None,
    };
    let rtc = RtcInputs {
        prev_chunk_left_over: npz(z, "left").f32s(),
        inference_delay: npz(z, "delay").i64s()[0] as u32,
        max_guidance_weight: npz(z, "weight").f32s()[0],
        execution_horizon: npz(z, "horizon").i64s()[0] as u32,
    };
    (obs, rtc)
}

fn rtc_reference(model: &str, test: &str) -> Option<Vec<u8>> {
    let path = package_from(&format!("CASCADIA_POLICY_RTC_REF_{model}"), test)?;
    Some(std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// Gate 1: guidance off is the plain chunk. Inside the RTC package, a call
/// with the fields at `max_guidance_weight` 0 (and a real leftover) is
/// bit-identical to a call without them, so the guidance contributes exactly
/// nothing. Against the package Intel's exporter makes without RTC the chunk
/// agrees within 1e-4: that export traces to a differently folded graph
/// (SmolVLA: 67,583 ops against 42,757), so the two compile to different
/// kernels and are not bit-identical even on CPU f32.
fn rtc_guidance_zero_is_the_plain_chunk(model: &str) {
    let test = format!(
        "{}_rtc_guidance_zero_is_the_plain_chunk",
        model.to_lowercase()
    );
    let (Some(rtc_dir), Some(plain_dir), Some(z)) = (
        package_from(&format!("CASCADIA_POLICY_PACKAGE_{model}_RTC"), &test),
        package_from(&format!("CASCADIA_POLICY_PACKAGE_{model}_NORTC"), &test),
        rtc_reference(model, &test),
    ) else {
        return;
    };
    let mut plain = OvPolicyEngine::load(&plain_dir, &cpu()).unwrap();
    let mut rtc = OvPolicyEngine::load(&rtc_dir, &cpu()).unwrap();
    assert_eq!(plain.spec().rtc_chunk_size, None);
    let [t, d] = rtc.spec().chunk;
    assert_eq!(rtc.spec().rtc_chunk_size, Some(t));
    assert_eq!(plain.spec().chunk, [t, d]);
    let (obs, fields) = rtc_case(&z, rtc.spec());
    let want = plain.infer(&obs).unwrap();
    let off = rtc.infer(&obs).unwrap();
    let mut zero = obs.clone();
    zero.rtc = Some(RtcInputs {
        max_guidance_weight: 0.0,
        ..fields.clone()
    });
    let zero = rtc.infer(&zero).unwrap();
    let mut on = obs.clone();
    on.rtc = Some(fields);
    let on = rtc.infer(&on).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let cross = max_abs_diff(&off.actions, &want.actions);
    eprintln!(
        "{model} cpu: weight 0 vs no fields bit-identical: {}; vs the plain package {cross:.2e} \
         (max abs); guidance on moves the chunk by {:.4}; {:.1} / {:.1} / {:.1} ms",
        bits(&zero.actions) == bits(&off.actions),
        max_abs_diff(&on.actions, &off.actions),
        off.infer_ms,
        zero.infer_ms,
        want.infer_ms,
    );
    assert_eq!(
        bits(&zero.actions),
        bits(&off.actions),
        "max_guidance_weight 0"
    );
    assert!(cross <= 1e-4, "no RTC fields vs the plain package: {cross}");
    assert!(
        max_abs_diff(&on.actions, &off.actions) > 1e-3,
        "guidance had no effect"
    );
}

#[test]
fn smolvla_rtc_guidance_zero_is_the_plain_chunk() {
    rtc_guidance_zero_is_the_plain_chunk("SMOLVLA");
}

#[test]
fn pi05_rtc_guidance_zero_is_the_plain_chunk() {
    rtc_guidance_zero_is_the_plain_chunk("PI05");
}

/// Gate 2: with guidance on, the engine on the f32 twin matches Intel's
/// PyTorch RTC path within 1e-4 on CPU f32 (and its plain path too).
fn rtc_matches_intels_pytorch(model: &str) {
    let test = format!("{}_rtc_matches_intels_pytorch", model.to_lowercase());
    let (Some(dir), Some(z)) = (
        package_from(&format!("CASCADIA_POLICY_PACKAGE_{model}_RTC_F32"), &test),
        rtc_reference(model, &test),
    ) else {
        return;
    };
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let (mut obs, fields) = rtc_case(&z, engine.spec());
    let plain = engine.infer(&obs).unwrap();
    obs.rtc = Some(fields);
    let on = engine.infer(&obs).unwrap();
    let (want_on, want_plain) = (
        npz(&z, "actions_rtc").f32s(),
        npz(&z, "actions_plain").f32s(),
    );
    let (d_on, d_plain) = (
        max_abs_diff(&on.actions, &want_on),
        max_abs_diff(&plain.actions, &want_plain),
    );
    eprintln!(
        "{model} cpu f32 vs PyTorch: rtc {d_on:.2e}, plain {d_plain:.2e}; rtc effect {:.4} \
         (max abs); {:.1} ms",
        max_abs_diff(&want_on, &want_plain),
        on.infer_ms
    );
    assert_eq!(on.actions.len(), want_on.len());
    assert!(d_on <= 1e-4, "rtc: max abs diff {d_on} > 1e-4");
    assert!(d_plain <= 1e-4, "plain: max abs diff {d_plain} > 1e-4");
}

#[test]
fn smolvla_rtc_matches_intels_pytorch() {
    rtc_matches_intels_pytorch("SMOLVLA");
}

#[test]
fn pi05_rtc_matches_intels_pytorch() {
    rtc_matches_intels_pytorch("PI05");
}

/// Gate 3: on the GPU at f16, RTC chunks are finite and stay as close to the
/// CPU f32 result as the plain chunk does (twice its deviation, plus 1e-3).
fn rtc_gpu_f16_stays_in_the_f16_envelope(model: &str) {
    let test = format!(
        "{}_rtc_gpu_f16_stays_in_the_f16_envelope",
        model.to_lowercase()
    );
    let (Some(dir), Some(z)) = (
        package_from(&format!("CASCADIA_POLICY_PACKAGE_{model}_RTC"), &test),
        rtc_reference(model, &test),
    ) else {
        return;
    };
    if !gpu_enabled(&test) {
        return;
    }
    let mut cpu_engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let (obs, fields) = rtc_case(&z, cpu_engine.spec());
    let mut on = obs.clone();
    on.rtc = Some(fields);
    let (cpu_plain, cpu_on) = (
        cpu_engine.infer(&obs).unwrap(),
        cpu_engine.infer(&on).unwrap(),
    );
    drop(cpu_engine);
    let gpu = OvPolicyOptions {
        device: "GPU".into(),
        ..OvPolicyOptions::default()
    };
    let mut engine = OvPolicyEngine::load(&dir, &gpu).unwrap();
    let warm = engine.warmup().unwrap();
    let mut ms = (Vec::new(), Vec::new());
    let (mut g_plain, mut g_on) = (None, None);
    for _ in 0..10 {
        let p = engine.infer(&obs).unwrap();
        let r = engine.infer(&on).unwrap();
        ms.0.push(p.infer_ms);
        ms.1.push(r.infer_ms);
        g_plain = Some(p);
        g_on = Some(r);
    }
    let (g_plain, g_on) = (g_plain.unwrap(), g_on.unwrap());
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (d_plain, d_on) = (
        max_abs_diff(&g_plain.actions, &cpu_plain.actions),
        max_abs_diff(&g_on.actions, &cpu_on.actions),
    );
    eprintln!(
        "{model} gpu f16: plain p50 {:.1} ms, rtc p50 {:.1} ms, warmup {warm:?}; vs cpu f32: \
         plain {d_plain:.2e}, rtc {d_on:.2e}",
        median(&mut ms.0),
        median(&mut ms.1)
    );
    assert!(g_on.actions.iter().all(|x| x.is_finite()));
    assert!(
        d_on <= 2.0 * d_plain + 1e-3,
        "rtc f16 deviation {d_on} is outside the plain chunk's envelope ({d_plain})"
    );
}

#[test]
fn smolvla_rtc_gpu_f16_stays_in_the_f16_envelope() {
    rtc_gpu_f16_stays_in_the_f16_envelope("SMOLVLA");
}

#[test]
fn pi05_rtc_gpu_f16_stays_in_the_f16_envelope() {
    rtc_gpu_f16_stays_in_the_f16_envelope("PI05");
}

/// openpi's DROID client sends two cameras and an 8-wide state; the package serves the third
/// slot empty and pads the state for the graph, so the engine's actions must be LeRobot's for
/// the same frames, state, task and noise. Only the first 8 action dims are DROID's.
#[test]
fn pi05_droid_cpu_infer_matches_lerobot() {
    let test = "pi05_droid_cpu_infer_matches_lerobot";
    let Some(dir) = package_from("CASCADIA_POLICY_PACKAGE_PI05_DROID", test) else {
        return;
    };
    let z = reference(&dir);
    let mut engine = OvPolicyEngine::load(&dir, &cpu()).unwrap();
    let spec = engine.spec().clone();
    assert_eq!(spec.cameras, vec!["base_0_rgb", "left_wrist_0_rgb"]);
    assert_eq!((spec.height, spec.width, spec.state_len), (224, 224, 8));
    assert_eq!(spec.chunk, [15, 32]);
    assert_eq!(
        (spec.noise, spec.noise_shape.clone()),
        (NoiseMode::Input, Some(vec![15, 32]))
    );
    let (base, wrist) = (npz(&z, "base"), npz(&z, "wrist"));
    let frames = base.shape[0];
    for (key, a) in [("base", &base), ("wrist", &wrist)] {
        assert_eq!(
            (a.descr.as_str(), a.shape.as_slice()),
            ("|u1", &[frames, 224, 224, 3][..]),
            "reference {key}: expected uint8 frames of 224x224x3"
        );
    }
    let (states, noise, want) = (
        npz(&z, "state").f32s(),
        npz(&z, "noise").f32s(),
        npz(&z, "actions").f32s(),
    );
    let task = npz(&z, "task").string();
    let px = 224 * 224 * 3;
    let image = |a: &Npy, i: usize| Image {
        height: 224,
        width: 224,
        data: a.data[i * px..(i + 1) * px].to_vec(),
    };
    let (mut max, mut sum, mut mag, mut n) = (0.0f32, 0.0f64, 0.0f64, 0usize);
    for i in 0..frames {
        let obs = Observation {
            images: vec![
                ("base_0_rgb".into(), image(&base, i)),
                ("left_wrist_0_rgb".into(), image(&wrist, i)),
            ],
            state: states[i * 8..(i + 1) * 8].to_vec(),
            prompt: task.clone(),
            seed: None,
            noise: Some(noise[i * 480..(i + 1) * 480].to_vec()),
            rtc: None,
        };
        let a = engine.infer(&obs).unwrap();
        assert_eq!((a.steps, a.dim), (15, 32));
        let w = &want[i * 480..(i + 1) * 480];
        for (g, r) in a.actions.chunks(32).zip(w.chunks(32)) {
            for (x, y) in g[..8].iter().zip(&r[..8]) {
                let d = (x - y).abs();
                max = max.max(d);
                sum += f64::from(d);
                mag += f64::from(y.abs());
                n += 1;
            }
        }
    }
    let (mean, mag) = (sum / n as f64, mag / n as f64);
    eprintln!(
        "pi05_droid engine cpu f32 vs lerobot f32 (first 8 dims, {frames} frames): max abs {max:.3e} \
         mean abs {mean:.3e}, mean |action| {mag:.4}"
    );
    assert!(max <= 5e-3, "max abs diff {max}");
}
