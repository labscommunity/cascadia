//! `OvPolicyEngine`: a `policy_package` on the OpenVINO runtime. The policy
//! IR follows the device; the tokenizer always runs on CPU, in its own
//! runtime handle, because its string ops are CPU-only.

use std::path::Path;

use cascadia_engine::{EngineError, EngineResult, PolicyEngine};
use cascadia_types::{ActionChunk, Observation, PolicySpec};

use super::manifest::LoadOptions;
use super::pipeline::{PolicyPipeline, Precision};

/// How to load a package.
#[derive(Clone, Debug)]
pub struct OvPolicyOptions {
    /// OpenVINO device for the policy IR: `GPU` (or `GPU.<n>`) or `CPU`.
    pub device: String,
    /// GPU compute precision; `None` keeps the device default (f16 on GPU).
    pub precision: Option<Precision>,
    pub manifest: LoadOptions,
}

impl Default for OvPolicyOptions {
    fn default() -> Self {
        Self {
            device: "GPU".into(),
            precision: None,
            manifest: LoadOptions::default(),
        }
    }
}

pub struct OvPolicyEngine {
    pipeline: PolicyPipeline,
}

impl OvPolicyEngine {
    /// Load, compile and check a package directory. Nothing is served from a
    /// package whose manifest, ports or tokenizer disagree.
    #[cfg(not(feature = "openvino"))]
    pub fn load(dir: &Path, opts: &OvPolicyOptions) -> EngineResult<Self> {
        let _ = (dir, opts);
        Err(EngineError::Backend(
            "the ov-policy engine needs a build with --features openvino".into(),
        ))
    }

    /// Load, compile and check a package directory. Nothing is served from a
    /// package whose manifest, ports or tokenizer disagree.
    #[cfg(feature = "openvino")]
    pub fn load(dir: &Path, opts: &OvPolicyOptions) -> EngineResult<Self> {
        use super::manifest::PolicyManifest;
        use super::pipeline::{compile_config, PROMPT_MAX_BYTES, RANDOM_UNIFORM};
        use super::spec::ServerFacts;
        use crate::runtime::map_ov_err;
        use cascadia_ov_genai_shim::Runtime;
        use sha2::Digest;

        let manifest = PolicyManifest::load(dir, opts.manifest)?;
        let (device, config) = compile_config(&opts.device, opts.precision)?;
        let xml = dir.join(&manifest.model_artifact);
        let rt = Runtime::compile_inspected(utf8(&xml)?, &device, None, &config)
            .map_err(|e| compile_failed(&xml, &device, opts.precision, map_ov_err(e)))?;
        let inputs = (0..rt.input_count())
            .map(|i| rt.input_port(i))
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_ov_err)?;
        let outputs = (0..rt.output_count())
            .map(|i| rt.output_port(i))
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_ov_err)?;
        let has_random_uniform = rt.has_op_type(RANDOM_UNIFORM).map_err(map_ov_err)?;
        let tokenizer = native::OvTokenizer::load(&dir.join(manifest.tokenizer_artifact()))?;
        let model = dir.file_name().map_or_else(
            || dir.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let manifest_bytes = std::fs::read(dir.join("manifest.json"))
            .map_err(|e| EngineError::InvalidConfig(format!("read manifest.json: {e}")))?;
        let manifest_sha256 = sha2::Sha256::digest(&manifest_bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let precision = super::pipeline::compiled_precision(&device, opts.precision);
        let mut pipeline = PolicyPipeline::new(
            manifest,
            ServerFacts {
                model,
                device,
                prompt_max_bytes: PROMPT_MAX_BYTES,
            },
            Box::new(native::OvModel {
                rt,
                outputs: outputs.clone(),
            }),
            &inputs,
            &outputs,
            has_random_uniform,
            Box::new(tokenizer),
        )?;
        pipeline.set_identity(manifest_sha256, precision);
        Ok(Self { pipeline })
    }
}

impl PolicyEngine for OvPolicyEngine {
    fn spec(&self) -> &PolicySpec {
        self.pipeline.spec()
    }

    fn warmup(&mut self) -> EngineResult<Vec<u32>> {
        self.pipeline.warmup()
    }

    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        self.pipeline.infer(obs)
    }

    fn close(&mut self) {
        self.pipeline.close();
    }
}

#[cfg(feature = "openvino")]
fn utf8(p: &Path) -> EngineResult<&str> {
    p.to_str()
        .ok_or_else(|| EngineError::InvalidConfig(format!("path is not UTF-8: {}", p.display())))
}

/// A failed policy compile as a load error; on the GPU at f32 it says that
/// f32 needs about twice the device memory of the f16 default.
#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
fn compile_failed(
    xml: &Path,
    device: &str,
    precision: Option<Precision>,
    e: EngineError,
) -> EngineError {
    match e {
        EngineError::Backend(m) => {
            let hint = if device.starts_with("GPU") && precision == Some(Precision::F32) {
                "; --precision f32 needs about twice the GPU memory of the f16 default, \
                 which may not fit an integrated GPU: try without --precision f32"
            } else {
                ""
            };
            EngineError::Backend(format!(
                "compiling {} on {device} failed: {m}{hint}",
                xml.display()
            ))
        }
        other => other,
    }
}

#[cfg(feature = "openvino")]
fn context(path: &Path, e: EngineError) -> EngineError {
    match e {
        EngineError::Backend(m) => EngineError::Backend(format!("{}: {m}", path.display())),
        other => other,
    }
}

#[cfg(feature = "openvino")]
pub use native::OvTokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_compile_failure_is_a_load_error_with_a_gpu_f32_hint() {
        let xml = Path::new("pkg/pi0.xml");
        let e = || EngineError::Backend("[GPU] ProgramBuilder build failed!".into());
        let m = compile_failed(xml, "GPU", Some(Precision::F32), e()).to_string();
        assert!(
            m.contains("compiling pkg/pi0.xml on GPU failed: [GPU] ProgramBuilder"),
            "{m}"
        );
        assert!(m.contains("try without --precision f32"), "{m}");
        for (device, precision) in [
            ("GPU", None),
            ("CPU", None),
            ("GPU.1", Some(Precision::F16)),
        ] {
            let m = compile_failed(xml, device, precision, e()).to_string();
            assert!(m.contains(&format!("on {device} failed")), "{m}");
            assert!(!m.contains("--precision"), "{m}");
        }
        assert!(matches!(
            compile_failed(xml, "GPU", None, EngineError::InvalidConfig("x".into())),
            EngineError::InvalidConfig(_)
        ));
    }
}

#[cfg(feature = "openvino")]
mod native {
    use std::path::Path;

    use cascadia_engine::{EngineError, EngineResult};
    use cascadia_ov_genai_shim::{
        find_tokenizers_extension, DType, PluginConfig, PortInfo, Runtime,
    };

    use super::super::pipeline::{HostTensor, PolicyModel, PromptTokenizer, Tokens};
    use super::super::special_tokens as special;
    use super::{context, utf8};
    use crate::runtime::map_ov_err;

    /// The tokenizer IR's state variable that switches truncation.
    const TRUNCATION: &str = "truncation";

    fn index_of(ports: &[PortInfo], name: &str) -> EngineResult<usize> {
        ports
            .iter()
            .position(|p| p.aliases.iter().any(|a| a == name))
            .ok_or_else(|| EngineError::InvalidConfig(format!("no output named {name:?}")))
    }

    pub struct OvModel {
        pub rt: Runtime,
        pub outputs: Vec<PortInfo>,
    }

    impl PolicyModel for OvModel {
        fn set_input(&mut self, name: &str, t: &HostTensor) -> EngineResult<()> {
            self.rt
                .set_input(name, t.dtype(), &t.shape, &t.to_bytes())
                .map_err(map_ov_err)
        }

        fn infer(&mut self) -> EngineResult<()> {
            self.rt.infer().map_err(map_ov_err)
        }

        fn output_f32(&mut self, name: &str) -> EngineResult<(Vec<usize>, Vec<f32>)> {
            let idx = index_of(&self.outputs, name)?;
            let (dtype, shape, bytes) = self.rt.output(idx).map_err(map_ov_err)?;
            let values = match dtype {
                DType::F32 => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                DType::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                other => {
                    return Err(EngineError::Backend(format!(
                        "output {name:?} is {other:?}; expected f32 or f16"
                    )))
                }
            };
            Ok((shape, values))
        }
    }

    /// A package tokenizer compiled on CPU with the `openvino_tokenizers`
    /// extension the loaded runtime ships.
    pub struct OvTokenizer {
        rt: Runtime,
        input: String,
        ids: usize,
        mask: usize,
        truncation: bool,
        special: Vec<String>,
    }

    impl OvTokenizer {
        pub fn load(xml: &Path) -> EngineResult<Self> {
            let ext = find_tokenizers_extension().map_err(map_ov_err)?;
            let mut rt = Runtime::compile_inspected(
                utf8(xml)?,
                "CPU",
                Some(utf8(&ext)?),
                &PluginConfig::new(),
            )
            .map_err(|e| context(xml, map_ov_err(e)))?;
            if rt.input_count() != 1 {
                return Err(EngineError::InvalidConfig(format!(
                    "{}: a tokenizer takes one input, this one has {}",
                    xml.display(),
                    rt.input_count()
                )));
            }
            let input = rt.input_name(0).map_err(map_ov_err)?;
            let outputs = (0..rt.output_count())
                .map(|i| rt.output_port(i))
                .collect::<Result<Vec<_>, _>>()
                .map_err(map_ov_err)?;
            let ids = index_of(&outputs, "input_ids")?;
            let mask = index_of(&outputs, "attention_mask")?;
            let truncation = rt.set_state_bool(TRUNCATION, false).map_err(map_ov_err)?;
            Ok(Self {
                rt,
                input,
                ids,
                mask,
                truncation,
                special: special_tokens(xml),
            })
        }

        fn row(&self, idx: usize, what: &str) -> EngineResult<Vec<i64>> {
            let (dtype, shape, bytes) = self.rt.output(idx).map_err(map_ov_err)?;
            if shape.len() != 2 || shape[0] != 1 {
                return Err(EngineError::Backend(format!(
                    "tokenizer {what} is {shape:?}; expected [1, n]"
                )));
            }
            Ok(match dtype {
                DType::I64 => bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
                    .collect(),
                DType::I32 => bytes
                    .chunks_exact(4)
                    .map(|c| i64::from(i32::from_le_bytes(c.try_into().expect("4 bytes"))))
                    .collect(),
                DType::Bool | DType::I8 => bytes.iter().map(|&b| i64::from(b)).collect(),
                other => {
                    return Err(EngineError::Backend(format!(
                        "tokenizer {what} is {other:?}; expected an integer type"
                    )))
                }
            })
        }
    }

    /// The IR's `SpecialTokensSplit` pattern plus `tokenizer.json`'s added
    /// tokens beside it; an unreadable artefact adds nothing.
    fn special_tokens(xml: &Path) -> Vec<String> {
        let mut out = Vec::new();
        if let (Ok(text), Ok(bin)) = (
            std::fs::read_to_string(xml),
            std::fs::read(xml.with_extension("bin")),
        ) {
            out.extend(special::from_tokenizer_ir(&text, &bin).unwrap_or_default());
        }
        if let Ok(json) = std::fs::read_to_string(xml.with_file_name("tokenizer.json")) {
            for t in special::from_tokenizer_json(&json) {
                if !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        out
    }

    impl PromptTokenizer for OvTokenizer {
        fn has_truncation_state(&self) -> bool {
            self.truncation
        }

        fn special_tokens(&self) -> Vec<String> {
            self.special.clone()
        }

        fn tokenize(&mut self, text: &str) -> EngineResult<Tokens> {
            if self.truncation {
                // Assign writes the variable back every call; set it each time
                // so nothing (a state reset) can turn truncation back on.
                self.rt
                    .set_state_bool(TRUNCATION, false)
                    .map_err(map_ov_err)?;
            }
            self.rt
                .set_input_strings(&self.input, &[text])
                .map_err(map_ov_err)?;
            self.rt.infer().map_err(map_ov_err)?;
            let ids = self.row(self.ids, "input_ids")?;
            let mask: Vec<bool> = self
                .row(self.mask, "attention_mask")?
                .iter()
                .map(|&m| m != 0)
                .collect();
            if ids.len() != mask.len() {
                return Err(EngineError::Backend(format!(
                    "tokenizer returned {} ids and {} mask values",
                    ids.len(),
                    mask.len()
                )));
            }
            Ok(Tokens { ids, mask })
        }
    }
}
