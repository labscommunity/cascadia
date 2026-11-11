//! Single-stage engine backed by `openvino_genai::LLMPipeline`.
//!
//! Mirrors `cascadia/worker/engines/openvino/genai_engine.py`. Three classes
//! of acceleration:
//!
//! * **FastDraft speculative decode** when `draft_model_path` is set.
//! * **Prompt Lookup decoding** when `prompt_lookup_ngram > 0`.
//! * **Plugin properties** (CACHE_DIR, KV_CACHE_PRECISION,
//!   DYNAMIC_QUANTIZATION_GROUP_SIZE) via `PluginConfig`.
//!
//! Usage, and the finish reason of a plain request, use the LlmPipeline
//! tokenizer's count of the decoded text: `perf_metrics` has reported
//! `max_new_tokens` when EOS fired early on short greedy decodes. A request
//! with an output constraint also consults the generated-token count for its
//! finish reason, since re-tokenising constrained output can undercount; on
//! OpenVINO GenAI 2026.4.1 that count matched short early-EOS answers on GPU
//! and CPU.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, RandomState};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{
    clean_backend_message, CbFinish, CbHandle, CbPipeline, CbRead, CbSchedulerConfig, CbStatus,
    Error as OvError, GenConfig, LlmPipeline, PluginConfig,
};
use cascadia_types::{
    Chunk, FinishReason, GenerationTask, GrammarKind, LoadProgress, PeerLayout, SamplingParams,
    ShardSpec, TaskId,
};
use futures::stream;
use tracing::{info, warn};

#[derive(Default)]
pub struct OvGenaiBuilder {
    pub model_path: String,
    pub device: String,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    /// Extra `(key, value)` OV plugin properties (PERFORMANCE_HINT,
    /// INFERENCE_PRECISION_HINT, NPU knobs, …) plumbed verbatim from the CLI.
    pub ov_properties: Vec<(String, String)>,
    pub draft_model_path: Option<String>,
    pub draft_device: Option<String>,
    pub speculative_k: u32,
    pub prompt_lookup_ngram: u32,
    /// True when the API pre-renders the model's chat template (so the engine
    /// tells ov-genai to skip its internal apply — lets `enable_thinking` work).
    pub prompt_pretemplated: bool,
    /// Continuous batching (#20): serve concurrent requests through one
    /// `ContinuousBatchingPipeline` instead of one-generation-per-`step()`.
    pub continuous_batching: Option<CbSchedulerConfig>,
    pipe: Option<LlmPipeline>,
    cb_pipe: Option<CbPipeline>,
}

impl OvGenaiBuilder {
    pub fn new(model_path: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            device: device.into(),
            cache_dir: None,
            kv_cache_precision: None,
            dyn_quant_group: None,
            ov_properties: Vec::new(),
            draft_model_path: None,
            draft_device: None,
            speculative_k: 0,
            prompt_lookup_ngram: 0,
            prompt_pretemplated: false,
            continuous_batching: None,
            pipe: None,
            cb_pipe: None,
        }
    }

    /// Enable continuous batching (#20) with the given scheduler knobs.
    pub fn with_continuous_batching(mut self, sched: CbSchedulerConfig) -> Self {
        self.continuous_batching = Some(sched);
        self
    }

    /// Set when the API renders the chat template itself (template loaded).
    pub fn with_prompt_pretemplated(mut self, v: bool) -> Self {
        self.prompt_pretemplated = v;
        self
    }

    pub fn with_cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }
    pub fn with_kv_cache_precision(mut self, prec: impl Into<String>) -> Self {
        self.kv_cache_precision = Some(prec.into());
        self
    }
    pub fn with_dyn_quant_group(mut self, group: impl Into<String>) -> Self {
        self.dyn_quant_group = Some(group.into());
        self
    }
    /// Append extra `(key, value)` OV plugin properties (CLI perf flags).
    pub fn with_ov_properties(mut self, props: Vec<(String, String)>) -> Self {
        self.ov_properties.extend(props);
        self
    }
    pub fn with_draft(
        mut self,
        path: impl Into<String>,
        device: impl Into<String>,
        k: u32,
    ) -> Self {
        self.draft_model_path = Some(path.into());
        self.draft_device = Some(device.into());
        self.speculative_k = k;
        self
    }
    pub fn with_prompt_lookup(mut self, ngram: u32) -> Self {
        self.prompt_lookup_ngram = ngram;
        self
    }

    fn build_plugin_config(&self) -> PluginConfig {
        let mut cfg = PluginConfig::new();
        if let Some(dir) = &self.cache_dir {
            cfg = cfg.with("CACHE_DIR", dir);
        }
        if let Some(prec) = &self.kv_cache_precision {
            cfg = cfg.with("KV_CACHE_PRECISION", prec);
        }
        if let Some(group) = &self.dyn_quant_group {
            cfg = cfg.with("DYNAMIC_QUANTIZATION_GROUP_SIZE", group);
        }
        for (key, val) in &self.ov_properties {
            cfg = cfg.with(key, val);
        }
        cfg
    }
}

/// ov::genai builds its tokenizer from OpenVINO IRs in the model dir. Without
/// them the pipeline still constructs, then every request returns an empty
/// string with no error — reject the directory instead of serving silence.
fn check_tokenizer_irs(dir: &str) -> EngineResult<()> {
    let d = PathBuf::from(dir);
    let missing: Vec<&str> = ["openvino_tokenizer.xml", "openvino_detokenizer.xml"]
        .into_iter()
        .filter(|f| !d.join(f).exists())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    // A staged tree is a common mistake here: it has no tokenizer IRs because the
    // staged engines tokenize in Rust. Say so — and name the engine that matches
    // this tree — rather than telling the user to re-export a model they already
    // exported. Two tree shapes exist: export_shards.py / the gemma4 exporters
    // write pipeline_config.json, while the surgery exporters write manifest.json.
    let read_key = |file: &str, key: &str| -> Option<String> {
        std::fs::read_to_string(d.join(file))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v[key].as_str().map(str::to_owned))
    };
    let engine = match (
        read_key("pipeline_config.json", "export_version"),
        read_key("manifest.json", "arch"),
    ) {
        (Some(v), _) if v.starts_with("gemma4") => Some("gemma4"),
        (Some(_), _) => Some("ov-runtime"),
        // qwen3_5 (dense Qwen3.8) and qwen3_5_moe (Qwen3.5/3.6) share the
        // surgery exporter and the staged engine.
        (_, Some(a)) if a.starts_with("qwen3_5") => Some("qwen35"),
        (_, Some(_)) => Some("sparse-moe"),
        _ => None,
    };
    if let Some(engine) = engine {
        return Err(EngineError::InvalidConfig(format!(
            "{dir} is a staged tree, which ov-genai cannot serve. Use --engine {engine}."
        )));
    }
    Err(EngineError::InvalidConfig(format!(
        "{} missing from {dir}. ov-genai needs the tokenizer IRs beside the model IR; \
         an export without them generates empty output. Re-export with \
         `pip install \"optimum-intel[openvino]\"` + `optimum-cli export openvino`, \
         or download a pre-exported IR.",
        missing.join(" + "),
    )))
}

#[async_trait]
impl Builder for OvGenaiBuilder {
    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        if peers.upstream.is_some() || peers.downstream.is_some() {
            return Err(EngineError::PeerRejected(
                "ov-genai is single-stage only; do not configure peers".into(),
            ));
        }
        Ok(())
    }

    async fn load(&mut self, shard: ShardSpec) -> EngineResult<LoadStream> {
        if !(shard.is_first_stage && shard.is_last_stage) {
            return Err(EngineError::ShardRejected(
                "ov-genai requires --total 1".into(),
            ));
        }
        if self.draft_model_path.is_some() && self.prompt_lookup_ngram > 0 {
            return Err(EngineError::InvalidConfig(
                "--draft-model and --prompt-lookup are mutually exclusive (both set \
                 GenerationConfig.num_assistant_tokens). Use one."
                    .into(),
            ));
        }
        if self.continuous_batching.is_some()
            && (self.draft_model_path.is_some() || self.prompt_lookup_ngram > 0)
        {
            return Err(EngineError::InvalidConfig(
                "--cb is incompatible with --draft-model / --prompt-lookup: the \
                 continuous-batching scheduler owns batch composition. Drop one."
                    .into(),
            ));
        }

        if !PathBuf::from(&self.model_path).exists() {
            return Err(EngineError::ModelNotFound(self.model_path.clone()));
        }
        if let Some(dpath) = &self.draft_model_path {
            if !PathBuf::from(dpath).exists() {
                return Err(EngineError::ModelNotFound(dpath.clone()));
            }
        }

        check_tokenizer_irs(&self.model_path)?;
        if let Some(draft) = &self.draft_model_path {
            check_tokenizer_irs(draft)?;
        }

        let mut progress = vec![LoadProgress::message(format!(
            "locating IR at {}",
            self.model_path
        ))];

        let plugin = self.build_plugin_config();
        // VLM-layout export (Qwen3.5/3.6 etc.): language model + separate
        // embeddings IRs, no `openvino_model.xml`. Needs VLMPipeline —
        // LLMPipeline fails on it with "Port for tensor name input_ids
        // was not found". Served text-only.
        let model_dir = PathBuf::from(&self.model_path);
        let is_vlm_layout = model_dir.join("openvino_language_model.xml").exists()
            && !model_dir.join("openvino_model.xml").exists();
        if let Some(sched) = &self.continuous_batching {
            if is_vlm_layout {
                return Err(EngineError::InvalidConfig(
                    "--cb is not supported on VLM-layout exports; \
                     ContinuousBatchingPipeline needs a plain LM export"
                        .into(),
                ));
            }
            progress.push(LoadProgress::message(format!(
                "compiling ContinuousBatchingPipeline on {} (cache {} GB, max_num_seqs {}, \
                 max_batched_tokens {})",
                self.device, sched.cache_size_gb, sched.max_num_seqs, sched.max_num_batched_tokens,
            )));
            let pipe = CbPipeline::new(&self.model_path, &self.device, sched, &plugin)
                .map_err(map_ov_err)?;
            progress.push(LoadProgress::ready());
            self.cb_pipe = Some(pipe);
            return Ok(Box::pin(stream::iter(progress)));
        }
        let pipe = if is_vlm_layout {
            if self.draft_model_path.is_some() {
                return Err(EngineError::InvalidConfig(
                    "--draft-model is not supported on VLM-layout exports; \
                     use --prompt-lookup instead"
                        .into(),
                ));
            }
            let prompt_lookup = self.prompt_lookup_ngram > 0;
            progress.push(LoadProgress::message(format!(
                "VLM-layout export detected; compiling VLMPipeline{} on {}",
                if prompt_lookup {
                    " + prompt_lookup"
                } else {
                    ""
                },
                self.device
            )));
            LlmPipeline::vlm(&self.model_path, &self.device, prompt_lookup, &plugin)
                .map_err(map_ov_err)?
        } else if let Some(dpath) = &self.draft_model_path {
            progress.push(LoadProgress::message(format!(
                "loading draft model {dpath}"
            )));
            progress.push(LoadProgress::message(format!(
                "compiling LLMPipeline + draft on {}",
                self.device
            )));
            LlmPipeline::with_draft(
                &self.model_path,
                &self.device,
                dpath,
                self.draft_device.as_deref().unwrap_or(&self.device),
                &plugin,
            )
            .map_err(map_ov_err)?
        } else if self.prompt_lookup_ngram > 0 {
            progress.push(LoadProgress::message(format!(
                "compiling LLMPipeline + prompt_lookup (n={}) on {}",
                self.prompt_lookup_ngram, self.device
            )));
            LlmPipeline::with_prompt_lookup(&self.model_path, &self.device, &plugin)
                .map_err(map_ov_err)?
        } else {
            progress.push(LoadProgress::message(format!(
                "compiling LLMPipeline on {}",
                self.device
            )));
            LlmPipeline::new(&self.model_path, &self.device, &plugin).map_err(map_ov_err)?
        };
        progress.push(LoadProgress::ready());

        self.pipe = Some(pipe);
        Ok(Box::pin(stream::iter(progress)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        if let Some(pipe) = self.cb_pipe {
            return Ok(Box::new(OvGenaiCbEngine {
                active: Vec::new(),
                pipe: Box::new(pipe),
                device: self.device,
                prompt_pretemplated: self.prompt_pretemplated,
                next_request_id: 0,
                max_tokens_default: 256,
                idle_steps: 0,
                step_failures: 0,
                suspect_failures: 0,
                latched_at: None,
                blamed_in_run: HashMap::new(),
                canary_failed_in_run: false,
                structured_unavailable: None,
                quarantined: QuarantineSet::default(),
                strikes: StrikeSet::default(),
                compiled: QuarantineSet::default(),
                schema_keys: RandomState::new(),
                schema_budget: SchemaBudget::default(),
            }));
        }
        let pipe = self.pipe.ok_or(EngineError::NotLoaded)?;
        Ok(Box::new(OvGenaiEngine {
            pipe,
            device: self.device,
            draft_model: self.draft_model_path.is_some(),
            speculative_k: self.speculative_k,
            prompt_lookup_ngram: self.prompt_lookup_ngram,
            prompt_pretemplated: self.prompt_pretemplated,
            pending: Vec::new(),
            max_tokens_default: 256,
            structured_unavailable: None,
            schema_budget: SchemaBudget::default(),
        }))
    }
}

fn map_ov_err(err: OvError) -> EngineError {
    match err {
        OvError::Stub => {
            EngineError::Backend("openvino-genai shim built without the `openvino` feature".into())
        }
        OvError::Utf8(s) => EngineError::InvalidConfig(s),
        OvError::Native(s) => EngineError::Backend(s),
    }
}

/// The JSON schema `response_format: {"type":"json_object"}` maps to. It is
/// the server's own fixed schema, never a client's, so a failed batch is
/// never blamed on it: quarantining it would refuse every json_object
/// request.
const JSON_OBJECT_PROBE: &str = r#"{"type":"object","additionalProperties":true}"#;

/// Schema, prompt and token budget for the warmup structured-output probe.
/// The prompt does not ask for JSON and the schema allows exactly one
/// answer, so the expected output appears only when the constraint is
/// really applied. The answer is a handful of tokens; the rest of the budget
/// is room for the whitespace the grammar allows between them.
const PROBE_SCHEMA: &str = r#"{"type":"object","properties":{"k":{"const":"zq"}},"required":["k"],"additionalProperties":false}"#;
const PROBE_PROMPT: &str = "Say hello.";
const PROBE_MAX_TOKENS: u32 = 48;

/// Request id of the CB structured-output probe, and `- 1` of its retry (the
/// plain warmup uses `u64::MAX`); none can collide with the sequential
/// request-id counter.
const PROBE_REQUEST_ID: u64 = u64::MAX - 1;

/// How many schemas the CB engine remembers as having failed to compile.
const QUARANTINE_CAPACITY: usize = 256;
/// How long a schema that failed to compile is refused.
const QUARANTINE_TTL: Duration = Duration::from_secs(10 * 60);

/// The JSON schema a task's output must follow, if it carries one.
fn grammar_json_schema(task: &GenerationTask) -> Option<&str> {
    match &task.grammar {
        Some(g) => match g.kind {
            GrammarKind::JsonSchema => Some(g.body.as_str()),
        },
        None => None,
    }
}

/// The schema text GenAI is sent for `task`: the client's, with every
/// unbounded `integer` bounded to 18 digits
/// ([`cascadia_grammar::bound_integers`]), since xgrammar, like llguidance,
/// allows an integer of any length and a small model in JSON mode can repeat
/// digits until max_tokens. Key order is kept. A schema that is not JSON, or
/// has nothing to bound, goes as the client wrote it. The schema budget and
/// quarantine still key on the client's text.
fn genai_json_schema(task: &GenerationTask) -> Option<String> {
    let schema = grammar_json_schema(task)?;
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(schema) else {
        return Some(schema.to_owned());
    };
    match cascadia_grammar::bound_integers(&mut value) {
        0 => Some(schema.to_owned()),
        n if n > GENAI_MAX_BOUNDED_INTEGERS => {
            tracing::warn!(
                integers = n,
                "response_format integer length not bounded: the schema has more than \
                 {GENAI_MAX_BOUNDED_INTEGERS} integers to bound"
            );
            Some(schema.to_owned())
        }
        _ => Some(value.to_string()),
    }
}

/// Most integers [`genai_json_schema`] bounds in one schema. xgrammar
/// compiles each bounded integer's range on its own, a cost the xgrammar
/// guard doesn't price: 1000 required ones took 1.37 s against 0.10 s
/// unbounded, 200 took 0.17 s against 0.02 s (xgrammar 0.1.31, Qwen2.5
/// tokenizer, one thread). Past this the schema goes as the client wrote it,
/// as the llguidance engines do for a schema with too many integers.
const GENAI_MAX_BOUNDED_INTEGERS: usize = 256;

/// The message inside a shim error, without the shim's own prefix.
fn ov_message(err: OvError) -> String {
    match err {
        OvError::Native(m) => m,
        other => other.to_string(),
    }
}

/// Submit-time check of a task's output constraint: refuse it when this model
/// cannot do structured output, when xgrammar would take too long to compile
/// the schema (it compiles inside the request with no time limit, and a
/// worker stuck in a minutes-long compile answers nothing else), or when the
/// schema does not compile. Tasks without a grammar pass straight through.
fn check_grammar(
    task: &GenerationTask,
    unavailable: Option<&str>,
    validate: impl FnOnce(&str) -> Result<(), OvError>,
) -> EngineResult<()> {
    // The text GenAI will compile: priced and validated as sent.
    let Some(schema) = genai_json_schema(task) else {
        return Ok(());
    };
    let schema = schema.as_str();
    if let Some(reason) = unavailable {
        return Err(EngineError::InvalidConfig(format!(
            "structured output is not available for this model: {reason}"
        )));
    }
    // A schema that is not JSON is left to `validate` to report. (One nested
    // deeper than serde_json's limit of 128 can't be: the API parses each
    // request body with that limit.)
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(schema) {
        if let Some(msg) = cascadia_grammar::xgrammar_cost_error(&value) {
            return Err(EngineError::InvalidConfig(format!(
                "response_format json_schema: {msg}"
            )));
        }
    }
    validate(schema).map_err(|err| {
        EngineError::InvalidConfig(format!(
            "response_format json_schema: {}",
            client_message(err, "schema validation failed")
        ))
    })
}

/// The one line of a shim error a client may see: the backend's raw text,
/// logged here in full, reduced by [`clean_backend_message`].
fn client_message(err: OvError, what: &str) -> String {
    let raw = ov_message(err);
    let clean = clean_backend_message(&raw);
    if clean != raw {
        warn!(raw = %raw, "{what}");
    }
    clean
}

/// What a client of a failed CB step is told when its own schema is not
/// blamed for it. Carries no backend text: that text can name another
/// client's schema.
const CB_BATCH_FAILED: &str =
    "cb step failed: the batch this request ran in failed; retry the request";

/// What a client of a failed CB step is told when its schema is one of
/// several that may have caused it (the backend's reason may come from
/// another of them, so it is not passed on), or is the one suspect but the
/// reason does not read as a grammar error.
const CB_BATCH_SCHEMA_SUSPECT: &str =
    "cb step failed: a schema in this batch may have caused it; retry the request";

/// What the client of a failed CB step whose schema is the one blamed for
/// it is told when the reason reads as a grammar error: the backend's raw
/// text (logged in full by the caller) can carry build paths, so it is
/// reduced by [`clean_backend_message`].
fn step_failure_message(err: &OvError) -> String {
    let detail = match err {
        OvError::Native(m) => clean_backend_message(m),
        other => other.to_string(),
    };
    format!("cb step failed: {detail}")
}

/// The probe schema's one answer, without whitespace.
const PROBE_ANSWER: &str = r#"{"k":"zq"}"#;

/// `text` with the whitespace outside JSON strings removed.
fn strip_json_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let (mut in_string, mut escaped) = (false, false);
    for c in text.chars() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
            out.push(c);
        } else if !matches!(c, ' ' | '\t' | '\n' | '\r') {
            out.push(c);
        }
    }
    out
}

/// Decide the warmup structured-output probe. `None` means available: the
/// trimmed output is exactly the probe schema's one answer `{"k":"zq"}`, or
/// it is empty or a proper prefix of that answer (the grammar allows
/// whitespace between tokens, so the budget can end before the object is
/// complete: inconclusive, not a failure). `Some(reason)` means unavailable;
/// an error's text is cleaned for clients.
fn probe_verdict(r: Result<String, OvError>) -> Option<String> {
    match r {
        Ok(text) => {
            let t = text.trim();
            if t.is_empty() {
                info!("structured output probe inconclusive: no output within the token budget");
                return None;
            }
            let compact = strip_json_whitespace(t);
            if compact.len() < PROBE_ANSWER.len() && PROBE_ANSWER.starts_with(&compact) {
                info!(
                    output = %t,
                    "structured output probe inconclusive: the token budget ended inside the answer"
                );
                return None;
            }
            let want = serde_json::json!({"k": "zq"});
            if serde_json::from_str::<serde_json::Value>(t).is_ok_and(|v| v == want) {
                None
            } else {
                let head: String = t.chars().take(40).collect();
                Some(format!("the output constraint was not applied: {head}"))
            }
        }
        Err(err) => Some(client_message(err, "structured output probe failed")),
    }
}

/// Run the probe, and once more if it fails: one failed generation at
/// startup is not enough to turn structured output off for the run.
fn probe_with_retry(mut run: impl FnMut() -> Result<String, OvError>) -> Result<String, OvError> {
    match run() {
        Err(err) => {
            warn!(error = %err, "structured output probe failed; retrying once");
            run()
        }
        ok => ok,
    }
}

/// Whether a GenAI runtime build string (`2026.4.1.0-3408-…`) is older than
/// 2026.4. `None` when it does not start with `<major>.<minor>`.
fn runtime_older_than_2026_4(build: &str) -> Option<bool> {
    let head = build.split('-').next()?;
    let mut parts = head.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor) < (2026, 4))
}

/// Why structured output cannot run on this engine, decided without running
/// the warmup probe: a GenAI runtime older than 2026.4 (see
/// [`version_gate`]), an NPU device (see [`npu_gate`]) or a draft model (see
/// [`draft_model_gate`]). `None` means run the probe.
fn structured_gate(
    device: &str,
    draft_model: bool,
    version: Result<cascadia_ov_genai_shim::OvVersion, OvError>,
) -> Option<String> {
    version_gate(version)
        .or_else(|| npu_gate(device))
        .or_else(|| draft_model_gate(draft_model))
}

/// True when `device` is an NPU or lists one: `NPU`, `NPU.0`, or a
/// `HETERO:`/`AUTO:`/`MULTI:` list such as `HETERO:GPU,NPU`. Case-insensitive.
fn uses_npu(device: &str) -> bool {
    let list = device.split_once(':').map_or(device, |(_, list)| list);
    list.split(',')
        .any(|d| d.trim().to_ascii_uppercase().starts_with("NPU"))
}

/// Why structured output cannot run on an NPU device. On GenAI 2026.4.1 the
/// NPU pipeline lets tokens the grammar forbids through: a `json_object`
/// request returned raw newlines inside strings (invalid JSON, finish
/// `stop`) where the same model on GPU returned valid JSON. A probe can pass
/// by luck, so the device is refused outright.
fn npu_gate(device: &str) -> Option<String> {
    uses_npu(device).then(|| {
        "not reliable on NPU with this OpenVINO GenAI version (the output \
         constraint is not fully applied); use GPU or CPU for response_format"
            .to_owned()
    })
}

/// Why structured output cannot run on a stateful pipeline with a draft
/// model, without trying it. On GenAI 2026.4.1 a draft model's proposals
/// reach the grammar with token ids outside the main model's vocabulary
/// (xgrammar logs "token id ... is out of range") and the constrained output
/// can loop on whitespace to the token cap, while the same request without
/// the draft passes. A probe can pass by luck, so the pairing is refused
/// outright. Prompt lookup proposes tokens from the prompt itself and keeps
/// the probe. `None` (run the probe) without a draft model.
fn draft_model_gate(draft_model: bool) -> Option<String> {
    draft_model.then(|| {
        "not supported with a draft model (speculative decoding); \
         run without --draft-model for response_format"
            .to_owned()
    })
}

/// Why structured output cannot run on the loaded GenAI runtime, without
/// trying it: an output constraint on GenAI older than 2026.4 crashes the
/// process. `None` (run the probe) when the version is newer or unknown.
fn version_gate(v: Result<cascadia_ov_genai_shim::OvVersion, OvError>) -> Option<String> {
    let v = v.ok()?;
    runtime_older_than_2026_4(&v.runtime_genai)?.then(|| {
        format!(
            "needs OpenVINO GenAI 2026.4 or newer; this runtime is {}",
            v.runtime_genai
        )
    })
}

/// The stateful engine's finish reason. GenAI does not report one, so it is
/// inferred: `length` when the tokenizer's count of the decoded text reaches
/// the cap, else `stop`. For a request with an output constraint the
/// generated-token count is consulted too: re-tokenising constrained output
/// can undercount badly (a long run of newlines merges into a few tokens).
/// That count is 0 when the backend's performance metrics are unavailable,
/// which leaves the tokenizer to decide. A plain request decides on the
/// tokenizer count alone.
fn stateful_finish(
    tokenizer_count: u32,
    generated: u32,
    max_new: u32,
    has_grammar: bool,
) -> FinishReason {
    if tokenizer_count >= max_new || (has_grammar && generated >= max_new) {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

/// Whether a backend error reads as a grammar or schema compile error.
fn looks_like_grammar_error(raw: &str) -> bool {
    let raw = raw.to_ascii_lowercase();
    ["grammar", "json schema", "json_schema"]
        .iter()
        .any(|k| raw.contains(k))
}

/// The final chunk for a stateful `generate` that failed. When the request
/// carries a client's schema and the backend's error reads as a grammar or
/// schema error (the grammar is compiled inside `generate`), it is refused
/// as an invalid request with the backend's cleaned message. Any other error
/// (a device error, resource exhaustion), the server's own `json_object`
/// schema and requests without a schema keep the plain failure.
fn generate_failure_chunk(task: &GenerationTask, err: OvError, resource: bool) -> Chunk {
    let client_schema = grammar_json_schema(task).is_some_and(|s| s != JSON_OBJECT_PROBE);
    let grammar_error = match &err {
        OvError::Native(raw) => looks_like_grammar_error(raw),
        _ => false,
    };
    if client_schema && grammar_error && !resource {
        Chunk::rejected(
            task.task_id.clone(),
            format!(
                "response_format json_schema: {}",
                client_message(err, "generate failed with a schema")
            ),
        )
    } else {
        Chunk::error(task.task_id.clone(), err.to_string())
    }
}

/// Log the probe's outcome and turn it into the engine's
/// `structured_unavailable` value.
fn record_probe(verdict: Option<String>) -> Option<String> {
    match &verdict {
        Some(reason) => warn!(
            reason = %reason,
            "structured output (response_format) unavailable for this model"
        ),
        None => info!("structured output available"),
    }
    verdict
}

/// A set of schemas (by body hash): those that recently failed to compile on
/// the CB pipeline, or those that recently compiled. Bounded and
/// time-limited: the oldest entry is dropped past capacity, and an entry
/// stops counting after [`QUARANTINE_TTL`].
#[derive(Default)]
struct QuarantineSet {
    order: VecDeque<(u64, Instant)>,
    members: HashSet<u64>,
}

impl QuarantineSet {
    fn prune(&mut self, now: Instant) {
        // Entries are only ever appended with a non-decreasing time, so the
        // expired ones are all at the front.
        while let Some(&(h, at)) = self.order.front() {
            if now.saturating_duration_since(at) < QUARANTINE_TTL {
                break;
            }
            self.order.pop_front();
            self.members.remove(&h);
        }
    }

    fn contains(&mut self, h: u64, now: Instant) -> bool {
        self.prune(now);
        self.members.contains(&h)
    }

    fn insert(&mut self, h: u64, now: Instant) {
        self.prune(now);
        if self.members.contains(&h) {
            return;
        }
        if self.order.len() >= QUARANTINE_CAPACITY {
            if let Some((old, _)) = self.order.pop_front() {
                self.members.remove(&old);
            }
        }
        self.order.push_back((h, now));
        self.members.insert(h);
    }

    fn remove(&mut self, h: u64) {
        if self.members.remove(&h) {
            self.order.retain(|&(m, _)| m != h);
        }
    }

    /// Insert `h`, or restart its time if it is already in.
    fn refresh(&mut self, h: u64, now: Instant) {
        self.remove(h);
        self.insert(h, now);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.order.len()
    }
}

/// Strikes against schemas (by body hash) that were among several new
/// schemas in a failed CB step, so no single one could be blamed. Bounded
/// like [`QuarantineSet`]: an entry lapses [`QUARANTINE_TTL`] after its last
/// strike, and past [`QUARANTINE_CAPACITY`] the stalest one is dropped.
#[derive(Default)]
struct StrikeSet {
    strikes: HashMap<u64, (u32, Instant)>,
}

impl StrikeSet {
    /// Record a strike against `h`; returns its strikes within the window.
    fn strike(&mut self, h: u64, now: Instant) -> u32 {
        self.strikes
            .retain(|_, (_, at)| now.saturating_duration_since(*at) < QUARANTINE_TTL);
        if !self.strikes.contains_key(&h) && self.strikes.len() >= QUARANTINE_CAPACITY {
            if let Some(oldest) = self
                .strikes
                .iter()
                .min_by_key(|(_, (_, at))| *at)
                .map(|(k, _)| *k)
            {
                self.strikes.remove(&oldest);
            }
        }
        let entry = self.strikes.entry(h).or_insert((0, now));
        entry.0 += 1;
        entry.1 = now;
        entry.0
    }

    /// Drop the strikes against `h`.
    fn clear(&mut self, h: u64) {
        self.strikes.remove(&h);
    }

    /// Take back `n` of the strikes against `h`; the entry goes when none
    /// are left.
    fn take_back(&mut self, h: u64, n: u32) {
        if let Some((count, _)) = self.strikes.get_mut(&h) {
            *count = count.saturating_sub(n);
            if *count == 0 {
                self.strikes.remove(&h);
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.strikes.len()
    }
}

/// Strikes at which a schema seen in several shared failed steps is
/// quarantined.
const QUARANTINE_STRIKES: u32 = 2;

/// Most distinct `response_format` schemas one pipeline lets through to
/// GenAI. Its xgrammar backend keeps every compiled grammar for the
/// pipeline's life, with no size limit and no way to evict one: about
/// 30–50 MB each with a 151k-token vocabulary (process RSS, xgrammar 0.1.31),
/// so a client sending ever-new schemas would grow the worker until it runs
/// out of memory. 64 of them is about 2–3 GB.
const MAX_DISTINCT_SCHEMAS: usize = 64;

/// The distinct schemas let through to GenAI since the pipeline loaded (their
/// text, at most 64 of the API's 64 KiB): past [`MAX_DISTINCT_SCHEMAS`] a new
/// one is refused, one already let through still runs (GenAI compiles it
/// from its cache).
#[derive(Default)]
struct SchemaBudget {
    seen: HashSet<String>,
    /// New schemas refused since the budget filled.
    refused: u64,
    /// Times the budget-full warning was logged: once per pipeline.
    full_warnings: u32,
}

/// The schema [`SchemaBudget`] counts for `task`: a client's. json_object's
/// fixed schema is one grammar per pipeline, compiled once, so it neither
/// takes a slot nor is ever refused (the budget bounds the client schemas'
/// grammars, plus that one).
fn counted_schema(task: &GenerationTask) -> Option<&str> {
    grammar_json_schema(task).filter(|s| *s != JSON_OBJECT_PROBE)
}

impl SchemaBudget {
    /// Refuses `task` when its schema is new and the budget is full, and
    /// counts the refusal. Checked before the request is handed to GenAI;
    /// [`Self::record`] counts it once GenAI has taken it. The first refusal
    /// is logged at WARN (the budget is full for the pipeline's life), each
    /// one at DEBUG with the running count.
    fn check(&mut self, task: &GenerationTask) -> EngineResult<()> {
        let Some(schema) = counted_schema(task) else {
            return Ok(());
        };
        if self.seen.contains(schema) || self.seen.len() < MAX_DISTINCT_SCHEMAS {
            return Ok(());
        }
        self.refused += 1;
        if self.full_warnings == 0 {
            self.full_warnings += 1;
            tracing::warn!(
                distinct = self.seen.len(),
                limit = MAX_DISTINCT_SCHEMAS,
                "ov-genai: the response_format schema budget is full: new schemas are refused \
                 until the worker restarts (GenAI keeps every compiled grammar); schemas already \
                 served keep working. Further refusals are logged at debug"
            );
        }
        tracing::debug!(
            refused = self.refused,
            limit = MAX_DISTINCT_SCHEMAS,
            "ov-genai: refused a new response_format schema: the schema budget is full"
        );
        Err(EngineError::InvalidConfig(format!(
            "structured output is not available for this model: this worker has reached its \
             limit of {MAX_DISTINCT_SCHEMAS} distinct response_format schemas per pipeline \
             (GenAI keeps each compiled grammar until the worker restarts); schemas it has \
             already served keep working, and restarting the worker clears the limit"
        )))
    }

    /// Counts `task`'s schema as let through to GenAI.
    fn record(&mut self, task: &GenerationTask) {
        if let Some(schema) = counted_schema(task) {
            if !self.seen.contains(schema) {
                self.seen.insert(schema.to_owned());
            }
        }
    }
}

pub struct OvGenaiEngine {
    pipe: LlmPipeline,
    /// The OpenVINO device string the pipeline was compiled for.
    device: String,
    /// Built with a draft model (speculative decoding), as opposed to prompt
    /// lookup, which also sets `num_assistant_tokens`.
    draft_model: bool,
    speculative_k: u32,
    prompt_lookup_ngram: u32,
    prompt_pretemplated: bool,
    pending: Vec<GenerationTask>,
    max_tokens_default: u32,
    /// Set by the warmup probe when this model cannot do structured output;
    /// tasks with a grammar are then refused at submit with this reason.
    structured_unavailable: Option<String>,
    schema_budget: SchemaBudget,
}

impl Engine for OvGenaiEngine {
    fn warmup(&mut self) {
        let cfg = self.gen_config(
            /*max_tokens*/ 4,
            /*temperature*/ 0.0,
            /*enable_thinking*/ false,
            &SamplingParams::default(),
            None,
        );
        match self.pipe.generate("Hi", &cfg) {
            Ok(_) => info!(
                spec_k = self.speculative_k,
                prompt_lookup = self.prompt_lookup_ngram,
                "ov-genai warmup ok"
            ),
            Err(err) => warn!(error = %err, "ov-genai warmup failed"),
        }

        // Structured-output probe: one real generation under the probe
        // schema, through this engine's own config (prompt lookup included),
        // so a model or pipeline that cannot run with an output constraint
        // is found here rather than per request. A draft model skips it: see
        // `draft_model_gate`.
        if let Some(reason) = structured_gate(
            &self.device,
            self.draft_model,
            cascadia_ov_genai_shim::ov_version(),
        ) {
            self.structured_unavailable = record_probe(Some(reason));
            return;
        }
        let cfg = self.gen_config(
            PROBE_MAX_TOKENS,
            0.0,
            false,
            &SamplingParams::default(),
            Some(PROBE_SCHEMA.to_owned()),
        );
        let result = probe_with_retry(|| {
            self.pipe.generate(PROBE_PROMPT, &cfg).map(|r| {
                match r.text.strip_prefix(PROBE_PROMPT) {
                    Some(rest) => rest.to_owned(),
                    None => r.text,
                }
            })
        });
        self.structured_unavailable = record_probe(probe_verdict(result));
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        // No forced-prefix path in the ov-genai pipelines: accepting a resumed task
        // would regenerate from scratch and report success on a corrupt "resume".
        // The enterprise shard backend declines these pre-submit (its
        // shard_is_ov_genai guard); this is the engine-side backstop for any
        // caller that misses that guard.
        if task.resume_ids().is_some() {
            return Err(EngineError::Backend(
                "resume_unsupported: ov-genai has no forced-prefix path".to_string(),
            ));
        }
        if self.pending.iter().any(|t| t.task_id == task.task_id) {
            return Ok(());
        }
        if self.pending.len() >= crate::dist_spec::MAX_PENDING_TASKS {
            tracing::warn!(
                queued = self.pending.len(),
                cap = crate::dist_spec::MAX_PENDING_TASKS,
                "ov-genai: pending queue at cap; rejecting task"
            );
            return Err(EngineError::QueueFull {
                queued: self.pending.len(),
                cap: crate::dist_spec::MAX_PENDING_TASKS,
            });
        }
        check_grammar(&task, self.structured_unavailable.as_deref(), |s| {
            self.pipe.validate_json_schema(s)
        })?;
        self.schema_budget.check(&task)?;
        self.schema_budget.record(&task);
        self.pending.push(task);
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Monolithic step(): a generation already running holds the engine
        // mutex for its whole duration, so cancel() (also &mut self) can only
        // run between tasks — it drops queued tasks before they start.
        // Interrupting an in-flight ov-genai generation needs a StreamerBase
        // callback returning CANCEL (C++ shim work) — tracked as a follow-up.
        self.pending.retain(|t| &t.task_id != task_id);
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let task = self.pending.remove(0);
        let max_new = if task.max_tokens > 0 {
            task.max_tokens
        } else {
            self.max_tokens_default
        };
        let cfg = self.gen_config(
            max_new,
            task.temperature,
            task.enable_thinking,
            &task.sampling,
            genai_json_schema(&task),
        );

        let started = Instant::now();
        let result = match self.pipe.generate(&task.prompt, &cfg) {
            Ok(r) => r,
            Err(err) => {
                warn!(task = %task.task_id, error = %err, "generate failed");
                // An error chunk, not an empty final_marker: the latter reads
                // as a SUCCESSFUL empty completion, so the client got a 200
                // with no content and cascadia_tasks_failed_total never
                // moved — the failure was booked as finish_reason="stop" with
                // zero tokens. Same class as the poisoned-head bug guarded in
                // qwen36.rs. This is the default engine, so it was the common
                // failure that the failure metric could not see.
                let resource = cascadia_ov_genai_shim::last_error_resource_exhausted();
                let final_chunk = generate_failure_chunk(&task, err, resource);
                return Ok(vec![(task.task_id, final_chunk)]);
            }
        };
        let elapsed = started.elapsed().as_secs_f64();

        let mut text = result.text;
        if let Some(stripped) = text.strip_prefix(&task.prompt) {
            text = stripped.trim_start().to_string();
        }
        // Prefer tokenizer-based count for usage; fall back to perf_metrics value.
        let tokens = self
            .pipe
            .count_tokens(&text)
            .unwrap_or(result.generated_tokens);
        let tok_s = if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        };
        info!(
            task = %task.task_id,
            tokens,
            elapsed_s = elapsed,
            tok_s,
            "task done"
        );

        // Carry the real token counts on the final chunk so the API's
        // `usage` block is populated (issue #55 — previously always 0 for
        // this engine because the whole response lands on one chunk with no
        // per-token count). `prompt_tokens` is the rendered-prompt token
        // count; it is exact when the API pre-templated the prompt
        // (`skip_chat_template`) and a slight undercount when ov-genai
        // applies its own template internally (the template wrapping isn't
        // counted). None when the tokenizer is unavailable.
        // finish_reason: ov-genai's GenResult doesn't report why it stopped,
        // so infer — reaching the token cap is `length`, anything short of it
        // (EOS / stop) is `stop`. Exact for the common case; an EOS landing
        // exactly at the cap reports `length` (OpenAI is also ambiguous here).
        let finish = stateful_finish(
            tokens,
            result.generated_tokens,
            max_new,
            task.grammar.is_some(),
        );
        let mut chunk = Chunk::final_marker(task.task_id.clone(), text)
            .with_n_tokens(tokens)
            .with_finish_reason(finish);
        if let Some(prompt_tokens) = self.pipe.count_tokens(&task.prompt) {
            chunk = chunk.with_prompt_tokens(prompt_tokens);
        }
        Ok(vec![(task.task_id, chunk)])
    }
}

/// Scheduler iterations the CB warmup will drive before giving up. The warmup
/// request asks for 4 tokens, so a healthy pipeline terminates in a handful.
const WARMUP_MAX_STEPS: usize = 64;

/// Consecutive scheduler iterations in which NO in-flight request produced a
/// token before the engine declares the batch stalled.
///
/// The liveness chunk emitted on such an iteration (see [`OvGenaiCbEngine`])
/// deliberately hides them from the runner's no-progress guard, so this is the
/// engine's own replacement for it — without a bound, a genuinely wedged
/// pipeline that keeps returning success would stream empty chunks until every
/// request hit max_tokens.
///
/// The counter only advances while the WHOLE batch is quiet (any request
/// producing a token resets it), so the largest legitimate run is one request's
/// chunked prefill: ceil(prompt_tokens / max_num_batched_tokens). At the
/// ov-genai default 256-token window this covers a ~512K-token prompt, which is
/// far beyond any supported context, while still terminating.
const MAX_IDLE_STEPS: u32 = 2048;

/// Consecutive `pipe.step()` failures after which the CB pipeline is treated as
/// unusable and new work is refused at `submit()`. Low, because a scheduler
/// iteration failing repeatedly means the device or the paged-attention cache
/// is gone, not that one request was unlucky.
const MAX_STEP_FAILURES: u32 = 3;

/// Step failures since the last successful step that are blamed on new
/// schemas. A step that fails with a request whose schema has not yet
/// produced output in flight is not counted toward [`MAX_STEP_FAILURES`],
/// whatever else shares the step, and that schema is refused after its first
/// or second such failure. At this many the engine checks the device itself
/// with a canary (see [`OvGenaiCbEngine::run_canary`]): if the canary
/// succeeds the schemas keep the blame and the count starts over; if it fails
/// the step counts.
const MAX_SUSPECT_FAILURES: u32 = 8;

/// How long a latched CB pipeline waits after its last canary before the next
/// request runs another to see whether it recovered. Client requests are
/// refused until one succeeds.
const LATCH_RETRY: Duration = Duration::from_secs(30);

/// Token budget of the canary request.
const CANARY_MAX_TOKENS: u32 = 4;

/// Whether a latched pipeline's next canary is due at `now`, the last one (or
/// the failure that latched it) having run at `last`.
fn canary_due(last: Option<Instant>, now: Instant) -> bool {
    last.is_some_and(|at| now.saturating_duration_since(at) >= LATCH_RETRY)
}

/// The slice of [`CbPipeline`] the continuous-batching engine actually uses.
///
/// Exists so the engine's state machine is testable without OpenVINO. The real
/// implementation is the FFI type; tests drive a scripted mock. Without it the
/// engine is untestable in a stub build — `CbPipeline::new` returns
/// `Error::Stub`, and every method on a stub pipeline errors, so `step()` can
/// only ever reach its failure branch.
trait CbPipe: Send {
    fn add_request(
        &self,
        request_id: u64,
        prompt: &str,
        cfg: &GenConfig,
    ) -> Result<Box<dyn CbReq>, OvError>;
    fn step(&self) -> Result<(), OvError>;
    fn count_tokens(&self, text: &str) -> Option<u32>;
    /// Must only be called from the engine's own `&mut self` methods: the
    /// tokenizer's structured-output state is not safe to touch while a step
    /// or add_request runs on the same pipeline.
    fn validate_json_schema(&self, schema: &str) -> Result<(), OvError>;
}

/// One enrolled request, as the engine sees it.
trait CbReq: Send {
    fn read(&mut self) -> Result<CbRead, OvError>;
    fn cancel(&self) -> Result<(), OvError>;
}

impl CbPipe for CbPipeline {
    fn add_request(
        &self,
        request_id: u64,
        prompt: &str,
        cfg: &GenConfig,
    ) -> Result<Box<dyn CbReq>, OvError> {
        Ok(Box::new(CbPipeline::add_request(
            self, request_id, prompt, cfg,
        )?))
    }
    fn step(&self) -> Result<(), OvError> {
        CbPipeline::step(self)
    }
    fn count_tokens(&self, text: &str) -> Option<u32> {
        CbPipeline::count_tokens(self, text)
    }
    fn validate_json_schema(&self, schema: &str) -> Result<(), OvError> {
        CbPipeline::validate_json_schema(self, schema)
    }
}

impl CbReq for CbHandle {
    fn read(&mut self) -> Result<CbRead, OvError> {
        CbHandle::read(self)
    }
    fn cancel(&self) -> Result<(), OvError> {
        CbHandle::cancel(self)
    }
}

/// Per-request generation state on the continuous-batching engine.
struct ActiveCbTask {
    task_id: TaskId,
    handle: Box<dyn CbReq>,
    total_tokens: u32,
    max_tokens: u32,
    prompt_tokens: Option<u32>,
    started: Instant,
    /// Hash of the task's JSON schema; `None` without one and for the
    /// json_object schema (the warmup probe already compiled that).
    grammar_hash: Option<u64>,
    /// False until a read returned at least one new token.
    produced: bool,
}

/// Continuous-batching ov-genai engine (#20). Unlike [`OvGenaiEngine`]'s
/// monolithic step (one whole generation per call), every `step()` advances
/// the batch by one scheduler iteration and emits one text delta per request
/// that progressed — the runner's per-task chunk buffers route each delta to
/// its owner. Which enrolled requests actually run in a given iteration is the
/// scheduler's choice, bounded by `max_num_seqs` / `max_num_batched_tokens`
/// and KV-cache pressure; a request may sit out a step entirely. `submit()`
/// enrolls into the running batch immediately, so a request arriving
/// mid-generation joins the next iteration instead of waiting for the batch
/// to drain.
///
/// A step that progresses the batch without producing any token — every
/// iteration of a chunked prefill does this — still emits a zero-token chunk.
/// An empty Vec from `step()` means "nothing in flight", and the runner's
/// no-progress guard relies on that distinction.
pub struct OvGenaiCbEngine {
    active: Vec<ActiveCbTask>,
    pipe: Box<dyn CbPipe>,
    /// The OpenVINO device string the pipeline was compiled for.
    device: String,
    prompt_pretemplated: bool,
    next_request_id: u64,
    max_tokens_default: u32,
    /// Consecutive iterations in which the whole batch produced nothing.
    idle_steps: u32,
    /// Counted `pipe.step()` failures since the last successful step or
    /// canary. At MAX_STEP_FAILURES the pipeline is treated as unusable and
    /// `submit()` admits no client work until a canary succeeds.
    step_failures: u32,
    /// Step failures blamed on new schemas since the last successful step
    /// or canary. At [`MAX_SUSPECT_FAILURES`] a canary decides.
    suspect_failures: u32,
    /// With `step_failures` at the limit: when the latch was set or its last
    /// canary ran. The next canary runs [`LATCH_RETRY`] after it.
    latched_at: Option<Instant>,
    /// Schemas quarantined or struck during the current run of failures, with
    /// the strikes each got in it. They are released (and only those strikes
    /// taken back) if the run turns out to have been the device's fault.
    blamed_in_run: HashMap<u64, u32>,
    /// A canary failed during the current run of failures.
    canary_failed_in_run: bool,
    /// Set by the warmup probe when this model cannot do structured output.
    structured_unavailable: Option<String>,
    /// Schemas that recently took a scheduler iteration down with them.
    quarantined: QuarantineSet,
    /// Strikes against schemas that shared a failed step with others.
    strikes: StrikeSet,
    /// Schemas that recently produced output, so compiled: never a suspect
    /// in a failed step.
    compiled: QuarantineSet,
    /// Random keys for [`Self::schema_hash`], drawn once per engine.
    schema_keys: RandomState,
    schema_budget: SchemaBudget,
}

impl Engine for OvGenaiCbEngine {
    fn warmup(&mut self) {
        let cfg = build_gen_config(
            0,
            0,
            self.prompt_pretemplated,
            /*max_tokens*/ 4,
            /*temperature*/ 0.0,
            /*enable_thinking*/ false,
            &SamplingParams::default(),
            None,
        );
        // u64::MAX cannot collide with the sequential request-id counter.
        match self.pipe.add_request(u64::MAX, "Hi", &cfg) {
            Ok(mut handle) => {
                // Only a terminal status means the pipeline actually served a
                // request. Reporting "ok" after a failed step told operators
                // the worker was healthy when it was not, and dropped the
                // native OV message that says why.
                let mut reached_terminal = false;
                for _ in 0..WARMUP_MAX_STEPS {
                    if let Err(err) = self.pipe.step() {
                        warn!(error = %err, "cb warmup step failed");
                        break;
                    }
                    match handle.read() {
                        Ok(r) if r.status != CbStatus::Running => {
                            reached_terminal = true;
                            break;
                        }
                        Ok(_) => {}
                        Err(err) => {
                            warn!(error = %err, "cb warmup read failed");
                            break;
                        }
                    }
                }
                if reached_terminal {
                    info!("ov-genai continuous-batching warmup ok");
                } else {
                    warn!(
                        max_steps = WARMUP_MAX_STEPS,
                        "cb warmup never reached a terminal status; pipeline may be unhealthy"
                    );
                }
            }
            Err(err) => warn!(error = %err, "ov-genai cb warmup failed"),
        }

        // `--cb` refuses a draft model at load, so only the device and the
        // runtime decide here.
        if let Some(reason) =
            structured_gate(&self.device, false, cascadia_ov_genai_shim::ov_version())
        {
            self.structured_unavailable = record_probe(Some(reason));
            return;
        }
        let mut request_id = PROBE_REQUEST_ID;
        let result = probe_with_retry(|| {
            let r = self.run_structured_probe(request_id);
            request_id -= 1;
            r
        });
        self.structured_unavailable = record_probe(probe_verdict(result));
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        // No forced-prefix path in the ov-genai pipelines: accepting a resumed task
        // would regenerate from scratch and report success on a corrupt "resume".
        // The enterprise shard backend declines these pre-submit (its
        // shard_is_ov_genai guard); this is the engine-side backstop for any
        // caller that misses that guard.
        if task.resume_ids().is_some() {
            return Err(EngineError::Backend(
                "resume_unsupported: ov-genai has no forced-prefix path".to_string(),
            ));
        }
        if self.active.iter().any(|t| t.task_id == task.task_id) {
            return Ok(());
        }
        // A pipeline that has failed every recent scheduler iteration will fail
        // this request too. Admitting it means paying add_request and a step
        // before returning the same 503, forever, with nothing distinguishing a
        // transient hiccup from a dead device. Refuse up front instead. No
        // client request is admitted while latched: a request arriving
        // LATCH_RETRY or more after the last check runs an internal canary
        // first, and only its success clears this.
        if self.step_failures >= MAX_STEP_FAILURES {
            let now = Instant::now();
            let recovered = canary_due(self.latched_at, now) && {
                self.latched_at = Some(now);
                self.run_canary()
            };
            if !recovered {
                return Err(EngineError::Backend(format!(
                    "cb pipeline unusable: {} counted scheduler failures since the last \
                     successful step; it is checked again on a request {} s or more after \
                     the last check",
                    self.step_failures,
                    LATCH_RETRY.as_secs()
                )));
            }
            info!("cb pipeline recovered: the health check succeeded");
            self.end_failure_run(true);
        }
        if self.active.len() >= crate::dist_spec::MAX_PENDING_TASKS {
            tracing::warn!(
                queued = self.active.len(),
                cap = crate::dist_spec::MAX_PENDING_TASKS,
                "ov-genai cb: pending queue at cap; rejecting task"
            );
            return Err(EngineError::QueueFull {
                queued: self.active.len(),
                cap: crate::dist_spec::MAX_PENDING_TASKS,
            });
        }
        let schema = grammar_json_schema(&task);
        let grammar_hash = schema
            .filter(|s| *s != JSON_OBJECT_PROBE)
            .map(|s| self.schema_hash(s));
        if let Some(h) = grammar_hash {
            if self.quarantined.contains(h, Instant::now()) {
                return Err(EngineError::InvalidConfig(
                    "response_format json_schema: this schema failed to compile on this \
                     server; retry later or change the schema"
                        .to_string(),
                ));
            }
        }
        check_grammar(&task, self.structured_unavailable.as_deref(), |s| {
            self.pipe.validate_json_schema(s)
        })?;
        self.schema_budget.check(&task)?;
        let max_tokens = if task.max_tokens > 0 {
            task.max_tokens
        } else {
            self.max_tokens_default
        };
        let cfg = build_gen_config(
            0,
            0,
            self.prompt_pretemplated,
            max_tokens,
            task.temperature,
            task.enable_thinking,
            &task.sampling,
            genai_json_schema(&task),
        );
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let handle = self
            .pipe
            .add_request(request_id, &task.prompt, &cfg)
            .map_err(map_ov_err)?;
        self.schema_budget.record(&task);
        self.active.push(ActiveCbTask {
            prompt_tokens: self.pipe.count_tokens(&task.prompt),
            task_id: task.task_id,
            handle,
            total_tokens: 0,
            max_tokens,
            started: Instant::now(),
            grammar_hash,
            produced: false,
        });
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Per-request abort — the CB scheduler frees the request's KV blocks
        // on the next iteration; every other in-flight request is untouched.
        // (The monolithic engine can only drop queued tasks; see
        // OvGenaiEngine::cancel.)
        if let Some(idx) = self.active.iter().position(|t| &t.task_id == task_id) {
            let t = self.active.swap_remove(idx);
            if let Err(err) = t.handle.cancel() {
                warn!(task = %task_id, error = %err, "cb cancel failed");
            }
        }
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.active.is_empty() {
            return Ok(Vec::new());
        }
        if let Err(err) = self.pipe.step() {
            // A failed scheduler iteration poisons every in-flight request:
            // fail them all explicitly rather than spinning on a broken batch.
            // Each drained task still gets its own error chunk — that per-task
            // routing is why this returns Ok rather than an engine-level Err.
            //
            // OV compiles a request's grammar on its first sampled step, so a
            // step that fails while such a request is new may have failed on
            // that compile. Its schema is a suspect: the one suspect in a
            // step is refused for a while, and with several there is no
            // telling which one failed, so each gets a strike and a schema
            // that keeps turning up in failed steps is refused anyway
            // (pairing a failing schema with another does not dodge it).
            // Resource exhaustion and json_object (the server's own schema)
            // are never blamed on one.
            //
            // A step with a suspect in flight is not counted toward the
            // failure latch, whatever shares the step, so a client cannot
            // latch the worker with bad schemas. Client work never decides
            // whether the device is healthy: after MAX_SUSPECT_FAILURES such
            // steps an internal canary does. A step with no suspect counts as
            // any other failure.
            //
            // A schema that already produced output (in this batch or
            // recently) compiled, so it is never a suspect.
            let resource = cascadia_ov_genai_shim::last_error_resource_exhausted();
            let mut suspects: Vec<u64> = if resource {
                Vec::new()
            } else {
                let now = Instant::now();
                let streaming: HashSet<u64> = self
                    .active
                    .iter()
                    .filter(|t| t.produced)
                    .filter_map(|t| t.grammar_hash)
                    .collect();
                let mut new: Vec<u64> = self
                    .active
                    .iter()
                    .filter(|t| !t.produced)
                    .filter_map(|t| t.grammar_hash)
                    .filter(|h| !streaming.contains(h))
                    .collect();
                new.retain(|h| !self.compiled.contains(*h, now));
                new
            };
            // Several requests with one schema are one suspect.
            suspects.sort_unstable();
            suspects.dedup();
            let drained: Vec<ActiveCbTask> = self.active.drain(..).collect();
            for t in &drained {
                // The canary below must run on its own.
                let _ = t.handle.cancel();
            }
            let excused = !suspects.is_empty() && {
                self.suspect_failures += 1;
                self.suspect_failures < MAX_SUSPECT_FAILURES || {
                    let healthy = self.run_canary();
                    if healthy {
                        // A canary that failed earlier in this run means the
                        // device, not the schemas, failed those steps.
                        self.end_failure_run(self.canary_failed_in_run);
                    } else {
                        self.canary_failed_in_run = true;
                    }
                    healthy
                }
            };
            let mut refused = 0;
            if excused {
                let now = Instant::now();
                if let [h] = suspects[..] {
                    self.quarantined.insert(h, now);
                    self.blamed_in_run.entry(h).or_insert(0);
                    refused = 1;
                } else {
                    for h in &suspects {
                        if self.strikes.strike(*h, now) >= QUARANTINE_STRIKES {
                            self.quarantined.insert(*h, now);
                            refused += 1;
                        }
                        *self.blamed_in_run.entry(*h).or_insert(0) += 1;
                    }
                }
            } else {
                self.step_failures += 1;
                if self.step_failures >= MAX_STEP_FAILURES {
                    self.latched_at = Some(Instant::now());
                }
            }
            // Only the client(s) of the one schema blamed for the step see
            // the backend's reason (cleaned), and only when it reads as a
            // grammar or schema error: any other reason (a device error) is
            // not about their schema. With several suspects the reason may
            // come from any of them, and anyone else's request did not cause
            // it, so they get a fixed text: the reason can name another
            // client's schema.
            let grammar_error =
                matches!(&err, OvError::Native(raw) if looks_like_grammar_error(raw));
            let failed: Vec<(TaskId, Chunk)> = drained
                .into_iter()
                .map(|t| {
                    let suspect = excused && t.grammar_hash.is_some_and(|h| suspects.contains(&h));
                    let msg = match (suspect, suspects.len()) {
                        (true, 1) if grammar_error => step_failure_message(&err),
                        (true, _) => CB_BATCH_SCHEMA_SUSPECT.to_string(),
                        (false, _) => CB_BATCH_FAILED.to_string(),
                    };
                    (t.task_id.clone(), Chunk::error(t.task_id, msg))
                })
                .collect();
            if !suspects.is_empty() {
                warn!(
                    schemas = suspects.len(),
                    refused,
                    counted = !excused,
                    suspect_failures = self.suspect_failures,
                    "cb step failed with new schema-constrained requests in the batch"
                );
            }
            warn!(
                error = %err,
                in_flight = failed.len(),
                counted = self.step_failures,
                "cb step failed"
            );
            return Ok(failed);
        }
        // A canary that failed in this run means the device, not the schemas,
        // failed those steps.
        self.end_failure_run(self.canary_failed_in_run);

        let mut out = Vec::new();
        let mut idx = 0;
        while idx < self.active.len() {
            let read = match self.active[idx].handle.read() {
                Ok(r) => r,
                Err(err) => {
                    let t = self.active.swap_remove(idx);
                    warn!(task = %t.task_id, error = %err, "cb read failed");
                    let chunk = Chunk::error(t.task_id.clone(), format!("cb read failed: {err}"));
                    out.push((t.task_id, chunk));
                    continue;
                }
            };
            if read.resynced {
                warn!(
                    task = %self.active[idx].task_id,
                    "cb detokenizer re-decode stopped extending the emitted text; \
                     re-anchored (a short run of output may be missing)"
                );
            }
            self.active[idx].total_tokens += read.new_tokens;
            if read.new_tokens > 0 && !self.active[idx].produced {
                self.active[idx].produced = true;
                // Output means the schema compiled, so it did not cause an
                // earlier failed step it shared with others, and is not a
                // suspect in a later one.
                if let Some(h) = self.active[idx].grammar_hash {
                    self.strikes.clear(h);
                    self.quarantined.remove(h);
                    self.compiled.refresh(h, Instant::now());
                }
            }
            match read.status {
                CbStatus::Running => {
                    // Every chunk carries ITS OWN token increment in n_tokens
                    // — the API sums per-chunk counts, so a cumulative total
                    // here double-counts (measured 2N-1 on-HW). Emit even
                    // when the text delta is empty (UTF-8 hold-back) so those
                    // tokens are still counted.
                    if !read.text_delta.is_empty() || read.new_tokens > 0 {
                        let t = &self.active[idx];
                        out.push((
                            t.task_id.clone(),
                            Chunk::token(t.task_id.clone(), 0, read.text_delta)
                                .with_n_tokens(read.new_tokens),
                        ));
                    }
                    idx += 1;
                }
                CbStatus::Ignored => {
                    // OpenVINO ran out of KV cache and abandoned the request.
                    // Finalising it like a normal completion handed the client
                    // HTTP 200, finish_reason "stop" and a truncated body —
                    // indistinguishable from a model that chose to stop. Fail
                    // it loudly and name the knobs that fix it.
                    let t = self.active.swap_remove(idx);
                    warn!(
                        task = %t.task_id,
                        tokens = t.total_tokens,
                        in_flight = self.active.len(),
                        "cb scheduler dropped a request: KV cache exhausted"
                    );
                    let chunk = Chunk::error(
                        t.task_id.clone(),
                        format!(
                            "continuous-batching scheduler dropped this request after {} \
                             tokens: KV cache exhausted. Raise --cb-cache-size or lower \
                             --cb-max-num-seqs.",
                            t.total_tokens
                        ),
                    );
                    out.push((t.task_id, chunk));
                }
                CbStatus::Finished | CbStatus::Cancelled => {
                    let t = self.active.swap_remove(idx);
                    let elapsed = t.started.elapsed().as_secs_f64();
                    let tok_s = if elapsed > 0.0 {
                        t.total_tokens as f64 / elapsed
                    } else {
                        0.0
                    };
                    info!(
                        task = %t.task_id,
                        tokens = t.total_tokens,
                        elapsed_s = elapsed,
                        tok_s,
                        in_flight = self.active.len(),
                        "cb task done"
                    );
                    // Prefer OpenVINO's own reason. It reports NONE when the
                    // terminal read carried no new output, so keep the
                    // token-count inference as the fallback — it is what the
                    // monolithic path uses and is right for the common case.
                    let finish = match (read.status, read.finish) {
                        (CbStatus::Cancelled, _) => FinishReason::Cancelled,
                        (_, CbFinish::Stop) => FinishReason::Stop,
                        (_, CbFinish::Length) => FinishReason::Length,
                        (_, CbFinish::Unknown) if t.total_tokens >= t.max_tokens => {
                            FinishReason::Length
                        }
                        (_, CbFinish::Unknown) => FinishReason::Stop,
                    };
                    // n_tokens is this chunk's increment, not the cumulative
                    // total — interim chunks already carried theirs.
                    let mut chunk = Chunk::final_marker(t.task_id.clone(), read.text_delta)
                        .with_n_tokens(read.new_tokens)
                        .with_finish_reason(finish);
                    if let Some(prompt_tokens) = t.prompt_tokens {
                        chunk = chunk.with_prompt_tokens(prompt_tokens);
                    }
                    out.push((t.task_id, chunk));
                }
            }
        }
        // Liveness. Every other engine returns an empty Vec only when nothing
        // is in flight, so the runner reads three consecutive empty steps as a
        // wedged engine and fails the stream (MAX_CONSECUTIVE_EMPTY_STEPS,
        // cascadia-runner). A CB prefill legitimately spans
        // ceil(prompt_tokens / max_num_batched_tokens) iterations that produce
        // no token for ANY request, so without this a prompt longer than ~3x
        // the batch window is killed mid-prefill before its first token
        // (reproduced on-HW: 1211 tokens at the default 256-token window).
        // One chunk is enough — any non-empty step resets the counter for
        // every stream. n_tokens MUST be Some(0): the SSE path does
        // `n_tokens.unwrap_or(1)` and would otherwise inflate reported usage.
        //
        // Hiding these iterations from the runner's guard means the engine owes
        // it a replacement, or a truly wedged pipeline would heartbeat until
        // every request hit max_tokens: MAX_IDLE_STEPS bounds the quiet run and
        // fails the batch loudly past it.
        if out.is_empty() && !self.active.is_empty() {
            self.idle_steps += 1;
            if self.idle_steps > MAX_IDLE_STEPS {
                warn!(
                    idle_steps = self.idle_steps,
                    in_flight = self.active.len(),
                    "cb pipeline produced no tokens for {MAX_IDLE_STEPS} consecutive \
                     iterations; failing in-flight requests"
                );
                self.idle_steps = 0;
                return Ok(self
                    .active
                    .drain(..)
                    .map(|t| {
                        let chunk = Chunk::error(
                            t.task_id.clone(),
                            format!(
                                "cb pipeline stalled: no tokens produced for \
                                 {MAX_IDLE_STEPS} scheduler iterations"
                            ),
                        );
                        (t.task_id, chunk)
                    })
                    .collect());
            }
            let t = &self.active[0];
            out.push((
                t.task_id.clone(),
                Chunk::token(t.task_id.clone(), 0, String::new()).with_n_tokens(0),
            ));
        } else {
            self.idle_steps = 0;
        }
        Ok(out)
    }
}

impl OvGenaiCbEngine {
    /// Check the device with an internal canary: a short plain request (no
    /// schema), run on its own, as the warmup does. True when it finishes.
    /// Only ever called with nothing else in flight.
    fn run_canary(&mut self) -> bool {
        let cfg = build_gen_config(
            0,
            0,
            self.prompt_pretemplated,
            CANARY_MAX_TOKENS,
            0.0,
            false,
            &SamplingParams::default(),
            None,
        );
        // A sequential id: a canary abandoned by a failed step can never
        // collide with the next one.
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let mut handle = match self.pipe.add_request(request_id, "Hi", &cfg) {
            Ok(h) => h,
            Err(err) => {
                warn!(error = %err, "cb health check could not start");
                return false;
            }
        };
        for _ in 0..WARMUP_MAX_STEPS {
            if let Err(err) = self.pipe.step() {
                warn!(error = %err, "cb health check step failed");
                let _ = handle.cancel();
                return false;
            }
            match handle.read() {
                Ok(r) if r.status == CbStatus::Running => {}
                Ok(r) if r.status == CbStatus::Finished => return true,
                Ok(r) => {
                    warn!(status = ?r.status, "cb health check did not finish");
                    return false;
                }
                Err(err) => {
                    warn!(error = %err, "cb health check read failed");
                    let _ = handle.cancel();
                    return false;
                }
            }
        }
        warn!(
            max_steps = WARMUP_MAX_STEPS,
            "cb health check never finished"
        );
        let _ = handle.cancel();
        false
    }

    /// The key the schema sets above use for `body`. Keyed with this
    /// engine's random keys, so a client cannot craft a schema that collides
    /// with another's. Kept in memory only: never logged, stored or sent.
    fn schema_hash(&self, body: &str) -> u64 {
        self.schema_keys.hash_one(body)
    }

    /// End the current run of failures: a step or canary succeeded. With
    /// `release` (the device was at fault during the run) the schemas blamed
    /// in it are released; otherwise they stay refused.
    fn end_failure_run(&mut self, release: bool) {
        if release {
            for (h, strikes) in self.blamed_in_run.drain() {
                self.quarantined.remove(h);
                self.strikes.take_back(h, strikes);
            }
        }
        self.blamed_in_run.clear();
        self.canary_failed_in_run = false;
        self.step_failures = 0;
        self.suspect_failures = 0;
        self.latched_at = None;
    }

    /// Run the warmup structured-output probe: one request under the probe
    /// schema, driven to a terminal status within [`WARMUP_MAX_STEPS`].
    /// Returns the text it produced, or nothing when the scheduler dropped
    /// or cancelled it (an incomplete probe proves nothing).
    fn run_structured_probe(&mut self, request_id: u64) -> Result<String, OvError> {
        let cfg = build_gen_config(
            0,
            0,
            self.prompt_pretemplated,
            PROBE_MAX_TOKENS,
            0.0,
            false,
            &SamplingParams::default(),
            Some(PROBE_SCHEMA.to_owned()),
        );
        let mut handle = self.pipe.add_request(request_id, PROBE_PROMPT, &cfg)?;
        let mut text = String::new();
        for _ in 0..WARMUP_MAX_STEPS {
            if let Err(err) = self.pipe.step() {
                let _ = handle.cancel();
                return Err(err);
            }
            let r = match handle.read() {
                Ok(r) => r,
                Err(err) => {
                    let _ = handle.cancel();
                    return Err(err);
                }
            };
            text.push_str(&r.text_delta);
            match r.status {
                CbStatus::Running => {}
                CbStatus::Finished => return Ok(text),
                CbStatus::Ignored | CbStatus::Cancelled => {
                    warn!(
                        status = ?r.status,
                        "structured output probe did not complete; treating it as inconclusive"
                    );
                    return Ok(String::new());
                }
            }
        }
        // Not finished in time: judge what it produced so far, and free it.
        let _ = handle.cancel();
        Ok(text)
    }
}

/// Shared `GenConfig` assembly for both ov-genai engines (the CB engine
/// passes 0 for the speculative knobs — the CB scheduler owns batching).
/// `json_schema` constrains the output to that schema; `None` leaves it free.
#[allow(clippy::too_many_arguments)]
fn build_gen_config(
    speculative_k: u32,
    prompt_lookup_ngram: u32,
    prompt_pretemplated: bool,
    max_tokens: u32,
    temperature: f32,
    enable_thinking: bool,
    sampling: &SamplingParams,
    json_schema: Option<String>,
) -> GenConfig {
    let do_sample = temperature > 0.0;
    let mut cfg = GenConfig {
        max_new_tokens: max_tokens,
        do_sample,
        temperature: if do_sample { temperature } else { 0.0 },
        num_assistant_tokens: 0,
        max_ngram_size: 0,
        // Hybrid: only when the API rendered the template for us — i.e. a
        // template loaded AND this request is thinking-OFF — do we tell
        // ov-genai to skip its own apply. Thinking-ON keeps ov-genai's
        // native templating (the legacy join the API emitted), untouched.
        skip_chat_template: prompt_pretemplated && !enable_thinking,
        // OpenAI sampling knobs (#14). top_p/top_k take effect in the
        // sampling regime (do_sample, i.e. temperature > 0); penalties and
        // seed are forwarded regardless.
        //
        // `seed` maps to GenerationConfig.rng_seed. What that buys you differs
        // per pipeline, so the caveat lives on the monolithic engine's
        // gen_config (the pipeline it applies to) rather than here — this
        // function also builds configs for the continuous-batching path.
        top_p: if sampling.top_p > 0.0 {
            sampling.top_p.min(1.0)
        } else {
            1.0
        },
        top_k: sampling.top_k,
        frequency_penalty: sampling.frequency_penalty,
        presence_penalty: sampling.presence_penalty,
        seed: sampling.seed,
        json_schema,
    };
    if speculative_k > 0 {
        cfg.num_assistant_tokens = speculative_k;
    }
    if prompt_lookup_ngram > 0 {
        cfg.num_assistant_tokens = speculative_k.max(5);
        cfg.max_ngram_size = prompt_lookup_ngram;
    }
    cfg
}

impl OvGenaiEngine {
    /// SEED CAVEAT (validated on-HW, OpenVINO GenAI 2026.1; not re-checked on
    /// the current 2026.2 floor): `seed` is forwarded to
    /// `GenerationConfig.rng_seed`, but it does NOT make ov-genai reproducible
    /// across requests on this reused `LLMPipeline`. OV-GenAI's
    /// `StatefulLLMPipeline` only re-seeds when the seed VALUE CHANGES between
    /// consecutive `generate()` calls (`if (m_sampler.get_seed() !=
    /// config.rng_seed) set_seed(...)`) and never calls `clear_request_info`,
    /// so two identical seeds in a row reuse the advanced RNG and diverge.
    /// Different seeds DO change the output, and the Rust-sampler engines
    /// (ov-runtime/sparse-moe) honor seed fully — they re-seed per request.
    ///
    /// This is a property of `StatefulLLMPipeline`, so it does not describe
    /// [`OvGenaiCbEngine`]: `ContinuousBatchingPipeline` builds a fresh request
    /// context per request. Whether that yields per-request reproducibility in
    /// practice is untested here.
    fn gen_config(
        &self,
        max_tokens: u32,
        temperature: f32,
        enable_thinking: bool,
        sampling: &SamplingParams,
        json_schema: Option<String>,
    ) -> GenConfig {
        stateful_gen_config(
            self.speculative_k,
            self.prompt_lookup_ngram,
            self.prompt_pretemplated,
            max_tokens,
            temperature,
            enable_thinking,
            sampling,
            json_schema,
        )
    }
}

/// The stateful engine's `GenConfig` (see [`OvGenaiEngine::gen_config`]); a
/// free function so it is testable without a pipeline.
#[allow(clippy::too_many_arguments)]
fn stateful_gen_config(
    speculative_k: u32,
    prompt_lookup_ngram: u32,
    prompt_pretemplated: bool,
    max_tokens: u32,
    temperature: f32,
    enable_thinking: bool,
    sampling: &SamplingParams,
    json_schema: Option<String>,
) -> GenConfig {
    build_gen_config(
        speculative_k,
        prompt_lookup_ngram,
        prompt_pretemplated,
        max_tokens,
        temperature,
        enable_thinking,
        sampling,
        json_schema,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    // ---- CB engine state machine -------------------------------------------
    //
    // Driven through the CbPipe seam with a scripted mock, so the whole state
    // machine runs without OpenVINO. Before the seam none of this was reachable:
    // a stub CbPipeline errors on every call, so step() could only ever take its
    // failure branch.

    #[derive(Default)]
    struct MockState {
        /// Scripted reads per enrolled request, in enrollment order.
        reads: Vec<VecDeque<Result<CbRead, OvError>>>,
        /// Returned once a request's script runs out.
        default_read: Option<CbRead>,
        step_fails: bool,
        /// The failed step's backend text, when not the default.
        step_err: Option<String>,
        /// With `step_fails`, fail a step only when the request added last
        /// carries a schema: bad schemas on a healthy device.
        fail_only_with_grammar: bool,
        last_added_grammar: bool,
        /// The schema text the last added request handed to GenAI.
        last_schema: Option<String>,
        steps: usize,
        added: Vec<u64>,
        cancelled: Vec<u64>,
        /// Returned by `validate_json_schema` when set.
        validate_err: Option<String>,
        validate_calls: usize,
    }

    struct MockPipe {
        st: Arc<Mutex<MockState>>,
    }
    struct MockReq {
        st: Arc<Mutex<MockState>>,
        idx: usize,
        id: u64,
    }

    impl CbPipe for MockPipe {
        fn add_request(
            &self,
            request_id: u64,
            _prompt: &str,
            cfg: &GenConfig,
        ) -> Result<Box<dyn CbReq>, OvError> {
            let mut st = self.st.lock().unwrap();
            st.added.push(request_id);
            st.last_added_grammar = cfg.json_schema.is_some();
            st.last_schema = cfg.json_schema.clone();
            let idx = st.added.len() - 1;
            while st.reads.len() <= idx {
                st.reads.push(VecDeque::new());
            }
            Ok(Box::new(MockReq {
                st: self.st.clone(),
                idx,
                id: request_id,
            }))
        }
        fn step(&self) -> Result<(), OvError> {
            let mut st = self.st.lock().unwrap();
            st.steps += 1;
            if st.step_fails && (!st.fail_only_with_grammar || st.last_added_grammar) {
                Err(OvError::Native(
                    st.step_err
                        .clone()
                        .unwrap_or_else(|| "mock step failure".into()),
                ))
            } else {
                Ok(())
            }
        }
        fn count_tokens(&self, _text: &str) -> Option<u32> {
            Some(7)
        }
        fn validate_json_schema(&self, _schema: &str) -> Result<(), OvError> {
            let mut st = self.st.lock().unwrap();
            st.validate_calls += 1;
            match &st.validate_err {
                Some(m) => Err(OvError::Native(m.clone())),
                None => Ok(()),
            }
        }
    }

    impl CbReq for MockReq {
        fn read(&mut self) -> Result<CbRead, OvError> {
            let mut st = self.st.lock().unwrap();
            if let Some(r) = st.reads.get_mut(self.idx).and_then(|q| q.pop_front()) {
                return r;
            }
            match &st.default_read {
                Some(r) => Ok(r.clone()),
                None => Err(OvError::Native("mock: script exhausted".into())),
            }
        }
        fn cancel(&self) -> Result<(), OvError> {
            self.st.lock().unwrap().cancelled.push(self.id);
            Ok(())
        }
    }

    fn read_of(status: CbStatus, tokens: u32, text: &str, finish: CbFinish) -> CbRead {
        CbRead {
            text_delta: text.to_string(),
            new_tokens: tokens,
            status,
            finish,
            resynced: false,
        }
    }

    /// A read that reports progress but no output — every iteration of a
    /// chunked prefill looks like this.
    fn idle_read() -> CbRead {
        read_of(CbStatus::Running, 0, "", CbFinish::Unknown)
    }

    fn mock_engine(st: &Arc<Mutex<MockState>>) -> OvGenaiCbEngine {
        OvGenaiCbEngine {
            active: Vec::new(),
            pipe: Box::new(MockPipe { st: st.clone() }),
            device: "CPU".into(),
            prompt_pretemplated: false,
            next_request_id: 0,
            max_tokens_default: 256,
            idle_steps: 0,
            step_failures: 0,
            suspect_failures: 0,
            latched_at: None,
            blamed_in_run: HashMap::new(),
            canary_failed_in_run: false,
            structured_unavailable: None,
            quarantined: QuarantineSet::default(),
            strikes: StrikeSet::default(),
            compiled: QuarantineSet::default(),
            schema_keys: RandomState::new(),
            schema_budget: SchemaBudget::default(),
        }
    }

    fn task(id: &str, max_tokens: u32) -> GenerationTask {
        GenerationTask::new(id, "hello").with_max_tokens(max_tokens)
    }

    /// Option B: the CB engine has no forced-prefix path — a resumed submit
    /// must be declined with the `resume_unsupported:` sentinel in the error,
    /// BEFORE any request is admitted (silently accepting would regenerate
    /// from scratch and present it as a resume). Any rewording that drops the
    /// sentinel silently breaks the scheduler's re-route matching.
    #[test]
    fn cb_submit_declines_resumed_task_with_sentinel() {
        let st = Arc::new(Mutex::new(MockState::default()));
        let mut e = mock_engine(&st);
        let mut t = task("t1", 8);
        t.resume_token_ids = Some(vec![1, 2]);
        let err = e.submit(t).expect_err("resumed submit must be declined");
        assert!(
            err.to_string().contains("resume_unsupported:"),
            "decline must carry the sentinel the scheduler matches on: {err}"
        );
        // Nothing was admitted.
        assert!(e.active.is_empty());
    }

    /// The B1 fix: a step that progressed the batch without producing a token
    /// must still report progress, or the runner's no-progress guard kills a
    /// perfectly healthy request mid-prefill. n_tokens must be an explicit 0 —
    /// the SSE path does `n_tokens.unwrap_or(1)`.
    #[test]
    fn cb_step_emits_a_zero_token_liveness_chunk_when_nothing_was_produced() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 8)).unwrap();

        let out = e.step().unwrap();
        assert_eq!(out.len(), 1, "must not return an empty Vec while in flight");
        let (tid, chunk) = &out[0];
        assert_eq!(tid, "t1");
        assert!(chunk.text.is_empty());
        assert!(!chunk.is_final);
        assert_eq!(chunk.n_tokens, Some(0), "explicit 0, not None");
        assert!(chunk.error.is_none());
    }

    /// ...but the liveness chunk hides those iterations from the runner's
    /// guard, so the engine owes it a bound. At the real MAX_IDLE_STEPS.
    #[test]
    fn cb_liveness_is_bounded_so_a_wedged_batch_still_fails() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 4096)).unwrap();

        for i in 0..MAX_IDLE_STEPS {
            let out = e.step().unwrap();
            assert!(
                out[0].1.error.is_none(),
                "failed early at idle step {i} of {MAX_IDLE_STEPS}"
            );
        }
        // One past the bound: the batch is declared stalled.
        let out = e.step().unwrap();
        assert_eq!(out.len(), 1);
        let err = out[0].1.error.as_deref().expect("must fail past the bound");
        assert!(err.contains("stalled"), "{err}");
        assert!(e.active.is_empty(), "stalled requests must be dropped");
    }

    /// A single producing iteration resets the idle run — otherwise a long but
    /// healthy generation would eventually trip the bound.
    #[test]
    fn cb_progress_resets_the_idle_counter() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        {
            let mut g = st.lock().unwrap();
            g.reads.push(VecDeque::new());
            for _ in 0..MAX_IDLE_STEPS {
                g.reads[0].push_back(Ok(idle_read()));
            }
            g.reads[0].push_back(Ok(read_of(CbStatus::Running, 1, "x", CbFinish::Unknown)));
            for _ in 0..MAX_IDLE_STEPS {
                g.reads[0].push_back(Ok(idle_read()));
            }
        }
        let mut e = mock_engine(&st);
        e.submit(task("t1", 4096)).unwrap();
        for i in 0..(2 * MAX_IDLE_STEPS + 1) {
            let out = e.step().unwrap();
            assert!(
                out[0].1.error.is_none(),
                "tripped at step {i} despite progress"
            );
        }
    }

    /// The B2 fix: a request OpenVINO abandoned for want of KV cache must not
    /// be finalised like a normal completion.
    #[test]
    fn cb_ignored_status_is_reported_as_an_error() {
        let st = Arc::new(Mutex::new(MockState::default()));
        st.lock()
            .unwrap()
            .reads
            .push(VecDeque::from(vec![Ok(read_of(
                CbStatus::Ignored,
                0,
                "",
                CbFinish::Unknown,
            ))]));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 8)).unwrap();

        let out = e.step().unwrap();
        assert_eq!(out.len(), 1);
        let c = &out[0].1;
        let err = c
            .error
            .as_deref()
            .expect("eviction must surface as an error");
        assert!(err.contains("KV cache exhausted"), "{err}");
        assert!(c.is_final);
        assert!(e.active.is_empty());
    }

    /// Prefer OpenVINO's own reason over guessing from token counts.
    #[test]
    fn cb_final_chunk_prefers_the_reported_finish_reason() {
        let st = Arc::new(Mutex::new(MockState::default()));
        st.lock()
            .unwrap()
            .reads
            .push(VecDeque::from(vec![Ok(read_of(
                CbStatus::Finished,
                1,
                "!",
                CbFinish::Length,
            ))]));
        let mut e = mock_engine(&st);
        // Only 1 token against a cap of 4096 — inference alone would say Stop.
        e.submit(task("t1", 4096)).unwrap();
        let out = e.step().unwrap();
        assert_eq!(out[0].1.finish_reason, Some(FinishReason::Length));
    }

    /// ...and fall back to the token-count inference when it reports nothing,
    /// which is what happens when the terminal read carries no new output.
    #[test]
    fn cb_final_chunk_falls_back_to_token_inference() {
        for (max_tokens, produced, want) in [
            (4u32, 4u32, FinishReason::Length),
            (4096, 1, FinishReason::Stop),
        ] {
            let st = Arc::new(Mutex::new(MockState::default()));
            st.lock()
                .unwrap()
                .reads
                .push(VecDeque::from(vec![Ok(read_of(
                    CbStatus::Finished,
                    produced,
                    "x",
                    CbFinish::Unknown,
                ))]));
            let mut e = mock_engine(&st);
            e.submit(task("t1", max_tokens)).unwrap();
            let out = e.step().unwrap();
            assert_eq!(out[0].1.finish_reason, Some(want), "max={max_tokens}");
        }
    }

    /// Regression for the token accounting fixed in 7c8e6db: the API SUMS
    /// per-chunk counts, so each chunk must carry its own increment. Emitting
    /// the running total here measured 2N-1 on hardware.
    #[test]
    fn cb_chunks_carry_per_chunk_token_increments_not_the_total() {
        let st = Arc::new(Mutex::new(MockState::default()));
        st.lock().unwrap().reads.push(VecDeque::from(vec![
            Ok(read_of(CbStatus::Running, 3, "abc", CbFinish::Unknown)),
            Ok(read_of(CbStatus::Running, 4, "defg", CbFinish::Unknown)),
            Ok(read_of(CbStatus::Finished, 2, "hi", CbFinish::Unknown)),
        ]));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 4096)).unwrap();

        let mut counts = Vec::new();
        let mut text = String::new();
        for _ in 0..3 {
            for (_, c) in e.step().unwrap() {
                counts.push(c.n_tokens.unwrap());
                text.push_str(&c.text);
            }
        }
        assert_eq!(counts, vec![3, 4, 2], "must be increments, not 3,7,9");
        assert_eq!(text, "abcdefghi");
    }

    /// A failed scheduler iteration poisons the whole batch: every in-flight
    /// request gets its own error chunk rather than one stream eating it.
    #[test]
    fn cb_step_failure_fails_every_in_flight_request() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 8)).unwrap();
        e.submit(task("t2", 8)).unwrap();

        let out = e.step().unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|(_, c)| c.error.is_some() && c.is_final));
        assert!(e.active.is_empty());
    }

    /// ...and once it keeps failing, stop admitting work onto a dead pipeline
    /// instead of 503-ing every new request forever.
    #[test]
    fn cb_submit_refuses_work_after_repeated_step_failures() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..MAX_STEP_FAILURES {
            e.submit(task(&format!("t{i}"), 8)).unwrap();
            let _ = e.step().unwrap();
        }
        let err = e.submit(task("late", 8)).expect_err("must refuse");
        assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
    }

    #[test]
    fn cb_submit_is_idempotent_and_honours_the_queue_cap() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 8)).unwrap();
        e.submit(task("t1", 8)).unwrap();
        assert_eq!(
            e.active.len(),
            1,
            "resubmitting a live task must be a no-op"
        );
        assert_eq!(
            st.lock().unwrap().added.len(),
            1,
            "and must not re-enroll it"
        );

        for i in 1..crate::dist_spec::MAX_PENDING_TASKS {
            e.submit(task(&format!("f{i}"), 8)).unwrap();
        }
        let err = e.submit(task("overflow", 8)).expect_err("cap must reject");
        assert!(matches!(err, EngineError::QueueFull { .. }), "{err:?}");
    }

    /// The trait contract: after cancel returns, step emits nothing more for
    /// that task.
    #[test]
    fn cb_cancel_drops_the_task_and_aborts_it_upstream() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 8)).unwrap();
        e.submit(task("t2", 8)).unwrap();
        e.cancel(&"t1".to_string());

        assert_eq!(e.active.len(), 1);
        assert_eq!(st.lock().unwrap().cancelled, vec![0], "must abort upstream");
        for _ in 0..5 {
            for (tid, _) in e.step().unwrap() {
                assert_ne!(tid, "t1", "cancelled task must not emit chunks");
            }
        }
    }
    use cascadia_engine::EngineError;
    use cascadia_types::{PeerEndpoint, PeerLayout, ShardSpec};

    // ---- structured output (JSON schema) ------------------------------------

    fn schema_task(id: &str, body: &str) -> GenerationTask {
        task(id, 8).with_grammar(cascadia_types::GrammarSpec {
            kind: cascadia_types::GrammarKind::JsonSchema,
            body: body.to_string(),
        })
    }

    const SCHEMA_A: &str = r#"{"type":"object","properties":{"a":{"type":"integer"}}}"#;

    #[test]
    fn stateful_generate_failure_is_a_client_error_only_when_it_names_the_grammar() {
        for raw in [
            "xgrammar: failed to compile the schema",
            "Invalid JSON Schema: unsupported keyword",
            "bad json_schema",
            "Grammar construction failed",
        ] {
            let c = generate_failure_chunk(
                &schema_task("s", SCHEMA_A),
                OvError::Native(raw.into()),
                false,
            );
            assert!(c.invalid_request, "{raw}");
            assert_eq!(
                c.error,
                Some(format!("response_format json_schema: {raw}")),
                "{raw}"
            );
        }
        let raw = "Exception from src/plugins/intel_gpu/src/runtime/ocl/ocl_stream.cpp:1: \
                   [GPU] CL_DEVICE_NOT_AVAILABLE: device lost";
        let c = generate_failure_chunk(
            &schema_task("s", SCHEMA_A),
            OvError::Native(raw.into()),
            false,
        );
        assert!(
            !c.invalid_request,
            "a device error with a schema stays a 503"
        );
        assert_eq!(c.error, Some(OvError::Native(raw.into()).to_string()));
    }

    #[test]
    fn stateful_generate_failure_with_a_schema_is_a_client_error() {
        let err = || {
            OvError::Native(
                "Exception from src/core.cpp:1:\n[10:00:00] src/grammar.cc:42: bad schema\n".into(),
            )
        };
        let c = generate_failure_chunk(&schema_task("s", SCHEMA_A), err(), false);
        assert!(c.invalid_request);
        assert!(c.is_final);
        assert_eq!(
            c.error.as_deref(),
            Some("response_format json_schema: bad schema")
        );

        // Not the client's schema, or not the schema's fault: unchanged.
        for (t, resource) in [
            (task("p", 8), false),
            (schema_task("j", JSON_OBJECT_PROBE), false),
            (schema_task("r", SCHEMA_A), true),
        ] {
            let c = generate_failure_chunk(&t, err(), resource);
            assert!(!c.invalid_request, "{}", t.task_id);
            assert_eq!(c.error, Some(err().to_string()), "{}", t.task_id);
        }
    }
    const SCHEMA_B: &str = r#"{"type":"object","properties":{"b":{"type":"string"}}}"#;

    #[test]
    fn build_gen_config_sets_json_schema() {
        let sp = SamplingParams::default();
        let cfg = build_gen_config(0, 0, false, 8, 0.0, false, &sp, Some(SCHEMA_A.to_owned()));
        assert_eq!(cfg.json_schema.as_deref(), Some(SCHEMA_A));
        let cfg = build_gen_config(0, 0, false, 8, 0.0, false, &sp, None);
        assert_eq!(cfg.json_schema, None);
    }

    #[test]
    fn stateful_gen_config_keeps_speculative_knobs_with_a_schema() {
        let cfg = stateful_gen_config(
            3,
            2,
            false,
            16,
            0.0,
            false,
            &SamplingParams::default(),
            Some(SCHEMA_A.to_owned()),
        );
        assert_eq!(cfg.json_schema.as_deref(), Some(SCHEMA_A));
        assert_eq!(cfg.num_assistant_tokens, 5);
        assert_eq!(cfg.max_ngram_size, 2);
        assert_eq!(cfg.max_new_tokens, 16);
    }

    /// An integer with no bound goes to GenAI with ±(10^18 - 1), so a small
    /// model can't repeat digits until max_tokens; a bound the client set is
    /// kept, nested and `$ref`'d integers and `["integer","null"]` are bounded,
    /// and key order stays the client's.
    #[test]
    fn a_schema_with_too_many_integers_goes_to_genai_as_written() {
        // Each bounded integer is a range xgrammar compiles on its own: 1000
        // of them took 1.37 s against 0.10 s unbounded (xgrammar 0.1.31,
        // Qwen2.5 tokenizer, one thread), a cost the xgrammar guard doesn't
        // price. Past the cap the client's schema goes as written.
        let many = |n: usize| {
            let props: Vec<String> = (0..n)
                .map(|i| format!(r#""p{i}":{{"type":"integer"}}"#))
                .collect();
            format!(
                r#"{{"type":"object","properties":{{{}}}}}"#,
                props.join(",")
            )
        };
        let at_cap = many(GENAI_MAX_BOUNDED_INTEGERS);
        assert_ne!(
            genai_json_schema(&schema_task("s", &at_cap)).unwrap(),
            at_cap,
            "up to the cap, integers are bounded"
        );
        let over = many(GENAI_MAX_BOUNDED_INTEGERS + 1);
        assert_eq!(genai_json_schema(&schema_task("s", &over)).unwrap(), over);
    }

    #[test]
    fn the_schema_sent_to_genai_bounds_its_integers() {
        let cap = cascadia_grammar::INTEGER_DIGITS_CAP;
        let sent = genai_json_schema(&schema_task("s", SCHEMA_A)).unwrap();
        assert_eq!(
            sent,
            format!(
                r#"{{"type":"object","properties":{{"a":{{"type":"integer","maximum":{cap},"minimum":-{cap}}}}}}}"#
            )
        );
        let body = r##"{"type":"object","properties":{"z":{"$ref":"#/$defs/N"},"y":{"type":["integer","null"]},"x":{"type":"array","items":{"type":"integer","minimum":0}},"w":{"type":"number"}},"$defs":{"N":{"type":"integer","maximum":7}}}"##;
        let sent: serde_json::Value =
            serde_json::from_str(&genai_json_schema(&schema_task("s", body)).unwrap()).unwrap();
        assert_eq!(
            sent["$defs"]["N"],
            serde_json::json!({"type":"integer","maximum":7,"minimum":-cap})
        );
        assert_eq!(sent["properties"]["y"]["maximum"], cap);
        assert_eq!(
            sent["properties"]["x"]["items"],
            serde_json::json!({"type":"integer","minimum":0,"maximum":cap})
        );
        assert_eq!(
            sent["properties"]["w"],
            serde_json::json!({"type":"number"})
        );
        let keys: Vec<&str> = sent["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["z", "y", "x", "w"]);
        // Not JSON: sent as it is, for GenAI to report.
        assert_eq!(
            genai_json_schema(&schema_task("s", "{nope")).as_deref(),
            Some("{nope")
        );
        assert_eq!(genai_json_schema(&task("p", 8)), None);
    }

    /// The CB engine hands GenAI the bounded schema and checks that text,
    /// while its budget of distinct schemas counts the client's text.
    #[test]
    fn the_cb_engine_sends_and_checks_the_bounded_schema() {
        let st = Arc::new(Mutex::new(MockState::default()));
        let mut e = mock_engine(&st);
        e.submit(schema_task("s", SCHEMA_A)).unwrap();
        let sent = st.lock().unwrap().last_schema.clone().unwrap();
        assert!(sent.contains(r#""maximum":999999999999999999"#), "{sent}");
        assert!(e.schema_budget.seen.contains(SCHEMA_A));
        let mut validated = None;
        check_grammar(&schema_task("s", SCHEMA_A), None, |s| {
            validated = Some(s.to_owned());
            Ok(())
        })
        .unwrap();
        assert_eq!(validated.as_deref(), Some(sent.as_str()));
    }

    #[test]
    fn check_grammar_paths() {
        // No grammar: validate must not run.
        let plain = task("p", 8);
        assert!(check_grammar(&plain, Some("x"), |_| panic!("validate called")).is_ok());

        let t = schema_task("s", SCHEMA_A);
        let err = check_grammar(&t, Some("no xgrammar"), |_| Ok(())).unwrap_err();
        match err {
            EngineError::InvalidConfig(m) => assert_eq!(
                m,
                "structured output is not available for this model: no xgrammar"
            ),
            other => panic!("{other:?}"),
        }

        let err =
            check_grammar(&t, None, |_| Err(OvError::Native("bad schema".into()))).unwrap_err();
        match err {
            EngineError::InvalidConfig(m) => {
                assert_eq!(m, "response_format json_schema: bad schema")
            }
            other => panic!("{other:?}"),
        }

        let mut seen = None;
        assert!(check_grammar(&t, None, |s| {
            seen = Some(s.to_owned());
            Ok(())
        })
        .is_ok());
        assert_eq!(
            seen.as_deref(),
            genai_json_schema(&t).as_deref(),
            "validates the schema GenAI is sent"
        );
    }

    #[test]
    fn cb_submit_rejects_an_invalid_schema_before_enrolling() {
        let st = Arc::new(Mutex::new(MockState {
            validate_err: Some("unsupported keyword".into()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        let err = e.submit(schema_task("t1", SCHEMA_A)).unwrap_err();
        assert!(matches!(err, EngineError::InvalidConfig(_)), "{err:?}");
        assert!(err.to_string().contains("unsupported keyword"), "{err}");
        assert!(st.lock().unwrap().added.is_empty());
        assert!(e.active.is_empty());
        // A plain task never runs the validator.
        e.submit(task("t2", 8)).unwrap();
        assert_eq!(st.lock().unwrap().validate_calls, 1);
    }

    #[test]
    fn a_new_schema_past_the_budget_is_refused_and_a_seen_one_still_runs() {
        let st = Arc::new(Mutex::new(MockState::default()));
        let mut e = mock_engine(&st);
        let body = |i: usize| {
            format!(r#"{{"type":"object","properties":{{"f{i}":{{"type":"integer"}}}}}}"#)
        };
        for i in 0..MAX_DISTINCT_SCHEMAS {
            e.submit(schema_task(&format!("t{i}"), &body(i))).unwrap();
        }
        // One more distinct schema: refused, in the class a gateway moves to
        // another node.
        match e.submit(schema_task("over", &body(MAX_DISTINCT_SCHEMAS))) {
            Err(EngineError::InvalidConfig(m)) => assert_eq!(m, BUDGET_FULL_400),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
        // A schema it already let through is served from GenAI's cache.
        e.submit(schema_task("again", &body(0))).unwrap();
        // Plain requests are never counted or refused.
        e.submit(GenerationTask::new("plain", "hi")).unwrap();
    }

    /// The 400 a new schema gets once the budget is full. The node maps it
    /// by its prefix, "structured output is not available".
    const BUDGET_FULL_400: &str = "structured output is not available for this model: \
        this worker has reached its limit of 64 distinct response_format schemas per pipeline \
        (GenAI keeps each compiled grammar until the worker restarts); schemas it has already \
        served keep working, and restarting the worker clears the limit";

    /// A full budget refuses each new schema with the same 400, counts every
    /// refusal, and warns once, however many it refuses.
    #[test]
    fn a_full_budget_refuses_with_the_limit_counts_and_warns_once() {
        // The budget as the stateful engine's submit uses it (check, then record).
        let mut budget = SchemaBudget::default();
        for i in 0..MAX_DISTINCT_SCHEMAS {
            let t = schema_task("t", &full_budget_body(i));
            budget.check(&t).unwrap();
            budget.record(&t);
        }
        assert_eq!((budget.refused, budget.full_warnings), (0, 0));
        for n in 1..=5u64 {
            let t = schema_task("over", &full_budget_body(1000 + n as usize));
            match budget.check(&t) {
                Err(EngineError::InvalidConfig(m)) => assert_eq!(m, BUDGET_FULL_400),
                other => panic!("expected InvalidConfig, got {other:?}"),
            }
            assert_eq!(budget.refused, n);
        }
        assert_eq!(budget.full_warnings, 1);
        // A schema already served is not a refusal.
        budget
            .check(&schema_task("again", &full_budget_body(0)))
            .unwrap();
        assert_eq!(budget.refused, 5);

        // The CB engine.
        let st = Arc::new(Mutex::new(MockState::default()));
        let mut e = mock_engine(&st);
        for i in 0..MAX_DISTINCT_SCHEMAS {
            e.submit(schema_task(&format!("t{i}"), &full_budget_body(i)))
                .unwrap();
        }
        for n in 1..=3u64 {
            match e.submit(schema_task("over", &full_budget_body(1000 + n as usize))) {
                Err(EngineError::InvalidConfig(m)) => assert_eq!(m, BUDGET_FULL_400),
                other => panic!("expected InvalidConfig, got {other:?}"),
            }
        }
        assert_eq!(
            (e.schema_budget.refused, e.schema_budget.full_warnings),
            (3, 1)
        );
    }

    fn full_budget_body(i: usize) -> String {
        format!(r#"{{"type":"object","properties":{{"f{i}":{{"type":"integer"}}}}}}"#)
    }

    /// json_object is one fixed grammar per pipeline: a full budget never
    /// refuses it, even as its first request, and it takes no slot. The API
    /// sends this same text for `json_object` and for a `json_schema` whose
    /// schema is missing, `null`, `true` or `{}` (pinned in cascadia-api's
    /// `every_any_object_spelling_reaches_the_engine_as_the_fixed_text`).
    #[test]
    fn a_full_schema_budget_never_refuses_json_object() {
        // The CB engine.
        let st = Arc::new(Mutex::new(MockState::default()));
        let mut e = mock_engine(&st);
        for i in 0..MAX_DISTINCT_SCHEMAS {
            e.submit(schema_task(&format!("t{i}"), &full_budget_body(i)))
                .unwrap();
        }
        e.submit(schema_task("obj", JSON_OBJECT_PROBE)).unwrap();
        assert_eq!(e.schema_budget.seen.len(), MAX_DISTINCT_SCHEMAS);
        // The budget as the stateful engine's submit uses it (check, then record).
        let mut budget = SchemaBudget::default();
        for i in 0..MAX_DISTINCT_SCHEMAS {
            let t = schema_task("t", &full_budget_body(i));
            budget.check(&t).unwrap();
            budget.record(&t);
        }
        assert!(budget
            .check(&schema_task("over", &full_budget_body(999)))
            .is_err());
        let obj = schema_task("obj", JSON_OBJECT_PROBE);
        budget.check(&obj).unwrap();
        budget.record(&obj);
        assert!(!budget.seen.contains(JSON_OBJECT_PROBE));
    }

    #[test]
    fn check_grammar_refuses_a_schema_xgrammar_would_take_minutes_on() {
        // 6000 optional properties: the schema that held a worker in
        // xgrammar's compile for over 15 minutes.
        let props: Vec<String> = (0..6000).map(|i| format!("\"p{i}\":{{}}")).collect();
        let wide = format!(
            r#"{{"type":"object","properties":{{{}}}}}"#,
            props.join(",")
        );
        let t = schema_task("w", &wide);
        let err = check_grammar(&t, None, |_| panic!("validate called")).unwrap_err();
        match err {
            EngineError::InvalidConfig(m) => assert!(
                m.starts_with(
                    "response_format json_schema: xgrammar would take too long to compile a \
                     grammar for this schema"
                ),
                "{m}"
            ),
            other => panic!("{other:?}"),
        }

        // Through submit: refused before `validate`, nothing enrolled.
        let st = Arc::new(Mutex::new(MockState::default()));
        let mut e = mock_engine(&st);
        let err = e.submit(schema_task("t1", &wide)).unwrap_err();
        assert!(
            err.to_string().contains("xgrammar would take too long"),
            "{err}"
        );
        assert_eq!(st.lock().unwrap().validate_calls, 0);
        assert!(st.lock().unwrap().added.is_empty());
        // An ordinary schema still reaches `validate`.
        e.submit(schema_task("t2", SCHEMA_A)).unwrap();
        assert_eq!(st.lock().unwrap().validate_calls, 1);
    }

    #[test]
    fn cb_submit_validates_after_queue_cap() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..crate::dist_spec::MAX_PENDING_TASKS {
            e.submit(task(&format!("f{i}"), 8)).unwrap();
        }
        let err = e.submit(schema_task("s", SCHEMA_A)).unwrap_err();
        assert!(matches!(err, EngineError::QueueFull { .. }), "{err:?}");
        assert_eq!(st.lock().unwrap().validate_calls, 0);
    }

    #[test]
    fn cb_bad_schema_batched_with_plain_requests_cannot_latch() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..=MAX_STEP_FAILURES {
            let schema = numbered_schema(i);
            e.submit(schema_task(&format!("s{i}"), &schema)).unwrap();
            e.submit(task(&format!("p{i}"), 8)).unwrap();
            let out = e.step().unwrap();
            assert_eq!(out.len(), 2, "every in-flight request still fails");
            assert_eq!(e.step_failures, 0, "a new schema in the step: not counted");
            let err = e.submit(schema_task("again", &schema)).unwrap_err();
            assert!(matches!(err, EngineError::InvalidConfig(_)), "{err:?}");
            assert!(err.to_string().contains("failed to compile"), "{err}");
        }
        e.submit(task("plain", 8)).unwrap();
        e.submit(schema_task("other", SCHEMA_B)).unwrap();
    }

    #[test]
    fn cb_step_failure_after_the_grammar_task_produced_still_counts() {
        let st = Arc::new(Mutex::new(MockState::default()));
        st.lock()
            .unwrap()
            .reads
            .push(VecDeque::from(vec![Ok(read_of(
                CbStatus::Running,
                1,
                "{",
                CbFinish::Unknown,
            ))]));
        let mut e = mock_engine(&st);
        e.submit(schema_task("t1", SCHEMA_A)).unwrap();
        let out = e.step().unwrap();
        assert!(out[0].1.error.is_none());
        st.lock().unwrap().step_fails = true;
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 1);
        st.lock().unwrap().step_fails = false;
        e.submit(schema_task("t2", SCHEMA_A)).unwrap();
    }

    /// A distinct schema per `i`.
    fn numbered_schema(i: u32) -> String {
        format!(r#"{{"type":"object","properties":{{"p{i}":{{"type":"string"}}}}}}"#)
    }

    #[test]
    fn cb_consecutive_failures_with_new_schemas_do_not_latch() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("t1", SCHEMA_A)).unwrap();
        let _ = e.step().unwrap();
        e.submit(schema_task("t2", SCHEMA_B)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 0);
        // B was the only schema in its failed step: refused from then on.
        let err = e.submit(schema_task("t3", SCHEMA_B)).unwrap_err();
        assert!(err.to_string().contains("failed to compile"), "{err}");
        for (i, id) in ["t4", "t5"].into_iter().enumerate() {
            e.submit(schema_task(id, &numbered_schema(i as u32)))
                .unwrap();
            let _ = e.step().unwrap();
        }
        assert_eq!(e.step_failures, 0);
        assert_eq!(e.suspect_failures, 4);
        e.submit(task("plain", 8)).unwrap();
        e.submit(schema_task("t6", SCHEMA_C)).unwrap();
    }

    #[test]
    fn canary_is_due_once_the_retry_interval_passes() {
        let now = Instant::now();
        let ago = |d: Duration| now.checked_sub(d).unwrap();
        assert!(!canary_due(None, now));
        assert!(!canary_due(Some(now), now));
        assert!(!canary_due(
            Some(ago(LATCH_RETRY - Duration::from_secs(1))),
            now
        ));
        assert!(canary_due(Some(ago(LATCH_RETRY)), now));
        assert!(canary_due(Some(ago(LATCH_RETRY * 2)), now));
    }

    fn finished_read() -> CbRead {
        read_of(CbStatus::Finished, 1, "", CbFinish::Stop)
    }

    /// Drive `n` failed steps, each with one new plain request.
    fn fail_plain_steps(e: &mut OvGenaiCbEngine, n: u32, tag: &str) {
        for i in 0..n {
            e.submit(task(&format!("{tag}{i}"), 8)).unwrap();
            let _ = e.step().unwrap();
        }
    }

    fn past_retry() -> Instant {
        Instant::now().checked_sub(LATCH_RETRY).unwrap()
    }

    #[test]
    fn cb_many_new_schema_failures_on_a_healthy_device_never_latch() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            fail_only_with_grammar: true,
            default_read: Some(finished_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..10 {
            let schema = numbered_schema(i);
            e.submit(schema_task(&format!("s{i}"), &schema)).unwrap();
            let _ = e.step().unwrap();
            assert_eq!(e.step_failures, 0, "failure {}", i + 1);
            assert!(
                e.submit(schema_task("again", &schema)).is_err(),
                "schema {i} is blamed"
            );
        }
        // Ten client requests and one canary at the eighth failure.
        assert_eq!(st.lock().unwrap().added.len(), 11);
        assert_eq!(e.suspect_failures, 2);
        e.submit(task("plain", 8)).unwrap();
    }

    #[test]
    fn cb_dead_device_backstop_counts_after_a_failed_canary() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(finished_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..MAX_SUSPECT_FAILURES {
            e.submit(schema_task(&format!("s{i}"), &numbered_schema(i)))
                .unwrap();
            let _ = e.step().unwrap();
        }
        assert_eq!(e.step_failures, 1, "the canary failed: the eighth counts");
        // The counted failure blamed no schema.
        let last = numbered_schema(MAX_SUSPECT_FAILURES - 1);
        st.lock().unwrap().step_fails = false;
        e.submit(schema_task("last", &last)).unwrap();
    }

    #[test]
    fn cb_latched_worker_admits_nothing_until_a_canary_succeeds() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(finished_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        fail_plain_steps(&mut e, MAX_STEP_FAILURES, "t");
        let added = || st.lock().unwrap().added.len();
        let before = added();

        // Within the retry interval: refused, no canary, nothing enrolled.
        let err = e.submit(task("early", 8)).expect_err("latched");
        assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
        assert!(
            err.to_string().contains("since the last successful step"),
            "{err}"
        );
        assert!(e.submit(schema_task("s", SCHEMA_A)).is_err());
        assert_eq!(added(), before);
        assert!(e.active.is_empty());

        // Interval passed, device still dead: the canary runs, fails, and the
        // wait starts over.
        e.latched_at = Some(past_retry());
        assert!(e.submit(task("retry1", 8)).is_err());
        assert_eq!(added(), before + 1, "only the canary was enrolled");
        assert!(e.active.is_empty());
        assert!(e.submit(task("retry2", 8)).is_err());
        assert_eq!(added(), before + 1, "no canary before the next interval");

        // Device back: the next canary clears the latch and the request is
        // admitted.
        st.lock().unwrap().step_fails = false;
        e.latched_at = Some(past_retry());
        e.submit(task("retry3", 8)).unwrap();
        assert_eq!(added(), before + 3, "canary, then the client request");
        assert_eq!(e.step_failures, 0);
        assert_eq!(e.suspect_failures, 0);
        assert!(e.latched_at.is_none());
    }

    #[test]
    fn cb_schemas_blamed_while_the_device_was_failing_are_released() {
        // A failed canary, then a successful one, ends the run.
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(finished_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..MAX_SUSPECT_FAILURES {
            e.submit(schema_task(&format!("s{i}"), &numbered_schema(i)))
                .unwrap();
            let _ = e.step().unwrap();
        }
        assert!(e.submit(schema_task("x", &numbered_schema(0))).is_err());
        st.lock().unwrap().fail_only_with_grammar = true;
        e.submit(schema_task("n", SCHEMA_A)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 0, "the canary succeeded");
        e.submit(schema_task("x", &numbered_schema(0)))
            .expect("released");
        e.cancel(&"x".to_string());
        let err = e.submit(schema_task("a", SCHEMA_A)).unwrap_err();
        assert!(err.to_string().contains("failed to compile"), "{err}");

        // The latch clearing ends the run.
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(finished_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("s", SCHEMA_A)).unwrap();
        let _ = e.step().unwrap();
        fail_plain_steps(&mut e, MAX_STEP_FAILURES, "t");
        st.lock().unwrap().step_fails = false;
        e.latched_at = Some(past_retry());
        e.submit(task("p", 8)).unwrap();
        e.submit(schema_task("s2", SCHEMA_A)).expect("released");
    }

    #[test]
    fn cb_ordinary_success_after_a_failed_canary_releases_the_blame() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(finished_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..MAX_SUSPECT_FAILURES {
            e.submit(schema_task(&format!("s{i}"), &numbered_schema(i)))
                .unwrap();
            let _ = e.step().unwrap();
        }
        assert!(e.canary_failed_in_run);
        assert!(e.submit(schema_task("x", &numbered_schema(0))).is_err());
        st.lock().unwrap().step_fails = false;
        st.lock().unwrap().default_read = Some(idle_read());
        e.submit(task("p", 8)).unwrap();
        let _ = e.step().unwrap();
        e.submit(schema_task("x", &numbered_schema(0)))
            .expect("released: the device failed during that run");
        assert!(e.blamed_in_run.is_empty());
        assert!(!e.canary_failed_in_run);
    }

    #[test]
    fn cb_release_keeps_strikes_from_earlier_runs() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(finished_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        let a = e.schema_hash(SCHEMA_A);
        // Earlier run, ended by an ordinary success: A keeps one strike.
        e.submit(schema_task("a1", SCHEMA_A)).unwrap();
        e.submit(schema_task("b1", SCHEMA_B)).unwrap();
        let _ = e.step().unwrap();
        st.lock().unwrap().step_fails = false;
        e.submit(task("p", 8)).unwrap();
        let _ = e.step().unwrap();
        e.cancel(&"p".to_string());
        assert_eq!(e.strikes.strikes.get(&a).map(|s| s.0), Some(1));

        // A run on a failing device: A is struck again (and refused), then a
        // failed and a successful canary release what this run added.
        st.lock().unwrap().step_fails = true;
        e.submit(schema_task("a2", SCHEMA_A)).unwrap();
        e.submit(schema_task("c2", SCHEMA_C)).unwrap();
        let _ = e.step().unwrap();
        assert!(e.submit(schema_task("a3", SCHEMA_A)).is_err());
        for i in 1..MAX_SUSPECT_FAILURES {
            e.submit(schema_task(&format!("s{i}"), &numbered_schema(i)))
                .unwrap();
            let _ = e.step().unwrap();
        }
        assert!(e.canary_failed_in_run);
        st.lock().unwrap().fail_only_with_grammar = true;
        e.submit(schema_task("n", &numbered_schema(100))).unwrap();
        let _ = e.step().unwrap();
        assert!(!e.canary_failed_in_run);
        assert_eq!(
            e.strikes.strikes.get(&a).map(|s| s.0),
            Some(1),
            "the earlier run's strike stays"
        );
        assert_eq!(e.strikes.strikes.get(&e.schema_hash(SCHEMA_C)), None);
        e.submit(schema_task("a4", SCHEMA_A)).expect("released");
    }

    #[test]
    fn cb_schemas_blamed_before_an_ordinary_success_stay_refused() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("s", SCHEMA_A)).unwrap();
        let _ = e.step().unwrap();
        st.lock().unwrap().step_fails = false;
        e.submit(task("p", 8)).unwrap();
        let _ = e.step().unwrap();
        assert!(e.submit(schema_task("s2", SCHEMA_A)).is_err());
    }

    #[test]
    fn cb_suspect_failures_count_once_the_backstop_is_reached() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..MAX_SUSPECT_FAILURES {
            e.submit(schema_task(&format!("t{i}"), &numbered_schema(i)))
                .unwrap();
            let _ = e.step().unwrap();
            let want = u32::from(i + 1 >= MAX_SUSPECT_FAILURES);
            assert_eq!(e.step_failures, want, "failure {}", i + 1);
        }
        assert_eq!(e.suspect_failures, MAX_SUSPECT_FAILURES);

        // A successful step resets both counters.
        st.lock().unwrap().step_fails = false;
        st.lock().unwrap().default_read = Some(idle_read());
        e.submit(task("plain", 8)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 0);
        assert_eq!(e.suspect_failures, 0);
    }

    #[test]
    fn cb_step_failure_with_json_object_counts() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("t1", JSON_OBJECT_PROBE)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 1);
        st.lock().unwrap().step_fails = false;
        e.submit(schema_task("t2", JSON_OBJECT_PROBE)).unwrap();
    }

    #[test]
    fn cb_step_failure_without_grammar_still_latches() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        for i in 0..MAX_STEP_FAILURES {
            e.submit(task(&format!("t{i}"), 8)).unwrap();
            let _ = e.step().unwrap();
            assert_eq!(e.step_failures, i + 1);
        }
        assert!(matches!(
            e.submit(task("late", 8)),
            Err(EngineError::Backend(_))
        ));
    }

    #[test]
    fn quarantine_set_is_bounded_and_expires() {
        let now = Instant::now();
        let mut q = QuarantineSet::default();
        for h in 0..300u64 {
            q.insert(h, now);
        }
        assert_eq!(q.len(), 256);
        for h in 0..44u64 {
            assert!(!q.contains(h, now), "{h} should have been dropped");
        }
        for h in 44..300u64 {
            assert!(q.contains(h, now), "{h} should be kept");
        }

        let mut q = QuarantineSet::default();
        q.insert(7, now);
        q.insert(7, now);
        assert_eq!(q.len(), 1);
        assert!(q.contains(7, now + std::time::Duration::from_secs(9 * 60)));
        assert!(!q.contains(7, now + std::time::Duration::from_secs(11 * 60)));
        assert_eq!(q.len(), 0, "expired entries are pruned");
    }

    #[test]
    fn probe_verdict_cases() {
        assert_eq!(probe_verdict(Ok(" {\"k\": \"zq\"}\n".into())), None);
        assert_eq!(probe_verdict(Ok("   \n".into())), None, "inconclusive");
        assert_eq!(probe_verdict(Ok(String::new())), None, "inconclusive");
        // The budget ran out part-way through the one allowed answer.
        for text in [
            "{",
            "\n{ \"k\"",
            "{\"k\" :\n \"z",
            "{\"k\":\"zq\"",
            "{\"k\":\"zq\" ",
        ] {
            assert_eq!(
                probe_verdict(Ok(text.into())),
                None,
                "{text:?} is inconclusive"
            );
        }
        for text in [
            "hello",
            "{\"a\":1}",
            "{\"k\":\"zz\"}",
            "{\"k\":\"z q",
            "{\"k\":\"zq\"}}",
            "[",
        ] {
            let r = probe_verdict(Ok(text.into())).expect("must fail");
            assert_eq!(r, format!("the output constraint was not applied: {text}"));
        }
        let long = "x".repeat(100);
        let r = probe_verdict(Ok(long)).expect("must fail");
        assert_eq!(
            r,
            format!("the output constraint was not applied: {}", "x".repeat(40))
        );
        let r = probe_verdict(Err(OvError::Native(
            "Exception from src/core.cpp:1:\n[10:00:00] src/grammar.cc:42: no grammar backend\n"
                .into(),
        )))
        .expect("must fail");
        assert_eq!(r, "no grammar backend");
    }

    #[test]
    fn probe_is_retried_once_after_an_error() {
        let mut calls = 0;
        let r = probe_with_retry(|| {
            calls += 1;
            if calls == 1 {
                Err(OvError::Native("transient".into()))
            } else {
                Ok("{}".into())
            }
        });
        assert_eq!(r.unwrap(), "{}");
        assert_eq!(calls, 2);

        let mut calls = 0;
        let r = probe_with_retry(|| {
            calls += 1;
            Err(OvError::Native(format!("fail {calls}")))
        });
        assert_eq!(ov_message(r.unwrap_err()), "fail 2");
        assert_eq!(calls, 2);

        let mut calls = 0;
        let _ = probe_with_retry(|| {
            calls += 1;
            Ok(String::new())
        });
        assert_eq!(calls, 1);
    }

    #[test]
    fn structured_gate_skips_the_probe_on_npu_and_with_a_draft_model() {
        const NPU: &str = "not reliable on NPU with this OpenVINO GenAI version (the output \
                           constraint is not fully applied); use GPU or CPU for response_format";
        const DRAFT: &str = "not supported with a draft model (speculative decoding); \
                             run without --draft-model for response_format";
        let gate = |b: &OvGenaiBuilder| {
            structured_gate(&b.device, b.draft_model_path.is_some(), Err(OvError::Stub))
        };
        let builder = |device: &str| OvGenaiBuilder::new("m", device);
        for device in ["NPU", "npu", "NPU.0", "HETERO:GPU,NPU", "AUTO:NPU.1,CPU"] {
            assert_eq!(gate(&builder(device)).as_deref(), Some(NPU), "{device}");
        }
        for device in ["GPU", "GPU.1", "CPU", "HETERO:GPU,CPU", "AUTO"] {
            assert_eq!(gate(&builder(device)), None, "{device}: the probe runs");
        }
        assert_eq!(
            gate(&builder("GPU").with_draft("d", "GPU", 3)).as_deref(),
            Some(DRAFT)
        );
        assert_eq!(
            gate(&builder("CPU").with_prompt_lookup(3)),
            None,
            "prompt lookup keeps the probe"
        );
        let old = cascadia_ov_genai_shim::OvVersion {
            built_genai: "2026.4.1".into(),
            built_core: "2026.4.1".into(),
            runtime_genai: "2026.2.0".into(),
            runtime_core: "2026.2.0".into(),
        };
        assert_eq!(
            structured_gate("GPU", false, Ok(old)).as_deref(),
            Some("needs OpenVINO GenAI 2026.4 or newer; this runtime is 2026.2.0")
        );
    }

    #[test]
    fn runtime_version_gate() {
        assert_eq!(
            runtime_older_than_2026_4("2026.2.0.0-21903-abc"),
            Some(true)
        );
        assert_eq!(runtime_older_than_2026_4("2025.9.1"), Some(true));
        assert_eq!(
            runtime_older_than_2026_4("2026.4.1.0-3408-ab01d1771e4"),
            Some(false)
        );
        assert_eq!(runtime_older_than_2026_4("2026.4"), Some(false));
        assert_eq!(runtime_older_than_2026_4("2027.0.0"), Some(false));
        for garbage in ["", "garbage", "2026", "x.4.1", "2026.y"] {
            assert_eq!(runtime_older_than_2026_4(garbage), None, "{garbage}");
        }

        let v = |runtime: &str| cascadia_ov_genai_shim::OvVersion {
            built_genai: "2026.4.1".into(),
            built_core: "2026.4.1".into(),
            runtime_genai: runtime.into(),
            runtime_core: runtime.into(),
        };
        assert_eq!(
            version_gate(Ok(v("2026.2.0.0-21903-abc"))).as_deref(),
            Some("needs OpenVINO GenAI 2026.4 or newer; this runtime is 2026.2.0.0-21903-abc")
        );
        assert_eq!(version_gate(Ok(v("2026.4.1.0-3408-ab01d1771e4"))), None);
        assert_eq!(version_gate(Ok(v("unknown"))), None);
        assert_eq!(version_gate(Err(OvError::Stub)), None);
    }

    #[test]
    fn check_grammar_cleans_the_backend_message() {
        let t = schema_task("s", SCHEMA_A);
        let err = check_grammar(&t, None, |_| {
            Err(OvError::Native(
                "Exception from src/x.cpp:3:\n[12:00:01] src/json_schema.cc:77: bad schema\n"
                    .into(),
            ))
        })
        .unwrap_err();
        match err {
            EngineError::InvalidConfig(m) => {
                assert_eq!(m, "response_format json_schema: bad schema")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn cb_step_failure_with_two_new_schemas_quarantines_neither() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("t1", SCHEMA_A)).unwrap();
        e.submit(schema_task("t2", SCHEMA_B)).unwrap();
        let out = e.step().unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(e.step_failures, 0, "only new schemas in the step");
        assert_eq!(e.quarantined.len(), 0);
        st.lock().unwrap().step_fails = false;
        e.submit(schema_task("t3", SCHEMA_A)).unwrap();
        e.submit(schema_task("t4", SCHEMA_B)).unwrap();
    }

    const SCHEMA_C: &str = r#"{"type":"object","properties":{"c":{"type":"boolean"}}}"#;

    #[test]
    fn cb_step_failure_with_one_schema_twice_quarantines_it() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("t1", SCHEMA_A)).unwrap();
        e.submit(schema_task("t2", SCHEMA_A)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 0);
        st.lock().unwrap().step_fails = false;
        assert!(e.submit(schema_task("t3", SCHEMA_A)).is_err());
    }

    #[test]
    fn cb_schema_failing_in_two_shared_steps_is_quarantined() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("t1", SCHEMA_A)).unwrap();
        e.submit(schema_task("t2", SCHEMA_B)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.quarantined.len(), 0);

        // A successful step re-arms the excuse.
        st.lock().unwrap().step_fails = false;
        e.submit(task("plain", 8)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.suspect_failures, 0);
        e.cancel(&"plain".to_string());

        st.lock().unwrap().step_fails = true;
        e.submit(schema_task("t3", SCHEMA_A)).unwrap();
        e.submit(schema_task("t4", SCHEMA_C)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 0);
        st.lock().unwrap().step_fails = false;
        let err = e.submit(schema_task("t5", SCHEMA_A)).unwrap_err();
        assert!(err.to_string().contains("failed to compile"), "{err}");
        e.submit(schema_task("t6", SCHEMA_C)).unwrap();
        e.submit(schema_task("t7", SCHEMA_B)).unwrap();
    }

    #[test]
    fn cb_schema_that_produced_output_loses_its_strikes() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("t1", SCHEMA_A)).unwrap();
        e.submit(schema_task("t2", SCHEMA_B)).unwrap();
        let _ = e.step().unwrap();

        // A request under schema A produces a token: A compiled fine.
        {
            let mut s = st.lock().unwrap();
            s.step_fails = false;
            assert_eq!(s.reads.len(), 2, "t3 is the third request");
            s.reads.push(VecDeque::from(vec![Ok(read_of(
                CbStatus::Running,
                1,
                "{",
                CbFinish::Unknown,
            ))]));
        }
        e.submit(schema_task("t3", SCHEMA_A)).unwrap();
        let out = e.step().unwrap();
        assert!(out.iter().all(|(_, c)| c.error.is_none()), "{out:?}");

        st.lock().unwrap().step_fails = true;
        e.submit(schema_task("t4", SCHEMA_A)).unwrap();
        e.submit(schema_task("t5", SCHEMA_C)).unwrap();
        let _ = e.step().unwrap();
        st.lock().unwrap().step_fails = false;
        // A compiled, so C was the one suspect in that step.
        e.submit(schema_task("t6", SCHEMA_A)).unwrap();
        assert!(e.submit(schema_task("t7", SCHEMA_C)).is_err());
    }

    /// One scripted read that produces a token, for request `n` (enrollment
    /// order).
    fn script_output(st: &Arc<Mutex<MockState>>, n: usize, status: CbStatus) {
        let mut s = st.lock().unwrap();
        while s.reads.len() <= n {
            s.reads.push(VecDeque::new());
        }
        s.reads[n].push_back(Ok(read_of(status, 1, "{", CbFinish::Unknown)));
    }

    #[test]
    fn cb_schema_streaming_for_one_client_is_not_blamed_for_a_failed_step() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        script_output(&st, 0, CbStatus::Running);
        e.submit(schema_task("a", SCHEMA_A)).unwrap();
        let out = e.step().unwrap();
        assert!(out.iter().all(|(_, c)| c.error.is_none()), "{out:?}");
        assert!(e.active[0].produced);

        // Client B, same schema, joins; an unrelated failure hits the step.
        e.submit(schema_task("b", SCHEMA_A)).unwrap();
        st.lock().unwrap().step_fails = true;
        let out = e.step().unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(e.quarantined.len(), 0, "the schema compiled already");
        assert_eq!(e.strikes.len(), 0, "the schema compiled already");
        assert_eq!(e.suspect_failures, 0);
        assert_eq!(e.step_failures, 1, "no suspect: counts as any failure");
        st.lock().unwrap().step_fails = false;
        e.submit(schema_task("c", SCHEMA_A)).unwrap();
    }

    #[test]
    fn cb_schema_that_produced_earlier_is_not_blamed_later() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        script_output(&st, 0, CbStatus::Finished);
        e.submit(schema_task("a", SCHEMA_A)).unwrap();
        let _ = e.step().unwrap();
        assert!(e.active.is_empty(), "the first request finished");

        // Later, A and C are new together in a failed step: only C is a
        // suspect, so C alone is refused.
        st.lock().unwrap().step_fails = true;
        e.submit(schema_task("a2", SCHEMA_A)).unwrap();
        e.submit(schema_task("c", SCHEMA_C)).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.step_failures, 0);
        assert_eq!(e.strikes.len(), 0);
        st.lock().unwrap().step_fails = false;
        e.submit(schema_task("a3", SCHEMA_A)).unwrap();
        let err = e.submit(schema_task("c2", SCHEMA_C)).unwrap_err();
        assert!(err.to_string().contains("failed to compile"), "{err}");
    }

    #[test]
    fn cb_schema_that_produced_leaves_quarantine() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        // A request admitted before its schema was quarantined produces.
        e.submit(schema_task("a", SCHEMA_A)).unwrap();
        let h = e.schema_hash(SCHEMA_A);
        e.quarantined.insert(h, Instant::now());
        script_output(&st, 0, CbStatus::Running);
        let _ = e.step().unwrap();
        assert_eq!(e.quarantined.len(), 0);
        e.submit(schema_task("b", SCHEMA_A)).unwrap();
    }

    #[test]
    fn cb_step_failure_message_is_cleaned_for_clients() {
        let err = OvError::Native(
            "Exception from src/core.cpp:12:\n[09:00:01] /build/xgrammar-src/cpp/grammar.cc:88: \
             bad grammar near {\"secret\":1}\n"
                .into(),
        );
        assert_eq!(
            step_failure_message(&err),
            "cb step failed: bad grammar near {\"secret\":1}"
        );
        assert_eq!(
            step_failure_message(&OvError::Native("mock step failure".into())),
            "cb step failed: mock step failure"
        );

        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(task("plain", 8)).unwrap();
        let out = e.step().unwrap();
        assert_eq!(out[0].1.error.as_deref(), Some(CB_BATCH_FAILED));
    }

    /// Backend text naming a build path and a schema fragment.
    const RAW_STEP_ERR: &str = "Exception from src/core.cpp:12:\n[09:00:01] \
        /build/xgrammar-src/cpp/grammar.cc:88: bad grammar near {\"secret\":1}\n";

    /// Each task's error text in `out`, by task id.
    fn errors_by_task(out: &[(TaskId, Chunk)]) -> HashMap<String, String> {
        out.iter()
            .map(|(id, c)| (id.to_string(), c.error.clone().expect("an error chunk")))
            .collect()
    }

    #[test]
    fn cb_only_the_blamed_schema_sees_the_backend_reason() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        // Schema B has compiled: its task is streaming.
        script_output(&st, 0, CbStatus::Running);
        e.submit(schema_task("other", SCHEMA_B)).unwrap();
        let _ = e.step().unwrap();
        e.submit(schema_task("suspect", SCHEMA_A)).unwrap();
        e.submit(task("plain", 8)).unwrap();
        e.submit(schema_task("json", JSON_OBJECT_PROBE)).unwrap();
        {
            let mut s = st.lock().unwrap();
            s.step_fails = true;
            s.step_err = Some(RAW_STEP_ERR.into());
        }
        let errors = errors_by_task(&e.step().unwrap());
        assert_eq!(
            errors["suspect"],
            "cb step failed: bad grammar near {\"secret\":1}"
        );
        for id in ["other", "plain", "json"] {
            assert_eq!(errors[id], CB_BATCH_FAILED, "{id}");
        }
        assert_eq!(e.quarantined.len(), 1, "the suspect's schema was blamed");
    }

    #[test]
    fn cb_the_blamed_schema_sees_the_reason_only_for_a_grammar_error() {
        // A device error in the step that compiled the suspect's schema is
        // not about that schema: its client gets the suspect text, not the
        // backend's reason.
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            step_err: Some(
                "Exception from src/plugin.cpp:7:\n/build/gpu/queue.cpp:3: device lost\n".into(),
            ),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("suspect", SCHEMA_A)).unwrap();
        e.submit(task("plain", 8)).unwrap();
        let errors = errors_by_task(&e.step().unwrap());
        assert_eq!(errors["suspect"], CB_BATCH_SCHEMA_SUSPECT);
        assert_eq!(errors["plain"], CB_BATCH_FAILED);
        assert_eq!(e.quarantined.len(), 1, "the suspect's schema was blamed");
    }

    #[test]
    fn cb_two_suspects_do_not_see_the_backend_reason() {
        let st = Arc::new(Mutex::new(MockState {
            step_fails: true,
            step_err: Some(RAW_STEP_ERR.into()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        e.submit(schema_task("a", SCHEMA_A)).unwrap();
        e.submit(schema_task("b", SCHEMA_B)).unwrap();
        e.submit(task("plain", 8)).unwrap();
        let errors = errors_by_task(&e.step().unwrap());
        assert_eq!(e.strikes.len(), 2);
        // Either schema may have failed the step, so neither client is
        // shown the backend's reason: it may come from the other's schema.
        for id in ["a", "b"] {
            assert_eq!(errors[id], CB_BATCH_SCHEMA_SUSPECT, "{id}");
        }
        assert_eq!(errors["plain"], CB_BATCH_FAILED);
        assert!(
            errors.values().all(|m| !m.contains("grammar")),
            "{errors:?}"
        );
    }

    #[test]
    fn cb_failure_with_no_suspect_gives_every_client_the_fixed_text() {
        let st = Arc::new(Mutex::new(MockState {
            default_read: Some(idle_read()),
            ..Default::default()
        }));
        let mut e = mock_engine(&st);
        script_output(&st, 0, CbStatus::Running);
        e.submit(schema_task("compiled", SCHEMA_A)).unwrap();
        let _ = e.step().unwrap();
        e.submit(task("plain", 8)).unwrap();
        e.submit(schema_task("json", JSON_OBJECT_PROBE)).unwrap();
        e.submit(schema_task("same", SCHEMA_A)).unwrap();
        {
            let mut s = st.lock().unwrap();
            s.step_fails = true;
            s.step_err = Some(RAW_STEP_ERR.into());
        }
        let out = e.step().unwrap();
        assert_eq!(out.len(), 4);
        for (id, msg) in errors_by_task(&out) {
            assert_eq!(msg, CB_BATCH_FAILED, "{id}");
        }
        assert_eq!(e.step_failures, 1);
    }

    #[test]
    fn cb_schema_hash_is_keyed_per_engine() {
        let st = Arc::new(Mutex::new(MockState::default()));
        let (e1, e2) = (mock_engine(&st), mock_engine(&st));
        assert_eq!(e1.schema_hash(SCHEMA_A), e1.schema_hash(SCHEMA_A));
        assert_ne!(e1.schema_hash(SCHEMA_A), e1.schema_hash(SCHEMA_B));
        assert_ne!(
            e1.schema_hash(SCHEMA_A),
            e2.schema_hash(SCHEMA_A),
            "two engines must not share hash keys"
        );
        use std::hash::{Hash, Hasher};
        let mut fixed = std::hash::DefaultHasher::new();
        SCHEMA_A.hash(&mut fixed);
        assert_ne!(
            e1.schema_hash(SCHEMA_A),
            fixed.finish(),
            "not the unkeyed hash"
        );
    }

    #[test]
    fn strike_set_counts_within_the_window_and_is_bounded() {
        let now = Instant::now();
        let mut s = StrikeSet::default();
        assert_eq!(s.strike(1, now), 1);
        assert_eq!(s.strike(1, now), 2);
        let later = now + std::time::Duration::from_secs(11 * 60);
        assert_eq!(s.strike(1, later), 1, "an old strike has expired");
        let mut s = StrikeSet::default();
        for h in 0..300u64 {
            s.strike(h, now + std::time::Duration::from_millis(h));
        }
        assert_eq!(s.len(), 256);
        assert_eq!(s.strike(0, now + std::time::Duration::from_secs(1)), 1);
        assert_eq!(s.strike(299, now + std::time::Duration::from_secs(1)), 2);
    }

    /// A CB engine whose plain warmup request finishes at once, with
    /// `probe` scripted as the reads of each probe attempt in turn.
    fn cb_warmup_with(
        probe: Vec<Vec<Result<CbRead, OvError>>>,
    ) -> (OvGenaiCbEngine, Arc<Mutex<MockState>>) {
        let st = Arc::new(Mutex::new(MockState::default()));
        {
            let mut s = st.lock().unwrap();
            s.reads.push(VecDeque::from(vec![Ok(read_of(
                CbStatus::Finished,
                1,
                "hi",
                CbFinish::Stop,
            ))]));
            for reads in probe {
                s.reads.push(VecDeque::from(reads));
            }
        }
        let mut e = mock_engine(&st);
        e.warmup();
        (e, st)
    }

    #[test]
    fn cb_warmup_probe_verdicts() {
        let finished = |text: &str| Ok(read_of(CbStatus::Finished, 6, text, CbFinish::Stop));
        let (e, _) = cb_warmup_with(vec![vec![finished("{\"k\":\"zq\"}")]]);
        assert_eq!(e.structured_unavailable, None);

        let (e, _) = cb_warmup_with(vec![vec![finished("Hello there!")]]);
        assert_eq!(
            e.structured_unavailable.as_deref(),
            Some("the output constraint was not applied: Hello there!")
        );

        // A probe the scheduler dropped proves nothing either way.
        let (e, _) = cb_warmup_with(vec![vec![Ok(read_of(
            CbStatus::Ignored,
            2,
            "{\"k",
            CbFinish::Unknown,
        ))]]);
        assert_eq!(e.structured_unavailable, None);

        // A failed attempt is retried once, under a fresh request id.
        let (e, st) = cb_warmup_with(vec![
            vec![Err(OvError::Native("transient".into()))],
            vec![finished("{\"k\":\"zq\"}")],
        ]);
        assert_eq!(e.structured_unavailable, None);
        let st = st.lock().unwrap();
        assert_eq!(
            st.added,
            vec![u64::MAX, PROBE_REQUEST_ID, PROBE_REQUEST_ID - 1]
        );
        drop(st);

        let (e, _) = cb_warmup_with(vec![
            vec![Err(OvError::Native("first".into()))],
            vec![Err(OvError::Native(
                "[01:02:03] src/x.cpp:9: second".into(),
            ))],
        ]);
        assert_eq!(e.structured_unavailable.as_deref(), Some("second"));
    }

    #[test]
    fn stateful_finish_with_a_grammar_reports_length_when_either_count_hits_the_cap() {
        // A whitespace run re-tokenises into far fewer tokens than were
        // generated: the generated count still says the cap was reached.
        assert_eq!(stateful_finish(100, 4096, 4096, true), FinishReason::Length);
        assert_eq!(
            stateful_finish(4096, 4096, 4096, true),
            FinishReason::Length
        );
        assert_eq!(stateful_finish(50, 50, 4096, true), FinishReason::Stop);
        // No generated count (perf metrics unavailable): the tokenizer decides.
        assert_eq!(stateful_finish(4096, 0, 4096, true), FinishReason::Length);
        assert_eq!(stateful_finish(50, 0, 4096, true), FinishReason::Stop);
    }

    #[test]
    fn stateful_finish_without_a_grammar_uses_the_tokenizer_count_only() {
        // The generated count can read as the cap when EOS fires early, so a
        // plain request never consults it.
        assert_eq!(stateful_finish(50, 4096, 4096, false), FinishReason::Stop);
        assert_eq!(stateful_finish(100, 4096, 4096, false), FinishReason::Stop);
        assert_eq!(stateful_finish(4096, 0, 4096, false), FinishReason::Length);
        assert_eq!(stateful_finish(4096, 50, 4096, false), FinishReason::Length);
        assert_eq!(stateful_finish(50, 0, 4096, false), FinishReason::Stop);
    }

    #[test]
    fn ov_properties_reach_plugin_config() {
        // Intercepts the PluginConfig the builder hands to
        // ov::Core::compile_model: every injected (key, value) — the CLI's
        // --ov-* perf flags — must land in its entries, alongside the
        // pre-existing cache_dir wiring.
        let b = OvGenaiBuilder::new("/x", "GPU")
            .with_cache_dir("/tmp/ovc")
            .with_ov_properties(vec![
                ("PERFORMANCE_HINT".into(), "LATENCY".into()),
                ("INFERENCE_PRECISION_HINT".into(), "f16".into()),
            ]);
        let cfg = b.build_plugin_config();
        let has = |k: &str, v: &str| cfg.entries.iter().any(|(ek, ev)| ek == k && ev == v);
        assert!(has("CACHE_DIR", "/tmp/ovc"));
        assert!(has("PERFORMANCE_HINT", "LATENCY"));
        assert!(has("INFERENCE_PRECISION_HINT", "f16"));
    }

    #[tokio::test]
    async fn rejects_peer_layout() {
        let mut b = OvGenaiBuilder::new("/x", "CPU");
        let layout = PeerLayout::first_of(PeerEndpoint::new("h", 1));
        let res = b.connect(layout).await;
        assert!(matches!(res, Err(EngineError::PeerRejected(_))));
    }

    #[tokio::test]
    async fn rejects_multi_stage_shard() {
        let mut b = OvGenaiBuilder::new("/some/missing/path", "CPU");
        let multi = ShardSpec {
            model_id: "x".into(),
            layer_start: 0,
            layer_end: 0,
            total_layers: 0,
            device: "CPU".into(),
            is_first_stage: true,
            is_last_stage: false,
            tp_size: 1,
            tp_rank: 0,
        };
        let res = b.load(multi).await;
        assert!(matches!(res, Err(EngineError::ShardRejected(_))));
    }

    #[tokio::test]
    async fn rejects_draft_and_prompt_lookup_combo() {
        let mut b = OvGenaiBuilder::new("/some/missing/path", "CPU")
            .with_draft("/draft/missing", "CPU", 5)
            .with_prompt_lookup(3);
        let shard = ShardSpec::single_stage("model", "CPU");
        let res = b.load(shard).await;
        assert!(matches!(res, Err(EngineError::InvalidConfig(_))));
    }

    #[tokio::test]
    async fn rejects_missing_model() {
        let mut b = OvGenaiBuilder::new("/definitely/does/not/exist", "CPU");
        let shard = ShardSpec::single_stage("model", "CPU");
        let res = b.load(shard).await;
        assert!(matches!(res, Err(EngineError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn cb_rejects_draft_and_prompt_lookup() {
        for b in [
            OvGenaiBuilder::new("/missing", "CPU")
                .with_continuous_batching(CbSchedulerConfig::default())
                .with_draft("/draft/missing", "CPU", 5),
            OvGenaiBuilder::new("/missing", "CPU")
                .with_continuous_batching(CbSchedulerConfig::default())
                .with_prompt_lookup(3),
        ] {
            let mut b = b;
            let shard = ShardSpec::single_stage("model", "CPU");
            let err = b.load(shard).await.err().expect("must reject");
            let msg = err.to_string();
            assert!(
                matches!(err, EngineError::InvalidConfig(_)) && msg.contains("--cb"),
                "{msg}"
            );
        }
    }

    #[tokio::test]
    async fn cb_rejects_vlm_layout() {
        let dir = tempfile::tempdir().unwrap();
        for f in [
            "openvino_language_model.xml",
            "openvino_tokenizer.xml",
            "openvino_detokenizer.xml",
        ] {
            std::fs::write(dir.path().join(f), "").unwrap();
        }
        let mut b = OvGenaiBuilder::new(dir.path().to_str().unwrap(), "CPU")
            .with_continuous_batching(CbSchedulerConfig::default());
        let shard = ShardSpec::single_stage("model", "CPU");
        let err = b.load(shard).await.err().expect("must reject");
        assert!(err.to_string().contains("VLM-layout"), "{err}");
    }

    /// Stub builds surface a Backend error from CbPipeline::new — proves the
    /// CB branch is reached with a valid LM-layout dir (and on the AI-PC lane
    /// with `--features openvino` this same path compiles a real pipeline).
    #[cfg(not(feature = "openvino"))]
    #[tokio::test]
    async fn cb_load_reaches_pipeline_construction() {
        let dir = tempfile::tempdir().unwrap();
        for f in [
            "openvino_model.xml",
            "openvino_tokenizer.xml",
            "openvino_detokenizer.xml",
        ] {
            std::fs::write(dir.path().join(f), "").unwrap();
        }
        let mut b = OvGenaiBuilder::new(dir.path().to_str().unwrap(), "CPU")
            .with_continuous_batching(CbSchedulerConfig::default());
        let shard = ShardSpec::single_stage("model", "CPU");
        let err = b.load(shard).await.err().expect("stub must error");
        assert!(
            matches!(err, EngineError::Backend(_)),
            "expected stub Backend error, got {err}"
        );
    }

    #[tokio::test]
    async fn build_before_load_fails() {
        let b = Box::new(OvGenaiBuilder::new("/x", "CPU"));
        let res = b.build();
        assert!(matches!(res, Err(EngineError::NotLoaded)));
    }

    #[test]
    fn tokenizer_irs_are_required() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("openvino_model.xml"), "").unwrap();
        // Model IR present, tokenizer IRs absent: this used to serve empty strings.
        let err = check_tokenizer_irs(&dir.path().to_string_lossy())
            .unwrap_err()
            .to_string();
        assert!(err.contains("openvino_tokenizer.xml"), "{err}");
        assert!(err.contains("openvino_detokenizer.xml"), "{err}");

        for f in ["openvino_tokenizer.xml", "openvino_detokenizer.xml"] {
            std::fs::write(dir.path().join(f), "").unwrap();
        }
        assert!(check_tokenizer_irs(&dir.path().to_string_lossy()).is_ok());
    }

    #[test]
    fn a_staged_tree_names_the_engine_that_can_serve_it() {
        // The (file, key, value) triples each exporter actually writes — not a
        // fabricated combination. export_shards.py and the gemma4 exporters write
        // pipeline_config.json; the surgery exporters write manifest.json only.
        for (file, key, value, engine) in [
            (
                "pipeline_config.json",
                "export_version",
                "v5_canonical_inputs",
                "ov-runtime",
            ),
            (
                "pipeline_config.json",
                "export_version",
                "gemma4_cached_v1",
                "gemma4",
            ),
            (
                "pipeline_config.json",
                "export_version",
                "gemma4_cached_v1.1",
                "gemma4",
            ),
            ("manifest.json", "arch", "qwen3_5_moe", "qwen35"),
            ("manifest.json", "arch", "qwen3_5", "qwen35"),
            ("manifest.json", "arch", "minimax_m2", "sparse-moe"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(file), format!("{{\"{key}\": \"{value}\"}}")).unwrap();
            let err = check_tokenizer_irs(&dir.path().to_string_lossy())
                .unwrap_err()
                .to_string();
            assert!(err.contains("staged tree"), "{err}");
            assert!(err.contains(&format!("--engine {engine}")), "{err}");
        }
    }
}
