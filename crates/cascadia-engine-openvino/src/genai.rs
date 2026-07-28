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
//! Token counting uses the LlmPipeline tokenizer (per the c57 lesson —
//! `perf_metrics` defaults to `max_new_tokens` when EOS fires early on
//! short greedy decodes, which inflates reported tok/s).

use std::path::PathBuf;
use std::time::Instant;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{
    CbFinish, CbHandle, CbPipeline, CbSchedulerConfig, CbStatus, Error as OvError, GenConfig,
    LlmPipeline, PluginConfig,
};
use cascadia_types::{
    Chunk, FinishReason, GenerationTask, LoadProgress, PeerLayout, SamplingParams, ShardSpec,
    TaskId,
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
        (_, Some(a)) if a == "qwen3_5_moe" => Some("qwen36-moe"),
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
                pipe,
                prompt_pretemplated: self.prompt_pretemplated,
                next_request_id: 0,
                max_tokens_default: 256,
                idle_steps: 0,
                step_failures: 0,
            }));
        }
        let pipe = self.pipe.ok_or(EngineError::NotLoaded)?;
        Ok(Box::new(OvGenaiEngine {
            pipe,
            speculative_k: self.speculative_k,
            prompt_lookup_ngram: self.prompt_lookup_ngram,
            prompt_pretemplated: self.prompt_pretemplated,
            pending: Vec::new(),
            max_tokens_default: 256,
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

pub struct OvGenaiEngine {
    pipe: LlmPipeline,
    speculative_k: u32,
    prompt_lookup_ngram: u32,
    prompt_pretemplated: bool,
    pending: Vec<GenerationTask>,
    max_tokens_default: u32,
}

impl Engine for OvGenaiEngine {
    fn warmup(&mut self) {
        let cfg = self.gen_config(
            /*max_tokens*/ 4,
            /*temperature*/ 0.0,
            /*enable_thinking*/ false,
            &SamplingParams::default(),
        );
        match self.pipe.generate("Hi", &cfg) {
            Ok(_) => info!(
                spec_k = self.speculative_k,
                prompt_lookup = self.prompt_lookup_ngram,
                "ov-genai warmup ok"
            ),
            Err(err) => warn!(error = %err, "ov-genai warmup failed"),
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
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
        );

        let started = Instant::now();
        let result = match self.pipe.generate(&task.prompt, &cfg) {
            Ok(r) => r,
            Err(err) => {
                warn!(task = %task.task_id, error = %err, "generate failed");
                let final_chunk = Chunk::final_marker(task.task_id.clone(), "");
                return Ok(vec![(task.task_id, final_chunk)]);
            }
        };
        let elapsed = started.elapsed().as_secs_f64();

        let mut text = result.text;
        if let Some(stripped) = text.strip_prefix(&task.prompt) {
            text = stripped.trim_start().to_string();
        }
        // Prefer tokenizer-based count; fall back to perf_metrics value.
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
        let finish = if tokens >= max_new {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
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

/// Per-request generation state on the continuous-batching engine.
struct ActiveCbTask {
    task_id: TaskId,
    handle: CbHandle,
    total_tokens: u32,
    max_tokens: u32,
    prompt_tokens: Option<u32>,
    started: Instant,
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
    pipe: CbPipeline,
    prompt_pretemplated: bool,
    next_request_id: u64,
    max_tokens_default: u32,
    /// Consecutive iterations in which the whole batch produced nothing.
    idle_steps: u32,
    /// Consecutive `pipe.step()` failures. Past MAX_STEP_FAILURES the pipeline
    /// is treated as unusable and `submit()` stops admitting work.
    step_failures: u32,
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
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.active.iter().any(|t| t.task_id == task.task_id) {
            return Ok(());
        }
        // A pipeline that has failed every recent scheduler iteration will fail
        // this request too. Admitting it means paying add_request and a step
        // before returning the same 503, forever, with nothing distinguishing a
        // transient hiccup from a dead device. Refuse up front instead; a
        // single successful step clears this.
        if self.step_failures >= MAX_STEP_FAILURES {
            return Err(EngineError::Backend(format!(
                "cb pipeline unusable: {} consecutive scheduler iterations failed; \
                 restart the worker",
                self.step_failures
            )));
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
        );
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let handle = self
            .pipe
            .add_request(request_id, &task.prompt, &cfg)
            .map_err(map_ov_err)?;
        self.active.push(ActiveCbTask {
            prompt_tokens: self.pipe.count_tokens(&task.prompt),
            task_id: task.task_id,
            handle,
            total_tokens: 0,
            max_tokens,
            started: Instant::now(),
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
            self.step_failures += 1;
            warn!(
                error = %err,
                in_flight = self.active.len(),
                consecutive = self.step_failures,
                "cb step failed"
            );
            let failed = self
                .active
                .drain(..)
                .map(|t| {
                    let chunk = Chunk::error(t.task_id.clone(), format!("cb step failed: {err}"));
                    (t.task_id, chunk)
                })
                .collect();
            return Ok(failed);
        }
        self.step_failures = 0;

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

/// Shared `GenConfig` assembly for both ov-genai engines (the CB engine
/// passes 0 for the speculative knobs — the CB scheduler owns batching).
fn build_gen_config(
    speculative_k: u32,
    prompt_lookup_ngram: u32,
    prompt_pretemplated: bool,
    max_tokens: u32,
    temperature: f32,
    enable_thinking: bool,
    sampling: &SamplingParams,
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
    ) -> GenConfig {
        build_gen_config(
            self.speculative_k,
            self.prompt_lookup_ngram,
            self.prompt_pretemplated,
            max_tokens,
            temperature,
            enable_thinking,
            sampling,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_engine::EngineError;
    use cascadia_types::{PeerEndpoint, PeerLayout, ShardSpec};

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
            ("manifest.json", "arch", "qwen3_5_moe", "qwen36-moe"),
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
