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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cascadia_engine::{
    Builder, Engine, EngineError, EngineResult, InflightCancel, LoadStream, PendingLoad,
    SubmitAdmission,
};
use cascadia_ov_genai_shim::{
    CbFinish, CbHandle, CbPipeline, CbRead, CbSchedulerConfig, CbStatus, Error as OvError,
    GenConfig, GenResult, LlmPipeline, PluginConfig, PollFn,
};
use cascadia_types::{
    Chunk, FinishReason, GenerationTask, LoadProgress, PeerLayout, SamplingParams, ShardSpec,
    TaskId,
};
use futures::stream;
use tracing::{info, warn};

use crate::genai_worker::{
    lock_pipe, run_job, Done, Event, GenJob, GenParams, Job, Outcome, Recv, SharedPipe,
    StatefulWorker,
};

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
    /// The server's known context window. Without `--cb` one `generate()`
    /// call makes the whole answer, so its limit is cut to what the prompt
    /// leaves of this window.
    pub context_window: Option<u32>,
    /// End a running stateful generation at a `stop` match. Default
    /// true; off when the server has a marker dialect, whose filtered text is
    /// not what the engine sees (the CLI decides).
    pub engine_stop_check: bool,
    /// Stream the stateful path: hand out each streamer write's text as it
    /// is generated (see the `stream_writes` method). Off by default.
    pub stream_writes: bool,
    pipe: Option<Box<dyn StatefulPipe>>,
    cb_pipe: Option<CbPipeline>,
    /// The shard's `model_id` from `load`: the `model` label of the engine's
    /// own metrics.
    model_id: String,
}

impl Default for OvGenaiBuilder {
    fn default() -> Self {
        Self::new(String::new(), String::new())
    }
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
            context_window: None,
            engine_stop_check: true,
            stream_writes: false,
            pipe: None,
            cb_pipe: None,
            model_id: String::new(),
        }
    }

    /// Whether the stateful engine ends a running generation at a `stop`
    /// match (see [`Self::engine_stop_check`]).
    pub fn with_engine_stop_check(mut self, on: bool) -> Self {
        self.engine_stop_check = on;
        self
    }

    /// Stream the default (stateful) path. On, `step()` returns the text as
    /// token chunks while the generation runs, one streamer write late (only
    /// text two consecutive decodes agree on)
    /// (on a thread that owns the pipeline), a zero-token progress chunk
    /// when nothing new came within [`STEP_POLL`], and a final chunk with
    /// the rest of the answer and its `completion_tokens`. The deltas plus
    /// the final's text are the answer the non-streamed path returns. Off,
    /// the whole answer comes on one final chunk, as before. `--cb` streams
    /// either way. A task whose client does not stream
    /// ([`GenerationTask::client_streams`] false) gets that one final chunk
    /// with the exact answer either way (it still runs on the worker, with
    /// progress chunks while it waits).
    pub fn stream_writes(mut self, on: bool) -> Self {
        self.stream_writes = on;
        self
    }

    /// The server's known context window (see [`Self::context_window`]).
    pub fn with_context_window(mut self, window: Option<u32>) -> Self {
        self.context_window = window;
        self
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
        self.model_id = shard.model_id.clone();
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

        self.pipe = Some(Box::new(pipe));
        Ok(Box::pin(stream::iter(progress)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        if let Some(pipe) = self.cb_pipe {
            return Ok(Box::new(OvGenaiCbEngine {
                active: Vec::new(),
                pipe: Box::new(pipe),
                prompt_pretemplated: self.prompt_pretemplated,
                next_request_id: 0,
                max_tokens_default: 256,
                idle_steps: 0,
                step_failures: 0,
            }));
        }
        let pipe = self.pipe.ok_or(EngineError::NotLoaded)?;
        let params = GenParams {
            speculative_k: self.speculative_k,
            prompt_lookup_ngram: self.prompt_lookup_ngram,
            prompt_pretemplated: self.prompt_pretemplated,
            max_tokens_default: 256,
            context_window: self.context_window,
            engine_stop_check: self.engine_stop_check,
            stream_text: self.stream_writes,
        };
        Ok(Box::new(OvGenaiEngine::new(pipe, params, self.model_id)?))
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

/// The answer limit for one stateful `generate()` call: `max_new`, cut to
/// what is left of the server's known context window after the prompt, so
/// the prompt plus the answer never run past it. Unchanged when either count
/// is unknown, or when the prompt alone reaches the window (the pipeline
/// accepted it, so that window is not its real limit).
pub(crate) fn stateful_max_new(
    max_new: u32,
    prompt_tokens: Option<u32>,
    window: Option<u32>,
) -> u32 {
    match (prompt_tokens, window) {
        (Some(prompt), Some(window)) if prompt < window => max_new.min(window - prompt),
        _ => max_new,
    }
}

/// The slice of [`LlmPipeline`] the stateful engine uses: a test seam like
/// [`CbPipe`], so `step()`'s cancel and stop handling runs without OpenVINO.
pub(crate) trait StatefulPipe: Send {
    fn count_tokens(&self, text: &str) -> Option<u32>;
    fn generate(&self, prompt: &str, cfg: &GenConfig) -> Result<GenResult, OvError>;
    fn generate_polled(
        &self,
        prompt: &str,
        cfg: &GenConfig,
        tail_tokens: usize,
        stream_text: bool,
        poll: &mut PollFn<'_>,
    ) -> Result<GenResult, OvError>;
}

impl StatefulPipe for LlmPipeline {
    fn count_tokens(&self, text: &str) -> Option<u32> {
        LlmPipeline::count_tokens(self, text)
    }
    fn generate(&self, prompt: &str, cfg: &GenConfig) -> Result<GenResult, OvError> {
        LlmPipeline::generate(self, prompt, cfg)
    }
    fn generate_polled(
        &self,
        prompt: &str,
        cfg: &GenConfig,
        tail_tokens: usize,
        stream_text: bool,
        poll: &mut PollFn<'_>,
    ) -> Result<GenResult, OvError> {
        LlmPipeline::generate_polled(self, prompt, cfg, tail_tokens, stream_text, poll)
    }
}

/// Calls [`InflightCancel::end`] when dropped, so every path out of a begun
/// task (success, error, a panic unwinding through `step`) frees the handle.
struct EndInflight(Arc<InflightCancel>);

impl Drop for EndInflight {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// The final chunk for a cancelled task: empty, `Cancelled`, with
/// the prompt's token count when it is known, so the runner books the prompt.
/// No text and no `n_tokens`: no TTFT sample, 0 generated tokens.
fn cancelled_chunk(task_id: &TaskId, prompt_tokens: Option<u32>) -> Chunk {
    let mut chunk =
        Chunk::final_marker(task_id.clone(), "").with_finish_reason(FinishReason::Cancelled);
    chunk.prompt_tokens = prompt_tokens;
    chunk
}

/// The stateful engine's refusals in `submit` that depend on the task alone.
///
/// No forced-prefix path in the ov-genai pipelines: accepting a resumed task would regenerate
/// from scratch and report success on a corrupt "resume". The enterprise shard backend declines
/// these pre-submit (its shard_is_ov_genai guard); this is the engine-side backstop for any caller
/// that misses that guard.
fn check_stateful_task(task: &GenerationTask) -> EngineResult<()> {
    if task.resume_ids().is_some() {
        return Err(EngineError::Backend(
            "resume_unsupported: ov-genai has no forced-prefix path".to_string(),
        ));
    }
    Ok(())
}

/// [`check_stateful_task`] without the engine lock. A generation holds that lock for the whole
/// answer, so the runner checks a submit that finds the engine busy with this and queues it,
/// instead of holding the caller (and its response headers) until the generation ends.
///
/// It also reports the engine's pending queue against [`crate::dist_spec::MAX_PENDING_TASKS`], so
/// the runner refuses a submit that would only be refused at drain before any response starts.
struct StatefulAdmission {
    /// [`OvGenaiEngine::pending_len`].
    pending: Arc<AtomicUsize>,
}

impl SubmitAdmission for StatefulAdmission {
    fn check(&self, task: &GenerationTask) -> EngineResult<()> {
        check_stateful_task(task)
    }

    fn pending_load(&self) -> Option<PendingLoad> {
        Some(PendingLoad {
            pending: self.pending.load(Ordering::SeqCst),
            cap: crate::dist_spec::MAX_PENDING_TASKS,
        })
    }
}

/// How long a streaming `step()` waits for the generation's next event before
/// it returns a progress chunk, so the engine lock is free at least this
/// often (a queued submit or a cancel gets in) and a long prefill still
/// counts as progress for the runner.
pub const STEP_POLL: Duration = Duration::from_millis(50);

pub struct OvGenaiEngine {
    pipe: SharedPipe,
    params: GenParams,
    pending: Vec<GenerationTask>,
    /// `pending.len()`, kept current in `submit`, `step` and `cancel` for the
    /// [`StatefulAdmission`], which reads it without the engine lock.
    pending_len: Arc<AtomicUsize>,
    /// The runner signals a cancel here while `step()` holds the engine
    /// lock (or, streaming, while the worker runs the generation); the poll
    /// closure reads it once per streamer write.
    inflight: Arc<InflightCancel>,
    /// Set once the first poll of the engine's life has told which thread
    /// runs ov-genai's streamer (logged once, for the hardware check).
    streamer_thread_logged: bool,
    /// With [`OvGenaiBuilder::stream_writes`]: the worker and its task.
    stream: Option<Streaming>,
    /// The `model` label of the engine's own metrics.
    model_id: String,
}

/// The streaming engine's worker and the task it runs.
struct Streaming {
    worker: StatefulWorker,
    running: Option<Running>,
}

/// The task the worker is generating.
struct Running {
    task_id: TaskId,
    prompt_tokens: Option<u32>,
    prompt_reported: bool,
    /// Ends the task on the inflight handle when the task is over.
    _end: EndInflight,
}

impl OvGenaiEngine {
    fn new(pipe: Box<dyn StatefulPipe>, params: GenParams, model_id: String) -> EngineResult<Self> {
        let pipe: SharedPipe = Arc::new(Mutex::new(pipe));
        let inflight = Arc::new(InflightCancel::new());
        let stream = if params.stream_text {
            Some(Streaming {
                worker: spawn_worker(&pipe, &inflight)?,
                running: None,
            })
        } else {
            None
        };
        Ok(Self {
            pipe,
            params,
            pending: Vec::new(),
            pending_len: Arc::new(AtomicUsize::new(0)),
            inflight,
            streamer_thread_logged: false,
            stream,
            model_id,
        })
    }

    /// Publish `pending.len()` to the [`StatefulAdmission`].
    fn sync_pending_len(&self) {
        self.pending_len.store(self.pending.len(), Ordering::SeqCst);
    }

    /// Take the next queued task and begin it on the inflight handle. `Err`
    /// carries the final chunk of a task cancelled before it began.
    #[allow(clippy::result_large_err)]
    fn begin_next(&mut self) -> Option<Result<(GenerationTask, EndInflight), (TaskId, Chunk)>> {
        if self.pending.is_empty() {
            return None;
        }
        let task = self.pending.remove(0);
        self.sync_pending_len();
        let end = EndInflight(self.inflight.clone());
        if self.inflight.begin(&task.task_id) {
            info!(task = %task.task_id, tokens = 0u64, elapsed_s = 0.0, "task cancelled");
            let chunk = cancelled_chunk(&task.task_id, None);
            return Some(Err((task.task_id, chunk)));
        }
        Some(Ok((task, end)))
    }

    /// The whole generation inside this call, one final chunk.
    fn step_inline(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let (task, _end) = match self.begin_next() {
            None => return Ok(Vec::new()),
            Some(Err(cancelled)) => return Ok(vec![cancelled]),
            Some(Ok(begun)) => begun,
        };
        let done = {
            let pipe = lock_pipe(&self.pipe);
            run_job(
                &**pipe,
                &self.params,
                &self.inflight,
                &task,
                !self.streamer_thread_logged,
                &mut |_| {},
            )
        };
        let chunk = final_chunk(
            &self.inflight,
            &mut self.streamer_thread_logged,
            &self.model_id,
            &task.task_id,
            done,
            None,
        );
        Ok(vec![(task.task_id, chunk)])
    }

    /// Start the next task on the worker if none runs, then return what it
    /// produced within [`STEP_POLL`]: its deltas, its final, or a progress
    /// chunk.
    fn step_streaming(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let idle = self.stream.as_ref().is_some_and(|s| s.running.is_none());
        if idle {
            let (task, end) = match self.begin_next() {
                None => return Ok(Vec::new()),
                Some(Err(cancelled)) => return Ok(vec![cancelled]),
                Some(Ok(begun)) => begun,
            };
            let task_id = task.task_id.clone();
            // A client that does not stream takes the one-shot route on the
            // worker: no per-write decode, one final chunk with the exact
            // visible text, whatever the stream would have sent.
            let params = GenParams {
                stream_text: task.client_streams,
                ..self.params.clone()
            };
            let job = Job::Generate(Box::new(GenJob {
                task,
                params,
                probe_thread: !self.streamer_thread_logged,
            }));
            let Some(stream) = self.stream.as_mut() else {
                return Ok(Vec::new());
            };
            if let Err(job) = stream.worker.send(job) {
                // The thread is gone: a new one takes the job.
                warn!(task = %task_id, "ov-genai worker thread is gone; starting a new one");
                let sent = spawn_worker(&self.pipe, &self.inflight)
                    .map_err(|e| e.to_string())
                    .and_then(|worker| {
                        stream.worker = worker;
                        stream
                            .worker
                            .send(job)
                            .map_err(|_| "the new worker thread is gone too".to_string())
                    });
                if let Err(err) = sent {
                    let chunk = Chunk::error(
                        task_id.clone(),
                        format!("ov-genai worker thread unavailable: {err}"),
                    );
                    return Ok(vec![(task_id, chunk)]);
                }
            }
            stream.running = Some(Running {
                task_id,
                prompt_tokens: None,
                prompt_reported: false,
                _end: end,
            });
        }
        let Some(stream) = self.stream.as_mut() else {
            return Ok(Vec::new());
        };
        let Some(running) = stream.running.as_mut() else {
            return Ok(Vec::new());
        };
        let task_id = running.task_id.clone();
        let mut out = Vec::new();
        let mut wait = Some(STEP_POLL);
        loop {
            match stream.worker.next(wait.take()) {
                Recv::Idle => break,
                Recv::Event(Event::Started { prompt_tokens }) => {
                    running.prompt_tokens = prompt_tokens;
                }
                Recv::Event(Event::Delta { text, n_tokens }) => {
                    let mut chunk =
                        Chunk::token(task_id.clone(), 0, text).with_n_tokens(n_tokens as u32);
                    if let Some(n) = running.prompt_tokens {
                        chunk = chunk.with_first_prompt_tokens(&mut running.prompt_reported, n);
                    }
                    out.push((task_id.clone(), chunk));
                }
                Recv::Event(Event::Done(mut done)) => {
                    done.prompt_tokens = done.prompt_tokens.or(running.prompt_tokens);
                    let chunk = final_chunk(
                        &self.inflight,
                        &mut self.streamer_thread_logged,
                        &self.model_id,
                        &task_id,
                        *done,
                        Some(&*running),
                    );
                    out.push((task_id.clone(), chunk));
                    // Ends the task on the inflight handle.
                    stream.running = None;
                    break;
                }
                Recv::Gone => {
                    warn!(task = %task_id, "ov-genai worker thread exited mid-generation");
                    out.push((
                        task_id.clone(),
                        Chunk::error(task_id.clone(), "ov-genai worker thread exited"),
                    ));
                    stream.running = None;
                    // The next task gets a new thread.
                    match spawn_worker(&self.pipe, &self.inflight) {
                        Ok(worker) => stream.worker = worker,
                        Err(err) => warn!(error = %err, "ov-genai worker thread: respawn failed"),
                    }
                    break;
                }
            }
        }
        if out.is_empty() {
            out.push((task_id.clone(), Chunk::progress(task_id)));
        }
        Ok(out)
    }
}

impl Drop for OvGenaiEngine {
    fn drop(&mut self) {
        // A streaming generation still running ends at its next write; the
        // worker is then joined.
        if let Some(stream) = self.stream.as_ref() {
            if stream.running.is_some() {
                self.inflight.shutdown();
            }
        }
    }
}

fn spawn_worker(pipe: &SharedPipe, inflight: &Arc<InflightCancel>) -> EngineResult<StatefulWorker> {
    StatefulWorker::spawn(pipe.clone(), inflight.clone())
        .map_err(|e| EngineError::Backend(format!("ov-genai worker thread: {e}")))
}

/// The final chunk of a generation that ran to `done`: cancelled, failed, or
/// the answer. Inline (`running` is `None`) the answer is the whole text with
/// its token count, as the non-streamed path always returned it; streaming,
/// it is what the deltas did not carry, the carried tokens and the answer's
/// `completion_tokens`.
fn final_chunk(
    inflight: &InflightCancel,
    streamer_thread_logged: &mut bool,
    model_id: &str,
    task_id: &TaskId,
    done: Done,
    running: Option<&Running>,
) -> Chunk {
    if let Some(threaded_streamer) = done.threaded_streamer {
        *streamer_thread_logged = true;
        info!(threaded_streamer, "ov-genai streamer thread");
    }
    // A generation the job finished stays finished when the flag went up
    // after it; a failure with the flag up ends as cancelled.
    let cancelled = |held_tokens: usize| {
        info!(
            task = %task_id,
            tokens = done.polled_tokens,
            elapsed_s = done.elapsed_s,
            "task cancelled"
        );
        let chunk = cancelled_chunk(task_id, done.prompt_tokens);
        // Streaming: the held writes' tokens, so the API's running sum
        // counts every token generated.
        if held_tokens > 0 {
            chunk.with_n_tokens(held_tokens as u32)
        } else {
            chunk
        }
    };
    let f = match done.outcome {
        Outcome::Cancelled { held_tokens } => return cancelled(held_tokens),
        Outcome::Failed { held_tokens, .. } if inflight.cancelled() => {
            return cancelled(held_tokens)
        }
        Outcome::Failed { error: err, .. } => {
            warn!(task = %task_id, error = %err, "generate failed");
            // Chunk::error, not an empty final_marker: the latter reads as
            // a SUCCESSFUL empty completion, so the client got a 200 with
            // no content and cascadia_tasks_failed_total never moved —
            // the failure was booked as finish_reason="stop" with zero
            // tokens. Same class as the poisoned-head bug guarded in
            // qwen36.rs. This is the default engine, so it was the common
            // failure that the failure metric could not see.
            return Chunk::error(task_id.clone(), err.to_string());
        }
        Outcome::Finished(f) => f,
    };
    if f.stop_missing {
        warn!(
            task = %task_id,
            "engine stopped on a stop the returned text lacks (stop-text invariant)"
        );
    }
    let tokens = f.completion_tokens;
    let elapsed = done.elapsed_s;
    let tok_s = if elapsed > 0.0 {
        tokens as f64 / elapsed
    } else {
        0.0
    };
    if f.stopped_on_match {
        info!(
            task = %task_id,
            tokens,
            elapsed_s = elapsed,
            "task stopped at stop string"
        );
    } else {
        info!(
            task = %task_id,
            tokens,
            elapsed_s = elapsed,
            tok_s,
            ttft_ms = f.ttft_ms,
            "task done"
        );
    }

    // Carry the real token counts on the final chunk so the API's
    // `usage` block is populated (issue #55). `prompt_tokens` is the
    // rendered-prompt token count; it is exact when the API pre-templated
    // the prompt (`skip_chat_template`) and a slight undercount when
    // ov-genai applies its own template internally (the template wrapping
    // isn't counted). None when the tokenizer is unavailable.
    // finish_reason: ov-genai's GenResult doesn't report why it stopped,
    // so infer — reaching the token cap is `length`, anything short of it
    // (EOS / stop) is `stop`. Exact for the common case; an EOS landing
    // exactly at the cap reports `length` (OpenAI is also ambiguous here).
    // A stop match ended it: `stop`, whatever the count. The text keeps
    // the match; the API's StopMatcher makes the cut.
    let finish = if !f.stopped_on_match && tokens >= f.max_new {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };
    let mut chunk = match (running, f.stream) {
        (Some(_), Some(rest)) => {
            if rest.diverged {
                // The text sent is not a prefix of the answer (a rewrite of
                // text two decodes agreed on, or a final decode that
                // disagrees). The parity check on hardware fails on this
                // line.
                warn!(
                    task = %task_id,
                    sent_bytes = rest.sent_bytes,
                    visible_bytes = f.visible.len(),
                    "ov_genai_stream_divergence: the streamed text is not a prefix of the answer"
                );
                cascadia_metrics::OV_GENAI_STREAM_DIVERGENCE_TOTAL
                    .with_label_values(&[model_id])
                    .inc();
            }
            Chunk::final_marker(task_id.clone(), rest.rest)
                .with_n_tokens(rest.carried_tokens as u32)
                .with_completion_tokens(tokens)
        }
        _ => Chunk::final_marker(task_id.clone(), f.visible).with_n_tokens(tokens),
    };
    chunk = chunk.with_finish_reason(finish);
    if let Some(prompt_tokens) = done.prompt_tokens {
        chunk = chunk.with_prompt_tokens(prompt_tokens);
    }
    chunk
}

impl Engine for OvGenaiEngine {
    fn warmup(&mut self) {
        let cfg = self.params.gen_config(
            /*max_tokens*/ 4,
            /*temperature*/ 0.0,
            /*enable_thinking*/ false,
            &SamplingParams::default(),
        );
        match lock_pipe(&self.pipe).generate("Hi", &cfg) {
            Ok(_) => info!(
                spec_k = self.params.speculative_k,
                prompt_lookup = self.params.prompt_lookup_ngram,
                "ov-genai warmup ok"
            ),
            Err(err) => warn!(error = %err, "ov-genai warmup failed"),
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        check_stateful_task(&task)?;
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
        self.sync_pending_len();
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Without streaming a running generation holds the engine mutex for
        // its whole step(), so this (&mut self) only runs between tasks and
        // drops a queued one. Streaming, it can run while the worker
        // generates. Either way a running task is ended through `inflight`
        // instead, which the runner signals without the lock (and has
        // signalled for this id too); `forget` leaves a running task alone.
        // A queued task will now never begin, so drop that early entry.
        self.pending.retain(|t| &t.task_id != task_id);
        self.sync_pending_len();
        self.inflight.forget(task_id);
    }

    fn inflight_cancel(&self) -> Option<Arc<InflightCancel>> {
        Some(self.inflight.clone())
    }

    fn submit_admission(&self) -> Option<Arc<dyn SubmitAdmission>> {
        Some(Arc::new(StatefulAdmission {
            pending: self.pending_len.clone(),
        }))
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.stream.is_some() {
            self.step_streaming()
        } else {
            self.step_inline()
        }
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
    /// `prompt_tokens` has ridden a chunk already (the first one).
    prompt_reported: bool,
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
    pipe: Box<dyn CbPipe>,
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
            prompt_reported: false,
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
        // (The stateful engine drops queued tasks here and ends its running
        // one through the in-flight handle; see OvGenaiEngine::cancel.)
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
                        let t = &mut self.active[idx];
                        let mut chunk = Chunk::token(t.task_id.clone(), 0, read.text_delta)
                            .with_n_tokens(read.new_tokens);
                        // The first emitted chunk carries the prompt count
                        // too, so a consumer that stops early has it.
                        if let Some(n) = t.prompt_tokens {
                            chunk = chunk.with_first_prompt_tokens(&mut t.prompt_reported, n);
                        }
                        out.push((t.task_id.clone(), chunk));
                    }
                    idx += 1;
                }
                CbStatus::Ignored => {
                    // OpenVINO ran out of KV cache and abandoned the request.
                    // Name the knobs that fix it either way.
                    let t = self.active.swap_remove(idx);
                    let outcome = if t.total_tokens > 0 {
                        "length"
                    } else {
                        "error"
                    };
                    cascadia_metrics::KV_CACHE_EXHAUSTED_TOTAL
                        .with_label_values(&[outcome])
                        .inc();
                    if t.total_tokens > 0 {
                        warn!(
                            task = %t.task_id,
                            tokens = t.total_tokens,
                            in_flight = self.active.len(),
                            "cb scheduler ran out of KV cache: request ended with \
                             finish_reason length after {} tokens. Raise --cb-cache-size \
                             or lower --cb-max-num-seqs.",
                            t.total_tokens
                        );
                    } else {
                        warn!(
                            task = %t.task_id,
                            in_flight = self.active.len(),
                            "cb scheduler dropped a request before it produced output: KV \
                             cache exhausted. Raise --cb-cache-size or lower --cb-max-num-seqs."
                        );
                    }
                    if t.total_tokens > 0 {
                        // It already produced output: keep it and end like an
                        // answer that hit its limit (`length`), counted as the
                        // Finished branch counts, rather than losing the text
                        // to an error.
                        let mut chunk = Chunk::final_marker(t.task_id.clone(), read.text_delta)
                            .with_n_tokens(read.new_tokens)
                            .with_finish_reason(FinishReason::Length);
                        if let Some(prompt_tokens) = t.prompt_tokens {
                            chunk = chunk.with_prompt_tokens(prompt_tokens);
                        }
                        out.push((t.task_id, chunk));
                        continue;
                    }
                    // Nothing produced: finalising it like a normal completion
                    // would hand the client HTTP 200 with an empty body, so
                    // fail it loudly.
                    let chunk = Chunk::error(
                        t.task_id.clone(),
                        "continuous-batching scheduler dropped this request before it \
                         produced output: KV cache exhausted. Raise --cb-cache-size or \
                         lower --cb-max-num-seqs."
                            .to_string(),
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
pub(crate) fn build_gen_config(
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[test]
    fn stateful_answers_end_where_prompt_plus_answer_fill_the_window() {
        // (max_new, prompt, window) -> limit
        for (max_new, prompt, window, want) in [
            // An unset limit is the whole window: cut to what the prompt left.
            (4096, Some(1000), Some(4096), 3096),
            // A smaller explicit limit stays.
            (256, Some(1000), Some(4096), 256),
            (3096, Some(1000), Some(4096), 3096),
            // Unknown prompt count or window: unchanged.
            (4096, None, Some(4096), 4096),
            (4096, Some(1000), None, 4096),
            // A prompt that alone reaches the window: unchanged.
            (4096, Some(4096), Some(4096), 4096),
            (4096, Some(5000), Some(4096), 4096),
            // One token of room left.
            (4096, Some(4095), Some(4096), 1),
        ] {
            assert_eq!(
                stateful_max_new(max_new, prompt, window),
                want,
                "max_new {max_new}, prompt {prompt:?}, window {window:?}"
            );
        }
    }

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
        steps: usize,
        added: Vec<u64>,
        cancelled: Vec<u64>,
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
            _cfg: &GenConfig,
        ) -> Result<Box<dyn CbReq>, OvError> {
            let mut st = self.st.lock().unwrap();
            st.added.push(request_id);
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
            if st.step_fails {
                Err(OvError::Native("mock step failure".into()))
            } else {
                Ok(())
            }
        }
        fn count_tokens(&self, _text: &str) -> Option<u32> {
            Some(7)
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
            prompt_pretemplated: false,
            next_request_id: 0,
            max_tokens_default: 256,
            idle_steps: 0,
            step_failures: 0,
        }
    }

    /// A task whose client streams: with `stream_writes` on, the streamed
    /// route (a client that does not stream takes the one-shot route).
    fn task(id: &str, max_tokens: u32) -> GenerationTask {
        GenerationTask::new(id, "hello")
            .with_max_tokens(max_tokens)
            .with_client_streams(true)
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

    /// A request OpenVINO abandoned for want of KV cache before it produced
    /// anything must not be finalised like a normal (empty) completion.
    #[test]
    fn cb_ignored_status_is_reported_as_an_error() {
        let exhausted = || {
            cascadia_metrics::KV_CACHE_EXHAUSTED_TOTAL
                .with_label_values(&["error"])
                .get()
        };
        let before = exhausted();
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
        assert!(
            !err.contains("after"),
            "nothing was produced, so no count: {err}"
        );
        assert!(c.is_final);
        assert!(e.active.is_empty());
        assert!(exhausted() > before, "not counted");
    }

    /// A request dropped for want of KV cache AFTER it produced output keeps
    /// that output: its last text goes out and it ends with `length`, like an
    /// answer that hit its limit, not as an error.
    #[test]
    fn cb_ignored_after_output_ends_with_length_and_keeps_the_text() {
        let exhausted = || {
            cascadia_metrics::KV_CACHE_EXHAUSTED_TOTAL
                .with_label_values(&["length"])
                .get()
        };
        let before = exhausted();
        let st = Arc::new(Mutex::new(MockState::default()));
        st.lock().unwrap().reads.push(VecDeque::from(vec![
            Ok(read_of(CbStatus::Running, 3, "abc", CbFinish::Unknown)),
            Ok(read_of(CbStatus::Ignored, 2, "de", CbFinish::Unknown)),
        ]));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 4096)).unwrap();

        let first = e.step().unwrap();
        let out = e.step().unwrap();
        assert_eq!(out.len(), 1);
        let c = &out[0].1;
        assert!(c.error.is_none(), "{:?}", c.error);
        assert!(c.is_final);
        assert_eq!(c.text, "de");
        assert_eq!(c.n_tokens, Some(2));
        assert_eq!(c.finish_reason, Some(FinishReason::Length));
        assert_eq!(c.prompt_tokens, Some(7));
        assert!(e.active.is_empty());
        let mut total = 0;
        first[0].1.accumulate_tokens(&mut total);
        c.accumulate_tokens(&mut total);
        assert_eq!(total, 5);
        assert!(exhausted() > before, "not counted");
    }

    /// Tokens produced by the very read that reports the drop count too.
    #[test]
    fn cb_ignored_on_the_first_read_with_tokens_ends_with_length() {
        let st = Arc::new(Mutex::new(MockState::default()));
        st.lock()
            .unwrap()
            .reads
            .push(VecDeque::from(vec![Ok(read_of(
                CbStatus::Ignored,
                1,
                "x",
                CbFinish::Unknown,
            ))]));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 8)).unwrap();
        let out = e.step().unwrap();
        let c = &out[0].1;
        assert!(c.error.is_none(), "{:?}", c.error);
        assert_eq!(c.text, "x");
        assert_eq!(c.finish_reason, Some(FinishReason::Length));
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

    /// The prompt count rides the first emitted chunk as well as the
    /// final one, so a consumer that stops early (the API on a `stop` match)
    /// still has it. A zero-token liveness chunk before it does not count.
    #[test]
    fn cb_first_and_final_chunks_carry_the_prompt_count() {
        let st = Arc::new(Mutex::new(MockState::default()));
        st.lock().unwrap().reads.push(VecDeque::from(vec![
            Ok(idle_read()),
            Ok(read_of(CbStatus::Running, 3, "abc", CbFinish::Unknown)),
            Ok(read_of(CbStatus::Running, 4, "defg", CbFinish::Unknown)),
            Ok(read_of(CbStatus::Finished, 2, "hi", CbFinish::Unknown)),
        ]));
        let mut e = mock_engine(&st);
        e.submit(task("t1", 4096)).unwrap();
        let mut counts = Vec::new();
        for _ in 0..4 {
            for (_, c) in e.step().unwrap() {
                if !c.is_progress() {
                    counts.push(c.prompt_tokens);
                }
            }
        }
        assert_eq!(counts, vec![Some(7), None, Some(7)]);
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

    // ---- stateful engine: polled generate ---------------------
    //
    // Driven through the StatefulPipe seam: the mock feeds the engine's poll
    // closure one scripted piece at a time, as the shim's streamer does.

    use cascadia_engine::InflightCancel;
    use cascadia_ov_genai_shim::{GenResult, PollAction, PollReply, PollWrite, POLL_PANICKED};

    #[derive(Default)]
    struct StatefulState {
        /// (token text, n_tokens) per streamer write. Each entry is one
        /// "token" for the tail decode (the shim decodes ids, not writes).
        pieces: Vec<(String, usize)>,
        /// The `tail_tokens` of every generate_polled call.
        tail_tokens: Vec<usize>,
        /// The `stream_text` of every generate_polled call.
        stream_text: Vec<bool>,
        /// Every poll call's (window, window_from), as the closure received
        /// them: with `stream_text` the decode of every token so far (the
        /// engine never confirms), else empty.
        windows: Vec<(Vec<u8>, usize)>,
        /// Every reply the closure gave.
        replies: Vec<PollReply>,
        /// Panic in the next this-many count_tokens calls (a panic inside the
        /// generation, after begin).
        count_panics: usize,
        /// Wait here before generating (the test holds the other side).
        gate: Option<Arc<std::sync::Barrier>>,
        /// Sleep this long before the first write (a long prefill).
        prefill: Option<std::time::Duration>,
        /// Before piece i is polled, cancel the running task through this
        /// handle (what the runner does on a disconnect).
        cancel_before: Option<(usize, Arc<InflightCancel>, TaskId)>,
        /// After a CANCEL answer, return Err (as a native generate may).
        err_after_cancel: bool,
        /// At piece i, fail the way the shim does after the poll panicked.
        poll_panic_at: Option<usize>,
        /// At piece i, pass this tail instead of the decode (a detokenizer
        /// whose window decode disagrees with the final decode).
        tail_override: Option<(usize, String)>,
        /// Call the poll closure from a spawned thread, as ov-genai does when
        /// it runs the streamer on its worker thread (the CB adapter).
        poll_from_thread: bool,
        /// Instead of `pieces`: one entry per streamer write, each the token
        /// texts it carries (a speculative round writes several ids at once).
        writes: Vec<Vec<String>>,
        /// With `stream_text`: the window of write i instead of the decode
        /// of the pieces so far, and the last one is the answer's decode (a
        /// detokenizer the pieces don't predict); raw bytes, so a window can
        /// end inside a character.
        window_script: Vec<Vec<u8>>,
        /// At the n-th count_tokens call (1: the prompt, 2: the answer),
        /// cancel this task through this handle: a cancel that lands after
        /// the generation is over.
        cancel_at_count: Option<(usize, Arc<InflightCancel>, TaskId)>,
        count_calls: usize,
        /// Pieces polled (the end flush not included).
        consumed: usize,
        /// Every poll call: (text, n_tokens, answer).
        polls: Vec<(String, usize, PollAction)>,
        plain_calls: usize,
        polled_calls: usize,
        last_cfg: Option<GenConfig>,
    }

    struct MockStatefulPipe {
        st: Arc<Mutex<StatefulState>>,
    }

    impl StatefulPipe for MockStatefulPipe {
        fn count_tokens(&self, text: &str) -> Option<u32> {
            {
                let mut st = self.st.lock().unwrap();
                st.count_calls += 1;
                if let Some((at, h, id)) = &st.cancel_at_count {
                    if *at == st.count_calls {
                        h.cancel(id);
                    }
                }
                if st.count_panics > 0 {
                    st.count_panics -= 1;
                    drop(st);
                    panic!("mock: count_tokens panicked");
                }
            }
            Some(text.split_whitespace().count() as u32)
        }
        fn generate(&self, _prompt: &str, cfg: &GenConfig) -> Result<GenResult, OvError> {
            let mut st = self.st.lock().unwrap();
            st.plain_calls += 1;
            st.last_cfg = Some(cfg.clone());
            Ok(GenResult {
                text: "warm".into(),
                generated_tokens: 1,
                ttft_ms: None,
            })
        }
        fn generate_polled(
            &self,
            prompt: &str,
            cfg: &GenConfig,
            tail_tokens: usize,
            stream_text: bool,
            poll: &mut PollFn<'_>,
        ) -> Result<GenResult, OvError> {
            let (gate, prefill) = {
                let mut st = self.st.lock().unwrap();
                st.stream_text.push(stream_text);
                (st.gate.clone(), st.prefill)
            };
            if let Some(gate) = gate {
                gate.wait();
            }
            if let Some(prefill) = prefill {
                std::thread::sleep(prefill);
            }
            // What the shim hands the closure, recorded as the closure got it.
            let mut poll = |tail: &str, window: &[u8], n_tokens: usize| -> PollAction {
                let w = PollWrite {
                    tail,
                    window,
                    window_from: 0,
                    n_tokens,
                };
                self.st
                    .lock()
                    .unwrap()
                    .windows
                    .push((w.window.to_vec(), w.window_from));
                let reply = poll(w);
                self.st.lock().unwrap().replies.push(reply);
                reply.action
            };
            if self.st.lock().unwrap().poll_from_thread {
                return std::thread::scope(|sc| {
                    sc.spawn(|| self.run_polled(prompt, cfg, tail_tokens, stream_text, &mut poll))
                        .join()
                        .expect("mock streamer thread")
                });
            }
            self.run_polled(prompt, cfg, tail_tokens, stream_text, &mut poll)
        }
    }

    impl MockStatefulPipe {
        fn run_polled(
            &self,
            _prompt: &str,
            cfg: &GenConfig,
            tail_tokens: usize,
            stream_text: bool,
            poll: &mut (dyn FnMut(&str, &[u8], usize) -> PollAction + Send),
        ) -> Result<GenResult, OvError> {
            let (tail_override, script) = {
                let st = self.st.lock().unwrap();
                (st.tail_override.clone(), st.window_script.clone())
            };
            let script = if stream_text { script } else { Vec::new() };
            let (pieces, tokens, ends, cancel_before, err_after_cancel, poll_panic_at) = {
                let mut st = self.st.lock().unwrap();
                // Per write: (text, n_tokens); per token: its text; per write:
                // the token count after it.
                let (pieces, tokens): (Vec<(String, usize)>, Vec<String>) = if st.writes.is_empty()
                {
                    let toks = st.pieces.iter().map(|(t, _)| t.clone()).collect();
                    (st.pieces.clone(), toks)
                } else {
                    let pieces = st.writes.iter().map(|w| (w.concat(), w.len())).collect();
                    (pieces, st.writes.concat())
                };
                let ends: Vec<usize> = if st.writes.is_empty() {
                    (1..=pieces.len()).collect()
                } else {
                    st.writes
                        .iter()
                        .scan(0, |acc, w| {
                            *acc += w.len();
                            Some(*acc)
                        })
                        .collect()
                };
                st.polled_calls += 1;
                st.tail_tokens.push(tail_tokens);
                st.last_cfg = Some(cfg.clone());
                (
                    pieces,
                    tokens,
                    ends,
                    st.cancel_before.clone(),
                    st.err_after_cancel,
                    st.poll_panic_at,
                )
            };
            let mut text = String::new();
            let mut n_total = 0usize;
            for (i, (piece, n)) in pieces.iter().enumerate() {
                if poll_panic_at == Some(i) {
                    return Err(OvError::Native(POLL_PANICKED.into()));
                }
                if let Some((at, h, id)) = &cancel_before {
                    if *at == i {
                        h.cancel(id);
                    }
                }
                text.push_str(piece);
                n_total += n;
                // What the shim passes: a decode of the last `tail_tokens`
                // tokens, or nothing.
                let tail: String =
                    if let Some((_, t)) = tail_override.as_ref().filter(|(at, _)| *at == i) {
                        t.clone()
                    } else if tail_tokens > 0 && *n > 0 {
                        let end = ends[i];
                        tokens[end.saturating_sub(tail_tokens)..end].concat()
                    } else {
                        String::new()
                    };
                // With `stream_text`: the decode of every token so far.
                let window = if let Some(w) = script.get(i) {
                    w.clone()
                } else if stream_text && *n > 0 {
                    tokens[..ends[i]].concat().into_bytes()
                } else {
                    Vec::new()
                };
                let action = poll(&tail, &window, *n);
                {
                    let mut st = self.st.lock().unwrap();
                    st.consumed += 1;
                    st.polls.push((tail, *n, action));
                }
                if action != PollAction::Running {
                    // The end flush comes after a STOP/CANCEL too.
                    let flush = poll("", &[], 0);
                    self.st
                        .lock()
                        .unwrap()
                        .polls
                        .push((String::new(), 0, flush));
                    if action == PollAction::Cancel && err_after_cancel {
                        return Err(OvError::Native("mock: generate failed after cancel".into()));
                    }
                    return Ok(GenResult {
                        text: script
                            .get(i)
                            .map_or(text, |w| String::from_utf8_lossy(w).into_owned()),
                        generated_tokens: n_total as u32,
                        ttft_ms: None,
                    });
                }
            }
            let flush = poll("", &[], 0);
            self.st
                .lock()
                .unwrap()
                .polls
                .push((String::new(), 0, flush));
            Ok(GenResult {
                text: script
                    .last()
                    .map_or(text, |w| String::from_utf8_lossy(w).into_owned()),
                generated_tokens: n_total as u32,
                ttft_ms: None,
            })
        }
    }

    fn pieces(texts: &[&str]) -> Vec<(String, usize)> {
        texts.iter().map(|t| (t.to_string(), 1)).collect()
    }

    /// Both values of [`OvGenaiBuilder::stream_writes`]: every stateful
    /// behaviour holds for the one-final-chunk path and the streaming one.
    const MODES: [bool; 2] = [false, true];

    /// The stateful engine as `build()` makes it, over the mock pipe.
    fn stateful_engine_with(
        st: &Arc<Mutex<StatefulState>>,
        configure: impl FnOnce(OvGenaiBuilder) -> OvGenaiBuilder,
    ) -> Box<dyn Engine> {
        let mut b = configure(OvGenaiBuilder::new("/x", "CPU"));
        b.pipe = Some(Box::new(MockStatefulPipe { st: st.clone() }));
        Box::new(b).build().expect("build")
    }

    fn stateful_engine(st: &Arc<Mutex<StatefulState>>) -> Box<dyn Engine> {
        stateful_engine_with(st, |b| b)
    }

    /// The stateful engine, streaming or not.
    fn stateful_engine_in(st: &Arc<Mutex<StatefulState>>, stream: bool) -> Box<dyn Engine> {
        stateful_engine_with(st, |b| b.stream_writes(stream))
    }

    /// Step until a final chunk: every chunk but progress, the final last.
    fn until_final(e: &mut Box<dyn Engine>) -> Vec<Chunk> {
        let mut out = Vec::new();
        for _ in 0..10_000 {
            for (_, c) in e.step().unwrap() {
                if c.is_progress() {
                    continue;
                }
                let fin = c.is_final;
                out.push(c);
                if fin {
                    return out;
                }
            }
        }
        panic!("no final chunk: {out:?}");
    }

    /// The answer as a client gets it: every chunk's text, joined.
    fn joined(chunks: &[Chunk]) -> String {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    /// The API's completion count over `chunks` (`accumulate_tokens`).
    fn usage(chunks: &[Chunk]) -> u32 {
        let mut total = 0;
        for c in chunks {
            c.accumulate_tokens(&mut total);
        }
        total
    }

    fn stop_task(id: &str, max_tokens: u32, stops: &[&str]) -> GenerationTask {
        task(id, max_tokens).with_sampling(SamplingParams {
            stop: stops.iter().map(|s| s.to_string()).collect(),
            ..SamplingParams::default()
        })
    }

    std::thread_local! {
        /// Where the test subscriber writes this thread's log lines, if anywhere.
        static LOG_BUF: std::cell::RefCell<Option<Arc<Mutex<Vec<u8>>>>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Writes into the current thread's [`LOG_BUF`], else nowhere.
    struct ThreadLogBuf;

    impl std::io::Write for ThreadLogBuf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            LOG_BUF.with(|cell| {
                if let Some(buf) = cell.borrow().as_ref() {
                    buf.lock().unwrap().extend_from_slice(b);
                }
            });
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Run `f` and return the INFO log lines it wrote on this thread.
    ///
    /// One global INFO subscriber for the whole test binary, writing per
    /// thread. A scoped subscriber (`with_default`) per test raced with
    /// parallel tests that hit the same log callsites with no subscriber:
    /// tracing could cache such a callsite as never enabled and drop the
    /// line the test reads.
    fn capture_logs<R>(f: impl FnOnce() -> R) -> (R, String) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let sub = tracing_subscriber::fmt()
                .with_writer(|| ThreadLogBuf)
                .with_ansi(false)
                .with_max_level(tracing::Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(sub).expect("one test subscriber");
        });
        let buf = Arc::new(Mutex::new(Vec::new()));
        LOG_BUF.with(|cell| *cell.borrow_mut() = Some(buf.clone()));
        let r = f();
        LOG_BUF.with(|cell| *cell.borrow_mut() = None);
        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        (r, out)
    }

    fn is_empty_cancelled(c: &Chunk) -> bool {
        c.is_final
            && c.error.is_none()
            && c.text.is_empty()
            && c.finish_reason == Some(FinishReason::Cancelled)
    }

    /// A builder that hands the runner an engine built beforehand.
    struct Prebuilt(Option<Box<dyn Engine>>);

    #[async_trait::async_trait]
    impl Builder for Prebuilt {
        async fn connect(&mut self, _peers: PeerLayout) -> EngineResult<()> {
            Ok(())
        }
        async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(mut self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
            Ok(self.0.take().unwrap())
        }
    }

    /// The first poll of the engine's life logs whether ov-genai runs the
    /// streamer inline (the stateful backend) or on its own worker thread
    /// (the CB adapter, speculative pipelines); later polls and steps don't.
    /// Streaming, "inline" is the engine's own worker thread.
    #[test]
    fn stateful_engine_logs_the_streamer_thread_once() {
        for stream in MODES {
            for (threaded, want) in [
                (false, "threaded_streamer=false"),
                (true, "threaded_streamer=true"),
            ] {
                let st = Arc::new(Mutex::new(StatefulState::default()));
                {
                    let mut s = st.lock().unwrap();
                    s.pieces = pieces(&["one", " two", " three"]);
                    s.poll_from_thread = threaded;
                }
                let mut e = stateful_engine_in(&st, stream);
                e.submit(task("t1", 64)).unwrap();
                e.submit(task("t2", 64)).unwrap();
                let (_, logs) = capture_logs(|| {
                    until_final(&mut e);
                    until_final(&mut e);
                });
                assert!(logs.contains(want), "{stream} {threaded}: {logs}");
                assert_eq!(
                    logs.matches("ov-genai streamer thread").count(),
                    1,
                    "{stream} {threaded}: logged once for the engine's life: {logs}"
                );
            }
        }
    }

    /// The stateful engine holds its lock for a whole generation, so the
    /// runner queues a submit that finds it busy after checking this: it must
    /// refuse exactly what `submit` refuses on the task alone, with the same
    /// error, and pass everything else.
    #[test]
    fn stateful_admission_refuses_a_resumed_task_like_submit() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            let mut e = stateful_engine_in(&st, stream);
            let admission = e
                .submit_admission()
                .expect("the stateful engine hands out an admission");
            let mut resumed = task("t1", 8);
            resumed.resume_token_ids = Some(vec![1, 2]);
            let checked = admission
                .check(&resumed)
                .expect_err("a resumed task is refused without the engine lock");
            let submitted = e.submit(resumed).expect_err("and by submit");
            assert!(
                checked.to_string().contains("resume_unsupported:"),
                "{checked}"
            );
            assert_eq!(checked.to_string(), submitted.to_string());
            admission
                .check(&task("t2", 8))
                .expect("a plain task passes the admission");
        }
    }

    /// The runner refuses a queued submit past the engine's own pending cap
    /// before any response starts, from the admission's `pending_load`: it
    /// must follow the engine's pending queue through submit, step and cancel.
    #[test]
    fn stateful_admission_tracks_the_pending_queue() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["a"]);
            let mut e = stateful_engine_in(&st, stream);
            let admission = e.submit_admission().unwrap();
            let load = |pending| {
                Some(PendingLoad {
                    pending,
                    cap: crate::dist_spec::MAX_PENDING_TASKS,
                })
            };
            assert_eq!(admission.pending_load(), load(0));
            e.submit(task("t1", 8)).unwrap();
            e.submit(task("t2", 8)).unwrap();
            e.submit(task("t3", 8)).unwrap();
            assert_eq!(admission.pending_load(), load(3));
            e.submit(task("t1", 8)).unwrap();
            assert_eq!(
                admission.pending_load(),
                load(3),
                "a duplicate takes no place"
            );
            until_final(&mut e);
            assert_eq!(admission.pending_load(), load(2), "a stepped task leaves");
            e.cancel(&TaskId::from("t3"));
            assert_eq!(admission.pending_load(), load(1), "a cancelled task leaves");
            let mut resumed = task("t4", 8);
            resumed.resume_token_ids = Some(vec![1]);
            assert!(e.submit(resumed).is_err());
            assert_eq!(
                admission.pending_load(),
                load(1),
                "a refused task takes no place"
            );
            // An admission handed out later reads the same count.
            assert_eq!(e.submit_admission().unwrap().pending_load(), load(1));
        }
    }

    #[test]
    fn stateful_engine_exposes_its_inflight_handle() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            let e = stateful_engine_in(&st, stream);
            let a = e.inflight_cancel().expect("stateful engine has a handle");
            let b = e.inflight_cancel().expect("same handle again");
            assert!(Arc::ptr_eq(&a, &b));
        }
    }

    #[test]
    fn stateful_cancel_mid_generate_ends_within_one_piece_and_is_empty_cancelled() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            let mut e = stateful_engine_in(&st, stream);
            let h = e.inflight_cancel().unwrap();
            {
                let mut s = st.lock().unwrap();
                s.pieces = pieces(&["one", " two", " three", " four", " five", " six"]);
                s.cancel_before = Some((2, h.clone(), TaskId::from("t1")));
            }
            e.submit(task("t1", 64)).unwrap();
            let (out, logs) = capture_logs(|| until_final(&mut e));
            let last = out.last().unwrap();
            assert!(is_empty_cancelled(last), "{stream}: {last:?}");
            // Streaming, one write late: the first write's text went out on
            // the second; the second's is held, and its token is on the
            // cancelled final, so the API counts both.
            let want = if stream { "one" } else { "" };
            assert_eq!(joined(&out), want, "{stream}");
            assert_eq!(usage(&out), if stream { 2 } else { 0 }, "{stream}");
            let s = st.lock().unwrap();
            // The cancel landed before piece 2's poll: that poll answers Cancel.
            assert_eq!(s.consumed, 3, "at most one more piece after the cancel");
            assert_eq!(s.polls[2].2, PollAction::Cancel);
            // Idempotent: the end flush after the Cancel also says Cancel.
            assert_eq!(
                s.polls.last().unwrap(),
                &(String::new(), 0, PollAction::Cancel)
            );
            assert!(logs.contains("task cancelled"), "{logs}");
            assert!(logs.contains("tokens=3"), "{logs}");
            // end() ran: the handle is free for the next task.
            assert!(!h.cancelled());
        }
    }

    /// An interrupted generation still books its prompt: the cancelled final
    /// carries `prompt_tokens` (counted before generate), with no text and no
    /// `n_tokens`, so no TTFT sample and 0 generated tokens when nothing went
    /// out before the cancel.
    #[tokio::test]
    async fn stateful_interrupt_books_the_prompt_tokens() {
        use futures::StreamExt;
        for (stream, model) in [
            (false, "ov-genai-interrupt-prompt-model"),
            (true, "ov-genai-interrupt-prompt-model-streaming"),
        ] {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            let e = stateful_engine_in(&st, stream);
            let h = e.inflight_cancel().unwrap();
            {
                let mut s = st.lock().unwrap();
                s.pieces = pieces(&["one", " two", " three"]);
                // What Runner::interrupt does: signal the handle, nothing else.
                // Streaming, before the first write, so nothing went out.
                let at = if stream { 0 } else { 1 };
                s.cancel_before = Some((at, h, TaskId::from("t-intr")));
            }
            let runner = Arc::new(cascadia_runner::Runner::new(Box::new(Prebuilt(Some(e)))));
            runner
                .start(
                    PeerLayout::single_stage(),
                    ShardSpec::single_stage(model, "CPU"),
                )
                .await
                .unwrap();
            let prompt = || {
                cascadia_metrics::TOKENS_PROMPT_TOTAL
                    .with_label_values(&[model])
                    .get()
            };
            let generated = || {
                cascadia_metrics::TOKENS_GENERATED_TOTAL
                    .with_label_values(&[model])
                    .get()
            };
            let ttft = || {
                cascadia_metrics::GENERATION_TTFT_SECONDS
                    .with_label_values(&[model])
                    .get_sample_count()
            };
            let (p0, g0, t0) = (prompt(), generated(), ttft());
            let mut chunks = runner.generate_async(task("t-intr", 64)).await.unwrap();
            let mut last = None;
            while let Some(c) = chunks.next().await {
                last = Some(c);
            }
            let last = last.expect("a final chunk");
            assert!(is_empty_cancelled(&last), "{stream}: {last:?}");
            assert_eq!(last.n_tokens, None);
            // The mock counts the prompt "hello" as 1 token.
            assert_eq!(prompt() - p0, 1, "{stream}: the prompt is booked");
            assert_eq!(generated() - g0, 0, "{stream}");
            assert_eq!(ttft() - t0, 0, "{stream}: no TTFT sample");
        }
    }

    #[test]
    fn stateful_cancel_before_begin_never_generates() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["a"]);
            let mut e = stateful_engine_in(&st, stream);
            let h = e.inflight_cancel().unwrap();
            e.submit(task("t1", 64)).unwrap();
            // The runner signals the handle while the task is still queued.
            h.cancel(&TaskId::from("t1"));
            let out = e.step().unwrap();
            assert_eq!(out.len(), 1, "{stream}");
            assert!(is_empty_cancelled(&out[0].1), "{:?}", out[0].1);
            let s = st.lock().unwrap();
            assert_eq!((s.polled_calls, s.plain_calls), (0, 0));
            drop(s);
            // end() ran: a next task begins clean.
            assert!(!h.begin(&TaskId::from("t2")));
            h.end();
        }
    }

    #[test]
    fn stateful_engine_cancel_forgets_a_queued_task() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["a"]);
            let mut e = stateful_engine_in(&st, stream);
            let h = e.inflight_cancel().unwrap();
            e.submit(task("t1", 64)).unwrap();
            // Runner order: signal the handle (an early entry), then Engine::cancel.
            h.cancel(&TaskId::from("t1"));
            e.cancel(&TaskId::from("t1"));
            // The early entry is gone: a resubmit of the same id runs.
            e.submit(task("t1", 64)).unwrap();
            let out = until_final(&mut e);
            assert_eq!(joined(&out), "a", "{stream}");
            assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));
        }
    }

    #[test]
    fn stateful_stop_match_stops_keeps_the_match_and_logs() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["one", " two", " tw", "enty", " more", " more"]);
            let mut e = stateful_engine_in(&st, stream);
            e.submit(stop_task("t1", 64, &["twenty"])).unwrap();
            let (out, logs) = capture_logs(|| until_final(&mut e));
            let c = out.last().unwrap();
            assert!(c.error.is_none(), "{c:?}");
            assert_eq!(joined(&out), "one two twenty", "{stream}");
            assert_eq!(c.finish_reason, Some(FinishReason::Stop));
            let s = st.lock().unwrap();
            assert_eq!(s.consumed, 4, "{stream}");
            assert_eq!(s.polls[3].2, PollAction::Stop);
            // Idempotent at the end flush.
            assert_eq!(s.polls.last().unwrap().2, PollAction::Stop);
            assert!(logs.contains("task stopped at stop string"), "{logs}");
            assert!(logs.contains("tokens=3"), "{logs}");
        }
    }

    #[test]
    fn stateful_stop_check_off_runs_to_the_end() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["one", " twenty", " more"]);
            let mut e = stateful_engine_with(&st, |b| {
                b.with_engine_stop_check(false).stream_writes(stream)
            });
            e.submit(stop_task("t1", 64, &["twenty"])).unwrap();
            let out = until_final(&mut e);
            assert_eq!(joined(&out), "one twenty more", "{stream}");
            let s = st.lock().unwrap();
            assert_eq!(s.consumed, 3);
            assert!(s.polls.iter().all(|p| p.2 == PollAction::Running));
        }
    }

    #[test]
    fn stateful_poll_panic_is_a_chunk_error() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            {
                let mut s = st.lock().unwrap();
                s.pieces = pieces(&["a", " b", " c"]);
                s.poll_panic_at = Some(1);
            }
            let mut e = stateful_engine_in(&st, stream);
            let h = e.inflight_cancel().unwrap();
            e.submit(task("t1", 64)).unwrap();
            let out = until_final(&mut e);
            let c = out.last().unwrap();
            assert!(c.is_final);
            assert!(c.error.is_some(), "a poll panic must fail the task: {c:?}");
            assert!(c.text.is_empty());
            // end() ran on the error path too.
            assert!(!h.begin(&TaskId::from("t2")));
            h.end();
        }
    }

    #[test]
    fn stateful_err_after_cancel_is_cancelled_not_an_error() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            let mut e = stateful_engine_in(&st, stream);
            let h = e.inflight_cancel().unwrap();
            {
                let mut s = st.lock().unwrap();
                s.pieces = pieces(&["a", " b", " c"]);
                s.cancel_before = Some((1, h.clone(), TaskId::from("t1")));
                s.err_after_cancel = true;
            }
            e.submit(task("t1", 64)).unwrap();
            let out = until_final(&mut e);
            assert!(is_empty_cancelled(out.last().unwrap()), "{out:?}");
            assert!(!h.cancelled());
        }
    }

    #[test]
    fn stateful_without_stop_keeps_text_finish_and_counts() {
        for stream in MODES {
            for (max_tokens, want) in [(64u32, FinishReason::Stop), (3, FinishReason::Length)] {
                let st = Arc::new(Mutex::new(StatefulState::default()));
                st.lock().unwrap().pieces = pieces(&["alpha", " beta", " gamma"]);
                let mut e = stateful_engine_in(&st, stream);
                e.submit(task("t1", max_tokens)).unwrap();
                let (out, logs) = capture_logs(|| until_final(&mut e));
                let c = out.last().unwrap();
                assert!(c.is_final && c.error.is_none(), "{c:?}");
                assert_eq!(joined(&out), "alpha beta gamma");
                assert_eq!(usage(&out), 3, "{stream}");
                if stream {
                    assert_eq!(c.completion_tokens, Some(3));
                } else {
                    assert_eq!(c.n_tokens, Some(3));
                }
                // The mock's count of the prompt "hello".
                assert_eq!(c.prompt_tokens, Some(1));
                assert_eq!(c.finish_reason, Some(want), "max_tokens {max_tokens}");
                assert!(logs.contains("task done"), "{logs}");
                let s = st.lock().unwrap();
                assert_eq!((s.polled_calls, s.plain_calls), (1, 0));
                assert_eq!(s.last_cfg.as_ref().unwrap().max_new_tokens, max_tokens);
            }
        }
    }

    /// A stop split over two tokens is found: without streaming the tail
    /// decode spans them, streaming the check runs on each write's newest
    /// visible text.
    #[test]
    fn stateful_stop_across_a_token_boundary() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["a", " sto", "p!", " more"]);
            let mut e = stateful_engine_in(&st, stream);
            e.submit(stop_task("t1", 64, &["stop"])).unwrap();
            let out = until_final(&mut e);
            assert_eq!(joined(&out), "a stop!");
            assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));
            let s = st.lock().unwrap();
            assert_eq!(s.consumed, 3);
            // The window covers the longest stop plus slack; streaming has none.
            let want = if stream { 0 } else { 4 + 16 };
            assert_eq!(s.tail_tokens, [want], "{stream}");
        }
    }

    /// With nothing to match, the shim decodes nothing per token.
    #[test]
    fn stateful_tail_tokens_is_zero_without_an_active_stop_check() {
        for stream in MODES {
            let cases: [(&str, bool, Vec<&str>); 3] = [
                ("no stops", true, vec![]),
                ("only empty stops", true, vec![""]),
                ("check off", false, vec!["x"]),
            ];
            for (name, check, stops) in cases {
                let st = Arc::new(Mutex::new(StatefulState::default()));
                st.lock().unwrap().pieces = pieces(&["a", " x", " b"]);
                let mut e = stateful_engine_with(&st, |b| {
                    b.with_engine_stop_check(check).stream_writes(stream)
                });
                e.submit(stop_task("t1", 64, &stops)).unwrap();
                let out = until_final(&mut e);
                assert_eq!(joined(&out), "a x b", "{name}");
                let s = st.lock().unwrap();
                assert_eq!(s.tail_tokens, [0], "{name}");
                assert!(s.polls.iter().all(|p| p.0.is_empty()), "{name}");
            }
        }
    }

    /// What the closure does with the window: without the streaming switch
    /// it gets none (an empty window from id 0) and the answer is today's one
    /// final chunk; with it, the chunks come from the scripted windows (a
    /// decode the pieces don't predict), one write late. The engine never
    /// confirms a window either way.
    #[test]
    fn stateful_polls_with_stream_text_only_when_streaming() {
        for stream in MODES {
            for stops in [vec![], vec!["x"]] {
                let st = Arc::new(Mutex::new(StatefulState::default()));
                {
                    let mut s = st.lock().unwrap();
                    s.pieces = pieces(&["a", " b", " c"]);
                    s.window_script = vec!["A".into(), "A B".into(), "A B C".into()];
                }
                let mut e = stateful_engine_in(&st, stream);
                e.submit(stop_task("t1", 64, &stops)).unwrap();
                let out = until_final(&mut e);
                let got: Vec<(&str, Option<u32>, bool)> = out
                    .iter()
                    .map(|c| (c.text.as_str(), c.n_tokens, c.is_final))
                    .collect();
                if stream {
                    assert_eq!(
                        got,
                        [
                            ("A", Some(1), false),
                            (" B", Some(1), false),
                            (" C", Some(1), true)
                        ],
                        "{stops:?}"
                    );
                } else {
                    assert_eq!(got, [("a b c", Some(3), true)], "{stops:?}");
                }
                let s = st.lock().unwrap();
                assert_eq!(s.stream_text, [stream], "{stops:?}");
                if !stream {
                    assert!(s.windows.iter().all(|w| w == &(Vec::new(), 0)));
                }
                assert_eq!(s.replies.len(), 4);
                assert!(
                    s.replies.iter().all(|r| r.confirm_upto.is_none()),
                    "never a confirm: {:?}",
                    s.replies
                );
            }
        }
    }

    /// One stateful generation over `toks` (one token per write) with
    /// `stops`, with the engine stop check on or off, streaming or not:
    /// (every chunk, writes consumed).
    fn run_stateful(
        prompt: &str,
        toks: &[&str],
        stops: &[&str],
        check: bool,
        stream: bool,
    ) -> (Vec<Chunk>, usize) {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().pieces = pieces(toks);
        let mut e = stateful_engine_with(&st, |b| {
            b.with_engine_stop_check(check).stream_writes(stream)
        });
        let mut t = stop_task("t1", 64, stops);
        t.prompt = prompt.into();
        e.submit(t).unwrap();
        let out = until_final(&mut e);
        let consumed = st.lock().unwrap().consumed;
        (out, consumed)
    }

    /// The text before the first `stop`: what the API's cut leaves.
    fn cut_at<'a>(text: &'a str, stop: &str) -> &'a str {
        text.split(stop).next().unwrap_or(text)
    }

    /// An empty prompt: the strip_prefix + trim_start apply, and the
    /// leading "\n"s are not part of the answer. The engine must not stop on
    /// them, so the API cut gives "Hi", the same as with the check off.
    /// Without streaming the tail check is not armed inside its window and
    /// runs to the end; streaming, it stops at the first visible "\n".
    #[test]
    fn stateful_empty_prompt_does_not_stop_in_leading_whitespace() {
        let toks = ["\n", "\n", "Hi", "\n", "There"];
        let (off, _) = run_stateful("", &toks, &["\n"], false, false);
        let off = joined(&off);
        assert_eq!(off, "Hi\nThere");
        let (on, consumed) = run_stateful("", &toks, &["\n"], true, false);
        assert_eq!(joined(&on), off);
        assert_eq!(on.last().unwrap().finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            consumed,
            toks.len(),
            "not armed inside the window: no engine stop"
        );
        let (streamed, consumed) = run_stateful("", &toks, &["\n"], true, true);
        assert_eq!(joined(&streamed), "Hi\n");
        assert_eq!(consumed, 4, "stopped at the first visible stop");
        assert_eq!(cut_at(&joined(&streamed), "\n"), cut_at(&off, "\n"));
    }

    /// The model repeats the prompt: the strip removes it, so a stop inside
    /// the echo must not end the generation. The answer is built on
    /// "There", the same as with the check off.
    #[test]
    fn stateful_stop_in_a_prompt_echo_keeps_the_visible_answer() {
        let toks = ["Hi", "\n", "There", "\n"];
        let (off, _) = run_stateful("Hi", &toks, &["\n"], false, false);
        assert_eq!(joined(&off), "There\n");
        for stream in MODES {
            let (on, consumed) = run_stateful("Hi", &toks, &["\n"], true, stream);
            assert_eq!(joined(&on), joined(&off), "{stream}");
            assert_eq!(consumed, 4, "{stream}");
        }
    }

    /// A chat-style prompt the output doesn't start with: no strip, so the
    /// check is armed from the first token.
    #[test]
    fn stateful_chat_prompt_arms_the_stop_at_once() {
        for stream in MODES {
            let (c, consumed) = run_stateful(
                "<|user|>count<|assistant|>",
                &["one", "\n", "two", "\n"],
                &["\n"],
                true,
                stream,
            );
            assert_eq!(consumed, 2);
            assert_eq!(joined(&c), "one\n");
            assert_eq!(c.last().unwrap().finish_reason, Some(FinishReason::Stop));
        }
    }

    /// An echo longer than the tail window: without streaming the visible
    /// text starts after the window decode stopped being the whole text, so
    /// the check never arms (latency only; the answer equals the check-off
    /// run). Streaming holds the echo and checks the answer after it.
    #[test]
    fn stateful_a_long_echo_never_arms() {
        // Stop "x": window 1 + 16 = 17 tokens; the echo is 20.
        let echo: Vec<String> = (0..20).map(|i| format!("w{i} ")).collect();
        let prompt = echo.concat();
        let mut toks: Vec<&str> = echo.iter().map(String::as_str).collect();
        toks.extend(["x", " y", " x", " z"]);
        let (on, consumed) = run_stateful(&prompt, &toks, &["x"], true, false);
        let (off, _) = run_stateful(&prompt, &toks, &["x"], false, false);
        assert_eq!(consumed, toks.len());
        assert_eq!(joined(&on), "x y x z");
        assert_eq!(joined(&on), joined(&off));
        let (streamed, consumed) = run_stateful(&prompt, &toks, &["x"], true, true);
        assert_eq!(joined(&streamed), "x");
        assert_eq!(consumed, 21);
        assert_eq!(cut_at(&joined(&streamed), "x"), cut_at(&joined(&off), "x"));
    }

    /// The window is the last `tail_tokens` tokens, not the whole history:
    /// stop "x" asks for 1 + 16 = 17. (Without streaming only: streaming
    /// decodes no tail.)
    #[test]
    fn stateful_tail_window_is_the_last_tail_tokens() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        let names: Vec<String> = (0..30).map(|i| format!(" p{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        st.lock().unwrap().pieces = pieces(&refs);
        let mut e = stateful_engine(&st);
        e.submit(stop_task("t1", 64, &["x"])).unwrap();
        e.step().unwrap();
        let s = st.lock().unwrap();
        assert_eq!(s.tail_tokens, [17]);
        assert_eq!(s.consumed, 30);
        let want: String = names[30 - 17..].concat();
        // The last write's tail (the end call after it is empty).
        assert_eq!(s.polls[29].0, want);
        assert_eq!(s.polls[29].1, 1);
        assert_eq!(s.polls[30], (String::new(), 0, PollAction::Running));
    }

    /// The stop-text invariant broken (a window decode that matched while the
    /// visible text doesn't hold the stop): keep the visible text, and say so
    /// in a warn the hardware check fails on. (Without streaming only:
    /// streaming checks the visible text itself, not a separate decode.)
    #[test]
    fn stateful_stop_missing_from_both_texts_keeps_visible_and_warns() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["  alpha", " beta", " gamma"]);
            s.tail_override = Some((1, "STOP".into()));
        }
        let mut e = stateful_engine(&st);
        // The default chat-style prompt: armed at once, no strip.
        e.submit(stop_task("t1", 64, &["STOP"])).unwrap();
        let (out, logs) = capture_logs(|| e.step().unwrap());
        assert_eq!(st.lock().unwrap().consumed, 2);
        assert_eq!(out[0].1.text, "  alpha beta");
        assert_eq!(out[0].1.finish_reason, Some(FinishReason::Stop));
        assert!(
            logs.contains("engine stopped on a stop the returned text lacks (stop-text invariant)"),
            "{logs}"
        );
    }

    /// A speculative round writes up to `num_assistant_tokens` + 1 ids at
    /// once, so the window grows by that much with a draft model or
    /// prompt-lookup on. Streaming decodes no tail at all.
    #[test]
    fn stateful_tail_tokens_adds_the_round_with_draft_or_prompt_lookup() {
        type Configure = fn(OvGenaiBuilder) -> OvGenaiBuilder;
        let cases: [(&str, Configure, usize); 3] = [
            ("plain", |b| b, 4 + 16),
            ("draft k=7", |b| b.with_draft("/d", "CPU", 7), 4 + 16 + 8),
            // prompt-lookup proposes max(k, 5) candidates per round.
            ("prompt-lookup", |b| b.with_prompt_lookup(3), 4 + 16 + 6),
        ];
        for stream in MODES {
            for (name, configure, want) in cases {
                let st = Arc::new(Mutex::new(StatefulState::default()));
                st.lock().unwrap().pieces = pieces(&["a", " b"]);
                let mut e = stateful_engine_with(&st, |b| configure(b).stream_writes(stream));
                e.submit(stop_task("t1", 64, &["stop"])).unwrap();
                until_final(&mut e);
                let want = if stream { 0 } else { want };
                assert_eq!(st.lock().unwrap().tail_tokens, [want], "{name} {stream}");
            }
        }
    }

    /// One write carries a whole speculative round of 23 ids, with the stop
    /// at its start: the window without the round slack (6 + 16 = 22 ids)
    /// misses it, with the slack the engine stops on that write; streaming
    /// finds it in that write's text.
    #[test]
    fn stateful_stop_at_the_start_of_a_long_round_is_found() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            {
                let mut round = vec!["twenty".to_string()];
                round.extend(std::iter::repeat_n(" a".to_string(), 22));
                let mut s = st.lock().unwrap();
                s.writes = vec![vec!["one ".into()], round, vec![" more".into()]];
            }
            let mut e =
                stateful_engine_with(&st, |b| b.with_draft("/d", "CPU", 22).stream_writes(stream));
            e.submit(stop_task("t1", 64, &["twenty"])).unwrap();
            let out = until_final(&mut e);
            let s = st.lock().unwrap();
            assert_eq!(s.polls[1].1, 23, "one write of 23 ids");
            assert_eq!(s.consumed, 2, "stopped on the long write");
            assert_eq!(s.polls[1].2, PollAction::Stop);
            assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));
            assert!(joined(&out).starts_with("one twenty"));
            assert_eq!(usage(&out), 24, "{stream}");
        }
    }

    /// A match on the last allowed token is `stop`, not `length`.
    #[test]
    fn stateful_stop_beats_length() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["a", " b", " stop", " x"]);
            let mut e = stateful_engine_in(&st, stream);
            e.submit(stop_task("t1", 3, &["stop"])).unwrap();
            let out = until_final(&mut e);
            assert_eq!(joined(&out), "a b stop");
            assert_eq!(usage(&out), 3);
            assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));
        }
    }

    /// Without streaming a panic inside step after begin still ends the
    /// handle's task. (Streaming, the worker catches it:
    /// `stateful_worker_panic_costs_one_task`.)
    #[test]
    fn stateful_panic_in_step_still_ends_the_inflight_task() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().count_panics = 1;
        let mut e = stateful_engine(&st);
        let h = e.inflight_cancel().unwrap();
        e.submit(task("t1", 64)).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| e.step()));
        assert!(r.is_err(), "count_tokens panicked inside step");
        // t1 is not current: its cancel doesn't set the flag.
        h.cancel(&TaskId::from("t1"));
        assert!(!h.cancelled());
        // A new task begins clean and becomes current.
        assert!(!h.begin(&TaskId::from("t2")));
        h.cancel(&TaskId::from("t2"));
        assert!(h.cancelled());
        h.end();
    }

    #[test]
    fn stateful_warmup_uses_plain_generate() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            let mut e = stateful_engine_in(&st, stream);
            e.warmup();
            let s = st.lock().unwrap();
            assert_eq!((s.plain_calls, s.polled_calls), (1, 0));
            assert_eq!(s.last_cfg.as_ref().unwrap().max_new_tokens, 4);
        }
    }

    #[test]
    fn stateful_window_still_caps_max_new() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            st.lock().unwrap().pieces = pieces(&["x"]);
            let mut e = stateful_engine_with(&st, |b| {
                b.with_context_window(Some(10)).stream_writes(stream)
            });
            // The mock counts 3 prompt tokens: 10 - 3 = 7 left.
            let mut t = task("t1", 4096);
            t.prompt = "a b c".into();
            e.submit(t).unwrap();
            until_final(&mut e);
            let s = st.lock().unwrap();
            assert_eq!(s.last_cfg.as_ref().unwrap().max_new_tokens, 7);
        }
    }

    #[test]
    fn stateful_draft_and_prompt_lookup_builds_use_the_polled_seam() {
        type Configure = fn(OvGenaiBuilder) -> OvGenaiBuilder;
        let cases: [(&str, Configure); 2] = [
            ("draft", |b| b.with_draft("/d", "CPU", 5)),
            ("prompt-lookup", |b| b.with_prompt_lookup(3)),
        ];
        for stream in MODES {
            for (name, configure) in cases {
                let st = Arc::new(Mutex::new(StatefulState::default()));
                st.lock().unwrap().pieces = pieces(&["a", " stop", " b"]);
                let mut e = stateful_engine_with(&st, |b| configure(b).stream_writes(stream));
                assert!(e.inflight_cancel().is_some(), "{name}");
                e.submit(stop_task("t1", 64, &["stop"])).unwrap();
                let out = until_final(&mut e);
                assert_eq!(joined(&out), "a stop", "{name}");
                let s = st.lock().unwrap();
                assert_eq!(s.polled_calls, 1, "{name}");
                let cfg = s.last_cfg.as_ref().unwrap();
                match name {
                    "draft" => assert_eq!(cfg.num_assistant_tokens, 5),
                    _ => assert_eq!(cfg.max_ngram_size, 3),
                }
            }
        }
    }

    // ---- stateful engine: streaming ---------------------------------------

    /// One write late: each write sends, as a token chunk, the text the write
    /// before it added (the text two decodes agree on), with that write's
    /// token count; the first chunk books the prompt. The final carries the
    /// last write's text and token, the answer's count, the prompt count and
    /// the finish reason.
    #[test]
    fn stateful_streams_one_chunk_per_write() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().pieces = pieces(&["one", " two", " three"]);
        let mut e = stateful_engine_in(&st, true);
        e.submit(task("t1", 64)).unwrap();
        let out = until_final(&mut e);
        assert_eq!(out.len(), 3, "{out:?}");
        let deltas: Vec<(&str, Option<u32>, bool)> = out[..2]
            .iter()
            .map(|c| (c.text.as_str(), c.n_tokens, c.is_final))
            .collect();
        assert_eq!(deltas, [("one", Some(1), false), (" two", Some(1), false)]);
        assert_eq!(
            out[0].prompt_tokens,
            Some(1),
            "the first chunk books the prompt"
        );
        assert_eq!(out[1].prompt_tokens, None);
        let fin = &out[2];
        assert!(fin.is_final && fin.error.is_none(), "{fin:?}");
        assert_eq!(fin.text, " three");
        assert_eq!(fin.n_tokens, Some(1));
        assert_eq!(fin.completion_tokens, Some(3));
        assert_eq!(fin.prompt_tokens, Some(1));
        assert_eq!(fin.finish_reason, Some(FinishReason::Stop));
        assert_eq!(usage(&out), 3);
    }

    /// The default stays one final chunk with the whole answer and its count,
    /// exactly what the non-streamed path always returned.
    #[test]
    fn stateful_stream_writes_off_keeps_one_final_chunk() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().pieces = pieces(&["one", " two", " three"]);
        let mut e = stateful_engine(&st);
        e.submit(task("t1", 64)).unwrap();
        let out = e.step().unwrap();
        assert_eq!(out.len(), 1);
        let want = Chunk::final_marker("t1", "one two three")
            .with_n_tokens(3)
            .with_finish_reason(FinishReason::Stop)
            .with_prompt_tokens(1);
        assert_eq!(
            serde_json::to_string(&out[0].1).unwrap(),
            serde_json::to_string(&want).unwrap()
        );
        assert_eq!(st.lock().unwrap().stream_text, [false]);
    }

    /// One generation at a time: the second task's first chunk comes after
    /// the first task's final.
    #[test]
    fn stateful_second_task_waits_for_the_first() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().pieces = pieces(&["a", " b"]);
        let mut e = stateful_engine_in(&st, true);
        e.submit(task("t1", 64)).unwrap();
        e.submit(task("t2", 64)).unwrap();
        let mut order = Vec::new();
        for _ in 0..10_000 {
            for (id, c) in e.step().unwrap() {
                if !c.is_progress() {
                    order.push((id.to_string(), c.text.clone(), c.is_final));
                }
            }
            if order.iter().filter(|o| o.2).count() == 2 {
                break;
            }
        }
        let ids: Vec<&str> = order.iter().map(|o| o.0.as_str()).collect();
        // Each task: its first write's text on the second write, then the final.
        assert_eq!(ids, ["t1", "t1", "t2", "t2"], "{order:?}");
        assert!(order[1].2 && order[3].2);
        assert_eq!(st.lock().unwrap().polled_calls, 2);
    }

    /// While the generation has nothing new (a held prefill), `step()` returns
    /// one progress chunk for the running task instead of blocking.
    #[test]
    fn stateful_step_returns_progress_while_the_generation_is_held() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        let gate = Arc::new(std::sync::Barrier::new(2));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["a", " b"]);
            s.gate = Some(gate.clone());
        }
        let mut e = stateful_engine_in(&st, true);
        e.submit(task("t1", 64)).unwrap();
        for _ in 0..3 {
            let out = e.step().unwrap();
            assert_eq!(out.len(), 1, "{out:?}");
            assert_eq!(out[0].0, TaskId::from("t1"));
            assert!(out[0].1.is_progress(), "{:?}", out[0].1);
        }
        assert_eq!(st.lock().unwrap().consumed, 0);
        gate.wait();
        let out = until_final(&mut e);
        assert_eq!(joined(&out), "a b");
    }

    /// Through the runner: a prefill of about 20 step waits is progress, not
    /// a stall (the runner closes a stream after a few empty steps).
    #[tokio::test]
    async fn stateful_long_prefill_survives_the_no_progress_guard() {
        use futures::StreamExt;
        const MODEL: &str = "ov-genai-long-prefill-model";
        let st = Arc::new(Mutex::new(StatefulState::default()));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["one", " two"]);
            s.prefill = Some(STEP_POLL * 20);
        }
        let e = stateful_engine_in(&st, true);
        let runner = Arc::new(cascadia_runner::Runner::new(Box::new(Prebuilt(Some(e)))));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage(MODEL, "CPU"),
            )
            .await
            .unwrap();
        let mut chunks = runner.generate_async(task("t-prefill", 64)).await.unwrap();
        let mut all = Vec::new();
        while let Some(c) = chunks.next().await {
            all.push(c);
        }
        let progress = all.iter().filter(|c| c.is_progress()).count();
        let out: Vec<Chunk> = all.into_iter().filter(|c| !c.is_progress()).collect();
        let last = out.last().expect("a final chunk");
        assert!(last.is_final && last.error.is_none(), "{last:?}");
        assert_eq!(joined(&out), "one two");
        assert!(progress >= 10, "about 20 progress chunks: {progress}");
    }

    /// The API's running sum of `n_tokens` is the tokens generated at every
    /// chunk, held writes included, so a cut at a cancel, a stop or the
    /// window counts what was generated; the final's count replaces the sum.
    #[test]
    fn stateful_usage_on_cancel_stop_and_window() {
        /// The running sum after each chunk.
        fn running(chunks: &[Chunk]) -> Vec<u32> {
            let mut total = 0;
            chunks
                .iter()
                .map(|c| {
                    c.accumulate_tokens(&mut total);
                    total
                })
                .collect()
        }

        // Stop, after an echo the stream holds: the held writes ride on the
        // first delta, which comes one write late; the stop is found at the
        // write that completes it.
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().pieces = pieces(&["Q", "\n", " one", " two", " stop", " x"]);
        let mut e = stateful_engine_in(&st, true);
        let mut t = stop_task("t1", 64, &["stop"]);
        t.prompt = "Q".into();
        e.submit(t).unwrap();
        let out = until_final(&mut e);
        assert_eq!(joined(&out), "one two stop");
        assert_eq!(running(&out), [3, 4, 3], "{out:?}");
        assert_eq!(st.lock().unwrap().consumed, 5);

        // Cancel: what went out is what was generated before it.
        let st = Arc::new(Mutex::new(StatefulState::default()));
        let mut e = stateful_engine_in(&st, true);
        let h = e.inflight_cancel().unwrap();
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["a", " b", " c", " d"]);
            s.cancel_before = Some((3, h, TaskId::from("t1")));
        }
        e.submit(task("t1", 64)).unwrap();
        let out = until_final(&mut e);
        assert!(is_empty_cancelled(out.last().unwrap()));
        // " c" is held (one write late): its token is on the cancelled final.
        assert_eq!(running(&out), [1, 2, 3]);

        // The window: the limit is cut to 7 - 3 = 4; each delta's sum is the
        // tokens of the text sent so far, so the sum reaches the window only
        // on the final and the API's window guard never cuts off the held
        // last write.
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().pieces = pieces(&["a", " b", " c", " d"]);
        let mut e =
            stateful_engine_with(&st, |b| b.with_context_window(Some(7)).stream_writes(true));
        let mut t = task("t1", 4096);
        t.prompt = "p q r".into();
        e.submit(t).unwrap();
        let out = until_final(&mut e);
        assert_eq!(
            st.lock().unwrap().last_cfg.as_ref().unwrap().max_new_tokens,
            4
        );
        assert_eq!(running(&out), [1, 2, 3, 4]);
        let fin = out.last().unwrap();
        assert_eq!(fin.finish_reason, Some(FinishReason::Length));
        assert_eq!(fin.prompt_tokens, Some(3));
    }

    /// A panic inside a streaming generation (here the prompt count) costs
    /// that task an error chunk; the engine and its handle go on, and the
    /// next task runs.
    #[test]
    fn stateful_worker_panic_costs_one_task() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["a", " b"]);
            s.count_panics = 1;
        }
        let mut e = stateful_engine_in(&st, true);
        let h = e.inflight_cancel().unwrap();
        e.submit(task("t1", 64)).unwrap();
        e.submit(task("t2", 64)).unwrap();
        let first = until_final(&mut e);
        let fail = first.last().unwrap();
        assert!(
            fail.error
                .as_deref()
                .is_some_and(|m| m.contains("panicked")),
            "{fail:?}"
        );
        // t1 is over on the handle: its cancel sets nothing.
        h.cancel(&TaskId::from("t1"));
        assert!(!h.cancelled());
        let second = until_final(&mut e);
        assert_eq!(joined(&second), "a b");
        assert!(second.last().unwrap().error.is_none());
    }

    /// The engine built directly, streaming, so a test can reach its worker.
    fn streaming_engine(st: &Arc<Mutex<StatefulState>>) -> OvGenaiEngine {
        let params = GenParams {
            speculative_k: 0,
            prompt_lookup_ngram: 0,
            prompt_pretemplated: false,
            max_tokens_default: 256,
            context_window: None,
            engine_stop_check: true,
            stream_text: true,
        };
        OvGenaiEngine::new(
            Box::new(MockStatefulPipe { st: st.clone() }),
            params,
            String::new(),
        )
        .unwrap()
    }

    /// A worker thread that is gone mid-generation fails that task with an
    /// error chunk; the next task runs on a new thread.
    #[test]
    fn stateful_worker_exit_fails_the_running_task_and_respawns() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        let gate = Arc::new(std::sync::Barrier::new(2));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["a", " b"]);
            s.gate = Some(gate.clone());
        }
        let mut e = streaming_engine(&st);
        let h = e.inflight_cancel().unwrap();
        e.submit(task("t1", 64)).unwrap();
        let out = e.step().unwrap();
        assert!(out[0].1.is_progress(), "{out:?}");
        // The engine loses its thread while t1 generates.
        let old = std::mem::replace(
            &mut e.stream.as_mut().unwrap().worker,
            StatefulWorker::gone(),
        );
        let out = e.step().unwrap();
        assert_eq!(out.len(), 1);
        let c = &out[0].1;
        assert!(c.is_final, "{c:?}");
        assert!(
            c.error
                .as_deref()
                .is_some_and(|m| m.contains("worker thread exited")),
            "{c:?}"
        );
        assert!(!h.cancelled(), "t1 is over on the handle");
        // Let the lost thread finish and go.
        st.lock().unwrap().gate = None;
        gate.wait();
        drop(old);
        e.submit(task("t2", 64)).unwrap();
        let mut boxed: Box<dyn Engine> = Box::new(e);
        let out = until_final(&mut boxed);
        assert_eq!(joined(&out), "a b");
    }

    /// A worker thread that is gone between tasks: the next task's job goes
    /// to a new thread.
    #[test]
    fn stateful_worker_gone_before_a_task_is_replaced() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().pieces = pieces(&["a", " b"]);
        let mut e = streaming_engine(&st);
        e.stream.as_mut().unwrap().worker = StatefulWorker::gone();
        e.submit(task("t1", 64)).unwrap();
        let mut boxed: Box<dyn Engine> = Box::new(e);
        let out = until_final(&mut boxed);
        assert_eq!(joined(&out), "a b");
        assert!(out.last().unwrap().error.is_none());
    }

    /// Dropping a streaming engine mid-generation ends the generation at its
    /// next write and joins the worker.
    #[test]
    fn stateful_dropping_a_streaming_engine_mid_generation_ends_it() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        {
            let mut s = st.lock().unwrap();
            let many: Vec<String> = (0..1000).map(|i| format!(" w{i}")).collect();
            s.pieces = many.into_iter().map(|w| (w, 1)).collect();
            s.prefill = Some(std::time::Duration::from_millis(200));
        }
        let mut e = stateful_engine_in(&st, true);
        e.submit(task("t1", 4096)).unwrap();
        assert!(e.step().unwrap()[0].1.is_progress());
        let t0 = std::time::Instant::now();
        drop(e);
        assert!(t0.elapsed() < std::time::Duration::from_secs(5));
        let s = st.lock().unwrap();
        assert_eq!(s.consumed, 1, "ended at its first write");
        assert_eq!(s.polls[0].2, PollAction::Cancel);
    }

    /// A stop split across two and three writes is found on the newest
    /// visible text (at the write that completes it, not one write late),
    /// and the match stays in the answer for the API's cut.
    #[test]
    fn stateful_stream_stop_split_across_writes() {
        for (toks, consumed) in [
            (vec!["say ", "<e", "nd>", " more"], 3),
            (vec!["say ", "<", "en", "d>", " more"], 4),
        ] {
            let (out, n) = run_stateful("<chat>", &toks, &["<end>"], true, true);
            assert_eq!(joined(&out), "say <end>", "{toks:?}");
            assert_eq!(n, consumed, "{toks:?}");
            assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));
        }
    }

    /// A stop inside the echoed prompt (held, never handed out) does not end
    /// the generation; the same stop in the answer does.
    #[test]
    fn stateful_stream_stop_inside_the_prompt_echo_is_not_a_stop() {
        let prompt = "Q: one;A:";
        let toks = ["Q: ", "one", ";", "A:", " two", ";", " three"];
        let (out, consumed) = run_stateful(prompt, &toks, &[";"], true, true);
        assert_eq!(joined(&out), "two;");
        assert_eq!(consumed, 6);
        let (off, _) = run_stateful(prompt, &toks, &[";"], false, false);
        assert_eq!(cut_at(&joined(&out), ";"), cut_at(&joined(&off), ";"));
    }

    /// Speculative rounds write several ids at once: each round's text goes
    /// out (one write late) with its id count, and the counts add up to the
    /// tokens generated.
    #[test]
    fn stateful_stream_speculative_multi_id_writes() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        st.lock().unwrap().writes = vec![
            vec!["a".into(), " b".into(), " c".into()],
            vec![" d".into()],
            vec![" e".into(), " f".into()],
        ];
        let mut e = stateful_engine_with(&st, |b| b.with_prompt_lookup(3).stream_writes(true));
        e.submit(task("t1", 64)).unwrap();
        let out = until_final(&mut e);
        let got: Vec<(&str, Option<u32>)> =
            out.iter().map(|c| (c.text.as_str(), c.n_tokens)).collect();
        assert_eq!(
            got,
            [("a b c", Some(3)), (" d", Some(1)), (" e f", Some(2))]
        );
        assert!(out[2].is_final);
        assert_eq!(usage(&out), 6);
        assert_eq!(joined(&out), "a b c d e f");
    }

    /// A cancel while text is held (an echo and the whitespace after it):
    /// the cancelled final carries the held writes' tokens, so the API's sum
    /// counts every token generated.
    #[test]
    fn stateful_cancel_during_held_text_counts_the_held_tokens() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        let mut e = stateful_engine_in(&st, true);
        let h = e.inflight_cancel().unwrap();
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["Q", "\n", " one", " two"]);
            s.cancel_before = Some((2, h, TaskId::from("t1")));
        }
        let mut t = task("t1", 64);
        t.prompt = "Q".into();
        e.submit(t).unwrap();
        let out = until_final(&mut e);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(is_empty_cancelled(&out[0]), "{out:?}");
        assert_eq!(out[0].n_tokens, Some(2));
        assert_eq!(usage(&out), 2);
    }

    /// A generation the worker finished stays finished when the cancel flag
    /// goes up after it (a client gone just as the answer completed): the
    /// final carries the rest, completion_tokens and the finish reason.
    #[test]
    fn stateful_finished_wins_over_a_later_cancel() {
        for stream in MODES {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            let mut e = stateful_engine_in(&st, stream);
            let h = e.inflight_cancel().unwrap();
            {
                let mut s = st.lock().unwrap();
                s.pieces = pieces(&["a", " b"]);
                // The answer's count runs after generate() returned and the
                // job checked the flag.
                s.cancel_at_count = Some((2, h.clone(), TaskId::from("t1")));
            }
            e.submit(task("t1", 64)).unwrap();
            let out = until_final(&mut e);
            // The cancel went in after the generation was over.
            assert_eq!(st.lock().unwrap().count_calls, 2, "{stream}");
            let fin = out.last().unwrap();
            assert!(fin.error.is_none(), "{stream}: {fin:?}");
            assert_eq!(fin.finish_reason, Some(FinishReason::Stop), "{stream}");
            assert_eq!(joined(&out), "a b", "{stream}");
            assert_eq!(usage(&out), 2, "{stream}");
            if stream {
                assert_eq!(fin.text, " b");
                assert_eq!(fin.completion_tokens, Some(2));
            }
        }
    }

    /// One streamed generation over scripted window decodes (one id per
    /// write) with `stops`: (every chunk, writes consumed).
    fn run_stream_script(windows: &[&str], stops: &[&str]) -> (Vec<Chunk>, usize) {
        let raw: Vec<&[u8]> = windows.iter().map(|w| w.as_bytes()).collect();
        run_stream_script_raw(&raw, stops)
    }

    /// [`run_stream_script`] over raw window bytes (a window can end inside a
    /// character, or hold a byte no character takes).
    fn run_stream_script_raw(windows: &[&[u8]], stops: &[&str]) -> (Vec<Chunk>, usize) {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&vec!["x"; windows.len()]);
            s.window_script = windows.iter().map(|w| w.to_vec()).collect();
        }
        let mut e = stateful_engine_in(&st, true);
        e.submit(stop_task("t1", 64, stops)).unwrap();
        let out = until_final(&mut e);
        let consumed = st.lock().unwrap().consumed;
        (out, consumed)
    }

    /// A cleanup rule rewrites `it '` to `it's`: the stop `' ` is in no
    /// later decode, so the generation must not stop on it (a running copy of
    /// the pieces would still hold `it ' a`).
    #[test]
    fn stateful_stream_a_rewrite_does_not_fake_a_stop() {
        let (out, consumed) = run_stream_script(&["it '", "it's", "it's a", "it's a b"], &["' "]);
        assert_eq!(consumed, 4, "{out:?}");
        assert_eq!(joined(&out), "it's a b");
    }

    /// A rewrite that completes a stop (`it '` to `it's\n`, stop `\n`) stops
    /// at that write, as the non-streamed tail decode would.
    #[test]
    fn stateful_stream_a_rewrite_does_not_hide_a_stop() {
        let (out, consumed) = run_stream_script(&["it '", "it's\n", "it's\nmore"], &["\n"]);
        assert_eq!(consumed, 2, "{out:?}");
        assert_eq!(joined(&out), "it's\n");
        assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));
    }

    /// A detokenizer that rewrites text two decodes agreed on (`it 's` to
    /// `it's`): a client that does not stream gets the final decode byte for
    /// byte, in one chunk, with no per-write decode, although `stream_writes`
    /// is on; a streaming client gets what the stream rules send (the agreed
    /// text, then the answer past the re-anchored text) and the divergence is
    /// logged.
    #[test]
    fn stateful_a_non_streamed_client_gets_the_exact_answer_past_a_rewrite() {
        let windows = ["it '", "it 's", "it's a"];
        let answer = "it's a";
        let run = |client_streams: bool| {
            let st = Arc::new(Mutex::new(StatefulState::default()));
            {
                let mut s = st.lock().unwrap();
                // The pieces' concat is the final decode, as the shim returns it.
                s.pieces = pieces(&["it", "'s", " a"]);
                s.window_script = windows.iter().map(|w| w.as_bytes().to_vec()).collect();
            }
            let mut e = stateful_engine_with(&st, |mut b| {
                b.model_id = "divergence-model".into();
                b.stream_writes(true)
            });
            e.submit(task("t1", 64).with_client_streams(client_streams))
                .unwrap();
            let (out, logs) = capture_logs(|| until_final(&mut e));
            let stream_text = st.lock().unwrap().stream_text.clone();
            (out, logs, stream_text)
        };

        let diverged = || {
            cascadia_metrics::OV_GENAI_STREAM_DIVERGENCE_TOTAL
                .with_label_values(&["divergence-model"])
                .get()
        };
        let (plain, logs, stream_text) = run(false);
        assert_eq!(plain.len(), 1, "{plain:?}");
        assert!(plain[0].is_final && plain[0].error.is_none(), "{plain:?}");
        assert_eq!(joined(&plain).as_bytes(), answer.as_bytes());
        // The mock counts the answer's words.
        assert_eq!(plain[0].n_tokens, Some(2));
        assert_eq!(stream_text, [false], "no per-write decode");
        assert!(!logs.contains("ov_genai_stream_divergence"), "{logs}");
        assert_eq!(diverged(), 0, "a non-streamed answer never diverges");

        let (streamed, logs, stream_text) = run(true);
        assert_eq!(stream_text, [true]);
        // The agreed `it '`, then the answer past the re-anchored text.
        assert_eq!(joined(&streamed), "it ''s a", "{streamed:?}");
        assert_eq!(usage(&streamed), 2, "{streamed:?}");
        assert!(logs.contains("ov_genai_stream_divergence"), "{logs}");
        assert_eq!(diverged(), 1, "the divergence is counted under its model");
    }

    /// A U+FFFD that stays (a byte no character takes) refuses the check of
    /// the write it ends; the next write's tail still reaches back over the
    /// text before it, so a stop there is found at that write, not missed.
    #[test]
    fn stateful_stream_a_stop_before_a_lasting_replacement_is_found() {
        let windows: [&[u8]; 4] = [b"a", b"a\n\x80", b"a\n\x80b", b"a\n\x80bc"];
        let (out, consumed) = run_stream_script_raw(&windows, &["\n"]);
        assert_eq!(consumed, 3, "{out:?}");
        assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));
    }

    /// A stop-completing id that also starts a multibyte character (`是的`
    /// plus `\xE4`, stop `的`) stops one id later, once the character is
    /// whole: the same write the non-streamed tail check (`是的\u{FFFD}`,
    /// refused) stops at.
    #[test]
    fn stateful_stream_stops_on_the_same_id_as_without_streaming() {
        let windows: [&[u8]; 3] = [
            b"\xE6\x98\xAF\xE7\x9A\x84\xE4",
            "是的中".as_bytes(),
            "是的中x".as_bytes(),
        ];
        let (out, streamed) = run_stream_script_raw(&windows, &["的"]);
        assert_eq!(out.last().unwrap().finish_reason, Some(FinishReason::Stop));

        // Without streaming: the shim's lossy tail of the same ids.
        let st = Arc::new(Mutex::new(StatefulState::default()));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["是的", "中", "x"]);
            s.tail_override = Some((0, "是的\u{FFFD}".into()));
        }
        let mut e = stateful_engine_in(&st, false);
        e.submit(stop_task("t1", 64, &["的"])).unwrap();
        let plain = until_final(&mut e);
        assert_eq!(
            plain.last().unwrap().finish_reason,
            Some(FinishReason::Stop)
        );
        let plain_consumed = st.lock().unwrap().consumed;
        assert_eq!(plain_consumed, 2, "{plain:?}");
        assert_eq!(streamed, plain_consumed, "{out:?}");
    }

    /// A generation that fails while the cancel flag goes up ends as
    /// cancelled, and still books the tokens of the writes whose text was
    /// held (an echo and the whitespace after it), as a plain cancel does.
    #[test]
    fn stateful_failed_with_the_cancel_flag_up_books_the_held_tokens() {
        let st = Arc::new(Mutex::new(StatefulState::default()));
        {
            let mut s = st.lock().unwrap();
            s.pieces = pieces(&["Q", "\n", " one", " two"]);
            s.poll_panic_at = Some(3);
        }
        let pipe = MockStatefulPipe { st: st.clone() };
        let params = GenParams {
            speculative_k: 0,
            prompt_lookup_ngram: 0,
            prompt_pretemplated: false,
            max_tokens_default: 256,
            context_window: None,
            engine_stop_check: true,
            stream_text: true,
        };
        let inflight = InflightCancel::new();
        let id = TaskId::from("t1");
        assert!(!inflight.begin(&id));
        let mut t = task("t1", 64);
        t.prompt = "Q".into();
        let mut deltas = 0;
        let done = run_job(&pipe, &params, &inflight, &t, false, &mut |ev| {
            if matches!(ev, Event::Delta { .. }) {
                deltas += 1;
            }
        });
        assert_eq!(deltas, 0, "every write's text was still held");
        // The flag goes up after the job looked at it.
        inflight.cancel(&id);
        let chunk = final_chunk(&inflight, &mut false, "m", &id, done, None);
        assert!(is_empty_cancelled(&chunk), "{chunk:?}");
        assert_eq!(chunk.n_tokens, Some(3));
    }
}
