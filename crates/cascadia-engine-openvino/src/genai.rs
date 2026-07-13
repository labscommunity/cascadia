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
use cascadia_ov_genai_shim::{Error as OvError, GenConfig, LlmPipeline, PluginConfig};
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
    pipe: Option<LlmPipeline>,
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
            pipe: None,
        }
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
    // A `cascadia shard` tree is a common mistake here: it has no tokenizer IRs
    // because the staged engines tokenize in Rust. Say so rather than telling
    // the user to re-export a model they already exported.
    if d.join("pipeline_config.json").exists() {
        return Err(EngineError::InvalidConfig(format!(
            "{dir} is a `cascadia shard` tree, which ov-genai cannot serve. \
             Use --engine ov-runtime."
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

impl OvGenaiEngine {
    fn gen_config(
        &self,
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
            skip_chat_template: self.prompt_pretemplated && !enable_thinking,
            // OpenAI sampling knobs (#14). top_p/top_k take effect in the
            // sampling regime (do_sample, i.e. temperature > 0); penalties and
            // seed are forwarded regardless.
            //
            // SEED CAVEAT (validated on-HW, OpenVINO GenAI 2026.1): `seed` is
            // forwarded to GenerationConfig.rng_seed, but it does NOT make
            // ov-genai reproducible across requests on this reused
            // `LLMPipeline`. OV-GenAI's StatefulLLMPipeline only re-seeds when
            // the seed VALUE CHANGES between consecutive generate() calls
            // (`if (m_sampler.get_seed() != config.rng_seed) set_seed(...)`)
            // and never calls `clear_request_info`, so two identical seeds in a
            // row reuse the advanced RNG and diverge. Different seeds DO change
            // the output, and the Rust-sampler engines (ov-runtime/sparse-moe)
            // honor seed fully (they re-seed per request). Per-request
            // reproducible ov-genai needs the ContinuousBatchingPipeline (fresh
            // request context per request) — tracked with #20.
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
        if self.speculative_k > 0 {
            cfg.num_assistant_tokens = self.speculative_k;
        }
        if self.prompt_lookup_ngram > 0 {
            cfg.num_assistant_tokens = self.speculative_k.max(5);
            cfg.max_ngram_size = self.prompt_lookup_ngram;
        }
        cfg
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
    async fn build_before_load_fails() {
        let b = Box::new(OvGenaiBuilder::new("/x", "CPU"));
        let res = b.build();
        assert!(matches!(res, Err(EngineError::NotLoaded)));
    }
}
