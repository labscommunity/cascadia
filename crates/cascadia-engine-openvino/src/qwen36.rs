//! Qwen3.6-35B-A3B staged engine (M3' v1): runs the IR-surgery shard
//! chain (tools/qwen36_surgery/export_qwen36_moe.py) in-process on one
//! box. Greedy-only, batch=1, per the spec's v1 invariants; the decode
//! loop is a port of `tools/qwen36_surgery/proto_m3_decode.py`, which
//! measured 64/64 greedy token parity vs the whole model.
//!
//! Stage dirs are stateful OV IRs (DeltaNet conv/ssm + attention KV as
//! ReadValue/Assign). Linear state cannot be trimmed: every task starts
//! with `reset_state()` on every stage, and cancellation mid-decode also
//! resets (position-0 re-entry is the only recovery — spec §4.1).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;
use tokenizers::Tokenizer;
use tracing::{info, warn};

const HIDDEN: usize = 2048;
const MROPE_ROWS: usize = 4;

#[derive(serde::Deserialize)]
struct Manifest {
    arch: String,
    stages: Vec<StageInfo>,
}

#[derive(serde::Deserialize)]
struct StageInfo {
    stage: usize,
    layer_start: u32,
    layer_end: u32,
}

pub struct Qwen36Builder {
    pub shards_dir: String,
    pub device: String,
    pub max_tokens_default: u32,
    emb: Option<Runtime>,
    stages: Option<Vec<Runtime>>,
    tokenizer: Option<Tokenizer>,
    eos: Option<u32>,
}

impl Qwen36Builder {
    pub fn new(shards_dir: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            shards_dir: shards_dir.into(),
            device: device.into(),
            max_tokens_default: 256,
            emb: None,
            stages: None,
            tokenizer: None,
            eos: None,
        }
    }
}

fn read_eos(dir: &Path) -> Option<u32> {
    let gc = std::fs::read_to_string(dir.join("generation_config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&gc).ok()?;
    match &v["eos_token_id"] {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as u32),
        serde_json::Value::Array(a) => a.first()?.as_u64().map(|x| x as u32),
        _ => None,
    }
}

#[async_trait]
impl Builder for Qwen36Builder {
    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        if peers.upstream.is_some() || peers.downstream.is_some() {
            return Err(EngineError::PeerRejected(
                "qwen36-moe v1 is single-box (all stages in-process); \
                 do not configure peers"
                    .into(),
            ));
        }
        Ok(())
    }

    async fn load(&mut self, shard: ShardSpec) -> EngineResult<LoadStream> {
        if !(shard.is_first_stage && shard.is_last_stage) {
            return Err(EngineError::ShardRejected(
                "qwen36-moe v1 requires --total 1 (in-process stage chain)".into(),
            ));
        }
        let dir = PathBuf::from(&self.shards_dir);
        let manifest_path = dir.join("manifest.json");
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path)
                .map_err(|e| EngineError::ModelNotFound(format!("{}: {e}", manifest_path.display())))?,
        )
        .map_err(|e| EngineError::InvalidConfig(format!("manifest.json: {e}")))?;
        if manifest.arch != "qwen3_5_moe" {
            return Err(EngineError::InvalidConfig(format!(
                "manifest arch {:?} is not qwen3_5_moe",
                manifest.arch
            )));
        }

        let mut progress = vec![LoadProgress::message(format!(
            "qwen36-moe: {} stages from {}",
            manifest.stages.len(),
            dir.display()
        ))];
        let plugin = PluginConfig::new();

        let emb_xml = dir.join("openvino_text_embeddings_model.xml");
        progress.push(LoadProgress::message("compiling text-embeddings IR".to_string()));
        let emb = Runtime::compile(
            emb_xml.to_str().unwrap_or_default(),
            &self.device,
            &plugin,
        )
        .map_err(map_ov)?;

        let mut stages = Vec::with_capacity(manifest.stages.len());
        for s in &manifest.stages {
            let xml = dir.join(format!("stage{}", s.stage)).join("stage.xml");
            progress.push(LoadProgress::message(format!(
                "compiling stage{} (layers {}..{}) on {}",
                s.stage, s.layer_start, s.layer_end, self.device
            )));
            stages.push(
                Runtime::compile(xml.to_str().unwrap_or_default(), &self.device, &plugin)
                    .map_err(map_ov)?,
            );
        }

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| EngineError::InvalidConfig(format!("tokenizer.json: {e}")))?;

        self.eos = read_eos(&dir);
        self.emb = Some(emb);
        self.stages = Some(stages);
        self.tokenizer = Some(tokenizer);
        progress.push(LoadProgress::ready());
        Ok(Box::pin(stream::iter(progress)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        Ok(Box::new(Qwen36Engine {
            emb: self.emb.ok_or(EngineError::NotLoaded)?,
            stages: self.stages.ok_or(EngineError::NotLoaded)?,
            tokenizer: self.tokenizer.ok_or(EngineError::NotLoaded)?,
            eos: self.eos,
            max_tokens_default: self.max_tokens_default,
            pending: Vec::new(),
            cancelled: HashSet::new(),
        }))
    }
}

fn map_ov(err: cascadia_ov_genai_shim::Error) -> EngineError {
    match err {
        cascadia_ov_genai_shim::Error::Stub => EngineError::Backend(
            "qwen36-moe requires the `openvino` feature (stub build)".into(),
        ),
        cascadia_ov_genai_shim::Error::Utf8(s) => EngineError::InvalidConfig(s),
        cascadia_ov_genai_shim::Error::Native(s) => EngineError::Backend(s),
    }
}

pub struct Qwen36Engine {
    emb: Runtime,
    stages: Vec<Runtime>,
    tokenizer: Tokenizer,
    eos: Option<u32>,
    max_tokens_default: u32,
    pending: Vec<GenerationTask>,
    cancelled: HashSet<TaskId>,
}

fn le_bytes_i64(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_bytes_i32(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_bytes_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn f32_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Qwen36Engine {
    fn embed(&mut self, tok: u32) -> EngineResult<Vec<f32>> {
        let name = self
            .emb
            .input_names()
            .map_err(map_ov)?
            .into_iter()
            .next()
            .ok_or_else(|| EngineError::Backend("embeddings IR has no inputs".into()))?;
        self.emb
            .set_input(&name, DType::I64, &[1, 1], &le_bytes_i64(&[tok as i64]))
            .map_err(map_ov)?;
        self.emb.infer().map_err(map_ov)?;
        let (dtype, _shape, bytes) = self.emb.output(0).map_err(map_ov)?;
        if !matches!(dtype, DType::F32) {
            return Err(EngineError::Backend(format!(
                "embeddings output dtype {dtype:?}, expected f32"
            )));
        }
        Ok(f32_from_le(&bytes))
    }

    /// One T=1 pass through the stage chain. `step` is the absolute
    /// position; returns the last stage's first output (hidden or logits).
    fn chain_step(&mut self, embeds: &[f32], step: usize) -> EngineResult<Vec<f32>> {
        let mask = vec![1i64; step + 1];
        let pos = vec![step as i64; MROPE_ROWS];
        let zeros_embeds = vec![0f32; HIDDEN];
        let mut hidden: Vec<f32> = embeds.to_vec();
        let n = self.stages.len();
        for (j, st) in self.stages.iter_mut().enumerate() {
            let names = st.input_names().map_err(map_ov)?;
            for name in &names {
                match name.as_str() {
                    "stage_hidden" => st
                        .set_input(name, DType::F32, &[1, 1, HIDDEN], &le_bytes_f32(&hidden))
                        .map_err(map_ov)?,
                    s if s.contains("embed") => {
                        // first stage: real embeds; mid stages: dummy
                        // (upstream ShapeOf chains read shapes only)
                        let data = if j == 0 { &hidden } else { &zeros_embeds };
                        st.set_input(name, DType::F32, &[1, 1, HIDDEN], &le_bytes_f32(data))
                            .map_err(map_ov)?;
                    }
                    s if s.contains("attention_mask") => st
                        .set_input(name, DType::I64, &[1, step + 1], &le_bytes_i64(&mask))
                        .map_err(map_ov)?,
                    s if s.contains("position") => st
                        .set_input(name, DType::I64, &[MROPE_ROWS, 1, 1], &le_bytes_i64(&pos))
                        .map_err(map_ov)?,
                    s if s.contains("beam") => st
                        .set_input(name, DType::I32, &[1], &le_bytes_i32(&[0]))
                        .map_err(map_ov)?,
                    other => {
                        return Err(EngineError::Backend(format!(
                            "stage{j}: unexpected input {other:?}"
                        )))
                    }
                }
            }
            st.infer().map_err(map_ov)?;
            let (dtype, _shape, bytes) = st.output(0).map_err(map_ov)?;
            if !matches!(dtype, DType::F32) {
                return Err(EngineError::Backend(format!(
                    "stage{j} output dtype {dtype:?}, expected f32"
                )));
            }
            hidden = f32_from_le(&bytes);
            let _ = n;
        }
        Ok(hidden)
    }

    fn reset_all(&mut self) {
        for st in self.stages.iter_mut() {
            if let Err(e) = st.reset_state() {
                warn!(error = %e, "qwen36: stage reset_state failed");
            }
        }
    }
}

impl Engine for Qwen36Engine {
    fn warmup(&mut self) {
        self.reset_all();
        match self.embed(1000).and_then(|e| self.chain_step(&e, 0)) {
            Ok(_) => info!("qwen36-moe warmup ok"),
            Err(e) => warn!(error = %e, "qwen36-moe warmup failed"),
        }
        self.reset_all();
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|t| t.task_id == task.task_id) {
            return Ok(());
        }
        self.pending.push(task);
        Ok(())
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let task = self.pending.remove(0);
        let max_tokens = if task.max_tokens > 0 {
            task.max_tokens
        } else {
            self.max_tokens_default
        } as usize;

        // Linear state cannot be trimmed: position-0 entry per task.
        self.reset_all();
        self.cancelled.remove(&task.task_id);

        let enc = match self.tokenizer.encode(task.prompt.as_str(), true) {
            Ok(e) => e,
            Err(e) => {
                warn!(task = %task.task_id, error = %e, "tokenize failed");
                return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
            }
        };
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

        let started = Instant::now();
        let mut step = 0usize;
        let mut logits: Vec<f32> = Vec::new();
        let mut gen_ids: Vec<u32> = Vec::new();
        let mut run = |engine: &mut Self, tok: u32, step: usize| -> EngineResult<Vec<f32>> {
            let e = engine.embed(tok)?;
            engine.chain_step(&e, step)
        };

        // prefill (T=1 steps), then greedy decode
        let mut failed = false;
        for &tok in &prompt_ids {
            match run(self, tok, step) {
                Ok(l) => logits = l,
                Err(e) => {
                    warn!(task = %task.task_id, error = %e, "prefill failed");
                    failed = true;
                    break;
                }
            }
            step += 1;
        }
        if !failed {
            for _ in 0..max_tokens {
                if self.cancelled.remove(&task.task_id) {
                    info!(task = %task.task_id, "qwen36: cancelled mid-decode; resetting state");
                    break;
                }
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                if Some(next) == self.eos {
                    break;
                }
                gen_ids.push(next);
                match run(self, next, step) {
                    Ok(l) => logits = l,
                    Err(e) => {
                        warn!(task = %task.task_id, error = %e, "decode failed");
                        break;
                    }
                }
                step += 1;
            }
        }
        // Any exit path leaves state dirty for the NEXT task; reset now so
        // the position-0 invariant holds unconditionally (spec §4.1.3).
        self.reset_all();

        let text = self.tokenizer.decode(&gen_ids, true).unwrap_or_default();
        let elapsed = started.elapsed().as_secs_f64();
        let tok_s = if elapsed > 0.0 {
            gen_ids.len() as f64 / elapsed
        } else {
            0.0
        };
        info!(
            task = %task.task_id,
            prompt_tokens = prompt_ids.len(),
            tokens = gen_ids.len(),
            elapsed_s = elapsed,
            tok_s,
            "qwen36 task done"
        );
        vec![(
            task.task_id.clone(),
            Chunk::final_marker(task.task_id, text),
        )]
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Pending (not started): drop. In-flight: flag checked per token.
        self.pending.retain(|t| &t.task_id != task_id);
        self.cancelled.insert(task_id.clone());
    }
}
