# /v1/decisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve typed decisions (yes/no, choice and score questions over a piece of evidence) for all five open decision models Cascadia can run: JEV-9B, JEV-27B, Clef-flash (9B), Clef (27B) and Kev-9B. Single-box and multi-machine pipeline, int8 weights. Two wire formats over one engine path: **OpenAI's Decisions API at `POST /v1/decisions`** (so OpenAI SDK clients work unchanged), and the **Jev/SystemOne format at `POST /v1/systemone`** (JEV's and Kev's own servers' path).

**Architecture:** The API owns each model family's prompt format (a *codec*). It tokenizes the request into one or more prefill-only *passes* and submits each as a `GenerationTask` carrying a `DecisionReadout`. The `qwen35` engine prefills the exact token ids and generates nothing. The machine holding the final hidden states (single box, or the last pipeline rank) runs the model's *decision head* and returns raw per-option logits on the final chunk. The API turns those logits into calibrated probabilities and answers in the format of the route that was called: OpenAI-format (`/v1/decisions`) or Jev/SystemOne-format (`/v1/systemone`). Both formats translate into one internal question model (`schema::DecisionRequest`) before the codec runs, so the models see identical prompts either way. The heads are:
- **Verbalizer** (JEV): lm_head logits for 24 fixed tokens at the last position.
- **Pointer** (Kev): `k(h_opt)·q(h_decide)/√256` over final hidden states at chosen positions.
- **Joint** (Clef): Cloudflare's joint schema head, as an OpenVINO IR, over all-position final hidden states.

Model trees are built by an extended exporter: Intel's stock int8 IR, plus an optional weight swap from the model's own checkpoint, plus an optional LoRA graft, plus a `final_hidden` output on the last stage, plus `decision.json` and `decision/` head assets.

**Tech Stack:** Rust (axum, serde, tokenizers 0.21, cascadia-ov-genai-shim), Python 3.12 (openvino 2026.4, numpy, safetensors, torch for export only), pytest.

**Evidence base (read before starting).** `experiments/` is gitignored, so it is **not** in the worktree. Everything below lives in the main checkout at `/Users/blokchainaholic/Desktop/clabs/tahoma/experiments` (`$SPIKE_DIR`). Read and port from there; tools that need spike files at run time use the `SPIKE_DIR` env var with that default.
- `experiments/QUANT_RESULTS.md`: int4 vs int8 for all three 9B models (int8 = 98.8–100%).
- `experiments/jev-spike/RESULTS.md`, `experiments/clef-spike/RESULTS.md`: the mechanisms.
- Spike code being ported:
  - `experiments/jev-spike/swapq.py` → weight swap
  - `experiments/jev-spike/vl/serve_decide.py` (`s1_pass`) → JEV codec (`jev.py` is the spike's measurement script; it sorted choices, the server does not)
  - `experiments/kev-spike/kev.py` → Kev codec
  - `experiments/clef-spike/flash/joint_schema_model.py` + `step2_head_export.py` → Clef codec and head export
  - `experiments/jev-spike/ref_stream.py` + `experiments/kev-spike/kev_ref.py` → streamed fp32 references

## Global Constraints

- **Weights are int8**, built from Intel's stock `OpenVINO/Qwen3.5-9B-int8-ov` (9B models) or `OpenVINO/Qwen3.8-27B-int8-ov` (27B models). int4 is not supported for decision trees: it measured 87.8–96.3% agreement versus int8's 98.8–100%.
- **Engine:** batch=1. A decision task never decodes; the engine emits exactly one final chunk carrying `decision_logits`.
- **Wire formats** (both text-only; images are a 400):
  - `POST /v1/decisions` follows **OpenAI's Decisions API** (developers.openai.com/api/docs/guides/decisions).
    - Request: `{model, input: string | [{role: "user", content: string | [{type: "input_text", text} | {type: "input_image", image_url}]}], questions: [{type: "predicate"|"choice"|"score", name, instructions, choices?: [{value, description?}], levels?: [{label, description?}]}]}`.
    - Response: `{answers: [{type: "predicate", name, probability} | {type: "choice", name, choice, probabilities: [{value, probability}], confidence} | {type: "score", name, score, probabilities: [{value: index, label, probability}], confidence}]}`.
    - Errors: `{"error": {"message", "type", "code"}}`.
    - The `refusal` answer type is never emitted, because these models do not refuse.
  - `POST /v1/systemone` follows the **Jev/SystemOne format**.
    - Request: `{model, state, questions:{id:{type, instructions, criteria}}, images?}`.
    - Response: `{model, answers:{id:{...}}, usage:{input_tokens, output_tokens:0}}`.
- **Confidence** uses TypeSafe's system-one-adapter 0.2.1 formulas on both routes. OpenAI's own documented examples match them exactly: choice [0.95, 0.02, 0.01, 0.02] → 0.93 and score [0.1, 0.7, 0.2] → 0.55.
- **Probabilities:** softmax in f64; reported values are rounded to 4 decimals.
- **Chat paths stay byte-identical:** no workspace-wide Cargo feature changes (in particular serde_json `preserve_order` stays off), no change to existing request/response types beyond additive `#[serde(default)]` fields, and every engine change is gated on `GenerationTask.decision.is_some()` or the manifest's `final_hidden_output`.
- **Prompt fidelity:** each codec reproduces its reference implementation's text exactly: key order, separators, Python `str()`/`json.dumps` formatting. The request's JSON order is preserved with `schema::OrderedValue`, never `serde_json::Value` (whose map sorts keys).
- **Exporter memory:** every export step processes one stage at a time and stays under 7 GB physical footprint. 9B trees are cut into 8 stages for that reason. The one exception is the Clef-27B export (2 pipeline stages of ~13 GB swapped constants), which needs a ~16 GB machine (Task 22).
- **Acceptance per model:** two gates. (1) **Golden:** the server's prompts, positions and spans equal the reference encoders' exactly, and its answers agree with the spike's saved full-precision references (Task 16). (2) **Parity** against a streamed fp32 reference on the 50-record set (`tools/decision/records.py`, 82 questions), measured through the running server and reported per question type. Both of these must hold:
  - raw top-1 agreement ≥ 96% (at most 3 flips of 82);
  - **no flip where the reference's own top-1 margin is ≥ 0.05**.

  Rationale: in the spike, int8 moved probabilities by at most 0.08, and the only miss (JEV) was a 0.501/0.499 tie. Flips are tolerated only where the full-precision model itself was undecided. JEV has 5 such questions under a 0.02 margin, so excluding near-ties outright would let too much through.
- **Long jobs:** exports, references and server bring-up run in the background with a Monitor that watches for completion and for failure signatures (Traceback, Error, killed, CAP, MISMATCH, FLIP, non-zero exit).
- **Runbook environment** (Task 15 Step 6 and Tasks 18–22). Every command starts from this preamble:

  ```bash
  export SPIKE_DIR=/Users/blokchainaholic/Desktop/clabs/tahoma/experiments
  PY=$SPIKE_DIR/clef-spike/.venv/bin/python
  ```

  `$PY` runs every tool and is the command `run_capped.py` launches. That venv (transformers 5.18, torch 2.14, openvino 2026.4.1, safetensors 0.8) is where the decision tools were developed and tested; `tools/requirements.txt`'s transformers range is unverified for `codec_golden.py`'s tokenizer-only directories. `cascadia` is this branch's release build on `PATH` (`cargo build --release -p cascadia --features openvino`), against the OpenVINO SDK of the venv's version (2026.4.1, the CI `OV_VERSION` pin). Manual `curl` calls to the decision routes must send `-H 'content-type: application/json'` (axum answers 415 otherwise); the tools set it themselves.
- **Secrets:** never read `.env*`, keys or tokens. HF downloads use the already-configured CLI login. Subagent prompts carry the same rule.
- **Commits:** no `Co-Authored-By` or session trailers. No push or PR without the user's approval.
- **Tests:** Rust: `cargo test -p <crate>`. Python: `python -m pytest tools/tests/<file> -v` (OpenVINO-dependent tests `pytest.importorskip("openvino")`).

---

## File Structure

**Rust: types (`crates/cascadia-types/src/`)**
- `decision.rs` (new): `DecisionReadout` (with `cache_prefix`, `validate`, `warm_limit`), `ReadoutHead`, `JointQuestion`, `JointOption`, `HiddenNeed`, `MAX_DECISION_OUTPUTS`, `MAX_JOINT_QUESTIONS`.
- `task.rs` (modify): `GenerationTask.decision: Option<DecisionReadout>`.
- `chunk.rs` (modify): `Chunk.decision_logits: Option<Vec<f32>>` + `with_decision_logits`.
- `lib.rs` (modify): `pub mod decision; pub use decision::*;`

**Rust: API (`crates/cascadia-api/src/decisions/`)**
- `mod.rs`: `DecisionService` (profile + tokenizer + codec + token budgets), `DecisionError`, `sanitize_user_text`, `Sanitizer`, `prefill_work`, `softmax`.
- `schema.rs`: request/response types, `OrderedValue` (order-preserving JSON + Python-compatible renderers), validation, `answer_from_probs`.
- `profile.rs`: `decision.json` model (`DecisionProfile`).
- `jev.rs`, `kev.rs`, `clef.rs`: one codec per family (prompt → passes; logits → probabilities).
- `handler.rs`: the shared `evaluate` core and the `POST /v1/systemone` handler.
- `openai.rs`: OpenAI Decisions API types, the request → internal and answer → OpenAI conversions, and the `POST /v1/decisions` handler (Task 8c).
- `lib.rs` (modify): module, routes `/v1/decisions` (OpenAI) and `/v1/systemone`, `AppState.{decisions, decision_dump}`, `Config.{decisions, decision_dump}`.

**Rust: mock engine:** `crates/cascadia-engine-mock/src/lib.rs` (modify): deterministic decision logits, plus a failure sentinel.

**Rust: OpenVINO engine (`crates/cascadia-engine-openvino/src/`)**
- `decision.rs` (new): `HeadAssets` (load `decision/`), `Collector`, `run_head`, length-prefixed frame body codecs.
- `qwen36.rs` (modify):
  - manifest `final_hidden_output`
  - capture the last stage's output 1 (only while a decision pass needs it)
  - decision admission (validate, bounded warm resume, prefix snapshots at `cache_prefix`) and finish (single box)
  - pipeline frames `FRAME_DECIDE`/`FRAME_DECIDE_ACK`/`FRAME_DECIDE_RUN`/`FRAME_DECISION` with chunked bodies; HELLO_ACK capability bits; rank 0 tokenizes through `admitted_prompt_ids`
- `lib.rs` (modify): `mod decision;`

**Rust: CLI:** `crates/cascadia-cli/src/lib.rs` (modify): load `DecisionService` when `<model>/decision.json` exists and the engine is qwen35; `--decision-dump FILE`.

**Python: exporter (`tools/`)**
- `qwen36_surgery/weight_swap.py` (new): low-memory int8 weight swap and LoRA graft (port of the spike's `swapq.py`).
- `qwen36_surgery/export_qwen36_moe.py` (modify): `weights`, `lora`, `lora_scale` and `final_hidden` options.
- `decision/clef_head.py` (new): Clef joint head, the reference module and the tensorized export.
- `decision_export.py` (new): one command per family that writes the shard tree, `decision.json` and `decision/`.
- `decision/records.py` (new): the 50-record parity set.
- `decision/codec_golden.py` (new): server dump versus the reference encoders, and server answers versus the saved spike references.
- `decision/reference_stream.py` (new): streamed fp32 reference for all families.
- `decision/parity_check.py` (new): server versus reference agreement (per type; gate = raw ≥ 96% and no confident flip).
- `decision/openai_equivalence.py` (new): `/v1/decisions` (OpenAI format) must answer identically to `/v1/systemone` on the records.
- `decision/backbone_diff.py` (new): checks whether a checkpoint is byte-identical to its stock base.
- `tools/tests/test_weight_swap.py`, `tools/tests/test_decision_export.py`, `tools/tests/test_clef_head.py`, `tools/tests/test_decision_tools.py`, `tools/tests/test_reference_stream.py` (new).

**Docs:** `docs/decisions.md` (new in Task 8): API; Tasks 18–22 append each model's recipe and measured numbers.

---

### Task 1: Decision types on the task/chunk wire

**Files:**
- Create: `crates/cascadia-types/src/decision.rs`
- Modify: `crates/cascadia-types/src/lib.rs`, `crates/cascadia-types/src/task.rs`, `crates/cascadia-types/src/chunk.rs`
- Modify: every `GenerationTask { .. }` literal in the workspace (add `decision: None`), and every `Chunk { .. }` literal (add `decision_logits: None`).

**Interfaces:**
- Produces:
  - `cascadia_types::{DecisionReadout, ReadoutHead, JointQuestion, JointOption, HiddenNeed}`
  - `ReadoutHead::output_len(&self) -> usize`
  - `ReadoutHead::hidden_need(&self) -> HiddenNeed`
  - `DecisionReadout.cache_prefix: Option<u32>`: positions `[0, cache_prefix)` are shared by every pass of one request (Kev's state); the engine snapshots there and may warm-resume from it
  - `DecisionReadout::validate(&self, vocab: usize, max_tokens: usize) -> Result<(), String>`: bounds every id, position, span and count; run on EVERY admission (single box and pipeline peers)
  - `DecisionReadout::warm_limit(&self) -> usize`: the longest prefix the engine may restore from the prefix cache without losing a row or logit the head reads
  - `cascadia_types::{MAX_DECISION_OUTPUTS, MAX_JOINT_QUESTIONS}`
  - `GenerationTask.decision: Option<DecisionReadout>`
  - `Chunk.decision_logits: Option<Vec<f32>>`
  - `Chunk::with_decision_logits(self, Vec<f32>) -> Self`

- [ ] **Step 1: Write the failing tests** (append to `crates/cascadia-types/src/decision.rs`, which you create with only the test module first)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbalizer_round_trips_and_sizes() {
        let r = DecisionReadout {
            prompt_ids: vec![1, 2, 3],
            head: ReadoutHead::Verbalizer { token_ids: vec![10, 11] },
            cache_prefix: None,
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"kind\":\"verbalizer\""));
        let back: DecisionReadout = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
        assert_eq!(r.head.output_len(), 2);
        assert_eq!(r.head.hidden_need(), HiddenNeed::None);
    }

    #[test]
    fn pointer_needs_its_positions() {
        let h = ReadoutHead::Pointer { decide_pos: 9, option_pos: vec![4, 7] };
        assert_eq!(h.output_len(), 2);
        assert_eq!(h.hidden_need(), HiddenNeed::Positions(vec![4, 7, 9]));
    }

    #[test]
    fn joint_needs_all_rows_and_counts_options() {
        let h = ReadoutHead::Joint {
            questions: vec![
                JointQuestion { qtype: 1, span: [2, 4], options: vec![JointOption { span: [5, 6] }, JointOption { span: [7, 9] }] },
                JointQuestion { qtype: 0, span: [10, 11], options: vec![JointOption { span: [12, 13] }] },
            ],
        };
        assert_eq!(h.output_len(), 3);
        assert_eq!(h.hidden_need(), HiddenNeed::All);
    }

    fn readout(prompt_len: u32, head: ReadoutHead) -> DecisionReadout {
        DecisionReadout { prompt_ids: (0..prompt_len).collect(), head, cache_prefix: None }
    }

    #[test]
    fn validate_rejects_out_of_range_ids_positions_and_spans() {
        let ok = readout(10, ReadoutHead::Pointer { decide_pos: 9, option_pos: vec![3, 5] });
        assert!(ok.validate(100, 64).is_ok());
        assert!(readout(10, ReadoutHead::Verbalizer { token_ids: vec![100] }).validate(100, 64).is_err(), "id >= vocab");
        assert!(readout(0, ReadoutHead::Verbalizer { token_ids: vec![1] }).validate(100, 64).is_err(), "empty prompt");
        assert!(readout(65, ReadoutHead::Verbalizer { token_ids: vec![1] }).validate(100, 64).is_err(), "prompt > max_tokens");
        let mut bad_id = readout(3, ReadoutHead::Verbalizer { token_ids: vec![1] });
        bad_id.prompt_ids[1] = 100;
        assert!(bad_id.validate(100, 64).is_err(), "prompt id >= vocab");
        assert!(readout(10, ReadoutHead::Pointer { decide_pos: 10, option_pos: vec![3] }).validate(100, 64).is_err());
        let span = |a: u32, b: u32| ReadoutHead::Joint { questions: vec![JointQuestion { qtype: 0, span: [a, b], options: vec![JointOption { span: [1, 2] }] }] };
        assert!(readout(10, span(4, 4)).validate(100, 64).is_err(), "empty span");
        assert!(readout(10, span(5, 4)).validate(100, 64).is_err(), "reversed span");
        assert!(readout(10, span(4, 11)).validate(100, 64).is_err(), "span past prompt");
        assert!(readout(10, ReadoutHead::Joint { questions: vec![] }).validate(100, 64).is_err(), "no questions");
        let overlap = ReadoutHead::Joint { questions: vec![JointQuestion { qtype: 1, span: [1, 2], options: vec![JointOption { span: [3, 6] }, JointOption { span: [4, 7] }] }] };
        assert!(readout(10, overlap).validate(100, 64).is_err(), "overlapping option spans");
        assert!(readout(10, ReadoutHead::Joint { questions: vec![JointQuestion { qtype: 3, span: [1, 2], options: vec![JointOption { span: [2, 3] }] }] }).validate(100, 64).is_err(), "bad qtype");
        let mut cp = readout(10, ReadoutHead::Verbalizer { token_ids: vec![1] });
        cp.cache_prefix = Some(10);
        assert!(cp.validate(100, 64).is_err(), "cache_prefix must leave >= 1 token to prefill");
    }

    #[test]
    fn warm_limit_keeps_every_row_the_head_reads() {
        assert_eq!(readout(10, ReadoutHead::Verbalizer { token_ids: vec![1] }).warm_limit(), 9);
        assert_eq!(readout(10, ReadoutHead::Pointer { decide_pos: 9, option_pos: vec![6, 4] }).warm_limit(), 4);
        let j = ReadoutHead::Joint { questions: vec![JointQuestion { qtype: 0, span: [1, 2], options: vec![JointOption { span: [2, 3] }] }] };
        assert_eq!(readout(10, j).warm_limit(), 0);
    }

    #[test]
    fn task_without_decision_field_still_parses() {
        let t: crate::GenerationTask = serde_json::from_str(r#"{"task_id":"t","prompt":"p"}"#).unwrap();
        assert!(t.decision.is_none());
    }

    #[test]
    fn chunk_carries_decision_logits() {
        let c = crate::Chunk::final_marker("t", "").with_decision_logits(vec![0.5, -1.0]);
        let s = serde_json::to_string(&c).unwrap();
        let back: crate::Chunk = serde_json::from_str(&s).unwrap();
        assert_eq!(back.decision_logits, Some(vec![0.5, -1.0]));
        let plain = serde_json::to_string(&crate::Chunk::final_marker("t", "")).unwrap();
        assert!(!plain.contains("decision_logits"));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-types decision`
Expected: compile errors (`DecisionReadout` etc. not defined).

- [ ] **Step 3: Implement the types** (top of `crates/cascadia-types/src/decision.rs`)

```rust
//! Decision readout: a prefill-only pass whose result is per-option logits
//! computed by the model's decision head (see docs/decisions.md). The API's
//! codec tokenizes and lays out the prompt; the engine prefills `prompt_ids`
//! exactly (no chat template, no think block, no special tokens added) and
//! never decodes.

use serde::{Deserialize, Serialize};

/// Upper bound on logits one readout may request (Clef: options summed over questions).
pub const MAX_DECISION_OUTPUTS: usize = 4096;
/// Upper bound on questions in one joint (Clef) readout.
pub const MAX_JOINT_QUESTIONS: usize = 256;
/// Prompt-length bound for a decision pass when the engine does not know the model window
/// (pipeline ranks other than 0 never read `max_position_embeddings`).
pub const MAX_DECISION_PROMPT: usize = 131_072;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionReadout {
    /// Exact ids to prefill, already rendered and tokenized by the codec.
    pub prompt_ids: Vec<u32>,
    pub head: ReadoutHead,
    /// `prompt_ids[..cache_prefix]` is shared by every pass of one request (Kev: the state).
    /// The engine snapshots the prefix cache there so later passes prefill only their tail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_prefix: Option<u32>,
}

/// Which head reads the prefilled state. Positions/spans index `prompt_ids`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadoutHead {
    /// JEV: lm_head logits at the last position for these vocab ids.
    Verbalizer { token_ids: Vec<u32> },
    /// Kev: pointer head over final-norm hidden states at `option_pos`
    /// (each option's closing token) against `decide_pos`.
    Pointer { decide_pos: u32, option_pos: Vec<u32> },
    /// Clef: joint schema head over all final-norm hidden states.
    Joint { questions: Vec<JointQuestion> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JointQuestion {
    /// 0 = noul, 1 = choice, 2 = score (Clef's QUESTION_TYPES).
    pub qtype: u8,
    /// Half-open [start, end) token span of the instruction text.
    pub span: [u32; 2],
    pub options: Vec<JointOption>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JointOption {
    /// Half-open [start, end) token span of the option's rendered semantics.
    pub span: [u32; 2],
}

/// Final-norm hidden rows the head needs collected during prefill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HiddenNeed {
    None,
    /// Sorted, de-duplicated absolute positions.
    Positions(Vec<u32>),
    All,
}

impl DecisionReadout {
    /// Bounds-check everything an engine indexes with. Readouts cross a trust boundary (HTTP
    /// clients via the codec, and pipeline peers via FRAME_DECIDE), so the engine calls this on
    /// every admission instead of trusting the sender.
    pub fn validate(&self, vocab: usize, max_tokens: usize) -> Result<(), String> {
        let n = self.prompt_ids.len();
        if n == 0 || n > max_tokens {
            return Err(format!("decision prompt has {n} tokens; allowed 1..={max_tokens}"));
        }
        if let Some(&bad) = self.prompt_ids.iter().find(|&&t| t as usize >= vocab) {
            return Err(format!("decision prompt id {bad} >= vocab {vocab}"));
        }
        if let Some(c) = self.cache_prefix {
            if c as usize >= n {
                return Err(format!("cache_prefix {c} must be < prompt length {n}"));
            }
        }
        let out = self.head.output_len();
        if out == 0 || out > MAX_DECISION_OUTPUTS {
            return Err(format!("decision head asks for {out} logits; allowed 1..={MAX_DECISION_OUTPUTS}"));
        }
        let span_ok = |s: &[u32; 2]| s[0] < s[1] && s[1] as usize <= n;
        match &self.head {
            ReadoutHead::Verbalizer { token_ids } => {
                if let Some(&bad) = token_ids.iter().find(|&&t| t as usize >= vocab) {
                    return Err(format!("verbalizer id {bad} >= vocab {vocab}"));
                }
            }
            ReadoutHead::Pointer { decide_pos, option_pos } => {
                if let Some(&bad) = option_pos.iter().chain(std::iter::once(decide_pos)).find(|&&p| p as usize >= n) {
                    return Err(format!("pointer position {bad} outside the {n}-token prompt"));
                }
            }
            ReadoutHead::Joint { questions } => {
                if questions.is_empty() || questions.len() > MAX_JOINT_QUESTIONS {
                    return Err(format!("joint readout has {} questions; allowed 1..={MAX_JOINT_QUESTIONS}", questions.len()));
                }
                // Spans must be in prompt order and disjoint (question span, its options, next question…),
                // exactly as the codec lays them out. Overlapping spans would let a peer multiply
                // lm_head reads and span-mask size without bound.
                let mut cursor = 0u32;
                for (i, q) in questions.iter().enumerate() {
                    if q.qtype > 2 || q.options.is_empty() {
                        return Err(format!("joint question {i}: bad qtype or no options"));
                    }
                    for s in std::iter::once(&q.span).chain(q.options.iter().map(|o| &o.span)) {
                        if !span_ok(s) || s[0] < cursor {
                            return Err(format!("joint question {i}: span {s:?} empty, out of range or out of order"));
                        }
                        cursor = s[1];
                    }
                }
            }
        }
        Ok(())
    }

    /// Longest prefix the engine may restore from its prefix cache for this pass. A restored
    /// prefix is never re-run, so no hidden row or logit inside it is produced: the limit is
    /// the first position the head reads (Verbalizer: the last token's logits; Joint: every row).
    pub fn warm_limit(&self) -> usize {
        match &self.head {
            ReadoutHead::Verbalizer { .. } => self.prompt_ids.len() - 1,
            ReadoutHead::Pointer { decide_pos, option_pos } => option_pos.iter().chain(std::iter::once(decide_pos)).copied().min().unwrap_or(0) as usize,
            ReadoutHead::Joint { .. } => 0,
        }
    }
}

impl ReadoutHead {
    /// Number of logits the engine returns for this head.
    pub fn output_len(&self) -> usize {
        match self {
            ReadoutHead::Verbalizer { token_ids } => token_ids.len(),
            ReadoutHead::Pointer { option_pos, .. } => option_pos.len(),
            ReadoutHead::Joint { questions } => questions.iter().map(|q| q.options.len()).sum(),
        }
    }

    pub fn hidden_need(&self) -> HiddenNeed {
        match self {
            ReadoutHead::Verbalizer { .. } => HiddenNeed::None,
            ReadoutHead::Pointer { decide_pos, option_pos } => {
                let mut v = option_pos.clone();
                v.push(*decide_pos);
                v.sort_unstable();
                v.dedup();
                HiddenNeed::Positions(v)
            }
            ReadoutHead::Joint { .. } => HiddenNeed::All,
        }
    }
}
```

In `crates/cascadia-types/src/lib.rs` add next to the other modules:

```rust
pub mod decision;
pub use decision::*;
```

In `crates/cascadia-types/src/task.rs`, add as the last field of `GenerationTask` (after `resume_token_ids`):

```rust
    /// Decision readout (docs/decisions.md). `Some` ⇒ the engine prefills
    /// `decision.prompt_ids` verbatim and emits ONE final chunk carrying
    /// `Chunk::decision_logits`; `prompt`, sampling and `max_tokens` are ignored.
    /// Engines without decision support fail the task with a
    /// `decision_unsupported:`-prefixed error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<crate::DecisionReadout>,
```

In `crates/cascadia-types/src/chunk.rs`, add as the last field of `Chunk`:

```rust
    /// Raw per-option logits of a decision readout, on the FINAL chunk only
    /// (length = `ReadoutHead::output_len`). `None` on every generation chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_logits: Option<Vec<f32>>,
```

Add `decision_logits: None,` to the `Chunk::token`, `Chunk::final_marker` and `Chunk::error` constructors. Add the builder next to the other `with_*` methods:

```rust
    pub fn with_decision_logits(mut self, logits: Vec<f32>) -> Self {
        self.decision_logits = Some(logits);
        self
    }
```

- [ ] **Step 4: Fix every struct literal in the workspace**

Run: `cargo build --workspace --all-targets 2>&1 | grep -E "missing field .(decision|decision_logits)." -B2 | grep -- "-->"`
For each reported location, add `decision: None,` after `resume_token_ids: ...,` (GenerationTask) or `decision_logits: None,` (Chunk). Repeat until `cargo build --workspace --all-targets` succeeds.

- [ ] **Step 5: Run tests**

Run: `cargo test -p cascadia-types`
Expected: PASS, including the 7 new tests (the validate test now also covers overlapping spans).

- [ ] **Step 6: Commit**

```bash
git add -A crates/
git commit -m "feat(types): decision readout on GenerationTask and decision_logits on Chunk"
```

---

### Task 2: Mock engine answers decision tasks

**Files:**
- Modify: `crates/cascadia-engine-mock/src/lib.rs`

**Interfaces:**
- Consumes: `GenerationTask.decision`, `ReadoutHead::output_len`, `Chunk::with_decision_logits`.
- Produces: for a decision task, one final chunk with `decision_logits[i] = -(i as f32)` (the first option always wins) and `prompt_tokens = prompt_ids.len()`. A decision task whose `prompt_ids` contain the sentinel id `999_999` instead fails with `Chunk::error(id, "mock engine failure")`. Task 8's API tests rely on both.

- [ ] **Step 1: Write the failing test** (in the existing `#[cfg(test)] mod tests` of the mock crate)

```rust
    #[test]
    fn decision_task_returns_one_final_chunk_with_logits() {
        let mut e = MockEngine::new();
        let mut t = GenerationTask {
            task_id: "d1".into(),
            prompt: String::new(),
            max_tokens: 8,
            temperature: 0.0,
            logprobs: 0,
            sampling: Default::default(),
            enable_thinking: false,
            trust_remote_code: false,
            tenant: String::new(),
            resume_token_ids: None,
            decision: None,
        };
        t.decision = Some(cascadia_types::DecisionReadout {
            prompt_ids: vec![5, 6, 7, 8],
            head: cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1, 2, 3] },
            cache_prefix: None,
        });
        e.submit(t).unwrap();
        let out = e.step().unwrap();
        assert_eq!(out.len(), 1);
        let c = &out[0].1;
        assert!(c.is_final);
        assert_eq!(c.decision_logits, Some(vec![0.0, -1.0, -2.0]));
        assert_eq!(c.prompt_tokens, Some(4));
        assert!(e.step().unwrap().is_empty());
    }

    #[test]
    fn decision_task_with_sentinel_id_fails() {
        let mut e = MockEngine::new();
        let t = GenerationTask {
            task_id: "d2".into(), prompt: String::new(), max_tokens: 8, temperature: 0.0, logprobs: 0,
            sampling: Default::default(), enable_thinking: false, trust_remote_code: false, tenant: String::new(),
            resume_token_ids: None,
            decision: Some(cascadia_types::DecisionReadout {
                prompt_ids: vec![5, 999_999],
                head: cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] },
                cache_prefix: None,
            }),
        };
        e.submit(t).unwrap();
        let out = e.step().unwrap();
        assert_eq!(out[0].1.error.as_deref(), Some("mock engine failure"));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-engine-mock decision_task`
Expected: FAIL (the mock echoes words; `decision_logits` is `None`).

- [ ] **Step 3: Implement** in `MockEngine::step`, right after the `__engine_error__` sentinel block:

```rust
        if let Some(d) = &task.decision {
            if d.prompt_ids.contains(&999_999) {
                return Ok(vec![(task.task_id.clone(), Chunk::error(task.task_id, "mock engine failure"))]);
            }
            let logits: Vec<f32> = (0..d.head.output_len()).map(|i| -(i as f32)).collect();
            let chunk = Chunk::final_marker(task.task_id.clone(), "")
                .with_prompt_tokens(d.prompt_ids.len() as u32)
                .with_finish_reason(cascadia_types::FinishReason::Stop)
                .with_decision_logits(logits);
            return Ok(vec![(task.task_id, chunk)]);
        }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-engine-mock`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-engine-mock/src/lib.rs
git commit -m "feat(mock): deterministic decision logits for decision tasks"
```

---

### Task 3: API decision schema, validation and answers

**Files:**
- Create: `crates/cascadia-api/src/decisions/mod.rs`, `crates/cascadia-api/src/decisions/schema.rs`
- Modify: `crates/cascadia-api/src/lib.rs` (add `pub mod decisions;`)
- Modify: `crates/cascadia-api/Cargo.toml` (add `indexmap = { version = "2", features = ["serde"] }` and `tokenizers = { workspace = true }`)

**Interfaces:**
- Produces:
  - `decisions::schema::{DecisionRequest, Question, QuestionType, DecisionResponse, Answer, Usage}`
  - `DecisionRequest::validate(&self, limits: &Limits) -> Result<(), DecisionError>`
  - `answer_from_probs(q: &Question, probs: &[f64]) -> Answer`
  - `decisions::DecisionError` (with `status()`), `decisions::schema::Limits { max_questions: usize, max_options: usize, max_levels: usize, min_levels: usize }`
  - `schema::{choice_confidence, score_confidence, round_prob}`: TypeSafe's `system-one-adapter 0.2.1` formulas (as mirrored in Kev's `kev/api.py`) and Python `round(x, 4)`
  - `Question::option_keys(&self) -> Vec<String>`: report keys in option order (choice = criteria names in request order; noul = `["false","true"]`; score = `"0".."L-1"`)
  - `schema::OrderedValue`: JSON value whose objects keep request key order (custom `Deserialize`/`Serialize`)
  - `OrderedValue::py_json(&self, sort_keys: bool, compact: bool) -> String`: Python `json.dumps(v, ensure_ascii=False, sort_keys=…, separators=(",",":") if compact else (", ",": "))`, floats included (`py_float` reproduces Python's `float.__repr__`: `1e+16`, `1e-05`, `1.0`)
  - `pub fn py_float(x: f64) -> String`
  - `OrderedValue::py_str(&self) -> String`: Python `str()` for scalars (`True`/`False`, `None` → `"None"`); containers fall back to `py_json(false, false)`
  - `OrderedValue::{get(&self, key) -> Option<&OrderedValue>, as_str(&self) -> Option<&str>, is_null(&self) -> bool}`

- [ ] **Step 1: Write the failing tests** (bottom of `schema.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Parse from TEXT: `serde_json::json!` builds a sorted Map (no preserve_order), so going
    /// through a `Value` would lose the order these tests check.
    fn req(v: serde_json::Value) -> DecisionRequest {
        serde_json::from_str(&v.to_string()).unwrap()
    }
    fn req_str(s: &str) -> DecisionRequest {
        serde_json::from_str(s).unwrap()
    }

    const LIMITS: Limits = Limits { max_questions: 3, max_options: 16, min_levels: 2, max_levels: 10 };

    #[test]
    fn parses_jev_body_preserving_question_order() {
        let r = req_str(r#"{
            "model": "m",
            "state": {"msg": "charged twice"},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
                "urgent": {"type": "noul", "instructions": "Urgent?"},
                "sev": {"type": "score", "instructions": "How bad?", "criteria": ["none", "minor", "major"]}
            }
        }"#);
        let ids: Vec<&str> = r.questions.keys().map(String::as_str).collect();
        assert_eq!(ids, ["team", "urgent", "sev"]);
        assert_eq!(r.questions["team"].option_keys(), ["billing", "tech"]);
        assert_eq!(r.questions["urgent"].option_keys(), ["false", "true"]);
        assert_eq!(r.questions["sev"].option_keys(), ["0", "1", "2"]);
        r.validate(&LIMITS).unwrap();
    }

    #[test]
    fn rejects_empty_questions_images_and_bad_criteria() {
        let base = |q: serde_json::Value| req(serde_json::json!({"model": "m", "state": "s", "questions": q}));
        assert!(base(serde_json::json!({})).validate(&LIMITS).is_err());
        assert!(base(serde_json::json!({"c": {"type": "choice", "criteria": {}}})).validate(&LIMITS).is_err());
        assert!(base(serde_json::json!({"s": {"type": "score", "criteria": ["only"]}})).validate(&LIMITS).is_err());
        let mut many = serde_json::Map::new();
        for i in 0..17 { many.insert(format!("o{i}"), serde_json::json!("x")); }
        assert!(base(serde_json::json!({"c": {"type": "choice", "criteria": many}})).validate(&LIMITS).is_err());
        assert!(base(serde_json::json!({"a": {"type": "noul"}, "b": {"type": "noul"}, "c": {"type": "noul"}, "d": {"type": "noul"}})).validate(&LIMITS).is_err(), "over max_questions");
        assert!(base(serde_json::json!({"": {"type": "noul"}})).validate(&LIMITS).is_err(), "empty question id");
        let img = req(serde_json::json!({"model": "m", "state": "s", "images": ["data:x"], "questions": {"u": {"type": "noul"}}}));
        assert!(matches!(img.validate(&LIMITS), Err(DecisionError::BadRequest(m)) if m.contains("images")));
    }

    #[test]
    fn ordered_value_keeps_request_order_and_renders_like_python() {
        let v: OrderedValue = serde_json::from_str(r#"{"b": 1, "a": {"y": true, "x": null}, "é": [1.5, "q\"t"]}"#).unwrap();
        assert_eq!(v.py_json(false, false), r#"{"b": 1, "a": {"y": true, "x": null}, "é": [1.5, "q\"t"]}"#);
        assert_eq!(v.py_json(true, true), r#"{"a":{"x":null,"y":true},"b":1,"é":[1.5,"q\"t"]}"#);
        assert_eq!(OrderedValue::Bool(true).py_str(), "True");
        assert_eq!(OrderedValue::Null.py_str(), "None");
        // duplicate keys: last value wins, first position kept (Python dict semantics)
        let d: OrderedValue = serde_json::from_str(r#"{"k": 1, "z": 2, "k": 3}"#).unwrap();
        assert_eq!(d.py_json(false, true), r#"{"k":3,"z":2}"#);
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"{"b":1,"a":{"y":true,"x":null},"é":[1.5,"q\"t"]}"#);
    }

    #[test]
    fn many_keys_parse_in_linear_time() {
        let body = format!("{{{}}}", (0..200_000).map(|i| format!("\"k{i}\":{i}")).collect::<Vec<_>>().join(","));
        let t = std::time::Instant::now();
        let v: OrderedValue = serde_json::from_str(&body).unwrap();
        assert!(matches!(v, OrderedValue::Object(ref m) if m.len() == 200_000));
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "{:?}", t.elapsed());
    }

    #[test]
    fn floats_render_like_python_repr() {
        // expected values are Python's repr() output
        for (x, want) in [(1e16, "1e+16"), (1e-05, "1e-05"), (0.0001, "0.0001"), (1.0, "1.0"), (1.5, "1.5"),
                          (123456789012345678.0, "1.2345678901234568e+17"), (-0.0, "-0.0"), (0.1, "0.1"),
                          (1e22, "1e+22"), (5e-324, "5e-324"), (2.5e-05, "2.5e-05"), (100.0, "100.0"), (1e15, "1000000000000000.0")] {
            assert_eq!(py_float(x), want, "{x:e}");
        }
        let v: OrderedValue = serde_json::from_str(r#"{"a": 1e16, "b": 3, "c": 2.50}"#).unwrap();
        assert_eq!(v.py_json(false, false), r#"{"a": 1e+16, "b": 3, "c": 2.5}"#);
    }

    #[test]
    fn choice_criteria_keep_request_order_not_alphabetical() {
        let r = req_str(r#"{"model": "m", "state": "s", "questions": {"c": {"type": "choice", "criteria": {"zeta": "z", "alpha": "a"}}}}"#);
        assert_eq!(r.questions["c"].option_keys(), ["zeta", "alpha"]);
    }

    #[test]
    fn answers_follow_systemone_shape() {
        let r = req_str(r#"{"model": "m", "state": "s", "questions": {
            "c": {"type": "choice", "criteria": {"a": "x", "b": "y"}},
            "n": {"type": "noul"},
            "s": {"type": "score", "criteria": ["lo", "mid", "hi"]}}}"#);
        // TypeSafe choice confidence: (p_max - 1/K) / (1 - 1/K) = (0.75 - 0.5) / 0.5
        let c = serde_json::to_value(answer_from_probs(&r.questions["c"], &[0.25, 0.75])).unwrap();
        assert_eq!(c, serde_json::json!({"type": "choice", "choice": "b", "confidence": 0.5, "probabilities": {"a": 0.25, "b": 0.75}}));
        let n = serde_json::to_value(answer_from_probs(&r.questions["n"], &[0.1, 0.9])).unwrap();
        assert_eq!(n, serde_json::json!({"type": "noul", "noul": 0.9}));
        let s = serde_json::to_value(answer_from_probs(&r.questions["s"], &[0.2, 0.3, 0.5])).unwrap();
        assert_eq!(s["score"], serde_json::json!(1.3));
        // TypeSafe score confidence: max(0, 1 - E|i - mode| / D); mode 2, E = 0.7, D = 2/3 → clipped to 0
        assert_eq!(s["confidence"], serde_json::json!(0.0));
        assert_eq!(s["legend"], serde_json::json!({"0": "lo", "1": "mid", "2": "hi"}));
    }

    #[test]
    fn typesafe_confidence_and_python_rounding() {
        assert_eq!(choice_confidence(&[0.25, 0.25, 0.25, 0.25]), 0.0);
        assert_eq!(choice_confidence(&[1.0]), 1.0);
        assert!((score_confidence(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert_eq!(score_confidence(&[1.0 / 3.0; 3]), 0.0);
        // Python rounds the EXACT binary value: 0.00125 is stored as 0.0012500000000000000260…, so it rounds up,
        // and 0.00015 is stored as 0.000149999999999999986…, so it rounds down (Python: 0.0013 and 0.0001).
        assert_eq!(round_prob(0.00125), 0.0013);
        assert_eq!(round_prob(0.00015), 0.0001);
        assert_eq!(round_prob(0.123456), 0.1235);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-api decisions::schema`
Expected: compile error (module missing).

- [ ] **Step 3: Implement `decisions/mod.rs`** (initial version; Task 4 adds `DecisionService`)

```rust
//! Typed decisions over decision models (docs/decisions.md): POST /v1/decisions (OpenAI format,
//! openai.rs) and POST /v1/systemone (Jev/SystemOne format, handler.rs), sharing one core.

pub mod schema;

use axum::http::StatusCode;

#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    TooLong(String),
    #[error("this model does not serve decisions")]
    NotServed,
    #[error("{0}")]
    Engine(String),
}

impl DecisionError {
    pub fn status(&self) -> StatusCode {
        match self {
            DecisionError::BadRequest(_) => StatusCode::BAD_REQUEST,
            DecisionError::TooLong(_) => StatusCode::PAYLOAD_TOO_LARGE,
            DecisionError::NotServed => StatusCode::NOT_IMPLEMENTED,
            DecisionError::Engine(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}
```

- [ ] **Step 4: Implement `decisions/schema.rs`**

```rust
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::DecisionError;

/// JSON value that keeps object keys in REQUEST order. `serde_json::Value`'s map sorts keys unless the
/// workspace-wide `preserve_order` feature is on, and turning that on would reorder every chat response.
/// Codecs must render exactly what their reference implementations saw, so request JSON is parsed into this.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedValue>),
    Object(Vec<(String, OrderedValue)>),
}

impl<'de> Deserialize<'de> for OrderedValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = OrderedValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<OrderedValue, E> { Ok(OrderedValue::Null) }
            fn visit_none<E>(self) -> Result<OrderedValue, E> { Ok(OrderedValue::Null) }
            fn visit_some<D2: serde::Deserializer<'de>>(self, d: D2) -> Result<OrderedValue, D2::Error> { OrderedValue::deserialize(d) }
            fn visit_bool<E>(self, b: bool) -> Result<OrderedValue, E> { Ok(OrderedValue::Bool(b)) }
            fn visit_i64<E>(self, n: i64) -> Result<OrderedValue, E> { Ok(OrderedValue::Number(n.into())) }
            fn visit_u64<E>(self, n: u64) -> Result<OrderedValue, E> { Ok(OrderedValue::Number(n.into())) }
            fn visit_f64<E: serde::de::Error>(self, n: f64) -> Result<OrderedValue, E> {
                serde_json::Number::from_f64(n).map(OrderedValue::Number).ok_or_else(|| E::custom("non-finite number"))
            }
            fn visit_str<E>(self, s: &str) -> Result<OrderedValue, E> { Ok(OrderedValue::String(s.to_owned())) }
            fn visit_string<E>(self, s: String) -> Result<OrderedValue, E> { Ok(OrderedValue::String(s)) }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut a: A) -> Result<OrderedValue, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(OrderedValue::Array(v))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut a: A) -> Result<OrderedValue, A::Error> {
                let mut v: Vec<(String, OrderedValue)> = Vec::new();
                // key → index in `v`: O(n) even for objects with many keys (this runs before any limit).
                let mut at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                while let Some((k, x)) = a.next_entry::<String, OrderedValue>()? {
                    // Python dict semantics: a repeated key keeps its first position, takes the last value.
                    match at.get(&k) {
                        Some(&i) => v[i].1 = x,
                        None => {
                            at.insert(k.clone(), v.len());
                            v.push((k, x));
                        }
                    }
                }
                Ok(OrderedValue::Object(v))
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for OrderedValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            OrderedValue::Null => s.serialize_unit(),
            OrderedValue::Bool(b) => s.serialize_bool(*b),
            OrderedValue::Number(n) => n.serialize(s),
            OrderedValue::String(x) => s.serialize_str(x),
            OrderedValue::Array(a) => {
                let mut q = s.serialize_seq(Some(a.len()))?;
                for x in a {
                    q.serialize_element(x)?;
                }
                q.end()
            }
            OrderedValue::Object(m) => {
                let mut q = s.serialize_map(Some(m.len()))?;
                for (k, x) in m {
                    q.serialize_entry(k, x)?;
                }
                q.end()
            }
        }
    }
}

impl OrderedValue {
    pub fn is_null(&self) -> bool {
        matches!(self, OrderedValue::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            OrderedValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&OrderedValue> {
        match self {
            OrderedValue::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Python `json.dumps(v, ensure_ascii=False, sort_keys=sort_keys,
    /// separators=(",", ":") if compact else (", ", ": "))`. String escaping is serde_json's, which
    /// matches Python with ensure_ascii=False (escapes `"` `\` and control chars as `\n`/`\u001f`,
    /// keeps non-ASCII). Floats use `py_float`, Python's repr. Known gap: integers beyond u64 parse
    /// as f64 (serde_json has no arbitrary precision here), where Python keeps an int.
    pub fn py_json(&self, sort_keys: bool, compact: bool) -> String {
        let mut out = String::new();
        self.write_py(&mut out, sort_keys, compact);
        out
    }

    fn write_py(&self, out: &mut String, sort: bool, compact: bool) {
        let (item, kv) = if compact { (",", ":") } else { (", ", ": ") };
        match self {
            OrderedValue::Null => out.push_str("null"),
            OrderedValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            OrderedValue::Number(n) => out.push_str(&py_number(n)),
            OrderedValue::String(s) => out.push_str(&serde_json::to_string(s).expect("string serializes")),
            OrderedValue::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    x.write_py(out, sort, compact);
                }
                out.push(']');
            }
            OrderedValue::Object(m) => {
                let mut entries: Vec<&(String, OrderedValue)> = m.iter().collect();
                if sort {
                    // Python sorts str keys by code point; UTF-8 byte order is the same order.
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (i, (k, x)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    out.push_str(&serde_json::to_string(k).expect("key serializes"));
                    out.push_str(kv);
                    x.write_py(out, sort, compact);
                }
                out.push('}');
            }
        }
    }

    /// Python `str(v)` for scalars (Kev's render): `True`/`False`, `None`, numbers as written.
    pub fn py_str(&self) -> String {
        match self {
            OrderedValue::Null => "None".into(),
            OrderedValue::Bool(true) => "True".into(),
            OrderedValue::Bool(false) => "False".into(),
            OrderedValue::Number(n) => py_number(n),
            OrderedValue::String(s) => s.clone(),
            other => other.py_json(false, false),
        }
    }
}

/// Python's `float.__repr__` (what `json.dumps` and `str()` print): the shortest round-trip digits,
/// fixed notation for exponents −4..16 (always with a fractional part, `1.0`), scientific otherwise
/// with a signed two-digit-minimum exponent (`1e+16`, `1e-05`). Rust's `{:e}` already gives the
/// shortest digits; only the layout differs.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (sign, mant) = match mant.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        format!("{sign}{m}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    }
}

fn py_number(n: &serde_json::Number) -> String {
    if n.is_f64() {
        py_float(n.as_f64().expect("f64 number"))
    } else {
        n.to_string()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: String,
    /// Any JSON value, in request order; codecs render it.
    pub state: OrderedValue,
    pub questions: IndexMap<String, Question>,
    #[serde(default)]
    pub images: Vec<serde_json::Value>,
    #[serde(default)]
    pub videos: Vec<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QuestionType,
    #[serde(default)]
    pub instructions: Option<OrderedValue>,
    /// choice: object name → description (request order); score: array of level descriptions;
    /// noul: optional {"true": ..., "false": ...} overrides.
    #[serde(default)]
    pub criteria: Option<OrderedValue>,
}

/// Per-profile limits (the codec decides them: JEV's head has 16 choice slots). `max_questions`
/// bounds engine work per request: JEV and Kev run one prefill pass per question.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_questions: usize,
    pub max_options: usize,
    pub min_levels: usize,
    pub max_levels: usize,
}

impl Question {
    /// (name, description) in REQUEST order.
    pub fn choice_criteria(&self) -> Vec<(String, OrderedValue)> {
        match &self.criteria {
            Some(OrderedValue::Object(m)) => m.clone(),
            _ => Vec::new(),
        }
    }

    pub fn score_levels(&self) -> Vec<OrderedValue> {
        match &self.criteria {
            Some(OrderedValue::Array(a)) => a.clone(),
            _ => Vec::new(),
        }
    }

    pub fn option_keys(&self) -> Vec<String> {
        match self.qtype {
            QuestionType::Noul => vec!["false".into(), "true".into()],
            QuestionType::Choice => self.choice_criteria().into_iter().map(|(k, _)| k).collect(),
            QuestionType::Score => (0..self.score_levels().len()).map(|i| i.to_string()).collect(),
        }
    }
}

impl DecisionRequest {
    pub fn validate(&self, limits: &Limits) -> Result<(), DecisionError> {
        let bad = |m: String| Err(DecisionError::BadRequest(m));
        if !self.images.is_empty() || !self.videos.is_empty() {
            return bad("images and videos are not supported by this server (text-only decision models)".into());
        }
        if self.questions.is_empty() || self.questions.len() > limits.max_questions {
            return bad(format!("between 1 and {} questions are allowed, got {}", limits.max_questions, self.questions.len()));
        }
        for (id, q) in &self.questions {
            if id.is_empty() {
                return bad("question ids must be non-empty".into());
            }
            match q.qtype {
                QuestionType::Noul => {}
                QuestionType::Choice => {
                    let n = q.choice_criteria().len();
                    if n == 0 || n > limits.max_options {
                        return bad(format!("{id}: choice criteria must have 1..={} options, got {n}", limits.max_options));
                    }
                }
                QuestionType::Score => {
                    let n = q.score_levels().len();
                    if n < limits.min_levels || n > limits.max_levels {
                        return bad(format!("{id}: score criteria must have {}..={} levels, got {n}", limits.min_levels, limits.max_levels));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul { noul: f64 },
    Choice { choice: String, confidence: f64, probabilities: IndexMap<String, f64> },
    Score { score: f64, confidence: f64, legend: IndexMap<String, OrderedValue>, probabilities: IndexMap<String, f64> },
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

/// Python `round(x, 4)`: both round the EXACT binary value half-to-even, and Rust's `{:.4}`
/// formatting is exact, so the 4th decimal matches Kev's and Clef's own servers.
pub fn round_prob(x: f64) -> f64 {
    format!("{x:.4}").parse().expect("formatted float parses")
}

fn normalize(p: &[f64]) -> Vec<f64> {
    let t: f64 = p.iter().sum();
    if t == 0.0 { vec![1.0 / p.len() as f64; p.len()] } else { p.iter().map(|x| x / t).collect() }
}

/// TypeSafe system-one-adapter 0.2.1 (mirrored in kev/api.py): (p_max - 1/K) / (1 - 1/K);
/// 0 at uniform, 1 at certainty, 1 for a single option.
pub fn choice_confidence(p: &[f64]) -> f64 {
    let k = p.len() as f64;
    if p.len() == 1 {
        return 1.0;
    }
    let m = normalize(p).into_iter().fold(f64::NEG_INFINITY, f64::max);
    (m - 1.0 / k) / (1.0 - 1.0 / k)
}

/// TypeSafe: max(0, 1 - E|level - mode| / D), D = mean absolute deviation of a uniform
/// distribution over the L levels around (L-1)/2; mode = first most likely level.
pub fn score_confidence(p: &[f64]) -> f64 {
    let l = p.len();
    if l == 1 {
        return 1.0;
    }
    let p = normalize(p);
    let mode = p.iter().enumerate().fold(0, |b, (i, &x)| if x > p[b] { i } else { b });
    let mid = (l as f64 - 1.0) / 2.0;
    let d = (0..l).map(|i| (i as f64 - mid).abs()).sum::<f64>() / l as f64;
    let e: f64 = p.iter().enumerate().map(|(i, x)| x * (i as f64 - mode as f64).abs()).sum();
    (1.0 - e / d).max(0.0)
}

/// Jev/SystemOne answers with TypeSafe's semantics (the wire format's owner): choice = argmax
/// (first on ties) with normalized confidence; score = probability-weighted level with the MAD
/// confidence; noul = p(true). `probs` is in `Question::option_keys` order.
pub fn answer_from_probs(q: &Question, probs: &[f64]) -> Answer {
    let keys = q.option_keys();
    let dist: IndexMap<String, f64> = keys.iter().cloned().zip(probs.iter().map(|&p| round_prob(p))).collect();
    let argmax = probs.iter().enumerate().fold(0, |b, (i, &p)| if p > probs[b] { i } else { b });
    match q.qtype {
        QuestionType::Noul => Answer::Noul { noul: round_prob(probs[1]) },
        QuestionType::Choice => Answer::Choice { choice: keys[argmax].clone(), confidence: round_prob(choice_confidence(probs)), probabilities: dist },
        QuestionType::Score => Answer::Score {
            score: round_prob(probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()),
            confidence: round_prob(score_confidence(probs)),
            legend: keys.into_iter().zip(q.score_levels()).collect(),
            probabilities: dist,
        },
    }
}
```

Add `pub mod decisions;` near the top of `crates/cascadia-api/src/lib.rs`, and the two dependencies to `crates/cascadia-api/Cargo.toml` under `[dependencies]`:

```toml
indexmap = { version = "2", features = ["serde"] }
tokenizers = { workspace = true }

[target.'cfg(unix)'.dependencies]
libc = "0.2"
```

(`libc` provides `O_NOFOLLOW` for Task 8's `--decision-dump` open; it is already in the workspace lockfile through other crates.)

- [ ] **Step 5: Run tests**

Run: `cargo test -p cascadia-api decisions::schema`
Expected: PASS (8 tests).

- [ ] **Step 6: Commit**

```bash
git add crates/cascadia-api
git commit -m "feat(api): /v1/decisions request schema, validation and SystemOne answers"
```

---

### Task 4: Decision profile and service

**Files:**
- Create: `crates/cascadia-api/src/decisions/profile.rs`
- Modify: `crates/cascadia-api/src/decisions/mod.rs`

**Interfaces:**
- Consumes: `schema::{DecisionRequest, Limits}`, `DecisionError`.
- Produces:
  - `profile::{DecisionProfile, Family, JevProfile, KevProfile, ClefProfile}` and `DecisionProfile::load(path) -> Result<DecisionProfile, String>`
  - `decisions::Pass { readout: cascadia_types::DecisionReadout, parts: Vec<PassPart> }`
  - `decisions::PassPart { question_id: String, logits: std::ops::Range<usize> }`
  - `trait Codec: Send + Sync`:
    - `fn limits(&self) -> Limits`
    - `fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError>`
    - `fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64>`
  - `DecisionService { codec: Box<dyn Codec>, max_tokens: usize, max_total_tokens: usize, max_questions: usize }`, `DecisionService::load(model_dir: &Path) -> Result<Option<Arc<DecisionService>>, String>` (None when there is no `decision.json`)
  - `pub(crate) fn softmax(z: &[f64]) -> Vec<f64>`
  - `pub fn prefill_work(passes: &[Pass], prefix_reuse: bool) -> usize` (Task 8's budget): the shared prefix counts once only when `prefix_reuse` is true (single box with the prefix cache on); otherwise every pass counts in full
  - `DecisionService.prefix_reuse: bool` (set by the CLI in Task 8b; `false` by default and in pipeline mode)
  - `DecisionService::limits(&self) -> Limits`: the codec's limits with `max_questions` taken from `decision.json` (default 32). The handler validates against this, never `codec.limits()` directly.
  - `pub(crate) fn sanitize_user_text(text: &str) -> String`: Kev's `user_tokens` rewrite (`<|name|>` → `<¦name¦>`)
  - `pub(crate) struct Sanitizer` with `Sanitizer::from_tokenizer(&Tokenizer)` and `apply(&self, &str) -> String`: the pipe rule plus every other angle-bracket added token of the model's tokenizer (`<think>`, `</think>`, `<tool_call>`, …) rewritten to `<¦…¦>`. Every codec owns one and passes every caller-controlled string through it, so caller text can never become a control or added token.

`decision.json` format (written by Task 15):

```json
{"family": "jev", "max_tokens": 32768, "max_total_tokens": 131072, "max_questions": 32,
 "jev": {"verbalizers": ["false","true","0","1","2","3","4","5","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"],
         "verbalizer_ids": [3721, 1802, 15, 16, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
         "bias": [24 floats], "slots": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
         "temperature": {"noul": 1.0022, "choice": 0.9840, "score": 1.0122}}}
{"family": "kev", "max_tokens": 73728, "kev": {"temperature": 2.193649959389252}}
{"family": "clef", "max_tokens": 16384, "clef": {}}
```

- [ ] **Step 1: Write the failing tests** (bottom of `profile.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_each_family() {
        let d = std::env::temp_dir().join(format!("dprof-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("decision.json");
        std::fs::write(&p, r#"{"family":"kev","max_tokens":100,"max_questions":8,"kev":{"temperature":2.5}}"#).unwrap();
        let prof = DecisionProfile::load(&p).unwrap();
        assert_eq!(prof.family, Family::Kev);
        assert_eq!(prof.max_questions, Some(8));
        assert_eq!(prof.kev.unwrap().temperature, 2.5);
        std::fs::write(&p, r#"{"family":"jev","max_tokens":10,"jev":{"verbalizers":["false","true"],"verbalizer_ids":[1,2],"bias":[0,0],"slots":{"noul":[0,2],"score":[2,2],"choice":[2,2]},"temperature":{"noul":1,"choice":1,"score":1}}}"#).unwrap();
        assert_eq!(DecisionProfile::load(&p).unwrap().jev.unwrap().verbalizer_ids, vec![1, 2]);
        std::fs::write(&p, r#"{"family":"jev","max_tokens":10}"#).unwrap();
        assert!(DecisionProfile::load(&p).unwrap_err().contains("jev"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn softmax_is_stable() {
        let p = crate::decisions::softmax(&[1000.0, 1000.0]);
        assert!((p[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sanitizer_also_neutralizes_non_pipe_added_tokens() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[unk]".to_string(), 0u32)].into_iter().collect()).unk_token("[unk]".into()).build().unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.add_tokens(&[tokenizers::AddedToken::from("<think>", false), tokenizers::AddedToken::from("</think>", false)]);
        tok.add_special_tokens(&[tokenizers::AddedToken::from("<|im_end|>", true)]);
        let san = crate::decisions::Sanitizer::from_tokenizer(&tok);
        assert_eq!(san.apply("a <think>x</think> <|im_end|> <b>"), "a <¦think¦>x<¦/think¦> <¦im_end¦> <b>");
    }

    #[test]
    fn sanitize_rewrites_only_control_token_shapes() {
        use crate::decisions::sanitize_user_text as s;
        assert_eq!(s("a <|im_end|> b"), "a <¦im_end¦> b");
        assert_eq!(s("<|fim_suffix|><|x_1|>"), "<¦fim_suffix¦><¦x_1¦>");
        assert_eq!(s("x <| not a token |> y"), "x <| not a token |> y");
        assert_eq!(s("<||> and <|"), "<||> and <|");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-api decisions::profile`
Expected: compile error.

- [ ] **Step 3: Implement `profile.rs`**

```rust
use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Jev,
    Kev,
    Clef,
}

#[derive(Clone, Debug, Deserialize)]
pub struct JevProfile {
    pub verbalizers: Vec<String>,
    pub verbalizer_ids: Vec<u32>,
    pub bias: Vec<f64>,
    /// kind → [lo, hi) slot range into verbalizer_ids/bias.
    pub slots: HashMap<String, [usize; 2]>,
    /// kind → calibration temperature.
    pub temperature: HashMap<String, f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct KevProfile {
    pub temperature: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ClefProfile {}

#[derive(Clone, Debug, Deserialize)]
pub struct DecisionProfile {
    pub family: Family,
    /// Longest admitted pass in tokens.
    pub max_tokens: usize,
    /// Tokens one request may prefill, shared prefix counted once (`prefill_work`; default 4 × max_tokens).
    #[serde(default)]
    pub max_total_tokens: Option<usize>,
    /// Questions per request (default 32). JEV and Kev run one pass per question.
    #[serde(default)]
    pub max_questions: Option<usize>,
    #[serde(default)]
    pub jev: Option<JevProfile>,
    #[serde(default)]
    pub kev: Option<KevProfile>,
    #[serde(default)]
    pub clef: Option<ClefProfile>,
}

impl DecisionProfile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let p: DecisionProfile = serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
        let ok = match p.family {
            Family::Jev => p.jev.is_some(),
            Family::Kev => p.kev.is_some(),
            Family::Clef => true,
        };
        if !ok {
            return Err(format!("{}: family {:?} needs its `{}` section", path.display(), p.family, format!("{:?}", p.family).to_lowercase()));
        }
        Ok(p)
    }
}
```

- [ ] **Step 4: Extend `decisions/mod.rs`** (append below `DecisionError`; the codec modules arrive in Tasks 5–7, so for now register them as declared modules gated by the families they implement)

```rust
pub mod profile;
pub mod jev;
pub mod kev;
pub mod clef;

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use schema::{DecisionRequest, Limits};

/// One prefill-only engine pass and which slice of its logits answers which question.
#[derive(Clone, Debug)]
pub struct Pass {
    pub readout: cascadia_types::DecisionReadout,
    pub parts: Vec<PassPart>,
}

#[derive(Clone, Debug)]
pub struct PassPart {
    pub question_id: String,
    pub logits: Range<usize>,
}

pub trait Codec: Send + Sync {
    fn limits(&self) -> Limits;
    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError>;
    fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64>;
}

pub struct DecisionService {
    pub codec: Box<dyn Codec>,
    pub max_tokens: usize,
    pub max_total_tokens: usize,
    pub max_questions: usize,
    /// The engine can resume a shared prefix (single box, prefix cache on). Only then does the
    /// budget count Kev's state once; pipeline ranks re-prefill it per question.
    pub prefix_reuse: bool,
}

// `cascadia_api::Config` derives Debug, and `Box<dyn Codec>` cannot.
impl std::fmt::Debug for DecisionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionService").field("max_tokens", &self.max_tokens).finish_non_exhaustive()
    }
}

impl DecisionService {
    /// `Ok(None)` when the model directory has no `decision.json` (a plain chat model).
    pub fn load(model_dir: &Path) -> Result<Option<Arc<DecisionService>>, String> {
        let path = model_dir.join("decision.json");
        if !path.exists() {
            return Ok(None);
        }
        let prof = profile::DecisionProfile::load(&path)?;
        let tok = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("{}: tokenizer.json: {e}", model_dir.display()))?;
        let codec: Box<dyn Codec> = match prof.family {
            profile::Family::Jev => Box::new(jev::JevCodec::new(tok, prof.jev.clone().unwrap())?),
            profile::Family::Kev => Box::new(kev::KevCodec::new(tok, prof.kev.clone().unwrap())?),
            profile::Family::Clef => Box::new(clef::ClefCodec::new(tok, prof.max_tokens)?),
        };
        let max_total_tokens = prof.max_total_tokens.unwrap_or(prof.max_tokens * 4);
        let max_questions = prof.max_questions.unwrap_or(32);
        Ok(Some(Arc::new(DecisionService { codec, max_tokens: prof.max_tokens, max_total_tokens, max_questions, prefix_reuse: false })))
    }

    /// The codec's option limits with the deployment's question cap.
    pub fn limits(&self) -> Limits {
        Limits { max_questions: self.max_questions, ..self.codec.limits() }
    }
}

/// kev.model.user_tokens, shared by every codec: `<|name|>` (name = [A-Za-z0-9_]+) → `<¦name¦>`.
/// HF tokenizers match added/special tokens inside text even with add_special_tokens=false, so
/// without this, state or option text could forge `<|im_end|>`, `<|fim_suffix|>` etc. and steer
/// the decision. Template pieces are tokenized separately and keep their real special tokens.
pub(crate) fn sanitize_user_text(text: &str) -> String {
    let mut safe = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        safe.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find("|>") {
            Some(end) if end > 0 && after[..end].chars().all(|c| c.is_ascii_alphanumeric() || c == '_') => {
                safe.push_str("<¦");
                safe.push_str(&after[..end]);
                safe.push_str("¦>");
                rest = &after[end + 2..];
            }
            _ => {
                safe.push_str("<|");
                rest = after;
            }
        }
    }
    safe.push_str(rest);
    safe
}

/// Tokens a request prefills. With `prefix_reuse` (single box, prefix cache on) a prefix every pass
/// shares (`cache_prefix`) is prefilled once and resumed, so it counts once; without it (pipeline,
/// cache off) every pass is prefilled in full. This is an upper bound only when reuse is on: an
/// evicted or over-budget snapshot re-prefills, which the per-pass `max_tokens` cap still bounds.
pub fn prefill_work(passes: &[Pass], prefix_reuse: bool) -> usize {
    let full: usize = passes.iter().map(|p| p.readout.prompt_ids.len()).sum();
    if !prefix_reuse {
        return full;
    }
    let shared = passes.iter().filter_map(|p| p.readout.cache_prefix).max().unwrap_or(0) as usize;
    let tails: usize = passes
        .iter()
        .map(|p| p.readout.prompt_ids.len().saturating_sub(p.readout.cache_prefix.unwrap_or(0) as usize))
        .sum();
    (shared + tails).min(full)
}

/// Caller-text neutralizer bound to one tokenizer: the `<|name|>` rule plus every other
/// angle-bracket added token (Qwen registers `<think>`, `</think>`, `<tool_call>`, … as added tokens,
/// and HF tokenizers match those inside plain text too). Longest tokens are rewritten first.
pub(crate) struct Sanitizer {
    added: Vec<String>,
}

impl Sanitizer {
    pub(crate) fn from_tokenizer(tok: &tokenizers::Tokenizer) -> Self {
        let mut added: Vec<String> = tok
            .get_added_tokens_decoder()
            .values()
            .map(|t| t.content.clone())
            .filter(|c| c.len() > 2 && c.starts_with('<') && c.ends_with('>') && !(c.starts_with("<|") && c.ends_with("|>")))
            .collect();
        added.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        added.dedup();
        Self { added }
    }

    pub(crate) fn apply(&self, text: &str) -> String {
        let mut s = sanitize_user_text(text);
        for a in &self.added {
            if s.contains(a.as_str()) {
                s = s.replace(a.as_str(), &format!("<¦{}¦>", &a[1..a.len() - 1]));
            }
        }
        s
    }
}

pub(crate) fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}
```

Create the three codec files with a compile-only skeleton so this task builds. Each is replaced in full by its own task:

```rust
// crates/cascadia-api/src/decisions/jev.rs  (Task 5 replaces this file)
use super::{profile::JevProfile, schema::{DecisionRequest, Limits}, Codec, DecisionError, Pass, PassPart};
pub struct JevCodec;
impl JevCodec {
    pub fn new(_tok: tokenizers::Tokenizer, _p: JevProfile) -> Result<Self, String> { Err("jev codec not built yet".into()) }
}
impl Codec for JevCodec {
    fn limits(&self) -> Limits { Limits { max_questions: 0, max_options: 0, min_levels: 0, max_levels: 0 } }
    fn encode(&self, _r: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> { Err(DecisionError::NotServed) }
    fn probabilities(&self, _r: &DecisionRequest, _p: &PassPart, _l: &[f32]) -> Vec<f64> { Vec::new() }
}
```

Copy the same skeleton into `kev.rs` (struct `KevCodec`, `new(_tok, _p: super::profile::KevProfile)`) and `clef.rs` (struct `ClefCodec`, `new(_tok, _max_tokens: usize)`).

- [ ] **Step 5: Run tests**

Run: `cargo test -p cascadia-api decisions`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/cascadia-api/src/decisions
git commit -m "feat(api): decision profile, codec trait and DecisionService loader"
```

---

### Task 5: JEV codec

Port of AutoTrust's `vl/serve_decide.py` `s1_pass` (System 1, strategy `single`), which is JEV's own server. One pass per question; the prompt is plain text, tokenized **without** special tokens:

```
[kind] {kind}
[state] {state}
[question] {question}
[options]
{lines}
[decision]:
```

- **noul:** lines `false\ntrue`, slots `noul`.
- **choice:** lines `A) {name}: {desc}` in **request order**, slots `choice[0..n]`. This is serve_decide's order (`opts` as the caller gives them). The spike's `jev.py` sorted the names, so the 98.8% figure was measured on sorted prompts; Task 18 re-measures JEV-9B with request order before that number is quoted.
- **score:** lines `0\n…\n5`, slots `score`. JEV only knows a fixed 0–5 scale (serve_decide has no level count), so the 6-way distribution is mapped onto the caller's L levels by `rescale_score`: slot j sits at x = j·(L−1)/5 and splits its mass linearly between levels ⌊x⌋ and ⌈x⌉. L = 6 is the identity; the expected value scales exactly with the range. Task 18 measures this mapping against the full-precision reference.
- **Caller text** (state, instructions, criteria names/descriptions) passes through the codec's `Sanitizer` (Task 4) before it enters the prompt, so it cannot forge control or added tokens. Text without such shapes is unchanged, so normal prompts still match the reference byte for byte.
- **State rendering mirrors serve_decide `_parts`:**
  - a string is used verbatim;
  - an object goes through `json.dumps` (`py_json(false, false)`);
  - an **array** is concatenated element by element: a string element verbatim, a `{"type": "text", "text": …}` element as its text, anything else through `json.dumps`;
  - an image element (`{"image": …}` or `{"type": "image_url", …}`) is a 400, since this server is text-only.

The probability for slot `i` is `softmax_i((logit_i + bias_i) / T_kind)`. The state renders exactly as the reference's `json.dumps(state, ensure_ascii=False)`: a string verbatim, anything else via `OrderedValue::py_json(false, false)` (request key order, **spaced** `", "`/`": "` separators). Choice descriptions and instructions render the same way.

**Files:**
- Modify (replace skeleton): `crates/cascadia-api/src/decisions/jev.rs`

**Interfaces:**
- Consumes: `JevProfile`, `Codec`, `Pass`, `PassPart`, `softmax`, `schema::{Question, QuestionType}`.
- Produces: `JevCodec::new(tok, profile) -> Result<JevCodec, String>`; `JevCodec::prompt_text(&self, req, qid) -> String` (pub for tests); `pub fn rescale_score(p6: &[f64], levels: usize) -> Vec<f64>`.

- [ ] **Step 1: Write the failing tests** (bottom of `jev.rs`). The tests build a tiny WordLevel tokenizer so they need no model files.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::decisions::schema::DecisionRequest;

    fn tiny_tokenizer() -> tokenizers::Tokenizer {
        // whitespace word-level vocab: every word maps to its index; unknowns to 0
        let words = ["[unk]", "[kind]", "noul", "choice", "score", "[state]", "[question]", "[options]", "[decision]:", "false", "true"];
        let vocab: std::collections::HashMap<String, u32> = words.iter().enumerate().map(|(i, w)| (w.to_string(), i as u32)).collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder().vocab(vocab.into_iter().collect()).unk_token("[unk]".into()).build().unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::WhitespaceSplit));
        t
    }

    fn profile() -> JevProfile {
        let verbalizers: Vec<String> = ["false", "true", "0", "1", "2", "3", "4", "5", "A", "B", "C"].iter().map(|s| s.to_string()).collect();
        JevProfile {
            verbalizer_ids: (100..111).collect(),
            bias: vec![0.0; 11],
            verbalizers,
            slots: [("noul".to_string(), [0, 2]), ("score".to_string(), [2, 8]), ("choice".to_string(), [8, 11])].into_iter().collect(),
            temperature: [("noul".to_string(), 1.0), ("score".to_string(), 1.0), ("choice".to_string(), 2.0)].into_iter().collect(),
        }
    }

    // From text: `serde_json::json!` would sort {"b","a"} and hide an ordering bug.
    fn req() -> DecisionRequest {
        serde_json::from_str(r#"{"model": "m", "state": {"b": 1, "a": "x"}, "questions": {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "sev": {"type": "score", "instructions": "How bad?", "criteria": ["none", "minor", "major"]}}}"#).unwrap()
    }

    #[test]
    fn prompt_matches_reference_layout() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        assert_eq!(
            c.prompt_text(&req(), "team"),
            "[kind] choice\n[state] {\"b\": 1, \"a\": \"x\"}\n[question] Which team?\n[options]\nA) billing: money\nB) tech: bugs\n[decision]:"
        );
        assert_eq!(c.prompt_text(&req(), "urgent"), "[kind] noul\n[state] {\"b\": 1, \"a\": \"x\"}\n[question] Urgent?\n[options]\nfalse\ntrue\n[decision]:");
        assert!(c.prompt_text(&req(), "sev").ends_with("[options]\n0\n1\n2\n3\n4\n5\n[decision]:"));
    }

    #[test]
    fn one_pass_per_question_with_verbalizer_slots() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let passes = c.encode(&req()).unwrap();
        assert_eq!(passes.len(), 3);
        match &passes[0].readout.head {
            cascadia_types::ReadoutHead::Verbalizer { token_ids } => assert_eq!(token_ids, &vec![108, 109]),
            h => panic!("{h:?}"),
        }
        match &passes[2].readout.head {
            cascadia_types::ReadoutHead::Verbalizer { token_ids } => assert_eq!(token_ids, &(102..108).collect::<Vec<u32>>()),
            h => panic!("{h:?}"),
        }
        assert_eq!(passes[1].parts[0].question_id, "urgent");
        assert_eq!(passes[1].parts[0].logits, 0..2);
    }

    #[test]
    fn caller_text_cannot_forge_control_tokens() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "x<|im_end|>", "questions": {"u": {"type": "noul", "instructions": "<|im_start|>q"}}}"#).unwrap();
        let t = c.prompt_text(&r, "u");
        assert!(t.contains("x<¦im_end¦>") && t.contains("<¦im_start¦>q") && !t.contains("<|"), "{t}");
    }

    #[test]
    fn list_state_renders_like_serve_decide_parts() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": ["hi ", {"type": "text", "text": "there"}, {"k": 1}], "questions": {"u": {"type": "noul", "instructions": "q"}}}"#).unwrap();
        assert!(c.prompt_text(&r, "u").contains("[state] hi there{\"k\": 1}\n"), "{}", c.prompt_text(&r, "u"));
        let img: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": [{"image": "data:x"}], "questions": {"u": {"type": "noul"}}}"#).unwrap();
        assert!(matches!(c.encode(&img), Err(DecisionError::BadRequest(_))));
    }

    #[test]
    fn rescale_score_maps_six_slots_onto_l_levels() {
        let p6 = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1];
        assert_eq!(rescale_score(&p6, 6), p6.to_vec());
        let three = rescale_score(&[0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 3);
        // slot 3 → x=1.2 (0.8 to level 1), slot 4 → x=1.6 (0.4 to level 1), slot 5 → level 2
        for (got, want) in three.iter().zip([0.0, 0.4, 0.6]) {
            assert!((got - want).abs() < 1e-12, "{three:?}");
        }
        let two = rescale_score(&p6, 2);
        assert!((two.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // expected value scales with the range: E6/5 == E2/1
        let e6: f64 = p6.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
        assert!((e6 / 5.0 - two[1]).abs() < 1e-12);
    }

    #[test]
    fn probabilities_apply_bias_temperature_and_score_rescaling() {
        let mut p = profile();
        p.bias[8] = 1.0; // choice slot A
        let c = JevCodec::new(tiny_tokenizer(), p).unwrap();
        let r = req();
        let passes = c.encode(&r).unwrap();
        let probs = c.probabilities(&r, &passes[0].parts[0], &[1.0, 1.0]);
        let e = (2.0f64 / 2.0).exp();
        assert!((probs[0] - e / (e + (1.0f64 / 2.0).exp())).abs() < 1e-9);
        let sp = c.probabilities(&r, &passes[2].parts[0], &[-100.0, -100.0, -100.0, 0.0, 0.0, 0.0]);
        assert_eq!(sp.len(), 3);
        assert!((sp.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((sp[1] - 0.4).abs() < 1e-9 && (sp[2] - 0.6).abs() < 1e-9, "{sp:?}");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-api decisions::jev`
Expected: FAIL / compile error (skeleton).

- [ ] **Step 3: Implement** (replace `jev.rs` above the test module)

```rust
//! JEV codec (autotrust/JEV-9B, JEV-27B): ports `vl/serve_decide.py` System 1.
use super::profile::JevProfile;
use super::schema::{DecisionRequest, Limits, OrderedValue, Question, QuestionType};
use super::{softmax, Codec, DecisionError, Pass, PassPart, Sanitizer};
use cascadia_types::{DecisionReadout, ReadoutHead};

pub struct JevCodec {
    tok: tokenizers::Tokenizer,
    p: JevProfile,
    san: Sanitizer,
}

fn kind(q: &Question) -> &'static str {
    match q.qtype {
        QuestionType::Noul => "noul",
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
    }
}

/// The reference's `json.dumps(v, ensure_ascii=False)` for non-strings (spaced separators, request
/// order). Used for instructions and choice descriptions; the codec sanitizes the result.
fn render(v: &OrderedValue) -> String {
    match v {
        OrderedValue::String(s) => s.clone(),
        OrderedValue::Null => String::new(),
        other => other.py_json(false, false),
    }
}

fn is_image_part(v: &OrderedValue) -> bool {
    v.get("image").is_some() || v.get("type").and_then(|t| t.as_str()) == Some("image_url")
}

/// serve_decide `_parts`: string verbatim, object via json.dumps, array concatenated part by part.
fn render_state(v: &OrderedValue) -> String {
    match v {
        OrderedValue::String(s) => s.clone(),
        OrderedValue::Array(parts) => parts
            .iter()
            .map(|p| match p {
                OrderedValue::String(s) => s.clone(),
                p if p.get("type").and_then(|t| t.as_str()) == Some("text") => p.get("text").map(render).unwrap_or_default(),
                other => other.py_json(false, false),
            })
            .collect(),
        other => other.py_json(false, false),
    }
}

/// JEV's 0–5 score distribution mapped onto `levels` levels: slot j sits at x = j·(L−1)/5 and
/// splits its mass linearly between ⌊x⌋ and ⌈x⌉. Identity for L = 6.
pub fn rescale_score(p6: &[f64], levels: usize) -> Vec<f64> {
    let mut out = vec![0.0; levels];
    let top = (p6.len() - 1) as f64;
    for (j, &p) in p6.iter().enumerate() {
        let x = j as f64 * (levels - 1) as f64 / top;
        let lo = x.floor() as usize;
        let w = x - lo as f64;
        out[lo] += p * (1.0 - w);
        if w > 0.0 {
            out[lo + 1] += p * w;
        }
    }
    out
}

impl JevCodec {
    pub fn new(tok: tokenizers::Tokenizer, p: JevProfile) -> Result<Self, String> {
        for k in ["noul", "score", "choice"] {
            if !p.slots.contains_key(k) || !p.temperature.contains_key(k) {
                return Err(format!("decision.json jev section lacks slots/temperature for {k}"));
            }
        }
        if p.verbalizer_ids.len() != p.bias.len() || p.verbalizers.len() != p.bias.len() {
            return Err("decision.json jev: verbalizers, verbalizer_ids and bias must have equal length".into());
        }
        let san = Sanitizer::from_tokenizer(&tok);
        Ok(Self { tok, p, san })
    }

    fn slot(&self, k: &str) -> [usize; 2] {
        self.p.slots[k]
    }

    pub fn prompt_text(&self, req: &DecisionRequest, qid: &str) -> String {
        let q = &req.questions[qid];
        let lines: Vec<String> = match q.qtype {
            QuestionType::Noul => vec!["false".into(), "true".into()],
            QuestionType::Score => (0..6).map(|i| i.to_string()).collect(),
            QuestionType::Choice => {
                let [lo, _] = self.slot("choice");
                q.choice_criteria()
                    .iter()
                    .enumerate()
                    .map(|(i, (name, desc))| {
                        let label = &self.p.verbalizers[lo + i];
                        let name = self.san.apply(name);
                        let d = self.san.apply(&render(desc));
                        if d.is_empty() { format!("{label}) {name}") } else { format!("{label}) {name}: {d}") }
                    })
                    .collect()
            }
        };
        let question = self.san.apply(&q.instructions.as_ref().map(render).unwrap_or_else(|| qid.to_string()));
        format!(
            "[kind] {}\n[state] {}\n[question] {}\n[options]\n{}\n[decision]:",
            kind(q),
            self.san.apply(&render_state(&req.state)),
            question,
            lines.join("\n")
        )
    }

    fn slots_for(&self, q: &Question) -> (usize, usize) {
        let [lo, hi] = self.slot(kind(q));
        match q.qtype {
            QuestionType::Choice => (lo, lo + q.choice_criteria().len()),
            _ => (lo, hi),
        }
    }
}

impl Codec for JevCodec {
    fn limits(&self) -> Limits {
        let [clo, chi] = self.slot("choice");
        let [slo, shi] = self.slot("score");
        // >16 choices would need serve_decide's tournament (multi-pass); rejected with 400 for now
        // (see Open Questions). Score levels: any L, mapped from JEV's 0-5 scale (rescale_score).
        let _ = (slo, shi);
        Limits { max_questions: 32, max_options: chi - clo, min_levels: 2, max_levels: 256 }
    }

    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> {
        if let OrderedValue::Array(parts) = &req.state {
            if parts.iter().any(is_image_part) {
                return Err(DecisionError::BadRequest("image parts in state are not supported (text-only server)".into()));
            }
        }
        let mut out = Vec::with_capacity(req.questions.len());
        for (qid, q) in &req.questions {
            let text = self.prompt_text(req, qid);
            let enc = self.tok.encode(text.as_str(), false).map_err(|e| DecisionError::BadRequest(format!("tokenize: {e}")))?;
            let (lo, hi) = self.slots_for(q);
            let token_ids = self.p.verbalizer_ids[lo..hi].to_vec();
            let n = token_ids.len();
            out.push(Pass {
                readout: DecisionReadout { prompt_ids: enc.get_ids().to_vec(), head: ReadoutHead::Verbalizer { token_ids }, cache_prefix: None },
                parts: vec![PassPart { question_id: qid.clone(), logits: 0..n }],
            });
        }
        Ok(out)
    }

    fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64> {
        let q = &req.questions[&part.question_id];
        let (lo, _) = self.slots_for(q);
        let t = self.p.temperature[kind(q)];
        let z: Vec<f64> = logits.iter().enumerate().map(|(i, &l)| (l as f64 + self.p.bias[lo + i]) / t).collect();
        let p = softmax(&z);
        if q.qtype == QuestionType::Score {
            return rescale_score(&p, q.score_levels().len());
        }
        p
    }
}
```

The test's `state` is `{"b": 1, "a": "x"}` and must render in request order with spaced separators. That comes from `OrderedValue` (Task 3). **Do not** enable serde_json's `preserve_order` feature: it unifies across the workspace and would reorder chat responses.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-api decisions::jev`
Expected: PASS (6 tests). Then `cargo test -p cascadia-api` (the whole crate stays green) and `cargo tree -p cascadia-api -e features -i serde_json | grep -c preserve_order`, which must print `0`.

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-api
git commit -m "feat(api): JEV decision codec (verbalizer head, calibrated per-kind softmax)"
```

---

### Task 6: Kev codec

Port of `experiments/kev-spike/kev.py` (itself from `jaredpalmer/kev` `kev/api.py` + `kev/model.py`). One pass per question:

```
[<|fim_prefix|> state] [<|fim_middle|> instr] (<|box_start|> option <|box_end|>)* <|fim_suffix|>
```

The rules:
- **Caller text never becomes a control token:** `<|name|>` is rewritten to `<¦name¦>` before tokenizing.
- **Options per type:** `noul` = `["no", "yes"]`, with criteria descriptions appended as `name: desc`; `choice` = `name` or `name: desc`; `score` = rendered level descriptions.
- **Head positions:** `decide_pos` is the `<|fim_suffix|>` index, and `option_pos[i]` is the index of option i's `<|box_end|>`.
- **Probabilities:** `softmax(z / T)`.
- **State reuse:** every pass of a request starts with the same `[<|fim_prefix|> state]` ids, so the readout sets `cache_prefix = state.len()`. The engine snapshots its prefix cache there on the first question and resumes from it for the rest, which is how Kev's own `model.py` serves (`probs_with_prefix`: prefill the state once, branch per question). Positions are unchanged, and every row the pointer head reads lies after the prefix.

Kev's `render`:
- a string, number or bool uses Python `str()` (`OrderedValue::py_str`): `True`/`False` capitalized;
- a list becomes `- item` lines;
- an object becomes `key: value` lines in **request order**, nested values indented under `key:`;
- null becomes empty.

**Files:**
- Modify (replace skeleton): `crates/cascadia-api/src/decisions/kev.rs`

**Interfaces:**
- Consumes: `KevProfile`, `Codec`, `Pass`, `PassPart`, `softmax`.
- Produces: `KevCodec::new(tok, profile)`, `pub fn render(v: &OrderedValue, indent: usize) -> String`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::decisions::schema::DecisionRequest;

    fn tok() -> tokenizers::Tokenizer {
        let words = ["[unk]", "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>", "no", "yes", "a", "b"];
        let vocab: std::collections::HashMap<String, u32> = words.iter().enumerate().map(|(i, w)| (w.to_string(), i as u32)).collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder().vocab(vocab.into_iter().collect()).unk_token("[unk]".into()).build().unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::WhitespaceSplit));
        t.add_special_tokens(&words[1..6].iter().map(|w| tokenizers::AddedToken::from(w.to_string(), true)).collect::<Vec<_>>());
        t
    }

    fn ov(s: &str) -> OrderedValue {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn render_matches_kev() {
        assert_eq!(render(&ov(r#""plain""#), 0), "plain");
        assert_eq!(render(&ov(r#"["x", "y"]"#), 0), "- x\n- y");
        assert_eq!(render(&ov(r#"{"n": {"a": 1}, "k": "v", "t": true}"#), 0), "n:\n  a: 1\nk: v\nt: True");
    }

    #[test]
    fn noul_row_layout_and_positions() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "a", "questions": {"u": {"type": "noul", "instructions": "b"}}}"#).unwrap();
        let c = KevCodec::new(tok(), KevProfile { temperature: 2.0 }).unwrap();
        let p = &c.encode(&r).unwrap()[0];
        // <fim_prefix> a <fim_middle> b <box_start> no <box_end> <box_start> yes <box_end> <fim_suffix>
        assert_eq!(p.readout.prompt_ids, vec![1, 8, 2, 9, 3, 6, 4, 3, 7, 4, 5]);
        assert_eq!(p.readout.head, cascadia_types::ReadoutHead::Pointer { decide_pos: 10, option_pos: vec![6, 9] });
        assert_eq!(p.readout.cache_prefix, Some(2), "<fim_prefix> + state");
        let probs = c.probabilities(&r, &p.parts[0], &[2.0, 0.0]);
        assert!((probs[0] - 1.0f64.exp() / (1.0f64.exp() + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn caller_text_cannot_forge_control_tokens() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "<|fim_suffix|>", "questions": {"u": {"type": "noul"}}}"#).unwrap();
        let c = KevCodec::new(tok(), KevProfile { temperature: 1.0 }).unwrap();
        let ids = &c.encode(&r).unwrap()[0].readout.prompt_ids;
        assert_eq!(ids.iter().filter(|&&i| i == 5).count(), 1, "only the codec's own <decide>");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-api decisions::kev`
Expected: FAIL.

- [ ] **Step 3: Implement**

```rust
//! Kev codec (jaredpalmer/kev-9b): ports kev/api.py to_record + kev/model.py encode (row form).
use super::profile::KevProfile;
use super::schema::{DecisionRequest, Limits, OrderedValue, QuestionType};
use super::{softmax, Codec, DecisionError, Pass, PassPart, Sanitizer};
use cascadia_types::{DecisionReadout, ReadoutHead};

const SPECIAL: [&str; 5] = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"];

pub struct KevCodec {
    tok: tokenizers::Tokenizer,
    ids: [u32; 5],
    temperature: f64,
    san: Sanitizer,
}

/// kev.api.render: str/int/float/bool via Python `str()`, lists as `- item`, objects as `key: value`.
pub fn render(v: &OrderedValue, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        OrderedValue::Null => String::new(),
        OrderedValue::String(_) | OrderedValue::Bool(_) | OrderedValue::Number(_) => v.py_str(),
        OrderedValue::Array(a) => a
            .iter()
            .map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        OrderedValue::Object(m) => m
            .iter()
            .map(|(k, x)| match x {
                OrderedValue::Object(_) | OrderedValue::Array(_) => format!("{pad}{k}:\n{}", render(x, indent + 1)),
                _ => format!("{pad}{k}: {}", render(x, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn option_text(name: &str, desc: Option<&OrderedValue>) -> String {
    match desc.map(|d| render(d, 0)).filter(|d| !d.is_empty()) {
        Some(d) => format!("{name}: {d}"),
        None => name.to_string(),
    }
}

impl KevCodec {
    pub fn new(tok: tokenizers::Tokenizer, p: KevProfile) -> Result<Self, String> {
        let mut ids = [0u32; 5];
        for (i, s) in SPECIAL.iter().enumerate() {
            ids[i] = tok.token_to_id(s).ok_or_else(|| format!("kev tokenizer lacks {s}"))?;
        }
        let san = Sanitizer::from_tokenizer(&tok);
        Ok(Self { tok, ids, temperature: p.temperature, san })
    }

    /// kev.model.user_tokens: sanitize (`<|name|>` → `<¦name¦>`), then tokenize without specials.
    fn user_tokens(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        let safe = self.san.apply(text);
        let enc = self.tok.encode(safe.as_str(), false).map_err(|e| DecisionError::BadRequest(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }
}

impl Codec for KevCodec {
    fn limits(&self) -> Limits {
        Limits { max_questions: 32, max_options: 256, min_levels: 2, max_levels: 256 }
    }

    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> {
        let [s_id, q_id, o_id, c_id, d_id] = self.ids;
        let mut state = vec![s_id];
        state.extend(self.user_tokens(&render(&req.state, 0))?);
        let mut out = Vec::new();
        for (qid, q) in &req.questions {
            let opts: Vec<String> = match q.qtype {
                QuestionType::Noul => {
                    let c = q.criteria.as_ref();
                    vec![option_text("no", c.and_then(|m| m.get("false"))), option_text("yes", c.and_then(|m| m.get("true")))]
                }
                QuestionType::Choice => q.choice_criteria().iter().map(|(k, v)| option_text(k, Some(v))).collect(),
                QuestionType::Score => q.score_levels().iter().map(|x| render(x, 0)).collect(),
            };
            let mut row = state.clone();
            row.push(q_id);
            row.extend(self.user_tokens(&q.instructions.as_ref().map(|v| render(v, 0)).unwrap_or_default())?);
            let mut ends = Vec::with_capacity(opts.len());
            for o in &opts {
                row.push(o_id);
                row.extend(self.user_tokens(o)?);
                row.push(c_id);
                ends.push((row.len() - 1) as u32);
            }
            row.push(d_id);
            let decide_pos = (row.len() - 1) as u32;
            let n = ends.len();
            out.push(Pass {
                readout: DecisionReadout { prompt_ids: row, head: ReadoutHead::Pointer { decide_pos, option_pos: ends }, cache_prefix: Some(state.len() as u32) },
                parts: vec![PassPart { question_id: qid.clone(), logits: 0..n }],
            });
        }
        Ok(out)
    }

    fn probabilities(&self, _req: &DecisionRequest, _part: &PassPart, logits: &[f32]) -> Vec<f64> {
        softmax(&logits.iter().map(|&z| z as f64 / self.temperature).collect::<Vec<_>>())
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-api decisions::kev`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-api/src/decisions/kev.rs
git commit -m "feat(api): Kev decision codec (pointer head, unforgeable delimiters)"
```

---

### Task 7: Clef codec

Port of `encode_record` from `experiments/clef-spike/flash/joint_schema_model.py` (Cloudflare, Apache-2.0). There is **one pass per request**, covering all questions. The layout, with spans recorded relative to the schema and shifted after the state is placed:
1. Prefix: `<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n`
2. The state.
3. `\n\nSCHEMA FIELDS:\n`, then per field: `\nFIELD {i}\nID: {id}\nTYPE: {type}\nINSTRUCTION: ` + instructions + `\nALLOWED OPTIONS:\n` + per option (`OPTION {k}: ` + `{"description": ..., "option_id": ...}` + `\n`) + `END FIELD\n`
4. Suffix: `\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:`

Rules that must match the reference:
- **Rendering:** `render(x)` = the string itself, or `json.dumps(x, ensure_ascii=False, separators=(",",":"), sort_keys=True)`.
- **Option order:** `noul` = `true`, `false` (default descriptions overridable by criteria); `choice` = criteria keys **sorted**; `score` = index order.
- **Tokenization:** each piece is tokenized separately with `add_special_tokens=False`, and the ids are concatenated.
- **Truncation:** the state is cut to `max_length - (prefix + schema + suffix)` tokens.

The engine returns one logit per option in question order; probabilities are a softmax per question. Engine order differs from the API's `option_keys` order in two places, and the codec maps both back before `answer_from_probs`:
- **noul:** Clef orders the options `true, false` (`question_options`), while `option_keys` is `false, true` and `answer_from_probs` reads `probs[1]` as p(true). The codec returns `[p_false, p_true]`. (Without this every Clef noul answer is inverted, and parity would not notice: see Task 16's golden check.)
- **choice:** Clef sorts the keys; the codec returns probabilities in the request's criteria order.

All caller-controlled text (question ids, state, instructions, option ids and descriptions) passes through the codec's `Sanitizer` after rendering, so it cannot forge `<|im_end|>`/`<|im_start|>`. The fixed template pieces are tokenized separately and keep their real special tokens.

**Files:**
- Modify (replace skeleton): `crates/cascadia-api/src/decisions/clef.rs`

**Interfaces:**
- Consumes: `Codec`, `Pass`, `PassPart`, `softmax`, `cascadia_types::{ReadoutHead::Joint, JointQuestion, JointOption}`.
- Produces: `ClefCodec::new(tok, max_tokens)`; `pub fn render_sorted(v: &OrderedValue) -> String`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::decisions::schema::DecisionRequest;

    fn tok() -> tokenizers::Tokenizer {
        // byte-level-free char tokenizer: one token per char keeps span arithmetic checkable
        let mut vocab: std::collections::HashMap<String, u32> = (32u8..127).map(|b| ((b as char).to_string(), b as u32)).collect();
        vocab.insert("\n".into(), 10);
        vocab.insert("[unk]".into(), 0);
        let model = tokenizers::models::wordlevel::WordLevel::builder().vocab(vocab.into_iter().collect()).unk_token("[unk]".into()).build().unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::split::Split::new(tokenizers::pre_tokenizers::split::SplitPattern::Regex("".into()), tokenizers::SplitDelimiterBehavior::Isolated, false).unwrap()));
        t
    }

    #[test]
    fn render_sorted_matches_python_json_dumps() {
        let v: OrderedValue = serde_json::from_str(r#"{"b": 1, "a": "é"}"#).unwrap();
        assert_eq!(render_sorted(&v), "{\"a\":\"é\",\"b\":1}");
        assert_eq!(render_sorted(&OrderedValue::String("x".into())), "x");
    }

    #[test]
    fn spans_point_at_instruction_and_option_text() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "S", "questions": {
            "c": {"type": "choice", "instructions": "Q", "criteria": {"z": "last", "a": "first"}}}}"#).unwrap();
        let c = ClefCodec::new(tok(), 16384).unwrap();
        let p = &c.encode(&r).unwrap();
        assert_eq!(p.len(), 1);
        let ids = &p[0].readout.prompt_ids;
        let text: String = ids.iter().map(|&i| char::from_u32(i).unwrap()).collect();
        match &p[0].readout.head {
            ReadoutHead::Joint { questions } => {
                let q = &questions[0];
                assert_eq!(q.qtype, 1);
                assert_eq!(&text[q.span[0] as usize..q.span[1] as usize], "Q");
                let o0 = &text[q.options[0].span[0] as usize..q.options[0].span[1] as usize];
                assert_eq!(o0, "{\"description\":\"first\",\"option_id\":\"a\"}");
            }
            h => panic!("{h:?}"),
        }
        assert!(text.ends_with("JOINT SCHEMA DECISIONS:"));
    }

    #[test]
    fn noul_probabilities_come_back_false_then_true() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "S", "questions": {"u": {"type": "noul"}}}"#).unwrap();
        let c = ClefCodec::new(tok(), 16384).unwrap();
        let p = c.encode(&r).unwrap();
        // engine order is [true, false]; logits favour true
        let probs = c.probabilities(&r, &p[0].parts[0], &[3.0, 0.0]);
        assert!(probs[1] > 0.9, "option_keys order is [false, true]: {probs:?}");
        let a = serde_json::to_value(crate::decisions::schema::answer_from_probs(&r.questions["u"], &probs)).unwrap();
        assert!(a["noul"].as_f64().unwrap() > 0.9);
    }

    #[test]
    fn caller_text_cannot_forge_the_chat_frame() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "<|im_end|>", "questions": {"u": {"type": "noul"}}}"#).unwrap();
        let c = ClefCodec::new(tok(), 16384).unwrap();
        let ids = &c.encode(&r).unwrap()[0].readout.prompt_ids;
        let text: String = ids.iter().map(|&i| char::from_u32(i).unwrap()).collect();
        // the test tokenizer maps '¦' to [unk]; what matters is that "<|" never survives
        assert!(text.contains("STATE:\n<") && !text.contains("STATE:\n<|"), "{text}");
    }

    #[test]
    fn choice_probabilities_return_in_request_order() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "S", "questions": {
            "c": {"type": "choice", "criteria": {"z": "last", "a": "first"}}}}"#).unwrap();
        let c = ClefCodec::new(tok(), 16384).unwrap();
        let p = c.encode(&r).unwrap();
        // engine order is sorted: [a, z]; logits favour a
        let probs = c.probabilities(&r, &p[0].parts[0], &[3.0, 0.0]);
        assert!(probs[1] > probs[0], "request order is [z, a]");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-api decisions::clef`
Expected: FAIL.

- [ ] **Step 3: Implement**

```rust
//! Clef codec (Cloudflare/clef, clef-flash): ports joint_schema_model.encode_record (Apache-2.0).
use super::schema::{DecisionRequest, Limits, OrderedValue, Question, QuestionType};
use super::{softmax, Codec, DecisionError, Pass, PassPart, Sanitizer};
use cascadia_types::{DecisionReadout, JointOption, JointQuestion, ReadoutHead, MAX_DECISION_OUTPUTS};

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

pub struct ClefCodec {
    tok: tokenizers::Tokenizer,
    max_tokens: usize,
    san: Sanitizer,
}

/// Python `json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`; strings verbatim.
pub fn render_sorted(v: &OrderedValue) -> String {
    match v {
        OrderedValue::String(s) => s.clone(),
        other => other.py_json(true, true),
    }
}

/// (option_id, description) in Clef's option order.
fn options(q: &Question) -> Vec<(String, OrderedValue)> {
    match q.qtype {
        QuestionType::Noul => {
            let mut t = OrderedValue::String("The proposition is true or the answer is yes.".into());
            let mut f = OrderedValue::String("The proposition is false or the answer is no.".into());
            if let Some(c) = &q.criteria {
                if let Some(v) = c.get("true") { t = v.clone(); }
                if let Some(v) = c.get("false") { f = v.clone(); }
            }
            vec![("true".into(), t), ("false".into(), f)]
        }
        QuestionType::Choice => {
            let mut c = q.choice_criteria();
            c.sort_by(|a, b| a.0.cmp(&b.0));
            c
        }
        QuestionType::Score => q.score_levels().into_iter().enumerate().map(|(i, v)| (i.to_string(), v)).collect(),
    }
}

fn qtype_code(q: &Question) -> u8 {
    match q.qtype {
        QuestionType::Noul => 0,
        QuestionType::Choice => 1,
        QuestionType::Score => 2,
    }
}

fn qtype_str(q: &Question) -> &'static str {
    match q.qtype {
        QuestionType::Noul => "noul",
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
    }
}

impl ClefCodec {
    pub fn new(tok: tokenizers::Tokenizer, max_tokens: usize) -> Result<Self, String> {
        let san = Sanitizer::from_tokenizer(&tok);
        Ok(Self { tok, max_tokens, san })
    }

    fn ids(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        Ok(self.tok.encode(text, false).map_err(|e| DecisionError::BadRequest(format!("tokenize: {e}")))?.get_ids().to_vec())
    }
}

impl Codec for ClefCodec {
    fn limits(&self) -> Limits {
        Limits { max_questions: 32, max_options: 256, min_levels: 2, max_levels: 256 }
    }

    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> {
        // One joint pass carries every option of every question; the engine caps a readout at
        // MAX_DECISION_OUTPUTS logits, so reject here with 400 instead of an opaque engine 503.
        let total: usize = req.questions.values().map(|q| options(q).len()).sum();
        if total > MAX_DECISION_OUTPUTS {
            return Err(DecisionError::BadRequest(format!("{total} options across all questions; the limit is {MAX_DECISION_OUTPUTS}")));
        }
        let mut schema = self.ids("\n\nSCHEMA FIELDS:\n")?;
        let mut questions = Vec::new();
        for (i, (qid, q)) in req.questions.iter().enumerate() {
            schema.extend(self.ids(&format!("\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ", i + 1, self.san.apply(qid), qtype_str(q)))?);
            let qs = schema.len();
            let instr = match &q.instructions {
                Some(v) if !(v.is_null() || v.as_str() == Some("")) => render_sorted(v),
                _ => qid.clone(),
            };
            schema.extend(self.ids(&self.san.apply(&instr))?);
            let qe = schema.len();
            schema.extend(self.ids("\nALLOWED OPTIONS:\n")?);
            let mut opts = Vec::new();
            for (k, (oid, desc)) in options(q).into_iter().enumerate() {
                schema.extend(self.ids(&format!("OPTION {}: ", k + 1))?);
                let os = schema.len();
                let mut sem = vec![("option_id".to_string(), OrderedValue::String(oid))];
                if !desc.is_null() {
                    sem.push(("description".to_string(), desc));
                }
                schema.extend(self.ids(&self.san.apply(&render_sorted(&OrderedValue::Object(sem))))?);
                opts.push((os as u32, schema.len() as u32));
                schema.extend(self.ids("\n")?);
            }
            schema.extend(self.ids("END FIELD\n")?);
            questions.push((qid.clone(), qtype_code(q), [qs as u32, qe as u32], opts));
        }
        let prefix = self.ids(&format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"))?;
        let suffix = self.ids("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        if fixed > self.max_tokens {
            return Err(DecisionError::TooLong(format!("schema needs {fixed} tokens before the state; the limit is {}", self.max_tokens)));
        }
        let mut state = self.ids(&self.san.apply(&render_sorted(&req.state)))?;
        state.truncate(self.max_tokens - fixed);
        let shift = (prefix.len() + state.len()) as u32;
        let mut prompt_ids = prefix;
        prompt_ids.extend(state);
        prompt_ids.extend(schema);
        prompt_ids.extend(suffix);
        let mut parts = Vec::new();
        let mut cursor = 0usize;
        let joint: Vec<JointQuestion> = questions
            .into_iter()
            .map(|(qid, qtype, span, opts)| {
                let n = opts.len();
                parts.push(PassPart { question_id: qid, logits: cursor..cursor + n });
                cursor += n;
                JointQuestion {
                    qtype,
                    span: [span[0] + shift, span[1] + shift],
                    options: opts.into_iter().map(|(a, b)| JointOption { span: [a + shift, b + shift] }).collect(),
                }
            })
            .collect();
        Ok(vec![Pass { readout: DecisionReadout { prompt_ids, head: ReadoutHead::Joint { questions: joint }, cache_prefix: None }, parts }])
    }

    fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64> {
        let q = &req.questions[&part.question_id];
        let p = softmax(&logits.iter().map(|&z| z as f64).collect::<Vec<_>>());
        match q.qtype {
            // engine order [true, false] → option_keys order [false, true]
            QuestionType::Noul => return vec![p[1], p[0]],
            QuestionType::Score => return p,
            QuestionType::Choice => {}
        }
        // engine order = sorted keys; report in request order
        let mut sorted: Vec<String> = q.choice_criteria().into_iter().map(|(k, _)| k).collect();
        sorted.sort();
        q.option_keys().iter().map(|k| p[sorted.iter().position(|s| s == k).unwrap()]).collect()
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-api decisions::clef`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-api/src/decisions/clef.rs
git commit -m "feat(api): Clef decision codec (joint head spans, encode_record port)"
```

---

### Task 8: Shared decision core and `POST /v1/systemone` (API crate)

**Files:**
- Create: `crates/cascadia-api/src/decisions/handler.rs`
- Modify:
  - `crates/cascadia-api/src/decisions/mod.rs`: `pub mod handler;`
  - `crates/cascadia-api/src/lib.rs`:
    - `AppState.decisions: Option<Arc<decisions::DecisionService>>` and `AppState.decision_dump: Option<Arc<std::sync::Mutex<std::fs::File>>>`
    - `Config.decisions` (same type, default `None`) and `Config.decision_dump: Option<PathBuf>` (default `None`), copied/opened in `make_router_with_stats`
    - route `.route("/v1/systemone", post(decisions::handler::decide))` (Task 8c adds `/v1/decisions`)

**Interfaces:**
- Consumes: `DecisionService`, `Codec`, `schema::*`, `answer_from_probs`, `state.runner.generate_async`, `state.permits`.
- Produces: HTTP behaviour:
  - 200 with a `DecisionResponse`
  - 400 on validation errors
  - 413 when the state is more than `max_total_tokens × 8` bytes (checked before tokenizing), a pass exceeds `max_tokens`, the request's prefill work exceeds `max_total_tokens`, or the engine says the prompt is too long. Prefill work (`prefill_work`) counts a prefix shared by all passes (`cache_prefix`, Kev's state) once **only when** `svc.prefix_reuse` is true (single box with the prefix cache on): `max(cache_prefix) + Σ(len − cache_prefix)`. A Kev request with a 64k-token state and 5 questions then costs ~64k + 5 branches. In pipeline mode or with the cache off, every pass counts in full, because the engine re-prefills the state per question.
  - 501 when the model has no `decision.json`, or the engine reports `decision_unsupported`
  - 503 on any other engine error, and at capacity
  - Client-facing error bodies are fixed strings per category. The engine's or tokenizer's text (paths, OpenVINO messages) is logged with `warn!` and never returned.
- Produces: `Config.decision_dump` (the CLI flag `--decision-dump FILE` arrives in Task 8b; startup only, not an env var). When set, the handler appends one JSON line per pass, `{"parts": [[qid, start, end], ...], "readout": DecisionReadout}`, to FILE, which is opened once with mode 0600 (and tightened to 0600 if it already exists; symlinks refused). It records request-derived token ids, so the CLI logs a startup warning. Task 17's reference and Task 16's golden check read this file.

- [ ] **Step 1: Write the failing tests.** Add them to the `#[cfg(test)] mod tests` in `crates/cascadia-api/src/lib.rs`, next to `make_app`. They use a Kev profile with the tiny tokenizer, because Kev's codec has no slot limits.

```rust
    async fn make_decision_app() -> Router {
        make_decision_app_full(1000, 4000, None).await
    }

    async fn make_decision_app_with(max_tokens: usize, max_total_tokens: usize) -> Router {
        make_decision_app_full(max_tokens, max_total_tokens, None).await
    }

    async fn make_decision_app_full(max_tokens: usize, max_total_tokens: usize, dump: Option<std::path::PathBuf>) -> Router {
        let mut runner = Runner::new(Box::new(MockBuilder::new()));
        runner.start(PeerLayout::single_stage(), ShardSpec::single_stage("mock-model", "CPU")).await.unwrap();
        let words = ["[unk]", "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"];
        let vocab: std::collections::HashMap<String, u32> = words.iter().enumerate().map(|(i, w)| (w.to_string(), i as u32)).collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder().vocab(vocab.into_iter().collect()).unk_token("[unk]".into()).build().unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::WhitespaceSplit));
        tok.add_special_tokens(&words[1..].iter().map(|w| tokenizers::AddedToken::from(w.to_string(), true)).collect::<Vec<_>>());
        let codec = decisions::kev::KevCodec::new(tok, decisions::profile::KevProfile { temperature: 1.0 }).unwrap();
        let mut cfg = Config::default();
        cfg.decisions = Some(Arc::new(decisions::DecisionService { codec: Box::new(codec), max_tokens, max_total_tokens, max_questions: 32, prefix_reuse: true }));
        cfg.decision_dump = dump;
        cfg.max_body_bytes = 1 << 20;
        make_router_with_config(Arc::new(runner), "mock-model", cfg)
    }

    /// Same as `post_decisions`; named for tests whose body is large (the router's body limit is
    /// `Config::default().max_body_bytes`, raise it in `make_decision_app_full` if it is below the body).
    async fn post_decisions_sized(app: Router, payload: &str) -> (StatusCode, Value, String) {
        post_decisions(app, payload).await
    }

    /// Raw JSON text: `serde_json::json!` sorts object keys, which would hide question-order bugs.
    async fn post_decisions(app: Router, payload: &str) -> (StatusCode, Value, String) {
        let response = app
            .oneshot(Request::builder().method("POST").uri("/v1/systemone").header("content-type", "application/json").body(Body::from(payload.to_owned())).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 65536).await.unwrap();
        let text = String::from_utf8_lossy(&body).into_owned();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null), text)
    }

    #[tokio::test]
    async fn decisions_answer_every_question_in_order() {
        let (status, v, body) = post_decisions(make_decision_app().await, r#"{
            "model": "mock-model", "state": "a b",
            "questions": {
                "team": {"type": "choice", "instructions": "who", "criteria": {"billing": "x", "tech": "y"}},
                "urgent": {"type": "noul"},
                "sev": {"type": "score", "criteria": ["lo", "mid", "hi"]}}}"#).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        // Order is checked on the RAW body: parsing into serde_json::Value sorts keys.
        let (t, u, s) = (body.find("\"team\"").unwrap(), body.find("\"urgent\"").unwrap(), body.find("\"sev\"").unwrap());
        assert!(t < u && u < s, "answers out of request order: {body}");
        // mock logits [0, -1, ...] ⇒ first option wins
        assert_eq!(v["answers"]["team"]["choice"], "billing");
        assert!(v["answers"]["urgent"]["noul"].as_f64().unwrap() < 0.5);
        assert_eq!(v["usage"]["output_tokens"], 0);
        assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
        assert_eq!(v["model"], "mock-model");
    }

    #[tokio::test]
    async fn decisions_on_a_chat_model_is_501() {
        let (status, _, _) = post_decisions(make_app().await, r#"{"model": "m", "state": "s", "questions": {"u": {"type": "noul"}}}"#).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn decisions_validation_is_400() {
        let (status, v, _) = post_decisions(make_decision_app().await, r#"{"model": "m", "state": "s", "questions": {}}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    }

    #[tokio::test]
    async fn decisions_over_the_total_token_budget_is_413() {
        // state prefix counted once + 32 branches of ~8 tokens > max_total_tokens = 40
        let qs: Vec<String> = (0..32).map(|i| format!(r#""q{i}": {{"type": "noul"}}"#)).collect();
        let body = format!(r#"{{"model": "m", "state": "a b", "questions": {{{}}}}}"#, qs.join(","));
        let app = make_decision_app_with(1000, 40).await;
        let (status, v, _) = post_decisions(app, &body).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
    }

    #[test]
    fn prefill_work_counts_the_shared_prefix_once() {
        let pass = |len: u32, cp: Option<u32>| decisions::Pass {
            readout: cascadia_types::DecisionReadout { prompt_ids: (0..len).collect(), head: cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] }, cache_prefix: cp },
            parts: vec![],
        };
        // Kev: 5 questions over a 1000-token state, 20-token branches
        let kev: Vec<_> = (0..5).map(|_| pass(1020, Some(1000))).collect();
        assert_eq!(decisions::prefill_work(&kev, true), 1000 + 5 * 20);
        // no reuse (pipeline / cache off): the state is re-prefilled per question
        assert_eq!(decisions::prefill_work(&kev, false), 5 * 1020);
        // JEV/Clef: no shared prefix, every pass counts in full
        assert_eq!(decisions::prefill_work(&[pass(300, None), pass(200, None)], true), 500);
    }

    #[tokio::test]
    async fn decisions_dump_writes_one_line_per_pass() {
        let path = std::env::temp_dir().join(format!("decision-dump-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        // a pre-existing world-readable file must be tightened to 0600 at startup
        std::fs::write(&path, "").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        let app = make_decision_app_full(1000, 4000, Some(path.clone())).await;
        let (status, v, _) = post_decisions(app, r#"{"model": "m", "state": "a", "questions": {"u": {"type": "noul"}, "w": {"type": "noul"}}}"#).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 2, "pre-existing empty file plus 2 appended lines");
        assert_eq!(lines[0]["parts"][0][0], "u");
        assert!(lines[0]["readout"]["prompt_ids"].as_array().unwrap().len() > 2);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn decisions_oversized_state_is_rejected_before_tokenizing() {
        let big = "a ".repeat(4000 * 8);
        let body = format!(r#"{{"model": "m", "state": "{big}", "questions": {{"u": {{"type": "noul"}}}}}}"#);
        let (status, _, _) = post_decisions_sized(make_decision_app_with(1000, 4000).await, &body).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn decisions_engine_error_text_is_not_returned() {
        // the mock fails a task whose prompt_ids contain the sentinel id 999_999 (Task 2)
        let (status, v, body) = post_decisions(make_decision_app().await, r#"{"model": "m", "state": "__engine_error__", "questions": {"u": {"type": "noul"}}}"#).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["error"], "decision engine error");
        assert!(!body.contains("mock"), "{body}");
    }
```

If `make_router_with_config` doesn't exist under that name, use the existing constructor that takes a `Config` (search: `grep -n "pub fn make_router" crates/cascadia-api/src/lib.rs`); `make_router_with_stats(runner, id, cfg, Arc::new(ApiStats::default()))` always works.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-api decisions_`
Expected: compile error (`Config.decisions` missing).

- [ ] **Step 3: Implement `handler.rs`**

```rust
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use indexmap::IndexMap;

use super::schema::{self, answer_from_probs, Answer, DecisionRequest, DecisionResponse, Usage};
use super::DecisionError;
use crate::AppState;

fn err(e: DecisionError) -> Response {
    (e.status(), Json(serde_json::json!({ "error": e.to_string() }))).into_response()
}

/// `generate_async` failures: log the EngineError (it can carry paths, peer and OpenVINO text),
/// return a fixed message per category. The chat handlers' `engine_error_response` echoes
/// `err.to_string()` and must not be reused here.
fn decision_engine_error(e: cascadia_engine::EngineError) -> DecisionError {
    use cascadia_engine::EngineError as E;
    tracing::warn!(error = %e, "decision submit failed");
    match e {
        E::PromptTooLong(_) => DecisionError::TooLong("decision prompt too long".into()),
        E::QueueFull { .. } => DecisionError::Engine("engine at capacity or unavailable".into()),
        _ => DecisionError::Engine("decision engine error".into()),
    }
}

/// Engine/head failures: log the detail (paths, OpenVINO text), return a fixed message.
fn engine_failure(reason: &str) -> DecisionError {
    tracing::warn!(reason, "decision pass failed");
    if reason.starts_with("decision_unsupported") {
        return DecisionError::NotServed;
    }
    DecisionError::Engine("decision engine error".into())
}

/// POST /v1/systemone: the Jev/SystemOne wire format.
pub async fn decide(State(state): State<AppState>, Json(req): Json<DecisionRequest>) -> Response {
    match evaluate(&state, &req).await {
        Ok((answers, input_tokens)) => Json(DecisionResponse { model: state.model_id.clone(), answers, usage: Usage { input_tokens, output_tokens: 0 } }).into_response(),
        Err(e) => err(e),
    }
}

/// The core both wire formats share: validate, encode, budget, run every pass, and return the
/// answers in request order plus the prompt tokens prefilled. Errors are `DecisionError`s, so each
/// route renders them in its own format (SystemOne `{"error": msg}`, OpenAI `{"error": {...}}`).
pub(crate) async fn evaluate(state: &AppState, req: &DecisionRequest) -> Result<(IndexMap<String, Answer>, u64), DecisionError> {
    let Some(svc) = state.decisions.clone() else {
        return Err(DecisionError::NotServed);
    };
    if let Err(e) = req.validate(&svc.limits()) {
        return Err(e);
    }
    // Take the permit BEFORE tokenizing: encode() is CPU work proportional to the request.
    let _permit = match state.permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return Err(DecisionError::Engine("engine at capacity; retry after current requests complete".into())),
    };
    // Cheap pre-check before any tokenization: a token is at least one byte, and this bound is
    // generous (~8 bytes per token) so honest requests never hit it.
    let state_bytes = match &req.state {
        schema::OrderedValue::String(s) => s.len(),
        other => other.py_json(false, true).len(),
    };
    if state_bytes > svc.max_total_tokens.saturating_mul(8) {
        return Err(DecisionError::TooLong(format!("state is {state_bytes} bytes; the limit is {}", svc.max_total_tokens * 8)));
    }
    // Tokenization is CPU work proportional to the request: run it off the async executor.
    let (svc2, req2) = (svc.clone(), req.clone());
    let encoded = match tokio::task::spawn_blocking(move || svc2.codec.encode(&req2)).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "decision encode task panicked");
            return Err(DecisionError::Engine("decision engine error".into()));
        }
    };
    let passes = match encoded {
        Ok(p) => p,
        Err(DecisionError::BadRequest(m)) => {
            tracing::warn!(error = %m, "decision encode failed");
            return Err(DecisionError::BadRequest("request could not be tokenized".into()));
        }
        Err(e) => return Err(e),
    };
    if let Some(p) = passes.iter().find(|p| p.readout.prompt_ids.len() > svc.max_tokens) {
        return Err(DecisionError::TooLong(format!("a decision pass is {} tokens; the limit is {}", p.readout.prompt_ids.len(), svc.max_tokens)));
    }
    let total = super::prefill_work(&passes, svc.prefix_reuse);
    if total > svc.max_total_tokens {
        return Err(DecisionError::TooLong(format!("this request needs {total} prefill tokens; the per-request limit is {}", svc.max_total_tokens)));
    }
    if let Some(f) = &state.decision_dump {
        use std::io::Write;
        // A poisoned lock (a panic mid-write) must not take the endpoint down: keep writing.
        let mut f = f.lock().unwrap_or_else(|e| e.into_inner());
        for p in &passes {
            let parts: Vec<_> = p.parts.iter().map(|x| (&x.question_id, x.logits.start, x.logits.end)).collect();
            if let Err(e) = writeln!(f, "{}", serde_json::json!({"parts": parts, "readout": p.readout})) {
                tracing::warn!(error = %e, "--decision-dump write failed");
                break;
            }
        }
    }
    let mut probs: IndexMap<String, Vec<f64>> = IndexMap::new();
    let mut input_tokens: u64 = 0;
    for pass in &passes {
        let task = cascadia_types::GenerationTask {
            task_id: format!("dec-{}", uuid::Uuid::new_v4().simple()),
            prompt: String::new(),
            max_tokens: 1,
            temperature: 0.0,
            logprobs: 0,
            sampling: Default::default(),
            enable_thinking: false,
            trust_remote_code: false,
            tenant: String::new(),
            resume_token_ids: None,
            decision: Some(pass.readout.clone()),
        };
        let mut stream = match state.runner.generate_async(task).await {
            Ok(s) => s,
            Err(e) => return Err(decision_engine_error(e)),
        };
        let mut logits = None;
        while let Some(chunk) = stream.next().await {
            if let Some(reason) = &chunk.error {
                return Err(engine_failure(reason));
            }
            if chunk.is_final {
                input_tokens += chunk.prompt_tokens.unwrap_or(pass.readout.prompt_ids.len() as u32) as u64;
                logits = chunk.decision_logits.clone();
            }
        }
        let Some(logits) = logits.filter(|l| l.len() == pass.readout.head.output_len()) else {
            return Err(engine_failure("engine returned no (or mis-sized) decision logits"));
        };
        for part in &pass.parts {
            probs.insert(part.question_id.clone(), svc.codec.probabilities(req, part, &logits[part.logits.clone()]));
        }
    }
    let answers = req.questions.iter().map(|(id, q)| (id.clone(), answer_from_probs(q, &probs[id]))).collect();
    Ok((answers, input_tokens))
}
```

`decision_engine_error` replaces the chat handlers' `crate::engine_error_response` (`lib.rs` ~3075), which returns `err.to_string()` to the client. The variants match `cascadia_engine::EngineError` (`crates/cascadia-engine/src/lib.rs`: `PromptTooLong(String)`, `QueueFull { queued, cap }`). `DecisionRequest` must derive `Clone` for the `spawn_blocking` move; it already does in Task 3.

The error test uses Task 2's mock sentinel: in this test's tiny tokenizer, map the word `__engine_error__` to id 999_999 (add it to `words` and give it that id when building the vocab).

In `lib.rs`:
- Add to `AppState`: `pub decisions: Option<Arc<decisions::DecisionService>>,` and `pub decision_dump: Option<Arc<std::sync::Mutex<std::fs::File>>>,`.
- Add to `Config`: `pub decisions: Option<Arc<decisions::DecisionService>>,` and `pub decision_dump: Option<std::path::PathBuf>,`, both `None` in `Config::default()`. `Config` derives `Debug`, which `DecisionService` implements by hand (Task 4).
- In `make_router_with_stats`, open the dump once and put both into the `AppState { .. }` literal:

```rust
    let decision_dump = cfg.decision_dump.as_ref().map(|p| {
        let mut o = std::fs::OpenOptions::new();
        o.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // 0600 on create; refuse to follow a symlink planted at the path.
            o.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let f = o.open(p).unwrap_or_else(|e| panic!("--decision-dump {}: {e}", p.display()));
        // `mode` only applies on create: tighten a pre-existing file too.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            f.set_permissions(std::fs::Permissions::from_mode(0o600))
                .unwrap_or_else(|e| panic!("--decision-dump {}: chmod 0600: {e}", p.display()));
        }
        Arc::new(std::sync::Mutex::new(f))
    });
```

  with `decisions: cfg.decisions.clone(), decision_dump,` in the literal.
- Add `.route("/v1/systemone", post(decisions::handler::decide))` after the `/v1/completions` route.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-api` and `cargo test -p cascadia-engine-mock`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-api
git commit -m "feat(api): shared decision core and POST /v1/systemone"
```

---

### Task 8b: CLI wiring and API docs

**Files:**
- Modify: `crates/cascadia-cli/src/lib.rs` (`WorkerArgs`, `RunArgs`, `WorkerArgs::single_node`, the API config block at line ~2337)
- Create: `docs/decisions.md`

**Interfaces:**
- Consumes: Task 8's `Config.{decisions, decision_dump}`, Task 4's `DecisionService` (`prefix_reuse` field).
- Produces:
  - `--decision-dump FILE` on `cascadia worker` and `cascadia run`;
  - `/v1/decisions` (OpenAI format) and `/v1/systemone` turned on for `--engine qwen35` when `<model>/decision.json` exists;
  - `DecisionService.prefix_reuse = (total <= 1 && prefix cache bytes > 0)`.

- [ ] **Step 1: Write the failing test** (in the existing run/worker threading test module, next to the test at line ~3810 that checks `prefix_cache_gb`)

```rust
    #[test]
    fn run_forwards_decision_dump() {
        let cli = Cli::try_parse_from(["cascadia", "run", "some-model-dir", "--engine", "qwen35", "--decision-dump", "/tmp/d.jsonl"])
            .expect("parse run argv");
        let Command::Run(args) = cli.cmd else { panic!("expected run subcommand") };
        assert_eq!(args.decision_dump.as_deref(), Some(std::path::Path::new("/tmp/d.jsonl")));
        let worker = WorkerArgs::single_node(
            args.model, args.device, args.engine, args.api, args.prefix_cache_gb, args.api_max_body_mb,
            Vec::new(), args.ep_placement, args.decision_dump,
        );
        assert_eq!(worker.decision_dump.as_deref(), Some(std::path::Path::new("/tmp/d.jsonl")));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-cli run_forwards_decision_dump`
Expected: compile error (no `decision_dump` field).

- [ ] **Step 3: Implement**

Add to `WorkerArgs` and `RunArgs`:

```rust
    /// Append every /v1/decisions pass (codec token ids + head layout) as JSONL to FILE, mode 0600.
    /// For parity checks only: the file holds request-derived data.
    #[arg(long)]
    pub decision_dump: Option<PathBuf>,
```

Add a trailing `decision_dump: Option<PathBuf>` parameter to `WorkerArgs::single_node` (line ~765) and set the field. Pass `args.decision_dump` at the `run` call site (line ~1077), and `None` at every other `single_node` call, including the existing tests at lines ~3827 and ~3860.

Right after `cfg.chat_template = chat_template;` (line ~2337):

```rust
        // Decision models: a `decision.json` beside the shard tree turns on /v1/decisions and /v1/systemone.
        // Only qwen35 implements decision passes; other engines would run them as empty chats.
        if matches!(args.engine, EngineKind::Qwen36Moe) {
            match cascadia_api::decisions::DecisionService::load(std::path::Path::new(&args.model)) {
                Ok(Some(svc)) => {
                    info!(model = %args.model, "decision.json found; serving /v1/decisions (OpenAI format) and /v1/systemone");
                    // The shared-prefix budget discount needs real reuse: a single box with the
                    // prefix cache on. Pipeline ranks re-prefill Kev's state per question.
                    let mut svc = std::sync::Arc::try_unwrap(svc).unwrap_or_else(|_| unreachable!("fresh Arc"));
                    svc.prefix_reuse = args.total <= 1 && prefix_cache_bytes > 0;
                    cfg.decisions = Some(std::sync::Arc::new(svc));
                }
                Ok(None) => {}
                Err(e) => warn!(model = %args.model, error = %e, "decision.json present but unusable; /v1/decisions disabled"),
            }
        } else if std::path::Path::new(&args.model).join("decision.json").exists() {
            warn!(engine = ?args.engine, "decision.json ignored: only --engine qwen35 serves /v1/decisions");
        }
        if let Some(p) = &args.decision_dump {
            warn!(path = %p.display(), "--decision-dump records request-derived token ids to disk; use for parity checks only");
            cfg.decision_dump = Some(p.clone());
        }
```

`prefix_cache_bytes` is the value `resolve_prefix_cache_bytes(args.prefix_cache_gb)?` already computes at line ~1992. Pass it down to this block if it isn't in scope there, and compare it the way its type requires (`> 0`).

Create `docs/decisions.md` with the API section: **both endpoints**, `POST /v1/decisions` (OpenAI Decisions API format, Task 8c: request/response shapes, the mapping notes, no `refusal`, OpenAI error envelope) and `POST /v1/systemone` (Jev/SystemOne format, Task 3), an OpenAI SDK example pointed at Cascadia (`OpenAI(base_url="http://HOST:PORT/v1", api_key="unused")` then `client.decisions.create(...)`), the status codes (Task 8), the limits (`max_questions` default 32 and `max_tokens`/`max_total_tokens` from `decision.json`; the shared-prefix discount applies only on a single box with the prefix cache on; the `--api-max-body-mb` body cap, default 1 MiB, which a long Kev state needs raised), the `--decision-dump` flag and its privacy note, the confidence semantics, and the limitations (text only, batch 1, int8 only, JEV choice ≤ 16 options). Tasks 18–22 append a section per model.

Two notes the docs must state plainly:
- **Confidence follows TypeSafe's system-one-adapter 0.2.1** (choice: `(p_max − 1/K)/(1 − 1/K)`; score: `1 − E|level − mode|/D`), the formulas Kev's server also uses, for every family and both routes. OpenAI's documented examples match them exactly (choice 0.93, score 0.55). Cloudflare's own Clef server reports max-probability instead. On the spike's records the two differ by 0.05 on average and up to 0.36 for Clef. Probabilities and the chosen answer are identical either way; only the `confidence` field differs.
- **JEV score answers are an approximation.** JEV only knows a fixed 0–5 scale and never sees the level descriptions, so the server maps its 6-way distribution linearly onto the requested L levels (`rescale_score`). On the 25 four-level score questions in the parity set, this mapping agreed with models that do read the descriptions (Kev and Clef) more often than truncation did: 20 vs 19 consistent with their choices, and a mean expected-score gap of 0.08 vs 0.13. Those two models agree with each other on only 15 of 25 score questions, so treat JEV score answers as indicative. For exact score semantics, use Kev or Clef.
- **OpenAI-format score levels** are rendered as one text per level (`"label: description"`, or `"label"`) before the codec sees them. This approximates the single-text levels the models were trained on.
- **OpenAI-format questions are "independent" in OpenAI's docs**; Clef decides all of a request's questions in one joint pass, by design.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-cli` and `cargo build --workspace`
Expected: PASS / builds.

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-cli docs/decisions.md
git commit -m "feat(cli): serve /v1/decisions for qwen35 decision trees; --decision-dump; API docs"
```

---

### Task 8c: OpenAI-format `POST /v1/decisions`

OpenAI's Decisions API (developers.openai.com/api/docs/guides/decisions) owns the `/v1/decisions` path shape, so Cascadia serves that format there and OpenAI SDK clients (`client.decisions.create(...)`) work unchanged. This task is a thin adapter: an OpenAI request becomes the internal `DecisionRequest`, the shared `evaluate` core runs it (Task 8), and the internal answers become OpenAI answers. Codecs, engine and models are untouched.

**Mapping**

- `input`:
  - a string becomes the state string;
  - a message list must have `role: "user"` only. Each message's `content` is either a string or parts. A message's parts are concatenated (`input_text` texts, the same rule as serve_decide `_parts`), and messages are joined with `"\n"`;
  - `input_image` is a 400, since this server is text-only.
- `predicate {name, instructions}` → `noul` (no criteria) → `{type: "predicate", name, probability: p(true)}`.
- `choice {name, instructions, choices: [{value, description?}]}` → `choice` with criteria `{key(value): description}` in request order → `{type: "choice", name, choice: <original value>, probabilities: [{value, probability}] (request order), confidence}`.
  - `value` may be a string or a number (the Go SDK types it as a union); `key(value)` is the string itself, or its JSON text for a number.
  - Duplicate keys are a 400.
- `score {name, instructions, levels: [{label, description?}]}` → `score` with level texts `"label: description"`, or `"label"` when there's no description → `{type: "score", name, score, probabilities: [{value: index, label, probability}], confidence}`.
  - This rendering is an approximation: the reference models were trained on one text per level, and nothing in the references says which form they score best on. JEV never sees level text (0–5 rescale), so only Kev and Clef are affected.
- `name` is required, non-empty and unique; it keys the internal question map, so answers come back in request order.
- `model` is accepted and ignored (one model per server), like the other routes.
- The response is exactly `{"answers": [...]}`, as in OpenAI's examples. `refusal` is never emitted.
- Errors are `{"error": {"message": "...", "type": "invalid_request_error" | "server_error", "code": null}}`, the shape OpenAI SDKs parse. The HTTP statuses are the shared core's (400, 413, 501, 503).
- Confidence is TypeSafe's, which matches OpenAI's examples exactly (Task 3).

**Files:**
- Create: `crates/cascadia-api/src/decisions/openai.rs`
- Modify: `crates/cascadia-api/src/decisions/mod.rs` (`pub mod openai;`), `crates/cascadia-api/src/lib.rs` (route `.route("/v1/decisions", post(decisions::openai::decide))` next to `/v1/systemone`; tests)

**Interfaces:**
- Consumes: Task 8's `handler::evaluate(&AppState, &DecisionRequest) -> Result<(IndexMap<String, Answer>, u64), DecisionError>`; Task 3's `DecisionRequest`, `Question`, `QuestionType`, `OrderedValue`, `Answer`.
- Produces:
  - `openai::{OaRequest, OaAnswer, OaResponse}`;
  - `openai::to_internal(&OaRequest) -> Result<(DecisionRequest, Vec<OaMeta>), DecisionError>`, where `OaMeta` keeps each question's name, original choice values and level labels for the reply;
  - `openai::to_openai(answers: &IndexMap<String, Answer>, meta: &[OaMeta]) -> OaResponse`;
  - `openai::decide` (the axum handler).

- [ ] **Step 1: Write the failing tests** (bottom of `openai.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn oa(s: &str) -> OaRequest {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn maps_every_question_type_and_input_shape() {
        let (r, meta) = to_internal(&oa(r#"{"model": "gpt-6-luna",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "charged "}, {"type": "input_text", "text": "twice"}]},
                      {"role": "user", "content": "order 7"}],
            "questions": [
              {"type": "choice", "name": "department", "instructions": "Which department?",
               "choices": [{"value": "billing", "description": "Payments."}, {"value": 2, "description": "Two."}, {"value": "other"}]},
              {"type": "predicate", "name": "urgent", "instructions": "Urgent?"},
              {"type": "score", "name": "severity", "instructions": "How bad?",
               "levels": [{"label": "Cosmetic", "description": "Appearance only."}, {"label": "Blocked"}]}]}"#)).unwrap();
        assert_eq!(r.state, OrderedValue::String("charged twice\norder 7".into()));
        let ids: Vec<&str> = r.questions.keys().map(String::as_str).collect();
        assert_eq!(ids, ["department", "urgent", "severity"]);
        assert_eq!(r.questions["department"].option_keys(), ["billing", "2", "other"]);
        assert_eq!(r.questions["department"].choice_criteria()[2].1, OrderedValue::Null);
        assert_eq!(r.questions["urgent"].qtype, QuestionType::Noul);
        assert_eq!(r.questions["severity"].score_levels(),
                   vec![OrderedValue::String("Cosmetic: Appearance only.".into()), OrderedValue::String("Blocked".into())]);
        assert_eq!(meta[0].choice_values[1], OrderedValue::Number(2.into()));
        assert_eq!(meta[2].level_labels, ["Cosmetic", "Blocked"]);
    }

    #[test]
    fn rejects_images_other_roles_and_bad_names() {
        let bad = |s: &str| matches!(to_internal(&oa(s)), Err(DecisionError::BadRequest(_)));
        assert!(bad(r#"{"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}], "questions": [{"type": "predicate", "name": "a", "instructions": "x"}]}"#));
        assert!(bad(r#"{"input": [{"role": "system", "content": "x"}], "questions": [{"type": "predicate", "name": "a", "instructions": "x"}]}"#));
        assert!(bad(r#"{"input": "x", "questions": [{"type": "predicate", "name": "a", "instructions": "x"}, {"type": "predicate", "name": "a", "instructions": "y"}]}"#));
        assert!(bad(r#"{"input": "x", "questions": [{"type": "predicate", "name": "", "instructions": "x"}]}"#));
        assert!(bad(r#"{"input": "x", "questions": [{"type": "choice", "name": "c", "instructions": "x", "choices": [{"value": "a"}, {"value": "a"}]}]}"#));
    }

    #[test]
    fn answers_render_in_openai_shape() {
        let (r, meta) = to_internal(&oa(r#"{"input": "x", "questions": [
            {"type": "choice", "name": "department", "instructions": "q", "choices": [{"value": "billing"}, {"value": "technical"}, {"value": "shipping"}, {"value": "other"}]},
            {"type": "score", "name": "severity", "instructions": "q", "levels": [{"label": "Cosmetic"}, {"label": "Workaround available"}, {"label": "Fully blocked"}]},
            {"type": "predicate", "name": "urgent", "instructions": "q"}]}"#)).unwrap();
        let mut answers = indexmap::IndexMap::new();
        answers.insert("department".to_string(), crate::decisions::schema::answer_from_probs(&r.questions["department"], &[0.95, 0.02, 0.01, 0.02]));
        answers.insert("severity".to_string(), crate::decisions::schema::answer_from_probs(&r.questions["severity"], &[0.1, 0.7, 0.2]));
        answers.insert("urgent".to_string(), crate::decisions::schema::answer_from_probs(&r.questions["urgent"], &[0.2, 0.8]));
        let v = serde_json::to_value(to_openai(&answers, &meta)).unwrap();
        // OpenAI's documented examples, reproduced exactly (confidence 0.93 and 0.55 included)
        assert_eq!(v["answers"][0], serde_json::json!({"type": "choice", "name": "department", "choice": "billing",
            "probabilities": [{"value": "billing", "probability": 0.95}, {"value": "technical", "probability": 0.02},
                              {"value": "shipping", "probability": 0.01}, {"value": "other", "probability": 0.02}],
            "confidence": 0.9333}));
        assert_eq!(v["answers"][1], serde_json::json!({"type": "score", "name": "severity", "score": 1.1,
            "probabilities": [{"value": 0, "label": "Cosmetic", "probability": 0.1}, {"value": 1, "label": "Workaround available", "probability": 0.7},
                              {"value": 2, "label": "Fully blocked", "probability": 0.2}],
            "confidence": 0.55}));
        assert_eq!(v["answers"][2], serde_json::json!({"type": "predicate", "name": "urgent", "probability": 0.8}));
        assert_eq!(v.as_object().unwrap().len(), 1, "only `answers`, as in OpenAI's responses");
    }
}
```

OpenAI's example shows `confidence: 0.93`; the server reports 4 decimals (`0.9333`), and `(0.95 − 0.25)/0.75 = 0.93333…`. The score case is exact: `1 − 0.3/(2/3) = 0.55`.

HTTP tests (in `lib.rs`'s test module, beside the Task 8 ones; they reuse `make_decision_app`, and the mock's first option wins):

```rust
    async fn post_openai(app: Router, payload: &str) -> (StatusCode, Value) {
        let response = app
            .oneshot(Request::builder().method("POST").uri("/v1/decisions").header("content-type", "application/json").body(Body::from(payload.to_owned())).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 65536).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn openai_decisions_route_answers_in_openai_shape() {
        let (status, v) = post_openai(make_decision_app().await, r#"{"model": "gpt-6-luna", "input": "I was charged twice for my order.",
            "questions": [{"type": "choice", "name": "department", "instructions": "Which department?",
              "choices": [{"value": "billing", "description": "Payments."}, {"value": "technical", "description": "Bugs."}]},
              {"type": "predicate", "name": "urgent", "instructions": "Urgent?"}]}"#).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["answers"][0]["type"], "choice");
        assert_eq!(v["answers"][0]["name"], "department");
        assert_eq!(v["answers"][0]["choice"], "billing");
        assert_eq!(v["answers"][0]["probabilities"][1]["value"], "technical");
        assert_eq!(v["answers"][1]["type"], "predicate");
        assert!(v["answers"][1]["probability"].as_f64().unwrap() < 0.5, "mock: first option (false) wins");
    }

    #[tokio::test]
    async fn openai_errors_use_openai_error_shape() {
        let (status, v) = post_openai(make_decision_app().await, r#"{"input": "x", "questions": []}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn openai_and_systemone_routes_agree() {
        let oa = r#"{"input": "a b", "questions": [{"type": "choice", "name": "team", "instructions": "who", "choices": [{"value": "billing", "description": "x"}, {"value": "tech", "description": "y"}]},
                    {"type": "score", "name": "sev", "instructions": "how", "levels": [{"label": "lo"}, {"label": "mid"}, {"label": "hi"}]}]}"#;
        let so = r#"{"model": "m", "state": "a b", "questions": {"team": {"type": "choice", "instructions": "who", "criteria": {"billing": "x", "tech": "y"}},
                    "sev": {"type": "score", "instructions": "how", "criteria": ["lo", "mid", "hi"]}}}"#;
        let (_, a) = post_openai(make_decision_app().await, oa).await;
        let (_, b, _) = post_decisions(make_decision_app().await, so).await;
        assert_eq!(a["answers"][0]["probabilities"][0]["probability"], b["answers"]["team"]["probabilities"]["billing"]);
        assert_eq!(a["answers"][1]["score"], b["answers"]["sev"]["score"]);
        assert_eq!(a["answers"][1]["confidence"], b["answers"]["sev"]["confidence"]);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-api openai`
Expected: compile error (module missing).

- [ ] **Step 3: Implement `openai.rs`**

```rust
//! OpenAI Decisions API wire format (developers.openai.com/api/docs/guides/decisions) at
//! POST /v1/decisions. A thin adapter over the shared core: the models see the same prompts as
//! through /v1/systemone.
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::schema::{Answer, DecisionRequest, OrderedValue, Question, QuestionType};
use super::DecisionError;
use crate::AppState;

#[derive(Clone, Debug, Deserialize)]
pub struct OaRequest {
    #[serde(default)]
    pub model: String,
    pub input: OaInput,
    pub questions: Vec<OaQuestion>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OaInput {
    Text(String),
    Messages(Vec<OaMessage>),
}

#[derive(Clone, Debug, Deserialize)]
pub struct OaMessage {
    #[serde(default = "user_role")]
    pub role: String,
    pub content: OaContent,
}

fn user_role() -> String {
    "user".into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OaContent {
    Text(String),
    Parts(Vec<OaPart>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OaPart {
    InputText { text: String },
    InputImage { image_url: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OaQuestion {
    Predicate { name: String, instructions: String },
    Choice { name: String, instructions: String, choices: Vec<OaChoice> },
    Score { name: String, instructions: String, levels: Vec<OaLevel> },
}

#[derive(Clone, Debug, Deserialize)]
pub struct OaChoice {
    pub value: OrderedValue,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OaLevel {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// What the reply needs back per question: its name, the caller's original choice values (their
/// JSON type is echoed, not the internal string key), and the level labels.
#[derive(Clone, Debug)]
pub struct OaMeta {
    pub name: String,
    pub choice_values: Vec<OrderedValue>,
    pub level_labels: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OaChoiceProb {
    pub value: OrderedValue,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OaLevelProb {
    pub value: usize,
    pub label: String,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OaAnswer {
    Predicate { name: String, probability: f64 },
    Choice { name: String, choice: OrderedValue, probabilities: Vec<OaChoiceProb>, confidence: f64 },
    Score { name: String, score: f64, probabilities: Vec<OaLevelProb>, confidence: f64 },
}

#[derive(Clone, Debug, Serialize)]
pub struct OaResponse {
    pub answers: Vec<OaAnswer>,
}

fn bad(m: impl Into<String>) -> DecisionError {
    DecisionError::BadRequest(m.into())
}

fn input_text(input: &OaInput) -> Result<String, DecisionError> {
    match input {
        OaInput::Text(t) => Ok(t.clone()),
        OaInput::Messages(msgs) => {
            let mut out = Vec::with_capacity(msgs.len());
            for m in msgs {
                if m.role != "user" {
                    return Err(bad(format!("input messages must have role \"user\", got {:?}", m.role)));
                }
                out.push(match &m.content {
                    OaContent::Text(t) => t.clone(),
                    OaContent::Parts(parts) => {
                        let mut s = String::new();
                        for p in parts {
                            match p {
                                OaPart::InputText { text } => s.push_str(text),
                                OaPart::InputImage { .. } => return Err(bad("input_image is not supported (text-only server)")),
                            }
                        }
                        s
                    }
                });
            }
            Ok(out.join("\n"))
        }
    }
}

fn choice_key(v: &OrderedValue) -> String {
    match v {
        OrderedValue::String(s) => s.clone(),
        other => other.py_json(false, true),
    }
}

/// OpenAI request → the internal model every codec consumes, plus what the reply needs.
pub fn to_internal(r: &OaRequest) -> Result<(DecisionRequest, Vec<OaMeta>), DecisionError> {
    let mut questions: IndexMap<String, Question> = IndexMap::new();
    let mut meta = Vec::with_capacity(r.questions.len());
    for q in &r.questions {
        let (name, instructions) = match q {
            OaQuestion::Predicate { name, instructions }
            | OaQuestion::Choice { name, instructions, .. }
            | OaQuestion::Score { name, instructions, .. } => (name, instructions),
        };
        if name.is_empty() {
            return Err(bad("every question needs a non-empty name"));
        }
        if questions.contains_key(name) {
            return Err(bad(format!("duplicate question name {name:?}")));
        }
        let mut m = OaMeta { name: name.clone(), choice_values: Vec::new(), level_labels: Vec::new() };
        let (qtype, criteria) = match q {
            OaQuestion::Predicate { .. } => (QuestionType::Noul, None),
            OaQuestion::Choice { choices, .. } => {
                let mut crit: Vec<(String, OrderedValue)> = Vec::with_capacity(choices.len());
                for c in choices {
                    let key = choice_key(&c.value);
                    if crit.iter().any(|(k, _)| *k == key) {
                        return Err(bad(format!("{name}: duplicate choice value {key:?}")));
                    }
                    crit.push((key, c.description.clone().map(OrderedValue::String).unwrap_or(OrderedValue::Null)));
                    m.choice_values.push(c.value.clone());
                }
                (QuestionType::Choice, Some(OrderedValue::Object(crit)))
            }
            OaQuestion::Score { levels, .. } => {
                let texts = levels
                    .iter()
                    .map(|l| {
                        m.level_labels.push(l.label.clone());
                        // One text per level, as the models were trained; see the mapping notes.
                        OrderedValue::String(match l.description.as_deref().filter(|d| !d.is_empty()) {
                            Some(d) => format!("{}: {d}", l.label),
                            None => l.label.clone(),
                        })
                    })
                    .collect();
                (QuestionType::Score, Some(OrderedValue::Array(texts)))
            }
        };
        questions.insert(name.clone(), Question { qtype, instructions: Some(OrderedValue::String(instructions.clone())), criteria });
        meta.push(m);
    }
    let req = DecisionRequest {
        model: r.model.clone(),
        state: OrderedValue::String(input_text(&r.input)?),
        questions,
        images: Vec::new(),
        videos: Vec::new(),
    };
    Ok((req, meta))
}

/// Internal answers (request order) → OpenAI answers.
pub fn to_openai(answers: &IndexMap<String, Answer>, meta: &[OaMeta]) -> OaResponse {
    let out = meta
        .iter()
        .map(|m| match &answers[&m.name] {
            Answer::Noul { noul } => OaAnswer::Predicate { name: m.name.clone(), probability: *noul },
            Answer::Choice { choice, confidence, probabilities } => {
                let keyed: Vec<(&String, &f64)> = probabilities.iter().collect();
                let idx = keyed.iter().position(|(k, _)| *k == choice).unwrap_or(0);
                OaAnswer::Choice {
                    name: m.name.clone(),
                    choice: m.choice_values[idx].clone(),
                    probabilities: keyed.iter().zip(&m.choice_values).map(|((_, p), v)| OaChoiceProb { value: v.clone(), probability: **p }).collect(),
                    confidence: *confidence,
                }
            }
            Answer::Score { score, confidence, probabilities, .. } => OaAnswer::Score {
                name: m.name.clone(),
                score: *score,
                probabilities: probabilities
                    .values()
                    .enumerate()
                    .map(|(i, p)| OaLevelProb { value: i, label: m.level_labels.get(i).cloned().unwrap_or_default(), probability: *p })
                    .collect(),
                confidence: *confidence,
            },
        })
        .collect();
    OaResponse { answers: out }
}

/// OpenAI's error envelope, which its SDKs parse into typed errors.
fn oa_err(e: DecisionError) -> Response {
    let kind = match e {
        DecisionError::BadRequest(_) | DecisionError::TooLong(_) => "invalid_request_error",
        DecisionError::NotServed | DecisionError::Engine(_) => "server_error",
    };
    (e.status(), Json(serde_json::json!({"error": {"message": e.to_string(), "type": kind, "code": null}}))).into_response()
}

/// POST /v1/decisions: the OpenAI Decisions API wire format.
pub async fn decide(State(state): State<AppState>, Json(r): Json<OaRequest>) -> Response {
    let (req, meta) = match to_internal(&r) {
        Ok(x) => x,
        Err(e) => return oa_err(e),
    };
    match super::handler::evaluate(&state, &req).await {
        Ok((answers, _input_tokens)) => Json(to_openai(&answers, &meta)).into_response(),
        Err(e) => oa_err(e),
    }
}
```

A malformed body is rejected by axum's `Json` extractor before `decide` runs, with axum's default 4xx body, not OpenAI's envelope. That matches the other routes and is acceptable: SDK clients only send well-formed bodies.

`Question`'s fields are `pub` (Task 3), so `to_internal` builds it directly. `DecisionRequest::validate` (question count, option limits, the per-family 16-choice JEV cap) still runs inside `evaluate`, so the OpenAI route gets every limit the SystemOne route has.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-api`
Expected: PASS (3 unit tests in `openai.rs` plus 3 HTTP tests, and every earlier test).

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-api
git commit -m "feat(api): OpenAI-format POST /v1/decisions over the shared decision core"
```

---

### Task 9: Engine decision heads (pure logic)

**Files:**
- Create: `crates/cascadia-engine-openvino/src/decision.rs`
- Modify: `crates/cascadia-engine-openvino/src/lib.rs` (`mod decision;`)

**Interfaces:**
- Consumes: `cascadia_types::{DecisionReadout, ReadoutHead, HiddenNeed}`, `cascadia_ov_genai_shim::{Runtime, DType, PluginConfig}`.
- Produces:
  - `decision::Collector::new(need: HiddenNeed, hidden: usize)`
  - `Collector::absorb(&mut self, t0: usize, n: usize, rows: &[f32])` (rows = `[n, hidden]` final-norm hidden for positions `t0..t0+n`)
  - `decision::HeadAssets::load(dir: &Path, hidden: usize) -> Result<HeadAssets, String>` (reads `decision/head.json`)
  - `decision::run_head(head: &ReadoutHead, assets: &mut HeadAssets, prompt_ids: &[u32], last_logits: &[f32], col: &Collector) -> Result<Vec<f32>, String>`
  - `Collector::absorb` returns `Result<(), String>`: for `HiddenNeed::All` it refuses a chunk whose `t0` is not the number of rows already held (a skipped or warm-restored prefix fails loudly instead of misaligning spans)
  - frame bodies for Task 11, all `[u32 BE byte length][JSON]` so a middle rank can relay them opaquely:
    - `decision::encode_readout(&DecisionReadout) -> Vec<u8>` / `decode_readout(&[u8]) -> Result<DecisionReadout, String>`
    - `decision::encode_result(&Result<Vec<f32>, String>) -> Vec<u8>` / `decode_result(&[u8]) -> Result<Result<Vec<f32>, String>, String>` (JSON `{"ok":[...]}` or `{"err":"..."}`)
  - `decision::MAX_DECISION_BODY_BYTES: usize = 16 << 20` (ceiling for either body; readers drain over-ceiling bodies to stay framed)

`decision/head.json` (written by Task 15):
- `{"kind": "verbalizer"}`
- `{"kind": "pointer", "dp": 256}`: alongside it, `pointer_q_w.f32` `[dp, hidden]`, `pointer_q_b.f32` `[dp]`, `pointer_k_w.f32`, `pointer_k_b.f32` (raw little-endian f32)
- `{"kind": "joint", "vocab": 248320}`: alongside it, `joint_head.xml`/`.bin` and `lm_head.f16` (`[vocab, hidden]` little-endian f16)

- [ ] **Step 1: Write the failing tests** (bottom of `decision.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::{HiddenNeed, ReadoutHead};

    #[test]
    fn collector_keeps_requested_rows_across_chunks() {
        let mut c = Collector::new(HiddenNeed::Positions(vec![1, 4]), 2);
        c.absorb(0, 3, &[0.0, 0.0, 1.0, 1.5, 2.0, 2.0]).unwrap();
        c.absorb(3, 2, &[3.0, 3.0, 4.0, 4.5]).unwrap();
        assert_eq!(c.row(1).unwrap(), &[1.0, 1.5]);
        assert_eq!(c.row(4).unwrap(), &[4.0, 4.5]);
        assert!(c.row(2).is_none());
    }

    #[test]
    fn collector_all_keeps_every_row_in_order() {
        let mut c = Collector::new(HiddenNeed::All, 1);
        c.absorb(0, 2, &[0.0, 1.0]).unwrap();
        c.absorb(2, 1, &[2.0]).unwrap();
        assert_eq!(c.all(), &[0.0, 1.0, 2.0]);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn collector_all_refuses_a_gap() {
        let mut c = Collector::new(HiddenNeed::All, 1);
        assert!(c.absorb(5, 1, &[0.0]).is_err(), "rows 0..5 were never seen (warm restore)");
    }

    #[test]
    fn verbalizer_picks_logit_rows() {
        let mut a = HeadAssets::Verbalizer;
        let head = ReadoutHead::Verbalizer { token_ids: vec![2, 0] };
        let c = Collector::new(HiddenNeed::None, 4);
        assert_eq!(run_head(&head, &mut a, &[], &[0.5, 1.5, 2.5], &c).unwrap(), vec![2.5, 0.5]);
        assert!(run_head(&ReadoutHead::Verbalizer { token_ids: vec![9] }, &mut a, &[], &[0.0], &c).is_err());
    }

    #[test]
    fn pointer_head_matches_hand_computation() {
        // hidden 2, dp 1: q = [1,0]·h + 0 ; k = [0,1]·h + 1 ; z = k*q / sqrt(1)
        let mut a = HeadAssets::Pointer { dp: 1, hidden: 2, q_w: vec![1.0, 0.0], q_b: vec![0.0], k_w: vec![0.0, 1.0], k_b: vec![1.0] };
        let mut c = Collector::new(HiddenNeed::Positions(vec![0, 1, 2]), 2);
        c.absorb(0, 3, &[0.0, 2.0, 0.0, 3.0, 2.0, 0.0]).unwrap(); // pos0=[0,2] pos1=[0,3] decide=pos2=[2,0]
        let head = ReadoutHead::Pointer { decide_pos: 2, option_pos: vec![0, 1] };
        // q = 2 ; k0 = 3 ; k1 = 4 ⇒ z = [6, 8]
        assert_eq!(run_head(&head, &mut a, &[], &[], &c).unwrap(), vec![6.0, 8.0]);
    }

    #[test]
    fn joint_span_mask_rows_average_their_span() {
        let m = span_mask(&[[0, 2], [2, 3]], 3);
        assert_eq!(m, vec![0.5, 0.5, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn frame_bodies_round_trip() {
        let r = cascadia_types::DecisionReadout { prompt_ids: vec![1, 2], head: ReadoutHead::Pointer { decide_pos: 1, option_pos: vec![0] }, cache_prefix: None };
        assert_eq!(decode_readout(&encode_readout(&r)).unwrap(), r);
        assert_eq!(decode_result(&encode_result(&Ok(vec![1.5, -2.0]))).unwrap(), Ok(vec![1.5, -2.0]));
        assert_eq!(decode_result(&encode_result(&Err("boom".into()))).unwrap(), Err("boom".to_string()));
        assert!(decode_result(&[0, 0, 0, 9]).is_err(), "truncated");
        assert!(decode_readout(&[0xff, 0xff, 0xff, 0xff]).is_err(), "length beyond the buffer");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-engine-openvino decision::`
Expected: compile error.

- [ ] **Step 3: Implement `decision.rs`**

```rust
//! Decision heads run where the final hidden states live (single box or the last pipeline rank).
//! See docs/decisions.md. Pure logic here; qwen36.rs drives it.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use cascadia_types::{DecisionReadout, HiddenNeed, ReadoutHead};
use serde::{Deserialize, Serialize};

pub struct Collector {
    need: HiddenNeed,
    hidden: usize,
    rows: BTreeMap<u32, Vec<f32>>,
    all: Vec<f32>,
}

impl Collector {
    pub fn new(need: HiddenNeed, hidden: usize) -> Self {
        Self { need, hidden, rows: BTreeMap::new(), all: Vec::new() }
    }

    pub fn wants_hidden(&self) -> bool {
        self.need != HiddenNeed::None
    }

    pub fn absorb(&mut self, t0: usize, n: usize, rows: &[f32]) -> Result<(), String> {
        if rows.len() < n * self.hidden {
            return Err(format!("final_hidden has {} values for {n} rows of {}", rows.len(), self.hidden));
        }
        match &self.need {
            HiddenNeed::None => {}
            HiddenNeed::All => {
                if t0 != self.len() {
                    return Err(format!("hidden rows arrive at {t0} but {} are held (prefix skipped?)", self.len()));
                }
                self.all.extend_from_slice(&rows[..n * self.hidden])
            }
            HiddenNeed::Positions(want) => {
                for &p in want {
                    let p = p as usize;
                    if p >= t0 && p < t0 + n {
                        let r = (p - t0) * self.hidden;
                        self.rows.insert(p as u32, rows[r..r + self.hidden].to_vec());
                    }
                }
            }
        }
        Ok(())
    }

    pub fn row(&self, p: u32) -> Option<&[f32]> {
        self.rows.get(&p).map(Vec::as_slice)
    }

    pub fn all(&self) -> &[f32] {
        &self.all
    }

    pub fn len(&self) -> usize {
        self.all.len() / self.hidden.max(1)
    }
}

pub enum HeadAssets {
    Verbalizer,
    Pointer { dp: usize, hidden: usize, q_w: Vec<f32>, q_b: Vec<f32>, k_w: Vec<f32>, k_b: Vec<f32> },
    Joint { runtime: Runtime, lm_head: PathBuf, vocab: usize, hidden: usize },
}

fn read_f32(path: &Path, expect: usize) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() != expect * 4 {
        return Err(format!("{}: {} bytes, expected {}", path.display(), b.len(), expect * 4));
    }
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

impl HeadAssets {
    pub fn load(model_dir: &Path, hidden: usize) -> Result<HeadAssets, String> {
        let dir = model_dir.join("decision");
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("head.json")).map_err(|e| format!("decision/head.json: {e}"))?,
        )
        .map_err(|e| format!("decision/head.json: {e}"))?;
        match meta["kind"].as_str() {
            Some("verbalizer") => Ok(HeadAssets::Verbalizer),
            Some("pointer") => {
                let dp = meta["dp"].as_u64().ok_or("pointer head.json lacks dp")? as usize;
                Ok(HeadAssets::Pointer {
                    dp,
                    hidden,
                    q_w: read_f32(&dir.join("pointer_q_w.f32"), dp * hidden)?,
                    q_b: read_f32(&dir.join("pointer_q_b.f32"), dp)?,
                    k_w: read_f32(&dir.join("pointer_k_w.f32"), dp * hidden)?,
                    k_b: read_f32(&dir.join("pointer_k_b.f32"), dp)?,
                })
            }
            Some("joint") => {
                let vocab = meta["vocab"].as_u64().ok_or("joint head.json lacks vocab")? as usize;
                let xml = dir.join("joint_head.xml");
                let runtime = Runtime::compile(xml.to_str().ok_or("non-utf8 path")?, "CPU", &PluginConfig::default())
                    .map_err(|e| format!("joint head compile: {e}"))?;
                Ok(HeadAssets::Joint { runtime, lm_head: dir.join("lm_head.f16"), vocab, hidden })
            }
            other => Err(format!("decision/head.json: unknown kind {other:?}")),
        }
    }
}

fn affine(w: &[f32], b: &[f32], x: &[f32], dp: usize, hidden: usize) -> Vec<f32> {
    (0..dp).map(|o| b[o] + w[o * hidden..(o + 1) * hidden].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect()
}

/// [rows, T] matrix whose row r averages span r (half-open), as Clef's `_mean_span`.
pub fn span_mask(spans: &[[u32; 2]], t: usize) -> Vec<f32> {
    let mut m = vec![0f32; spans.len() * t];
    for (r, s) in spans.iter().enumerate() {
        let (a, b) = (s[0] as usize, s[1] as usize);
        let w = 1.0 / (b - a) as f32;
        for p in a..b {
            m[r * t + p] = w;
        }
    }
    m
}

fn lm_head_mean(path: &Path, hidden: usize, vocab: usize, ids: &[u32]) -> Result<Vec<f32>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut acc = vec![0f32; hidden];
    let mut buf = vec![0u8; hidden * 2];
    for &id in ids {
        if id as usize >= vocab {
            return Err(format!("lexical id {id} >= vocab {vocab}"));
        }
        f.seek(SeekFrom::Start(id as u64 * hidden as u64 * 2)).map_err(|e| e.to_string())?;
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        for (a, c) in acc.iter_mut().zip(buf.chunks_exact(2)) {
            *a += half::f16::from_le_bytes([c[0], c[1]]).to_f32();
        }
    }
    let n = ids.len().max(1) as f32;
    Ok(acc.into_iter().map(|x| x / n).collect())
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn run_head(head: &ReadoutHead, assets: &mut HeadAssets, prompt_ids: &[u32], last_logits: &[f32], col: &Collector) -> Result<Vec<f32>, String> {
    match (head, assets) {
        (ReadoutHead::Verbalizer { token_ids }, HeadAssets::Verbalizer) => token_ids
            .iter()
            .map(|&t| last_logits.get(t as usize).copied().ok_or_else(|| format!("verbalizer id {t} outside logits ({})", last_logits.len())))
            .collect(),
        (ReadoutHead::Pointer { decide_pos, option_pos }, HeadAssets::Pointer { dp, hidden, q_w, q_b, k_w, k_b }) => {
            let hd = col.row(*decide_pos).ok_or("pointer: <decide> row was not collected")?;
            let q = affine(q_w, q_b, hd, *dp, *hidden);
            let scale = 1.0 / (*dp as f32).sqrt();
            option_pos
                .iter()
                .map(|&p| {
                    let ho = col.row(p).ok_or_else(|| format!("pointer: option row {p} was not collected"))?;
                    let k = affine(k_w, k_b, ho, *dp, *hidden);
                    Ok(k.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * scale)
                })
                .collect()
        }
        (ReadoutHead::Joint { questions }, HeadAssets::Joint { runtime, lm_head, vocab, hidden }) => {
            let t = col.len();
            if t != prompt_ids.len() {
                return Err(format!("joint: collected {t} rows for a {}-token prompt", prompt_ids.len()));
            }
            let mut spans: Vec<[u32; 2]> = questions.iter().map(|q| q.span).collect();
            let mut lexical = Vec::new();
            let mut option_q = Vec::new();
            for (qi, q) in questions.iter().enumerate() {
                for o in &q.options {
                    spans.push(o.span);
                    lexical.extend(lm_head_mean(lm_head, *hidden, *vocab, &prompt_ids[o.span[0] as usize..o.span[1] as usize])?);
                    option_q.push(qi as i64);
                }
            }
            let type_ids: Vec<i64> = questions.iter().map(|q| q.qtype as i64).collect();
            let n_opt = option_q.len();
            let mask = span_mask(&spans, t);
            let set = |rt: &mut Runtime, name: &str, dt: DType, shape: &[usize], data: &[u8]| rt.set_input(name, dt, shape, data).map_err(|e| format!("joint input {name}: {e}"));
            set(runtime, "hidden", DType::F32, &[t, *hidden], &le_f32(col.all()))?;
            set(runtime, "span_mask", DType::F32, &[spans.len(), t], &le_f32(&mask))?;
            set(runtime, "lexical", DType::F32, &[n_opt, *hidden], &le_f32(&lexical))?;
            set(runtime, "option_q", DType::I64, &[n_opt], &le_i64(&option_q))?;
            set(runtime, "type_ids", DType::I64, &[type_ids.len()], &le_i64(&type_ids))?;
            runtime.infer().map_err(|e| format!("joint infer: {e}"))?;
            let (_, _, bytes) = runtime.output(0).map_err(|e| format!("joint output: {e}"))?;
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        (h, _) => Err(format!("decision head {h:?} does not match this model's decision/ assets")),
    }
}

/// Ceiling for a FRAME_DECIDE / FRAME_DECISION body. A 73,728-token Kev readout is ~600 KB of JSON.
pub const MAX_DECISION_BODY_BYTES: usize = 16 << 20;

fn framed(json: Vec<u8>) -> Vec<u8> {
    let mut b = (json.len() as u32).to_be_bytes().to_vec();
    b.extend(json);
    b
}

fn unframe(b: &[u8]) -> Result<&[u8], String> {
    let n = u32::from_be_bytes(b.get(..4).ok_or("short decision body")?.try_into().unwrap()) as usize;
    if n > MAX_DECISION_BODY_BYTES {
        return Err(format!("decision body {n} bytes exceeds {MAX_DECISION_BODY_BYTES}"));
    }
    b.get(4..4usize.checked_add(n).ok_or("overflow")?).ok_or_else(|| "truncated decision body".to_string())
}

/// FRAME_DECIDE body: u32 BE length + JSON(DecisionReadout). The receiver still runs
/// `DecisionReadout::validate` before using it.
pub fn encode_readout(r: &DecisionReadout) -> Vec<u8> {
    framed(serde_json::to_vec(r).expect("readout serializes"))
}

pub fn decode_readout(b: &[u8]) -> Result<DecisionReadout, String> {
    serde_json::from_slice(unframe(b)?).map_err(|e| e.to_string())
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Ok(Vec<f32>),
    Err(String),
}

/// FRAME_DECISION body: u32 BE length + JSON `{"ok":[logits]}` or `{"err":"why"}`; the last rank
/// reports a head failure instead of hanging rank 0.
pub fn encode_result(r: &Result<Vec<f32>, String>) -> Vec<u8> {
    let o = match r {
        Ok(v) => Outcome::Ok(v.clone()),
        Err(e) => Outcome::Err(e.clone()),
    };
    framed(serde_json::to_vec(&o).expect("outcome serializes"))
}

pub fn decode_result(b: &[u8]) -> Result<Result<Vec<f32>, String>, String> {
    match serde_json::from_slice::<Outcome>(unframe(b)?).map_err(|e| e.to_string())? {
        Outcome::Ok(v) => Ok(Ok(v)),
        Outcome::Err(e) => Ok(Err(e)),
    }
}
```

Check `PluginConfig::default()` exists (`grep -n "impl Default for PluginConfig\|derive(.*Default" crates/cascadia-ov-genai-shim/src/lib.rs`). If it doesn't, construct the same `PluginConfig` that `qwen36.rs` uses for its stages (search `PluginConfig {` in `qwen36.rs`) and pass it into `HeadAssets::load` as an extra parameter.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-engine-openvino decision::`
Expected: PASS (7 tests; the joint runtime is not exercised: no OpenVINO in unit tests).

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-engine-openvino/src/decision.rs crates/cascadia-engine-openvino/src/lib.rs
git commit -m "feat(engine): decision head assets, hidden collector and frame bodies"
```

---

### Task 10: qwen35 engine serves decisions on a single box

**Files:**
- Modify: `crates/cascadia-engine-openvino/src/qwen36.rs`

**Interfaces:**
- Consumes: Task 1 (`DecisionReadout::{validate, warm_limit, cache_prefix}`), Task 9 (`Collector`, `HeadAssets`, `run_head`), `GenerationTask.decision`.
- Produces:
  - manifest key `final_hidden_output: bool` (default false), meaning the last stage has output 1 = final-norm hidden `[1, n, hidden]`
  - single-box decision tasks end with `Chunk::final_marker(..).with_prompt_tokens(n).with_finish_reason(Stop).with_decision_logits(v)`
  - `fn decision_admission(task: &GenerationTask) -> Option<Vec<u32>>` (pure, tested)
  - `fn decision_snapshot_boundaries(d: &DecisionReadout, cache_on: bool) -> Vec<usize>` (pure, tested)
  - `Qwen36Engine::admit_decision_check(&self, d: &DecisionReadout) -> Result<(), String>` (validate + capability; shared with Task 11's rank 0 and last rank)

Behaviour changes:
1. **`Manifest`:** add `#[serde(default)] final_hidden_output: bool`, threaded exactly like `last_logits_only` (builder field set in `load`, copied into the engine in `build`).
2. **Engine fields:**
   - `decision_assets: Option<decision::HeadAssets>`: loaded in the builder when `<model>/decision/head.json` exists **and** this process owns the last stage (`total <= 1` or `rank + 1 == total`).
   - `last_final_hidden: Option<Vec<f32>>` and `want_final_hidden: bool`. `chain_pass_inner` reads output 1 only while `want_final_hidden` is set, so chat traffic on a decision tree never pays the `[1, n, hidden]` copy.
   - `vocab: Option<usize>`, the bound `validate` checks ids against. It is read from the tree's `config.json` (`text_config.vocab_size`, else top-level `vocab_size`) on **every** rank, because pipeline ranks other than 0 have no tokenizer. It is read **tolerantly**, the same way `read_max_ctx` reads the window: a missing file or key logs a warning and leaves `None`. Chat loads never fail because of it; only decision tasks are refused (`decision_unsupported: tree config.json states no vocab_size`).
3. **`admitted_prompt_ids`:** `decision_admission(task)` takes priority, for both `submit`'s window guard and `step_local`.
4. **Admission** (`step_local`, before the warm-resume block):
   - `admit_decision_check`: `d.validate(vocab, self.max_ctx.unwrap_or(MAX_DECISION_PROMPT))` (`max_ctx` is only read on rank 0, so other ranks use the fixed bound), plus head assets present, plus `final_hidden_output` when the head needs hidden rows. A failure ends the task with `Chunk::error(id, "decision_unsupported: …")` for capability problems and `"decision_invalid: …"` for validation problems.
   - **Prefix cache:** a decision task never takes the kv_coord `take_warm` path. It restores at most `d.warm_limit()` tokens: `self.local_warm_or_cold(&prompt_ids[..d.warm_limit()])`. For Joint (Clef) the limit is 0, so the pass is always cold. For Verbalizer (JEV) it can reuse anything but the last token. For Pointer (Kev) it can reuse up to the first option row, which in practice means the state prefix.
   - **Snapshots:** the boundary list is `decision_snapshot_boundaries(d, enabled)` (just `cache_prefix`) instead of `chat_boundaries`. It then goes through the same `plan_snapshots` call, so budget and refresh-cost rules still apply. Decision passes therefore snapshot only the shared state prefix, never chat-style boundaries.
   - Set `self.want_final_hidden` on **every** admission, chat included: `task.decision.as_ref().is_some_and(|d| d.head.hidden_need() != HiddenNeed::None)`. A flag left over from a cancelled or refused decision therefore never makes later chat passes copy `final_hidden`.
5. **`run_span`:** after `chain_pass`, if the active task has a collector, `absorb(t0, n, &fh)`. An `Err` ends the task with `finalize_error("decision_invalid: …")`. Because of the warm limit above this cannot fire for a correct engine; it is the loud tripwire if one regresses.
6. **Post-prefill:** in `step_local`'s post-prefill `else` arm, a decision task calls `finish_decision()` instead of decoding.
7. **Reset:** `finish_decision`, `finalize_error`, `cancel` and the pipeline `InFrame::Reset` arm (Task 11b) all clear `want_final_hidden` and `last_final_hidden`. The per-admission assignment above is the backstop.

- [ ] **Step 1: Write the failing tests** (in the existing `#[cfg(test)]` module of `qwen36.rs`)

```rust
    fn decision_task(head: cascadia_types::ReadoutHead, cache_prefix: Option<u32>) -> GenerationTask {
        GenerationTask {
            task_id: "d".into(), prompt: "ignored".into(), max_tokens: 1, temperature: 0.0, logprobs: 0,
            sampling: Default::default(), enable_thinking: false, trust_remote_code: false,
            tenant: String::new(), resume_token_ids: None,
            decision: Some(cascadia_types::DecisionReadout { prompt_ids: vec![7, 8, 9, 10], head, cache_prefix }),
        }
    }

    #[test]
    fn decision_admission_uses_exact_ids_and_no_think_block() {
        let mut t = decision_task(cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] }, None);
        assert_eq!(decision_admission(&t), Some(vec![7, 8, 9, 10]));
        t.decision = None;
        assert_eq!(decision_admission(&t), None);
    }

    #[test]
    fn decision_snapshots_only_at_the_shared_prefix() {
        let t = decision_task(cascadia_types::ReadoutHead::Pointer { decide_pos: 3, option_pos: vec![2] }, Some(2));
        let d = t.decision.as_ref().unwrap();
        assert_eq!(decision_snapshot_boundaries(d, true), vec![2]);
        assert!(decision_snapshot_boundaries(d, false).is_empty(), "prefix cache disabled");
        let j = decision_task(cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] }, None);
        assert!(decision_snapshot_boundaries(j.decision.as_ref().unwrap(), true).is_empty());
    }

    #[test]
    fn manifest_reads_final_hidden_output() {
        let m = Manifest::parse(r#"{"arch":"qwen3_5","hidden_size":4096,"final_hidden_output":true,"stages":[]}"#).unwrap();
        assert!(m.final_hidden_output);
        let m = Manifest::parse(r#"{"arch":"qwen3_5","hidden_size":4096,"stages":[]}"#).unwrap();
        assert!(!m.final_hidden_output);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-engine-openvino decision_ manifest_reads_final_hidden`
Expected: compile errors.

- [ ] **Step 3: Implement**

Next to `admitted_prompt_ids` (line ~729):

```rust
/// Decision passes prefill the codec's exact ids: no tokenization, no think block.
fn decision_admission(task: &GenerationTask) -> Option<Vec<u32>> {
    task.decision.as_ref().map(|d| d.prompt_ids.clone())
}

/// A decision pass snapshots only where its request's passes share a prefix (Kev: the state);
/// chat-style `<|im_start|>` boundaries would only evict chat entries.
fn decision_snapshot_boundaries(d: &cascadia_types::DecisionReadout, cache_on: bool) -> Vec<usize> {
    match d.cache_prefix {
        Some(c) if cache_on => vec![c as usize],
        _ => Vec::new(),
    }
}
```

First lines of `admitted_prompt_ids`' body:

```rust
    if let Some(ids) = decision_admission(task) {
        return Ok(ids);
    }
```

In `Manifest`:

```rust
    /// Last stage exposes output 1 = final-norm hidden [1, n, hidden] for decision heads
    /// (exporter `--final-hidden`). Absent in generation-only trees (default false).
    #[serde(default)]
    final_hidden_output: bool,
```

The capability and validation check (an engine method, used here and by Task 11):

```rust
    /// Everything a decision pass needs before it touches a stage: a well-formed readout
    /// (ids < vocab, spans/positions inside the prompt: the readout may come from a peer)
    /// and, on the head-owning process, the assets and hidden output its head reads.
    fn admit_decision_check(&self, d: &cascadia_types::DecisionReadout) -> Result<(), String> {
        let vocab = self.vocab.ok_or("decision_unsupported: tree config.json states no vocab_size")?;
        d.validate(vocab, self.max_ctx.unwrap_or(cascadia_types::MAX_DECISION_PROMPT)).map_err(|e| format!("decision_invalid: {e}"))?;
        let owns_head = self.total <= 1 || self.rank + 1 == self.total;
        if owns_head {
            if self.decision_assets.is_none() {
                return Err("decision_unsupported: model has no decision/ head assets".into());
            }
            if d.head.hidden_need() != cascadia_types::HiddenNeed::None && !self.final_hidden_output {
                return Err("decision_unsupported: tree was exported without --final-hidden".into());
            }
        }
        Ok(())
    }
```

In `step_local`, right after `prompt_ids` is computed for the admitted task:

```rust
            if let Some(d) = &task.decision {
                if let Err(reason) = self.admit_decision_check(d) {
                    let id = task.task_id.clone();
                    return vec![(id.clone(), Chunk::error(id, reason))];
                }
            }
            // Every admission (chat too) sets it, so a stale flag can never outlive its task.
            self.want_final_hidden = task.decision.as_ref().is_some_and(|d| d.head.hidden_need() != cascadia_types::HiddenNeed::None);
```

Replace the head of the `let warm_prefix: usize = { … };` block (line ~2298) so decisions bypass kv_coord and bound the restore:

```rust
            let warm_prefix: usize = if let Some(d) = &task.decision {
                // Restore no further than the first row/logit the head reads (Joint: 0 = cold).
                self.local_warm_or_cold(&prompt_ids[..d.warm_limit()])
            } else {
                // … the existing kv_coord / local_warm_or_cold block, unchanged …
            };
```

and the `boundaries` computation (line ~2362):

```rust
            let boundaries = match &task.decision {
                Some(d) => decision_snapshot_boundaries(d, self.prefix_cache.enabled() && !self.stages_dirty),
                None => self
                    .im_start_id
                    .filter(|_| self.prefix_cache.enabled() && !self.stages_dirty)
                    .map(|im| crate::prefix_cache::chat_boundaries(&prompt_ids, im))
                    .unwrap_or_default(),
            };
```

`local_warm_or_cold(&prompt_ids[..0])` finds no prefix and resets cold, which is the Joint case. `plan_snapshots` already drops boundaries at or below `warm_prefix`.

In `chain_pass_inner`, before the stage loop: `let n_stages = self.stages.len(); let owns_head = self.total <= 1 || self.rank + 1 == self.total; let want_fh = self.want_final_hidden && self.final_hidden_output && owns_head; let mut final_hidden = None;`. Inside the loop, after `st.infer()` and the output(0) dtype check:

```rust
            if want_fh && j + 1 == n_stages {
                let (_, _, fh) = st.output(1).map_err(map_ov)?;
                final_hidden = Some(f32_from_le(&fh));
            }
```

After the loop: `self.last_final_hidden = final_hidden;` (a local avoids borrowing `self` while `self.stages` is iterated mutably).

In `run_span`, after `let out = self.chain_pass(&e, t0, t0 + n)?;`:

```rust
        if let Some(fh) = self.last_final_hidden.take() {
            if let Some(c) = self.active.as_mut().and_then(|t| t.collector.as_mut()) {
                c.absorb(t0, n, &fh).map_err(|e| EngineError::Backend(format!("decision_invalid: {e}")))?;
            }
        }
```

(`run_span`'s callers already turn an `Err` into `finalize_error`.)

`struct ActiveTask` gains `decision: Option<cascadia_types::DecisionReadout>` and `collector: Option<crate::decision::Collector>`. Every `ActiveTask { .. }` literal (step_local, step_pipe_first, and the three test literals at lines ~3464/3496/3579) sets them. step_pipe_first uses `decision: task.decision.clone(), collector: None`: rank 0 owns no head, but it must know the task is a decision so it collects the result instead of decoding (Task 11b step 6). step_local uses:

```rust
                collector: task.decision.as_ref().map(|d| crate::decision::Collector::new(d.head.hidden_need(), self.hidden)),
                decision: task.decision.clone(),
```

and the three test literals use `None, None`. Also clear `want_final_hidden` and `last_final_hidden` in `Qwen36Engine::cancel`.

In the post-prefill `else` arm of `step_local`, first thing:

```rust
            if t.decision.is_some() {
                return self.finish_decision();
            }
```

Next to `finalize`:

```rust
    /// End a decision pass: run the head on the collected state, emit ONE final chunk, reset.
    fn finish_decision(&mut self) -> Vec<(TaskId, Chunk)> {
        self.want_final_hidden = false;
        self.last_final_hidden = None;
        let Some(t) = self.active.as_ref() else { return Vec::new() };
        let d = t.decision.as_ref().expect("decision task");
        let empty = crate::decision::Collector::new(cascadia_types::HiddenNeed::None, self.hidden);
        let col = t.collector.as_ref().unwrap_or(&empty);
        let result = match self.decision_assets.as_mut() {
            Some(a) => crate::decision::run_head(&d.head, a, &d.prompt_ids, &t.logits, col),
            None => Err("decision_unsupported: no head assets".into()),
        };
        match result {
            Ok(logits) => {
                let n = t.prompt_ids.len() as u32;
                let id = t.task_id.clone();
                self.active = None;
                self.reset_all();
                vec![(id.clone(), Chunk::final_marker(id, "").with_prompt_tokens(n).with_finish_reason(FinishReason::Stop).with_decision_logits(logits))]
            }
            Err(e) => self.finalize_error(format!("decision head failed: {e}")),
        }
    }
```

Also clear `want_final_hidden` and `last_final_hidden` at the top of `finalize_error`.

Load assets in the builder's `load` (lines ~550–594), after the manifest is parsed:

```rust
        let owns_head = self.total <= 1 || self.rank + 1 == self.total;
        self.decision_assets = if owns_head && model_dir.join("decision").join("head.json").exists() {
            Some(crate::decision::HeadAssets::load(&model_dir, manifest.hidden()).map_err(EngineError::Backend)?)
        } else {
            None
        };
```

Read the vocabulary bound in the same place, on every rank and tolerantly (a chat tree without `config.json` must keep loading):

```rust
/// `text_config.vocab_size` (or top-level `vocab_size`) from the tree's config.json; `None` with a
/// warning when absent. Only decision admission needs it.
fn read_vocab(dir: &Path) -> Option<usize> {
    let raw = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "qwen35: no config.json; decision passes will be refused");
            return None;
        }
    };
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let v = cfg["text_config"]["vocab_size"].as_u64().or_else(|| cfg["vocab_size"].as_u64());
    if v.is_none() {
        warn!("qwen35: config.json states no vocab_size; decision passes will be refused");
    }
    v.map(|v| v as usize)
}
```

and `self.vocab = read_vocab(&dir);` next to the stage loop (outside the `rank == 0` block).

Move it, `final_hidden_output` and `vocab` into the engine in `build`. Use the variable names the surrounding code already uses for the model directory, rank and total.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-engine-openvino` and `cargo test -p cascadia-engine-openvino --features kv_coord`
Expected: PASS (existing + 3 new), both feature sets.

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-engine-openvino/src/qwen36.rs
git commit -m "feat(qwen35): single-box decision passes (exact-id prefill, bounded warm resume, head on final hidden)"
```

---

### Task 11: Pipeline capability bits and chunked body I/O

**Files:**
- Modify: `crates/cascadia-engine-openvino/src/qwen36.rs`

**Interfaces:**
- Consumes: Task 9's `MAX_DECISION_BODY_BYTES`.
- Produces (this task changes only the handshake and the body readers; the decision frames are Task 11b):
  - **Capability bits in the HELLO_ACK `pos` field.** Today `pos` carries the capture_v2 floor (0/1), combined with `min`. It becomes a bit set: `CAP_CAPTURE_V2 = 1` (bit 0, same values as today) and `CAP_DECISION = 2` (bit 1). The last rank replies `own_caps()`, each middle replies `own_caps() & downstream`, and rank 0 keeps `chain_caps = own_caps() & downstream`. An old rank replies 0 or 1, so bit 1 reads as "no decision frames" anywhere in the chain. An old rank 0 reading a new chain still takes `.min(CAPTURE_V2_CAP)` and keeps working. The HELLO payload and `validate_hello` are **unchanged**, so mixed chains handshake exactly as today.
  - **Frames.** All bodies are `[u32 BE byte length][JSON]`, read in `MAX_RAW_BYTES` chunks up to `MAX_DECISION_BODY_BYTES`. Over-ceiling bodies are drained so the stream stays framed.
    - `FRAME_DECIDE = 13` (header + readout body): rank 0 → chain, sent after RESET and before the prefill FORWARDs.
    - `FRAME_DECIDE_ACK = 14` (header only): chain → rank 0.
    - `FRAME_DECIDE_RUN = 15` (header only): rank 0 → chain, after the last prefill FORWARD.
    - `FRAME_DECISION = 16` (header + result body): last rank → rank 0, relayed verbatim by middles.
  - Rank 0 refuses a decision task with `decision_unsupported: a pipeline rank lacks decision frames (upgrade every rank)` when `chain_caps & CAP_DECISION == 0`.
  - `fn own_caps() -> u32`, `fn combine_caps(own: u32, down: u32) -> u32`, `fn chain_decision(caps: u32) -> bool` (pure, tested).
  - `trait RawRecv` implemented for both `ActivationServer` (the upstream end) and `ActivationClient` (the downstream end). Both already have `pub async fn recv_raw(&mut self, n: usize) -> TransportResult<Vec<u8>>` (`crates/cascadia-transport/src/lib.rs` ~1372 and ~1631), but no shared trait.
  - `async fn recv_body_chunked<G: RawRecv>(g, n, ceiling, what)`: the existing `recv_kv_body_chunked` moved out of `#[cfg(feature = "kv_coord")]`, made generic, with the ceiling as a parameter. The kv path calls it with `MAX_CARRY_BLOB_BYTES`.
  - `async fn recv_decision_body<G: RawRecv>(g, what)`, used by both ends.

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn capability_bits_degrade_to_old_ranks() {
        let own = own_caps();
        assert!(chain_decision(own), "a new build advertises decision frames");
        // an old last rank replies its capture floor (0 or 1): bit 1 is clear either way
        assert!(!chain_decision(combine_caps(own, 0)));
        assert!(!chain_decision(combine_caps(own, 1)));
        // all-new chain keeps both bits it has
        assert_eq!(combine_caps(own, own), own);
        // capture bit keeps today's meaning
        assert_eq!(combine_caps(own, 1) & CAP_CAPTURE_V2, own & CAP_CAPTURE_V2);
    }

    #[test]
    fn decision_frame_codes_are_distinct() {
        // CAPTURE/RESTORE codes (8..=12) exist only under the kv_coord feature (off by default).
        let decision = [FRAME_DECIDE, FRAME_DECIDE_ACK, FRAME_DECIDE_RUN, FRAME_DECISION];
        let mut all = vec![FRAME_HELLO, FRAME_HELLO_ACK, FRAME_HELLO_NAK, FRAME_RESET, FRAME_RESET_ACK, FRAME_FORWARD, FRAME_TOKEN];
        all.extend(decision);
        #[cfg(feature = "kv_coord")]
        all.extend([FRAME_CAPTURE, FRAME_CAPTURE_ACK, FRAME_RESTORE, FRAME_RESTORE_ACK, FRAME_CAPTURE_V2]);
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n);
        assert!(decision.iter().all(|c| !(8..=12).contains(c)), "decision codes must not reuse kv_coord codes");
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-engine-openvino capability_bits decision_frame_codes`
Expected: compile errors.

- [ ] **Step 3: Implement**

Constants and capability helpers next to the frame constants (line ~168):

```rust
/// Decision readout: rank 0 → chain, body = decision::encode_readout. Lockstep-acked.
const FRAME_DECIDE: u32 = 13;
const FRAME_DECIDE_ACK: u32 = 14;
/// After the prefill FORWARDs: run the head on the last rank.
const FRAME_DECIDE_RUN: u32 = 15;
/// Last rank → rank 0 (middles relay verbatim), body = decision::encode_result.
const FRAME_DECISION: u32 = 16;

/// HELLO_ACK `pos` is a capability bit set. Bit 0 keeps the pre-existing capture_v2 meaning
/// (old ranks send exactly 0 or 1 there), bit 1 = this rank speaks the decision frames.
const CAP_CAPTURE_V2: u32 = 1;
const CAP_DECISION: u32 = 2;

fn own_caps() -> u32 {
    (CAPTURE_V2_CAP & CAP_CAPTURE_V2) | CAP_DECISION
}

fn combine_caps(own: u32, down: u32) -> u32 {
    own & down
}

fn chain_decision(caps: u32) -> bool {
    caps & CAP_DECISION != 0
}
```

Handshake changes:
- Middle and last handler (line ~1940): replace `cap = cap.min(down_cap)` with `cap = combine_caps(cap, down_cap)`, where `cap` starts at `own_caps()` instead of `CAPTURE_V2_CAP`. Store `self.chain_caps = cap; self.chain_capture_v2 = (cap & CAP_CAPTURE_V2).min(CAPTURE_V2_CAP);`.
- `handshake_a` (line ~1233): `let caps = combine_caps(own_caps(), down_cap); self.chain_caps = caps; self.chain_capture_v2 = (caps & CAP_CAPTURE_V2).min(CAPTURE_V2_CAP);`, and log `caps`.
- Add the engine field `chain_caps: u32` (0 in the constructor at lines ~645/3208).

Both link ends need the chunked reader: the upstream reader (`ActivationServer`) for FRAME_DECIDE, and rank 0 or a middle's downstream link (`ActivationClient`) for FRAME_DECISION. Add a private trait over the two existing methods:

```rust
/// The one method chunked body reads need, on both ends of a pipeline link.
trait RawRecv {
    fn recv_raw(&mut self, n: usize) -> impl std::future::Future<Output = TransportResult<Vec<u8>>> + Send;
}

impl RawRecv for ActivationServer {
    fn recv_raw(&mut self, n: usize) -> impl std::future::Future<Output = TransportResult<Vec<u8>>> + Send {
        ActivationServer::recv_raw(self, n)
    }
}

impl RawRecv for ActivationClient {
    fn recv_raw(&mut self, n: usize) -> impl std::future::Future<Output = TransportResult<Vec<u8>>> + Send {
        ActivationClient::recv_raw(self, n)
    }
}
```

Move `recv_kv_body_chunked` (line ~217) out of the `kv_coord` gate as:

```rust
/// Read exactly `n` body bytes in ≤`MAX_RAW_BYTES` chunks, accepting at most `ceiling`.
/// Over-ceiling lengths are drained (discarded) to stay framed, then rejected.
async fn recv_body_chunked<G: RawRecv>(g: &mut G, n: usize, ceiling: usize, what: &str) -> Result<Vec<u8>, TransportError> {
    // … the existing body, with `MAX_CARRY_BLOB_BYTES` replaced by `ceiling` …
}
```

The kv_coord call sites become `recv_body_chunked(&mut *g, n, MAX_CARRY_BLOB_BYTES, "...")`. Readers of the framed decision bodies call:

```rust
async fn recv_decision_body<G: RawRecv>(g: &mut G, what: &str) -> Result<Vec<u8>, TransportError> {
    let lb = g.recv_raw(4).await?;
    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
    let mut body = lb;
    body.extend(recv_body_chunked(g, n, crate::decision::MAX_DECISION_BODY_BYTES, what).await?);
    Ok(body) // still length-prefixed: decode_readout/decode_result and relays take it whole
}
```

Every caller holds a tokio `MutexGuard`, so pass `&mut *g`, **not** `&mut g`, to `recv_body_chunked` and `recv_decision_body`, on both the upstream (`ActivationServer`) and downstream (`ActivationClient`) links. A generic `G: RawRecv` does not deref-coerce: `&mut g` would infer `G = MutexGuard<…>`, which has no impl (E0277). Add `TransportResult` to qwen36.rs's `use cascadia_transport::{…}` list.

Senders write the prefixed body with `send_raw` in `MAX_RAW_BYTES` slices (`for c in body.chunks(cascadia_transport::MAX_RAW_BYTES) { g.send_raw(c).await?; }`), matching how the kv path sends large bodies.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-engine-openvino` and `cargo test -p cascadia-engine-openvino --features kv_coord`
Expected: PASS, both feature sets (the kv_coord build exercises the moved, now generic `recv_body_chunked`; the handshake change must leave every existing pipeline test green).

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-engine-openvino/src/qwen36.rs
git commit -m "feat(qwen35): HELLO_ACK capability bits and generic chunked body reads"
```

---

### Task 11b: Pipeline decision frames

**Files:**
- Modify: `crates/cascadia-engine-openvino/src/qwen36.rs`

**Interfaces:**
- Consumes: Task 9 frame bodies (`encode_readout`/`decode_readout`, `encode_result`/`decode_result`), Task 10 (`admit_decision_check`, `finish_decision`, `want_final_hidden`), Task 11 (`chain_caps`, `chain_decision`, `recv_decision_body`).
- Produces:
  - the four frames on every rank role (rank 0, middle, last), as described under Task 11's Produces;
  - last-rank engine fields `pending_decision` and `last_logits: Vec<f32>`;
  - `fn take_pending(pending: &mut Option<(u32, Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String>)>, epoch: u32) -> Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String>` (pure, tested): the last rank's DECIDE_RUN lookup.

Frame correctness end to end (lockstep, epochs, relays) is checked by the two-rank smoke test in Task 15 Step 6, the first point where a 2-stage decision tree exists.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn take_pending_matches_epoch_and_surfaces_errors() {
        use cascadia_types::{DecisionReadout, HiddenNeed, ReadoutHead};
        let r = DecisionReadout { prompt_ids: vec![1, 2], head: ReadoutHead::Verbalizer { token_ids: vec![1] }, cache_prefix: None };
        let mut p = Some((7, Ok((r.clone(), crate::decision::Collector::new(HiddenNeed::None, 4)))));
        assert!(take_pending(&mut p, 8).is_err(), "stale epoch");
        assert!(p.is_none(), "taken even on mismatch so it cannot leak into the next task");
        let mut p = Some((7, Ok((r, crate::decision::Collector::new(HiddenNeed::None, 4)))));
        assert!(take_pending(&mut p, 7).is_ok());
        let mut p: Option<(u32, Result<(DecisionReadout, crate::decision::Collector), String>)> = Some((7, Err("decision_invalid: x".into())));
        assert_eq!(take_pending(&mut p, 7).unwrap_err(), "decision_invalid: x");
        assert!(take_pending(&mut None, 7).is_err(), "nothing pending");
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cascadia-engine-openvino take_pending`
Expected: compile error.

- [ ] **Step 3: Implement**

```rust
/// The last rank's DECIDE_RUN lookup: always takes the entry (a stale one must not leak into
/// the next task), and turns a missing, stale or invalid readout into the error rank 0 reports.
fn take_pending(
    pending: &mut Option<(u32, Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String>)>,
    epoch: u32,
) -> Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String> {
    match pending.take() {
        Some((e, r)) if e == epoch => r,
        Some((e, _)) => Err(format!("decision pending for epoch {e}, run asked for {epoch}")),
        None => Err("no decision pending for this epoch".into()),
    }
}
```

**`InFrame`** gains `Decide { epoch: u32, body: Vec<u8> }` and `DecideRun(u32)`. In `step_pipe_relay`'s reader `match kind`:

```rust
                FRAME_DECIDE => Ok(InFrame::Decide { epoch, body: recv_decision_body(&mut *g, "decide body").await? }),
                FRAME_DECIDE_RUN => Ok(InFrame::DecideRun(epoch)),
```

**Downstream handling** (middle and last ranks). New engine fields: `pending_decision: Option<(u32, Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String>)>` and `last_logits: Vec<f32>` (both empty in the constructors at lines ~645/3208). `InFrame::Reset` clears both, along with `want_final_hidden` and `last_final_hidden`.
- `InFrame::Decide { epoch, body }`:
  - Stale epoch (`!= self.peer_epoch`): drop it with a `warn!`, the same way FORWARD does.
  - **Middle:** `forward_decide_downstream(epoch, &body)` (send the header plus the chunked body, then expect a `FRAME_DECIDE_ACK` with the same epoch), then ack upstream.
  - **Last:** `let r = crate::decision::decode_readout(&body).and_then(|d| self.admit_decision_check(&d).map(|_| d));`. Store `(epoch, r.map(|d| (Collector::new(d.head.hidden_need(), self.hidden), d)))` (put the pair in the struct's field order). Set `self.want_final_hidden` from the readout, then ack upstream. An invalid readout is still acked, so the lockstep stays intact, and its error is reported at DECIDE_RUN.
- `InFrame::Forward` on the last rank: after `chain_pass`, if `pending_decision` is `Some((epoch, Ok(..)))` and `self.last_final_hidden.take()` is `Some(fh)`, call `collector.absorb(t0, n, &fh)`. On `Err(e)`, replace the entry with `Err(e)`. Keep this FORWARD's last-position logits row (the `logits` the existing code argmaxes) in `self.last_logits`. The TOKEN reply is unchanged; rank 0 ignores tokens during prefill.
- `InFrame::DecideRun(epoch)`:
  - **Middle:** relay the header downstream, read `FRAME_DECISION` plus `recv_decision_body`, and write both upstream unchanged.
  - **Last:** compute `let out: Result<Vec<f32>, String> = take_pending(&mut self.pending_decision, epoch).and_then(|(d, col)| crate::decision::run_head(&d.head, assets, &d.prompt_ids, &self.last_logits, &col));`. Reply with the `FRAME_DECISION` header plus `encode_result(&out)`, clear `want_final_hidden` and `last_logits`, and call `reset_all()`.

**Rank 0** (`step_pipe_first`, line ~1486):
1. **Tokenize through `admitted_prompt_ids`.** Replace the inline `tokenizer.encode(task.prompt.as_str(), true)` and think-block lines (lines ~1510–1527) with `let mut prompt_ids = match admitted_prompt_ids(tokenizer, &task) { … same error arm … };`. That function performs the identical encode and think-block append, so chat behaviour is unchanged. The Option-B resume-append block that follows stays as it is; a decision task carries no resume ids. Decision tasks now prefill the codec's ids.
2. **Check before anything else.** For a decision task, run `self.admit_decision_check(d)` (validate, since rank 0 doesn't own the head) and require `chain_decision(self.chain_caps)`. On failure, end the task with that `Chunk::error`.
3. **No warm resume.** A decision task skips the kv_coord `take_warm` block (`warm_prefix = 0`). Pipeline ranks have no local prefix cache, so Kev and JEV re-prefill per question in pipeline mode; only the single box gets state reuse.
4. **Announce the readout.** After `reset_exchange()`, call `self.decide_exchange(&readout)`: a copy of `reset_exchange` that sends the `FRAME_DECIDE` header plus the encoded readout body and expects `FRAME_DECIDE_ACK` with the same epoch.
5. **Prefill** with the normal FORWARDs.
6. **Collect the result instead of decoding.** At the point where a chat task would start decoding, call `self.decide_run_exchange() -> EngineResult<Result<Vec<f32>, String>>`. It is also patterned on `reset_exchange`: header out, `FRAME_DECISION` header in (check kind and epoch), `recv_decision_body`, `decode_result`.
7. **Emit.** `Ok(Ok(v))` gives the final chunk as in `finish_decision`. `Ok(Err(why))` gives `finalize_error(why)` (the handler maps `decision_unsupported`/`decision_invalid`). `Err(wire)` gives the existing wire-failure path.

- [ ] **Step 4: Run tests**

Run: `cargo test -p cascadia-engine-openvino` and `cargo test -p cascadia-engine-openvino --features kv_coord`
Expected: PASS, both feature sets.

- [ ] **Step 5: Commit**

```bash
git add crates/cascadia-engine-openvino/src/qwen36.rs
git commit -m "feat(qwen35): pipeline decision frames (DECIDE/RUN/DECISION) on rank 0, middles and last rank"
```

---

### Task 12: Low-memory int8 weight swap and LoRA graft (exporter)

Port of `experiments/jev-spike/swapq.py`, which was verified bit-exact against NNCF for int8. It runs per stage, inside the exporter's stage loop, so peak memory is one stage.

**Files:**
- Create: `tools/qwen36_surgery/weight_swap.py`
- Test: `tools/tests/test_weight_swap.py`

**Interfaces:**
- Produces:
  - `q_asym(w, bits, axis) -> (q uint8, zp uint8, scale f16)`
  - `q_sym8(w) -> (q int8, scale f16)`
  - `TRANSFORMS` (name → fn) and `FLOAT_KIND_TRANSFORM` (hf suffix → transform name: `input_layernorm/post_attention_layernorm/q_norm/k_norm/norm` → `1+w`, `A_log` → `exp`, `dt_bias` and `linear_attn.norm` → `w`)
  - `class Weights(dir)` with `.get(name) -> np.ndarray f32`, `.rows(name, a, b)`
  - `swap_stage(model, weights) -> dict` (a report `{"proj": {layer: n}, "conv": {layer: n}, "float": {layer_or_"final": [kinds]}}`)
  - `check_swap(report, layers: range, layer_types: list[str], last: bool) -> None`: raises if any layer of the stage missed a projection, its conv (linear layers), or an expected float constant. A renamed node on another base IR (e.g. Qwen3.8-27B) would otherwise silently keep stock weights.
  - `lora_targets(adapter_file) -> int` and `lora_scale_from_config(adapter_file) -> float`: `lora_alpha / r` from the `adapter_config.json` beside the adapter (`lora_alpha / sqrt(r)` with `use_rslora`). Raises when `rank_pattern` or `alpha_pattern` is non-empty, since one global scale would be wrong.
  - `swap_embeddings(src_xml, weights, out_dir)`
  - `graft_lora(model, adapter_file, scale) -> int` (branches added)
  - `add_final_hidden_output(model) -> None`

- [ ] **Step 1: Write the failing tests** (`tools/tests/test_weight_swap.py`)

```python
"""weight_swap: quantizers, transform table, LoRA graft and final-hidden output.

    python -m pytest tools/tests/test_weight_swap.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
import weight_swap as ws  # noqa: E402


def test_int8_asym_per_channel_formula():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 64)).astype(np.float32)
    q, zp, s = ws.q_asym(w, 8, axis=-1)
    assert q.dtype == np.uint8 and zp.shape == (4, 1) and s.dtype == np.float16
    lo, hi = w.min(-1, keepdims=True), w.max(-1, keepdims=True)
    exp_s = ((hi - lo) / 255).astype(np.float16)
    np.testing.assert_array_equal(s, exp_s)
    deq = (q.astype(np.float32) - zp) * s.astype(np.float32)
    assert np.abs(deq - w).max() <= float(s.astype(np.float32).max())


def test_sym8_rows():
    w = np.array([[1.0, -2.0], [0.0, 0.0]], dtype=np.float32)
    q, s = ws.q_sym8(w)
    np.testing.assert_array_equal(q[0], [64, -127])
    assert s[1, 0] > 0  # zero row keeps a positive scale


def test_check_swap_requires_every_layer_kind():
    full = {"proj": {0: 4, 1: 7}, "conv": {0: 1},
            "float": {0: ["input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.A_log",
                          "linear_attn.dt_bias", "linear_attn.norm.weight"],
                      1: ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                          "self_attn.k_norm.weight"],
                      "final": ["norm.weight"]}}
    ws.check_swap(full, range(0, 2), ["linear_attention", "full_attention"], last=True)
    missing_conv = {**full, "conv": {}}
    with pytest.raises(ValueError, match="conv"):
        ws.check_swap(missing_conv, range(0, 2), ["linear_attention", "full_attention"], last=True)
    no_final = {**full, "float": {k: v for k, v in full["float"].items() if k != "final"}}
    with pytest.raises(ValueError, match="norm"):
        ws.check_swap(no_final, range(0, 2), ["linear_attention", "full_attention"], last=True)


def test_lora_scale_comes_from_adapter_config(tmp_path):
    import json
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64, "rank_pattern": {}, "alpha_pattern": {}}))
    assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 2.0
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "use_rslora": True}))
    assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 8.0
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "rank_pattern": {"q_proj": 8}}))
    with pytest.raises(ValueError, match="rank_pattern"):
        ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors"))


def test_float_transform_table():
    w = np.array([0.5, -0.25], dtype=np.float32)
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["input_layernorm.weight"]](w), 1 + w)
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["linear_attn.A_log"]](w), np.exp(w))
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["linear_attn.dt_bias"]](w), w)


def test_graft_lora_adds_scaled_branch():
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    x = ops.parameter([1, 3], np.float32, name="x")
    w = ops.constant(np.eye(3, dtype=np.float32))
    mm = ops.matmul(x, w, False, True)
    mm.set_friendly_name("__module.model.model.language_model.layers.0.mlp.down_proj/ov_ext::linear/MatMul")
    model = ov.Model([ops.result(mm)], [x], "t")
    a = torch.ones(1, 3)
    b = torch.full((3, 1), 0.5)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "lora_test.safetensors")
    save_file({"base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": a,
               "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": b}, path)
    assert ws.graft_lora(model, path, 2.0) == 1
    out = ov.Core().compile_model(model, "CPU")([np.array([[1.0, 2.0, 3.0]], np.float32)])[0]
    # base = x ; lora = 2 * (x·1)·0.5 = sum(x) = 6
    np.testing.assert_allclose(out, [[7.0, 8.0, 9.0]], rtol=1e-6)


def test_add_final_hidden_output_exposes_lm_head_input():
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    x = ops.parameter([1, 2, 4], np.float32, name="x")
    norm = ops.multiply(x, ops.constant(np.float32(2.0)))
    mm = ops.matmul(norm, ops.constant(np.ones((5, 4), np.float32)), False, True)
    mm.set_friendly_name("__module.model.lm_head/ov_ext::linear/MatMul")
    model = ov.Model([ops.result(mm)], [x], "t")
    ws.add_final_hidden_output(model)
    assert len(model.outputs) == 2
    assert "final_hidden" in model.outputs[1].get_names()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tools/tests/test_weight_swap.py -v`
Expected: `ModuleNotFoundError: weight_swap`.

- [ ] **Step 3: Implement `tools/qwen36_surgery/weight_swap.py`.** Port `experiments/jev-spike/swapq.py` with these exact contents:

```python
"""Low-memory weight swap + LoRA graft for Qwen3.5-family stage IRs.

Intel's stock IRs (OpenVINO/Qwen3.5-9B-int8-ov, OpenVINO/Qwen3.8-27B-int8-ov) have the exact graph of any
fine-tune or sibling of the same architecture. For one stage (cut by export_qwen36_moe.extract_stage) every
weight constant is replaced with the target checkpoint's tensor quantized in the stock constant's scheme:
  u8 [out, in] + u8 zp + f16 scale  -> int8 asymmetric per output channel (bit-exact vs NNCF)
  u4 [out, G, 128] + u4 zp + f16    -> int4 asymmetric group 128 (NNCF data-free except .5 ties)
  float constants (norms, dt_bias, A)-> via FLOAT_KIND_TRANSFORM, calibrated against the stock pair
Embeddings (i8 symmetric per row) are swapped in row chunks. Peak memory: one stage.
"""
import json
import re
from pathlib import Path

import numpy as np


def q_asym(w, bits, axis):
    hi = 2 ** bits - 1
    lo_v, hi_v = w.min(axis=axis, keepdims=True), w.max(axis=axis, keepdims=True)
    scale = (hi_v - lo_v) / hi
    scale = np.where(scale == 0, np.finfo(np.float32).eps, scale)
    zp = np.clip(np.round(-lo_v / scale), 0, hi)
    q = np.clip(np.round(w / scale) + zp, 0, hi)
    return q.astype(np.uint8), zp.astype(np.uint8), scale.astype(np.float16)


def q_sym8(w):
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, np.finfo(np.float32).eps, scale)
    return np.clip(np.round(w / scale), -128, 127).astype(np.int8), scale.astype(np.float16)


TRANSFORMS = {"w": lambda w: w, "1+w": lambda w: 1.0 + w, "exp": np.exp}
FLOAT_KIND_TRANSFORM = {
    "input_layernorm.weight": "1+w",
    "post_attention_layernorm.weight": "1+w",
    "self_attn.q_norm.weight": "1+w",
    "self_attn.k_norm.weight": "1+w",
    "norm.weight": "1+w",
    "linear_attn.A_log": "exp",
    "linear_attn.dt_bias": "w",
    "linear_attn.norm.weight": "w",
}
FLOAT_KINDS = {
    r"layers\.(\d+)\.input_layernorm/aten::mul/Multiply_1$": "input_layernorm.weight",
    r"layers\.(\d+)\.post_attention_layernorm/aten::mul/Multiply_1$": "post_attention_layernorm.weight",
    r"layers\.(\d+)\.linear_attn\.norm/aten::mul/Multiply_1$": "linear_attn.norm.weight",
    r"layers\.(\d+)\.linear_attn/aten::add/Add_2$": "linear_attn.dt_bias",
    r"layers\.(\d+)\.linear_attn/aten::neg/Multiply_1$": "linear_attn.A_log",
    r"layers\.(\d+)\.self_attn\.q_norm/aten::mul/Multiply_1$": "self_attn.q_norm.weight",
    r"layers\.(\d+)\.self_attn\.k_norm/aten::mul/Multiply_1$": "self_attn.k_norm.weight",
    r"language_model\.norm/aten::mul/Multiply_1$": "norm.weight",
}
PROJ = re.compile(r"language_model\.layers\.(\d+)\.([\w.]+)\._openvino_orig_weight(/zero_point|/scale)?$")
CONV = re.compile(r"language_model\.layers\.(\d+)\.linear_attn/aten::_convolution/Reshape(/zero_point|/scale)?$")
SLOT = {None: "w", "/zero_point": "zp", "/scale": "s"}


class Weights:
    """HF safetensors checkpoint, read tensor by tensor (text-model names relative to the layer stack)."""

    def __init__(self, d):
        from safetensors import safe_open  # noqa: F401  (import check)
        self.d = Path(d)
        self.idx = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]
        self.prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in self.idx) else "model."
        self._f = {}

    def _file(self, name):
        from safetensors import safe_open
        f = self.idx[name]
        if f not in self._f:
            self._f[f] = safe_open(self.d / f, "pt")
        return self._f[f]

    def get(self, name):
        n = self.prefix + name
        return self._file(n).get_tensor(n).float().numpy()

    def rows(self, name, a, b):
        n = self.prefix + name
        return self._file(n).get_slice(n)[a:b].float().numpy()


def _classify(model):
    import openvino as ov
    proj, conv, floats = {}, {}, []
    for op in model.get_ops():
        if op.get_type_name() != "Constant":
            continue
        nm = op.get_friendly_name()
        m = PROJ.search(nm)
        if m and "lm_head" not in nm:
            proj.setdefault((int(m.group(1)), m.group(2)), {})[SLOT[m.group(3)]] = op
            continue
        m = CONV.search(nm)
        if m:
            conv.setdefault(int(m.group(1)), {})[SLOT[m.group(2)]] = op
            continue
        if op.get_element_type() not in (ov.Type.f32, ov.Type.f16) or int(np.prod(list(op.get_output_shape(0)) or [1])) <= 1:
            continue
        for ti in op.output(0).get_target_inputs():
            cn = ti.get_node().get_friendly_name()
            for pat, hf in FLOAT_KINDS.items():
                mm = re.search(pat, cn)
                if mm:
                    floats.append((op, (f"layers.{mm.group(1)}." if mm.groups() else "") + hf))
                    break
    return proj, conv, floats


def _replace(op, arr, et):
    from openvino import opset13 as ops
    new = ops.constant(arr, dtype=et)
    new.set_friendly_name(op.get_friendly_name())
    for t in list(op.output(0).get_target_inputs()):
        t.replace_source_output(new.output(0))


def swap_stage(model, weights):
    """Replace every weight constant of a cut stage with `weights` (a Weights). Returns a report
    of what was swapped per layer, for `check_swap`."""
    import openvino as ov
    proj, conv, floats = _classify(model)
    report = {"proj": {}, "conv": {}, "float": {}}
    for (layer, path), c in proj.items():
        w = weights.get(f"layers.{layer}.{path}.weight")
        shp = list(c["w"].get_output_shape(0))
        if c["w"].get_element_type() == ov.Type.u4:
            q, zp, s = q_asym(w.reshape(shp), 4, axis=-1)
            _replace(c["w"], q, ov.Type.u4)
            _replace(c["zp"], zp, ov.Type.u4)
        else:
            q, zp, s = q_asym(w, 8, axis=-1)
            _replace(c["w"], q, ov.Type.u8)
            _replace(c["zp"], zp, ov.Type.u8)
        _replace(c["s"], s, ov.Type.f16)
        report["proj"][layer] = report["proj"].get(layer, 0) + 1
    for layer, c in conv.items():
        shp = list(c["w"].get_output_shape(0))
        w = weights.get(f"layers.{layer}.linear_attn.conv1d.weight").reshape(shp[0], -1)
        q, zp, s = q_asym(w, 8, axis=-1)
        _replace(c["w"], q.reshape(shp), ov.Type.u8)
        _replace(c["zp"], zp.reshape(list(c["zp"].get_output_shape(0))), ov.Type.u8)
        _replace(c["s"], s.reshape(list(c["s"].get_output_shape(0))), ov.Type.f16)
        report["conv"][layer] = report["conv"].get(layer, 0) + 1
    for op, hf in floats:
        kind = re.sub(r"layers\.\d+\.", "", hf)
        v = TRANSFORMS[FLOAT_KIND_TRANSFORM[kind]](weights.get(hf)).astype(np.float32)
        _replace(op, v.reshape(list(op.get_output_shape(0))), op.get_element_type())
        m = re.match(r"layers\.(\d+)\.", hf)
        report["float"].setdefault(int(m.group(1)) if m else "final", []).append(kind)
    return report


LAYER_FLOATS = {
    "linear_attention": {"input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.A_log",
                         "linear_attn.dt_bias", "linear_attn.norm.weight"},
    "full_attention": {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                       "self_attn.k_norm.weight"},
}


def check_swap(report, layers, layer_types, last):
    """Every layer of the stage swapped its projections, its conv (linear layers) and every float
    kind its type carries; the last stage also swapped the final norm."""
    for i in layers:
        t = layer_types[i]
        if not report["proj"].get(i):
            raise ValueError(f"layer {i}: no projection swapped")
        if t == "linear_attention" and not report["conv"].get(i):
            raise ValueError(f"layer {i}: conv1d not swapped")
        missing = LAYER_FLOATS[t] - set(report["float"].get(i, []))
        if missing:
            raise ValueError(f"layer {i} ({t}): float constants not swapped: {sorted(missing)}")
    if last and "norm.weight" not in report["float"].get("final", []):
        raise ValueError("final norm not swapped")


def lora_targets(adapter_file):
    from safetensors import safe_open
    with safe_open(adapter_file, "pt") as f:
        return sum(1 for k in f.keys() if k.endswith(".lora_A.weight"))


def lora_scale_from_config(adapter_file):
    """PEFT scale from adapter_config.json beside the adapter: alpha/r (alpha/sqrt(r) with rsLoRA)."""
    cfg = json.loads((Path(adapter_file).parent / "adapter_config.json").read_text())
    for k in ("rank_pattern", "alpha_pattern"):
        if cfg.get(k):
            raise ValueError(f"adapter_config.json has a non-empty {k}; one global LoRA scale would be wrong")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r


def swap_embeddings(src_xml, weights, out_dir, chunk=16384):
    import openvino as ov
    model = ov.Core().read_model(src_xml)
    consts = {op.get_friendly_name(): op for op in model.get_ops() if op.get_type_name() == "Constant"}
    wop = next(op for nm, op in consts.items() if nm.endswith("_openvino_orig_weight"))
    sop = next(op for nm, op in consts.items() if nm.endswith("_openvino_orig_weight/scale"))
    rows, hid = list(wop.get_output_shape(0))
    q = np.empty((rows, hid), dtype=np.int8)
    s = np.empty((rows, 1), dtype=np.float16)
    for a in range(0, rows, chunk):
        q[a:a + chunk], s[a:a + chunk] = q_sym8(weights.rows("embed_tokens.weight", a, min(rows, a + chunk)))
    _replace(wop, q, ov.Type.i8)
    _replace(sop, s, ov.Type.f16)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(Path(out_dir) / "openvino_text_embeddings_model.xml"), compress_to_fp16=False)


def graft_lora(model, adapter_file, scale):
    """y += scale * (x A^T) B^T on every LoRA-targeted MatMul in `model` (fp32 constants, LoRA term
    first in the Add: OpenVINO's ARM CPU plugin produced NaN with fp16 constants / base-first adds)."""
    from openvino import opset13 as ops
    from safetensors.torch import load_file
    pairs = {}
    for k, v in load_file(adapter_file).items():
        m = re.match(r"base_model\.model\.(?:model\.)?(.+)\.lora_([AB])\.weight$", k)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = v.float().numpy()
    n = 0
    for op in list(model.get_ordered_ops()):
        if op.get_type_name() != "MatMul":
            continue
        m = re.search(r"__module\.(?:model\.)*(?:language_model\.)?((?:layers\.\d+\.[\w.]+)|lm_head)/(?:aten|ov_ext)::linear/MatMul$",
                      op.get_friendly_name())
        if not m or m.group(1) not in pairs:
            continue
        a, b = pairs[m.group(1)]["A"], pairs[m.group(1)]["B"]
        x = op.input_value(0)
        xa = ops.matmul(x, ops.constant(np.ascontiguousarray(a.T, dtype=np.float32)), False, False)
        xab = ops.matmul(xa, ops.constant(np.ascontiguousarray(b.T * scale, dtype=np.float32)), False, False)
        consumers = list(op.output(0).get_target_inputs())
        added = ops.add(xab, op.output(0))
        for c in consumers:
            c.replace_source_output(added.output(0))
        n += 1
    model.validate_nodes_and_infer_types()
    return n


def add_final_hidden_output(model):
    """Expose the lm_head MatMul's input (= final-norm hidden, all positions) as output 1 `final_hidden`."""
    from openvino import opset13 as ops
    mm = next(op for op in model.get_ops()
              if op.get_type_name() == "MatMul" and op.get_friendly_name().endswith("lm_head/ov_ext::linear/MatMul"))
    res = ops.result(mm.input_value(0))
    res.output(0).set_names({"final_hidden"})
    model.add_results([res])
    model.validate_nodes_and_infer_types()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tools/tests/test_weight_swap.py -v`
Expected: PASS (OpenVINO tests skip without OpenVINO; they must PASS in the dev venv that has it).

Note: the float-kind table was fitted on the Qwen3.5-9B stock IR. Task 22's first export step runs `check_swap` on every Qwen3.8-27B stage, so a node-name difference stops the export instead of producing a half-swapped model.

- [ ] **Step 5: Commit**

```bash
git add tools/qwen36_surgery/weight_swap.py tools/tests/test_weight_swap.py
git commit -m "feat(tools): low-memory int8 weight swap, LoRA graft and final_hidden output"
```

---

### Task 13: Exporter options for decision trees

**Files:**
- Modify: `tools/qwen36_surgery/export_qwen36_moe.py` (`run_export`, `main`)
- Test: `tools/tests/test_decision_export.py` (first part)

**Interfaces:**
- Consumes: `weight_swap.{Weights, swap_stage, check_swap, swap_embeddings, graft_lora, lora_targets, lora_scale_from_config, add_final_hidden_output}`.
- Produces: `run_export(model_dir, output_dir, num_stages=2, validate=False, layer_split=None, weights=None, lora=None, lora_scale=None, final_hidden=False)`; the manifest gains `"final_hidden_output": bool`; CLI flags `--weights DIR`, `--lora FILE`, `--lora-scale F` (override only), `--final-hidden`.
- Behaviour:
  - `lora_scale=None` means the scale comes from `adapter_config.json`; an explicit value that disagrees prints a warning.
  - Every swapped stage passes `check_swap`.
  - After the loop, the total number of grafted branches must equal `lora_targets(lora)`, or the export fails.
- **Memory:** peak ≈ one stage plus its swapped constants. A single-box 9B tree is therefore exported as **8 stages** (~1 GB of int8 each, under the 7 GB cap). The engine compiles every manifest stage when `total = 1`, so the stage count is independent of serving. Pipeline trees use one stage per rank. For JEV-27B (stock IR + LoRA graft, no swap) that stays small; for Clef-27B with `--weights` each of the 2 stages is ~13 GB of new constants, so that export needs a ~16 GB machine (stated in Task 22).

- [ ] **Step 1: Write the failing test** (`tools/tests/test_decision_export.py`)

```python
"""Exporter decision options and decision.json/assets writers.

    python -m pytest tools/tests/test_decision_export.py -v
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import export_qwen36_moe as ex  # noqa: E402


def test_run_export_accepts_decision_options():
    params = inspect.signature(ex.run_export).parameters
    for name in ("weights", "lora", "lora_scale", "final_hidden"):
        assert name in params, name
    assert params["final_hidden"].default is False


def test_lora_scale_defaults_to_adapter_config():
    assert inspect.signature(ex.run_export).parameters["lora_scale"].default is None


def test_manifest_records_final_hidden():
    spec = ex.ModelSpec(model_type="qwen3_5", hidden=8, num_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    m = ex.build_manifest(spec, "src", final_hidden=True)
    assert m["final_hidden_output"] is True
    assert ex.build_manifest(spec, "src")["final_hidden_output"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tools/tests/test_decision_export.py -v`
Expected: FAIL (no such parameters).

- [ ] **Step 3: Implement**
- `build_manifest(spec, source, final_hidden=False)`: add `"final_hidden_output": bool(final_hidden)` to the returned dict.
- `run_export`: extend the signature as in **Interfaces**, and pass `final_hidden` to `build_manifest`.
- Inside the stage loop, replace `stage = extract_stage(...)` with:

```python
        stage = extract_stage(xml, spec, a, b, first, last)
        if weights:
            from weight_swap import Weights, check_swap, swap_stage
            report = swap_stage(stage, Weights(weights))
            check_swap(report, range(a, b + 1), spec.layer_types, last)
            print(f"  stage{i}: swapped layers {a}..{b} from {weights}", flush=True)
        if lora:
            from weight_swap import graft_lora
            g = graft_lora(stage, lora, lora_scale)
            grafted += g
            print(f"  stage{i}: grafted {g} LoRA branches", flush=True)
        if last and final_hidden:
            from weight_swap import add_final_hidden_output
            add_final_hidden_output(stage)
```

Before the stage loop:

```python
    grafted = 0
    if lora:
        from weight_swap import lora_scale_from_config
        try:
            cfg_scale = lora_scale_from_config(lora)
        except FileNotFoundError:
            if lora_scale is None:
                raise SystemExit(f"no adapter_config.json beside {lora}: download it, or pass --lora-scale")
            cfg_scale = lora_scale
        if lora_scale is not None and abs(lora_scale - cfg_scale) > 1e-9:
            print(f"WARNING: --lora-scale {lora_scale} overrides adapter_config.json's {cfg_scale}", flush=True)
        lora_scale = cfg_scale if lora_scale is None else lora_scale
```

After the loop:

```python
    if lora:
        from weight_swap import lora_targets
        want = lora_targets(lora)
        if grafted != want:
            raise SystemExit(f"grafted {grafted} LoRA branches but the adapter has {want} targets")
```

- After `ov.save_model(...)`, add `del stage; import gc; gc.collect()` to keep peak memory at one stage.
- In the aux-file loop: when `weights` is set, skip copying `openvino_text_embeddings_model.{xml,bin}`, and instead call `weight_swap.swap_embeddings(os.path.join(model_dir, "openvino_text_embeddings_model.xml"), Weights(weights), output_dir)`. Also copy `tokenizer.json`/`config.json` from `weights` when they exist there (the fine-tune's tokenizer wins).
- `main`:

```python
    ap.add_argument("--weights", default=None, help="HF checkpoint whose weights replace the IR's (same architecture)")
    ap.add_argument("--lora", default=None, help="PEFT adapter_model.safetensors grafted as fp32 side branches")
    ap.add_argument("--lora-scale", type=float, default=None, help="override adapter_config.json's lora_alpha / r")
    ap.add_argument("--final-hidden", action="store_true", help="last stage also outputs final-norm hidden (decision heads)")
    ...
    run_export(args.model, args.out, args.total, args.validate, args.layer_split,
               weights=args.weights, lora=args.lora, lora_scale=args.lora_scale, final_hidden=args.final_hidden)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tools/tests/test_decision_export.py -v` and the existing exporter tests (`python -m pytest tools/tests -k qwen36 -v`)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/qwen36_surgery/export_qwen36_moe.py tools/tests/test_decision_export.py
git commit -m "feat(tools): exporter --weights/--lora/--final-hidden for decision trees"
```

---

### Task 14: Clef joint head module and OpenVINO export

The reference module is copied (Apache-2.0, attribution header) from Cloudflare's `joint_schema_model.py` (`EvidenceRoutingLayer`, `JointSchemaHead`). `TensorHead` is the spike's flat-tensor equivalent, verified to 7e-7 against the reference and 2.4e-6 in OpenVINO, with `cosine_similarity` rewritten because OpenVINO has no rule for it.

**Files:**
- Create: `tools/decision/__init__.py` (empty), `tools/decision/clef_head.py`
- Test: `tools/tests/test_clef_head.py`

**Interfaces:**
- Produces: `clef_head.{EvidenceRoutingLayer, JointSchemaHead, TensorHead, load_joint_head(head_dir) -> JointSchemaHead, export_joint_head(head_dir, out_xml) -> None}`. `load_joint_head` is also used by Task 17's reference. The OpenVINO IR has named inputs `hidden [T,H] f32`, `span_mask [Q+O,T] f32`, `lexical [O,H] f32`, `option_q [O] i64`, `type_ids [Q] i64`, and output `[O] f32`. These names are the contract with `decision.rs`.

- [ ] **Step 1: Write the failing test**

```python
"""Clef joint head: TensorHead == reference loop head; export names match decision.rs.

    python -m pytest tools/tests/test_clef_head.py -v
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from decision import clef_head as ch  # noqa: E402


def _record(hidden_size, t=20):
    torch.manual_seed(0)
    head = ch.JointSchemaHead(hidden_size=hidden_size, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).eval()
    hidden = torch.randn(t, hidden_size)
    q_spans = [(2, 4), (10, 12)]
    o_spans = [[(5, 6), (7, 9)], [(13, 15), (16, 18)]]
    lm_head = torch.randn(50, hidden_size)
    ids = torch.randint(0, 50, (t,))
    return head, hidden, q_spans, o_spans, lm_head, ids


def test_tensor_head_matches_reference():
    hs = 8
    head, hidden, q_spans, o_spans, lm_head, ids = _record(hs)
    ref = ch.reference_logits(head, hidden, ids, q_spans, o_spans, [1, 0], lm_head)
    spans = q_spans + [s for q in o_spans for s in q]
    mask = torch.zeros(len(spans), hidden.shape[0])
    for i, (a, b) in enumerate(spans):
        mask[i, a:b] = 1.0 / (b - a)
    lexical = torch.stack([lm_head[ids[a:b]].mean(0) for q in o_spans for (a, b) in q])
    option_q = torch.tensor([0, 0, 1, 1])
    got = ch.TensorHead(head)(hidden, mask, lexical, option_q, torch.tensor([1, 0]))
    assert torch.allclose(got, ref, atol=1e-5)


def test_export_uses_contract_input_names(tmp_path):
    ov = pytest.importorskip("openvino")
    head, *_ = _record(8)
    xml = tmp_path / "joint_head.xml"
    ch.export_module(ch.TensorHead(head), 8, str(xml))
    m = ov.Core().read_model(str(xml))
    assert [i.get_any_name() for i in m.inputs] == ["hidden", "span_mask", "lexical", "option_q", "type_ids"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tools/tests/test_clef_head.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/decision/clef_head.py`.** Write these parts into it:
- An Apache-2.0 attribution header naming Cloudflare `joint_schema_model.py`.
- `EvidenceRoutingLayer` and `JointSchemaHead`, copied verbatim from `experiments/clef-spike/flash/joint_schema_model.py` lines 242–459.
- `TensorHead`, copied from `experiments/clef-spike/step2_head_export.py` lines 37–94, with the `cosine` rewrite already applied there.
- These additions:

```python
def reference_logits(head, hidden, ids, q_spans, o_spans, qtypes, lm_head):
    """Run the reference loop head on one record described by raw spans (test helper)."""
    from types import SimpleNamespace
    questions = [SimpleNamespace(question_span=q, option_spans=tuple(o), question_type=t)
                 for q, o, t in zip(q_spans, o_spans, qtypes)]
    rec = SimpleNamespace(questions=tuple(questions))
    with torch.no_grad():
        out = head(hidden.unsqueeze(0), ids.unsqueeze(0), torch.ones(1, hidden.shape[0], dtype=torch.long), [rec], lm_head)[0]
    return torch.cat(out)


def export_module(module, hidden_size, out_xml):
    import openvino as ov
    t, q, o = 12, 2, 4
    example = (torch.randn(t, hidden_size), torch.full((q + o, t), 1.0 / t), torch.randn(o, hidden_size),
               torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0]))
    traced = torch.jit.trace(module.eval(), example, check_trace=False)
    model = ov.convert_model(traced, example_input=example,
                             input=[ov.PartialShape([-1, hidden_size]), ov.PartialShape([-1, -1]),
                                    ov.PartialShape([-1, hidden_size]), ov.PartialShape([-1]), ov.PartialShape([-1])])
    for inp, name in zip(model.inputs, ["hidden", "span_mask", "lexical", "option_q", "type_ids"]):
        inp.get_tensor().set_names({name})
    ov.save_model(model, out_xml, compress_to_fp16=False)


def load_joint_head(head_dir):
    """head_dir holds Cloudflare's joint_head.safetensors + joint_head_config.json."""
    import json
    from safetensors.torch import load_file
    cfg = json.loads(open(os.path.join(head_dir, "joint_head_config.json")).read())
    head = JointSchemaHead(**cfg)
    head.load_state_dict(load_file(os.path.join(head_dir, "joint_head.safetensors")), strict=True)
    return head.float().eval()


def export_joint_head(head_dir, out_xml):
    head = load_joint_head(head_dir)
    export_module(TensorHead(head), head.hidden_norm.normalized_shape[0], out_xml)
```

It starts with `from __future__ import annotations` (the copied code annotates with `EncodedRecord` and `jsm.JointSchemaHead`, which would raise NameError at import on Python 3.12), then `import math, os, torch`, `import torch.nn.functional as functional` (the copied bodies call `functional.normalize`/`functional.cosine_similarity`) and `import torch.nn.functional as F`. Strip the `jsm.` prefix from TensorHead's annotation.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tools/tests/test_clef_head.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/decision tools/tests/test_clef_head.py
git commit -m "feat(tools): Clef joint schema head module and OpenVINO export"
```

---

### Task 15: `decision_export.py`: one command per model family

**Files:**
- Create: `tools/decision_export.py`
- Test: `tools/tests/test_decision_export.py` (append)

**Interfaces:**
- Consumes: `export_qwen36_moe.run_export`, `decision.clef_head.export_joint_head`, `weight_swap.Weights`.
- Produces:
  - `write_jev_profile(adapter_dir, calibration_json, out_dir)`: reads `decision_head.json` + `calibration.json`; writes `decision.json` (family jev, `max_tokens` = config `max_position_embeddings`, default 32768) and `decision/head.json` `{"kind":"verbalizer"}`.
  - `write_kev_profile(head_pt, out_dir)`: reads `head.pt` with `torch.load(weights_only=True)`; writes `decision.json` (family kev, temperature) and `decision/head.json` `{"kind":"pointer","dp":256}` + `pointer_{q,k}_{w,b}.f32`.
  - `write_clef_profile(clef_dir, out_dir, max_length)`: writes `decision.json` (family clef), `decision/head.json` `{"kind":"joint","vocab":V}`, `decision/joint_head.xml/.bin` and `decision/lm_head.f16`.
  - CLI: `python tools/decision_export.py {jev,kev,clef} --base-ir DIR --out DIR [--stages N] [--weights DIR] [--lora FILE] [--lora-scale F] [--adapter-dir DIR] [--head-pt FILE] [--clef-dir DIR]`. `--stages` defaults to 8 for single-box trees (peak memory ≈ one stage); pipeline trees pass one stage per rank.

- [ ] **Step 1: Write the failing tests** (append to `tools/tests/test_decision_export.py`)

```python
import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import decision_export as de  # noqa: E402


def test_jev_profile_from_adapter_files(tmp_path):
    adapter = tmp_path / "adapter_vllm"
    adapter.mkdir()
    ids = list(range(100, 124))
    (adapter / "decision_head.json").write_text(json.dumps({
        "bias": [0.1] * 24, "verbalizer_ids": ids,
        "slots": {"num_slots": 24, "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                  "verbalizers": ["false", "true"] + [str(i) for i in range(6)] + [chr(65 + i) for i in range(16)]}}))
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"per_kind": {"noul": 1.0, "choice": 0.98, "score": 1.01}}))
    out = tmp_path / "tree"
    de.write_jev_profile(str(adapter), str(cal), str(out), max_tokens=4096)
    d = json.loads((out / "decision.json").read_text())
    assert d["family"] == "jev" and d["max_tokens"] == 4096
    assert d["jev"]["verbalizer_ids"] == ids and d["jev"]["slots"]["choice"] == [8, 24]
    assert d["jev"]["temperature"]["choice"] == 0.98
    assert json.loads((out / "decision" / "head.json").read_text()) == {"kind": "verbalizer"}


def test_kev_profile_writes_pointer_weights(tmp_path):
    torch = pytest.importorskip("torch")
    head = {"head": {"q.weight": torch.ones(256, 8), "q.bias": torch.zeros(256),
                     "k.weight": torch.full((256, 8), 2.0), "k.bias": torch.ones(256)},
            "temperature": 2.19}
    pt = tmp_path / "head.pt"
    torch.save(head, pt)
    out = tmp_path / "tree"
    de.write_kev_profile(str(pt), str(out), max_tokens=73728)
    d = json.loads((out / "decision.json").read_text())
    assert d == {"family": "kev", "max_tokens": 73728, "kev": {"temperature": 2.19}}
    kw = np.fromfile(out / "decision" / "pointer_k_w.f32", dtype="<f4")
    assert kw.shape == (256 * 8,) and kw[0] == 2.0
    assert json.loads((out / "decision" / "head.json").read_text()) == {"kind": "pointer", "dp": 256}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tools/tests/test_decision_export.py -v`
Expected: `ModuleNotFoundError: decision_export`.

- [ ] **Step 3: Implement `tools/decision_export.py`**

```python
"""Build a Cascadia decision-model tree (docs/decisions.md).

  jev:  python tools/decision_export.py jev  --base-ir STOCK_INT8_IR --adapter-dir JEV/adapter_vllm --out TREE [--stages N]
  kev:  python tools/decision_export.py kev  --base-ir STOCK_INT8_IR --weights QWEN35_9B_BASE --lora KEV/adapter_model.safetensors \
                                             --head-pt KEV/head.pt --out TREE
  clef: python tools/decision_export.py clef --base-ir STOCK_INT8_IR --weights CLEF_DIR --clef-dir CLEF_DIR --out TREE

The stage IRs come from export_qwen36_moe.run_export (one stage in memory at a time).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "qwen36_surgery"))
sys.path.insert(0, os.path.dirname(__file__))


def _write(out_dir, profile, head):
    out = Path(out_dir)
    (out / "decision").mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(profile, indent=1))
    (out / "decision" / "head.json").write_text(json.dumps(head))


def write_jev_profile(adapter_dir, calibration_json, out_dir, max_tokens=32768):
    h = json.loads((Path(adapter_dir) / "decision_head.json").read_text())
    cal = json.loads(Path(calibration_json).read_text())["per_kind"]
    slots = h["slots"]
    profile = {"family": "jev", "max_tokens": max_tokens, "jev": {
        "verbalizers": slots["verbalizers"], "verbalizer_ids": h["verbalizer_ids"], "bias": h["bias"],
        "slots": slots["ranges"], "temperature": {k: cal.get(k, 1.0) for k in ("noul", "choice", "score")}}}
    _write(out_dir, profile, {"kind": "verbalizer"})


def write_kev_profile(head_pt, out_dir, max_tokens=73728):
    import torch
    meta = torch.load(head_pt, map_location="cpu", weights_only=True)
    w = meta["head"]
    dp = w["q.weight"].shape[0]
    _write(out_dir, {"family": "kev", "max_tokens": max_tokens, "kev": {"temperature": float(meta["temperature"])}},
           {"kind": "pointer", "dp": int(dp)})
    for key, fname in (("q.weight", "pointer_q_w"), ("q.bias", "pointer_q_b"), ("k.weight", "pointer_k_w"), ("k.bias", "pointer_k_b")):
        w[key].float().numpy().astype("<f4").tofile(Path(out_dir) / "decision" / f"{fname}.f32")


def write_clef_profile(clef_dir, out_dir, max_tokens=16384, chunk=8192):
    from weight_swap import Weights
    from decision.clef_head import export_joint_head
    W = Weights(clef_dir)
    idx = W.idx
    vocab = None
    _write(out_dir, {"family": "clef", "max_tokens": max_tokens, "clef": {}}, {"kind": "joint", "vocab": 0})
    export_joint_head(clef_dir, str(Path(out_dir) / "decision" / "joint_head.xml"))
    # lm_head rows fp16 for the lexical prior, streamed in chunks (vocab x hidden)
    from safetensors import safe_open
    f = safe_open(Path(clef_dir) / idx["lm_head.weight"], "pt")
    sl = f.get_slice("lm_head.weight")
    vocab, hidden = sl.get_shape()
    with open(Path(out_dir) / "decision" / "lm_head.f16", "wb") as out:
        for a in range(0, vocab, chunk):
            sl[a:min(vocab, a + chunk)].float().numpy().astype("<f2").tofile(out)
    (Path(out_dir) / "decision" / "head.json").write_text(json.dumps({"kind": "joint", "vocab": int(vocab)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["jev", "kev", "clef"])
    ap.add_argument("--base-ir", required=True, help="Intel stock int8 IR dir (OpenVINO/Qwen3.5-9B-int8-ov or Qwen3.8-27B-int8-ov)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", type=int, default=8, help="stage cut count: 8 for a single box, one per rank for a pipeline")
    ap.add_argument("--layer-split", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=None, help="override adapter_config.json")
    ap.add_argument("--adapter-dir", default=None, help="jev: dir with decision_head.json + adapter_model.safetensors")
    ap.add_argument("--calibration", default=None, help="jev: calibration.json (default: ADAPTER_DIR/../calibration.json)")
    ap.add_argument("--head-pt", default=None, help="kev: head.pt")
    ap.add_argument("--clef-dir", default=None, help="clef: dir with joint_head.* and the checkpoint")
    a = ap.parse_args()
    from export_qwen36_moe import run_export

    if a.family == "jev":
        lora = a.lora or os.path.join(a.adapter_dir, "adapter_model.safetensors")
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights, lora=lora, lora_scale=a.lora_scale)
        cal = a.calibration or os.path.join(os.path.dirname(os.path.abspath(a.adapter_dir)), "calibration.json")
        cfg = json.loads((Path(a.base_ir) / "config.json").read_text())
        mpe = (cfg.get("text_config") or cfg).get("max_position_embeddings", 32768)
        write_jev_profile(a.adapter_dir, cal, a.out, max_tokens=min(int(mpe), 32768))
    elif a.family == "kev":
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights, lora=a.lora,
                   lora_scale=a.lora_scale, final_hidden=True)
        write_kev_profile(a.head_pt, a.out)
    else:
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights, final_hidden=True)
        write_clef_profile(a.clef_dir, a.out)
    print(f"decision tree written: {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tools/tests/test_decision_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/decision_export.py tools/tests/test_decision_export.py
git commit -m "feat(tools): decision_export.py builds jev/kev/clef decision trees"
```

- [ ] **Step 6: Pipeline smoke test** (manual, needs OpenVINO and Tasks 11/11b; one machine, two processes; nothing to commit)

This is the first point where a 2-stage decision tree exists, so it is where the pipeline frames (Task 11b) are exercised end to end. Build a 2-stage JEV-9B tree, then start the two ranks. Flags are as declared in `WorkerArgs`: the tree is `--model`, and the downstream peer is `--next`. Use the runbook environment from Global Constraints (`$SPIKE_DIR`, `$PY`, `cascadia` on `PATH`). Run each process in the background with a Monitor on its log, watching for `Traceback|panic|error|timed out|NAK|epoch|poisoned`.

```bash
$PY tools/decision_export.py jev --base-ir ~/models/qwen35-9b-int8 --adapter-dir ~/models/jev9b/adapter_vllm --stages 2 --out ~/trees/jev9b-int8-2st
start_ranks() {   # rank 1 first (it listens), then rank 0 (it dials rank 1 once, at startup)
  cascadia worker --model ~/trees/jev9b-int8-2st --engine qwen35 --device CPU --rank 1 --total 2 --listen 127.0.0.1:7801 > /tmp/rank1.log 2>&1 &
  sleep 5
  cascadia worker --model ~/trees/jev9b-int8-2st --engine qwen35 --device CPU --rank 0 --total 2 --next 127.0.0.1:7801 --api 127.0.0.1:8080 > /tmp/rank0.log 2>&1 &
  until curl -sf localhost:8080/health > /dev/null; do sleep 2; done
}
stop_ranks() { pkill -f 'cascadia worker --model' ; sleep 2; }
start_ranks
H='content-type: application/json'
curl -s -H "$H" localhost:8080/v1/systemone -d '{"model":"jev","state":"I was charged twice","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"charges","tech":"bugs"}}}}' | tee /tmp/small-base.json
curl -s -H "$H" localhost:8080/v1/decisions -d '{"model":"jev","input":"I was charged twice","questions":[{"type":"choice","name":"team","instructions":"Which team?","choices":[{"value":"billing","description":"charges"},{"value":"tech","description":"bugs"}]}]}'
```

Expected: 200 on both, with `"choice":"billing"` and identical probabilities. (Without the `content-type` header axum answers 415.) Then, on the same two ranks:

- **Large-body check.** A JEV state of at least 12k tokens, so FRAME_DECIDE (about 6–7 bytes of JSON per prompt id) is well past 64 KiB; JEV's `max_tokens` is 32768. It must return 200 and report `usage.input_tokens` ≥ 12000. Its answer is the baseline for the next two checks.

  ```bash
  $PY - <<'EOF'
  import json
  state = "The quick brown fox jumps over the lazy dog. " * 1500   # about 13.5k Qwen tokens, ~68 KB
  q = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs"}}}
  json.dump({"model": "jev", "state": state, "questions": q}, open("/tmp/jev-large.json", "w"))
  EOF
  curl -s -H "$H" --data-binary @/tmp/jev-large.json localhost:8080/v1/systemone -o /tmp/large-base.json -w '%{http_code}\n'
  $PY -c 'import json; r = json.load(open("/tmp/large-base.json")); print(r["usage"]["input_tokens"], r["answers"]["team"])'
  ```

- **Abort, then warm chat.** Drop a large decision mid-prefill, then at once send a chat request and the same decision again. Both must return 200, and the decision's answer must equal `/tmp/large-base.json` exactly (a fresh-server run).

  ```bash
  curl -s -H "$H" --max-time 1 --data-binary @/tmp/jev-large.json localhost:8080/v1/systemone; echo "abort exit=$?"   # 28 = client timed out mid-prefill
  curl -s -H "$H" localhost:8080/v1/chat/completions -d '{"model":"jev","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}' -o /dev/null -w 'chat %{http_code}\n'
  curl -s -H "$H" --data-binary @/tmp/jev-large.json localhost:8080/v1/systemone -o /tmp/large-after-abort.json -w 'decision %{http_code}\n'
  $PY -c 'import json; a, b = (json.load(open(f))["answers"] for f in ("/tmp/large-base.json", "/tmp/large-after-abort.json")); assert a == b, (a, b); print("same answer")'
  ```

- **Stale-epoch DECIDE.** Abort at several points of the large request's lifetime, so some aborts land after prefill and before FRAME_DECIDE_RUN, then send two decisions back to back. Every follow-up must return 200 with the baseline answer. A `dropping stale DECIDE frame` / `dropping stale frame` warning on rank 1 is the expected drop and is fine; `timed out`, `decision engine error`, `poisoned` or a NAK in either log fails the step.

  ```bash
  for t in 0.5 1 2 3 5; do
    curl -s -H "$H" --max-time $t --data-binary @/tmp/jev-large.json localhost:8080/v1/systemone > /dev/null
    curl -s -H "$H" --data-binary @/tmp/jev-large.json localhost:8080/v1/systemone -o /tmp/stale-a.json -w "t=$t a %{http_code} "
    curl -s -H "$H" localhost:8080/v1/systemone -d '{"model":"jev","state":"I was charged twice","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"charges","tech":"bugs"}}}}' -o /tmp/stale-b.json -w "b %{http_code}\n"
    $PY -c 'import json; L = lambda f: json.load(open(f))["answers"]; assert L("/tmp/stale-a.json") == L("/tmp/large-base.json") and L("/tmp/stale-b.json") == L("/tmp/small-base.json"); print("baseline answers")'
  done
  grep -nE 'stale|epoch|timed out|decision engine error|poisoned|NAK' /tmp/rank0.log /tmp/rank1.log
  ```

- **Old-rank check.** Rank 0 dials rank 1 once, at startup, and keeps its handshake, so restarting rank 1 alone only breaks the socket (a 503 or a timeout, not 501). Stop **both** ranks, start a `main`-build rank 1, then a fresh branch rank 0. The decision routes must answer 501 `decision_unsupported` and chat must still answer 200. Then restore the branch ranks.

  ```bash
  stop_ranks
  git worktree add /tmp/cascadia-main origin/main && (cd /tmp/cascadia-main && cargo build --release -p cascadia --features openvino)
  /tmp/cascadia-main/target/release/cascadia worker --model ~/trees/jev9b-int8-2st --engine qwen35 --device CPU --rank 1 --total 2 --listen 127.0.0.1:7801 > /tmp/rank1-main.log 2>&1 &
  sleep 5
  cascadia worker --model ~/trees/jev9b-int8-2st --engine qwen35 --device CPU --rank 0 --total 2 --next 127.0.0.1:7801 --api 127.0.0.1:8080 > /tmp/rank0.log 2>&1 &
  until curl -sf localhost:8080/health > /dev/null; do sleep 2; done
  curl -s -H "$H" localhost:8080/v1/systemone -d '{"model":"jev","state":"x","questions":{"u":{"type":"noul","instructions":"Is it?"}}}' -w ' %{http_code}\n'    # 501, "code":"decision_unsupported"
  curl -s -H "$H" localhost:8080/v1/decisions -d '{"model":"jev","input":"x","questions":[{"type":"predicate","name":"u","instructions":"Is it?"}]}' -w ' %{http_code}\n'   # 501
  curl -s -H "$H" localhost:8080/v1/chat/completions -d '{"model":"jev","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}' -o /dev/null -w 'chat %{http_code}\n'   # 200
  stop_ranks
  git worktree remove /tmp/cascadia-main
  ```

  The 501 must come back within the request timeout, not hang.

---

### Task 16: Parity tooling and the codec golden check

Two different questions need two different checks:
- **Are the codecs right?** The server's prompts, positions, spans and answers must match the reference implementations. That is the golden check (`codec_golden.py`): it compares the server's `--decision-dump` against independent Python encoders (Cloudflare's `encode_record`, Kev's `questions`, and a serve_decide-style JEV renderer), and the server's answers against the spike's saved full-precision references.
- **Is int8 close enough?** Task 17's `reference_stream.py` replays the server's own prompts in fp32, so it isolates quantization and nothing else.

Neither check can stand in for the other.

**Files:**
- Create: `tools/decision/records.py`, `tools/decision/parity_check.py`, `tools/decision/backbone_diff.py`, `tools/decision/codec_golden.py`, `tools/decision/openai_equivalence.py`
- Test: `tools/tests/test_decision_tools.py`

**Spike sources.** `experiments/` is gitignored and absent from worktrees. The spike files live in the main checkout at `/Users/blokchainaholic/Desktop/clabs/tahoma/experiments`. The tools find them through `SPIKE_DIR` (env var, defaulting to that path). Only `records.py` is copied into the repo; the golden check imports the spike encoders from `SPIKE_DIR` at run time and never vendors them.

**Interfaces:**
- `records.py`: a verbatim copy of `$SPIKE_DIR/clef-spike/records.py` (`records() -> list[dict]`, 50 records).
- `parity_check.py --url URL [--ref REF.json] [--min-agree 0.96] [--confident 0.05]`:
  - posts every record;
  - without `--ref` it only sends requests (so the server writes its dump);
  - with `--ref` it prints agreement per question type and lists every flip with the reference's top-1 margin. It exits non-zero when `agree / total < min_agree` **or** any flip has a reference margin ≥ `--confident`.
- `def gate(flip_margins: list[float], agree: int, total: int, min_agree: float, confident: float) -> tuple[bool, str]` (pure, tested).
- `codec_golden.py {jev,kev,clef} --url URL --dump DUMP.jsonl --tokenizer DIR [--adapter-dir DIR] [--answers]` exits non-zero on the first mismatch:
  - It notes the dump file's current size, POSTs the 50 records **plus `SYNTHETIC` golden-only records** to the server, and compares only the dump lines appended after that offset. The synthetic records cover every rendering branch the 50 records miss: a list state, a nested list-of-dicts state, noul criteria overrides, a dict-valued choice description, empty and null instructions, and floats (`1e-05`, `1e16`, `2.5`).
  - `prompt_ids`, positions and spans must equal the reference encoder's exactly. For Clef the prompt ids must also equal `reference_flash.pt`'s saved `input_ids`, a tokenizer-version-independent golden for the 50 records.
  - For JEV it also checks every pass's verbalizer `token_ids` against the adapter's `decision_head.json` (`--adapter-dir`), using serve_decide's slot logic: the kind's slots, the first `n` for choice.
  - With `--answers`, the server's top-1 on the 50 records must agree with the saved spike reference on ≥ 98%. For Kev and Clef that covers every question. For JEV it covers every question too (score references rescaled to the record's L levels first). `reference_jev.pt` was measured on sorted choices, so the answer run sends each JEV choice question with its criteria sorted (`jev_sorted`), and the prompts it dumps must equal `jev.questions()`'s ids exactly. (Fix round 1, FID-3: the plan code below still shows the earlier informational-only JEV choice line; the shipped `codec_golden.py` gates all 82.)
  - The 27B models have no saved spike reference; for them only the encoder comparison runs, and parity (Task 17) is the answer check.
- `openai_equivalence.py --url URL`: converts each of the 50 records into an OpenAI-format request (`to_openai(rec)`), POSTs it to `/v1/decisions` and the original to `/v1/systemone`, and exits non-zero unless every question's probabilities, choice/score and confidence are identical. The records' score levels are plain strings (no descriptions), so both routes build byte-identical prompts, and any difference is an adapter bug. Everything verified through `/v1/systemone` (golden, parity) then carries over to `/v1/decisions`.
- `backbone_diff.py A B [--per-layer] [--slice BYTES]`: matches tensors per layer index and suffix after stripping the repo's `model.language_model.` / `model.` prefix (vision and MTP tensors ignored), hashes both ends of every backbone tensor (all projections, norms, embeddings, final norm, lm_head), prints per-tensor / per-layer `SAME`/`DIFF`, and ends with one `VERDICT: STOCK | FINETUNED | UNKNOWN` line; UNKNOWN (no comparable decoder-layer tensors, different layer sets, or a dtype/shape mismatch) exits 2. (Fix round 1, BLD-1, replaces the sampled/gate_proj-only probe below.)
- `def answer_probs(ans: dict) -> list[float]` and `def margin(p: list[float]) -> float` (in parity_check, tested).

- [ ] **Step 1: Write the failing tests** (`tools/tests/test_decision_tools.py`)

```python
"""Decision parity tools: answer flattening, margins, dump alignment.

    python -m pytest tools/tests/test_decision_tools.py -v
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "decision"))
import codec_golden as cg  # noqa: E402
import parity_check as pc  # noqa: E402
from records import records  # noqa: E402


def test_records_are_the_fifty_spike_records():
    rs = records()
    assert len(rs) == 50 and sum(len(r["questions"]) for r in rs) == 82


def test_answer_probs_flattens_every_type():
    assert pc.answer_probs({"type": "noul", "noul": 0.75}) == [0.25, 0.75]
    assert pc.answer_probs({"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}}) == [0.1, 0.9]
    assert pc.answer_probs({"type": "score", "score": 1.2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}) == [0.2, 0.4, 0.4]


def test_margin_is_top1_minus_top2():
    assert abs(pc.margin([0.501, 0.499]) - 0.002) < 1e-12
    assert pc.margin([1.0]) == 1.0


def test_gate_tolerates_coin_flips_but_not_confident_flips():
    assert pc.gate([0.002, 0.01, 0.04], 79, 82, 0.96, 0.05)[0], "3 undecided flips: pass"
    assert not pc.gate([0.002, 0.01, 0.02, 0.03], 78, 82, 0.96, 0.05)[0], "4 flips: under 96%"
    ok, why = pc.gate([0.06], 81, 82, 0.96, 0.05)
    assert not ok and "confident" in why, "one flip the reference was sure about fails"


def test_synthetic_records_cover_the_risky_branches():
    kinds = {k for r in cg.SYNTHETIC for k in cg.branches(r)}
    assert {"list_state", "noul_criteria", "dict_description", "empty_instructions", "null_instructions", "float"} <= kinds


def test_records_convert_to_openai_requests():
    import openai_equivalence as oe
    r = records()[0]
    o = oe.to_openai(r)
    assert [q["name"] for q in o["questions"]] == list(r["questions"])
    assert o["input"] == (r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False))
    kinds = {q["type"] for rr in records() for q in oe.to_openai(rr)["questions"]}
    assert kinds == {"predicate", "choice", "score"}


def test_dump_passes_align_to_records(tmp_path):
    # JEV/Kev: one pass per question, in record then question order
    lines = []
    for r in records()[:2]:
        for qid in r["questions"]:
            lines.append({"parts": [[qid, 0, 2]], "readout": {"prompt_ids": [1, 2], "head": {"kind": "verbalizer", "token_ids": [1, 2]}}})
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    aligned = list(cg.align(str(p), records()[:2]))
    assert [len(passes) for _, passes in aligned] == [len(r["questions"]) for r in records()[:2]]
    assert aligned[0][0]["id"] == records()[0]["id"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tools/tests/test_decision_tools.py -v`
Expected: `ModuleNotFoundError` (`codec_golden`, `parity_check`, `records`).

- [ ] **Step 3: Write `records.py`, `parity_check.py` and `backbone_diff.py`**

`cp /Users/blokchainaholic/Desktop/clabs/tahoma/experiments/clef-spike/records.py tools/decision/records.py`

`tools/decision/parity_check.py`:

```python
"""Agreement of a running decisions server (SystemOne route, /v1/systemone) with a full-precision reference.

  python tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump d.jsonl        # drive the server, write d.parity.jsonl (this run's passes only)
  python tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref ref.json        # gate: raw >= 96% and no flip with reference margin >= 0.05
"""
import argparse
import json
import sys
import urllib.request
from collections import defaultdict

from records import records


def answer_probs(ans):
    if ans["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    return list(ans["probabilities"].values())


def margin(p):
    s = sorted(p, reverse=True)
    return s[0] - (s[1] if len(s) > 1 else 0.0)


def gate(flip_margins, agree, total, min_agree, confident):
    """Pass iff raw agreement >= min_agree AND no flip where the reference itself was confident."""
    sure = [m for m in flip_margins if m >= confident]
    if sure:
        return False, f"{len(sure)} confident flip(s) (reference margin >= {confident}): {[round(m, 3) for m in sure]}"
    rate = agree / max(1, total)
    if rate < min_agree:
        return False, f"agreement {rate:.3f} < {min_agree}"
    return True, f"agreement {agree}/{total} = {rate:.3f}, every flip had reference margin < {confident}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref", default=None, help="omit to only drive the server (it writes its --decision-dump)")
    ap.add_argument("--min-agree", type=float, default=0.96, help="raw top-1 agreement floor (0.96 = at most 3 flips of 82)")
    ap.add_argument("--confident", type=float, default=0.05, help="a flip where the reference margin is >= this fails the gate")
    a = ap.parse_args()
    ref = json.load(open(a.ref)) if a.ref else None
    per_type = defaultdict(lambda: [0, 0])
    agree = total = 0
    flip_margins = []
    for rec in records():
        body = json.dumps({"model": "m", "state": rec["state"], "questions": rec["questions"]}).encode()
        req = urllib.request.Request(a.url + "/v1/systemone", data=body, headers={"content-type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=1800))
        if ref is None:
            continue
        for qid, ans in resp["answers"].items():
            got, want = answer_probs(ans), ref[f"{rec['id']}.{qid}"]
            same = max(range(len(got)), key=got.__getitem__) == max(range(len(want)), key=want.__getitem__)
            kind = rec["questions"][qid]["type"]
            per_type[kind][0] += same
            per_type[kind][1] += 1
            agree += same
            total += 1
            if not same:
                flip_margins.append(margin(want))
                print(f"FLIP {rec['id']}.{qid} {kind} ref_margin={margin(want):.3f} "
                      f"got={[round(x, 3) for x in got]} ref={[round(x, 3) for x in want]}")
    if ref is None:
        print("requests sent (no reference given)")
        return
    for kind, (g, n) in sorted(per_type.items()):
        print(f"  {kind}: {g}/{n}")
    ok, why = gate(flip_margins, agree, total, a.min_agree, a.confident)
    print(("PASS " if ok else "FAIL ") + why)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

`tools/decision/openai_equivalence.py`:

```python
"""/v1/decisions (OpenAI format) must answer exactly like /v1/systemone on the 50 records.

  python tools/decision/openai_equivalence.py --url http://127.0.0.1:8080
"""
import argparse
import json
import sys
import urllib.request

from records import records

KIND = {"noul": "predicate", "choice": "choice", "score": "score"}


def to_openai(rec):
    """SystemOne record -> OpenAI Decisions request. String states pass through; object states go as
    their json.dumps text, which is what the JEV codec renders for objects anyway. Kev and Clef render
    object states structurally, so for them the equivalence check runs on string-state records only."""
    st = rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"], ensure_ascii=False)
    qs = []
    for qid, q in rec["questions"].items():
        o = {"type": KIND[q["type"]], "name": qid, "instructions": q.get("instructions") or qid}
        if q["type"] == "choice":
            o["choices"] = [{"value": k, "description": v} for k, v in q["criteria"].items()]
        elif q["type"] == "score":
            o["levels"] = [{"label": lvl} for lvl in q["criteria"]]
        qs.append(o)
    return {"model": "m", "input": st, "questions": qs}


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def flat_openai(a):
    if a["type"] == "predicate":
        return {"p": [1 - a["probability"], a["probability"]]}
    return {"p": [x["probability"] for x in a["probabilities"]], "c": a["confidence"], "s": a.get("score"), "choice": a.get("choice")}


def flat_systemone(a):
    if a["type"] == "noul":
        return {"p": [1 - a["noul"], a["noul"]]}
    return {"p": list(a["probabilities"].values()), "c": a["confidence"], "s": a.get("score"), "choice": a.get("choice")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--all-states", action="store_true", help="also object-state records (JEV only: identical rendering)")
    a = ap.parse_args()
    n = bad = 0
    for rec in records():
        if not a.all_states and not isinstance(rec["state"], str):
            continue
        oa = post(a.url + "/v1/decisions", to_openai(rec))
        so = post(a.url + "/v1/systemone", {"model": "m", "state": rec["state"], "questions": rec["questions"]})
        for ans in oa["answers"]:
            got, want = flat_openai(ans), flat_systemone(so["answers"][ans["name"]])
            n += 1
            if got != want:
                bad += 1
                print(f"DIFF {rec['id']}.{ans['name']}: openai={got} systemone={want}")
    print(f"{n - bad}/{n} questions identical across /v1/decisions and /v1/systemone")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

`tools/decision/backbone_diff.py`: copy `$SPIKE_DIR/clef-spike/step1_backbone_diff.py` verbatim. Then add `--per-layer`: for every decoder layer, compare the first 8 MB of `mlp.gate_proj.weight` (`layers.{i}.` under the auto-detected prefix) and print `layer {i}: SAME|DIFF`. This is the probe that found Clef-27B's fine-tune touching only layers 40–63. (Superseded in fix round 1, BLD-1: exact-name matching made it vacuous across text-only vs VLM layouts, and gate_proj alone misses attention, norm, embedding and lm_head changes; see the interface line above.)

- [ ] **Step 4: Write `codec_golden.py`**

```python
"""Golden check: the server's decision passes vs independent reference encoders, and its answers vs the
spike's saved full-precision references. Catches codec bugs that int8-vs-fp32 parity cannot see.

  python tools/decision/codec_golden.py kev  --url http://127.0.0.1:8080 --dump kev.jsonl  --tokenizer ~/models/kev9b [--answers]
  python tools/decision/codec_golden.py clef --url http://127.0.0.1:8080 --dump clef.jsonl --tokenizer ~/models/clef-flash [--answers]
  python tools/decision/codec_golden.py jev  --url http://127.0.0.1:8080 --dump jev.jsonl  --tokenizer ~/models/jev9b --adapter-dir ~/models/jev9b/adapter_vllm [--answers]
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from records import records

SPIKE_DIR = Path(os.environ.get("SPIKE_DIR", "/Users/blokchainaholic/Desktop/clabs/tahoma/experiments"))


def align(dump_path, recs):
    """Yield (record, [pass, ...]) by consuming dump lines until every question of the record is covered."""
    lines = [json.loads(l) for l in open(dump_path) if l.strip()]
    i = 0
    for rec in recs:
        need, got = set(rec["questions"]), []
        while need:
            if i >= len(lines):
                raise SystemExit(f"dump ends before record {rec['id']} is covered")
            got.append(lines[i])
            need -= {p[0] for p in lines[i]["parts"]}
            i += 1
        yield rec, got


def fail(msg):
    print(f"MISMATCH {msg}")
    sys.exit(1)


# Golden-only records: every rendering branch the 50 parity records never reach. No saved
# reference exists for them, so they are compared on prompts/positions only, never on answers.
SYNTHETIC = [
    {"id": "s00", "state": ["first turn. ", {"type": "text", "text": "second turn."}, {"k": 1.5}],
     "questions": {"u": {"type": "noul", "instructions": "Is this resolved?",
                         "criteria": {"true": "Yes, closed.", "false": "No, still open."}}}},
    {"id": "s01", "state": {"items": [{"sku": "a1", "price": 1e-05}, {"sku": "b2", "price": 1e16}], "total": 2.5},
     "questions": {"c": {"type": "choice", "instructions": "",
                         "criteria": {"refund": {"when": "charged twice", "max": 500}, "ignore": "no action"}}}},
    {"id": "s02", "state": "Plain text state.",
     "questions": {"s": {"type": "score", "instructions": None, "criteria": ["none", "minor", "major", "critical"]}}},
]


def branches(rec):
    out = set()
    if isinstance(rec["state"], list):
        out.add("list_state")
    if "1e" in json.dumps(rec["state"]) or ".5" in json.dumps(rec["state"]):
        out.add("float")
    for q in rec["questions"].values():
        if q["type"] == "noul" and q.get("criteria"):
            out.add("noul_criteria")
        if q["type"] == "choice" and any(isinstance(v, dict) for v in q["criteria"].values()):
            out.add("dict_description")
        if q.get("instructions") == "":
            out.add("empty_instructions")
        if "instructions" in q and q["instructions"] is None:
            out.add("null_instructions")
    return out


def golden_records():
    return records() + SYNTHETIC


def jev_state(state):
    """serve_decide _parts: str verbatim, dict json.dumps, list concatenated part by part."""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "".join(p if isinstance(p, str) else p["text"] if isinstance(p, dict) and p.get("type") == "text"
                       else json.dumps(p, ensure_ascii=False) for p in state)
    return json.dumps(state, ensure_ascii=False)


def jev_value(v):
    return "" if v is None else v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def jev_text(rec, qid, labels):
    """serve_decide s1_pass layout, caller order; mirrors the server codec's rendering rules."""
    q = rec["questions"][qid]
    kind = q["type"]
    if kind == "choice":
        lines = [f"{labels[i]}) {n}: {jev_value(d)}" if jev_value(d) else f"{labels[i]}) {n}" for i, (n, d) in enumerate(q["criteria"].items())]
    elif kind == "noul":
        lines = ["false", "true"]
    else:
        lines = [str(i) for i in range(6)]
    question = jev_value(q["instructions"]) if q.get("instructions") is not None else qid
    return f"[kind] {kind}\n[state] {jev_state(rec['state'])}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def check_jev(dump, tok, adapter_dir, recs):
    head = json.loads((Path(adapter_dir) / "decision_head.json").read_text())
    ranges = head["slots"]["ranges"]
    labels = head["slots"]["verbalizers"][ranges["choice"][0]:ranges["choice"][1]]
    n = 0
    for rec, passes in align(dump, recs):
        for p in passes:
            qid = p["parts"][0][0]
            q = rec["questions"][qid]
            want = tok(jev_text(rec, qid, labels), add_special_tokens=False).input_ids
            if p["readout"]["prompt_ids"] != want:
                fail(f"jev {rec['id']}.{qid}: prompt_ids differ")
            lo, hi = ranges[q["type"]]
            if q["type"] == "choice":
                hi = lo + len(q["criteria"])
            if p["readout"]["head"]["token_ids"] != head["verbalizer_ids"][lo:hi]:
                fail(f"jev {rec['id']}.{qid}: verbalizer token_ids differ from decision_head.json")
            n += 1
    print(f"jev: {n} passes match the serve_decide layout and decision_head.json")


def check_kev(dump, tok, adapter_dir, recs):
    sys.path.insert(0, str(SPIKE_DIR / "kev-spike"))
    import kev
    kev.records = lambda: recs  # kev.questions iterates kev.records(); point it at the golden set
    want = {qid: (row, d, ends) for qid, kind, row, d, ends, keys in kev.questions(tok)}
    n = 0
    for rec, passes in align(dump, recs):
        for p in passes:
            qid = p["parts"][0][0]
            row, d, ends = want[f"{rec['id']}.{qid}"]
            r = p["readout"]
            if r["prompt_ids"] != list(row) or r["head"]["decide_pos"] != d or r["head"]["option_pos"] != list(ends):
                fail(f"kev {rec['id']}.{qid}: row/positions differ")
            n += 1
    print(f"kev: {n} passes match kev.questions")


def check_clef(dump, tok, adapter_dir, recs):
    import torch
    sys.path.insert(0, str(SPIKE_DIR / "clef-spike" / "flash"))
    import joint_schema_model as jsm
    saved = {e["id"]: e["input_ids"] for e in torch.load(SPIKE_DIR / "clef-spike" / "reference_flash.pt", map_location="cpu", weights_only=True)}
    n = 0
    for rec, passes in align(dump, recs):
        (p,) = passes
        enc = jsm.encode_record(tok, rec, max_length=16384)
        r = p["readout"]
        if r["prompt_ids"] != list(enc.input_ids):
            fail(f"clef {rec['id']}: prompt_ids differ")
        if rec["id"] in saved and r["prompt_ids"] != list(saved[rec["id"]]):
            fail(f"clef {rec['id']}: prompt_ids differ from reference_flash.pt input_ids")
        for q, eq in zip(r["head"]["questions"], enc.questions):
            if q["qtype"] != eq.question_type or tuple(q["span"]) != tuple(eq.question_span) or \
                    [tuple(o["span"]) for o in q["options"]] != [tuple(s) for s in eq.option_spans]:
                fail(f"clef {rec['id']}.{eq.question_id}: spans/type differ")
        n += 1
    print(f"clef: {n} records match encode_record")


def check_answers(family, url):
    import torch
    ref_file = {"kev": SPIKE_DIR / "kev-spike" / "reference_kev.pt", "clef": SPIKE_DIR / "clef-spike" / "reference_flash.pt",
                "jev": SPIKE_DIR / "jev-spike" / "reference_jev.pt"}[family]
    raw = torch.load(ref_file, map_location="cpu", weights_only=True)
    if family == "clef":
        # reference_flash.pt is per RECORD: {id, input_ids, hidden, logits=[one tensor per question]}
        recs = {r["id"]: r for r in records()}
        ref = {f"{e['id']}.{qid}": {"probs": torch.as_tensor(lg).float().softmax(-1)}
               for e in raw for qid, lg in zip(recs[e["id"]]["questions"], e["logits"])}
    else:
        ref = {r["qid"]: r for r in raw}
    agree = total = 0
    for rec in records():
        body = json.dumps({"model": "m", "state": rec["state"], "questions": rec["questions"]}).encode()
        resp = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/systemone", data=body,
                                                                       headers={"content-type": "application/json"}), timeout=1800))
        for qid, ans in resp["answers"].items():
            r = ref.get(f"{rec['id']}.{qid}")
            if r is None:
                continue
            q = rec["questions"][qid]
            if family == "jev" and q["type"] == "choice":
                print(f"  (jev choice, informational) {rec['id']}.{qid}: server={top1(ans)} reference={reference_top1(family, q, r)}")
                continue
            if family == "jev" and q["type"] == "score":
                r = {"probs": torch.tensor(rescale_score(r["probs"].tolist(), len(q["criteria"])))}
            want = reference_top1(family, q, r)
            got = top1(ans)
            agree += got == want
            total += 1
            if got != want:
                print(f"ANSWER {rec['id']}.{qid}: server={got} reference={want}")
    print(f"{family}: server answers agree with the saved reference on {agree}/{total}")
    if agree < 0.98 * total:
        sys.exit(1)


def rescale_score(p6, levels):
    """Same mapping as the JEV codec's rescale_score (Task 5)."""
    out = [0.0] * levels
    for j, p in enumerate(p6):
        x = j * (levels - 1) / (len(p6) - 1)
        lo = int(x)
        w = x - lo
        out[lo] += p * (1 - w)
        if w > 0:
            out[lo + 1] += p * w
    return out


def post_all(url, recs):
    for rec in recs:
        body = json.dumps({"model": "m", "state": rec["state"], "questions": rec["questions"]}).encode()
        urllib.request.urlopen(urllib.request.Request(url + "/v1/systemone", data=body, headers={"content-type": "application/json"}), timeout=1800).read()


def top1(ans):
    if ans["type"] == "noul":
        return "true" if ans["noul"] >= 0.5 else "false"
    if ans["type"] == "choice":
        return ans["choice"]
    p = list(ans["probabilities"].values())
    return str(max(range(len(p)), key=p.__getitem__))


def reference_top1(family, q, r):
    """Name of the reference's top option in API terms (each spike stored probs in its own option order)."""
    i = int(r["probs"].argmax())
    kind = q["type"]
    if kind == "noul":
        order = ["true", "false"] if family == "clef" else ["false", "true"]
        return order[i]
    if kind == "choice":
        names = sorted(q["criteria"]) if family in ("clef", "jev") else list(q["criteria"])
        return names[i]
    return str(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["jev", "kev", "clef"])
    ap.add_argument("--url", required=True, help="the running server (started with --decision-dump DUMP)")
    ap.add_argument("--dump", required=True, help="the server's --decision-dump file (same machine)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter-dir", default=None, help="jev: dir with decision_head.json")
    ap.add_argument("--answers", action="store_true", help="also compare top-1 answers with the saved spike reference")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if a.family == "jev" and not a.adapter_dir:
        raise SystemExit("jev needs --adapter-dir (decision_head.json)")
    # Compare only what THIS run appends: earlier parity runs may already be in the dump.
    start = Path(a.dump).stat().st_size if Path(a.dump).exists() else 0
    recs = golden_records()
    post_all(a.url, recs)
    tail = Path(a.dump).read_bytes()[start:].decode()
    fresh = Path(a.dump).with_suffix(".golden.jsonl")
    fresh.write_text(tail)
    {"jev": check_jev, "kev": check_kev, "clef": check_clef}[a.family](str(fresh), tok, a.adapter_dir, recs)
    if a.answers:
        check_answers(a.family, a.url)


if __name__ == "__main__":
    main()
```

Before writing `reference_top1`, check the option order each spike reference stores, with `python -c "import torch; r=torch.load('…/reference_flash.pt', weights_only=True); print(r[0].keys())"`. The spike notes say: Clef stores `[true, false]` and sorted choices; Kev stores `[false, true]` and caller-order choices; JEV stores `[false, true]` and sorted choices. If a file carries option names, use them instead of the order table.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tools/tests/test_decision_tools.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add tools/decision/records.py tools/decision/parity_check.py tools/decision/backbone_diff.py tools/decision/codec_golden.py tools/decision/openai_equivalence.py tools/tests/test_decision_tools.py
git commit -m "feat(tools): decision parity check, codec golden check and backbone diff"
```

---

### Task 17: Streamed fp32 reference (`reference_stream.py`)

Generalizes `$SPIKE_DIR/jev-spike/ref_stream.py` and `$SPIKE_DIR/kev-spike/kev_ref.py`: one decoder layer resident at a time, LoRA merged per layer in fp32. It replays the server's dumped passes, so it measures int8 against full precision on identical inputs.

**Files:**
- Create: `tools/decision/reference_stream.py`
- Test: `tools/tests/test_reference_stream.py`

**Interfaces:**
- `reference_stream.py {jev,kev,clef} --weights DIR --dump DUMP.parity.jsonl --tree TREE --out REF.json [--lora FILE] [--lora-scale F] [--clef-dir DIR]`
  - `--dump` is the `<dump>.parity.jsonl` slice `parity_check.py --dump` writes (exactly one run of the 50 records); the server's whole dump also holds golden and other passes and fails alignment.
  - `--tree` is the exported tree. It supplies `decision.json` (JEV bias, temperatures, slots; Kev temperature) and `decision/` (Kev pointer weights).
  - Writes `{"<record>.<qid>": probs}` in `option_keys` order, the same order `parity_check` reads.
- `def stream_hidden(weights_dir, passes, lora=None, lora_scale=None, threads=4) -> list[torch.Tensor]`: final-norm hidden `[T, H]` for every pass.
- `def lm_rows(weights_dir, ids) -> torch.Tensor`: lm_head rows in fp32.
- `def family_probs(family, rec, passes, hidden, tree, weights_dir, lora, lora_scale, clef_dir) -> dict[str, list[float]]`

- [ ] **Step 1: Write the failing test** (`tools/tests/test_reference_stream.py`)

```python
"""reference_stream: layer-streamed fp32 forward == the full HF model on a tiny random Qwen3.5.

    python -m pytest tools/tests/test_reference_stream.py -v
"""
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "decision"))
import reference_stream as rs  # noqa: E402


def tiny(tmp_path):
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            layer_types=["linear_attention", "full_attention"],
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    torch.manual_seed(0)
    model = M.Qwen3_5ForCausalLM(cfg).float().eval()
    # clone: tied embeddings share storage with lm_head and safetensors refuses shared tensors
    sd = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in sd}}))
    cfg.save_pretrained(tmp_path)
    return model


def test_streamed_hidden_matches_full_model(tmp_path):
    model = tiny(tmp_path)
    ids = [3, 9, 27, 1, 44, 5]
    with torch.no_grad():
        full = model(torch.tensor([ids]), output_hidden_states=True).hidden_states[-1][0]
    (h,) = rs.stream_hidden(str(tmp_path), [{"readout": {"prompt_ids": ids}}], threads=1)
    assert torch.allclose(h, full, atol=1e-4), float((h - full).abs().max())
```

If `Qwen3_5TextConfig`'s linear-attention field names differ in the installed transformers, take them from `transformers/models/qwen3_5/configuration_qwen3_5.py`. The test only needs one linear and one full layer with small widths.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tools/tests/test_reference_stream.py -v`
Expected: `ModuleNotFoundError: reference_stream` (or SKIP without transformers ≥ 5.2; it must PASS in the export venv).

- [ ] **Step 3: Implement `tools/decision/reference_stream.py`**

```python
"""fp32 reference for a decision tree, one decoder layer resident at a time (LoRA merged per layer).

Replays the server's --decision-dump, so it measures quantization only (codec correctness is
codec_golden.py's job). Peak memory ~ one fp32 layer + the passes' activations.

  python tools/decision/reference_stream.py jev  --weights JEV_BF16 --lora JEV/adapter_vllm/adapter_model.safetensors --tree TREE --dump d.parity.jsonl --out ref.json
  python tools/decision/reference_stream.py kev  --weights QWEN35_9B_BASE --lora KEV/adapter_model.safetensors --tree TREE --dump d.parity.jsonl --out ref.json
  python tools/decision/reference_stream.py clef --weights CLEF_DIR --clef-dir CLEF_DIR --tree TREE --dump d.parity.jsonl --out ref.json
"""
import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwen36_surgery"))
from codec_golden import align  # noqa: E402
from records import records  # noqa: E402


class Ckpt:
    def __init__(self, d):
        self.d = Path(d)
        self.idx = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]
        self.prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in self.idx) else "model."
        self._f = {}

    def _file(self, name):
        f = self.idx[name]
        if f not in self._f:
            self._f[f] = safe_open(self.d / f, "pt")
        return self._f[f]

    def tensor(self, name):
        return self._file(name).get_tensor(name).float()

    def rows(self, name, ids):
        s = self._file(name).get_slice(name)
        return torch.stack([s[int(i):int(i) + 1][0] for i in ids]).float()


class Lora:
    def __init__(self, path, scale):
        self.f = safe_open(path, "pt") if path else None
        self.keys = set(self.f.keys()) if self.f else set()
        self.scale = scale

    def delta(self, module):
        a = f"base_model.model.{module}.lora_A.weight"
        if a not in self.keys:
            a = f"base_model.model.model.{module}.lora_A.weight"
            if a not in self.keys:
                return None
        return self.scale * (self.f.get_tensor(a.replace("lora_A", "lora_B")).float() @ self.f.get_tensor(a).float())


def text_config(weights_dir):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(weights_dir)
    cfg = getattr(cfg, "text_config", cfg)
    cfg._attn_implementation = "sdpa"
    return cfg


@torch.inference_mode()
def stream_hidden(weights_dir, passes, lora=None, lora_scale=None, threads=4):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    torch.set_num_threads(threads)
    ck, cfg = Ckpt(weights_dir), text_config(weights_dir)
    lo = Lora(lora, lora_scale)
    ids = [p["readout"]["prompt_ids"] for p in passes]
    acts = [ck.rows(f"{ck.prefix}embed_tokens.weight", x).unsqueeze(0) for x in ids]
    pos = [torch.arange(a.shape[1]).view(1, 1, -1).expand(4, 1, -1) for a in acts]
    rot = M.Qwen3_5TextRotaryEmbedding(cfg)
    pe = [rot(a, p[1:]) for a, p in zip(acts, pos)]
    for i in range(cfg.num_hidden_layers):
        layer = M.Qwen3_5DecoderLayer(cfg, i).float()
        sd = {}
        for name in layer.state_dict():
            w = ck.tensor(f"{ck.prefix}layers.{i}.{name}")
            if name.endswith(".weight"):
                d = lo.delta(f"layers.{i}.{name[:-len('.weight')]}")
                if d is not None:
                    w = w + d
            sd[name] = w
        layer.load_state_dict(sd, strict=True)
        layer.eval()
        for j in range(len(acts)):
            acts[j] = layer(acts[j], position_embeddings=pe[j], attention_mask=None, position_ids=pos[j][0])
        del layer, sd
        gc.collect()
        print(f"layer {i} ({cfg.layer_types[i]})", flush=True)
    norm = M.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.load_state_dict({"weight": ck.tensor(f"{ck.prefix}norm.weight")})
    return [norm(a[0]) for a in acts]


def lm_rows(weights_dir, ids):
    ck = Ckpt(weights_dir)
    name = "lm_head.weight" if "lm_head.weight" in ck.idx else f"{ck.prefix}embed_tokens.weight"
    return ck.rows(name, ids)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def rescale_score(p6, levels):
    out = [0.0] * levels
    for j, p in enumerate(p6):
        x = j * (levels - 1) / (len(p6) - 1)
        lo = math.floor(x)
        w = x - lo
        out[lo] += p * (1 - w)
        if w > 0:
            out[lo + 1] += p * w
    return out


def family_probs(family, rec, passes, hidden, tree, weights_dir, lora, lora_scale, clef_dir):
    prof = json.loads((Path(tree) / "decision.json").read_text())
    out = {}
    for p, h in zip(passes, hidden):
        head = p["readout"]["head"]
        if family == "jev":
            j = prof["jev"]
            qid = p["parts"][0][0]
            q = rec["questions"][qid]
            kind = q["type"]
            lo_slot = j["slots"][kind][0]
            vids = head["token_ids"]
            last = h[-1]
            logits = lm_rows(weights_dir, vids) @ last
            lo = Lora(lora, lora_scale)
            d = lo.delta("lm_head")
            if d is not None:
                logits = logits + d[vids] @ last
            z = [(float(l) + j["bias"][lo_slot + i]) / j["temperature"][kind] for i, l in enumerate(logits)]
            pr = softmax(z)
            out[qid] = rescale_score(pr, len(q["criteria"])) if kind == "score" else pr
        elif family == "kev":
            qid = p["parts"][0][0]
            dd = Path(tree) / "decision"
            dp = json.loads((dd / "head.json").read_text())["dp"]
            H = h.shape[1]
            ld = lambda n, shape: torch.from_numpy(np.fromfile(dd / f"{n}.f32", dtype="<f4").reshape(shape))
            qv = ld("pointer_q_w", (dp, H)) @ h[head["decide_pos"]] + ld("pointer_q_b", (dp,))
            kv = h[head["option_pos"]] @ ld("pointer_k_w", (dp, H)).T + ld("pointer_k_b", (dp,))
            z = (kv @ qv) / math.sqrt(dp) / prof["kev"]["temperature"]
            out[qid] = softmax(z.tolist())
        else:
            from decision.clef_head import TensorHead, load_joint_head
            th = TensorHead(load_joint_head(clef_dir)).eval()
            T = h.shape[0]
            spans = [q["span"] for q in head["questions"]] + [o["span"] for q in head["questions"] for o in q["options"]]
            mask = torch.zeros(len(spans), T)
            for r, (a, b) in enumerate(spans):
                mask[r, a:b] = 1.0 / (b - a)
            ids = p["readout"]["prompt_ids"]
            lexical = torch.stack([lm_rows(weights_dir, ids[o["span"][0]:o["span"][1]]).mean(0)
                                   for q in head["questions"] for o in q["options"]])
            option_q = torch.tensor([qi for qi, q in enumerate(head["questions"]) for _ in q["options"]])
            types = torch.tensor([q["qtype"] for q in head["questions"]])
            with torch.no_grad():
                logits = th(h, mask, lexical, option_q, types).tolist()
            for (qid, a, b) in p["parts"]:
                pr = softmax(logits[a:b])
                q = rec["questions"][qid]
                if q["type"] == "noul":
                    pr = [pr[1], pr[0]]
                elif q["type"] == "choice":
                    order = sorted(q["criteria"])
                    pr = [pr[order.index(k)] for k in q["criteria"]]
                out[qid] = pr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["jev", "kev", "clef"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=None)
    ap.add_argument("--clef-dir", default=None)
    a = ap.parse_args()
    scale = a.lora_scale
    if a.lora and scale is None:
        from weight_swap import lora_scale_from_config
        scale = lora_scale_from_config(a.lora)
    aligned = list(align(a.dump, records()))
    flat = [p for _, ps in aligned for p in ps]
    hidden = stream_hidden(a.weights, flat, a.lora, scale)
    ref, k = {}, 0
    for rec, ps in aligned:
        probs = family_probs(a.family, rec, ps, hidden[k:k + len(ps)], a.tree, a.weights, a.lora, scale, a.clef_dir)
        k += len(ps)
        for qid, pr in probs.items():
            ref[f"{rec['id']}.{qid}"] = pr
    Path(a.out).write_text(json.dumps(ref))
    print(f"wrote {len(ref)} reference distributions to {a.out}")


if __name__ == "__main__":
    main()
```

`family_probs` for Clef imports Task 14's `load_joint_head` from `decision.clef_head`. Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` beside the other two inserts so `decision.clef_head` resolves.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tools/tests/test_reference_stream.py tools/tests/test_clef_head.py -v`
Expected: PASS in the export venv (transformers ≥ 5.2, torch, safetensors).

- [ ] **Step 5: Commit**

```bash
git add tools/decision/reference_stream.py tools/decision/clef_head.py tools/tests/test_reference_stream.py
git commit -m "feat(tools): layer-streamed fp32 decision reference for jev/kev/clef"
```

---

## Model acceptance (Tasks 18–22)

Each model task is manual (it needs OpenVINO and downloads), stays under 7 GB per step unless stated otherwise, and ends by appending its recipe and measured numbers to `docs/decisions.md`. Every command below assumes the runbook environment from Global Constraints (`$SPIKE_DIR`, `$PY`, `cascadia` on `PATH`). Each task runs the same six checks:
1. **Export** the tree.
2. **Serve** it with `--decision-dump`, then drive the 50 records once with `parity_check.py --dump` (no `--ref`). That writes `<dump>.parity.jsonl`: exactly this run's passes, 0600, never overwritten without `--force`.
3. **Golden:** `codec_golden.py` (it slices its own `<dump>.golden.jsonl`), plus `--answers` against the running server.
4. **Reference:** `reference_stream.py --dump <dump>.parity.jsonl`. The whole dump also holds golden, answer, duplicate-request and long-state passes, so the reference never reads it directly; the slice makes the step order free.
5. **Parity:** `parity_check.py --ref` must print `PASS`: raw agreement ≥ 96% and no flip where the reference margin is ≥ 0.05.
6. **OpenAI route equivalence:** `openai_equivalence.py --url …` must report every question identical (JEV: add `--all-states`). This carries checks 3–5 over to `/v1/decisions`.

The dump is opened in append mode, so every task first removes the previous files: `rm -f /tmp/<model>.jsonl /tmp/<model>.parity.jsonl /tmp/<model>.golden.jsonl`.

Every long step (exports, reference runs, server bring-up) runs in the background with a Monitor on its log. The Monitor watches for completion **and** for failure signatures (`Traceback`, `Error`, `killed`, `CAP`, `MISMATCH`, `FLIP`, `epoch`, non-zero `exit=`), so silence can't look like success. Wrap exports and references in `$SPIKE_DIR/jev-spike/run_capped.py` (`CAP_GB=7`), which watches phys_footprint and kills the step if it goes over. `run_capped.py` is **macOS-only** (it loads `/usr/lib/libproc.dylib`): on a Linux box run the step under `/usr/bin/time -v` and a cgroup memory limit instead (`systemd-run --user --scope -p MemoryMax=7G …`).

### Task 18: JEV-9B

- [ ] **Step 1: Download and export**

```bash
hf download OpenVINO/Qwen3.5-9B-int8-ov --local-dir ~/models/qwen35-9b-int8
hf download autotrust/JEV-9B --include "adapter_vllm/*" --include calibration.json --include tokenizer.json --include tokenizer_config.json --local-dir ~/models/jev9b
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision_export.py jev --base-ir ~/models/qwen35-9b-int8 --adapter-dir ~/models/jev9b/adapter_vllm --out ~/trees/jev9b-int8
```

Expected: 8 stages, `grafted N LoRA branches` summing to the adapter's target count (201), and `decision tree written`.

- [ ] **Step 2: Serve and drive the parity run**

```bash
rm -f /tmp/jev9b.jsonl /tmp/jev9b.parity.jsonl /tmp/jev9b.golden.jsonl
cascadia run ~/trees/jev9b-int8 --engine qwen35 --device CPU --api 127.0.0.1:8080 --decision-dump /tmp/jev9b.jsonl > /tmp/jev9b-server.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health > /dev/null; do sleep 2; done
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump /tmp/jev9b.jsonl
```

Expected: `wrote 82 passes for 50 records to /tmp/jev9b.parity.jsonl`.

- [ ] **Step 3: Golden check**

```bash
$PY tools/decision/codec_golden.py jev --url http://127.0.0.1:8080 --dump /tmp/jev9b.jsonl --tokenizer ~/models/jev9b --adapter-dir ~/models/jev9b/adapter_vllm --answers
```

Expected:
- `jev: 86 passes match the serve_decide layout and decision_head.json` (82 record passes + 4 synthetic);
- `jev: 82 answer-run prompts equal jev.py's (choices sorted)`: `--answers` sends each choice question with its criteria sorted, which is the order `jev.py` and `reference_jev.pt` use, and checks the server's dumped prompt ids against `jev.questions()`;
- `jev: server answers agree with the saved reference on N/82` with N ≥ 81 (98%). All 82 are gated, the 33 choice answers included.

- [ ] **Step 4: Reference and parity**

The reference needs JEV's BF16 shards: `hf download autotrust/JEV-9B --include "model-*" --include "model.safetensors.index.json" --include config.json --local-dir ~/models/jev9b`.

```bash
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision/reference_stream.py jev --weights ~/models/jev9b \
    --lora ~/models/jev9b/adapter_vllm/adapter_model.safetensors --tree ~/trees/jev9b-int8 \
    --dump /tmp/jev9b.parity.jsonl --out /tmp/jev9b-ref.json
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref /tmp/jev9b-ref.json
```

Expected: `wrote 82 reference distributions`, then `PASS`. This is the **re-measure** that decision A requires: request-order choices and rescaled scores, replacing the spike's 98.8% (sorted choices, 6-way scores). Record the per-type lines too.

- [ ] **Step 4b: OpenAI route equivalence**: `$PY tools/decision/openai_equivalence.py --url http://127.0.0.1:8080 --all-states`. Expected: every question identical across `/v1/decisions` and `/v1/systemone`.

- [ ] **Step 5: Docs and commit** (append the JEV-9B section to `docs/decisions.md`: the recipe above, the measured agreement, per-type counts, flips)

```bash
git add docs/decisions.md
git commit -m "docs(decisions): JEV-9B recipe and measured parity"
```

### Task 19: Kev-9B

- [ ] **Step 1: Download and export**

```bash
hf download Qwen/Qwen3.5-9B-Base --revision 68c46c4b3498877f3ef123c856ecfde50c39f404 --local-dir ~/models/qwen35-9b-base
hf download jaredpalmer/kev-9b adapter_model.safetensors head.pt tokenizer.json --local-dir ~/models/kev9b
hf download jaredpalmer/kev-9b adapter_config.json --local-dir ~/models/kev9b \
    || cp $SPIKE_DIR/kev-spike/adapter_config.json ~/models/kev9b/adapter_config.json
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision_export.py kev --base-ir ~/models/qwen35-9b-int8 --weights ~/models/qwen35-9b-base \
    --lora ~/models/kev9b/adapter_model.safetensors --head-pt ~/models/kev9b/head.pt --out ~/trees/kev9b-int8
```

`adapter_config.json` is downloaded on its own, because `hf download` fails outright when a named file is missing. If the repo has none, the spike's copy (`$SPIKE_DIR/kev-spike/adapter_config.json`: r=16, lora_alpha=32, no DoRA, no rank/alpha patterns, the same 12 target modules) goes beside the adapter instead; note that in the docs. Do not pass `--lora-scale` as a fallback: the exporter and `reference_stream.py` both read `adapter_config.json`, and its refusals (DoRA, patterns, `lora_bias`) must always run. Expected: `check_swap` passes on all 8 stages, and 248 LoRA branches grafted.

- [ ] **Step 2: Serve and drive the parity run**

```bash
rm -f /tmp/kev9b.jsonl /tmp/kev9b.parity.jsonl /tmp/kev9b.golden.jsonl
cascadia run ~/trees/kev9b-int8 --engine qwen35 --device CPU --api 127.0.0.1:8080 --decision-dump /tmp/kev9b.jsonl > /tmp/kev9b-server.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health > /dev/null; do sleep 2; done
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump /tmp/kev9b.jsonl
```

Expected: `wrote 82 passes for 50 records to /tmp/kev9b.parity.jsonl`. Then check the state reuse works:
- the server log shows `qwen35 prefix-cache hit` from the second question of each multi-question record whose `<|fim_prefix|>` + state is ≥ 16 tokens (12 of the 24 multi-question records; shorter states are below `MIN_PREFIX_TOKENS` and never snapshot);
- one synthetic long-state request (a ~8k-token state with 5 noul questions) shows one cold prefill, then hits whose `warm_prefix` equals the state length:

  ```bash
  $PY - <<'EOF'
  import json
  state = "The quick brown fox jumps over the lazy dog. " * 900   # about 8k tokens
  qs = {f"q{i}": {"type": "noul", "instructions": f"Does the text mention animal number {i}?"} for i in range(5)}
  json.dump({"model": "kev", "state": state, "questions": qs}, open("/tmp/kev-long.json", "w"))
  EOF
  curl -s -H 'content-type: application/json' --data-binary @/tmp/kev-long.json http://127.0.0.1:8080/v1/systemone -w ' %{http_code}\n'
  grep -n 'prefix-cache hit' /tmp/kev9b-server.log | tail -4
  ```

- [ ] **Step 3: Golden check**

```bash
$PY tools/decision/codec_golden.py kev --url http://127.0.0.1:8080 --dump /tmp/kev9b.jsonl --tokenizer ~/models/kev9b --answers
```

Expected: `kev: 86 passes match kev.questions` (82 + 4 synthetic), and `kev: server answers agree with the saved reference on N/82` with N ≥ 81 against `reference_kev.pt`.

- [ ] **Step 4: Reference and parity**

```bash
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision/reference_stream.py kev --weights ~/models/qwen35-9b-base \
    --lora ~/models/kev9b/adapter_model.safetensors --tree ~/trees/kev9b-int8 \
    --dump /tmp/kev9b.parity.jsonl --out /tmp/kev9b-ref.json
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref /tmp/kev9b-ref.json
```

Expected: `PASS` (spike: 82/82).

- [ ] **Step 4b: OpenAI route equivalence**: `$PY tools/decision/openai_equivalence.py --url http://127.0.0.1:8080`. Expected: every question identical across `/v1/decisions` and `/v1/systemone`.

- [ ] **Step 5: Docs and commit** (Kev-9B section; `git commit -m "docs(decisions): Kev-9B recipe and measured parity"`)

### Task 20: Clef-flash

- [ ] **Step 1: Download and export**

```bash
hf download Cloudflare/clef-flash --local-dir ~/models/clef-flash
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision_export.py clef --base-ir ~/models/qwen35-9b-int8 --weights ~/models/clef-flash --clef-dir ~/models/clef-flash --out ~/trees/clef-flash-int8
```

- [ ] **Step 2: Serve and drive the parity run**

```bash
rm -f /tmp/clef-flash.jsonl /tmp/clef-flash.parity.jsonl /tmp/clef-flash.golden.jsonl
cascadia run ~/trees/clef-flash-int8 --engine qwen35 --device CPU --api 127.0.0.1:8080 --decision-dump /tmp/clef-flash.jsonl > /tmp/clef-flash-server.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health > /dev/null; do sleep 2; done
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump /tmp/clef-flash.jsonl
```

Expected: `wrote 50 passes for 50 records to /tmp/clef-flash.parity.jsonl` (one joint pass per record). Then send the same request twice and confirm both return 200. This is the regression check for the prefix-cache bug: Joint passes always prefill cold.

```bash
for i in 1 2; do curl -s -H 'content-type: application/json' http://127.0.0.1:8080/v1/systemone \
  -d '{"model":"clef","state":"I was charged twice for my order.","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"charges","tech":"bugs"}},"urgent":{"type":"noul","instructions":"Is it urgent?"}}}' -w ' %{http_code}\n'; done
```

- [ ] **Step 3: Golden check**

```bash
$PY tools/decision/codec_golden.py clef --url http://127.0.0.1:8080 --dump /tmp/clef-flash.jsonl --tokenizer ~/models/clef-flash --answers
```

Expected: `clef: 54 records match encode_record` (50 + 4 synthetic; the 50 also match the saved `input_ids`), and answers ≥ 98% (N ≥ 81 of 82) against `reference_flash.pt`. That answer check is what catches noul inversion.

- [ ] **Step 4: Reference and parity**

```bash
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision/reference_stream.py clef --weights ~/models/clef-flash \
    --clef-dir ~/models/clef-flash --tree ~/trees/clef-flash-int8 \
    --dump /tmp/clef-flash.parity.jsonl --out /tmp/clef-flash-ref.json
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref /tmp/clef-flash-ref.json
```

Expected: `PASS` (spike: 82/82).

- [ ] **Step 4b: OpenAI route equivalence**: `$PY tools/decision/openai_equivalence.py --url http://127.0.0.1:8080`. Expected: every question identical across `/v1/decisions` and `/v1/systemone`.

- [ ] **Step 5: Docs and commit** (Clef-flash section; `git commit -m "docs(decisions): Clef-flash recipe and measured parity"`)

### Two-machine runbook (Tasks 21 and 22)

Host A runs rank 0 (the API, the dump and every tool); host B runs rank 1. `RANK1_HOST` is host B's LAN address. Both hosts need the **whole tree at the same `--model` path** (rank ≥ 1 reads `config.json` and, as the last rank, the `decision/` assets), the same branch build of `cascadia`, and OpenVINO 2026.4.1:

```bash
# on host A, after the export
rsync -a ~/trees/<tree>/ RANK1_HOST:trees/<tree>/
# host B (rank 1 listens on the LAN; start it first)
cascadia worker --model ~/trees/<tree> --engine qwen35 --device CPU --rank 1 --total 2 --listen 0.0.0.0:7801 > /tmp/rank1.log 2>&1 &
# host A (rank 0 dials rank 1 once, at startup; the API stays on loopback because the dump holds request text)
rm -f /tmp/<model>.jsonl /tmp/<model>.parity.jsonl /tmp/<model>.golden.jsonl
cascadia worker --model ~/trees/<tree> --engine qwen35 --device CPU --rank 0 --total 2 --next RANK1_HOST:7801 \
    --api 127.0.0.1:8080 --decision-dump /tmp/<model>.jsonl > /tmp/rank0.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health > /dev/null; do sleep 2; done
```

Monitor both logs. If rank 1 restarts, restart rank 0 too (it never re-dials).

### Task 21: JEV-27B (pipeline, 2 machines)

The 27B models were never measured in the spike, so this task establishes the number instead of confirming one.

- [ ] **Step 1: Check the base, then export**

```bash
hf download OpenVINO/Qwen3.8-27B-int8-ov --local-dir ~/models/qwen38-27b-int8
hf download autotrust/JEV-27B --include "adapter_vllm/*" --include calibration.json --include tokenizer.json --include tokenizer_config.json --local-dir ~/models/jev27b
$PY tools/decision/backbone_diff.py autotrust/JEV-27B Qwen/Qwen3.8-27B --per-layer
```

`backbone_diff.py` matches tensors per layer index and suffix (a text-only `model.layers.N.*` checkpoint lines up with a VLM `model.language_model.layers.N.*` one), hashes every backbone tensor (all projections, norms, embeddings, final norm and lm_head) and ends with one line:
- `VERDICT: STOCK`: the backbone is stock Qwen3.8-27B. Export without `--weights`.
- `VERDICT: FINETUNED`: export with `--weights ~/models/jev27b` (download the shards first, below). `swap_stage` never swaps `lm_head`, and JEV reads verbalizer logits from it: if the per-tensor lines show `lm_head.weight` DIFF, stop and extend `swap_stage` to the lm_head constants before continuing.
- `VERDICT: UNKNOWN` (exit 2): nothing comparable was found. **Stop** and inspect the two repos' headers; do not export.

Either way the reference (Step 4) runs on JEV-27B's **own** shards, so it is independent of this call: a wrong STOCK verdict then fails parity instead of passing a circular one.

```bash
hf download autotrust/JEV-27B --include "model-*" --include "model.safetensors.index.json" --include config.json --local-dir ~/models/jev27b
# STOCK:
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision_export.py jev --base-ir ~/models/qwen38-27b-int8 --adapter-dir ~/models/jev27b/adapter_vllm --stages 2 --out ~/trees/jev27b-int8
# FINETUNED: the same command plus --weights ~/models/jev27b
```

- [ ] **Step 2: Serve on two machines** per the two-machine runbook above, with `<tree>` = `jev27b-int8` and `<model>` = `jev27b`. Then drive the parity run: `$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump /tmp/jev27b.jsonl` (expected: `wrote 82 passes for 50 records to /tmp/jev27b.parity.jsonl`).

- [ ] **Step 3: Golden**

```bash
$PY tools/decision/codec_golden.py jev --url http://127.0.0.1:8080 --dump /tmp/jev27b.jsonl --tokenizer ~/models/jev27b --adapter-dir ~/models/jev27b/adapter_vllm
```

No `--answers`: there is no saved 27B reference. Expected: `jev: 86 passes match the serve_decide layout and decision_head.json`.

- [ ] **Step 4: Reference and parity.** The reference always uses JEV-27B's own BF16 shards, the same backbone the tree serves (stock, or `--weights ~/models/jev27b`); never the stock Qwen3.8-27B checkpoint. The peak is about 4 GB (one fp32 decoder layer plus its loaded state dict, ~3.4 GB, plus one LoRA delta), under the cap.

```bash
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision/reference_stream.py jev --weights ~/models/jev27b \
    --lora ~/models/jev27b/adapter_vllm/adapter_model.safetensors --tree ~/trees/jev27b-int8 \
    --dump /tmp/jev27b.parity.jsonl --out /tmp/jev27b-ref.json
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref /tmp/jev27b-ref.json
```

Expected: `PASS`. If it fails, record the number, the per-type breakdown and the confident flips in the docs, and raise it with the user before calling the model supported.

- [ ] **Step 4b: OpenAI route equivalence**: `$PY tools/decision/openai_equivalence.py --url http://127.0.0.1:8080 --all-states`. Expected: every question identical across `/v1/decisions` and `/v1/systemone`.

- [ ] **Step 5: Docs and commit** (JEV-27B section, including the backbone verdict; `git commit -m "docs(decisions): JEV-27B recipe and measured parity"`)

### Task 22: Clef 27B (pipeline, 2 machines)

- [ ] **Step 1: Download and export.** This needs a machine with ~16 GB free: each of the 2 swapped stages is ~13 GB of new int8 constants. This is the one export above the 7 GB cap, and it is stated in the docs, so it is not wrapped in `run_capped.py`. `check_swap` runs on both stages, which is the first time the float-kind table meets the Qwen3.8-27B IR. A failure there means a node-name difference: fix `FLOAT_KINDS` and rerun. Never bypass the check. `swap_embeddings` also meets the 27B embeddings IR for the first time: if it stops with `embeddings are not i8 symmetric with one scale per row`, inspect that IR's embedding constants before changing anything.

```bash
hf download Cloudflare/clef --local-dir ~/models/clef
$PY tools/decision_export.py clef --base-ir ~/models/qwen38-27b-int8 --weights ~/models/clef --clef-dir ~/models/clef --stages 2 --out ~/trees/clef-int8
```

- [ ] **Step 2: Serve on two machines** per the two-machine runbook above, with `<tree>` = `clef-int8` and `<model>` = `clef27b`. Then drive the parity run: `$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump /tmp/clef27b.jsonl` (expected: `wrote 50 passes for 50 records to /tmp/clef27b.parity.jsonl`).

- [ ] **Step 3: Golden**

```bash
$PY tools/decision/codec_golden.py clef --url http://127.0.0.1:8080 --dump /tmp/clef27b.jsonl --tokenizer ~/models/clef
```

No `--answers`; the 50 records' saved `input_ids` check still applies, since Clef 27B shares the tokenizer. Expected: `clef: 54 records match encode_record`.

- [ ] **Step 4: Reference and parity**

```bash
CAP_GB=7 $PY $SPIKE_DIR/jev-spike/run_capped.py $PY tools/decision/reference_stream.py clef --weights ~/models/clef \
    --clef-dir ~/models/clef --tree ~/trees/clef-int8 \
    --dump /tmp/clef27b.parity.jsonl --out /tmp/clef27b-ref.json
$PY tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref /tmp/clef27b-ref.json
```

Expected: `PASS`, with the same escalation rule as Task 21.

- [ ] **Step 4b: OpenAI route equivalence**: `$PY tools/decision/openai_equivalence.py --url http://127.0.0.1:8080`. Expected: every question identical across `/v1/decisions` and `/v1/systemone`.

- [ ] **Step 5: Docs and commit** (Clef-27B section, including the 16 GB export note; `git commit -m "docs(decisions): Clef 27B recipe and measured parity"`)

---

## Self-Review Notes

- **Spec coverage:**
  - All five models: Tasks 18–22, one each (export → serve → golden → reference → parity → docs).
  - API: Tasks 3, 4 and 8 (docs/decisions.md starts in Task 8).
  - Codecs: Tasks 5–7.
  - Engine, single box and pipeline: Tasks 9–11.
  - Exporter: Tasks 12–15.
  - int8-only: the Global Constraints, and `decision_export` always uses the stock int8 IR.
  - Memory cap: 8-stage single-box exports, per-stage swap with `gc`, and `run_capped.py` on every step. Clef-27B's export is the one stated exception.
  - Acceptance: golden (Task 16) plus parity (Task 17's reference, Tasks 18–22).
- **Type consistency:**
  - `DecisionReadout { prompt_ids, head, cache_prefix }`, `ReadoutHead::{Verbalizer, Pointer, Joint}`, `JointQuestion { qtype, span, options }` and `JointOption { span }` are used identically in Tasks 1, 2, 5–7, 9, 10, 11, 16 and 17.
  - The joint IR input names `hidden/span_mask/lexical/option_q/type_ids` match between Task 9 (`run_head`) and Task 14 (`export_module`).
  - `decision/head.json` kinds and file names match between Task 9 (`HeadAssets::load`), Task 15 (writers) and Task 17 (Kev pointer files).
  - The `final_hidden_output` manifest key matches between Task 10 (`Manifest`) and Task 13 (`build_manifest`).
  - Probability order: every codec returns `option_keys` order (noul `[false, true]`, choice in request order, score by level); `parity_check.answer_probs` and `reference_stream.family_probs` use the same order.
  - The dump line `{"parts": [[qid, start, end]], "readout": …}` matches between Task 8 (writer), Task 16 (`align`) and Task 17.
- **Review round 1 (2026-10-04), applied:**
  - Clef noul order.
  - The prefix cache: bounded warm resume, never for Joint.
  - Pipeline rank 0 tokenizing through `admitted_prompt_ids`.
  - HELLO_ACK capability bits.
  - Chunked decision bodies.
  - Readout validation on every admission.
  - Question and total-token caps, with the permit taken before encoding.
  - Kev state reuse through `cache_prefix`.
  - The codec golden check.
  - The LoRA scale read from `adapter_config.json`.
  - JEV request order and score rescaling.
  - TypeSafe confidence and Python rounding.
  - 8-stage exports.
  - Control-token sanitizing in every codec.
  - Generic client errors.
  - `--decision-dump` replacing the env var.
  - Task splits (16/17, per-model 18–22) and `SPIKE_DIR`.
  - Name fixes, the qwen35-only CLI gate, `check_swap` and graft counts.
- **Stress test (2026-10-04), applied:**
  - Parity gate: raw ≥ 96% plus no confident flip, replacing near-tie exclusion. JEV has 5 of 82 reference margins under 0.02.
  - The total-token budget counts a shared prefix once (Kev state reuse).
  - `max_questions` is configurable in `decision.json`.
  - Docs notes: JEV score rescaling is an approximation; TypeSafe confidence differs from Clef's own server.
- **Review round 2 (2026-10-04), applied:**
  - Compile and crash fixes: the generic `RawRecv` trait for chunked reads on both link ends; the Clef golden check reading `reference_flash.pt` per record (plus the saved `input_ids`); rank 0 carrying `decision` on its `ActiveTask`; `last_logits` declared.
  - Chat isolation: `vocab` read tolerantly (`Option`, refusing only decisions); `want_final_hidden` reset on every admission, on cancel and on pipeline RESET.
  - Security:
    - decisions-specific `decision_engine_error` (no `err.to_string()` to clients);
    - `prefill_work(…, prefix_reuse)`: the shared-prefix discount applies only on a single box with the prefix cache on;
    - O(n) duplicate-key handling in `OrderedValue`;
    - `encode` in `spawn_blocking` behind a byte pre-check;
    - spans required in order and disjoint;
    - `MAX_DECISION_PROMPT` where `max_ctx` is unknown;
    - dump opened with `O_NOFOLLOW` and chmod 0600 on existing files;
    - `torch.load(weights_only=True)`;
    - Clef total options capped at `MAX_DECISION_OUTPUTS` in the API;
    - `Sanitizer` covering every angle-bracket added token, not just `<|…|>`.
  - Fidelity:
    - JEV state rendered like serve_decide `_parts` (lists concatenated, image parts rejected);
    - floats rendered with Python's repr (`py_float`);
    - JEV golden checks verbalizer `token_ids` against `decision_head.json` and gates noul and score answers;
    - synthetic golden-only records cover list states, criteria overrides, dict descriptions, empty and null instructions, and floats;
    - Kev reuse check reworded for the 16-token minimum, plus one long-state request.
  - Structure: Task 8 split into 8 (API) and 8b (CLI, docs); Task 11 split into 11 (capability bits, chunked I/O) and 11b (decision frames); the pipeline smoke test moved to Task 15 Step 6 with the real `cascadia worker` flags (`--model`, `--next`); the mock failure sentinel moved into Task 2; stale references fixed.
  - Round 3 (verify-only): consistency check clean. The code check verified the new pieces against the code, the transport API, tokenizers 0.21 and the saved references (it actually loaded them and ran the encoders), plus `py_float` on 200,000 random floats. Fixed its one finding: pass `&mut *g` from the MutexGuard to the generic readers. Also corrected the `codec_golden.py` usage lines.
  - Kept: the parity gate at 0.05.
- **OpenAI Decisions API (2026-10-07), added:** OpenAI published its Decisions API at the same path, `POST /v1/decisions`, with a different shape (`input`, a `questions` array with `name`, `predicate`/`choice`/`score`, array-valued probabilities). `/v1/decisions` now serves that format through the Task 8c adapter, and the Jev/SystemOne format moved to `/v1/systemone`. Both routes share Task 8's `evaluate` core, so codecs, engine and models are unchanged. `openai_equivalence.py` (Task 16, acceptance step 4b) proves the two routes answer identically on the records. OpenAI's documented confidences (0.93, 0.55) match the TypeSafe formulas already in Task 3. A failure caused only by flips with reference margins of 0.05–0.16 escalates to the user rather than rejecting the model, since int8 moved probabilities by up to 0.08 in the spike.
- **Known risks** to check while executing:
  1. `PluginConfig::default()` existence (Task 9 Step 3 gives the fallback).
  2. The exact name of `make_router_with_config` (Task 8 gives the fallback).
  3. qwen35 pipeline flags for `cascadia worker` are taken from `WorkerArgs` (`--model`, `--rank`, `--total`, `--listen`, `--next`; Task 15 Step 6). Re-check with `--help` if the CLI has moved.
  4. Whether JEV-27B's base (and lm_head) is stock (Task 21 Step 1 checks it, with a fallback).
  5. Floats render with Python's repr (`py_float`, tested against repr output). The remaining gap is integers beyond u64, which serde_json parses as f64 while Python keeps an int. The golden check's synthetic records include floats, so a regression shows up there.
  6. The Qwen3.8-27B IR's node names were never checked against `FLOAT_KINDS`; `check_swap` stops Task 22 if they differ.
  7. The 27B models were never measured; Tasks 21–22 establish their numbers, with an escalation rule if they fail the parity gate.

## Open Questions

### From 2026-10-04 review

- **JEV choices with more than 16 options.** serve_decide answers these with a tournament: groups of ≤16 in parallel, then a final among the group winners. That is multi-pass orchestration in the API. For now such requests get 400. Implementing it means a codec that returns passes in two rounds.
- **CI for the Python tests.** `tools/tests/test_{weight_swap,decision_export,clef_head,decision_tools,reference_stream}.py` run by hand; no workflow invokes pytest for `tools/tests`. The OpenVINO and transformers tests need those wheels in CI.
- **OpenAI `refusal` answers.** The adapter never emits them, because these models have no refusal behaviour. Callers that branch on `refusal` (as OpenAI's examples do) simply never take that branch. Revisit if a model with a refusal signal is added.
- **OpenAI score-level rendering.** OpenAI levels have `label` and `description`, while the models were trained on one text per level. `"label: description"` is a reasoned default, unmeasured. A small A/B against `"description"`-only on Kev and Clef would settle it.
- **Pipeline peer authentication.** Task 11b treats FRAME_DECIDE as untrusted and validates it, but the activation port itself has no authentication in this plan. Anyone who can reach a worker's port can open a link. This predates decisions, since it applies to chat frames too, but decisions widen what a peer can send.
- **A separate permit pool for decisions.** A decision request holds one chat permit for all of its passes (up to 32 questions), so long Kev or JEV requests can delay chat on a mixed server. Options: a dedicated small pool, or a per-request time budget.
