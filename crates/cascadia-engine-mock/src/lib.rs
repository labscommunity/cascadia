//! Mock engine for tests.
//!
//! Deterministically echoes the prompt back, one token per `step()`,
//! capped at `max_tokens`. Useful for testing the runner, API, and CLI
//! without real model weights. A tools request carrying the
//! `__mock_tool_call__` sentinel gets a scripted tool call instead (see
//! [`scripted_reply`]).

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;

#[derive(Default)]
pub struct MockEngine {
    pending: Vec<(GenerationTask, usize)>,
    fail_next_step: Option<String>,
    /// Scripted replies (tool-call sentinel), keyed by task id: the chunks
    /// `step()` emits instead of the echo. Removed on cancel and at the end.
    scripted: std::collections::HashMap<TaskId, Vec<String>>,
}

impl MockEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Test hook: make the next `step()` return `EngineError::Backend(msg)`.
    pub fn fail_next_step(&mut self, msg: impl Into<String>) {
        self.fail_next_step = Some(msg.into());
    }
}

impl Engine for MockEngine {
    fn warmup(&mut self) {}

    fn cancel(&mut self, task_id: &TaskId) {
        // Step-wise engine: drop the task from the queue so step() stops
        // emitting its words. cancel() and step() never overlap (both
        // &mut self behind the runner mutex), so this is race-free.
        self.pending.retain(|(t, _)| &t.task_id != task_id);
        self.scripted.remove(task_id);
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        // Images are accepted and ignored on purpose: the API's image tests
        // use the mock as an image-capable stand-in (the echo shows where each
        // placeholder landed). Real text-only engines refuse it in `submit`
        // via `cascadia_engine::refuse_media`. A native video item is refused,
        // as on every engine but ov-genai.
        cascadia_engine::refuse_video(&task, "mock")?;
        if self.pending.iter().any(|(t, _)| t.task_id == task.task_id) {
            return Ok(());
        }
        // Test sentinel: a prompt containing `__prompt_too_long__` is refused
        // at submit as over the window, the way a real engine refuses a
        // prompt past its context, so the API's over-window shape can be
        // tested on every route without a tokenizer.
        if task.prompt.contains("__prompt_too_long__") {
            return Err(EngineError::PromptTooLong(
                "prompt is 9000 tokens; the model's window is 8192".into(),
            ));
        }
        // Option B resume: the seed ids are already-emitted, so start the
        // echo cursor past them. They are never re-emitted by step().
        //
        // Id-FIDELITY, not just length: the mock's own emitted id for position
        // i is exactly `i` (see step()), so a resume seed claiming this stream
        // must be [0..K). Length-only checking let a splicer that shuffled or
        // corrupted ids pass every mock-based integration test (review).
        let seed = task.resume_ids().map(<[i32]>::to_vec).unwrap_or_default();
        if let Some((i, &bad)) = seed.iter().enumerate().find(|(i, &id)| id != *i as i32) {
            return Err(EngineError::Backend(format!(
                "mock resume seed id mismatch at {i}: got {bad}, this stream emitted {i}"
            )));
        }
        if let Some(script) = scripted_reply(&task) {
            self.scripted.insert(task.task_id.clone(), script);
        }
        self.pending.push((task, seed.len()));
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if let Some(msg) = self.fail_next_step.take() {
            return Err(EngineError::Backend(msg));
        }
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let (task, emitted) = self.pending.remove(0);
        // Test sentinel: a prompt containing `__engine_error__` fails the
        // task loud (emits an error chunk) so consumers' failure paths —
        // e.g. the API's 5xx mapping — can be exercised deterministically.
        if task.prompt.contains("__engine_error__") {
            self.scripted.remove(&task.task_id);
            return Ok(vec![(
                task.task_id.clone(),
                Chunk::error(task.task_id, "mock injected engine failure"),
            )]);
        }
        let max = task.max_tokens.max(1) as usize;
        let script = self.scripted.get(&task.task_id).cloned();
        let words: Vec<&str> = match script {
            Some(ref chunks) => chunks.iter().map(String::as_str).collect(),
            None => task.prompt.split_whitespace().collect(),
        };
        if emitted >= max || emitted >= words.len() {
            self.scripted.remove(&task.task_id);
            // Distinguish the stop reason like a real engine: exhausting the
            // prompt's words is a natural `stop`; hitting the token cap first
            // is `length`. Lets the API's finish_reason path be tested without
            // a real model.
            let reason = if emitted >= words.len() {
                cascadia_types::FinishReason::Stop
            } else {
                cascadia_types::FinishReason::Length
            };
            return Ok(vec![(
                task.task_id.clone(),
                Chunk::final_marker(task.task_id, "").with_finish_reason(reason),
            )]);
        }
        // Scripted chunks are emitted verbatim; echo words get a space.
        let text = match script {
            Some(_) => words[emitted].to_string(),
            None => format!("{} ", words[emitted]),
        };
        let chunk =
            Chunk::token(&task.task_id, emitted as i64, text).with_token_ids(vec![emitted as i64]);
        let task_id = task.task_id.clone();
        self.pending.push((task, emitted + 1));
        Ok(vec![(task_id, chunk)])
    }
}

/// Test sentinel for tool loops: a user turn containing it, on a request
/// that offers tools, gets one scripted call of the first tool.
const TOOL_CALL_SENTINEL: &str = "__mock_tool_call__";

/// The scripted reply for a tools request, or `None` for the plain echo.
/// Applies only when `tools` is a non-empty array and `tool_choice` is not
/// `"none"`; the first matching rule wins:
/// 1. last turn is `tool` -> `mock: got <first 40 chars of its content>`;
/// 2. last turn is a `user` turn with the sentinel -> one `<tool_call>` of
///    `tools[0]`, its required arguments filled by type (3 chunks);
/// 3. an earlier user turn had the sentinel, the last user turn doesn't ->
///    `mock: ok`.
fn scripted_reply(task: &GenerationTask) -> Option<Vec<String>> {
    let tools = task.tools.as_ref()?.as_array().filter(|t| !t.is_empty())?;
    if task.tool_choice.as_ref().and_then(|c| c.as_str()) == Some("none") {
        return None;
    }
    let last = task.messages.last()?;
    if last.role == "tool" {
        let head: String = last.content.chars().take(40).collect();
        return Some(word_chunks(&format!("mock: got {head}")));
    }
    if last.role == "user" && last.content.contains(TOOL_CALL_SENTINEL) {
        let function = &tools[0]["function"];
        let name = function["name"].as_str().unwrap_or_default();
        let params = &function["parameters"];
        let mut args = serde_json::Map::new();
        for key in params["required"].as_array().into_iter().flatten() {
            let Some(key) = key.as_str() else { continue };
            let value = match params["properties"][key]["type"].as_str() {
                Some("string") => serde_json::json!("mock"),
                Some("number" | "integer") => serde_json::json!(1),
                Some("boolean") => serde_json::json!(true),
                Some("array") => serde_json::json!([]),
                Some("object") => serde_json::json!({}),
                _ => serde_json::Value::Null,
            };
            args.insert(key.to_owned(), value);
        }
        // Built by hand so `name` precedes `arguments` whatever the map order.
        let call = format!(
            "{{\"name\":{},\"arguments\":{}}}",
            serde_json::Value::from(name),
            serde_json::Value::Object(args)
        );
        return Some(vec!["<tool_call>".into(), call, "</tool_call>".into()]);
    }
    let mut users = task.messages.iter().filter(|m| m.role == "user");
    let last_user = users.next_back()?;
    if !last_user.content.contains(TOOL_CALL_SENTINEL)
        && users.any(|m| m.content.contains(TOOL_CALL_SENTINEL))
    {
        return Some(word_chunks("mock: ok"));
    }
    None
}

/// Split `text` into word chunks that join back to exactly `text`.
fn word_chunks(text: &str) -> Vec<String> {
    text.split_inclusive(' ').map(str::to_owned).collect()
}

#[derive(Default)]
pub struct MockBuilder {
    connected: bool,
    loaded: bool,
}

impl MockBuilder {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Builder for MockBuilder {
    async fn connect(&mut self, _peers: PeerLayout) -> EngineResult<()> {
        self.connected = true;
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        self.loaded = true;
        let progress = vec![
            LoadProgress::message("mock load: starting"),
            LoadProgress::ready(),
        ];
        Ok(Box::pin(stream::iter(progress)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        if !self.loaded {
            return Err(EngineError::NotLoaded);
        }
        Ok(Box::new(MockEngine::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::{PeerLayout, ShardSpec};

    #[tokio::test]
    async fn build_after_connect_and_load() {
        let mut builder = MockBuilder::new();
        builder.connect(PeerLayout::single_stage()).await.unwrap();
        let _stream = builder
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .unwrap();
        let engine = Box::new(builder).build().expect("build should succeed");
        let _ = engine; // moved
    }

    #[tokio::test]
    async fn build_before_load_errors() {
        let mut builder = MockBuilder::new();
        builder.connect(PeerLayout::single_stage()).await.unwrap();
        let res = Box::new(builder).build();
        assert!(matches!(res, Err(EngineError::NotLoaded)));
    }

    #[test]
    fn submit_then_step_emits_words() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "the quick brown fox").with_max_tokens(2))
            .unwrap();
        let mut emitted = Vec::new();
        for _ in 0..6 {
            for (_, chunk) in e.step().unwrap() {
                emitted.push(chunk);
            }
        }
        // Two tokens then a final marker.
        assert_eq!(emitted.len(), 3);
        assert!(emitted[0].text.starts_with("the"));
        assert!(emitted[1].text.starts_with("quick"));
        assert!(emitted[2].is_final);
    }

    #[test]
    fn cancel_drops_pending_task_and_stops_emission() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "the quick brown fox jumps").with_max_tokens(64))
            .unwrap();
        // One token out, then cancel mid-generation.
        let first = e.step().unwrap();
        assert_eq!(first.len(), 1);
        assert!(first[0].1.text.starts_with("the"));
        e.cancel(&"t1".to_string());
        // Task is gone — no further compute / emission.
        assert!(e.step().unwrap().is_empty());
        // Cancelling an unknown id is a harmless no-op.
        e.cancel(&"nope".to_string());
    }

    #[test]
    fn duplicate_submit_is_noop() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "hi")).unwrap();
        e.submit(GenerationTask::new("t1", "hi")).unwrap();
        // We should still have exactly one task pending.
        let chunks = e.step().unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn cancel_pending_task_removes_it() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "alpha bravo")).unwrap();
        e.submit(GenerationTask::new("t2", "charlie delta"))
            .unwrap();
        e.cancel(&"t1".to_string());
        // Only t2 ever emits; t1 is gone.
        for _ in 0..8 {
            for (tid, _) in e.step().unwrap() {
                assert_eq!(tid, "t2");
            }
        }
    }

    #[test]
    fn cancel_active_task_lets_next_task_activate_immediately() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "alpha bravo charlie").with_max_tokens(8))
            .unwrap();
        // t1 is mid-generation (one token emitted).
        assert_eq!(e.step().unwrap()[0].0, "t1");
        e.cancel(&"t1".to_string());
        e.submit(GenerationTask::new("t2", "delta echo").with_max_tokens(8))
            .unwrap();
        // The very next step serves t2 — no draining of the abandoned t1.
        assert_eq!(e.step().unwrap()[0].0, "t2");
    }

    #[test]
    fn resume_emits_only_tail_tokens() {
        let mut e = MockEngine::new();
        // Prompt echoes to [a b c d]; resume_token_ids covers [a b] (the
        // mock's own ids for the first two positions), so only c/d are new.
        let mut task = GenerationTask::new("t1", "a b c d").with_max_tokens(64);
        task.resume_token_ids = Some(vec![0, 1]);
        e.submit(task).unwrap();
        let mut emitted = Vec::new();
        for _ in 0..8 {
            for (_, chunk) in e.step().unwrap() {
                emitted.push(chunk);
            }
        }
        // Exactly 2 new tokens (c, d) then a final marker — the prefix a/b
        // is never re-emitted.
        assert_eq!(emitted.len(), 3);
        assert!(emitted[0].text.starts_with('c'));
        assert!(emitted[1].text.starts_with('d'));
        assert!(emitted[2].is_final);
    }

    #[test]
    fn resume_with_wrong_ids_is_rejected() {
        let mut e = MockEngine::new();
        let mut task = GenerationTask::new("t1", "a b c d").with_max_tokens(64);
        // Shuffled ids: right length, wrong content — a corrupted accumulator.
        task.resume_token_ids = Some(vec![1, 0]);
        assert!(
            e.submit(task).is_err(),
            "length-only consumption would have accepted a corrupt prefix"
        );
    }

    #[test]
    fn resume_at_full_budget_emits_zero_new_tokens() {
        let mut e = MockEngine::new();
        // The seed alone exhausts max_tokens, so the task finals immediately
        // with no new tokens. The prompt has MORE words than the budget so the
        // finish reason provably comes from the BUDGET branch (Length), not
        // word exhaustion (Stop) — with a 4-word prompt both conditions were
        // true and the tie-break masked which branch fired.
        let mut task = GenerationTask::new("t1", "a b c d e f").with_max_tokens(4);
        task.resume_token_ids = Some(vec![0, 1, 2, 3]);
        e.submit(task).unwrap();
        let emitted = e.step().unwrap();
        assert_eq!(emitted.len(), 1);
        assert!(emitted[0].1.is_final);
        assert_eq!(
            emitted[0].1.finish_reason,
            Some(cascadia_types::FinishReason::Length),
            "an exhausted resume budget is a Length stop, like every real engine"
        );
    }

    #[test]
    fn injected_step_error_surfaces_as_err() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "hello world")).unwrap();
        e.fail_next_step("boom");
        let res = e.step();
        assert!(matches!(res, Err(EngineError::Backend(ref m)) if m == "boom"));
        // Error is one-shot; the engine resumes on the next step.
        assert!(!e.step().unwrap().is_empty());
    }

    // --- Scripted tool-call sentinel (`__mock_tool_call__`) ---

    fn turn(role: &str, content: &str) -> cascadia_types::ChatTurn {
        cascadia_types::ChatTurn {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    fn weather_tools() -> serde_json::Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city", "days"]
                }
            }
        }])
    }

    fn tools_task(
        id: &str,
        messages: Vec<cascadia_types::ChatTurn>,
        tools: Option<serde_json::Value>,
    ) -> GenerationTask {
        let prompt = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let mut task = GenerationTask::new(id, prompt).with_max_tokens(64);
        task.messages = messages;
        task.tools = tools;
        task
    }

    /// Steps the engine until the task's final chunk; returns the joined
    /// text, the finish reason and the emitted token ids.
    fn run_to_end(e: &mut MockEngine) -> (String, Option<cascadia_types::FinishReason>, Vec<i64>) {
        let (mut text, mut ids) = (String::new(), Vec::new());
        for _ in 0..64 {
            for (_, chunk) in e.step().unwrap() {
                text.push_str(&chunk.text);
                ids.extend(chunk.token_ids.iter().copied());
                if chunk.is_final {
                    return (text, chunk.finish_reason, ids);
                }
            }
        }
        panic!("task never finished; text so far: {text}");
    }

    #[test]
    fn sentinel_emits_one_call_then_stops() {
        let mut e = MockEngine::new();
        let task = tools_task(
            "t1",
            vec![turn("user", "weather __mock_tool_call__")],
            Some(weather_tools()),
        );
        e.submit(task).unwrap();
        let (text, reason, ids) = run_to_end(&mut e);
        assert_eq!(
            text,
            r#"<tool_call>{"name":"get_weather","arguments":{"city":"mock","days":1}}</tool_call>"#
        );
        assert_eq!(reason, Some(cascadia_types::FinishReason::Stop));
        assert_eq!(ids, vec![0, 1, 2], "three chunks, ids 0..n");
        assert!(
            e.scripted.is_empty(),
            "side state removed when the task ends"
        );
    }

    #[test]
    fn tool_result_gets_fixed_answer() {
        let mut e = MockEngine::new();
        let task = tools_task(
            "t1",
            vec![
                turn("user", "weather __mock_tool_call__"),
                turn("assistant", ""),
                turn("tool", "sunny and 20C"),
            ],
            Some(weather_tools()),
        );
        e.submit(task).unwrap();
        let (text, reason, _) = run_to_end(&mut e);
        assert_eq!(text, "mock: got sunny and 20C");
        assert_eq!(reason, Some(cascadia_types::FinishReason::Stop));
    }

    #[test]
    fn tool_result_is_cut_to_40_chars() {
        let mut e = MockEngine::new();
        let long = "a".repeat(50);
        let task = tools_task("t1", vec![turn("tool", &long)], Some(weather_tools()));
        e.submit(task).unwrap();
        let (text, _, _) = run_to_end(&mut e);
        assert_eq!(text, format!("mock: got {}", "a".repeat(40)));
    }

    #[test]
    fn later_turn_gets_ok() {
        let mut e = MockEngine::new();
        let task = tools_task(
            "t1",
            vec![
                turn("user", "weather __mock_tool_call__"),
                turn("assistant", ""),
                turn("tool", "sunny and 20C"),
                turn("user", "thanks"),
            ],
            Some(weather_tools()),
        );
        e.submit(task).unwrap();
        let (text, reason, _) = run_to_end(&mut e);
        assert_eq!(text, "mock: ok");
        assert_eq!(reason, Some(cascadia_types::FinishReason::Stop));
    }

    #[test]
    fn tool_choice_none_echoes() {
        let mut e = MockEngine::new();
        let mut task = tools_task(
            "t1",
            vec![turn("user", "weather __mock_tool_call__")],
            Some(weather_tools()),
        );
        task.tool_choice = Some(serde_json::json!("none"));
        e.submit(task).unwrap();
        let (text, _, _) = run_to_end(&mut e);
        assert_eq!(text, "weather __mock_tool_call__ ");
    }

    #[test]
    fn no_tools_echoes() {
        let mut e = MockEngine::new();
        let task = tools_task("t1", vec![turn("user", "weather __mock_tool_call__")], None);
        e.submit(task).unwrap();
        let (text, _, _) = run_to_end(&mut e);
        assert_eq!(text, "weather __mock_tool_call__ ");
        // An empty tools array is "no tools" too.
        let task = tools_task(
            "t2",
            vec![turn("user", "weather __mock_tool_call__")],
            Some(serde_json::json!([])),
        );
        e.submit(task).unwrap();
        let (text, _, _) = run_to_end(&mut e);
        assert_eq!(text, "weather __mock_tool_call__ ");
    }

    #[test]
    fn scripted_resume_and_cancel() {
        // Resume: a seed of the script's own ids [0..K) skips K chunks.
        let mut e = MockEngine::new();
        let mut task = tools_task(
            "t1",
            vec![turn("user", "weather __mock_tool_call__")],
            Some(weather_tools()),
        );
        task.resume_token_ids = Some(vec![0, 1]);
        e.submit(task).unwrap();
        let (text, _, ids) = run_to_end(&mut e);
        assert_eq!(text, "</tool_call>");
        assert_eq!(ids, vec![2]);
        // Cancel drops the scripted side state.
        let task = tools_task(
            "t2",
            vec![turn("user", "weather __mock_tool_call__")],
            Some(weather_tools()),
        );
        e.submit(task).unwrap();
        e.step().unwrap();
        e.cancel(&"t2".to_string());
        assert!(e.scripted.is_empty());
        assert!(e.step().unwrap().is_empty());
    }
}
