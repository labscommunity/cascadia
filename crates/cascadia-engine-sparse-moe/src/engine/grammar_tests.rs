//! Structured output on the pipeline engine: the one-task path and
//! multi-stream decode, on a single stage and across stages, driven by a fake
//! runner whose logits favour tokens the schema forbids.

use super::*;
use crate::constrained::test_env::{self, EOS, VOCAB, WEATHER};
use crate::constrained::GrammarSlot;

/// Favoured order: a control byte (never valid JSON), EOS, `}`, `"`, `x`.
fn preferences(width: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; width];
    let mut set = |i: usize, v: f32| {
        if i < width {
            l[i] = v;
        }
    };
    set(0x01, 20.0);
    set(EOS as usize, 10.0);
    set(usize::from(b'}'), 6.0);
    set(usize::from(b'"'), 5.0);
    set(usize::from(b'x'), 4.0);
    l
}

/// A `StagedRunner` with fixed logits and per-slot positions for streams.
struct JsonStub {
    width: usize,
    slots: Vec<Option<usize>>,
    batched: bool,
}

impl JsonStub {
    fn new() -> Self {
        Self {
            width: VOCAB,
            slots: Vec::new(),
            batched: false,
        }
    }

    /// Prefills a prompt as one batch (`ForwardBatchPrefill` on a pipeline).
    fn batched() -> Self {
        Self {
            batched: true,
            ..Self::new()
        }
    }
}

impl StagedRunner for JsonStub {
    fn arch_name(&self) -> &'static str {
        "json-stub"
    }
    fn hidden_size(&self) -> usize {
        4
    }
    fn max_seq(&self) -> usize {
        4096
    }
    fn eos_token_ids(&self) -> &[u32] {
        &[EOS]
    }
    fn reset(&mut self) {}
    fn embed_token(&self, token: u32) -> Vec<f32> {
        vec![token as f32; 4]
    }
    fn forward_layers(&mut self, hidden: Vec<f32>, _pos: usize, _token: Option<u32>) -> Vec<f32> {
        hidden
    }
    fn head_logits(&self, _hidden: &[f32]) -> Vec<f32> {
        preferences(self.width)
    }
    fn supports_batched_prefill(&self) -> bool {
        self.batched
    }
    fn stream_capacity(&self) -> usize {
        self.slots.len()
    }
    fn configure_streams(&mut self, n: usize) -> bool {
        self.slots = vec![None; n];
        true
    }
    fn open_stream(&mut self) -> Option<usize> {
        let i = self.slots.iter().position(Option::is_none)?;
        self.slots[i] = Some(0);
        Some(i)
    }
    fn open_stream_at(&mut self, slot: usize) -> bool {
        match self.slots.get_mut(slot) {
            Some(s) => {
                *s = Some(0);
                true
            }
            None => false,
        }
    }
    fn close_stream(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            *s = None;
        }
    }
    fn stream_pos(&self, slot: usize) -> Option<usize> {
        self.slots.get(slot).copied().flatten()
    }
    fn prefill_stream(&mut self, slot: usize, hidden: Vec<f32>, rows: usize) -> Vec<f32> {
        if let Some(Some(p)) = self.slots.get_mut(slot) {
            *p += rows;
        }
        hidden
    }
    fn decode_streams(&mut self, hidden: Vec<f32>, slots: &[usize]) -> Vec<f32> {
        for &s in slots {
            if let Some(Some(p)) = self.slots.get_mut(s) {
                *p += 1;
            }
        }
        hidden
    }
}

struct Rig {
    // Keeps the handle the engine holds alive.
    _rt: tokio::runtime::Runtime,
    eng: PipelineEngine<JsonStub>,
}

fn rig(runner: JsonStub, total: u32, streams: usize) -> Rig {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut eng = PipelineEngine::new(
        runner,
        Some(test_env::tokenizer()),
        StageTransport {
            upstream: None,
            downstream: None,
        },
        rt.handle().clone(),
        0,
        total,
        None,
    );
    eng.set_grammar_slot(GrammarSlot::from_factory(test_env::factory()));
    if streams > 0 {
        assert_eq!(eng.enable_streams(streams), streams);
    }
    Rig { _rt: rt, eng }
}

fn json_task(id: &str, max_tokens: u32) -> GenerationTask {
    let mut t = GenerationTask::new(id, "hi").with_max_tokens(max_tokens);
    t.grammar = Some(test_env::spec(WEATHER));
    t
}

fn plain_task(id: &str, max_tokens: u32) -> GenerationTask {
    GenerationTask::new(id, "hi").with_max_tokens(max_tokens)
}

/// What one task produced: its token ids, text and final chunk.
#[derive(Debug, Default)]
struct Outcome {
    tokens: Vec<u32>,
    text: String,
    last: Option<Chunk>,
}

/// Steps until every task in `ids` has its final chunk.
fn drive(eng: &mut PipelineEngine<JsonStub>, ids: &[&str]) -> HashMap<String, Outcome> {
    let mut out: HashMap<String, Outcome> = HashMap::new();
    for _ in 0..10_000 {
        if ids
            .iter()
            .all(|id| out.get(*id).is_some_and(|o| o.last.is_some()))
        {
            return out;
        }
        for (id, c) in eng.step().unwrap() {
            let o = out.entry(id).or_default();
            o.tokens.extend(c.token_ids.iter().map(|&t| t as u32));
            o.text.push_str(&c.text);
            if c.is_final {
                o.last = Some(c);
            }
        }
    }
    panic!("tasks did not finish: {out:?}");
}

/// Every token was allowed when it was picked, and the grammar finished.
fn assert_schema_valid(o: &Outcome) {
    let mut gm = test_env::mask(WEATHER);
    for &t in &o.tokens {
        assert!(gm.allowed(t), "token {t} was not allowed: {:?}", o.tokens);
        gm.accept(t).unwrap();
    }
    assert!(gm.finished(), "grammar not finished: {:?}", o.text);
    let v: serde_json::Value = serde_json::from_str(&o.text).expect("output parses");
    assert!(v["city"].is_string(), "{v}");
    assert!(
        ["sunny", "cloudy", "rainy"].contains(&v["conditions"].as_str().unwrap()),
        "{v}"
    );
}

fn final_chunk(o: &Outcome) -> &Chunk {
    o.last.as_ref().unwrap()
}

#[test]
fn single_stage_json_task_produces_schema_valid_json_and_stops() {
    let mut r = rig(JsonStub::new(), 1, 0);
    r.eng.submit(json_task("j", 400)).unwrap();
    let out = drive(&mut r.eng, &["j"]);
    let j = &out["j"];
    assert_schema_valid(j);
    assert_eq!(final_chunk(j).error, None);
    assert_eq!(final_chunk(j).finish_reason, Some(FinishReason::Stop));
}

#[test]
fn single_stage_json_task_finishing_at_exactly_max_tokens_is_stop() {
    let mut r = rig(JsonStub::new(), 1, 0);
    r.eng.submit(json_task("a", 400)).unwrap();
    let n = drive(&mut r.eng, &["a"])["a"].tokens.len() as u32;
    r.eng.submit(json_task("b", n)).unwrap();
    let out = drive(&mut r.eng, &["b"]);
    assert_eq!(out["b"].tokens.len() as u32, n);
    assert_schema_valid(&out["b"]);
    assert_eq!(
        final_chunk(&out["b"]).finish_reason,
        Some(FinishReason::Stop)
    );
}

#[test]
fn single_stage_json_task_cut_short_is_length_with_a_valid_prefix() {
    let mut r = rig(JsonStub::new(), 1, 0);
    r.eng.submit(json_task("j", 5)).unwrap();
    let out = drive(&mut r.eng, &["j"]);
    let j = &out["j"];
    assert_eq!(j.tokens.len(), 5);
    let mut gm = test_env::mask(WEATHER);
    for &t in &j.tokens {
        gm.accept(t).unwrap();
    }
    assert_eq!(final_chunk(j).finish_reason, Some(FinishReason::Length));
}

#[test]
fn single_stage_json_task_with_sampling_stays_inside_the_schema() {
    let mut r = rig(JsonStub::new(), 1, 0);
    for seed in 1..=8u64 {
        let id = format!("s{seed}");
        let mut t = json_task(&id, 4000);
        t.temperature = 1.0;
        t.sampling.seed = Some(seed);
        r.eng.submit(t).unwrap();
        let out = drive(&mut r.eng, &[id.as_str()]);
        assert_schema_valid(&out[&id]);
    }
}

#[test]
fn single_stage_plain_task_is_unchanged() {
    let mut r = rig(JsonStub::new(), 1, 0);
    r.eng.submit(plain_task("p", 24)).unwrap();
    let out = drive(&mut r.eng, &["p"]);
    let mut direct = JsonStub::new();
    let cfg = sampling_from_task(&plain_task("p", 24));
    let (want, _, _) = direct.generate_reason_timed(&[b'h'.into(), b'i'.into()], 24, &cfg);
    assert_eq!(out["p"].tokens, want);
    // The unconstrained favourite is the control byte the grammar forbids.
    assert_eq!(want[0], 0x01);
}

#[test]
fn grammar_dead_end_is_an_error_chunk_not_output() {
    // A head row too short to hold `{`: nothing the grammar allows exists.
    let mut runner = JsonStub::new();
    runner.width = usize::from(b'{');
    let mut r = rig(runner, 1, 0);
    r.eng.submit(json_task("j", 16)).unwrap();
    let out = drive(&mut r.eng, &["j"]);
    assert!(out["j"].tokens.is_empty(), "{:?}", out["j"]);
    assert!(final_chunk(&out["j"]).error.is_some());
}

#[test]
fn streams_mix_json_and_plain_tasks() {
    let mut alone = rig(JsonStub::new(), 1, 2);
    alone.eng.submit(plain_task("p", 30)).unwrap();
    let want = drive(&mut alone.eng, &["p"]).remove("p").unwrap();

    let mut r = rig(JsonStub::new(), 1, 2);
    r.eng.submit(json_task("j", 400)).unwrap();
    r.eng.submit(plain_task("p", 30)).unwrap();
    let out = drive(&mut r.eng, &["j", "p"]);
    assert_schema_valid(&out["j"]);
    assert_eq!(
        final_chunk(&out["j"]).finish_reason,
        Some(FinishReason::Stop)
    );
    assert_eq!(out["p"].tokens, want.tokens, "plain stream changed");
    assert_eq!(out["p"].text, want.text);
    assert_eq!(
        final_chunk(&out["p"]).finish_reason,
        final_chunk(&want).finish_reason
    );
}

#[test]
fn stream_json_task_finishing_at_exactly_max_tokens_is_stop() {
    let mut r = rig(JsonStub::new(), 1, 2);
    r.eng.submit(json_task("a", 400)).unwrap();
    let n = drive(&mut r.eng, &["a"])["a"].tokens.len() as u32;
    r.eng.submit(json_task("b", n)).unwrap();
    r.eng.submit(plain_task("p", 8)).unwrap();
    let out = drive(&mut r.eng, &["b", "p"]);
    assert_eq!(out["b"].tokens.len() as u32, n);
    assert_schema_valid(&out["b"]);
    assert_eq!(
        final_chunk(&out["b"]).finish_reason,
        Some(FinishReason::Stop)
    );
    assert_eq!(
        final_chunk(&out["p"]).finish_reason,
        Some(FinishReason::Length)
    );
}

#[test]
fn stream_dead_end_fails_only_that_stream() {
    let mut runner = JsonStub::new();
    runner.width = usize::from(b'{');
    let mut r = rig(runner, 1, 2);
    r.eng.submit(json_task("j", 16)).unwrap();
    r.eng.submit(plain_task("p", 6)).unwrap();
    let out = drive(&mut r.eng, &["j", "p"]);
    assert!(out["j"].tokens.is_empty());
    assert!(final_chunk(&out["j"]).error.is_some());
    assert_eq!(final_chunk(&out["p"]).error, None);
    assert_eq!(out["p"].tokens.len(), 6);
}

#[test]
fn submit_rejects_an_unbuildable_schema_with_invalid_config() {
    let mut r = rig(JsonStub::new(), 1, 0);
    let mut t = json_task("j", 8);
    t.grammar = Some(test_env::spec(r#"{"type":42}"#));
    assert!(matches!(
        r.eng.submit(t),
        Err(EngineError::InvalidConfig(_))
    ));
    assert!(r.eng.pending.is_empty());
}

#[test]
fn submit_without_a_factory_is_invalid_config() {
    let mut r = rig(JsonStub::new(), 1, 0);
    r.eng.set_grammar_slot(GrammarSlot::default());
    assert!(matches!(
        r.eng.submit(json_task("j", 8)),
        Err(EngineError::InvalidConfig(m)) if m.starts_with("structured output is not available for this model")
    ));
    r.eng.submit(plain_task("p", 8)).unwrap();
}

#[test]
fn submit_queues_json_on_a_multi_stage_pipeline() {
    let mut r = rig(JsonStub::new(), 2, 0);
    r.eng.submit(json_task("j", 8)).unwrap();
    assert_eq!(r.eng.pending.len(), 1);
    assert_eq!(r.eng.grammars.len(), 1);
}

#[test]
fn cancel_drops_the_compiled_grammar() {
    let mut r = rig(JsonStub::new(), 1, 0);
    r.eng.submit(json_task("j", 8)).unwrap();
    r.eng.cancel(&"j".to_string());
    assert!(r.eng.grammars.is_empty());
}

#[test]
fn speculative_decoding_is_skipped_for_grammar_tasks() {
    assert!(use_spec_decode(Some(4), 0.0, false));
    assert!(!use_spec_decode(Some(4), 0.0, true));
    assert!(!use_spec_decode(Some(4), 0.7, false));
    assert!(!use_spec_decode(None, 0.0, false));
}

#[test]
fn a_finished_grammar_at_the_cap_is_stop() {
    assert_eq!(finish_reason_for(8, 8, true), FinishReason::Stop);
    assert_eq!(finish_reason_for(8, 8, false), FinishReason::Length);
    assert_eq!(finish_reason_for(3, 8, false), FinishReason::Stop);
}

fn resumed_json_task(id: &str) -> GenerationTask {
    let mut t = json_task(id, 64);
    t.resume_token_ids = Some(br#"{"city": "P"#.iter().map(|&b| i32::from(b)).collect());
    t
}

fn assert_resume_declined(o: &Outcome) {
    assert!(o.tokens.is_empty(), "{o:?}");
    let reason = final_chunk(o).error.as_deref().expect("error chunk");
    assert!(reason.starts_with("resume_unsupported:"), "{reason}");
}

#[test]
fn single_stage_declines_a_resumed_json_task() {
    let mut r = rig(JsonStub::new(), 1, 0);
    r.eng.submit(resumed_json_task("j")).unwrap();
    assert_resume_declined(&drive(&mut r.eng, &["j"])["j"]);
    assert!(r.eng.grammars.is_empty());
}

#[test]
fn streams_decline_a_resumed_json_task() {
    let mut r = rig(JsonStub::new(), 1, 2);
    r.eng.submit(resumed_json_task("j")).unwrap();
    assert_resume_declined(&drive(&mut r.eng, &["j"])["j"]);
    assert!(r.eng.grammars.is_empty());
}

#[test]
fn a_resumed_plain_task_keeps_todays_behaviour() {
    for streams in [0, 2] {
        let mut r = rig(JsonStub::new(), 1, streams);
        let mut t = plain_task("p", 6);
        t.resume_token_ids = Some(vec![65]);
        r.eng.submit(t).unwrap();
        let out = drive(&mut r.eng, &["p"]);
        assert_eq!(final_chunk(&out["p"]).error, None, "streams={streams}");
        assert_eq!(out["p"].tokens.len(), 6);
    }
}

#[test]
fn early_returns_do_not_leak_the_compiled_grammar() {
    for streams in [0, 2] {
        let mut r = rig(JsonStub::new(), 1, streams);
        let mut t = json_task("e", 8);
        t.prompt = String::new();
        r.eng.submit(t).unwrap();
        drive(&mut r.eng, &["e"]);
        assert!(r.eng.grammars.is_empty(), "streams={streams}");
    }
}

// ---- multi-stage: the mask on the stage wire --------------------------------

use crate::constrained::UPGRADE_EVERY_STAGE;
use cascadia_transport::{ActivationClient, ActivationServer};

/// Rank 0 of a pipeline whose other ranks run on their own threads over
/// loopback links. The head is dropped first, which closes the links and
/// lets the workers exit.
struct Chain {
    head: PipelineEngine<JsonStub>,
    rt: tokio::runtime::Runtime,
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

type Link = (
    Arc<TokioMutex<ActivationServer>>,
    Arc<TokioMutex<ActivationClient>>,
);

/// `n` loopback links: `(server side, client side)`.
fn links(rt: &tokio::runtime::Runtime, n: usize) -> Vec<Link> {
    rt.block_on(async {
        let mut out = Vec::new();
        for _ in 0..n {
            let mut srv = ActivationServer::new("127.0.0.1", 0);
            srv.start().await.unwrap();
            let port = srv.port();
            let srv = Arc::new(TokioMutex::new(srv));
            let s = srv.clone();
            let accept = tokio::spawn(async move { s.lock().await.accept().await.unwrap() });
            let mut cli = ActivationClient::new("127.0.0.1", port);
            cli.connect_with_timeout(Duration::from_secs(5))
                .await
                .unwrap();
            accept.await.unwrap();
            out.push((srv, Arc::new(TokioMutex::new(cli))));
        }
        out
    })
}

fn head_engine(
    rt: &tokio::runtime::Runtime,
    runner: JsonStub,
    down: Arc<TokioMutex<ActivationClient>>,
    ranks: u32,
    streams: usize,
) -> PipelineEngine<JsonStub> {
    let mut head = PipelineEngine::new(
        runner,
        Some(test_env::tokenizer()),
        StageTransport {
            upstream: None,
            downstream: Some(down),
        },
        rt.handle().clone(),
        0,
        ranks,
        None,
    );
    head.set_grammar_slot(GrammarSlot::from_factory(test_env::factory()));
    if streams > 0 {
        assert_eq!(head.enable_streams(streams), streams);
    }
    head
}

fn chain(ranks: u32, streams: usize, make: fn() -> JsonStub) -> Chain {
    let rt = runtime();
    let mut l = links(&rt, ranks as usize - 1);
    let head = head_engine(&rt, make(), l[0].1.clone(), ranks, streams);
    for r in 1..ranks as usize {
        let up = l[r - 1].0.clone();
        let down = l.get(r).map(|x| x.1.clone());
        let mut w = PipelineEngine::new(
            make(),
            None,
            StageTransport {
                upstream: Some(up),
                downstream: down,
            },
            rt.handle().clone(),
            r as u32,
            ranks,
            None,
        );
        if streams > 0 {
            assert_eq!(w.enable_streams(streams), streams);
        }
        std::thread::spawn(move || while w.step().is_ok() {});
    }
    l.clear();
    Chain { head, rt }
}

#[test]
fn json_task_is_enforced_through_a_three_stage_chain() {
    for make in [JsonStub::new as fn() -> JsonStub, JsonStub::batched] {
        let mut c = chain(3, 0, make);
        c.head.submit(json_task("j", 400)).unwrap();
        let out = drive(&mut c.head, &["j"]);
        assert_schema_valid(&out["j"]);
        assert_eq!(
            final_chunk(&out["j"]).finish_reason,
            Some(FinishReason::Stop)
        );
        assert!(!c.head.stage_masks.latch.is_set());
        // A plain task on the same chain afterwards is unchanged.
        let mut alone = rig(make(), 1, 0);
        alone.eng.submit(plain_task("p", 12)).unwrap();
        let want = drive(&mut alone.eng, &["p"]).remove("p").unwrap();
        c.head.submit(plain_task("p", 12)).unwrap();
        let got = drive(&mut c.head, &["p"]).remove("p").unwrap();
        assert_eq!(got.tokens, want.tokens);
        assert_eq!(final_chunk(&got).error, None);
        let _ = &c.rt;
    }
}

#[test]
fn json_and_plain_streams_share_batches_across_stages() {
    let mut alone = rig(JsonStub::new(), 1, 3);
    alone.eng.submit(plain_task("p", 30)).unwrap();
    let want = drive(&mut alone.eng, &["p"]).remove("p").unwrap();

    let mut c = chain(3, 3, JsonStub::new);
    c.head.submit(json_task("j", 400)).unwrap();
    c.head.submit(plain_task("p", 30)).unwrap();
    // A prompt longer than one prefill window: its final window is masked.
    let mut long = json_task("l", 400);
    long.prompt = "x".repeat(300);
    c.head.submit(long).unwrap();
    let out = drive(&mut c.head, &["j", "p", "l"]);
    for id in ["j", "l"] {
        assert_schema_valid(&out[id]);
        assert_eq!(
            final_chunk(&out[id]).finish_reason,
            Some(FinishReason::Stop),
            "{id}"
        );
    }
    assert_eq!(out["p"].tokens, want.tokens, "plain stream changed");
    assert_eq!(final_chunk(&out["p"]).error, None);
    assert!(!c.head.stage_masks.latch.is_set());
}

/// A last stage that reads frames with this build's decoders but answers a
/// masked step with a plain `Token`, as a stage that does not acknowledge
/// masks would. Counts the masked frames it saw.
fn plain_answering_tail(
    rt: &tokio::runtime::Runtime,
    srv: Arc<TokioMutex<ActivationServer>>,
    masked: Arc<std::sync::atomic::AtomicUsize>,
) {
    let handle = rt.handle().clone();
    std::thread::spawn(move || {
        handle.block_on(async move {
            loop {
                match crate::dist::recv_kind_server(&srv).await {
                    Ok(Some(FrameKind::Reset)) => {}
                    Ok(Some(FrameKind::Forward | FrameKind::ForwardNoSample)) => {
                        let (_, _, _, step) = crate::dist::recv_forward_step_body_server(&srv)
                            .await
                            .unwrap();
                        if step.is_masked() {
                            masked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                        send_token_upstream(&srv, i64::from(b'a')).await.unwrap();
                    }
                    _ => return,
                }
            }
        })
    });
}

#[test]
fn an_unacknowledged_masked_step_fails_the_task_and_latches() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let masked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    plain_answering_tail(&rt, l[0].0.clone(), masked.clone());
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 0);

    head.submit(json_task("j1", 32)).unwrap();
    let out = drive(&mut head, &["j1"]);
    assert!(out["j1"].tokens.is_empty(), "no unmasked token is emitted");
    let first = final_chunk(&out["j1"]).error.clone().expect("error chunk");
    assert!(first.contains("likely cause"), "{first}");
    assert!(first.ends_with(UPGRADE_EVERY_STAGE), "{first}");
    assert!(head.stage_masks.latch.is_set());
    assert_eq!(masked.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Later json tasks fail at once, without a masked frame.
    head.submit(json_task("j2", 32)).unwrap();
    let out = drive(&mut head, &["j2"]);
    assert_eq!(
        final_chunk(&out["j2"]).error.as_deref(),
        Some(UPGRADE_EVERY_STAGE)
    );
    assert_eq!(masked.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Plain tasks keep working.
    head.submit(plain_task("p", 4)).unwrap();
    let out = drive(&mut head, &["p"]);
    assert_eq!(final_chunk(&out["p"]).error, None);
    assert_eq!(out["p"].tokens, vec![u32::from(b'a'); 4]);

    // After the re-probe interval the next json task probes again.
    head.stage_masks
        .latch
        .set_at(Instant::now() - crate::dist::MASK_REPROBE - Duration::from_secs(1));
    head.submit(json_task("j3", 32)).unwrap();
    let out = drive(&mut head, &["j3"]);
    assert!(final_chunk(&out["j3"])
        .error
        .as_deref()
        .is_some_and(|e| e.contains("likely cause")));
    assert_eq!(masked.load(std::sync::atomic::Ordering::SeqCst), 2);
}

/// A stream tail that answers a masked stream frame with a plain
/// `StreamTokens`.
fn plain_answering_stream_tail(
    rt: &tokio::runtime::Runtime,
    srv: Arc<TokioMutex<ActivationServer>>,
) {
    let handle = rt.handle().clone();
    std::thread::spawn(move || {
        handle.block_on(async move {
            loop {
                match crate::dist::recv_kind_server(&srv).await {
                    Ok(Some(FrameKind::StreamOpen)) => {
                        let (bid, slot, ..) = crate::dist::recv_stream_open_step_body_server(&srv)
                            .await
                            .unwrap();
                        send_stream_tokens_upstream(&srv, bid, &[(slot, i64::from(b'{'))])
                            .await
                            .unwrap();
                    }
                    Ok(Some(FrameKind::StreamClose)) => {
                        crate::dist::recv_stream_close_body_server(&srv)
                            .await
                            .unwrap();
                    }
                    _ => return,
                }
            }
        })
    });
}

#[test]
fn an_unacknowledged_masked_stream_frame_fails_and_latches() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    plain_answering_stream_tail(&rt, l[0].0.clone());
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 2);
    head.submit(json_task("j", 32)).unwrap();
    let out = drive(&mut head, &["j"]);
    assert!(out["j"].tokens.is_empty());
    let e = final_chunk(&out["j"]).error.clone().expect("error chunk");
    assert!(e.ends_with(UPGRADE_EVERY_STAGE), "{e}");
    assert!(head.stage_masks.latch.is_set());
}

/// With the real reply budgets: a middle stage's masked wait ends before
/// the wait of every stage above it, so its unconfirmed reply reaches rank 0
/// before rank 0 gives up.
#[test]
fn middle_masked_budgets_end_before_the_heads_with_the_real_budgets() {
    for head in [
        decode_reply_deadline(),
        prefill_reply_deadline(),
        PipelineEngine::<JsonStub>::reply_deadline(),
        PipelineEngine::<JsonStub>::reply_deadline_prefill(),
    ] {
        assert!(head > Duration::ZERO);
        for total in 2..=12u32 {
            let mut upstream = head;
            for rank in 1..total {
                let b = crate::dist::mid_mask_budget(head, rank, total);
                assert!(b < upstream, "{head:?} rank {rank}/{total}");
                upstream = b;
            }
        }
    }
    // A middle stage takes the prefill budget only for the first sampled
    // step after a reset (the prompt's last token), which is the step rank 0
    // waits on with the prefill budget; decode steps get the decode budget.
    assert!(decode_reply_deadline() <= prefill_reply_deadline());
}

// ---- mask evidence per link: never a frame an older worker dies on ----------

/// What a fixture last stage received: each frame's kind and whether it
/// carried a mask, and whether it exited.
#[derive(Default)]
struct TailLog {
    frames: std::sync::Mutex<Vec<(FrameKind, bool)>>,
    died: std::sync::atomic::AtomicBool,
    /// Masked frames that carried the repeat marker.
    repeats: std::sync::atomic::AtomicUsize,
}

impl TailLog {
    fn saw(&self, kind: FrameKind, masked: bool) {
        self.frames.lock().unwrap().push((kind, masked));
    }
    fn frames(&self) -> Vec<(FrameKind, bool)> {
        self.frames.lock().unwrap().clone()
    }
    fn died(&self) -> bool {
        self.died.load(std::sync::atomic::Ordering::SeqCst)
    }
    fn repeats(&self) -> usize {
        self.repeats.load(std::sync::atomic::Ordering::SeqCst)
    }
    fn masked(&self, kind: FrameKind) -> usize {
        self.frames()
            .iter()
            .filter(|&&(k, m)| k == kind && m)
            .count()
    }
}

/// The lowest token a mask row allows (the fixture's "sampler").
fn first_allowed(words: &[u32]) -> i64 {
    for (w, &bits) in words.iter().enumerate() {
        if bits != 0 {
            return (w * 32 + bits.trailing_zeros() as usize) as i64;
        }
    }
    -1
}

/// A last stage on the wire. `old`: the base build's Pipeline worker, which
/// reads frames with the original decoders, skips a `Forward` it can't
/// decode, and exits (closing its link) on a batched or stream frame it
/// can't decode. Otherwise a stage on this build that answers masked steps
/// with the lowest allowed token.
fn fixture_tail(
    rt: &tokio::runtime::Runtime,
    srv: Arc<TokioMutex<ActivationServer>>,
    old: bool,
) -> Arc<TailLog> {
    fixture_tail_with(rt, srv, old, false)
}

/// [`fixture_tail`]; with `forget` a new stage that answers every repeat
/// marker with the mask-miss token (a stage that lost its mask state).
fn fixture_tail_with(
    rt: &tokio::runtime::Runtime,
    srv: Arc<TokioMutex<ActivationServer>>,
    old: bool,
    forget: bool,
) -> Arc<TailLog> {
    use crate::dist::*;
    let log = Arc::new(TailLog::default());
    let l = log.clone();
    let handle = rt.handle().clone();
    let recv = MaskRecvLink::default();
    // A new stage resolves markers with its receive cache, unless it forgot.
    let resolve = move |l: &TailLog, c: MaskCols| {
        if c.is_repeat() {
            l.repeats.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if forget {
                return c;
            }
        }
        recv.resolve(c)
    };
    // The lowest allowed token of row `r`, the mask-miss token for a
    // marker that did not resolve.
    let pick = |c: &MaskCols, r: usize| {
        if c.is_repeat() {
            i64::from(cascadia_grammar::MASK_MISS_TOKEN)
        } else {
            c.words(r).map_or(-1, first_allowed)
        }
    };
    std::thread::spawn(move || {
        handle.block_on(async move {
            let die = |l: &TailLog| l.died.store(true, std::sync::atomic::Ordering::SeqCst);
            loop {
                let Ok(Some(kind)) = recv_kind_server(&srv).await else {
                    return;
                };
                let sample = matches!(
                    kind,
                    FrameKind::Forward
                        | FrameKind::ForwardBatchPrefill
                        | FrameKind::StreamOpen
                        | FrameKind::StreamDecode
                );
                match kind {
                    FrameKind::Reset => l.saw(kind, false),
                    FrameKind::Forward | FrameKind::ForwardNoSample if old => {
                        match recv_forward_body_server(&srv).await {
                            Ok(_) => {
                                l.saw(kind, false);
                                let t = if sample { i64::from(b'a') } else { -1 };
                                send_token_upstream(&srv, t).await.unwrap();
                            }
                            // Logged and skipped: the worker keeps running.
                            Err(_) => l.saw(kind, true),
                        }
                    }
                    FrameKind::ForwardBatchPrefill | FrameKind::ForwardBatchPrefillNoSample
                        if old =>
                    {
                        match recv_forward_batch_body_server(&srv).await {
                            Ok(_) => {
                                l.saw(kind, false);
                                let t = if sample { i64::from(b'a') } else { -1 };
                                send_token_upstream(&srv, t).await.unwrap();
                            }
                            Err(_) => {
                                l.saw(kind, true);
                                die(&l);
                                srv.lock().await.close().await;
                                return;
                            }
                        }
                    }
                    FrameKind::StreamOpen | FrameKind::StreamFeed | FrameKind::StreamDecode
                        if old =>
                    {
                        let ok = match kind {
                            FrameKind::StreamOpen => recv_stream_open_body_server(&srv)
                                .await
                                .map(|(b, s, ..)| (b, vec![s])),
                            FrameKind::StreamFeed => recv_stream_feed_body_server(&srv)
                                .await
                                .map(|(b, s, ..)| (b, vec![s])),
                            _ => recv_stream_decode_body_server(&srv)
                                .await
                                .map(|(b, rows, _)| (b, rows.iter().map(|r| r.0).collect())),
                        };
                        match ok {
                            Ok((bid, slots)) => {
                                l.saw(kind, false);
                                let toks: Vec<(u32, i64)> =
                                    slots.iter().map(|&s| (s, i64::from(b'a'))).collect();
                                send_stream_tokens_upstream(&srv, bid, &toks).await.unwrap();
                            }
                            Err(_) => {
                                l.saw(kind, true);
                                die(&l);
                                srv.lock().await.close().await;
                                return;
                            }
                        }
                    }
                    FrameKind::Forward
                    | FrameKind::ForwardNoSample
                    | FrameKind::ForwardBatchPrefill
                    | FrameKind::ForwardBatchPrefillNoSample => {
                        let (step, rows) =
                            if matches!(kind, FrameKind::Forward | FrameKind::ForwardNoSample) {
                                let (_, _, _, step) =
                                    recv_forward_step_body_server(&srv).await.unwrap();
                                (step, 1)
                            } else {
                                let (_, n, _, step) =
                                    recv_forward_batch_step_body_server(&srv).await.unwrap();
                                (step, n as usize)
                            };
                        let (_, _, cols) = step.split(4).unwrap();
                        l.saw(kind, cols.is_some());
                        let cols = cols.map(|c| match kind {
                            FrameKind::Forward => resolve(&l, c),
                            _ => c,
                        });
                        match cols {
                            Some(c) => {
                                let t = pick(&c, rows - 1);
                                send_token_ack_upstream(&srv, t, c.tag(rows - 1), MASK_ACK)
                                    .await
                                    .unwrap();
                            }
                            None => {
                                let t = if sample { i64::from(b'a') } else { -1 };
                                send_token_upstream(&srv, t).await.unwrap();
                            }
                        }
                    }
                    FrameKind::StreamOpen | FrameKind::StreamFeed | FrameKind::StreamDecode => {
                        let (bid, slots, step, last) = match kind {
                            FrameKind::StreamOpen => {
                                let (b, s, _, _, st) =
                                    recv_stream_open_step_body_server(&srv).await.unwrap();
                                (b, vec![s], st, true)
                            }
                            FrameKind::StreamFeed => {
                                let (b, s, _, f, _, st) =
                                    recv_stream_feed_step_body_server(&srv).await.unwrap();
                                (b, vec![s], st, f & STREAM_FEED_FINAL != 0)
                            }
                            _ => {
                                let (b, rows, st) =
                                    recv_stream_decode_step_body_server(&srv).await.unwrap();
                                (b, rows.iter().map(|r| r.0).collect(), st, true)
                            }
                        };
                        let (_, shape, cols) = step.split(4).unwrap();
                        l.saw(kind, cols.is_some());
                        let cols = cols.map(|c| resolve(&l, c));
                        if !last {
                            continue;
                        }
                        let rows = shape[1] as usize;
                        let toks: Vec<(u32, i64)> = slots
                            .iter()
                            .enumerate()
                            .map(|(i, &s)| {
                                // A prompt frame samples its last row.
                                let r = if kind == FrameKind::StreamDecode {
                                    i
                                } else {
                                    rows - 1
                                };
                                let t = match cols.as_ref().filter(|c| c.tag(r) != 0) {
                                    Some(c) => pick(c, r),
                                    None => i64::from(b'a'),
                                };
                                (s, t)
                            })
                            .collect();
                        match cols {
                            Some(_) => send_stream_tokens_ack_upstream(&srv, bid, MASK_ACK, &toks)
                                .await
                                .unwrap(),
                            None => send_stream_tokens_upstream(&srv, bid, &toks).await.unwrap(),
                        }
                    }
                    FrameKind::StreamClose => {
                        recv_stream_close_body_server(&srv).await.unwrap();
                        l.saw(kind, false);
                    }
                    _ => return,
                }
            }
        })
    });
    log
}

/// Steps `head` on its own thread until `done` holds for `log` (or 10 s).
/// The head may stay blocked waiting on a reply an older stage never sends;
/// that thread is left behind.
fn step_until(mut head: PipelineEngine<JsonStub>, log: &TailLog, done: impl Fn(&TailLog) -> bool) {
    std::thread::spawn(move || loop {
        if head.step().is_err() {
            return;
        }
    });
    let t0 = Instant::now();
    while !done(log) {
        assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", log.frames());
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn long_json_task(id: &str, max_tokens: u32) -> GenerationTask {
    let mut t = json_task(id, max_tokens);
    t.prompt = "a prompt of a few tokens".into();
    t
}

#[test]
fn an_old_worker_never_gets_a_masked_batch_frame_on_a_fresh_link() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let log = fixture_tail(&rt, l[0].0.clone(), true);
    let mut head = head_engine(&rt, JsonStub::batched(), l.remove(0).1, 2, 0);
    head.submit(long_json_task("j", 8)).unwrap();
    step_until(head, &log, |l| l.masked(FrameKind::Forward) > 0 || l.died());
    std::thread::sleep(Duration::from_millis(100));
    assert!(!log.died(), "the old worker exited: {:?}", log.frames());
    let frames = log.frames();
    assert!(
        frames.contains(&(FrameKind::ForwardBatchPrefillNoSample, false)),
        "{frames:?}"
    );
    assert!(
        !frames.iter().any(|&(k, m)| m && k != FrameKind::Forward),
        "{frames:?}"
    );
}

#[test]
fn an_old_worker_never_gets_a_masked_stream_frame_on_a_fresh_link() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let log = fixture_tail(&rt, l[0].0.clone(), true);
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 2);
    head.submit(json_task("j", 8)).unwrap();
    step_until(head, &log, |l| l.masked(FrameKind::Forward) > 0 || l.died());
    std::thread::sleep(Duration::from_millis(100));
    assert!(!log.died(), "the old worker exited: {:?}", log.frames());
    let frames = log.frames();
    assert_eq!(frames, vec![(FrameKind::Forward, true)], "only the probe");
}

#[test]
fn a_new_link_gets_evidence_from_the_first_json_request_then_masks_batches() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let log = fixture_tail(&rt, l[0].0.clone(), false);
    let mut head = head_engine(&rt, JsonStub::batched(), l.remove(0).1, 2, 0);
    assert!(!head.mask_evidence);
    head.submit(long_json_task("j1", 6)).unwrap();
    let out = drive(&mut head, &["j1"]);
    assert_eq!(final_chunk(&out["j1"]).error, None);
    assert!(head.mask_evidence);
    assert_eq!(log.masked(FrameKind::ForwardBatchPrefill), 0);
    head.submit(long_json_task("j2", 6)).unwrap();
    let out = drive(&mut head, &["j2"]);
    assert_eq!(final_chunk(&out["j2"]).error, None);
    assert_eq!(
        log.masked(FrameKind::ForwardBatchPrefill),
        1,
        "{:?}",
        log.frames()
    );
    let _ = &rt;
}

#[test]
fn a_new_stream_link_is_probed_once_and_the_evidence_resets_on_redial() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let log = fixture_tail(&rt, l[0].0.clone(), false);
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 2);
    head.submit(json_task("j1", 6)).unwrap();
    let out = drive(&mut head, &["j1"]);
    assert_eq!(final_chunk(&out["j1"]).error, None);
    assert!(head.mask_evidence);
    let frames = log.frames();
    assert_eq!(
        frames[0],
        (FrameKind::Forward, true),
        "probe first: {frames:?}"
    );
    assert_eq!(log.masked(FrameKind::StreamOpen), 1, "{frames:?}");
    head.submit(json_task("j2", 6)).unwrap();
    drive(&mut head, &["j2"]);
    assert_eq!(log.masked(FrameKind::Forward), 1, "no second probe");
    // The link keeper re-dialed: the new connection must be probed again.
    head.link_epoch
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    head.step().unwrap();
    assert!(!head.mask_evidence);
    head.submit(json_task("j3", 6)).unwrap();
    let out = drive(&mut head, &["j3"]);
    assert_eq!(final_chunk(&out["j3"]).error, None);
    assert_eq!(log.masked(FrameKind::Forward), 2, "{:?}", log.frames());
    let _ = &rt;
}

#[test]
fn draining_a_masked_frame_without_ack_sets_the_latch() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let (srv, down) = l.remove(0);
    let mut head = head_engine(&rt, JsonStub::new(), down, 2, 2);
    // A frame is in flight: keep the idle link keeper off the socket.
    head.link_busy.store(1, std::sync::atomic::Ordering::SeqCst);
    head.stream_inflight[0].push_back(StreamInFlight {
        batch_id: 7,
        slots: vec![0],
        open: false,
        sent_at: Instant::now(),
        masked: true,
    });
    rt.block_on(send_stream_tokens_upstream(&srv, 7, &[(0, 1)]))
        .unwrap();
    assert!(head.drain_stream_link());
    assert!(head.stage_masks.latch.is_set());
}

/// A masked stream frame whose reply fails (here the next stage goes away
/// before it answers; a timeout takes the same branch): on a link whose
/// stages acknowledged a mask it is a slow or dead stage, so json stays on;
/// without that evidence it latches.
#[test]
fn a_failed_masked_stream_reply_latches_only_without_evidence() {
    for evidence in [false, true] {
        let rt = runtime();
        let mut l = links(&rt, 1);
        let (srv, down) = l.remove(0);
        let mut head = head_engine(&rt, JsonStub::new(), down, 2, 2);
        head.link_busy.store(1, std::sync::atomic::Ordering::SeqCst);
        head.mask_evidence = evidence;
        head.stream_inflight[0].push_back(StreamInFlight {
            batch_id: 7,
            slots: vec![0],
            open: false,
            sent_at: Instant::now(),
            masked: true,
        });
        rt.block_on(async { srv.lock().await.close().await });
        let mut out = Vec::new();
        assert!(!head.step_stream_group(0, &mut out));
        assert_eq!(
            head.stage_masks.latch.is_set(),
            !evidence,
            "evidence {evidence}"
        );
        // The drain after a local failure takes the same rule.
        head.stage_masks.latch.clear();
        head.stream_inflight[0].push_back(StreamInFlight {
            batch_id: 8,
            slots: vec![0],
            open: false,
            sent_at: Instant::now(),
            masked: true,
        });
        assert!(!head.drain_stream_link());
        assert_eq!(
            head.stage_masks.latch.is_set(),
            !evidence,
            "drain, evidence {evidence}"
        );
    }
}

/// The head and tail pieces SparseMoE and OvMoe run on a masked step
/// (their runners need model files, so the exchange is driven directly):
/// rank 0's masked `Forward` and acknowledged wait, the last stage's
/// masked sample and answer.
#[test]
fn the_masked_single_row_exchange_round_trips() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let (srv, down) = l.remove(0);
    let tail = rt.spawn(async move {
        crate::dist::recv_kind_server(&srv).await.unwrap();
        let (_, cfg, push, step) = crate::dist::recv_forward_step_body_server(&srv)
            .await
            .unwrap();
        let (_, _, cols) = step.split(4).unwrap();
        let cols = cols.expect("masked");
        let mut history = Vec::new();
        let mut rng = 1;
        let (t, tag) = masked_tail_reply(
            &preferences(VOCAB),
            &mut history,
            &cfg,
            &mut rng,
            &cols,
            0,
            push,
        );
        crate::dist::send_token_ack_upstream(&srv, t, tag, MASK_ACK)
            .await
            .unwrap();
        history
    });
    let mut masks = StageMasks::default();
    let mut gm = test_env::mask(WEATHER);
    let slot = GrammarSlot::from_factory(test_env::factory());
    let (cols, tag) = stage_mask_cols(&mut masks, &slot, &mut gm, 1).unwrap();
    let t = rt
        .block_on(masked_forward_exchange(
            &down,
            &masks,
            3,
            &crate::sampling::SamplingConfig::default(),
            &[0.5; 4],
            [1, 1, 4],
            &cols,
            tag,
            Duration::from_secs(5),
            &crate::dist::MaskSendLink::default(),
        ))
        .unwrap();
    // The stub favours a control byte, EOS and `}`; the mask allows `{` only.
    assert_eq!(t, i64::from(b'{'));
    assert_eq!(rt.block_on(tail).unwrap(), vec![i64::from(b'{')]);
    assert!(!masks.latch.is_set());
}

/// The SparseMoE / OvMoe head against a last stage on an older build (it
/// skips a masked `Forward` it can't decode and never answers), driven step
/// by step as those heads run it (their runners need model files): the
/// json task's masked step times out with the upgrade error and latches,
/// the socket stays, later json tasks are refused at once, and a later
/// plain task on the same link (`Reset`, plain `Forward`) gets its token.
#[test]
fn a_masked_step_to_a_silent_old_tail_fails_then_a_plain_task_uses_the_link() {
    use crate::dist::{recv_token_reply_with, send_forward, send_reset};
    let rt = runtime();
    let mut l = links(&rt, 1);
    let (srv, down) = l.remove(0);
    let log = fixture_tail(&rt, srv, true);
    let cfg = crate::sampling::SamplingConfig::default();
    let mut masks = StageMasks::default();
    let slot = GrammarSlot::from_factory(test_env::factory());

    // The json task: Reset, then its first masked step.
    assert_eq!(masks.refusal(), None);
    rt.block_on(send_reset(&down)).unwrap();
    let mut gm = test_env::mask(WEATHER);
    let (cols, tag) = stage_mask_cols(&mut masks, &slot, &mut gm, 1).unwrap();
    let err = rt
        .block_on(masked_forward_exchange(
            &down,
            &masks,
            0,
            &cfg,
            &[0.5; 4],
            [1, 1, 4],
            &cols,
            tag,
            Duration::from_millis(300),
            &crate::dist::MaskSendLink::default(),
        ))
        .unwrap_err();
    assert!(err.contains("no reply within"), "{err}");
    assert!(err.ends_with(UPGRADE_EVERY_STAGE), "{err}");
    assert!(masks.latch.is_set());
    assert_eq!(masks.refusal().as_deref(), Some(UPGRADE_EVERY_STAGE));

    // A later plain task on the same link.
    rt.block_on(send_reset(&down)).unwrap();
    rt.block_on(send_forward(&down, 0, &cfg, &[0.5; 4], [1, 1, 4], true))
        .unwrap();
    let t = rt
        .block_on(recv_token_reply_with(
            &down,
            Duration::from_secs(5),
            Some(&masks.latch),
        ))
        .unwrap();
    assert_eq!(t, i64::from(b'a'));
    assert!(!log.died());
    assert_eq!(
        log.frames(),
        vec![
            (FrameKind::Reset, false),
            (FrameKind::Forward, true),
            (FrameKind::Reset, false),
            (FrameKind::Forward, false),
        ]
    );
    assert!(masks.latch.is_set(), "a plain reply proves nothing");
}

#[test]
fn a_middle_takes_the_prefill_budget_only_for_the_first_sampled_step() {
    let mut sampled = false;
    let (rank, total) = (1, 3);
    let pre = crate::dist::mid_mask_budget(prefill_reply_deadline(), rank, total);
    let dec = crate::dist::mid_mask_budget(decode_reply_deadline(), rank, total);
    // Prefill sync steps (not sampled) keep the prefill budget.
    assert_eq!(mid_forward_budget(&mut sampled, false, rank, total), pre);
    assert_eq!(mid_forward_budget(&mut sampled, true, rank, total), pre);
    assert_eq!(mid_forward_budget(&mut sampled, true, rank, total), dec);
    assert!(dec < decode_reply_deadline() && pre < prefill_reply_deadline());
    // A Reset clears it.
    sampled = false;
    assert_eq!(mid_forward_budget(&mut sampled, true, rank, total), pre);
}

/// Runs `drive` on its own thread and fails after 10 s instead of hanging on
/// a reply that never comes.
fn drive_bounded(
    mut eng: PipelineEngine<JsonStub>,
    ids: &'static [&'static str],
) -> (PipelineEngine<JsonStub>, HashMap<String, Outcome>) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let out = drive(&mut eng, ids);
        let _ = tx.send((eng, out));
    });
    rx.recv_timeout(Duration::from_secs(10))
        .expect("tasks did not finish within 10 s")
}

/// A stream-mode middle stage: a late acknowledged reply to a probe it
/// already answered upstream (its downstream was slow) turns up between
/// stream replies. It is dropped, the link stays, and rank 0 hears of it
/// before the next stream reply (its latch clears).
#[test]
fn a_stream_middle_drops_a_late_probe_ack_and_keeps_the_link() {
    let rt = runtime();
    let l = links(&rt, 2);
    // The late reply is already on the middle's downstream link.
    rt.block_on(crate::dist::send_token_ack_upstream(
        &l[1].0, -1, 7, MASK_ACK,
    ))
    .unwrap();
    let log = fixture_tail(&rt, l[1].0.clone(), false);
    let mut mid = PipelineEngine::new(
        JsonStub::new(),
        None,
        StageTransport {
            upstream: Some(l[0].0.clone()),
            downstream: Some(l[1].1.clone()),
        },
        rt.handle().clone(),
        1,
        3,
        None,
    );
    assert_eq!(mid.enable_streams(2), 2);
    std::thread::spawn(move || while mid.step().is_ok() {});
    let mut head = head_engine(&rt, JsonStub::new(), l[0].1.clone(), 3, 2);
    head.stage_masks.latch.set();
    head.submit(plain_task("p1", 6)).unwrap();
    let (mut head, out) = drive_bounded(head, &["p1"]);
    assert_eq!(final_chunk(&out["p1"]).error, None);
    assert_eq!(out["p1"].tokens, vec![u32::from(b'a'); 6]);
    assert!(
        !head.stage_masks.latch.is_set(),
        "the late ack reached rank 0"
    );
    // The streams keep running on the same link.
    head.submit(plain_task("p2", 4)).unwrap();
    let (_head, out) = drive_bounded(head, &["p2"]);
    assert_eq!(final_chunk(&out["p2"]).error, None);
    assert!(!log.died());
}

/// Steps until every task in `ids` has its final chunk, noting the step of
/// each task's first token and of its final chunk.
fn drive_timeline(
    eng: &mut PipelineEngine<JsonStub>,
    ids: &[&str],
) -> (HashMap<String, Outcome>, HashMap<String, (usize, usize)>) {
    let mut out: HashMap<String, Outcome> = HashMap::new();
    let mut first: HashMap<String, usize> = HashMap::new();
    let mut last: HashMap<String, usize> = HashMap::new();
    for n in 0..10_000 {
        if ids.iter().all(|id| last.contains_key(*id)) {
            let at = ids
                .iter()
                .map(|id| {
                    let id = id.to_string();
                    let f = first.get(&id).copied().unwrap_or(usize::MAX);
                    (id.clone(), (f, last[&id]))
                })
                .collect();
            return (out, at);
        }
        for (id, c) in eng.step().unwrap() {
            if !c.token_ids.is_empty() {
                first.entry(id.clone()).or_insert(n);
            }
            let o = out.entry(id.clone()).or_default();
            o.tokens.extend(c.token_ids.iter().map(|&t| t as u32));
            o.text.push_str(&c.text);
            if c.is_final {
                last.insert(id, n);
                o.last = Some(c);
            }
        }
    }
    panic!("tasks did not finish: {out:?}");
}

/// A stream head on a link with no mask evidence yet, one long plain stream
/// running: a json request (which needs the idle link for its probe) and a
/// short plain request queue behind it.
fn json_waiting_for_the_probe() -> (
    tokio::runtime::Runtime,
    PipelineEngine<JsonStub>,
    Arc<TailLog>,
) {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let log = fixture_tail(&rt, l[0].0.clone(), false);
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 3);
    head.submit(plain_task("p1", 40)).unwrap();
    for _ in 0..3 {
        head.step().unwrap();
    }
    assert_eq!(head.streams.len(), 1, "the long stream is running");
    // The fixture's sampler (lowest allowed token) never closes the JSON:
    // the json request ends at its cap.
    head.submit(json_task("j", 6)).unwrap();
    head.submit(plain_task("p2", 3)).unwrap();
    (rt, head, log)
}

#[test]
fn plain_requests_behind_a_json_request_waiting_for_its_probe_are_admitted() {
    let (_rt, mut head, log) = json_waiting_for_the_probe();
    let (out, at) = drive_timeline(&mut head, &["p1", "j", "p2"]);
    assert!(
        at["p2"].1 < at["p1"].1,
        "the short plain request waited for the long stream to drain: {at:?}"
    );
    assert_eq!(final_chunk(&out["p2"]).error, None);
    assert_eq!(final_chunk(&out["p1"]).error, None);
    assert_eq!(final_chunk(&out["j"]).error, None);
    assert_eq!(out["j"].tokens.len(), 6);
    assert_eq!(log.masked(FrameKind::Forward), 1, "probed once");
}

#[test]
fn a_json_request_waiting_past_its_patience_stops_new_admissions() {
    let (_rt, mut head, log) = json_waiting_for_the_probe();
    head.probe_wait_since = Some(Instant::now() - JSON_PROBE_PATIENCE - Duration::from_secs(1));
    let (out, at) = drive_timeline(&mut head, &["p1", "j", "p2"]);
    assert!(
        at["p1"].1 <= at["j"].0 && at["j"].0 <= at["p2"].0,
        "the link idles, the json request is probed and runs first: {at:?}"
    );
    assert_eq!(final_chunk(&out["j"]).error, None);
    assert_eq!(out["j"].tokens.len(), 6);
    assert_eq!(final_chunk(&out["p2"]).error, None);
    assert_eq!(log.masked(FrameKind::Forward), 1, "probed once");
}

// ---- repeated masks on the stage wire ----------------------------------------

/// A JSON string the fixture tail fills with the lowest allowed byte: every
/// step inside it has the same mask.
const STRING: &str = r#"{"type":"string"}"#;
/// Strings of one letter each: two streams with these never share a mask.
const XS: &str = r#"{"type":"string","enum":["xxxxxxxx"]}"#;
const YS: &str = r#"{"type":"string","enum":["yyyyyyyy"]}"#;

fn schema_task(id: &str, schema: &str, max_tokens: u32) -> GenerationTask {
    let mut t = GenerationTask::new(id, "hi").with_max_tokens(max_tokens);
    t.grammar = Some(test_env::spec(schema));
    t
}

/// Every token of `o` was allowed by `schema` when it was picked.
fn assert_allowed(o: &Outcome, schema: &str) {
    let mut gm = test_env::mask(schema);
    for &t in &o.tokens {
        assert!(
            gm.allowed(t),
            "token {t} not allowed by {schema}: {:?}",
            o.text
        );
        gm.accept(t).unwrap();
    }
}

/// The one-task path: a mask equal to the last one sent goes as the
/// marker, and the tail, resolving it, picks the same tokens.
#[test]
fn repeated_masks_go_as_markers_on_the_one_task_path() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let log = fixture_tail(&rt, l[0].0.clone(), false);
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 0);
    head.submit(schema_task("j", STRING, 8)).unwrap();
    let out = drive(&mut head, &["j"]);
    assert_eq!(final_chunk(&out["j"]).error, None);
    assert_eq!(out["j"].tokens.len(), 8);
    assert_allowed(&out["j"], STRING);
    assert!(log.repeats() >= 5, "{} markers", log.repeats());
    assert!(!head.stage_masks.latch.is_set());
    let _ = &rt;
}

/// Two json streams with the same mask, each in its own group so their
/// frames are in flight together: markers flow through a middle, and both
/// come out right.
#[test]
fn pipelined_json_streams_share_markers_through_a_middle() {
    let mut c = chain(3, 2, JsonStub::new);
    c.head.submit(schema_task("a", XS, 12)).unwrap();
    c.head.submit(schema_task("b", XS, 12)).unwrap();
    let out = drive(&mut c.head, &["a", "b"]);
    for id in ["a", "b"] {
        assert_eq!(final_chunk(&out[id]).error, None, "{id}");
        assert_eq!(out[id].text, "\"xxxxxxxx\"", "{id}");
    }
    assert!(c.head.mask_send.markers_sent() > 0);
    assert!(!c.head.stage_masks.latch.is_set());
}

/// Two pipelined json streams with different masks through a middle never
/// get each other's mask.
#[test]
fn pipelined_json_streams_with_different_masks_never_swap_them() {
    let mut c = chain(3, 2, JsonStub::new);
    c.head.submit(schema_task("x", XS, 12)).unwrap();
    c.head.submit(schema_task("y", YS, 12)).unwrap();
    let out = drive(&mut c.head, &["x", "y"]);
    assert_eq!(final_chunk(&out["x"]).error, None);
    assert_eq!(final_chunk(&out["y"]).error, None);
    assert_eq!(out["x"].text, "\"xxxxxxxx\"");
    assert_eq!(out["y"].text, "\"yyyyyyyy\"");
}

/// A marker the tail can't resolve fails that task with the retry message:
/// no latch, rank 0 forgets its mask, and a plain stream next to it keeps
/// going.
#[test]
fn a_mask_miss_fails_only_that_task_and_never_latches() {
    // The one-task path.
    let rt = runtime();
    let mut l = links(&rt, 1);
    let _log = fixture_tail_with(&rt, l[0].0.clone(), false, true);
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 0);
    head.submit(schema_task("j", STRING, 8)).unwrap();
    let out = drive(&mut head, &["j"]);
    let err = final_chunk(&out["j"]).error.clone().unwrap_or_default();
    assert!(err.contains(cascadia_grammar::MASK_MISS_MESSAGE), "{err}");
    assert!(!head.stage_masks.latch.is_set());
    assert!(head.mask_send.is_empty());

    // Streams: the json one fails, the plain one in the same pipeline does not.
    let rt = runtime();
    let mut l = links(&rt, 1);
    let _log = fixture_tail_with(&rt, l[0].0.clone(), false, true);
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 2);
    head.submit(schema_task("j", XS, 12)).unwrap();
    head.submit(plain_task("p", 12)).unwrap();
    let out = drive(&mut head, &["j", "p"]);
    let err = final_chunk(&out["j"]).error.clone().unwrap_or_default();
    assert!(err.contains(cascadia_grammar::MASK_MISS_MESSAGE), "{err}");
    assert_eq!(final_chunk(&out["p"]).error, None);
    assert_eq!(out["p"].tokens.len(), 12);
    assert!(!head.stage_masks.latch.is_set());
    assert!(head.mask_send.is_empty());
}

/// A re-dial is a new connection: the next json frame goes full, even when
/// it repeats the last mask sent on the old one.
#[test]
fn a_redial_starts_the_link_with_full_masks() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let log = fixture_tail(&rt, l[0].0.clone(), false);
    let mut head = head_engine(&rt, JsonStub::new(), l.remove(0).1, 2, 0);
    // One masked step each, all with the opening quote's mask.
    let one = |head: &mut PipelineEngine<JsonStub>, id: &str| {
        head.submit(schema_task(id, XS, 1)).unwrap();
        let out = drive(head, &[id]);
        assert_eq!(final_chunk(&out[id]).error, None);
    };
    one(&mut head, "j1");
    assert_eq!(log.repeats(), 0);
    one(&mut head, "j2");
    assert_eq!(log.repeats(), 1, "the same mask again: a marker");
    // The link keeper re-dialed between two requests.
    head.link_epoch
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    one(&mut head, "j3");
    assert_eq!(log.repeats(), 1, "full on the new connection");
    one(&mut head, "j4");
    assert_eq!(log.repeats(), 2);
    let _ = &rt;
}

/// A re-dial that fails keeps the link down and the retry spaced out: the
/// next step neither treats it as a new connection nor dials again at once.
#[test]
fn a_failed_redial_keeps_the_backoff_and_the_link_down() {
    let rt = runtime();
    let mut l = links(&rt, 1);
    let (srv, down) = l.remove(0);
    let mut head = head_engine(&rt, JsonStub::new(), down, 2, 2);
    // Nothing listens on the downstream's address any more.
    rt.block_on(async { srv.lock().await.close().await });
    drop(srv);
    head.peer_disconnected = true;
    let epoch = head.link_epoch.load(std::sync::atomic::Ordering::SeqCst);
    let _ = head.step();
    assert!(
        head.redial_next.is_some(),
        "the re-dial was tried and spaced out"
    );
    let _ = head.step();
    assert_eq!(
        head.link_epoch.load(std::sync::atomic::Ordering::SeqCst),
        epoch,
        "no new connection was counted"
    );
    assert!(head.peer_disconnected, "the link is still down");
    assert!(head.redial_next.is_some(), "the backoff is kept");
}

/// The last rank, a marker it can't resolve on a stream's prompt window:
/// it answers that row with the mask-miss token at once, without opening
/// the slot or running its layers.
#[test]
fn a_stream_window_miss_at_the_last_rank_runs_no_layer() {
    use crate::dist::*;
    let rt = runtime();
    let mut l = links(&rt, 1);
    let (srv, cli) = l.remove(0);
    let mut tail = PipelineEngine::new(
        JsonStub::new(),
        None,
        StageTransport {
            upstream: Some(srv),
            downstream: None,
        },
        rt.handle().clone(),
        1,
        2,
        None,
    );
    assert_eq!(tail.enable_streams(2), 2);
    let words = vec![u32::MAX; VOCAB.div_ceil(32)];
    let mut cols = MaskCols::new(1, words.len());
    cols.set(0, 7, &words);
    // The sender believes the tail holds this mask; the tail is fresh.
    let send = MaskSendLink::default();
    send.record_for_tests(&words);
    rt.block_on(send_stream_open_step(
        &cli,
        3,
        1,
        &crate::sampling::SamplingConfig::default(),
        &[0.5; 4],
        1,
        4,
        Some(&cols),
        Some(&send),
    ))
    .unwrap();
    assert_eq!(send.markers_sent(), 1);
    tail.step().unwrap();
    let reply = rt.block_on(async {
        assert_eq!(
            recv_kind_client_raw(&cli).await.unwrap(),
            Some(FrameKind::StreamTokensAck)
        );
        recv_stream_tokens_ack_body_client(&cli).await.unwrap()
    });
    assert_eq!(
        reply,
        (
            3,
            MASK_ACK,
            vec![(1, i64::from(cascadia_grammar::MASK_MISS_TOKEN))]
        )
    );
    assert_eq!(tail.runner.stream_pos(1), None, "the slot was never opened");
}
