use tahoma_types::{
    Chunk, GenerationTask, LoadProgress, PeerEndpoint, PeerLayout, ShardPlan, ShardSpec,
};

#[test]
fn generation_task_defaults() {
    let task = GenerationTask::new("t1", "hello");
    assert_eq!(task.max_tokens, 256);
    assert_eq!(task.temperature, 0.0);
    assert_eq!(task.logprobs, 0);
    assert!(!task.enable_thinking);
    assert!(!task.trust_remote_code);
}

#[test]
fn generation_task_builder() {
    let task = GenerationTask::new("t1", "hi")
        .with_max_tokens(64)
        .with_temperature(0.5);
    assert_eq!(task.max_tokens, 64);
    assert_eq!(task.temperature, 0.5);
}

#[test]
fn generation_task_serde_roundtrip() {
    let task = GenerationTask::new("t1", "prompt").with_max_tokens(8);
    let json = serde_json::to_string(&task).unwrap();
    let back: GenerationTask = serde_json::from_str(&json).unwrap();
    assert_eq!(task, back);
}

#[test]
fn chunk_token_constructor() {
    let c = Chunk::token("t1", 42, "hello");
    assert_eq!(c.token_id, 42);
    assert_eq!(c.text, "hello");
    assert!(!c.is_final);
}

#[test]
fn chunk_final_marker() {
    let c = Chunk::final_marker("t1", "");
    assert!(c.is_final);
}

#[test]
fn load_progress_ready() {
    let p = LoadProgress::ready();
    assert_eq!(p.bytes_loaded, 1);
    assert_eq!(p.bytes_total, Some(1));
    assert_eq!(p.message, "ready");
}

#[test]
fn shard_spec_num_layers() {
    let s = ShardSpec {
        model_id: "x".into(),
        layer_start: 4,
        layer_end: 12,
        total_layers: 32,
        device: "GPU".into(),
        is_first_stage: false,
        is_last_stage: false,
        tp_size: 1,
        tp_rank: 0,
    };
    assert_eq!(s.num_layers(), 8);
    assert!(!s.is_tp());
}

#[test]
fn shard_spec_single_stage() {
    let s = ShardSpec::single_stage("model-id", "GPU");
    assert!(s.is_first_stage);
    assert!(s.is_last_stage);
    assert_eq!(s.device, "GPU");
}

#[test]
fn shard_plan_uniform_even_split() {
    let plan =
        ShardPlan::uniform("model", 32, 4096, 32, 32000, 4, None).expect("uniform should succeed");
    assert_eq!(plan.total_stages(), 4);
    for (i, stage) in plan.stages.iter().enumerate() {
        assert_eq!(stage.num_layers(), 8);
        assert_eq!(stage.is_first_stage, i == 0);
        assert_eq!(stage.is_last_stage, i == 3);
    }
}

#[test]
fn shard_plan_uneven_remainder_to_last_stage() {
    let plan = ShardPlan::uniform("model", 33, 4096, 32, 32000, 4, None).unwrap();
    assert_eq!(plan.stages[0].num_layers(), 8);
    assert_eq!(plan.stages[3].num_layers(), 9);
}

#[test]
fn shard_plan_rejects_too_many_stages() {
    assert!(ShardPlan::uniform("m", 4, 0, 0, 0, 5, None).is_err());
}

#[test]
fn peer_endpoint_display() {
    assert_eq!(
        PeerEndpoint::new("127.0.0.1", 9100).to_string(),
        "127.0.0.1:9100"
    );
}

#[test]
fn peer_layout_constructors() {
    assert!(PeerLayout::single_stage().upstream.is_none());
    assert!(PeerLayout::single_stage().downstream.is_none());
    let first = PeerLayout::first_of(PeerEndpoint::new("h", 1));
    assert!(first.upstream.is_none());
    assert!(first.downstream.is_some());
    let last = PeerLayout::last_of(PeerEndpoint::new("h", 1));
    assert!(last.upstream.is_some());
    assert!(last.downstream.is_none());
}
