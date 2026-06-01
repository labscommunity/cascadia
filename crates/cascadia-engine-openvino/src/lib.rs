//! OpenVINO engines for cascadia.
//!
//! Currently exposes [`OvGenaiBuilder`] / [`OvGenaiEngine`] — the Rust
//! port of `cascadia/worker/engines/openvino/genai_engine.py`. Wraps
//! `cascadia-ov-genai-shim` (FFI to openvino-genai) under the
//! [`cascadia_engine::Engine`] and [`cascadia_engine::Builder`] traits.
//!
//! Two further engines from the Python tree (`ov-runtime`,
//! `ov-dist-spec`) are deferred — they need the lower-level
//! `openvino` crate (Core/CompiledModel/InferRequest) and the
//! distributed spec-decode protocol; tracked separately.

pub mod dist_spec;
pub mod genai;
pub mod rotary;
pub mod runtime;

pub use dist_spec::{
    DistributedMaskedReq, FrameKind, MaskedReq, OvDistSpecBuilder, OvDistSpecEngine,
    OvDistSpecWorkerBuilder, OvDistSpecWorkerEngine, SpecDecodeStats,
};
pub use genai::{OvGenaiBuilder, OvGenaiEngine};
pub use rotary::{ModelTextConfig, RopeScalingConfig, Rotary};
pub use runtime::{OvRuntimeBuilder, OvRuntimeEngine};

use cascadia_types::{GenerationTask, TaskId};

/// Remove `task_id` from `pending` and clear it from `active` if in-flight.
/// Returns `true` if the active task was cleared.
///
/// No reset: the next task's activation resets the engine (as on normal
/// completion). Direct mutation is safe — `cancel` and `step` never overlap
/// (both `&mut self`, engine held behind a `Mutex`).
pub(crate) fn clear_task<A>(
    pending: &mut Vec<GenerationTask>,
    active: &mut Option<A>,
    task_id: &TaskId,
    active_is: impl Fn(&A, &TaskId) -> bool,
) -> bool {
    pending.retain(|t| t.task_id != *task_id);
    if active.as_ref().is_some_and(|a| active_is(a, task_id)) {
        *active = None;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod cancel_tests {
    use super::clear_task;
    use cascadia_types::GenerationTask;

    /// Minimal stand-in for an engine's `ActiveTask` / `ActiveSpec`:
    /// both expose `.task.task_id`, which is all the matcher reads.
    struct FakeActive {
        task: GenerationTask,
    }

    fn gen_task(id: &str) -> GenerationTask {
        GenerationTask::new(id, "prompt")
    }

    #[test]
    fn drops_matching_pending_task() {
        let mut pending = vec![gen_task("a"), gen_task("b")];
        let mut active: Option<FakeActive> = None;
        let cleared = clear_task(&mut pending, &mut active, &"a".to_string(), |a, id| {
            a.task.task_id == *id
        });
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].task_id, "b");
        assert!(!cleared);
    }

    #[test]
    fn clears_matching_active() {
        let mut pending: Vec<GenerationTask> = vec![];
        let mut active = Some(FakeActive {
            task: gen_task("x"),
        });
        let cleared = clear_task(&mut pending, &mut active, &"x".to_string(), |a, id| {
            a.task.task_id == *id
        });
        assert!(active.is_none());
        assert!(cleared);
    }

    #[test]
    fn leaves_nonmatching_active() {
        let mut pending: Vec<GenerationTask> = vec![];
        let mut active = Some(FakeActive {
            task: gen_task("x"),
        });
        let cleared = clear_task(&mut pending, &mut active, &"y".to_string(), |a, id| {
            a.task.task_id == *id
        });
        assert!(active.is_some());
        assert!(!cleared);
    }

    #[test]
    fn unknown_id_is_noop() {
        let mut pending = vec![gen_task("a")];
        let mut active: Option<FakeActive> = None;
        let cleared = clear_task(&mut pending, &mut active, &"zzz".to_string(), |a, id| {
            a.task.task_id == *id
        });
        assert_eq!(pending.len(), 1);
        assert!(!cleared);
    }

    #[test]
    fn clears_active_and_leaves_unrelated_pending() {
        // Cancel the in-flight task; an unrelated queued task must survive.
        let mut pending = vec![gen_task("keep")];
        let mut active = Some(FakeActive {
            task: gen_task("x"),
        });
        let cleared = clear_task(&mut pending, &mut active, &"x".to_string(), |a, id| {
            a.task.task_id == *id
        });
        assert!(cleared);
        assert!(active.is_none());
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].task_id, "keep");
    }

    #[test]
    fn cancels_pending_while_active_keeps_running() {
        // Cancel a queued task; a different in-flight task keeps running.
        let mut pending = vec![gen_task("queued")];
        let mut active = Some(FakeActive {
            task: gen_task("inflight"),
        });
        let cleared = clear_task(&mut pending, &mut active, &"queued".to_string(), |a, id| {
            a.task.task_id == *id
        });
        assert!(!cleared);
        assert!(active.is_some());
        assert!(pending.is_empty());
    }
}
