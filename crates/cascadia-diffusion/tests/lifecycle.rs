use cascadia_diffusion::{lease::Gate, Error};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};
fn reserve(gate: &Gate, now: Instant) -> cascadia_diffusion::lease::LeaseId {
    gate.reserve(
        gate.boot(),
        "peer-a",
        &"a".repeat(64),
        BTreeSet::from(["unet_first".into()]),
        Duration::from_secs(10),
        now,
    )
    .unwrap()
}
#[test]
fn cancellation_and_expiry_retain_running_work() {
    for cancelled in [false, true] {
        let gate = Gate::new();
        let now = Instant::now();
        let id = reserve(&gate, now);
        let execution = gate
            .begin("peer-a", id, &"a".repeat(64), "unet_first", 1, now)
            .unwrap();
        if cancelled {
            gate.release("peer-a", id).unwrap();
        }
        let later = now + Duration::from_secs(11);
        assert!(gate.occupied(later));
        assert!(gate
            .begin("peer-a", id, &"a".repeat(64), "unet_first", 2, later)
            .is_err());
        assert!(gate
            .renew("peer-a", id, Duration::from_secs(10), later)
            .is_err());
        execution.complete(later);
        assert!(!gate.occupied(later));
        reserve(&gate, later);
    }
}
#[test]
fn lost_execution_quarantines_even_after_release_and_expiry() {
    let gate = Gate::new();
    let now = Instant::now();
    let id = reserve(&gate, now);
    let execution = gate
        .begin("peer-a", id, &"a".repeat(64), "unet_first", 1, now)
        .unwrap();
    gate.release("peer-a", id).unwrap();
    drop(execution);
    assert!(gate.quarantined());
    assert!(gate.occupied(now + Duration::from_secs(7200)));
    assert!(matches!(
        gate.reserve(
            gate.boot(),
            "peer-b",
            &"a".repeat(64),
            BTreeSet::from(["unet_first".into()]),
            Duration::from_secs(10),
            now + Duration::from_secs(7200)
        ),
        Err(Error::Quarantined)
    ));
}
#[test]
fn identity_sequence_component_and_boot_are_fenced() {
    let gate = Gate::new();
    let now = Instant::now();
    let id = reserve(&gate, now);
    for (owner, manifest, component, sequence) in [
        ("peer-b", "a", "unet_first", 1),
        ("peer-a", "b", "unet_first", 1),
        ("peer-a", "a", "unet_second", 1),
        ("peer-a", "a", "unet_first", 2),
    ] {
        assert!(gate
            .begin(owner, id, &manifest.repeat(64), component, sequence, now)
            .is_err());
    }
    assert!(gate.release("peer-b", id).is_err());
    let execution = gate
        .begin("peer-a", id, &"a".repeat(64), "unet_first", 1, now)
        .unwrap();
    assert!(gate
        .begin("peer-a", id, &"a".repeat(64), "unet_first", 1, now)
        .is_err());
    assert!(gate
        .begin("peer-a", id, &"a".repeat(64), "unet_first", 2, now)
        .is_err());
    execution.complete(now);
    assert!(gate
        .begin("peer-a", id, &"a".repeat(64), "unet_first", 1, now)
        .is_err());
    let replacement = Gate::new();
    assert!(replacement
        .begin("peer-a", id, &"a".repeat(64), "unet_first", 2, now)
        .is_err());
}
#[test]
fn drain_finishes_admitted_job_and_lifetime_cannot_be_extended() {
    let gate = Gate::new();
    let now = Instant::now();
    let id = reserve(&gate, now);
    gate.drain();
    gate.begin("peer-a", id, &"a".repeat(64), "unet_first", 1, now)
        .unwrap()
        .complete(now);
    for second in (5..3600).step_by(5) {
        gate.renew(
            "peer-a",
            id,
            Duration::from_secs(10),
            now + Duration::from_secs(second),
        )
        .unwrap();
    }
    assert!(gate
        .renew(
            "peer-a",
            id,
            Duration::from_secs(10),
            now + Duration::from_secs(3600)
        )
        .is_err());
    assert!(!gate.occupied(now + Duration::from_secs(3600)));
    assert!(matches!(
        gate.reserve(
            gate.boot(),
            "peer-b",
            &"a".repeat(64),
            BTreeSet::from(["unet_first".into()]),
            Duration::from_secs(10),
            now + Duration::from_secs(3600)
        ),
        Err(Error::Capacity)
    ));
}
