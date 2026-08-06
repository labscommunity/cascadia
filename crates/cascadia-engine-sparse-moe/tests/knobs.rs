//! `seed` is process-global and once-only, so this is ONE test in its own
//! integration binary — separate tests would race and their outcome would
//! depend on execution order.

use cascadia_engine_sparse_moe::k3::knobs::{get, seed, Overrides};

#[test]
fn seed_is_once_and_unseeded_defers_to_env() {
    assert!(get().simd.is_none(), "unseeded overrides must be all-None");
    assert!(seed(Overrides {
        simd: Some(false),
        ..Default::default()
    }));
    assert!(
        !seed(Overrides {
            simd: Some(true),
            ..Default::default()
        }),
        "second seed must be ignored"
    );
    assert_eq!(get().simd, Some(false));
    assert_eq!(get().max_seq, None, "unset fields stay None");
}
