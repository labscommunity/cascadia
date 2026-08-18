//! `layer_split` is the single source of truth for a rank's layer range.
//!
//! `load_staged` derives its slice from it, and the out-of-process scheduler
//! publishes the same range in a `ShardDescriptor` whose contiguity rule
//! consumes. A gap, an overlap, or an orphaned head layer does not surface as an
//! inference error — it breaks chain formation silently — so the guarantees are
//! asserted here rather than assumed by callers.

use cascadia_engine_sparse_moe::glm::loader::GlmManifest;
use cascadia_engine_sparse_moe::glm::stage::layer_split;

fn manifest(num_layers: usize, indexer_types: &[&str]) -> GlmManifest {
    GlmManifest {
        arch: "glm5".into(),
        num_layers,
        indexer_types: indexer_types.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

#[test]
fn even_split_covers_every_layer_exactly_once() {
    let m = manifest(8, &[]);
    let ranges: Vec<_> = (0..4).map(|r| layer_split(&m, r, 4).unwrap()).collect();
    assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 6), (6, 8)]);
}

#[test]
fn index_aligned_split_covers_every_layer_exactly_once() {
    // "full" layers compute their own top-k; "shared" layers reuse the carry, so
    // a rank must start on a "full" layer.
    let types = [
        "full", "shared", "shared", "full", "shared", "full", "shared", "shared",
    ];
    let m = manifest(8, &types);
    let ranges: Vec<_> = (0..2).map(|r| layer_split(&m, r, 2).unwrap()).collect();
    assert_eq!(ranges[0].0, 0, "rank 0 must start at layer 0");
    assert_eq!(ranges[1].1, 8, "last rank must end at num_layers");
    assert_eq!(ranges[0].1, ranges[1].0, "ranges must be contiguous");
}

#[test]
fn split_is_contiguous_and_complete_for_all_rank_counts() {
    for &n in &[8usize, 61, 79] {
        for &total in &[1u32, 2, 4] {
            let m = manifest(n, &[]);
            let mut prev_end = 0usize;
            for rank in 0..total {
                let (lo, hi) = layer_split(&m, rank, total).unwrap();
                assert_eq!(
                    lo, prev_end,
                    "gap/overlap at n={n} total={total} rank={rank}"
                );
                assert!(lo < hi, "rank {rank} of {total} owns zero layers at n={n}");
                prev_end = hi;
            }
            assert_eq!(prev_end, n, "layers orphaned at n={n} total={total}");
        }
    }
}

#[test]
fn index_aligned_split_is_contiguous_across_four_ranks() {
    // Realistic shape: full layers spaced so the even boundaries must snap.
    let types = [
        "full", "shared", "full", "shared", "full", "shared", "full", "shared", "full", "shared",
        "full", "shared",
    ];
    let m = manifest(12, &types);
    let mut prev_end = 0usize;
    for rank in 0..4 {
        let (lo, hi) = layer_split(&m, rank, 4).unwrap();
        assert_eq!(lo, prev_end, "gap/overlap at rank {rank}");
        assert!(lo < hi, "rank {rank} owns zero layers");
        prev_end = hi;
    }
    assert_eq!(prev_end, 12, "tail layers orphaned");
}

#[test]
fn rejects_rank_at_or_past_total() {
    let m = manifest(8, &[]);
    assert!(layer_split(&m, 4, 4).is_err());
    assert!(layer_split(&m, 9, 4).is_err());
}

#[test]
fn rejects_total_exceeding_layer_count() {
    let m = manifest(3, &[]);
    assert!(layer_split(&m, 3, 8).is_err());
}

#[test]
fn orphaned_head_layers_are_rejected() {
    // Layer 0 is "shared": index-aligned anchors rank 0 at fulls[0] == 1, which
    // would silently orphan layer 0. No scheduler rule catches that — rule 3
    // checks adjacency only — so it must be an error here, not a silent gap.
    let m = manifest(4, &["shared", "full", "shared", "full"]);
    let err = layer_split(&m, 0, 2).unwrap_err();
    assert!(err.contains("layer 0"), "unexpected message: {err}");
}
