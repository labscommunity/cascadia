//! Host-side, engine-dep-free helpers the engine's `lookup`/`reslice_to` reuse (plan M1) — kept
//! here so they are [local]-testable without the OpenVINO SDK.

/// Longest-common-prefix length of two token-id sequences. The holder's `lookup` returns this as
/// the negotiated `prefix_token_len` (§6 partial-prefix negotiation).
#[must_use]
pub fn longest_common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_basic() {
        assert_eq!(longest_common_prefix_len(&[1, 2, 3, 4], &[1, 2, 3, 9]), 3);
        assert_eq!(longest_common_prefix_len(&[1, 2], &[3, 4]), 0);
        assert_eq!(longest_common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(longest_common_prefix_len(&[], &[1, 2]), 0);
    }
}
