//! Deterministic FNV-1a-64 rendered as fixed-width lowercase hex.
//!
//! Used for wire-stable identity hashes (mDNS TXT `model_hash`, ring
//! membership digests). MUST NOT use `std::collections::hash_map::
//! DefaultHasher` — its output is not stable across builds/hosts, which
//! would make two nodes on different cascadia versions disagree.

/// FNV-1a-64 over `bytes`, rendered as 16 lowercase zero-padded hex chars.
/// Pure, endianness-independent, identical on every host and build.
pub fn fnv1a_hex(bytes: &[u8]) -> String {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors_are_stable() {
        // FNV-1a-64 reference vectors (canonical; hand-verified).
        assert_eq!(fnv1a_hex(b""), "cbf29ce484222325");
        assert_eq!(fnv1a_hex(b"a"), "af63dc4c8601ec8c");
    }

    #[test]
    fn output_is_always_16_hex_chars() {
        for s in ["", "a", "hello", "Qwen/Qwen3-1.7B", "\u{1f600}"] {
            let h = fnv1a_hex(s.as_bytes());
            assert_eq!(h.len(), 16, "hash of {s:?} not 16 chars: {h}");
            assert!(h
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        }
    }

    #[test]
    fn distinct_inputs_including_prefixes_differ() {
        assert_ne!(fnv1a_hex(b"model"), fnv1a_hex(b"model-v2"));
    }
}
