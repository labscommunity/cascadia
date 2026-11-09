use super::*;

const A: &[u32] = &[0b1010, 0, 7];
const B: &[u32] = &[0b0110, 1, 7];

#[test]
fn repeat_is_the_largest_width_and_never_zero() {
    assert_eq!(REPEAT, u32::MAX);
    assert_ne!(REPEAT, 0, "W = 0 is the sparse-moe mask-support probe");
    assert_eq!(MASK_MISS_TOKEN, -2);
    assert!(MASK_MISS_MESSAGE.contains("retry the request"));
}

#[test]
fn a_fresh_sender_sends_full_then_repeats_the_same_mask() {
    let mut send = MaskSendCache::default();
    assert!(!send.repeats(0, A), "nothing sent yet");
    send.record(0, A);
    assert!(send.repeats(0, A));
    assert!(!send.repeats(0, B), "a changed mask goes full");
    send.record(0, B);
    assert!(send.repeats(0, B));
    assert!(!send.repeats(0, A));
}

#[test]
fn the_sender_repeats_only_on_the_connection_it_recorded_on() {
    let mut send = MaskSendCache::default();
    send.record(3, A);
    assert!(send.repeats(3, A));
    assert!(!send.repeats(4, A), "a re-dialed link starts full");
}

#[test]
fn a_cleared_sender_sends_full() {
    let mut send = MaskSendCache::default();
    send.record(0, A);
    send.clear();
    assert!(!send.repeats(0, A));
    assert!(send.is_empty());
}

#[test]
fn the_receiver_resolves_full_masks_and_repeats_of_the_committed_one() {
    let mut recv = MaskRecvCache::default();
    assert_eq!(recv.resolve(WireMask::Full(A.to_vec())), Ok(A.to_vec()));
    // Not committed yet: a repeat right now is a miss.
    assert_eq!(recv.resolve(WireMask::Repeat), Err(MaskMiss));
    recv.commit(A);
    assert_eq!(recv.resolve(WireMask::Repeat), Ok(A.to_vec()));
    assert_eq!(recv.resolve(WireMask::Full(B.to_vec())), Ok(B.to_vec()));
    recv.commit(B);
    assert_eq!(recv.resolve(WireMask::Repeat), Ok(B.to_vec()));
}

#[test]
fn a_cleared_receiver_misses_a_repeat() {
    let mut recv = MaskRecvCache::default();
    recv.commit(A);
    recv.clear();
    assert_eq!(recv.resolve(WireMask::Repeat), Err(MaskMiss));
    assert_eq!(recv.resolve(WireMask::Full(A.to_vec())), Ok(A.to_vec()));
}

/// The two ends of one link stay in step: the sender records what it wrote,
/// the receiver commits what it decoded, and a repeat always resolves to the
/// mask the sender meant.
#[test]
fn sender_and_receiver_agree_over_a_mask_sequence() {
    let mut send = MaskSendCache::default();
    let mut recv = MaskRecvCache::default();
    let seq: [&[u32]; 6] = [A, A, A, B, B, A];
    let mut repeats = 0;
    for words in seq {
        let wire = if send.repeats(0, words) {
            repeats += 1;
            WireMask::Repeat
        } else {
            WireMask::Full(words.to_vec())
        };
        send.record(0, words);
        let got = recv.resolve(wire).unwrap();
        recv.commit(&got);
        assert_eq!(got, words);
    }
    assert_eq!(repeats, 3);
}
