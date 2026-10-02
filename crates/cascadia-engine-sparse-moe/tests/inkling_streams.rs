//! Multi-stream decode on the Inkling runner: a stream decoded inside a
//! batch of streams must equal the same stream decoded alone, bit for bit
//! (logits and greedy tokens), including a stream admitted while others are
//! mid-decode, and slots must be reusable after close.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

fn fixture() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    if dir.join("reference.json").exists() {
        Some(dir)
    } else {
        eprintln!("inkling_export fixture missing; skipping");
        None
    }
}

fn reference_prompt(dir: &PathBuf) -> (Vec<u32>, Vec<u32>) {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    (ids("prompt_ids"), ids("greedy_ids"))
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn load(dir: &PathBuf) -> InklingRunner {
    InklingRunner::load_staged(dir, 64, 0, 1, 0, 0, Some("eager".into()), None).expect("runner")
}

/// Classic single-sequence path: batched prefill, then per-token decode.
/// Returns (tokens, every step's logits) for `n` generated tokens.
fn solo(r: &mut InklingRunner, prompt: &[u32], n: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    r.reset();
    let hs = r.hidden_size();
    let mut batch = Vec::with_capacity(prompt.len() * hs);
    for &t in prompt {
        batch.extend(r.embed_token(t));
    }
    let h = r.forward_layers_batch(batch, 0, prompt.len());
    let mut logits_all = vec![r.head_logits(&h[(prompt.len() - 1) * hs..])];
    let mut toks = vec![argmax(&logits_all[0])];
    let mut pos = prompt.len();
    while toks.len() < n {
        let h = r.embed_token(*toks.last().unwrap());
        let h = r.forward_layers(h, pos, None);
        let l = r.head_logits(&h);
        toks.push(argmax(&l));
        logits_all.push(l);
        pos += 1;
    }
    (toks, logits_all)
}

struct Stream {
    slot: usize,
    toks: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

fn admit(r: &mut InklingRunner, prompt: &[u32]) -> Stream {
    let hs = r.hidden_size();
    let slot = r.open_stream().expect("free slot");
    assert_eq!(r.stream_pos(slot), 0);
    let mut batch = Vec::with_capacity(prompt.len() * hs);
    for &t in prompt {
        batch.extend(r.embed_token(t));
    }
    let h = r.prefill_stream(slot, batch, prompt.len());
    assert_eq!(r.stream_pos(slot), prompt.len());
    let l = r.head_logits(&h[(prompt.len() - 1) * hs..]);
    Stream {
        slot,
        toks: vec![argmax(&l)],
        logits: vec![l],
    }
}

/// One decode step over every listed stream (batched), sampling greedily.
fn step(r: &mut InklingRunner, streams: &mut [&mut Stream]) {
    let hs = r.hidden_size();
    let slots: Vec<usize> = streams.iter().map(|s| s.slot).collect();
    let mut batch = Vec::with_capacity(slots.len() * hs);
    for s in streams.iter() {
        batch.extend(r.embed_token(*s.toks.last().unwrap()));
    }
    let h = r.decode_streams(batch, &slots);
    let logits = r.head_logits_rows(&h, slots.len());
    let vocab = logits.len() / slots.len();
    for (i, s) in streams.iter_mut().enumerate() {
        let l = logits[i * vocab..(i + 1) * vocab].to_vec();
        s.toks.push(argmax(&l));
        s.logits.push(l);
    }
}

#[test]
fn streams_decoded_in_a_batch_match_each_stream_alone() {
    let Some(dir) = fixture() else { return };
    let (p1, want) = reference_prompt(&dir);
    let p2: Vec<u32> = p1.iter().rev().copied().collect();
    let p3: Vec<u32> = p1[3..].iter().map(|&t| (t + 7) % 120).collect();
    let n = 8;

    let mut r = load(&dir);
    let (t1, l1) = solo(&mut r, &p1, n);
    assert_eq!(
        t1, want,
        "classic path must reproduce the HF reference greedy ids"
    );
    let (t2, l2) = solo(&mut r, &p2, n);
    let (t3, l3) = solo(&mut r, &p3, n);

    // Multi-stream: p1 and p2 start together; p3 is admitted after two steps.
    assert_eq!(r.stream_capacity(), 0);
    assert!(r.configure_streams(2));
    assert_eq!(r.stream_capacity(), 2);
    let mut s1 = admit(&mut r, &p1);
    let mut s2 = admit(&mut r, &p2);
    assert!(
        r.open_stream().is_none(),
        "capacity 2 must refuse a third stream"
    );
    for _ in 0..2 {
        step(&mut r, &mut [&mut s1, &mut s2]);
    }
    // grow the pool and admit the third stream mid-flight
    assert!(r.configure_streams(3));
    let mut s3 = admit(&mut r, &p3);
    while s1.toks.len() < n {
        step(&mut r, &mut [&mut s3, &mut s1, &mut s2]);
    }
    while s3.toks.len() < n {
        step(&mut r, &mut [&mut s3]);
    }
    for (name, s, t, l) in [
        ("p1", &s1, &t1, &l1),
        ("p2", &s2, &t2, &l2),
        ("p3", &s3, &t3, &l3),
    ] {
        assert_eq!(
            &s.toks[..n],
            &t[..],
            "{name}: greedy tokens differ in the batch"
        );
        for (i, (a, b)) in s.logits.iter().zip(l.iter()).enumerate() {
            let ab: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
            let bb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
            assert_eq!(ab, bb, "{name}: step {i} logits are not bit-identical");
        }
    }

    // Slot reuse: close p2's slot, admit p1 again into it, decode alongside p3.
    r.close_stream(s2.slot);
    let mut s4 = admit(&mut r, &p1);
    assert_eq!(s4.slot, s2.slot, "a closed slot must be handed out again");
    while s4.toks.len() < n {
        step(&mut r, &mut [&mut s4, &mut s3]);
    }
    assert_eq!(&s4.toks[..n], &t1[..], "reused slot must start clean");
    for (i, (a, b)) in s4.logits.iter().zip(l1.iter()).enumerate() {
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "reused slot: step {i} logits differ"
        );
    }
}

#[test]
fn single_sequence_path_is_unchanged_without_streams() {
    let Some(dir) = fixture() else { return };
    let (p1, want) = reference_prompt(&dir);
    let mut r = load(&dir);
    let (t, _) = solo(&mut r, &p1, 8);
    assert_eq!(t, want);
    // and again after a reset, to show reset() still clears the live slot
    let (t, _) = solo(&mut r, &p1, 8);
    assert_eq!(t, want);
}
