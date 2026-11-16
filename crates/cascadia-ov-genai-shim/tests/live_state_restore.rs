//! KV restore into an empty request, on a real stateful language model IR.
//! Needs the `openvino` feature and `CASCADIA_LIVE_STATEFUL_XML` naming an IR
//! that takes `attention_mask`, `position_ids` (and `beam_idx`) plus either
//! `input_ids` (a full export, or stage 0 of a cut tree) or `hidden_states`
//! (a later stage; `CASCADIA_LIVE_HIDDEN` gives its width, default 4096):
//!
//! ```text
//! CASCADIA_LIVE_STATEFUL_XML=/path/to/stage_0/openvino_model.xml \
//! CASCADIA_LIVE_DEVICE=GPU \
//!   cargo test -p cascadia-ov-genai-shim --features openvino \
//!   --test live_state_restore -- --ignored --nocapture --test-threads 1
//! ```
//!
//! On GPU a restore into an empty request (just compiled, reset, or
//! recreated) used to make the next multi-token step fail with "opset1::Add
//! ... Argument shapes are inconsistent"; the shim now primes such a request
//! before restoring. Each case must give the logits of a run that never
//! restored.
#![cfg(feature = "openvino")]

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

const PREFIX: usize = 97;
const SUFFIX: usize = 22;

fn ids(start: usize, n: usize) -> Vec<i64> {
    (start..start + n)
        .map(|i| ((i * 37 + 11) % 1000 + 100) as i64)
        .collect()
}

fn bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One forward of `n` tokens from `start`, returning output 0 as f32.
fn step(rt: &mut Runtime, start: usize, n: usize) -> Vec<f32> {
    let names = rt.input_names().expect("input names");
    if names.iter().any(|n| n == "input_ids") {
        rt.set_input("input_ids", DType::I64, &[1, n], &bytes(&ids(start, n)))
            .expect("input_ids");
    } else {
        let h: usize = std::env::var("CASCADIA_LIVE_HIDDEN")
            .map(|v| v.parse().expect("CASCADIA_LIVE_HIDDEN"))
            .unwrap_or(4096);
        // k/2048 with |k| <= 1001: exact in f16 as well as f32.
        let vals =
            (start * h..(start + n) * h).map(|i| (((i * 7919) % 2003) as f32 - 1001.0) / 2048.0);
        // A cut stage's input may carry `hidden_states` as an alias only.
        let idx = (0..names.len())
            .find(|&i| {
                rt.input_aliases(i)
                    .is_ok_and(|a| a.iter().any(|n| n == "hidden_states"))
            })
            .expect("an input named hidden_states");
        let dtype = rt.input_dtype(idx).expect("hidden_states dtype");
        let rows: Vec<u8> = match dtype {
            DType::F32 => vals.flat_map(f32::to_le_bytes).collect(),
            DType::F16 => vals.flat_map(|x| f32_to_half(x).to_le_bytes()).collect(),
            other => panic!("hidden_states is {other:?}, expected f32 or f16"),
        };
        rt.set_input("hidden_states", dtype, &[1, n, h], &rows)
            .expect("hidden_states");
    }
    rt.set_input(
        "attention_mask",
        DType::I64,
        &[1, start + n],
        &bytes(&vec![1; start + n]),
    )
    .expect("attention_mask");
    let pos: Vec<i64> = (start as i64..(start + n) as i64).collect();
    rt.set_input("position_ids", DType::I64, &[1, n], &bytes(&pos))
        .expect("position_ids");
    if names.iter().any(|n| n == "beam_idx") {
        rt.set_input("beam_idx", DType::I32, &[1], &0i32.to_le_bytes())
            .expect("beam_idx");
    }
    rt.infer().expect("infer");
    let (dtype, _, raw) = rt.output(0).expect("output 0");
    match dtype {
        DType::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::F16 => raw
            .chunks_exact(2)
            .map(|c| half_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        other => panic!("output 0 is {other:?}, expected f32 or f16"),
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// f32 to f16 bits for the zero and normal-range values this test feeds.
fn f32_to_half(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!(
        (1..31).contains(&exp),
        "{x} is outside the f16 normal range"
    );
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

fn check(case: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{case}: output size");
    let peak = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let diff = got
        .iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("{case}: max|diff| {diff} (max|x| {peak})");
    assert!(
        diff <= 1e-2 * peak.max(1.0),
        "{case}: max|diff| {diff} of {peak}"
    );
}

#[test]
#[ignore = "needs CASCADIA_LIVE_STATEFUL_XML and an OpenVINO device"]
fn restore_into_an_empty_request_then_a_multi_token_step() {
    let xml = std::env::var("CASCADIA_LIVE_STATEFUL_XML").expect("CASCADIA_LIVE_STATEFUL_XML");
    let device = std::env::var("CASCADIA_LIVE_DEVICE").unwrap_or_else(|_| "GPU".into());
    // At CPU's default KV precision a restore is close but not exact (a stage
    // 1 diff of 0.32 on 13.6 was measured); with f16 it is exact. Run the CPU
    // control with CASCADIA_LIVE_KV_PRECISION=f16.
    let plugin = match std::env::var("CASCADIA_LIVE_KV_PRECISION") {
        Ok(p) => PluginConfig::new().with("KV_CACHE_PRECISION", p),
        Err(_) => PluginConfig::new(),
    };

    // Reference: prefill the prefix, capture its KV, then the suffix with no restore.
    let mut donor = Runtime::compile(&xml, &device, &plugin).expect("compile donor");
    step(&mut donor, 0, PREFIX);
    let blob = donor.get_state_blob().expect("get_state_blob");
    let want = step(&mut donor, PREFIX, SUFFIX);

    // A separately compiled request that has never run: the cross-chain recipient.
    let mut fresh = Runtime::compile(&xml, &device, &plugin).expect("compile recipient");
    fresh
        .set_state_blob(&blob)
        .expect("restore into a fresh request");
    check("fresh", &step(&mut fresh, PREFIX, SUFFIX), &want);

    // The same request after reset_state, as after the engine's load-time warm-up.
    fresh.reset_state().expect("reset_state");
    fresh
        .set_state_blob(&blob)
        .expect("restore after reset_state");
    check("reset", &step(&mut fresh, PREFIX, SUFFIX), &want);

    // After recreate_request, as after the engine scrubs a failed restore.
    fresh.recreate_request().expect("recreate_request");
    fresh
        .set_state_blob(&blob)
        .expect("restore after recreate_request");
    check("recreated", &step(&mut fresh, PREFIX, SUFFIX), &want);

    // Over live state, which never needed the prime.
    fresh
        .set_state_blob(&blob)
        .expect("restore over live state");
    check("live", &step(&mut fresh, PREFIX, SUFFIX), &want);
}
