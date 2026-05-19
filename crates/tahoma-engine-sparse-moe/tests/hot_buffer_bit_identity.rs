//! Bit-identity test for the hot-expert buffer.
//!
//! Build a synthetic safetensors model dir with a handful of fake
//! experts (per-tensor uniform byte stamps so it's trivially
//! falsifiable), instantiate the safetensors source, pack a hot
//! buffer with a subset of the experts, and assert each hot
//! buffer slice matches the source slice byte-for-byte.
//!
//! The kernel itself is deterministic over (gate_packed, gate_scale,
//! up_packed, up_scale, down_packed, down_scale, x) — proving that the
//! six byte slices are bit-identical between the cold (mmap) and hot
//! (packed) paths is equivalent to proving the output `Vec<f32>` will be
//! bit-identical too.
//!
//! Does NOT require the real K2.6 model — generates its own tiny
//! safetensors files at test runtime.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use serde_json::json;
use tahoma_engine_sparse_moe::hot_buffer::LayerHotBuffer;
use tahoma_int4_gemm::SafetensorsExpertSource;
use tahoma_int4_gemm::{GROUP_SIZE, HIDDEN, INTERMEDIATE, PACKED_COLS_IN, PACKED_COLS_MID};

const NUM_TEST_EXPERTS: u32 = 4;
const TEST_LAYER: u32 = 1;

/// Build one expert's six tensors as deterministic byte stamps.
/// The kernel never runs against these — we only check byte-equality
/// between the source slices and the hot buffer slices.
fn synth_expert_bytes(eid: u32) -> Vec<(String, &'static str, Vec<usize>, Vec<u8>)> {
    // Same six tensors the K2.6 exporter writes.
    let gate_packed_n = INTERMEDIATE * PACKED_COLS_IN * 4;
    let gate_scale_n = INTERMEDIATE * (HIDDEN / GROUP_SIZE) * 2;
    let up_packed_n = INTERMEDIATE * PACKED_COLS_IN * 4;
    let up_scale_n = INTERMEDIATE * (HIDDEN / GROUP_SIZE) * 2;
    let down_packed_n = HIDDEN * PACKED_COLS_MID * 4;
    let down_scale_n = HIDDEN * (INTERMEDIATE / GROUP_SIZE) * 2;

    let mut out: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = Vec::with_capacity(6);
    let base = format!("language_model.model.layers.{TEST_LAYER}.mlp.experts.{eid}");
    let mk = |n: usize, stamp: u8| vec![stamp; n];
    let stamp_base = (eid as u8).wrapping_mul(17).wrapping_add(1);
    out.push((
        format!("{base}.gate_proj.weight_packed"),
        "I32",
        vec![INTERMEDIATE, PACKED_COLS_IN],
        mk(gate_packed_n, stamp_base ^ 0x10),
    ));
    out.push((
        format!("{base}.gate_proj.weight_scale"),
        "BF16",
        vec![INTERMEDIATE, HIDDEN / GROUP_SIZE],
        mk(gate_scale_n, stamp_base ^ 0x20),
    ));
    out.push((
        format!("{base}.up_proj.weight_packed"),
        "I32",
        vec![INTERMEDIATE, PACKED_COLS_IN],
        mk(up_packed_n, stamp_base ^ 0x30),
    ));
    out.push((
        format!("{base}.up_proj.weight_scale"),
        "BF16",
        vec![INTERMEDIATE, HIDDEN / GROUP_SIZE],
        mk(up_scale_n, stamp_base ^ 0x40),
    ));
    out.push((
        format!("{base}.down_proj.weight_packed"),
        "I32",
        vec![HIDDEN, PACKED_COLS_MID],
        mk(down_packed_n, stamp_base ^ 0x50),
    ));
    out.push((
        format!("{base}.down_proj.weight_scale"),
        "BF16",
        vec![HIDDEN, INTERMEDIATE / GROUP_SIZE],
        mk(down_scale_n, stamp_base ^ 0x60),
    ));
    out
}

/// Write a minimal safetensors file containing one expert's six
/// tensors. Layout matches safetensors v1: 8-byte LE header length,
/// JSON metadata, then raw tensor data concatenated.
fn write_one_expert_shard(
    dir: &std::path::Path,
    shard_name: &str,
    eid: u32,
) -> Vec<(String, String)> {
    let tensors = synth_expert_bytes(eid);
    // Build the JSON metadata: each tensor gets {dtype, shape,
    // data_offsets: [start, end]}.
    let mut meta = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, dtype, shape, bytes) in &tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        let end = data.len();
        meta.insert(
            name.clone(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end]
            }),
        );
    }
    // Write the shard: 8-byte header_len LE + JSON + data
    let header = serde_json::Value::Object(meta).to_string();
    let path = dir.join(shard_name);
    let mut f = File::create(&path).expect("create shard");
    let len_le = (header.len() as u64).to_le_bytes();
    f.write_all(&len_le).expect("write header_len");
    f.write_all(header.as_bytes()).expect("write header");
    f.write_all(&data).expect("write data");
    f.flush().ok();
    drop(f);

    tensors
        .into_iter()
        .map(|(name, _, _, _)| (name, shard_name.to_string()))
        .collect()
}

fn write_safetensors_index(dir: &std::path::Path, weight_map: &HashMap<String, String>) {
    let idx = json!({
        "metadata": {"total_size": 0u64},
        "weight_map": weight_map,
    });
    let p = dir.join("model.safetensors.index.json");
    std::fs::write(&p, idx.to_string()).expect("write index");
}

#[test]
fn hot_buffer_matches_safetensors_source_byte_for_byte() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir: PathBuf = tmp.path().to_path_buf();

    // Write NUM_TEST_EXPERTS experts each into their own shard so the
    // safetensors source has to load multiple shards. Keeps the test
    // exercise more realistic than one giant shard.
    let mut weight_map: HashMap<String, String> = HashMap::new();
    for eid in 0..NUM_TEST_EXPERTS {
        let shard_name = format!("model-{eid:05}-of-{:05}.safetensors", NUM_TEST_EXPERTS);
        let assoc = write_one_expert_shard(&dir, &shard_name, eid);
        for (name, shard) in assoc {
            weight_map.insert(name, shard);
        }
    }
    write_safetensors_index(&dir, &weight_map);

    let source = SafetensorsExpertSource::open(&dir).expect("open source");

    // Pack a hot buffer with a subset of the experts. The order
    // matters for the packed layout, but slicing by eid should still
    // find each. Use a non-monotonic order to catch any "I assumed
    // sorted" bugs.
    let hot_set: Vec<u32> = vec![2, 0, 3];
    let hb = LayerHotBuffer::build(&source, TEST_LAYER, &hot_set).expect("build hot buffer");

    // Every expert in the hot set must round-trip byte-for-byte.
    for &eid in &hot_set {
        let view = hb.slice(eid).expect("present");
        let src = source.expert(TEST_LAYER, eid).expect("source expert");
        assert_eq!(view.gate_packed, src.gate_packed, "eid {eid} gate_packed");
        assert_eq!(view.gate_scale, src.gate_scale, "eid {eid} gate_scale");
        assert_eq!(view.up_packed, src.up_packed, "eid {eid} up_packed");
        assert_eq!(view.up_scale, src.up_scale, "eid {eid} up_scale");
        assert_eq!(view.down_packed, src.down_packed, "eid {eid} down_packed");
        assert_eq!(view.down_scale, src.down_scale, "eid {eid} down_scale");
    }

    // Expert id NOT in the hot set must miss.
    assert!(hb.slice(1).is_none(), "eid 1 was not packed; must miss");

    // Verify the packed buffer's expected size: per-expert stride
    // × number-of-hot-experts.
    let per_expert = src_per_expert_bytes();
    assert_eq!(hb.bytes(), per_expert * hot_set.len());
}

fn src_per_expert_bytes() -> usize {
    let gate_packed = INTERMEDIATE * PACKED_COLS_IN * 4;
    let gate_scale = INTERMEDIATE * (HIDDEN / GROUP_SIZE) * 2;
    let up_packed = gate_packed;
    let up_scale = gate_scale;
    let down_packed = HIDDEN * PACKED_COLS_MID * 4;
    let down_scale = HIDDEN * (INTERMEDIATE / GROUP_SIZE) * 2;
    gate_packed + gate_scale + up_packed + up_scale + down_packed + down_scale
}
