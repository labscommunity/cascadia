//! The prime that set_state_blob runs before a restore into an empty request,
//! exercised on CPU through the shim's test-only switches:
//! `CASCADIA_SHIM_FORCE_PRIME=1` runs the prime off GPU, and
//! `CASCADIA_SHIM_FAIL_PRIME=1` makes the prime throw before its inference.
//!
//! ```text
//! cargo test -p cascadia-ov-genai-shim --features openvino --test prime_seam -- --nocapture
//! ```
//!
//! The model is a stateful IR this test writes itself: `y = state + x`, and
//! the state takes `y`. Its ReadValue starts from `x - x`, so it has no
//! constants and needs no `.bin` file.
#![cfg(feature = "openvino")]

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

const IR: &str = r#"<?xml version="1.0"?>
<net name="prime_seam" version="11">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data shape="1,4" element_type="f32" />
			<output>
				<port id="0" precision="FP32" names="x">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="zeros" type="Subtract" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="state_read" type="ReadValue" version="opset6">
			<data variable_id="state" variable_type="f32" variable_shape="1,4" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sum" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="y">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="state_write" type="Assign" version="opset6">
			<data variable_id="state" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
		</layer>
		<layer id="5" name="y/result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
		<edge from-layer="0" from-port="0" to-layer="1" to-port="1" />
		<edge from-layer="1" from-port="2" to-layer="2" to-port="0" />
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
		<edge from-layer="0" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="3" from-port="2" to-layer="4" to-port="0" />
		<edge from-layer="3" from-port="2" to-layer="5" to-port="0" />
	</edges>
</net>
"#;

const FORCE: &str = "CASCADIA_SHIM_FORCE_PRIME";
const FAIL: &str = "CASCADIA_SHIM_FAIL_PRIME";
const PRIME_FAILED: &str = "GPU prime of an empty request failed";

// Small integers: exact in f32, f16 and bf16, whatever precision the CPU plugin picks.
const A: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const B: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
const A_PLUS_B: [f32; 4] = [11.0, 22.0, 33.0, 44.0];

fn set_x(rt: &mut Runtime, x: &[f32; 4]) {
    let raw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    rt.set_input("x", DType::F32, &[1, 4], &raw).expect("set x");
}

/// One inference with the inputs the request holds now; returns `y`.
fn infer(rt: &mut Runtime) -> Vec<f32> {
    rt.infer().expect("infer");
    let (dtype, shape, raw) = rt.output(0).expect("output 0");
    assert_eq!(dtype, DType::F32, "output dtype");
    assert_eq!(shape, [1, 4], "output shape");
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// The switches are process-global environment variables, so every case runs
// in this one #[test] fn, in a fixed order. Separate test fns would run on
// parallel threads and see each other's settings.
#[test]
fn prime_runs_fails_and_retries_on_cpu() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("prime_seam");
    std::fs::create_dir_all(&dir).expect("create IR dir");
    let xml = dir.join("prime_seam.xml");
    std::fs::write(&xml, IR).expect("write IR");
    let xml = xml.to_str().expect("utf-8 IR path");
    let plugin = PluginConfig::new();

    std::env::set_var(FORCE, "1");
    std::env::remove_var(FAIL);

    // Donor: one step from an empty state leaves state = A.
    let mut donor = Runtime::compile(xml, "CPU", &plugin).expect("compile donor");
    set_x(&mut donor, &A);
    assert_eq!(infer(&mut donor), A, "donor step");
    let blob = donor.get_state_blob().expect("get_state_blob");

    // A fresh request with a caller input already set.
    let mut rt = Runtime::compile(xml, "CPU", &plugin).expect("compile recipient");
    set_x(&mut rt, &B);

    // A failed prime fails the restore with the prime's error.
    std::env::set_var(FAIL, "1");
    let err = rt
        .set_state_blob(&blob)
        .expect_err("restore with a failed prime");
    assert!(err.to_string().contains(PRIME_FAILED), "first error: {err}");
    // The request stays empty, so the next restore runs the prime again.
    let err = rt
        .set_state_blob(&blob)
        .expect_err("second restore with a failed prime");
    assert!(
        err.to_string().contains(PRIME_FAILED),
        "second error: {err}"
    );
    // The caller's input is back: the empty state gives y = 0 + B, not 0 + 0.
    assert_eq!(infer(&mut rt), B, "caller input after a failed prime");

    // An empty request again: the prime runs, the restore succeeds and the
    // caller's input is back, so y = A + B.
    rt.recreate_request().expect("recreate_request");
    set_x(&mut rt, &B);
    std::env::remove_var(FAIL);
    rt.set_state_blob(&blob).expect("restore after a prime");
    assert_eq!(infer(&mut rt), A_PLUS_B, "step after a primed restore");

    // Over live state there is no prime, so a forced failure does not fire.
    std::env::set_var(FAIL, "1");
    rt.set_state_blob(&blob).expect("restore over live state");
    assert_eq!(infer(&mut rt), A_PLUS_B, "step after a live restore");

    std::env::remove_var(FAIL);
    std::env::remove_var(FORCE);
}
