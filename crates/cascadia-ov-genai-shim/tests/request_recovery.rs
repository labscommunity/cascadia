//! The shim's stage-recovery calls on a real CPU request: the request-state
//! probe, `wait_for`, `abandon_request` and its leak count, the calls that
//! must fail on an abandoned handle, and `recreate_request` after an abandon.
//!
//! ```text
//! cargo test -p cascadia-ov-genai-shim --features openvino --test request_recovery -- --nocapture
//! ```
//!
//! The model is the stateful IR of `prime_seam.rs`: `y = state + x`, and the
//! state takes `y`. This file has one #[test] fn: the leak count and the prime
//! switches are process-global, so parallel test fns would see each other.
#![cfg(feature = "openvino")]

use std::time::Duration;

use cascadia_ov_genai_shim::{leaked_request_count, DType, PluginConfig, RequestState, Runtime};

const IR: &str = r#"<?xml version="1.0"?>
<net name="request_recovery" version="11">
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

fn state(rt: &Runtime) -> RequestState {
    rt.request_state().expect("request_state")
}

#[test]
fn abandon_then_recreate_on_cpu() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("request_recovery");
    std::fs::create_dir_all(&dir).expect("create IR dir");
    let xml = dir.join("request_recovery.xml");
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

    // 1. A fresh request is idle. It never ran, so wait_for has nothing to wait on.
    let mut rt = Runtime::compile(xml, "CPU", &plugin).expect("compile recipient");
    assert_eq!(state(&rt), RequestState::Idle, "fresh request");
    assert!(
        !rt.wait_for(Duration::from_millis(10))
            .expect("wait_for fresh"),
        "wait_for on a request that never ran"
    );
    // A cancel of an idle request does nothing.
    rt.cancel().expect("cancel idle");
    assert_eq!(
        state(&rt),
        RequestState::Idle,
        "after cancel of an idle request"
    );
    // After a run, wait_for reports it complete and the request is idle.
    set_x(&mut rt, &B);
    assert_eq!(infer(&mut rt), B, "first step");
    assert!(
        rt.wait_for(Duration::from_millis(10))
            .expect("wait_for after a run"),
        "wait_for after a run"
    );
    assert_eq!(state(&rt), RequestState::Idle, "after a run");

    // 2. Abandon: no request, and exactly one more leaked request. The step above
    // left the state live, and the abandon does not clear that flag.
    let leaked = leaked_request_count();
    rt.abandon_request().expect("abandon_request");
    assert_eq!(state(&rt), RequestState::None, "after abandon");
    assert_eq!(
        leaked_request_count(),
        leaked + 1,
        "leak count after abandon"
    );

    // 3. The calls that need a request fail at once, with an error.
    rt.set_state_blob(&blob)
        .expect_err("set_state_blob on an abandoned request");
    rt.infer().expect_err("infer on an abandoned request");
    assert_eq!(state(&rt), RequestState::None, "after the failed calls");

    // 4. Recreate after the abandon gives an idle request.
    rt.recreate_request()
        .expect("recreate_request after abandon");
    assert_eq!(state(&rt), RequestState::Idle, "after recreate");
    assert_eq!(leaked_request_count(), leaked + 1, "recreate leaks nothing");
    set_x(&mut rt, &B);

    // 5. The new request is empty, so a restore into it primes again: a forced
    // prime failure fails the restore.
    std::env::set_var(FAIL, "1");
    let err = rt
        .set_state_blob(&blob)
        .expect_err("restore after recreate with a failed prime");
    assert!(err.to_string().contains(PRIME_FAILED), "error: {err}");
    std::env::remove_var(FAIL);
    rt.set_state_blob(&blob).expect("restore after recreate");
    // The restored state A plus the caller's input B.
    assert_eq!(infer(&mut rt), A_PLUS_B, "step after recreate and restore");

    std::env::remove_var(FORCE);
}
