//! C-ABI surface for calling the int4 GEMM from Python via ctypes.
//!
//! Two entry points:
//!
//!   - `tahoma_int4_open_source(model_dir, out_handle)` — opens the
//!     safetensors-direct source. Returns 0 on success.
//!   - `tahoma_int4_expert_forward(handle, layer, expert, x_f32,
//!     out_f32)` — runs one expert on one input vector.
//!
//! No memory ownership crosses the boundary other than the opaque
//! source handle; inputs/outputs are caller-allocated f32 arrays of
//! length HIDDEN = 7168.

#![allow(non_snake_case)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::sync::Arc;

use half::bf16;

use crate::kernel::expert_forward;
use crate::safetensors_source::SafetensorsExpertSource;
use crate::HIDDEN;

#[repr(C)]
pub struct TahomaInt4Source {
    inner: Arc<SafetensorsExpertSource>,
}

/// Open a safetensors-backed expert source. `out_handle` is set on
/// success; caller must free via `tahoma_int4_destroy_source`.
#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_open_source(
    model_dir: *const c_char,
    out_handle: *mut *mut TahomaInt4Source,
) -> c_int {
    if model_dir.is_null() || out_handle.is_null() {
        return -1;
    }
    let s = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let source = match SafetensorsExpertSource::open(s) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    let boxed = Box::new(TahomaInt4Source {
        inner: Arc::new(source),
    });
    *out_handle = Box::into_raw(boxed);
    0
}

#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_destroy_source(handle: *mut TahomaInt4Source) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle));
}

/// Force one expert's pages into the OS page cache by touching every
/// page of its six tensors. Useful for prewarming before timing /
/// generation, when the cold mmap page-in cost (~25 ms/expert) would
/// otherwise dominate the inner loop.
///
/// Returns 0 on success, or `n` (number of touched bytes consumed) for
/// caller inspection. Single u8 read per 4 KB page is enough.
#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_prewarm(
    handle: *mut TahomaInt4Source,
    layer: u32,
    expert: u32,
) -> c_int {
    if handle.is_null() {
        return -1;
    }
    let src = &(*handle).inner;
    let w = match src.expert(layer, expert) {
        Ok(w) => w,
        Err(_) => return -2,
    };
    // Volatile-read one byte every 4 KB to force page-in. Avoid being
    // optimized out by writing into a static sink.
    let mut sink: u64 = 0;
    for slice in [
        w.gate_packed,
        w.gate_scale,
        w.up_packed,
        w.up_scale,
        w.down_packed,
        w.down_scale,
    ] {
        let mut i = 0usize;
        while i < slice.len() {
            sink = sink.wrapping_add(std::ptr::read_volatile(&slice[i]) as u64);
            i += 4096;
        }
    }
    SINK.store(sink, std::sync::atomic::Ordering::Relaxed);
    0
}

static SINK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ---- Shell forward C-FFI ---------------------------------------------------

use crate::safetensors_source::SafetensorsShell;
use crate::shell::{
    shell_forward_decode, ShellOutputs, HIDDEN as SHELL_HIDDEN, NUM_HEADS, QK_HEAD_DIM, TOPK,
    V_HEAD_DIM,
};

/// Holds one layer's shell weights pinned to its safetensors mmaps.
#[repr(C)]
pub struct TahomaShell {
    inner: SafetensorsShell,
}

/// Same data, but quantized to int4 + bf16 scales and heap-resident.
/// Smaller working set (~77 MB/shell vs 295 MB), survives page cache
/// eviction pressure.
#[repr(C)]
pub struct TahomaShellInt4 {
    inner: crate::shell_int4::Int4Shell,
}

/// Load shell weights for layer `layer` from the source. Caller owns
/// the handle and must free via `tahoma_int4_destroy_shell`.
#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_open_shell(
    handle: *mut TahomaInt4Source,
    layer: u32,
    out: *mut *mut TahomaShell,
) -> c_int {
    if handle.is_null() || out.is_null() {
        return -1;
    }
    let src = &(*handle).inner;
    match src.shell(layer) {
        Ok(s) => {
            let boxed = Box::new(TahomaShell { inner: s });
            *out = Box::into_raw(boxed);
            0
        }
        Err(_) => -2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_destroy_shell(h: *mut TahomaShell) {
    if h.is_null() {
        return;
    }
    drop(Box::from_raw(h));
}

/// Run one shell forward (decode, seq=1). Caller provides:
///   x_f32:      [HIDDEN] f32 layer input
///   past_k:     [NUM_HEADS * past_seq_len * QK_HEAD_DIM] f32
///   past_v:     [NUM_HEADS * past_seq_len * V_HEAD_DIM] f32
///   past_seq_len: usize
///
/// And output buffers:
///   attn_out_post_norm: [HIDDEN] f32
///   attn_residual:      [HIDDEN] f32
///   shared_expert_out:  [HIDDEN] f32
///   routing_ids:        [TOPK=8] i64
///   routing_weights:    [TOPK=8] f32
///   present_k:          [NUM_HEADS * QK_HEAD_DIM] f32 (seq=1 only)
///   present_v:          [NUM_HEADS * V_HEAD_DIM] f32  (seq=1 only)
///
/// Returns 0 on success.
/// Build an int4-quantized variant of one shell from the safetensors
/// (read once, quantize on the fly, store in heap-resident buffers).
#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_open_shell_int4(
    handle: *mut TahomaInt4Source,
    layer: u32,
    out: *mut *mut TahomaShellInt4,
) -> c_int {
    if handle.is_null() || out.is_null() {
        return -1;
    }
    let src = &(*handle).inner;
    match src.shell(layer) {
        Ok(s) => {
            let q = crate::shell_int4::Int4Shell::from_safetensors(&s);
            let boxed = Box::new(TahomaShellInt4 { inner: q });
            *out = Box::into_raw(boxed);
            0
        }
        Err(_) => -2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_destroy_shell_int4(h: *mut TahomaShellInt4) {
    if h.is_null() {
        return;
    }
    drop(Box::from_raw(h));
}

/// Run int4 shell forward.
#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_shell_forward_int4(
    shell: *mut TahomaShellInt4,
    x_f32: *const f32,
    past_k: *const f32,
    past_v: *const f32,
    past_seq_len: usize,
    out_post_norm: *mut f32,
    out_residual: *mut f32,
    out_shared: *mut f32,
    out_ids: *mut i64,
    out_weights: *mut f32,
    out_present_k: *mut f32,
    out_present_v: *mut f32,
) -> c_int {
    if shell.is_null() {
        return -1;
    }
    let s = &(*shell).inner;
    let x = std::slice::from_raw_parts(x_f32, SHELL_HIDDEN);
    let pk = std::slice::from_raw_parts(past_k, NUM_HEADS * past_seq_len * QK_HEAD_DIM);
    let pv = std::slice::from_raw_parts(past_v, NUM_HEADS * past_seq_len * V_HEAD_DIM);
    let ShellOutputs {
        attn_out_post_norm,
        attn_residual,
        shared_expert_out,
        routing_ids,
        routing_weights,
        present_k,
        present_v,
    } = crate::shell_int4::shell_forward_decode_int4(s, x, pk, pv, past_seq_len);
    std::slice::from_raw_parts_mut(out_post_norm, SHELL_HIDDEN)
        .copy_from_slice(&attn_out_post_norm);
    std::slice::from_raw_parts_mut(out_residual, SHELL_HIDDEN).copy_from_slice(&attn_residual);
    std::slice::from_raw_parts_mut(out_shared, SHELL_HIDDEN).copy_from_slice(&shared_expert_out);
    std::slice::from_raw_parts_mut(out_ids, TOPK).copy_from_slice(&routing_ids);
    std::slice::from_raw_parts_mut(out_weights, TOPK).copy_from_slice(&routing_weights);
    std::slice::from_raw_parts_mut(out_present_k, NUM_HEADS * QK_HEAD_DIM)
        .copy_from_slice(&present_k);
    std::slice::from_raw_parts_mut(out_present_v, NUM_HEADS * V_HEAD_DIM)
        .copy_from_slice(&present_v);
    0
}

#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_shell_forward(
    shell: *mut TahomaShell,
    x_f32: *const f32,
    past_k: *const f32,
    past_v: *const f32,
    past_seq_len: usize,
    out_post_norm: *mut f32,
    out_residual: *mut f32,
    out_shared: *mut f32,
    out_ids: *mut i64,
    out_weights: *mut f32,
    out_present_k: *mut f32,
    out_present_v: *mut f32,
) -> c_int {
    if shell.is_null() {
        return -1;
    }
    let s = &(*shell).inner;
    let x = std::slice::from_raw_parts(x_f32, SHELL_HIDDEN);
    let pk = std::slice::from_raw_parts(past_k, NUM_HEADS * past_seq_len * QK_HEAD_DIM);
    let pv = std::slice::from_raw_parts(past_v, NUM_HEADS * past_seq_len * V_HEAD_DIM);
    let ShellOutputs {
        attn_out_post_norm,
        attn_residual,
        shared_expert_out,
        routing_ids,
        routing_weights,
        present_k,
        present_v,
    } = shell_forward_decode(s, x, pk, pv, past_seq_len);

    std::slice::from_raw_parts_mut(out_post_norm, SHELL_HIDDEN)
        .copy_from_slice(&attn_out_post_norm);
    std::slice::from_raw_parts_mut(out_residual, SHELL_HIDDEN).copy_from_slice(&attn_residual);
    std::slice::from_raw_parts_mut(out_shared, SHELL_HIDDEN).copy_from_slice(&shared_expert_out);
    std::slice::from_raw_parts_mut(out_ids, TOPK).copy_from_slice(&routing_ids);
    std::slice::from_raw_parts_mut(out_weights, TOPK).copy_from_slice(&routing_weights);
    std::slice::from_raw_parts_mut(out_present_k, NUM_HEADS * QK_HEAD_DIM)
        .copy_from_slice(&present_k);
    std::slice::from_raw_parts_mut(out_present_v, NUM_HEADS * V_HEAD_DIM)
        .copy_from_slice(&present_v);
    0
}

/// Run one expert. Caller provides x_f32 (HIDDEN floats) and an output
/// buffer (HIDDEN floats). Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn tahoma_int4_expert_forward(
    handle: *mut TahomaInt4Source,
    layer: u32,
    expert: u32,
    x_f32: *const f32,
    out_f32: *mut f32,
) -> c_int {
    if handle.is_null() || x_f32.is_null() || out_f32.is_null() {
        return -1;
    }
    let src = &(*handle).inner;
    let w = match src.expert(layer, expert) {
        Ok(w) => w,
        Err(_) => return -2,
    };
    let x_slice = std::slice::from_raw_parts(x_f32, HIDDEN);
    let x_bf16: Vec<bf16> = x_slice.iter().map(|v| bf16::from_f32(*v)).collect();
    let mut out_bf16 = vec![bf16::ZERO; HIDDEN];
    expert_forward(
        &x_bf16,
        w.gate_packed,
        w.gate_scale,
        w.up_packed,
        w.up_scale,
        w.down_packed,
        w.down_scale,
        &mut out_bf16,
    );
    let out_slice = std::slice::from_raw_parts_mut(out_f32, HIDDEN);
    for i in 0..HIDDEN {
        out_slice[i] = out_bf16[i].to_f32();
    }
    0
}
