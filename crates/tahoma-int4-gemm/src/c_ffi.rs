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
