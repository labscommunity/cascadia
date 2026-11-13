//! WDDM video-memory budget (Windows): what the display driver stack will actually let a process
//! keep resident on a GPU, as opposed to what OpenVINO reports.
//!
//! On an Intel iGPU the OpenVINO GPU plugin's `GPU_DEVICE_TOTAL_MEM_SIZE` follows the Intel
//! user-mode driver's GMM pinning limit
//! (`HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-…}\0000\GMM\GpuSystemMemoryPinninglimit`),
//! but residency is enforced by dxgkrnl's video memory manager, whose shared segment and
//! per-process budget follow
//! `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager\SystemPartitionCommitLimitPercentage`
//! (Intel's INF installs 57; the Intel Graphics Software "Shared GPU Memory Override" slider
//! writes both). A 32 GB box can therefore report a 24.8 GiB OpenVINO total while only 17.2 GiB
//! can be resident, and a stage sized to the OpenVINO number pages — the 17x decode cliff
//! measured on 32 GB Panther Lake boxes; 85 is the measured setting there. `cascadia doctor`
//! prints both numbers and `profile-stages` caps the GPU budget at the WDDM one.
//!
//! The query is the one Task Manager uses, straight from gdi32 (no SDK crate):
//! `D3DKMTEnumAdapters2` for the adapter handles, `D3DKMTQueryAdapterInfo` with
//! `KMTQAITYPE_GETSEGMENTSIZE` for the segment sizes (dxdiag's numbers) and `KMTQAITYPE_ADAPTERTYPE`
//! to skip software adapters, and `D3DKMTQueryVideoMemoryInfo` (LOCAL segment group, calling
//! process) for the budget. Struct layouts follow `d3dkmthk.h` and are pinned by the unit tests,
//! which compile on every platform; only the calls are Windows.

use std::ffi::c_void;

// ---- d3dkmthk.h layouts (every field 4- or 8-byte aligned; no packing pragmas) ----
//
// Only constructed by the Windows query path; the definitions stay unconditional so the layout
// tests run on the Mac/Linux CI too.

/// `LUID` — the adapter identity Windows hands out.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(windows), allow(dead_code))]
struct Luid {
    low_part: u32,
    high_part: i32,
}

/// `D3DKMT_ADAPTERINFO`: one entry of the `D3DKMTEnumAdapters2` array.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(windows), allow(dead_code))]
struct AdapterInfo {
    h_adapter: u32,
    adapter_luid: Luid,
    num_of_sources: u32,
    /// `BOOL bPrecisePresentRegionsPreferred`.
    precise_present_regions_preferred: i32,
}

/// `D3DKMT_ENUMADAPTERS2`: `p_adapters` null ⇒ the call only fills `num_adapters`.
#[repr(C)]
#[cfg_attr(not(windows), allow(dead_code))]
struct EnumAdapters2 {
    num_adapters: u32,
    p_adapters: *mut AdapterInfo,
}

/// `D3DKMT_SEGMENTSIZEINFO` (`KMTQAITYPE_GETSEGMENTSIZE`): the numbers dxdiag shows.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(windows), allow(dead_code))]
struct SegmentSizeInfo {
    dedicated_video_memory_size: u64,
    dedicated_system_memory_size: u64,
    shared_system_memory_size: u64,
}

/// `D3DKMT_QUERYADAPTERINFO`: `kind` is a `KMTQUERYADAPTERINFOTYPE`, the data pointer/size name
/// the caller's output buffer.
#[repr(C)]
#[cfg_attr(not(windows), allow(dead_code))]
struct QueryAdapterInfo {
    h_adapter: u32,
    kind: u32,
    p_private_driver_data: *mut c_void,
    private_driver_data_size: u32,
}

/// `D3DKMT_QUERYVIDEOMEMORYINFO`: `h_process` null ⇒ the calling process;
/// `memory_segment_group` 0 = LOCAL (the segment an iGPU's weights live in), 1 = NON_LOCAL.
#[repr(C)]
#[cfg_attr(not(windows), allow(dead_code))]
struct QueryVideoMemoryInfo {
    h_process: *mut c_void,
    h_adapter: u32,
    memory_segment_group: u32,
    budget: u64,
    current_usage: u64,
    available_for_reservation: u64,
    current_reservation: u64,
    physical_adapter_index: u32,
}

/// `D3DKMT_CLOSEADAPTER`.
#[repr(C)]
#[cfg_attr(not(windows), allow(dead_code))]
struct CloseAdapter {
    h_adapter: u32,
}

/// `KMTQUERYADAPTERINFOTYPE` values used here.
#[cfg_attr(not(windows), allow(dead_code))]
const KMTQAITYPE_GETSEGMENTSIZE: u32 = 3;
#[cfg_attr(not(windows), allow(dead_code))]
const KMTQAITYPE_ADAPTERTYPE: u32 = 15;
/// `D3DKMT_ADAPTERTYPE.SoftwareDevice` (bit 2; RenderSupported = 1, DisplaySupported = 2).
#[cfg_attr(not(windows), allow(dead_code))]
const ADAPTER_TYPE_SOFTWARE_DEVICE: u32 = 0x4;
/// `D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL`.
#[cfg_attr(not(windows), allow(dead_code))]
const MEMORY_SEGMENT_GROUP_LOCAL: u32 = 0;

#[cfg(windows)]
#[link(name = "gdi32")]
#[allow(non_snake_case)]
extern "system" {
    // All four return NTSTATUS (0 = STATUS_SUCCESS).
    fn D3DKMTEnumAdapters2(p: *mut EnumAdapters2) -> i32;
    fn D3DKMTQueryAdapterInfo(p: *mut QueryAdapterInfo) -> i32;
    fn D3DKMTQueryVideoMemoryInfo(p: *mut QueryVideoMemoryInfo) -> i32;
    fn D3DKMTCloseAdapter(p: *mut CloseAdapter) -> i32;
}

/// One hardware display adapter's WDDM memory picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBudget {
    /// Adapter LUID as `hhhhhhhh-llllllll`, for telling two adapters apart in a report.
    pub luid: String,
    /// Shared system memory segment (what dxdiag calls "Shared Memory").
    pub shared_system_bytes: u64,
    /// Dedicated video memory (0 on an iGPU, VRAM on a dGPU).
    pub dedicated_video_bytes: u64,
    /// Per-process budget of the LOCAL segment group for the calling process: the real
    /// residency cap. 0 when the driver did not report one.
    pub budget_bytes: u64,
    /// The calling process' current LOCAL usage.
    pub current_usage_bytes: u64,
}

/// Every hardware (non-software) display adapter's segment sizes and the calling process'
/// LOCAL budget on it. `Err` only when the enumeration itself fails; an adapter whose budget
/// query fails is skipped.
#[cfg(windows)]
pub fn query_gpu_budgets() -> Result<Vec<GpuBudget>, String> {
    let status = |st: i32| format!("NTSTATUS {:#010x}", st as u32);
    // SAFETY: plain C structs handed to gdi32 by pointer; every buffer outlives its call and is
    // sized by the matching `*_size` field, and the adapter handles are closed before return.
    unsafe {
        let mut e = EnumAdapters2 {
            num_adapters: 0,
            p_adapters: std::ptr::null_mut(),
        };
        let st = D3DKMTEnumAdapters2(&mut e);
        if st != 0 {
            return Err(format!("D3DKMTEnumAdapters2 (count): {}", status(st)));
        }
        if e.num_adapters == 0 {
            return Ok(Vec::new());
        }
        let mut adapters = vec![AdapterInfo::default(); e.num_adapters as usize];
        e.p_adapters = adapters.as_mut_ptr();
        let st = D3DKMTEnumAdapters2(&mut e);
        if st != 0 {
            return Err(format!("D3DKMTEnumAdapters2: {}", status(st)));
        }
        adapters.truncate(e.num_adapters as usize);

        let mut out = Vec::with_capacity(adapters.len());
        for a in &adapters {
            // Skip the Microsoft Basic Render Driver and friends: their "budget" is not a GPU's.
            let mut type_flags: u32 = 0;
            let mut q = QueryAdapterInfo {
                h_adapter: a.h_adapter,
                kind: KMTQAITYPE_ADAPTERTYPE,
                p_private_driver_data: (&mut type_flags as *mut u32).cast::<c_void>(),
                private_driver_data_size: std::mem::size_of::<u32>() as u32,
            };
            let software = D3DKMTQueryAdapterInfo(&mut q) == 0
                && type_flags & ADAPTER_TYPE_SOFTWARE_DEVICE != 0;
            if !software {
                let mut seg = SegmentSizeInfo::default();
                let mut q = QueryAdapterInfo {
                    h_adapter: a.h_adapter,
                    kind: KMTQAITYPE_GETSEGMENTSIZE,
                    p_private_driver_data: (&mut seg as *mut SegmentSizeInfo).cast::<c_void>(),
                    private_driver_data_size: std::mem::size_of::<SegmentSizeInfo>() as u32,
                };
                if D3DKMTQueryAdapterInfo(&mut q) != 0 {
                    seg = SegmentSizeInfo::default();
                }
                let mut v = QueryVideoMemoryInfo {
                    h_process: std::ptr::null_mut(),
                    h_adapter: a.h_adapter,
                    memory_segment_group: MEMORY_SEGMENT_GROUP_LOCAL,
                    budget: 0,
                    current_usage: 0,
                    available_for_reservation: 0,
                    current_reservation: 0,
                    physical_adapter_index: 0,
                };
                if D3DKMTQueryVideoMemoryInfo(&mut v) == 0 {
                    out.push(GpuBudget {
                        luid: format!(
                            "{:08x}-{:08x}",
                            a.adapter_luid.high_part as u32, a.adapter_luid.low_part
                        ),
                        shared_system_bytes: seg.shared_system_memory_size,
                        dedicated_video_bytes: seg.dedicated_video_memory_size,
                        budget_bytes: v.budget,
                        current_usage_bytes: v.current_usage,
                    });
                }
            }
            let mut c = CloseAdapter {
                h_adapter: a.h_adapter,
            };
            let _ = D3DKMTCloseAdapter(&mut c);
        }
        Ok(out)
    }
}

/// The largest reported per-process budget across adapters (an AI PC has one iGPU; with a dGPU
/// beside it the bigger budget is the one a large stage would be placed under). `None` when no
/// adapter reported a non-zero budget.
pub fn largest_budget(budgets: &[GpuBudget]) -> Option<u64> {
    budgets
        .iter()
        .map(|b| b.budget_bytes)
        .filter(|&b| b > 0)
        .max()
}

/// Placement budget for a GPU: OpenVINO's `GPU_DEVICE_TOTAL_MEM_SIZE` capped at the WDDM
/// per-process budget when one is known (0 / `None` = unknown ⇒ the OpenVINO number stands).
pub fn gpu_budget_cap(ov_total: u64, wddm_budget: Option<u64>) -> u64 {
    match wddm_budget {
        Some(b) if b > 0 => ov_total.min(b),
        _ => ov_total,
    }
}

/// Bytes as GiB, for the report lines.
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// `LUID` and `D3DKMT_ADAPTERINFO` are all 4-byte fields: 8 and 20 bytes, no padding.
    #[test]
    fn luid_and_adapter_info_match_d3dkmthk() {
        assert_eq!(size_of::<Luid>(), 8);
        assert_eq!(offset_of!(Luid, high_part), 4);
        assert_eq!(size_of::<AdapterInfo>(), 20);
        assert_eq!(offset_of!(AdapterInfo, adapter_luid), 4);
        assert_eq!(offset_of!(AdapterInfo, num_of_sources), 12);
        assert_eq!(
            offset_of!(AdapterInfo, precise_present_regions_preferred),
            16
        );
        assert_eq!(size_of::<CloseAdapter>(), 4);
    }

    /// Three `UINT64`s, in dxdiag's order.
    #[test]
    fn segment_size_info_matches_d3dkmthk() {
        assert_eq!(size_of::<SegmentSizeInfo>(), 24);
        assert_eq!(offset_of!(SegmentSizeInfo, dedicated_video_memory_size), 0);
        assert_eq!(offset_of!(SegmentSizeInfo, dedicated_system_memory_size), 8);
        assert_eq!(offset_of!(SegmentSizeInfo, shared_system_memory_size), 16);
    }

    /// The pointer-bearing structs on the 64-bit ABI the Windows builds use (x64; the padding
    /// after `num_adapters` / before the data pointer / after the last `UINT` is what a wrong
    /// field type would shift).
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn pointer_bearing_structs_match_the_x64_layout() {
        assert_eq!(size_of::<EnumAdapters2>(), 16);
        assert_eq!(offset_of!(EnumAdapters2, p_adapters), 8);

        assert_eq!(size_of::<QueryAdapterInfo>(), 24);
        assert_eq!(offset_of!(QueryAdapterInfo, kind), 4);
        assert_eq!(offset_of!(QueryAdapterInfo, p_private_driver_data), 8);
        assert_eq!(offset_of!(QueryAdapterInfo, private_driver_data_size), 16);

        assert_eq!(size_of::<QueryVideoMemoryInfo>(), 56);
        assert_eq!(offset_of!(QueryVideoMemoryInfo, h_adapter), 8);
        assert_eq!(offset_of!(QueryVideoMemoryInfo, memory_segment_group), 12);
        assert_eq!(offset_of!(QueryVideoMemoryInfo, budget), 16);
        assert_eq!(offset_of!(QueryVideoMemoryInfo, current_usage), 24);
        assert_eq!(
            offset_of!(QueryVideoMemoryInfo, available_for_reservation),
            32
        );
        assert_eq!(offset_of!(QueryVideoMemoryInfo, current_reservation), 40);
        assert_eq!(offset_of!(QueryVideoMemoryInfo, physical_adapter_index), 48);
    }

    fn budget(bytes: u64) -> GpuBudget {
        GpuBudget {
            luid: "0-1".into(),
            shared_system_bytes: 0,
            dedicated_video_bytes: 0,
            budget_bytes: bytes,
            current_usage_bytes: 0,
        }
    }

    /// The measured 32 GB Panther Lake numbers: a 24.77 GiB OpenVINO total against a 17.23 GiB
    /// budget is capped; after the registry change (26.06 GiB budget) the OpenVINO total stands.
    #[test]
    fn cap_takes_the_smaller_of_ov_total_and_budget() {
        let gib = |g: f64| (g * (1u64 << 30) as f64) as u64;
        assert_eq!(gpu_budget_cap(gib(24.77), Some(gib(17.23))), gib(17.23));
        assert_eq!(gpu_budget_cap(gib(24.77), Some(gib(26.06))), gib(24.77));
        // Unknown budget (query failed / zero) leaves the OpenVINO number alone.
        assert_eq!(gpu_budget_cap(gib(24.77), None), gib(24.77));
        assert_eq!(gpu_budget_cap(gib(24.77), Some(0)), gib(24.77));
    }

    #[test]
    fn largest_budget_skips_zero_and_picks_the_max() {
        assert_eq!(largest_budget(&[]), None);
        assert_eq!(largest_budget(&[budget(0)]), None);
        assert_eq!(largest_budget(&[budget(0), budget(5), budget(3)]), Some(5));
        assert!((gib(1 << 30) - 1.0).abs() < 1e-12);
    }
}
