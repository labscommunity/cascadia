//! Hand-rolled int4 GEMM for the Kimi K2.6 expert path.
//!
//! Why this exists: the rainier-exported per-expert OV IRs round-trip
//! their int4 weights through OV's CPU plugin one Linear at a time. The
//! per-call inference overhead (~5 ms) dominates for these tiny ~24 MB
//! experts — 480 expert calls × 5 ms = 2.4 s/tok lower bound. A pure
//! Rust kernel with memmap'd packed weights and AVX-512 SIMD is bound by
//! DRAM bandwidth (~140 GB/s on the Xeon Gold 6252 in the miner) and so
//! has a much tighter ceiling.
//!
//! Format: each expert is three int4-packed Linear weight matrices with
//! per-group bf16 scales, group_size=32, zero_point=8 (we apply the
//! standard `unsigned - 8` shift to get signed [-8, 7]). On-disk layout
//! matches the compressed-tensors PackedQuantizationCompressor format:
//!
//!   weight_packed: int32 [out_dim, ceil(in_dim / 8)]
//!   weight_scale:  bf16  [out_dim, in_dim / 32]
//!
//! Each int32 holds 8 nibbles strided across its 32 bits: nibble i at
//! bits [4i, 4i+4]. As little-endian bytes that comes out as
//! `[col0|col1, col2|col3, col4|col5, col6|col7]` (low/high nibble per
//! byte = even/odd column).
//!
//! The `ExpertWeights` struct points at an mmap'd region containing all
//! three projections (gate, up, down) plus their scales for one expert,
//! and [`expert_forward`] runs the full silu(gate*x) ⊙ up*x → down*y
//! pipeline.

pub mod c_ffi;
pub mod format;
pub mod kernel;
pub mod kernel_avx512;
pub mod kernel_avx512_multi;
pub mod kernel_bf16;
pub mod layer0_int4;
pub mod safetensors_source;
pub mod shell;
pub mod shell_int4;

pub use format::{ExpertWeights, GemmError};
pub use kernel::{dequant_gemv_int4, expert_forward};
pub use kernel_avx512::dequant_gemv_int4_auto;
pub use kernel_avx512_multi::dequant_gemm_int4_multi_auto;
pub use safetensors_source::{SafetensorsExpert, SafetensorsExpertSource, SafetensorsLayer0};

// Architecture constants for K2.6 — exposed as constants so the kernel
// and the on-disk format agree.
pub const HIDDEN: usize = 7168;
pub const INTERMEDIATE: usize = 2048;
pub const GROUP_SIZE: usize = 32;
pub const NIBBLES_PER_INT32: usize = 8;
pub const GROUPS_IN: usize = HIDDEN / GROUP_SIZE; // 224
pub const GROUPS_MID: usize = INTERMEDIATE / GROUP_SIZE; // 64
pub const PACKED_COLS_IN: usize = HIDDEN / NIBBLES_PER_INT32; // 896 (i32 cols per row)
pub const PACKED_COLS_MID: usize = INTERMEDIATE / NIBBLES_PER_INT32; // 256
