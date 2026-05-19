//! Measurement bin for the lazy-expert-load investigation
//! (perf/lazy-expert-load-080).
//!
//! Question: is OS-level mmap already free for our 384-experts × 60-layers
//! K2.6 layout, or does the metadata cost (VMA tracking, page-table state)
//! actually matter at startup?
//!
//! What this bin does:
//!
//! 1.  Opens `SafetensorsExpertSource` against a K2.6 model dir. Reports
//!     RSS / VmSize / shards mapped before any tensor is touched.
//! 2.  In `--mode shells` (default), walks every shell + dense layer 0 +
//!     embed_tokens — i.e. the load path the Runner takes today. Reports
//!     RSS + bytes mapped + elapsed time after each stage.
//! 3.  In `--mode all-experts`, additionally walks every (layer, expert)
//!     and pins its six tensor slices. This is the *worst-case* mmap
//!     footprint — what we would pay if every expert in every layer
//!     fired on a single workload.
//! 4.  In `--mode all-tensors`, walks every tensor in the safetensors
//!     index (~quarter-million for K2.6: shells + experts + heads). This
//!     forces an mmap of every shard that contains any model weight.
//! 5.  In `--mode populate`, additionally reads one byte from every page
//!     of every mmap'd region. This forces real page-faults so we can
//!     separate "VMA bookkeeping is cheap" from "actually touching the
//!     memory is cheap". (Slow — minutes — but it's how you tell the
//!     difference between virtual and resident memory.)
//!
//! Output: tab-separated table of (stage, secs, shards, vma_mb, rss_mb,
//! vmsize_mb). The caller decides if the numbers justify shipping the
//! per-expert lazy mode in `(A)`. See the source-level note on
//! `SafetensorsExpertSource::lazy_load` for the conclusion.
//!
//! Usage::
//!
//!   mmap_profile --model-dir <path> [--mode shells|all-experts|all-tensors|populate]
//!                                   [--layers <N>] [--experts <N>]

use std::path::PathBuf;
use std::time::Instant;

use tahoma_int4_gemm::{OpenOptions, SafetensorsExpertSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Shells,
    AllExperts,
    AllTensors,
    Populate,
}

fn parse_args() -> (PathBuf, Mode, u32, u32, bool) {
    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut mode = Mode::Shells;
    let mut layers: u32 = 60;
    let mut experts: u32 = 384;
    let mut lazy = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--mode" => {
                mode = match args.next().as_deref() {
                    Some("shells") => Mode::Shells,
                    Some("all-experts") => Mode::AllExperts,
                    Some("all-tensors") => Mode::AllTensors,
                    Some("populate") => Mode::Populate,
                    other => panic!("unknown --mode {:?}", other),
                };
            }
            "--layers" => layers = args.next().and_then(|s| s.parse().ok()).unwrap_or(60),
            "--experts" => experts = args.next().and_then(|s| s.parse().ok()).unwrap_or(384),
            "--lazy" => lazy = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    (
        model_dir.expect("--model-dir required"),
        mode,
        layers,
        experts,
        lazy,
    )
}

/// (rss_mb, vmsize_mb). On platforms we don't know how to query, both
/// fields are 0.
#[cfg(target_os = "linux")]
fn mem_mb() -> (f64, f64) {
    // /proc/self/status is a few KB; cheap to re-read each step.
    let s = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return (0.0, 0.0),
    };
    let mut rss_kb: u64 = 0;
    let mut vm_kb: u64 = 0;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kb = rest
                .trim()
                .trim_end_matches(" kB")
                .split_whitespace()
                .next()
                .and_then(|t| t.parse().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmSize:") {
            vm_kb = rest
                .trim()
                .trim_end_matches(" kB")
                .split_whitespace()
                .next()
                .and_then(|t| t.parse().ok())
                .unwrap_or(0);
        }
    }
    (rss_kb as f64 / 1024.0, vm_kb as f64 / 1024.0)
}

#[cfg(target_os = "macos")]
fn mem_mb() -> (f64, f64) {
    // No /proc on macOS. `ps -o rss=,vsz= -p <pid>` is the portable
    // path and avoids a Mach FFI dependency. Returns KB on Darwin.
    let pid = std::process::id().to_string();
    let out = match std::process::Command::new("ps")
        .args(["-o", "rss=,vsz=", "-p", &pid])
        .output()
    {
        Ok(o) => o,
        Err(_) => return (0.0, 0.0),
    };
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace();
    let rss_kb: u64 = it.next().and_then(|t| t.parse().ok()).unwrap_or(0);
    let vsz_kb: u64 = it.next().and_then(|t| t.parse().ok()).unwrap_or(0);
    (rss_kb as f64 / 1024.0, vsz_kb as f64 / 1024.0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn mem_mb() -> (f64, f64) {
    (0.0, 0.0)
}

fn print_header() {
    println!("stage\tsecs\tshards\tvma_mb\trss_mb\tvmsize_mb");
}

fn print_row(stage: &str, t0: Instant, src: &SafetensorsExpertSource) {
    let (rss, vmsize) = mem_mb();
    let vma_mb = src.shard_bytes_mapped() as f64 / (1024.0 * 1024.0);
    println!(
        "{stage}\t{:.3}\t{}\t{:.1}\t{:.1}\t{:.1}",
        t0.elapsed().as_secs_f64(),
        src.shards_mapped(),
        vma_mb,
        rss,
        vmsize,
    );
}

fn main() {
    let (model_dir, mode, layers, experts, lazy) = parse_args();
    let t_total = Instant::now();
    print_header();

    let opts = OpenOptions { lazy_load: lazy };
    let src = SafetensorsExpertSource::open_with_options(&model_dir, opts)
        .expect("open safetensors source");
    print_row("after_open", t_total, &src);

    // Stage 1: walk shells + layer-0 + embed_tokens — what the Runner does today.
    let t = Instant::now();
    let _ = src.layer0().expect("layer0");
    print_row("after_layer0", t, &src);

    let t = Instant::now();
    for lid in 1..layers {
        let _ = src
            .shell(lid)
            .unwrap_or_else(|e| panic!("shell L{lid}: {e}"));
    }
    print_row("after_shells", t, &src);

    let t = Instant::now();
    let _ = src.embed_tokens().expect("embed_tokens");
    print_row("after_embed", t, &src);

    if mode == Mode::Shells {
        println!("# total elapsed = {:.3}s", t_total.elapsed().as_secs_f64());
        return;
    }

    // Stage 2: walk every (layer, expert). Layer 0 is dense (no experts);
    // experts live in layers 1..layers.
    let t = Instant::now();
    let mut n_experts = 0u32;
    for lid in 1..layers {
        for eid in 0..experts {
            let _e = src
                .expert(lid, eid)
                .unwrap_or_else(|e| panic!("expert L{lid}/E{eid}: {e}"));
            n_experts += 1;
        }
    }
    print_row(&format!("after_all_experts({n_experts})"), t, &src);

    if mode == Mode::AllExperts {
        println!("# total elapsed = {:.3}s", t_total.elapsed().as_secs_f64());
        return;
    }

    // Stage 3: walk every tensor in the index — forces every shard to be
    // mmap'd, regardless of whether its tensors are part of the K2.6
    // dispatch path.
    let t = Instant::now();
    let mut n_tensors = 0u32;
    for name in src.tensor_names() {
        if let Err(e) = src.tensor_bytes(&name) {
            eprintln!("# tensor {name}: {e}");
            continue;
        }
        n_tensors += 1;
    }
    print_row(&format!("after_all_tensors({n_tensors})"), t, &src);

    if mode == Mode::AllTensors {
        println!("# total elapsed = {:.3}s", t_total.elapsed().as_secs_f64());
        return;
    }

    // Stage 4: actually fault every page of every shard. Reads one byte
    // every 4096 bytes — enough to force the kernel to populate the
    // page-table entry. RSS should approach VmSize after this completes.
    let t = Instant::now();
    let mut touched: u64 = 0;
    for name in src.tensor_names() {
        // tensor_bytes returns (Arc<Shard>, &'static [u8]) — we want
        // the bytes for stride-walk. Pin Arc by name to keep mmap alive.
        let (_pin, bytes) = match src.tensor_bytes(&name) {
            Ok(t) => t,
            Err(_) => continue,
        };
        // Volatile single-byte read every 4 KB. `read_volatile` keeps
        // the compiler from optimizing out the touch.
        let mut sink: u8 = 0;
        let mut off = 0usize;
        while off < bytes.len() {
            // SAFETY: bytes lives as long as the Arc<Shard> pin.
            sink = sink.wrapping_add(unsafe { std::ptr::read_volatile(&bytes[off]) });
            off += 4096;
            touched += 1;
        }
        // Force the compiler to keep `sink` live.
        std::hint::black_box(sink);
    }
    print_row(&format!("after_populate({touched}_pages)"), t, &src);
    println!("# total elapsed = {:.3}s", t_total.elapsed().as_secs_f64());
}
