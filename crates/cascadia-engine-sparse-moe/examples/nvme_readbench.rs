//! R1 go/no-go: does explicit concurrent expert-reading beat mmap-fault-on-touch
//! on a real NVMe? Models the per-token MoE read pattern: each "token" picks K
//! random expert bins from a dir and reads them two ways —
//!   A) MMAP+TOUCH: mmap each file, read every byte sequentially (what the GEMV
//!      does today — pages fault in on the compute path, one file at a time).
//!   B) EXPLICIT-PARALLEL: read_at the whole file into an owned buffer, K files
//!      concurrently (rayon) — the R1 change.
//! Reports MB/s + per-token wall time for each. Run on a dir of expert bins that
//! lives on the target NVMe. Drop caches between phases for a true cold read.
//!
//!   cargo run --release --example nvme_readbench -- <dir-of-bins-on-nvme> [K=8] [tokens=16]
//!
//! (Linux; uses std::os::unix::fs::FileExt::read_at.)

// `read_at` is the point of the B) phase and is unix-only, so the bench only
// exists there. The fleet is Windows, where `--all-targets` builds every
// example — without this gate the workspace does not compile at all.
#[cfg(not(unix))]
fn main() {
    eprintln!(
        "nvme_readbench measures std::os::unix::fs::FileExt::read_at against mmap \
         faulting; there is nothing to run on this platform."
    );
}

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(unix)]
use std::path::PathBuf;
#[cfg(unix)]
use std::time::Instant;

#[cfg(unix)]
use memmap2::Mmap;
#[cfg(unix)]
use rayon::prelude::*;

#[cfg(unix)]
fn list_bins(dir: &PathBuf) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "bin").unwrap_or(false))
        .collect();
    v.sort();
    v
}

// Deterministic LCG so token->expert selection is reproducible without rand.
#[cfg(unix)]
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 16
}

#[cfg(unix)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        args.get(1)
            .expect("usage: nvme_readbench <dir> [K] [tokens]"),
    );
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    let bins = list_bins(&dir);
    assert!(
        bins.len() >= k,
        "need >= {k} .bin files, found {}",
        bins.len()
    );
    let total_gb: f64 = bins
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum::<u64>() as f64
        / 1e9;
    println!(
        "[readbench] dir={} bins={} ({:.1} GB) K={k} tokens={tokens}",
        dir.display(),
        bins.len(),
        total_gb
    );
    println!("[readbench] NOTE: drop caches before each phase — `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`");

    // Reproducible per-token expert picks (same for both phases).
    let mut st = 0x1234_5678_9abc_def0u64;
    let picks: Vec<Vec<usize>> = (0..tokens)
        .map(|_| {
            (0..k)
                .map(|_| (lcg(&mut st) as usize) % bins.len())
                .collect()
        })
        .collect();

    let bytes_of = |idxs: &[usize]| -> u64 {
        idxs.iter()
            .map(|&i| std::fs::metadata(&bins[i]).unwrap().len())
            .sum()
    };

    // Phase A — mmap + touch every byte, one file at a time (today's path).
    let phase = std::env::var("PHASE").unwrap_or_else(|_| "both".into());
    if phase == "A" || phase == "both" {
        let t0 = Instant::now();
        let mut bytes = 0u64;
        let mut sink = 0u64;
        for tok in &picks {
            for &i in tok {
                let f = File::open(&bins[i]).unwrap();
                let mm = unsafe { Mmap::map(&f) }.unwrap();
                // touch every 4 KB page (forces the fault-in the GEMV would cause)
                let mut o = 0;
                while o < mm.len() {
                    sink = sink.wrapping_add(mm[o] as u64);
                    o += 4096;
                }
                bytes += mm.len() as u64;
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "[A mmap+touch ] {:.2} GB in {:.2}s = {:.0} MB/s | {:.1} ms/token (sink {})",
            bytes as f64 / 1e9,
            dt,
            bytes as f64 / 1e6 / dt,
            dt * 1e3 / tokens as f64,
            sink & 0xff
        );
    }

    if phase == "B" || phase == "both" {
        let t0 = Instant::now();
        let mut bytes = 0u64;
        let mut sink = 0u64;
        for tok in &picks {
            bytes += bytes_of(tok);
            // read the token's K experts concurrently, each one whole-file read_at.
            let bufs: Vec<Vec<u8>> = tok
                .par_iter()
                .map(|&i| {
                    let f = File::open(&bins[i]).unwrap();
                    let len = f.metadata().unwrap().len() as usize;
                    let mut buf = vec![0u8; len];
                    let mut off = 0usize;
                    while off < len {
                        let n = f.read_at(&mut buf[off..], off as u64).unwrap();
                        if n == 0 {
                            break;
                        }
                        off += n;
                    }
                    buf
                })
                .collect();
            for b in &bufs {
                sink = sink.wrapping_add(b.first().copied().unwrap_or(0) as u64);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "[B explicit|| ] {:.2} GB in {:.2}s = {:.0} MB/s | {:.1} ms/token (sink {})",
            bytes as f64 / 1e9,
            dt,
            bytes as f64 / 1e6 / dt,
            dt * 1e3 / tokens as f64,
            sink & 0xff
        );
    }
}
