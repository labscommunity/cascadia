//! Opt-in GPU keep-alive for the qwen35 pipeline (`CASCADIA_QWEN36_GPU_KEEPALIVE_MS=<n>`).
//!
//! Measured on the 122B 3-rank iGPU pipeline: a stage's infer is p50 ~33 ms but p95 250-550 ms
//! on every rank, while a standalone loop on the same stage has no gaps (p99 57 ms). Inserting a
//! 70 ms sleep between infers reproduces it (16% of steps take ~570 ms; frequency pinning does
//! not help), and running a tiny infer (~1.7 ms) every ~5 ms during the gap removes the stalls
//! entirely (p99 68 ms): the device idles between pipeline steps and pays a wake-up penalty on
//! the next real infer. So, while a rank is blocked on the network — a relay waiting for the
//! next upstream frame or for its downstream reply, rank 0 waiting for the downstream token —
//! one worker thread keeps the device busy with a synthetic infer every `n` ms, and goes idle
//! the moment the wait ends, so it never overlaps a real stage infer on the same device.
//!
//! Byte-identical behaviour when the variable is unset (nothing is compiled or spawned).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::warn;

/// Keep-alive period env var: milliseconds between synthetic infers while waiting (0/unset = off).
pub const KEEPALIVE_ENV: &str = "CASCADIA_QWEN36_GPU_KEEPALIVE_MS";

/// A wait longer than this is the idle gap BETWEEN tasks (a relay parks on its upstream socket
/// indefinitely), not between steps: the worker stops firing until the next wait begins, so an
/// idle chain does not spin its GPUs at a 30% duty cycle forever.
pub const MAX_CONTINUOUS_WAIT: Duration = Duration::from_secs(2);

/// Width of the synthetic model's `[1, SYNTHETIC_WIDTH]` f32 input.
pub const SYNTHETIC_WIDTH: usize = 16;

/// Parse the env var's value: `None` for unset, empty, 0, or anything that is not a positive
/// integer number of milliseconds.
pub fn parse_keepalive_ms(raw: Option<&str>) -> Option<Duration> {
    let ms: u64 = raw?.trim().parse().ok()?;
    (ms > 0).then(|| Duration::from_millis(ms))
}

/// [`parse_keepalive_ms`] over the process environment.
pub fn keepalive_period() -> Option<Duration> {
    parse_keepalive_ms(std::env::var(KEEPALIVE_ENV).ok().as_deref())
}

/// Minimal OpenVINO IR (v11): `Parameter f32 [1,16] -> Add(Const f32 [1,16]) -> Result`. The
/// constant's 64 bytes live at offset 0 of the companion `.bin` ([`synthetic_ir_bin`]).
pub fn synthetic_ir_xml() -> String {
    let w = SYNTHETIC_WIDTH;
    let dims = format!("<dim>1</dim><dim>{w}</dim>");
    format!(
        r#"<?xml version="1.0"?>
<net name="cascadia_qwen36_keepalive" version="11">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data shape="1,{w}" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="x">{dims}</port>
			</output>
		</layer>
		<layer id="1" name="c" type="Const" version="opset1">
			<data element_type="f32" shape="1,{w}" offset="0" size="{size}"/>
			<output>
				<port id="0" precision="FP32">{dims}</port>
			</output>
		</layer>
		<layer id="2" name="add" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<input>
				<port id="0" precision="FP32">{dims}</port>
				<port id="1" precision="FP32">{dims}</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="y">{dims}</port>
			</output>
		</layer>
		<layer id="3" name="y" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">{dims}</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
"#,
        size = w * 4
    )
}

/// The synthetic IR's weights: `SYNTHETIC_WIDTH` f32 ones, little-endian.
pub fn synthetic_ir_bin() -> Vec<u8> {
    std::iter::repeat(1.0f32.to_le_bytes())
        .take(SYNTHETIC_WIDTH)
        .flatten()
        .collect()
}

/// Write `keepalive.xml` + `keepalive.bin` under `dir` (created if needed); returns the xml path.
pub fn write_synthetic_ir(dir: &Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let xml = dir.join("keepalive.xml");
    std::fs::write(&xml, synthetic_ir_xml())?;
    std::fs::write(dir.join("keepalive.bin"), synthetic_ir_bin())?;
    Ok(xml)
}

/// The keep-alive worker: one thread that runs `infer` every `period` while [`begin_wait`] has
/// been called more recently than [`end_wait`], and parks otherwise. Owns the synthetic request
/// through the closure, so the stage requests are never touched from this thread.
///
/// [`begin_wait`]: KeepAlive::begin_wait
/// [`end_wait`]: KeepAlive::end_wait
pub struct KeepAlive {
    waiting: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    infers: Arc<AtomicU64>,
    /// µs per keep-alive infer, drained by the profiler (`ka.infer`); only filled when asked.
    samples: Arc<Mutex<Vec<f64>>>,
    thread: Option<JoinHandle<()>>,
}

impl KeepAlive {
    /// Start the worker. `infer` runs one synthetic inference (`Err` ⇒ the worker logs once and
    /// stops; [`is_alive`](Self::is_alive) then reads false). `record_samples` keeps per-infer µs
    /// for the profiler.
    pub fn spawn<F>(period: Duration, record_samples: bool, mut infer: F) -> Self
    where
        F: FnMut() -> Result<(), String> + Send + 'static,
    {
        let waiting = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let infers = Arc::new(AtomicU64::new(0));
        let samples: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let period = period.max(Duration::from_millis(1));
        let (w, s, f, n, smp) = (
            Arc::clone(&waiting),
            Arc::clone(&stop),
            Arc::clone(&failed),
            Arc::clone(&infers),
            Arc::clone(&samples),
        );
        let thread = thread::Builder::new()
            .name("qwen36-gpu-keepalive".into())
            .spawn(move || {
                let mut wait_started: Option<Instant> = None;
                loop {
                    if s.load(Ordering::Relaxed) {
                        return;
                    }
                    if !w.load(Ordering::Acquire) {
                        wait_started = None;
                        // Parked until `begin_wait` unparks us (or the period elapses).
                        thread::park_timeout(period);
                        continue;
                    }
                    let started = *wait_started.get_or_insert_with(Instant::now);
                    if started.elapsed() > MAX_CONTINUOUS_WAIT {
                        // Between tasks, not between steps: stay quiet until this wait ends.
                        thread::park_timeout(period);
                        continue;
                    }
                    let t0 = Instant::now();
                    if let Err(e) = infer() {
                        warn!(error = %e, "qwen36 GPU keep-alive infer failed; keep-alive OFF");
                        f.store(true, Ordering::Release);
                        return;
                    }
                    n.fetch_add(1, Ordering::Relaxed);
                    if record_samples {
                        smp.lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(t0.elapsed().as_micros() as f64);
                    }
                    thread::sleep(period);
                }
            })
            .ok();
        if thread.is_none() {
            warn!("qwen36 GPU keep-alive: thread spawn failed; keep-alive OFF");
            failed.store(true, Ordering::Release);
        }
        Self {
            waiting,
            stop,
            failed,
            infers,
            samples,
            thread,
        }
    }

    /// A blocking network wait starts: fire synthetic infers until [`end_wait`](Self::end_wait).
    pub fn begin_wait(&self) {
        self.waiting.store(true, Ordering::Release);
        if let Some(t) = &self.thread {
            t.thread().unpark();
        }
    }

    /// The wait returned: stop firing before the next real infer on the device.
    pub fn end_wait(&self) {
        self.waiting.store(false, Ordering::Release);
    }

    /// Keep-alive infers since the last call (then zero).
    pub fn take_infers(&self) -> u64 {
        self.infers.swap(0, Ordering::Relaxed)
    }

    /// Recorded per-infer µs since the last call (then empty).
    pub fn take_samples(&self) -> Vec<f64> {
        std::mem::take(&mut *self.samples.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// False once the worker has given up (infer error / spawn failure).
    pub fn is_alive(&self) -> bool {
        !self.failed.load(Ordering::Acquire)
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_value_parses_positive_millis_only() {
        assert_eq!(parse_keepalive_ms(None), None);
        assert_eq!(parse_keepalive_ms(Some("")), None);
        assert_eq!(parse_keepalive_ms(Some("0")), None);
        assert_eq!(parse_keepalive_ms(Some("x")), None);
        assert_eq!(parse_keepalive_ms(Some("-5")), None);
        assert_eq!(
            parse_keepalive_ms(Some(" 5 ")),
            Some(Duration::from_millis(5))
        );
    }

    #[test]
    fn synthetic_ir_is_consistent() {
        let xml = synthetic_ir_xml();
        for needle in [
            r#"type="Parameter""#,
            r#"type="Const""#,
            r#"type="Add""#,
            r#"type="Result""#,
            r#"shape="1,16""#,
            r#"offset="0" size="64""#,
            r#"version="11""#,
        ] {
            assert!(xml.contains(needle), "missing {needle}");
        }
        assert_eq!(xml.matches("<edge ").count(), 3);
        assert_eq!(synthetic_ir_bin().len(), SYNTHETIC_WIDTH * 4);
        let dir = tempfile::tempdir().unwrap();
        let xml_path = write_synthetic_ir(&dir.path().join("ka")).unwrap();
        assert!(xml_path.ends_with("keepalive.xml"));
        assert_eq!(
            std::fs::read(xml_path.with_extension("bin")).unwrap().len(),
            64
        );
    }

    /// The worker fires only between `begin_wait` and `end_wait`, counts what it fired, and
    /// stops cleanly on drop.
    #[test]
    fn worker_fires_only_while_waiting() {
        let fired = Arc::new(AtomicU64::new(0));
        let f = Arc::clone(&fired);
        let ka = KeepAlive::spawn(Duration::from_millis(1), true, move || {
            f.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        thread::sleep(Duration::from_millis(30));
        assert_eq!(fired.load(Ordering::Relaxed), 0, "idle before any wait");
        ka.begin_wait();
        thread::sleep(Duration::from_millis(60));
        ka.end_wait();
        let n = ka.take_infers();
        assert!(n > 0, "fired during the wait");
        assert_eq!(n, fired.load(Ordering::Relaxed));
        assert_eq!(ka.take_samples().len() as u64, n);
        assert_eq!(ka.take_infers(), 0, "take resets");
        // Give the worker time to observe end_wait, then make sure it stays quiet.
        thread::sleep(Duration::from_millis(10));
        let settled = fired.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(40));
        assert_eq!(
            fired.load(Ordering::Relaxed),
            settled,
            "idle after the wait"
        );
        assert!(ka.is_alive());
        drop(ka);
    }

    #[test]
    fn worker_gives_up_on_infer_error() {
        let ka = KeepAlive::spawn(Duration::from_millis(1), false, || Err("boom".into()));
        ka.begin_wait();
        let deadline = Instant::now() + Duration::from_secs(2);
        while ka.is_alive() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(!ka.is_alive());
        assert_eq!(ka.take_infers(), 0);
    }
}
