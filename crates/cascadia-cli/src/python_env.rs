//! Finding a Python that can run the embedded exporters, and building a private
//! one when the machine has none.
//!
//! [`resolve_python`] only looks: it never prompts or downloads, so `doctor` can
//! use it. [`python_for_export`] is what `shard` and `export` call: it adds the
//! offer to set up the private environment.
//!
//! The private environment installs only what the machine lacks. An existing
//! `uv`, a system Python 3.12 or 3.13 and any system package at exactly the
//! locked version are reused; a wrong version is left alone and shadowed by the
//! private copy. The system Python itself is never written to.

use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::{export_requirements, EXPORT_IMPORTS};

/// Importing torch, optimum.intel and nncf takes tens of seconds on a cold
/// Windows box; a hung interpreter must not hang the export.
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);

/// Prefix of the probes' result lines, so import-time chatter is ignored.
const PROBE_MARK: &str = "cascadia-probe";

/// `uv` downloaded when none is on PATH.
const UV_VERSION: &str = "0.13.7";

/// Oldest `uv` on PATH that setup uses; older ones lack flags it passes.
const UV_MIN_VERSION: &[u64] = &[0, 10, 4];

/// Disk needed before a setup starts: the packages, their download cache and
/// the interpreter. The largest measured was just over 2 GB, on Windows.
const SETUP_FREE_BYTES: u64 = 3 << 30;

/// What setup needs for this build's platform.
struct Platform {
    lock: &'static str,
    uv_asset: &'static str,
    uv_sha256: &'static str,
    /// What `sysconfig.get_platform()` of a usable system Python contains.
    python_platform: &'static str,
}

/// `None` where OpenVINO publishes no wheels, so there is nothing to install.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const PLATFORM: Option<Platform> = Some(Platform {
    lock: include_str!("../../../tools/export-locks/linux-x86_64.txt"),
    uv_asset: "uv-x86_64-unknown-linux-gnu.tar.gz",
    uv_sha256: "c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83",
    python_platform: "linux-x86_64",
});
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const PLATFORM: Option<Platform> = Some(Platform {
    lock: include_str!("../../../tools/export-locks/windows-x86_64.txt"),
    uv_asset: "uv-x86_64-pc-windows-msvc.zip",
    uv_sha256: "0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021",
    python_platform: "win-amd64",
});
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PLATFORM: Option<Platform> = Some(Platform {
    lock: include_str!("../../../tools/export-locks/macos-arm64.txt"),
    uv_asset: "uv-aarch64-apple-darwin.tar.gz",
    uv_sha256: "81b811b57e4624df7ed3aac148e0359100d9962f3d9dce00aec533dc2e086731",
    python_platform: "arm64",
});
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64"),
)))]
const PLATFORM: Option<Platform> = None;

/// How strict a Python on PATH has to be for the command at hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Need {
    /// `shard`: imports the core packages. A version outside the pins only
    /// warns, because many exports work on those.
    Imports,
    /// `export`: inside every pin, because the pinned optimum export fails
    /// outside them.
    Pins,
}

/// A Python interpreter and what it can do.
#[derive(Debug)]
pub(crate) struct PythonEnv {
    pub path: String,
    /// Version summary from the dependency probe; `None` if the imports failed
    /// (or `check_deps` was false), so callers can skip probing a second time.
    pub deps: Option<String>,
    /// Why this interpreter was chosen, for the line every run prints.
    pub why: String,
    /// The private environment: the exporter runs without the caller's
    /// `PYTHON*` variables, and hints name `cascadia python setup`.
    pub managed: bool,
    /// Held while the exporter runs, so `python remove` can't delete it.
    pub guard: Option<std::fs::File>,
}

impl PythonEnv {
    /// A command running this interpreter with the environment it needs.
    pub(crate) fn command(&self) -> Command {
        let mut cmd = Command::new(&self.path);
        if self.managed {
            scrub_python_env(&mut cmd);
            cmd.env("CASCADIA_MANAGED_PYTHON", "1");
        }
        cmd
    }
}

/// The arguments that run `code` with `python -c`, with the working folder off
/// `sys.path`: `-c` searches it first, so a `torch.py` in the folder someone runs
/// Cascadia from would be imported. `-P` does the same, but only from Python 3.11.
pub(crate) fn run_code(code: &str) -> [String; 2] {
    let no_cwd = "import sys; sys.path[:] = [p for p in sys.path if p]";
    ["-c".into(), format!("{no_cwd}\n{code}")]
}

/// `PYTHONPATH` would let a stray package replace the locked one, and
/// `PYTHONHOME` stops the interpreter from starting at all.
fn scrub_python_env(cmd: &mut Command) {
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("PYTHON") {
            cmd.env_remove(key);
        }
    }
}

/// True only if the interpreter runs AND exits 0. Spawning is not enough: the
/// Windows Store `python3.exe` alias spawns, prints "Python was not found" and
/// exits 9009 — treating that as an interpreter is the very bug we're fixing.
fn python_runs(p: &str) -> bool {
    Command::new(p)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Run `cmd` to exit, killing it at `timeout`. `Ok(stdout)` on exit 0, else why.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Result<String, String> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run: {e}"))?;
    // Drained on threads: a traceback can outgrow a pipe buffer and stall the child.
    let out = drain(child.stdout.take());
    let err = drain(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            res => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(match res {
                    Err(e) => format!("waiting failed: {e}"),
                    _ => format!("timed out after {}s", timeout.as_secs()),
                });
            }
        }
    };
    let stdout = out.join().unwrap_or_default();
    let stderr = err.join().unwrap_or_default();
    if !status.success() {
        return Err(String::from_utf8_lossy(&stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&stdout).into_owned())
}

fn drain<R: std::io::Read + Send + 'static>(r: Option<R>) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut r) = r {
            let _ = r.read_to_end(&mut buf);
        }
        buf
    })
}

/// Import the exporter's deps and report their versions. `None` if the imports
/// fail — that interpreter can't run the exporter.
fn probe_export_deps(mut cmd: Command) -> Option<String> {
    let code = format!(
        "{EXPORT_IMPORTS}; import sys; \
         print('python', sys.version.split()[0]); \
         print('torch', torch.__version__); \
         print('openvino', openvino.__version__); \
         print('transformers', transformers.__version__)"
    );
    run_with_timeout(cmd.args(run_code(&code)), PROBE_TIMEOUT).ok()
}

/// The module a pinned pip dist imports as: the dist name with `-` as `_`, as
/// most packages do. A new pin that imports otherwise fails the probe's import,
/// rejecting every candidate, until it is added here.
fn import_module(dist: &str) -> String {
    match dist {
        "optimum-intel" => "optimum.intel".into(),
        d => d.replace('-', "_"),
    }
}

/// Prefix of the line naming a requirement that didn't import.
const MISSING_MARK: &str = "cascadia-missing";

/// One subprocess: try every requirement and report each as a
/// `cascadia-probe <dist> <version>` or `cascadia-missing <dist>` line. A single
/// pass answers both "inside the pins?" and "can it run a shard export?", since
/// importing torch takes tens of seconds on a cold Windows box.
fn probe_pins(mut cmd: Command) -> Result<String, String> {
    let reqs: Vec<String> = export_requirements()
        .iter()
        .map(|(d, _)| format!("({d:?}, {:?})", import_module(d)))
        .collect();
    // The imported module's own `__version__` first, as `probe_export_deps`
    // prints it: under the SDK's setupvars the imported openvino can differ from
    // the pip wheel that `importlib.metadata` would report.
    let code = format!(
        "import importlib, importlib.metadata as md, sys\n\
         print('{PROBE_MARK}', 'python', sys.version.split()[0])\n\
         for d, m in [{}]:\n\
         \x20   try:\n\
         \x20       mod = importlib.import_module(m)\n\
         \x20   except Exception:\n\
         \x20       print('{MISSING_MARK}', d)\n\
         \x20       continue\n\
         \x20   v = getattr(mod, '__version__', None)\n\
         \x20   if not isinstance(v, str):\n\
         \x20       try:\n\
         \x20           v = md.version(d)\n\
         \x20       except md.PackageNotFoundError:\n\
         \x20           v = '?'\n\
         \x20   print('{PROBE_MARK}', d, v)\n",
        reqs.join(", ")
    );
    run_with_timeout(cmd.args(run_code(&code)), PROBE_TIMEOUT).map_err(|e| {
        let last = e.lines().last().unwrap_or("").to_string();
        format!("imports failed: {last}")
    })
}

/// Did a [`probe_pins`] run import everything a shard export needs?
fn core_imports_ok(probe_out: &str) -> bool {
    let core: Vec<&str> = EXPORT_IMPORTS
        .trim_start_matches("import ")
        .split(',')
        .map(str::trim)
        .collect();
    !probe_out
        .lines()
        .filter_map(|l| l.trim().strip_prefix(MISSING_MARK))
        .any(|d| core.contains(&d.trim()))
}

/// The version lines of a [`probe_pins`] run, as `probe_export_deps` prints them.
fn pin_summary(probe_out: &str) -> String {
    probe_out
        .lines()
        .filter_map(|l| l.trim().strip_prefix(PROBE_MARK))
        .map(str::trim)
        .filter(|l| {
            ["python ", "torch ", "openvino ", "transformers "]
                .iter()
                .any(|name| l.starts_with(name))
        })
        .map(|l| format!("{l}\n"))
        .collect()
}

/// `Ok` when the probe output names a version inside every requirement's range.
fn check_versions(probe_out: &str, reqs: &[(&str, &str)]) -> Result<(), String> {
    let found: HashMap<&str, &str> = probe_out
        .lines()
        .filter_map(|l| l.trim().strip_prefix(PROBE_MARK)?.trim().split_once(' '))
        .collect();
    let bad: Vec<String> = reqs
        .iter()
        .filter_map(|(dist, spec)| match found.get(dist) {
            Some(v) if satisfies(v, spec) => None,
            Some(v) => Some(format!("{dist} {v} (needs {spec})")),
            None => Some(format!("{dist} missing")),
        })
        .collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(bad.join(", "))
    }
}

/// `version` against a comma-joined specifier (`>=5.5,<5.6`) on release numbers.
/// Pre and dev releases fail, as pip excludes them when no clause names one.
/// Unparseable versions or operators fail.
fn satisfies(version: &str, spec: &str) -> bool {
    let Some((have, false)) = parse_version(version) else {
        return false;
    };
    spec.split(',').map(str::trim).all(|clause| {
        let Some((op, want)) = ["==", ">=", "<=", ">", "<"]
            .iter()
            .find_map(|op| clause.strip_prefix(op).map(|v| (*op, v)))
        else {
            return false;
        };
        let Some((want, _)) = parse_version(want) else {
            return false;
        };
        let ord = cmp_release(&have, &want);
        match op {
            "==" => ord.is_eq(),
            ">=" => ord.is_ge(),
            "<=" => ord.is_le(),
            ">" => ord.is_gt(),
            _ => ord.is_lt(),
        }
    })
}

/// Release numbers of a version and whether it is a pre/dev release. The local
/// label (`+cpu`) and openvino's build suffix (`-22982-…`) are dropped:
/// `2026.4.1-22982` -> ([2026, 4, 1], false), `2.6.0a0+git1` -> ([2, 6, 0], true).
fn parse_version(v: &str) -> Option<(Vec<u64>, bool)> {
    let v = v.trim().split(['+', '-']).next()?;
    let end = v
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(v.len());
    let (rel, tail) = v.split_at(end);
    let nums = rel
        .trim_end_matches('.')
        .split('.')
        .map(|n| n.parse().ok())
        .collect::<Option<Vec<u64>>>()?;
    let pre = !tail.is_empty() && !tail.to_ascii_lowercase().starts_with("post");
    Some((nums, pre))
}

/// Compare releases with missing trailing components as zero (5.5 == 5.5.0).
fn cmp_release(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    let at = |v: &[u64], i: usize| v.get(i).copied().unwrap_or(0);
    (0..a.len().max(b.len()))
        .map(|i| at(a, i).cmp(&at(b, i)))
        .find(|o| o.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Interpreters to try after any activated environment, best first.
fn python_candidates() -> Vec<String> {
    #[cfg_attr(not(target_os = "windows"), allow(unused_mut))]
    let mut candidates: Vec<String> = vec!["python3".into(), "python".into()];

    // python.org and winget installs often aren't on PATH. Walk Program Files,
    // then the per-user root, for `Python3xx\python.exe`, newest first.
    #[cfg(target_os = "windows")]
    for root in [
        std::env::var_os("ProgramFiles").map(PathBuf::from),
        std::env::var_os("ProgramFiles(x86)").map(PathBuf::from),
        std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("Programs").join("Python")),
    ]
    .into_iter()
    .flatten()
    {
        // The minor version as a number, so 3.10 sorts above 3.9.
        let mut found: Vec<(u32, PathBuf)> = std::fs::read_dir(&root)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter_map(|p| {
                let name = p.file_name()?.to_str()?;
                let minor = name.strip_prefix("Python3")?.parse().ok()?;
                Some((minor, p.join("python.exe")))
            })
            .filter(|(_, exe)| exe.exists())
            .collect();
        found.sort_by(|a, b| b.0.cmp(&a.0));
        candidates.extend(
            found
                .into_iter()
                .filter_map(|(_, exe)| exe.to_str().map(str::to_owned)),
        );
    }
    candidates
}

/// The interpreter of an activated venv or non-base conda environment. Conda's
/// `base` is left out: it activates itself in every shell, so it says nothing
/// about what the user wants here.
fn activated_python() -> Option<(String, &'static str)> {
    let (prefix, label) = match std::env::var_os("VIRTUAL_ENV") {
        Some(venv) => (PathBuf::from(venv), "activated venv"),
        None => {
            if std::env::var("CONDA_DEFAULT_ENV").is_ok_and(|e| e == "base") {
                return None;
            }
            (
                PathBuf::from(std::env::var_os("CONDA_PREFIX")?),
                "activated conda environment",
            )
        }
    };
    let exe = python_in(&prefix);
    let conda_windows = prefix.join("python.exe");
    let exe = if exe.exists() { exe } else { conda_windows };
    exe.exists()
        .then(|| (exe.to_string_lossy().into_owned(), label))
}

/// Resolve the interpreter to run the embedded exporter with. Only looks: it
/// never prompts, downloads or installs.
///
/// `--python`, then an activated environment, then the private environment,
/// then PATH. Falls back to any interpreter that runs, so the caller can still
/// surface the real `ModuleNotFoundError`.
pub(crate) fn resolve_python(
    explicit: Option<&str>,
    check_deps: bool,
    need: Need,
) -> Result<PythonEnv> {
    let env = |path: &str, deps, why: &str| PythonEnv {
        path: path.to_string(),
        deps,
        why: why.to_string(),
        managed: false,
        guard: None,
    };
    // An interpreter the user chose is used regardless. `export` also says when
    // it is outside the pins, the first thing to check if the export misbehaves.
    let probe = |p: &str| -> Option<String> {
        match (check_deps, need) {
            (false, _) => None,
            (true, Need::Imports) => probe_export_deps(Command::new(p)),
            (true, Need::Pins) => {
                let out = probe_pins(Command::new(p)).ok()?;
                if !core_imports_ok(&out) {
                    return None;
                }
                if let Err(why) = check_versions(&out, export_requirements()) {
                    eprintln!("warning: {p} is outside the exporter's pins: {why}");
                }
                Some(pin_summary(&out))
            }
        }
    };

    if let Some(p) = explicit {
        if let Some(deps) = probe(p) {
            return Ok(env(p, Some(deps), "--python"));
        }
        return if python_runs(p) {
            Ok(env(p, None, "--python"))
        } else {
            Err(anyhow!(
                "cannot run the interpreter passed to --python: {p:?}"
            ))
        };
    }

    if let Some((p, label)) = activated_python() {
        if !check_deps {
            return Ok(env(&p, None, label));
        }
        if let Some(deps) = probe(&p) {
            return Ok(env(&p, Some(deps), label));
        }
    }

    if let Some(found) = find_managed(check_deps) {
        return Ok(found);
    }

    let candidates = python_candidates();
    if !check_deps {
        return candidates
            .iter()
            .find(|p| python_runs(p))
            .map(|p| env(p, None, "first Python on PATH"))
            .ok_or_else(no_python_error);
    }

    let mut importer = None;
    let mut rejected = Vec::new();
    let mut runnable = None;
    for p in &candidates {
        match probe_pins(Command::new(p)) {
            Ok(out) => match check_versions(&out, export_requirements()) {
                Ok(()) => {
                    return Ok(env(
                        p,
                        Some(pin_summary(&out)),
                        "on PATH, meets the exporter's pins",
                    ))
                }
                Err(why) => {
                    rejected.push(format!("{p}: {why}"));
                    if importer.is_none() && core_imports_ok(&out) {
                        importer = Some((p.clone(), pin_summary(&out)));
                    }
                }
            },
            Err(why) => rejected.push(format!("{p}: {why}")),
        }
        if runnable.is_none() && python_runs(p) {
            runnable = Some(p.clone());
        }
    }
    match (need, importer) {
        (Need::Imports, Some((p, deps))) => {
            tracing::debug!(rejected = %rejected.join("; "), "no Python meets the pins");
            eprintln!(
                "warning: {p} has package versions the exporter isn't tested with; \
                 run `cascadia python setup` to use tested ones."
            );
            Ok(env(&p, Some(deps), "on PATH, outside the exporter's pins"))
        }
        _ => runnable
            .map(|p| env(&p, None, "on PATH, missing packages"))
            .ok_or_else(no_python_error),
    }
}

fn no_python_error() -> anyhow::Error {
    anyhow!(
        "no Python interpreter found on PATH. Install Python 3.10-3.13 or pass --python <path>."
    )
}

/// The interpreter inside a venv at `prefix`.
fn python_in(prefix: &Path) -> PathBuf {
    if cfg!(windows) {
        prefix.join("Scripts").join("python.exe")
    } else {
        prefix.join("bin").join("python")
    }
}

/// `CASCADIA_PYTHON_HOME`, else the platform's data dir: not its cache dir,
/// whose cleaners can remove parts of an environment and leave it looking fine.
fn managed_root() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("CASCADIA_PYTHON_HOME") {
        return Ok(PathBuf::from(home));
    }
    dirs::data_local_dir()
        .map(|d| d.join("cascadia").join("python"))
        .context("no user data directory on this machine; pass --python <path>")
}

/// A short name for the lock's content: a release with new pins gets a new
/// environment beside the old one.
fn lock_id(lock: &str) -> String {
    Sha256::digest(lock.replace('\r', "").as_bytes())
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Paths of one private environment.
struct Managed {
    root: PathBuf,
    dir: PathBuf,
    id: String,
}

const COMPLETE_MARKER: &str = "cascadia-complete";

impl Managed {
    fn current() -> Result<Option<Self>> {
        let Some(platform) = &PLATFORM else {
            return Ok(None);
        };
        let root = managed_root()?;
        let id = lock_id(platform.lock);
        Ok(Some(Self {
            dir: root.join(&id),
            root,
            id,
        }))
    }
    fn python(&self) -> PathBuf {
        python_in(&self.dir)
    }
    /// Written last, after the import test: anything else is unfinished.
    fn is_complete(&self) -> bool {
        self.dir.join(COMPLETE_MARKER).is_file() && self.python().exists()
    }
    fn lock_path(&self) -> PathBuf {
        self.root.join(format!("{}.lock", self.id))
    }
    fn open_lock(&self) -> Result<std::fs::File> {
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("creating {}", self.root.display()))?;
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.lock_path())
            .with_context(|| format!("opening {}", self.lock_path().display()))
    }
    fn env(&self, deps: Option<String>) -> PythonEnv {
        PythonEnv {
            path: self.python().to_string_lossy().into_owned(),
            deps,
            why: "Cascadia's private environment".into(),
            managed: true,
            guard: None,
        }
    }
}

/// The private environment, if it is finished and (when checked) imports.
fn find_managed(check_deps: bool) -> Option<PythonEnv> {
    let managed = Managed::current().ok().flatten()?;
    if !managed.is_complete() {
        return None;
    }
    if !check_deps {
        return Some(managed.env(None));
    }
    let mut env = managed.env(None);
    env.deps = probe_export_deps(env.command());
    match env.deps {
        Some(_) => Some(env),
        None => {
            eprintln!("Cascadia's Python packages are damaged; they need reinstalling.");
            None
        }
    }
}

/// One package of a lock: `name==version` and its `--hash` lines.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LockEntry {
    name: String,
    version: String,
    block: String,
}

fn normalize_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['_', '.'], "-")
}

/// Split a lock into packages. The locks are per platform, so they hold no
/// environment markers to evaluate.
fn parse_lock(lock: &str) -> Vec<LockEntry> {
    let mut entries: Vec<LockEntry> = Vec::new();
    for line in lock.lines() {
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if line.starts_with(char::is_whitespace) {
            if let Some(last) = entries.last_mut() {
                last.block.push_str(line);
                last.block.push('\n');
            }
            continue;
        }
        let spec = line.trim_end_matches(['\\', ' ']);
        if let Some((name, version)) = spec.split_once("==") {
            entries.push(LockEntry {
                name: normalize_name(name),
                version: version.trim().to_string(),
                block: format!("{line}\n"),
            });
        }
    }
    entries
}

/// The same release, ignoring the local label: a system `torch 2.15.0+cu124`
/// serves as well as the locked `2.15.0+cpu`, and is the largest package to skip.
fn same_release(a: &str, b: &str) -> bool {
    let base = |v: &str| v.split('+').next().unwrap_or(v).trim().to_string();
    base(a) == base(b)
}

/// The locked packages `installed` doesn't already have at exactly that version.
fn packages_to_install<'a>(
    lock: &'a [LockEntry],
    installed: &HashMap<String, String>,
) -> Vec<&'a LockEntry> {
    lock.iter()
        .filter(|e| {
            !installed
                .get(&e.name)
                .is_some_and(|have| same_release(have, &e.version))
        })
        .collect()
}

/// What an interpreter is, and which packages it can see. Standard library only.
const PY_INFO: &str = r#"
import importlib.metadata as m, json, struct, sys, sysconfig
d = {}
for x in m.distributions():
    n = x.metadata["Name"]
    if n:
        d.setdefault(n.lower().replace("_", "-").replace(".", "-"), x.version)
print("cascadia-probe " + json.dumps({
    "minor": list(sys.version_info[:2]),
    "venv": sys.prefix != sys.base_prefix,
    "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    "bits": struct.calcsize("P") * 8,
    "platform": sysconfig.get_platform(),
    "executable": sys.executable,
    "dists": d,
}))
"#;

#[derive(Debug, serde::Deserialize)]
struct PyInfo {
    minor: (u32, u32),
    venv: bool,
    free_threaded: bool,
    bits: u32,
    platform: String,
    executable: String,
    dists: HashMap<String, String>,
}

impl PyInfo {
    /// Can a private venv be layered on this interpreter? 3.11 can't install the
    /// lock and the exporter refuses 3.14; a venv isn't a base to layer on.
    fn is_usable_base(&self, platform: &str) -> bool {
        matches!(self.minor, (3, 12) | (3, 13))
            && !self.venv
            && !self.free_threaded
            && self.bits == 64
            && self.platform.contains(platform)
    }
}

fn py_info(mut cmd: Command) -> Result<PyInfo, String> {
    let out = run_with_timeout(cmd.args(run_code(PY_INFO)), PROBE_TIMEOUT)?;
    let line = out
        .lines()
        .find_map(|l| l.trim().strip_prefix(PROBE_MARK))
        .ok_or("no result from the interpreter")?;
    serde_json::from_str(line.trim()).map_err(|e| e.to_string())
}

/// A system Python to layer the private environment on, with its packages, by
/// absolute path: given a bare `python3`, uv prefers a Python it installed itself.
fn find_base_python(platform: &Platform) -> Option<(String, PyInfo)> {
    python_candidates().into_iter().find_map(|p| {
        let mut cmd = Command::new(&p);
        scrub_python_env(&mut cmd);
        let info = py_info(cmd).ok()?;
        info.is_usable_base(platform.python_platform)
            .then(|| (info.executable.clone(), info))
    })
}

/// `uv --version` output -> is it new enough to use?
fn uv_is_usable(version_output: &str) -> bool {
    version_output
        .split_whitespace()
        .nth(1)
        .and_then(parse_version)
        .is_some_and(|(v, _)| cmp_release(&v, UV_MIN_VERSION).is_ge())
}

/// `uv` on PATH when it is new enough. Packages are checked against the lock's
/// hashes whichever `uv` installs them.
fn uv_on_path() -> Option<PathBuf> {
    let out = Command::new("uv").arg("--version").output().ok()?;
    (out.status.success() && uv_is_usable(&String::from_utf8_lossy(&out.stdout)))
        .then(|| PathBuf::from("uv"))
}

fn own_uv_path(root: &Path) -> PathBuf {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    root.join("uv").join(UV_VERSION).join(exe)
}

/// Check a download against its pinned hash before anything reads it further.
fn verify_sha256(bytes: &[u8], want: &str) -> Result<()> {
    let got: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if got != want {
        bail!("checksum mismatch: expected {want}, got {got}");
    }
    Ok(())
}

/// Pull the `uv` binary out of a release archive. It is written to a fixed name,
/// so a path inside the archive can't place a file anywhere else.
#[cfg(unix)]
fn extract_uv(archive: &[u8], dest: &Path) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(archive);
    let mut tar = tar::Archive::new(gz);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if path.file_name().is_some_and(|n| n == "uv") {
            let mut out = std::fs::File::create(dest)?;
            std::io::copy(&mut entry, &mut out)?;
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
            return Ok(());
        }
    }
    bail!("no uv binary in the archive")
}

#[cfg(windows)]
fn extract_uv(archive: &[u8], dest: &Path) -> Result<()> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive))?;
    let mut entry = zip.by_name("uv.exe").context("no uv.exe in the archive")?;
    let mut out = std::fs::File::create(dest)?;
    std::io::copy(&mut entry, &mut out)?;
    Ok(())
}

/// Download the pinned `uv`, verify it and put it in place atomically.
/// How long to wait before each new try at a download.
const DOWNLOAD_RETRY_PAUSES: [Duration; 4] = [
    Duration::from_secs(2),
    Duration::from_secs(5),
    Duration::from_secs(10),
    Duration::from_secs(20),
];

/// GitHub can answer 504 for minutes on one route while working everywhere else.
/// A missing file stays missing, so that one fails at once.
fn worth_retrying(e: &reqwest::Error) -> bool {
    e.status()
        .is_none_or(|s| s.is_server_error() || s == reqwest::StatusCode::TOO_MANY_REQUESTS)
}

/// Run `attempt`, trying again after each pause while `retry` says it's worth it.
async fn with_retries<T, E, Fut>(
    pauses: &[Duration],
    retry: impl Fn(&E) -> bool,
    attempt: impl Fn() -> Fut,
) -> Result<T, E>
where
    E: std::fmt::Display,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    let mut pauses = pauses.iter();
    loop {
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(e) => match pauses.next() {
                Some(pause) if retry(&e) => {
                    eprintln!("Download failed, trying again in {}s: {e}", pause.as_secs());
                    tokio::time::sleep(*pause).await;
                }
                _ => return Err(e),
            },
        }
    }
}

async fn download_uv(platform: &Platform, dest: &Path) -> Result<()> {
    let url = format!(
        "https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{}",
        platform.uv_asset
    );
    eprintln!("Downloading uv {UV_VERSION}...");
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .build()?;
    let bytes = with_retries(&DOWNLOAD_RETRY_PAUSES, worth_retrying, || async {
        client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await
    })
    .await
    .with_context(|| format!("downloading {url}"))?;
    verify_sha256(&bytes, platform.uv_sha256).with_context(|| format!("verifying {url}"))?;
    let dir = dest.parent().context("uv path has no parent")?;
    std::fs::create_dir_all(dir)?;
    let tmp = tempfile::NamedTempFile::new_in(dir)?;
    extract_uv(&bytes, tmp.path())?;
    tmp.persist(dest)
        .map_err(|e| anyhow!("placing {}: {e}", dest.display()))?;
    Ok(())
}

/// A `uv` to install with: one already on the machine, else the pinned download.
async fn find_or_get_uv(platform: &Platform, root: &Path) -> Result<PathBuf> {
    if let Some(uv) = uv_on_path() {
        return Ok(uv);
    }
    let own = own_uv_path(root);
    if own.exists() {
        return Ok(own);
    }
    download_uv(platform, &own).await.with_context(|| {
        "could not download uv. Install uv yourself and re-run, or point at a working \
         Python with --python <path>"
    })?;
    Ok(own)
}

/// `UV_*` settings that keep working behind a company mirror or proxy. Everything
/// the lock's hashes protect against is safe to fetch from there. Any other
/// `UV_*` could change what is installed, so it isn't passed on.
const UV_ENV_ALLOW: &[&str] = &[
    "UV_DEFAULT_INDEX",
    "UV_INDEX",
    "UV_NATIVE_TLS",
    "UV_HTTP_TIMEOUT",
    "UV_PYTHON_INSTALL_MIRROR",
];

/// `uv` with Cascadia's own directories and none of the caller's settings.
fn uv_command(uv: &Path, root: &Path) -> Command {
    let mut cmd = Command::new(uv);
    for (key, _) in std::env::vars_os() {
        let key_str = key.to_string_lossy();
        if key_str.starts_with("UV_") && !UV_ENV_ALLOW.contains(&key_str.as_ref()) {
            cmd.env_remove(&key);
        }
    }
    scrub_python_env(&mut cmd);
    cmd.env_remove("VIRTUAL_ENV")
        .env_remove("CONDA_PREFIX")
        .env("UV_CACHE_DIR", root.join("uv-cache"))
        .env("UV_PYTHON_INSTALL_DIR", root.join("cpython"))
        .current_dir(root);
    cmd
}

fn run_uv(cmd: &mut Command, what: &str) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("running uv to {what}"))?;
    if !status.success() {
        bail!(
            "uv could not {what} ({status}). Setup needs PyPI, pytorch.org and GitHub; \
             without them, point at a working Python with --python <path>"
        );
    }
    Ok(())
}

/// Install `entries` into the environment, checked against the lock's hashes.
/// Keep the environment locked for as long as this `uv` runs, even if Cascadia is
/// killed: `uv` carries on writing, and a new setup has to wait for it. A lock lasts
/// until every handle to it is closed, and `uv` doesn't read stdin. Windows releases
/// the lock with Cascadia, but won't move the folder while `uv` has files open
/// there, so [`discard_dir`] waits instead.
fn hold_lock_for(cmd: &mut Command, env_lock: &std::fs::File) {
    #[cfg(unix)]
    if let Ok(held) = env_lock.try_clone() {
        cmd.stdin(held);
    }
    #[cfg(not(unix))]
    let _ = (cmd, env_lock);
}

fn install_packages(
    uv: &Path,
    root: &Path,
    python: &Path,
    entries: &[&LockEntry],
    env_lock: &std::fs::File,
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    for e in entries {
        file.write_all(e.block.as_bytes())?;
    }
    file.flush()?;
    let mut install = uv_command(uv, root);
    install
        .args(["pip", "install", "--no-config", "-r"])
        .arg(file.path())
        .arg("--python")
        .arg(python)
        .args([
            "--torch-backend",
            "cpu",
            "--require-hashes",
            "--no-deps",
            "--only-binary",
            ":all:",
        ]);
    hold_lock_for(&mut install, env_lock);
    run_uv(&mut install, "install the exporter's packages")
}

/// Refuse a setup that can't fit, rather than failing partway through.
fn check_free_space(root: &Path) -> Result<()> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let free = disks
        .list()
        .iter()
        .filter(|d| root.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space());
    match free {
        Some(free) if free < SETUP_FREE_BYTES => bail!(
            "setting up Python needs about {} GB free in {}, and only {:.1} GB is",
            SETUP_FREE_BYTES >> 30,
            root.display(),
            free as f64 / (1u64 << 30) as f64
        ),
        _ => Ok(()),
    }
}

/// Ask once, in a terminal. CI and scripts get an error naming the switch.
fn confirm_setup(question: &str, yes: bool) -> Result<bool> {
    if yes {
        return Ok(true);
    }
    if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
        bail!("{question}\nRe-run with --yes to install them, or --python <path> to use your own.");
    }
    eprint!("{question} [Y/n] ");
    std::io::stderr().flush().ok();
    let mut answer = String::new();
    let read = std::io::stdin().read_line(&mut answer)?;
    Ok(answer_is_yes(read, &answer))
}

/// Enter or a yes. Closed input isn't an answer, so nothing large downloads
/// because a terminal went away.
fn answer_is_yes(bytes_read: usize, answer: &str) -> bool {
    let a = answer.trim().to_ascii_lowercase();
    bytes_read > 0 && (a.is_empty() || a == "y" || a == "yes")
}

/// The one question setup asks. How it gets there (which `uv`, which Python) is
/// in the log line it prints when it finishes, not here.
fn setup_question(install: usize) -> String {
    format!(
        "Exporting needs {install} Python packages. Install them in Cascadia's own folder \
         (up to about 1.5 GB; nothing else on your system changes)?"
    )
}

/// Windows ships without the Visual C++ runtime torch and OpenVINO load.
fn missing_runtime_hint(failure: &str) -> Option<&'static str> {
    (failure.contains("WinError 126") || failure.contains("DLL load failed")).then_some(
        "This usually means the Microsoft Visual C++ Redistributable is missing. It needs \
         admin rights, so Cascadia doesn't install it: run \
         https://aka.ms/vs/17/release/vc_redist.x64.exe and try again.",
    )
}

/// Import everything the exporter needs and compare versions with the lock. uv
/// reporting success isn't enough: after a killed install, a re-sync exits 0 and
/// `pip check` passes on an environment that can't import.
fn verify_managed(python: &Path, lock: &[LockEntry]) -> Result<()> {
    let env = PythonEnv {
        path: python.to_string_lossy().into_owned(),
        deps: None,
        why: String::new(),
        managed: true,
        guard: None,
    };
    let imports = format!("{EXPORT_IMPORTS}; {}", crate::OPTIMUM_IMPORTS);
    if let Err(e) = run_with_timeout(env.command().args(run_code(&imports)), PROBE_TIMEOUT) {
        let last = e.lines().last().unwrap_or("");
        let hint = missing_runtime_hint(&e).unwrap_or("");
        bail!("the new environment can't import the exporter's packages: {last}\n{hint}");
    }
    let info = py_info(env.command()).map_err(|e| anyhow!(e))?;
    let wrong = packages_to_install(lock, &info.dists);
    if let Some(e) = wrong.first() {
        bail!(
            "the new environment doesn't hold {} {} ({} packages differ from the lock)",
            e.name,
            e.version,
            wrong.len()
        );
    }
    Ok(())
}

/// Build the environment from nothing. `layer` lets it see `base`'s packages, so
/// only the ones `base` lacks get installed.
fn build_env(
    uv: &Path,
    managed: &Managed,
    lock: &[LockEntry],
    base: Option<&(String, PyInfo)>,
    layer: bool,
    env_lock: &std::fs::File,
) -> Result<()> {
    discard_dir(&managed.dir)?;
    let mut venv = uv_command(uv, &managed.root);
    venv.args(["venv", "--no-config", "--no-project"])
        .arg(&managed.dir);
    match base {
        Some((p, _)) => {
            venv.arg("--python").arg(p);
            if layer {
                venv.arg("--system-site-packages");
            }
        }
        None => {
            venv.args(["--python", "3.12", "--managed-python"]);
        }
    }
    hold_lock_for(&mut venv, env_lock);
    run_uv(&mut venv, "create the environment")?;

    let none = HashMap::new();
    let seen = match base {
        Some((_, info)) if layer => &info.dists,
        _ => &none,
    };
    let install = packages_to_install(lock, seen);
    install_packages(uv, &managed.root, &managed.python(), &install, env_lock)?;
    verify_managed(&managed.python(), lock)?;
    std::fs::write(managed.dir.join(COMPLETE_MARKER), &managed.id)
        .context("marking the environment complete")?;
    let reused = lock.len() - install.len();
    match base.filter(|_| layer) {
        Some((p, _)) if reused > 0 => eprintln!(
            "Python packages ready: {} installed, {reused} reused from {p}.",
            install.len()
        ),
        _ => eprintln!("Python packages ready: {} installed.", install.len()),
    }
    Ok(())
}

/// Build the private environment if it is missing or damaged, asking first
/// unless `yes`. Holds the environment's lock: a second setup waits, and a killed
/// one leaves no marker, so the next one starts over. Re-syncing in place can't
/// repair it.
pub(crate) async fn setup_managed(yes: bool) -> Result<PythonEnv> {
    let Some(platform) = &PLATFORM else {
        bail!(
            "Cascadia can't set up Python on this platform: OpenVINO publishes no packages for \
             it. Point at a working Python with --python <path>."
        );
    };
    let managed = Managed::current()?.context("no private environment on this platform")?;
    let lock_file = managed.open_lock()?;
    if lock_file.try_lock().is_err() {
        eprintln!("Waiting for another Cascadia using the private environment...");
        lock_file
            .lock()
            .context("waiting for the environment lock")?;
    }
    let lock = parse_lock(platform.lock);
    let healthy = managed.is_complete() && probe_export_deps(managed.env(None).command()).is_some();
    if !healthy {
        let base = find_base_python(platform);
        let reused = base.as_ref().map_or(0, |(_, info)| {
            lock.len() - packages_to_install(&lock, &info.dists).len()
        });
        if !confirm_setup(&setup_question(lock.len() - reused), yes)? {
            bail!("Not installing. Use --python <path> to run with your own Python.");
        }
        check_free_space(&managed.root)?;
        let uv = find_or_get_uv(platform, &managed.root).await?;
        forget_old_packages(&uv, &managed, &lock);
        // Layer only when it saves something: otherwise the system's other
        // packages have no reason to be visible to the exporter.
        let layered = build_env(&uv, &managed, &lock, base.as_ref(), reused > 0, &lock_file);
        if let Err(e) = layered {
            if reused == 0 {
                return Err(e);
            }
            eprintln!(
                "warning: the environment layered on the system Python failed ({e:#}); \
                 rebuilding without it."
            );
            build_env(&uv, &managed, &lock, base.as_ref(), false, &lock_file)?;
        }
        remove_old_envs(&managed);
    }
    lock_file
        .unlock()
        .context("releasing the environment lock")?;
    let mut env = managed.env(None);
    env.deps = probe_export_deps(env.command());
    Ok(env)
}

/// Put back any package that no longer matches the lock, such as a reused system
/// package the user upgraded. Imports can pass on such an environment, so the
/// versions are what's compared; reading them takes milliseconds. Changing the
/// environment takes its lock, like a setup: not under a running export, and only
/// one repair, so the versions are read again once it's held.
async fn repair_managed(env: &PythonEnv) -> Result<()> {
    let (Some(platform), Some(managed)) = (&PLATFORM, Managed::current()?) else {
        return Ok(());
    };
    let lock = parse_lock(platform.lock);
    let info = py_info(env.command()).map_err(|e| anyhow!(e))?;
    if packages_to_install(&lock, &info.dists).is_empty() {
        return Ok(());
    }
    let lock_file = managed.open_lock()?;
    if lock_file.try_lock().is_err() {
        eprintln!("Waiting for another export to finish before restoring Python packages...");
        lock_file
            .lock()
            .context("waiting for the environment lock")?;
    }
    let info = py_info(env.command()).map_err(|e| anyhow!(e))?;
    let wrong = packages_to_install(&lock, &info.dists);
    if !wrong.is_empty() {
        let names: Vec<&str> = wrong.iter().map(|e| e.name.as_str()).collect();
        eprintln!(
            "Private Python environment: restoring {} package(s) that changed: {}",
            wrong.len(),
            names.join(", ")
        );
        let uv = find_or_get_uv(platform, &managed.root).await?;
        install_packages(&uv, &managed.root, &managed.python(), &wrong, &lock_file)?;
    }
    lock_file.unlock().context("releasing the environment lock")
}

/// The interpreter for `shard` or `export`: an existing one if it can do the job,
/// else the private environment, offering to build it first.
pub(crate) async fn python_for_export(
    explicit: Option<&str>,
    check_deps: bool,
    need: Need,
    yes: bool,
) -> Result<PythonEnv> {
    let found = resolve_python(explicit, check_deps, need);
    let usable = matches!(&found, Ok(env) if !check_deps || env.deps.is_some());
    let mut env = if usable || explicit.is_some() {
        found?
    } else {
        setup_managed(yes).await?
    };
    if env.managed {
        if check_deps {
            repair_managed(&env).await?;
        }
        if let Some(managed) = Managed::current()? {
            let guard = managed.open_lock()?;
            guard.lock_shared().context("locking the environment")?;
            env.guard = Some(guard);
        }
    }
    eprintln!("Using Python: {} ({})", env.path, env.why);
    Ok(env)
}

/// What a directory is renamed to while it waits to be deleted.
const TRASH_MARK: &str = ".trash-";

/// Delete `path`, retrying: just after a killed install, files can still be
/// appearing while the walk runs, and it fails with "Directory not empty".
fn remove_dir_with_retries(path: &Path) -> std::io::Result<()> {
    let mut last = Ok(());
    for attempt in 0..5 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(200));
        }
        match std::fs::remove_dir_all(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Ok(()) => return Ok(()),
            Err(e) => last = Err(e),
        }
    }
    last
}

/// How long to wait for whatever is holding a file in a directory to let go.
const RENAME_WAIT: Duration = Duration::from_secs(60);

/// Run `rename` until it works or `wait` is up, saying once that it's waiting.
fn rename_when_free(
    dir: &Path,
    wait: Duration,
    mut rename: impl FnMut() -> std::io::Result<()>,
) -> Result<()> {
    let start = std::time::Instant::now();
    let mut said = false;
    loop {
        let e = match rename() {
            Ok(()) => return Ok(()),
            Err(e) => e,
        };
        if start.elapsed() >= wait {
            if wait.is_zero() {
                return Err(e).with_context(|| format!("moving aside {}", dir.display()));
            }
            return Err(e).with_context(|| {
                format!(
                    "Something is still using {}, most likely an install that was cut off \
                     (uv.exe) or a virus scan. Wait for it to finish or end it, then try again.",
                    dir.display()
                )
            });
        }
        if !said {
            eprintln!(
                "Waiting for another program to let go of {}...",
                dir.display()
            );
            said = true;
        }
        std::thread::sleep(Duration::from_millis(500).min(wait));
    }
}

/// Get `dir` out of the way now, and delete it if that can be done. The rename
/// can't race anything writing into it, so a rebuild always starts clean;
/// anything the delete leaves behind goes with the next [`sweep_trash`].
fn discard_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let name = dir.file_name().unwrap_or_default().to_string_lossy();
    // Unique: an earlier discard's leftovers can still hold a name, even this
    // process's own when a layered build fails and is rebuilt.
    let pid = std::process::id();
    let trash = (0u32..)
        .map(|n| dir.with_file_name(format!("{name}{TRASH_MARK}{pid}-{n}")))
        .find(|p| !p.exists())
        .context("no free name to move the old environment to")?;
    // Only Windows refuses this: a cut-off install or a virus scan still holding
    // a file inside blocks the rename until it lets go.
    let wait = if cfg!(windows) {
        RENAME_WAIT
    } else {
        Duration::ZERO
    };
    rename_when_free(dir, wait, || std::fs::rename(dir, &trash))?;
    if let Err(e) = remove_dir_with_retries(&trash) {
        tracing::debug!(path = %trash.display(), error = %e, "left for the next setup");
    }
    Ok(())
}

/// Delete what an earlier [`discard_dir`] couldn't.
fn sweep_trash(root: &Path) {
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        if entry.file_name().to_string_lossy().contains(TRASH_MARK) {
            let _ = remove_dir_with_retries(&entry.path());
        }
    }
}

fn open_env_lock(root: &Path, name: &str) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join(format!("{name}.lock")))
}

/// Delete the environment `name` under `root`, unless an export is using it.
fn remove_env(root: &Path, name: &str) -> Result<bool> {
    let guard = open_env_lock(root, name)?;
    if guard.try_lock().is_err() {
        return Ok(false);
    }
    discard_dir(&root.join(name))?;
    drop(guard);
    std::fs::remove_file(root.join(format!("{name}.lock"))).ok();
    Ok(true)
}

fn env_names(root: &Path) -> Vec<String> {
    std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| is_env_dir_name(n))
        .collect()
}

/// A new lock leaves the previous environment behind: clear it once the new one
/// works. One in use is left for next time, along with the uv and Python it may
/// still need.
fn remove_old_envs(managed: &Managed) {
    sweep_trash(&managed.root);
    let old: Vec<String> = env_names(&managed.root)
        .into_iter()
        .filter(|name| *name != managed.id)
        .collect();
    let removed = old
        .iter()
        .filter(|name| matches!(remove_env(&managed.root, name), Ok(true)))
        .count();
    if !old.is_empty() && removed == old.len() {
        remove_old_tools(managed);
    }
}

/// Delete the uv and Python versions nothing here runs on any more.
fn remove_old_tools(managed: &Managed) {
    for entry in std::fs::read_dir(managed.root.join("uv"))
        .into_iter()
        .flatten()
        .flatten()
    {
        if entry.file_name() != UV_VERSION {
            let _ = remove_dir_with_retries(&entry.path());
        }
    }
    let Some(home) = python_home(&managed.dir) else {
        return;
    };
    for entry in std::fs::read_dir(managed.root.join("cpython"))
        .into_iter()
        .flatten()
        .flatten()
    {
        // uv's short names are links to the full one, so compare where they lead.
        let in_use = entry
            .path()
            .canonicalize()
            .is_ok_and(|p| home.starts_with(p));
        if !in_use && entry.file_name().to_string_lossy().starts_with("cpython-") {
            let _ = remove_dir_with_retries(&entry.path());
        }
    }
}

/// The Python installation a virtual environment runs on.
fn python_home(env_dir: &Path) -> Option<PathBuf> {
    std::fs::read_to_string(env_dir.join("pyvenv.cfg"))
        .ok()?
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == "home").then(|| PathBuf::from(value.trim()))
        })?
        .canonicalize()
        .ok()
}

/// What an environment has installed, from its `*.dist-info` folder names.
fn installed_in(env_dir: &Path) -> Vec<(String, String)> {
    let unix = std::fs::read_dir(env_dir.join("lib"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|version| version.path().join("site-packages"));
    let windows = std::iter::once(env_dir.join("Lib").join("site-packages"));
    unix.chain(windows)
        .flat_map(|site| std::fs::read_dir(site).into_iter().flatten().flatten())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let (dist, version) = name.strip_suffix(".dist-info")?.rsplit_once('-')?;
            Some((normalize_name(dist), version.to_string()))
        })
        .collect()
}

/// The packages in `installed` that the lock doesn't want at that version.
fn stale_packages(installed: &[(String, String)], lock: &[LockEntry]) -> Vec<String> {
    let mut stale: Vec<String> = installed
        .iter()
        .filter(|(name, version)| {
            !lock
                .iter()
                .any(|e| e.name == *name && same_release(&e.version, version))
        })
        .map(|(name, _)| name.clone())
        .collect();
    stale.sort();
    stale.dedup();
    stale
}

/// A new lock would otherwise leave every old package version in the download
/// cache for good. This runs before installing because uv clears all versions of
/// a package: what the new lock needs is then the only version cached. Installed
/// environments keep their files. If an older environment is still in use it
/// waits for next time, and if this fails the cost is only disk space.
fn forget_old_packages(uv: &Path, managed: &Managed, lock: &[LockEntry]) {
    let mut stale = Vec::new();
    for name in env_names(&managed.root) {
        if name == managed.id {
            continue;
        }
        let Ok(guard) = open_env_lock(&managed.root, &name) else {
            return;
        };
        if guard.try_lock().is_err() {
            return;
        }
        stale.extend(stale_packages(
            &installed_in(&managed.root.join(&name)),
            lock,
        ));
    }
    stale.sort();
    stale.dedup();
    if stale.is_empty() {
        return;
    }
    let mut cmd = uv_command(uv, &managed.root);
    cmd.args(["cache", "clean", "--no-config", "--quiet"])
        .args(&stale);
    if let Err(e) = run_with_timeout(&mut cmd, PROBE_TIMEOUT) {
        tracing::debug!(error = %e, "left old packages in the download cache");
    }
}

/// `cascadia python remove`: delete the private environment and everything setup
/// downloaded, keeping what a running export is using.
pub(crate) fn remove_managed() -> Result<()> {
    let root = managed_root()?;
    if !root.exists() {
        eprintln!("Nothing to remove.");
        return Ok(());
    }
    for name in env_names(&root) {
        if !remove_env(&root, &name)? {
            bail!(
                "An export is using {}; try again when it ends.",
                root.join(name).display()
            );
        }
    }
    remove_dir_with_retries(&root).with_context(|| format!("removing {}", root.display()))?;
    eprintln!("Removed {}", root.display());
    Ok(())
}

/// The name [`lock_id`] gives an environment's directory.
fn is_env_dir_name(name: &str) -> bool {
    name.len() == 12 && name.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The private environment's state, for `cascadia doctor`. Only looks.
pub(crate) fn managed_status() -> String {
    let Ok(root) = managed_root() else {
        return "unavailable: no user data directory".into();
    };
    let Ok(Some(managed)) = Managed::current() else {
        return "not available on this platform".into();
    };
    if managed.is_complete() {
        format!("ready at {}", managed.dir.display())
    } else if root.exists() {
        "unfinished (`cascadia python setup`)".to_string()
    } else {
        "not set up (only needed for `cascadia shard`; offered when needed)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: &str = "\
# Generated
# requirements-sha256: abc
Foo_Bar==1.2.3 \\
    --hash=sha256:aa \\
    --hash=sha256:bb
torch==2.15.0+cpu \\
    --hash=sha256:cc
";

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn only_enter_or_yes_agrees_to_install() {
        assert!(answer_is_yes(1, "\n"));
        assert!(answer_is_yes(2, "y\n"));
        assert!(answer_is_yes(5, " YES\r\n"));
        assert!(!answer_is_yes(2, "n\n"));
        assert!(!answer_is_yes(6, "maybe\n"));
        assert!(!answer_is_yes(0, ""), "closed input is not consent");
    }

    #[test]
    fn lock_splits_into_packages_keeping_their_hashes() {
        let lock = parse_lock(LOCK);
        assert_eq!(lock.len(), 2);
        assert_eq!(lock[0].name, "foo-bar");
        assert_eq!(lock[0].version, "1.2.3");
        assert_eq!(lock[0].block.matches("--hash").count(), 2);
        assert_eq!(lock[1].version, "2.15.0+cpu");
        assert!(lock[1].block.starts_with("torch==2.15.0+cpu"));
    }

    #[test]
    fn only_missing_or_different_packages_are_installed() {
        let lock = parse_lock(LOCK);
        let names = |have| -> Vec<String> {
            packages_to_install(&lock, &have)
                .iter()
                .map(|e| e.name.clone())
                .collect()
        };
        assert_eq!(names(map(&[])), ["foo-bar", "torch"], "nothing installed");
        assert_eq!(
            names(map(&[("foo-bar", "1.2.3"), ("torch", "2.15.0+cpu")])),
            Vec::<String>::new(),
            "exact versions are reused"
        );
        assert_eq!(
            names(map(&[("foo-bar", "1.2.4"), ("torch", "2.15.0")])),
            ["foo-bar"],
            "a wrong version is replaced; a local label doesn't make it wrong"
        );
        assert_eq!(
            names(map(&[("foo-bar", "1.2.3"), ("torch", "2.15.0+cu124")])),
            Vec::<String>::new(),
            "a GPU build of the same torch serves"
        );
    }

    fn info(minor: (u32, u32), venv: bool, bits: u32, platform: &str) -> PyInfo {
        PyInfo {
            minor,
            venv,
            free_threaded: false,
            bits,
            platform: platform.into(),
            executable: String::new(),
            dists: HashMap::new(),
        }
    }

    #[test]
    fn a_file_in_the_working_folder_is_not_imported() {
        let Some(python) = python_candidates().into_iter().find(|p| python_runs(p)) else {
            eprintln!("no Python here, skipping");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ran = dir.path().join("json.py.ran");
        std::fs::write(
            dir.path().join("json.py"),
            "open(__file__ + '.ran', 'w').close()\n",
        )
        .unwrap();
        let in_folder = || {
            let mut cmd = Command::new(&python);
            scrub_python_env(&mut cmd);
            cmd.current_dir(dir.path());
            cmd
        };
        let out = run_with_timeout(
            in_folder().args(run_code("import json; print('imported')")),
            PROBE_TIMEOUT,
        )
        .unwrap();
        assert_eq!(out.trim(), "imported");
        assert!(!ran.exists(), "the check imported the planted json.py");
        // The plant works: without run_code, Python does import it.
        in_folder().args(["-c", "import json"]).output().unwrap();
        assert!(ran.exists());
    }

    #[test]
    fn only_a_matching_system_python_is_a_base() {
        let p = "linux-x86_64";
        assert!(info((3, 12), false, 64, p).is_usable_base(p));
        assert!(info((3, 13), false, 64, p).is_usable_base(p));
        assert!(!info((3, 11), false, 64, p).is_usable_base(p), "too old");
        assert!(!info((3, 14), false, 64, p).is_usable_base(p), "refused");
        assert!(!info((3, 12), true, 64, p).is_usable_base(p), "a venv");
        assert!(!info((3, 12), false, 32, p).is_usable_base(p), "32-bit");
        assert!(!info((3, 12), false, 64, "linux-aarch64").is_usable_base(p));
        let mut t = info((3, 13), false, 64, p);
        t.free_threaded = true;
        assert!(!t.is_usable_base(p), "free-threaded");
    }

    #[test]
    fn versions_meet_or_miss_their_specifiers() {
        assert!(satisfies("5.5.4", ">=5.5,<5.6"));
        assert!(!satisfies("5.4.0", ">=5.5,<5.6"));
        assert!(!satisfies("5.6.0", ">=5.5,<5.6"));
        assert!(satisfies("2.15.0+cpu", ">=2.6"));
        assert!(satisfies(
            "2026.4.1-22982-e213a14-releases/2026/4",
            ">=2026.2"
        ));
        assert!(satisfies("5.5", ">=5.5.0"));
        assert!(!satisfies("2.6.0a0+git1", ">=2.0"), "pre-release");
        assert!(satisfies("2.6.0.post1", ">=2.6"));
        assert!(!satisfies("garbage", ">=1"));
        assert!(!satisfies("1.0", "~=1.0"), "unsupported operator");
    }

    #[test]
    fn missing_or_out_of_range_packages_fail_the_pin_check() {
        let reqs = [("torch", ">=2.6"), ("nncf", ">=2.18")];
        let ok = "chatter\ncascadia-probe torch 2.15.0\ncascadia-probe nncf 3.4.0\n";
        assert!(check_versions(ok, &reqs).is_ok());
        let old = "cascadia-probe torch 2.1.0\ncascadia-probe nncf 3.4.0\n";
        assert!(check_versions(old, &reqs)
            .unwrap_err()
            .contains("torch 2.1.0"));
        let gone = "cascadia-probe torch 2.15.0\n";
        assert!(check_versions(gone, &reqs)
            .unwrap_err()
            .contains("nncf missing"));
    }

    #[test]
    fn one_probe_answers_both_questions() {
        // A shard-only environment: no nncf or optimum-intel.
        let shard_only = "cascadia-probe python 3.12.1\n\
                          cascadia-probe torch 2.15.0\n\
                          cascadia-missing nncf\n\
                          cascadia-missing optimum-intel\n\
                          cascadia-probe nncf-lookalike 1\n";
        assert!(
            core_imports_ok(shard_only),
            "the optional ones can be missing"
        );
        let reqs = [("nncf", ">=2.18")];
        assert!(check_versions(shard_only, &reqs)
            .unwrap_err()
            .contains("nncf missing"));
        assert!(!core_imports_ok("cascadia-missing torch\n"));
        assert!(!core_imports_ok("cascadia-missing huggingface_hub\n"));
        assert_eq!(
            pin_summary("cascadia-probe python 3.12.1\ncascadia-probe torch 2.15.0\ncascadia-probe nncf 3.4.0\n"),
            "python 3.12.1\ntorch 2.15.0\n",
            "the same lines as the import-only probe"
        );
    }

    #[test]
    fn uv_on_path_must_be_new_enough() {
        assert!(uv_is_usable("uv 0.13.7 (abc 2026-01-01)"));
        assert!(uv_is_usable("uv 0.10.4"));
        assert!(!uv_is_usable("uv 0.9.5"));
        assert!(!uv_is_usable("something else"));
    }

    #[test]
    fn a_tampered_uv_download_is_refused() {
        let want = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        assert!(verify_sha256(b"hello", want).is_ok());
        assert!(verify_sha256(b"hello!", want).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn only_the_uv_binary_comes_out_of_the_archive() {
        let mut archive = Vec::new();
        {
            let gz = flate2::write::GzEncoder::new(&mut archive, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            for (path, body) in [("uv-x/uvx", &b"not this"[..]), ("uv-x/uv", &b"this"[..])] {
                let mut h = tar::Header::new_gnu();
                h.set_size(body.len() as u64);
                h.set_mode(0o644);
                h.set_cksum();
                tar.append_data(&mut h, path, body).unwrap();
            }
            tar.into_inner().unwrap().finish().unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("uv");
        extract_uv(&archive, &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"this");
        use std::os::unix::fs::PermissionsExt;
        assert_ne!(
            std::fs::metadata(&dest).unwrap().permissions().mode() & 0o111,
            0
        );
        assert!(extract_uv(b"not an archive", &dest).is_err());
    }

    #[test]
    fn discarding_clears_the_path_at_once_and_sweeping_clears_leftovers() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("abc");
        std::fs::create_dir_all(dir.join("lib/deep")).unwrap();
        std::fs::write(dir.join("lib/deep/f"), "x").unwrap();
        discard_dir(&dir).unwrap();
        assert!(!dir.exists());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        discard_dir(&dir).expect("nothing to discard is fine");

        let left = root.path().join(format!("abc{TRASH_MARK}1"));
        std::fs::create_dir_all(left.join("x")).unwrap();
        std::fs::create_dir_all(root.path().join("keep")).unwrap();
        sweep_trash(root.path());
        assert!(!left.exists());
        assert!(root.path().join("keep").exists());
    }

    #[test]
    fn discarding_never_collides_with_earlier_leftovers() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("abc");
        std::fs::create_dir_all(&dir).unwrap();
        // What a delete that failed leaves: this process's first trash name, full.
        let held = root
            .path()
            .join(format!("abc{TRASH_MARK}{}-0", std::process::id()));
        std::fs::create_dir_all(held.join("locked")).unwrap();
        discard_dir(&dir).expect("should pick another name");
        assert!(!dir.exists());
        assert!(held.join("locked").exists(), "the leftover is untouched");
    }

    #[test]
    fn a_held_directory_is_waited_for_then_named() {
        let denied = || std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        let dir = Path::new("env");

        // Lets go on the third try: no error.
        let mut tries = 0;
        rename_when_free(dir, Duration::from_secs(30), || {
            tries += 1;
            if tries < 3 {
                Err(denied())
            } else {
                Ok(())
            }
        })
        .expect("should wait for it");
        assert_eq!(tries, 3);

        // Never lets go: stops, and says what to do.
        let msg = format!(
            "{:#}",
            rename_when_free(dir, Duration::from_millis(1), || Err(denied())).unwrap_err()
        );
        assert!(msg.contains("still using env"), "{msg}");
        assert!(msg.contains("then try again"), "{msg}");

        // Not waiting at all: one try and the plain error.
        let mut tries = 0;
        let msg = format!(
            "{:#}",
            rename_when_free(dir, Duration::ZERO, || {
                tries += 1;
                Err(denied())
            })
            .unwrap_err()
        );
        assert_eq!(tries, 1);
        assert!(msg.contains("moving aside env"), "{msg}");
    }

    #[tokio::test]
    async fn downloads_are_tried_again_only_when_that_can_help() {
        use std::cell::Cell;
        let pauses = [Duration::ZERO; 3];
        let tries = Cell::new(0);

        // A passing failure: works on the third try.
        let got = with_retries(
            &pauses,
            |_: &String| true,
            || async {
                tries.set(tries.get() + 1);
                if tries.get() < 3 {
                    Err("504".to_string())
                } else {
                    Ok(7)
                }
            },
        )
        .await;
        assert_eq!(got, Ok(7));
        assert_eq!(tries.get(), 3);

        // It never recovers: stops when the pauses run out, with the last error.
        tries.set(0);
        let got: Result<u8, String> = with_retries(
            &pauses,
            |_| true,
            || async {
                tries.set(tries.get() + 1);
                Err("504".to_string())
            },
        )
        .await;
        assert_eq!(got, Err("504".to_string()));
        assert_eq!(tries.get(), 4);

        // Trying again can't help: one try.
        tries.set(0);
        let got: Result<u8, String> = with_retries(
            &pauses,
            |_| false,
            || async {
                tries.set(tries.get() + 1);
                Err("404".to_string())
            },
        )
        .await;
        assert_eq!(got, Err("404".to_string()));
        assert_eq!(tries.get(), 1);
    }

    /// What keeps a new setup from racing a `uv` that outlived a killed Cascadia.
    #[cfg(unix)]
    #[test]
    fn a_child_keeps_the_environment_locked_after_cascadia_lets_go() {
        let root = tempfile::tempdir().unwrap();
        let open = || open_env_lock(root.path(), "abc").unwrap();
        let cascadia = open();
        cascadia.try_lock().unwrap();

        let mut sleep = Command::new("sleep");
        sleep.arg("30");
        hold_lock_for(&mut sleep, &cascadia);
        let mut uv = sleep.spawn().unwrap();
        // Cascadia is gone: only the child's copy of the handle is left.
        drop(sleep);
        drop(cascadia);

        assert!(open().try_lock().is_err(), "still locked while it runs");
        uv.kill().unwrap();
        uv.wait().unwrap();
        // Not instantly: a child another test is starting holds a copy of every
        // open file until it execs.
        let deadline = Instant::now() + Duration::from_secs(10);
        while open().try_lock().is_err() {
            assert!(Instant::now() < deadline, "free once it has exited");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn only_what_the_lock_no_longer_wants_is_stale() {
        let lock = parse_lock(LOCK);
        let wanted = &lock[0];
        let installed = vec![
            (wanted.name.clone(), wanted.version.clone()),
            (wanted.name.clone(), format!("{}+cpu", wanted.version)),
            ("left-the-lock".to_string(), "1.0".to_string()),
            ("left-the-lock".to_string(), "1.0".to_string()),
        ];
        assert_eq!(stale_packages(&installed, &lock), ["left-the-lock"]);

        let moved = vec![(wanted.name.clone(), "0.0.0-older".to_string())];
        assert_eq!(
            stale_packages(&moved, &lock),
            std::slice::from_ref(&wanted.name)
        );
    }

    #[test]
    fn installed_packages_are_read_on_both_layouts() {
        let root = tempfile::tempdir().unwrap();
        let unix = root.path().join("unix/lib/python3.12/site-packages");
        std::fs::create_dir_all(unix.join("typing_extensions-4.1.dist-info")).unwrap();
        std::fs::create_dir_all(unix.join("typing_extensions")).unwrap();
        let windows = root.path().join("win/Lib/site-packages");
        std::fs::create_dir_all(windows.join("torch-2.15.0+cpu.dist-info")).unwrap();
        assert_eq!(
            installed_in(&root.path().join("unix")),
            [("typing-extensions".to_string(), "4.1".to_string())]
        );
        assert_eq!(
            installed_in(&root.path().join("win")),
            [("torch".to_string(), "2.15.0+cpu".to_string())]
        );
    }

    #[test]
    fn old_uv_and_python_go_and_the_ones_in_use_stay() {
        let root = tempfile::tempdir().unwrap();
        let managed = Managed {
            root: root.path().to_path_buf(),
            dir: root.path().join("abc"),
            id: "abc".to_string(),
        };
        let current = root.path().join("cpython/cpython-3.12.2-x");
        let old = root.path().join("cpython/cpython-3.12.1-x");
        let hidden = root.path().join("cpython/.temp");
        let uv_now = root.path().join("uv").join(UV_VERSION);
        let uv_old = root.path().join("uv/0.0.1");
        for dir in [&managed.dir, &current, &old, &hidden, &uv_now, &uv_old] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(
            managed.dir.join("pyvenv.cfg"),
            format!("home = {}\nversion = 3.12\n", current.join("bin").display()),
        )
        .unwrap();
        std::fs::create_dir_all(current.join("bin")).unwrap();

        remove_old_tools(&managed);

        assert!(current.exists(), "the Python in use stays");
        assert!(hidden.exists(), "uv's own files stay");
        assert!(!old.exists());
        assert!(uv_now.exists());
        assert!(!uv_old.exists());
    }

    #[test]
    fn only_lock_ids_count_as_environments() {
        assert!(is_env_dir_name(&lock_id(LOCK)));
        assert!(!is_env_dir_name("uv"));
        assert!(!is_env_dir_name("uv-cache"));
        assert!(!is_env_dir_name("cpython"));
        assert!(!is_env_dir_name(&format!("{}{TRASH_MARK}7", lock_id(LOCK))));
    }

    #[test]
    fn the_lock_id_ignores_line_endings() {
        assert_eq!(lock_id("a\nb\n"), lock_id("a\r\nb\r\n"));
        assert_ne!(lock_id("a\n"), lock_id("b\n"));
    }

    #[test]
    fn a_missing_runtime_gets_the_installer_link() {
        assert!(missing_runtime_hint("OSError: [WinError 126] module")
            .unwrap()
            .contains("vc_redist"));
        assert!(missing_runtime_hint("ImportError: DLL load failed while importing x").is_some());
        assert!(missing_runtime_hint("ModuleNotFoundError: yaml").is_none());
    }

    /// Reads a file from the repo root, for the checks on the committed locks.
    fn repo_file(path: &str) -> String {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .replace('\r', "")
    }

    const LOCK_FILES: [&str; 3] = [
        "tools/export-locks/linux-x86_64.txt",
        "tools/export-locks/windows-x86_64.txt",
        "tools/export-locks/macos-arm64.txt",
    ];

    /// Fails until `scripts/update-export-locks.sh` is re-run after a change to
    /// the requirements, so the private environment can't drift from them.
    #[test]
    fn locks_match_the_requirements_they_came_from() {
        let inputs =
            repo_file("tools/requirements.txt") + &repo_file("tools/requirements-managed.txt");
        let stamp: String = Sha256::digest(inputs.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        for path in LOCK_FILES {
            assert!(
                repo_file(path).contains(&format!("# requirements-sha256: {stamp}\n")),
                "{path} is stale: run scripts/update-export-locks.sh"
            );
        }
    }

    /// The exporter writes IRs and tokenizers the bundled runtime then loads.
    #[test]
    fn locks_use_the_openvino_release_bundles() {
        let release = repo_file(".github/workflows/release.yml");
        let bundled = release
            .lines()
            .find_map(|l| l.trim().strip_prefix("OV_VERSION:"))
            .expect("OV_VERSION in release.yml")
            .trim()
            .trim_matches('"')
            .to_string();
        for path in LOCK_FILES {
            let lock = parse_lock(&repo_file(path));
            let openvino = lock.iter().find(|e| e.name == "openvino").expect(path);
            let base = bundled.trim_end_matches(".0");
            assert_eq!(openvino.version, base, "{path} vs OV_VERSION {bundled}");
            let tokenizers = lock
                .iter()
                .find(|e| e.name == "openvino-tokenizers")
                .expect(path);
            assert!(tokenizers.version.starts_with(base), "{path}: tokenizers");
        }
    }

    /// Every lock entry names its version exactly and carries a hash, and every
    /// requirement is in the lock within its range.
    #[test]
    fn locks_are_complete_and_hashed() {
        for path in LOCK_FILES {
            let lock = parse_lock(&repo_file(path));
            assert!(lock.len() > 40, "{path}: {} packages", lock.len());
            for e in &lock {
                assert!(e.block.contains("--hash=sha256:"), "{path}: {}", e.name);
            }
            for (dist, spec) in export_requirements() {
                let e = lock
                    .iter()
                    .find(|e| e.name == normalize_name(dist))
                    .unwrap_or_else(|| panic!("{path} lacks {dist}"));
                assert!(satisfies(&e.version, spec), "{path}: {dist} {}", e.version);
            }
        }
    }
}
