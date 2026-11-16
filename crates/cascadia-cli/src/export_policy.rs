//! `cascadia export-policy`: a LeRobot SmolVLA or π0.5 checkpoint to a `policy_package`.
//!
//! Python runs only here, in a uv environment under the Cascadia cache built from the hashed
//! requirements embedded below. Every refusal comes before the checkpoint is downloaded: the host
//! checks (uv, memory, a serving process) before any environment exists, and the metadata checks
//! (gated tokenizer, statistics) from a small environment holding only `huggingface_hub`.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;

/// The driver the CLI runs in the export environment.
const DRIVER: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/export_policy/export_policy.py"
));
/// The export environment, compiled with hashes by `tools/export_policy/lock.sh`.
pub(crate) const REQUIREMENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/export_policy/requirements.txt"
));
/// `huggingface_hub` alone, at the same versions, for the metadata checks.
pub(crate) const PREFLIGHT_REQUIREMENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/export_policy/requirements-preflight.txt"
));

const PYTHON_VERSION: &str = "3.12";
/// torch and torchvision come from here as CPU wheels; every wheel is checked against its hash.
const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
const SUPPORTED: [&str; 2] = ["smolvla", "pi05"];
const GIB: u64 = 1 << 30;
/// Available RAM plus free swap an export needs: π0.5 peaked at 14.3 GB RSS plus up to 10 GB swap.
const MIN_AVAILABLE: u64 = 32 * GIB;
/// Physical RAM floor; a 16 GiB VM reports about 15.6 GiB.
const MIN_PHYSICAL: u64 = 15 * GIB;
const GEMMA_TERMS: &str = "https://ai.google.dev/gemma/terms";
const GEMMA_PROHIBITED_USE: &str = "https://ai.google.dev/gemma/prohibited_use_policy";

/// Export a LeRobot SmolVLA or π0.5 checkpoint from Hugging Face to an OpenVINO `policy_package`
/// for `cascadia run --engine ov-policy`.
#[derive(Parser, Debug, Clone)]
pub struct ExportPolicyArgs {
    /// Hugging Face model id of the checkpoint, e.g. `lerobot/smolvla_libero`.
    pub model: String,
    /// Package directory to write; must not exist or be empty.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Export the real-time chunking ports (SmolVLA and π0.5): the package serves the whole chunk.
    #[arg(long)]
    pub rtc: bool,
    /// Normalization statistics for a checkpoint that ships none: a JSON file
    /// `{"state": {...}, "action": {...}}`, each with `mean`/`std`, `q01`/`q99` or `min`/`max`.
    #[arg(long, value_name = "FILE")]
    pub stats: Option<PathBuf>,
    /// Run the traced input through PyTorch and the package's IR (CPU f32, and GPU f16 when a
    /// GPU is present) and print the max and mean differences.
    #[arg(long)]
    pub verify: bool,
}

/// What the host checks look at.
#[derive(Debug, Clone, Default)]
pub(crate) struct HostFacts {
    pub uv: Option<PathBuf>,
    pub git: Option<PathBuf>,
    pub total_ram: u64,
    pub available_ram: u64,
    pub free_swap: u64,
    pub processes: Vec<ProcessFacts>,
    pub own_pid: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct ProcessFacts {
    pub pid: u32,
    pub name: String,
    pub args: Vec<String>,
}

/// The driver's `preflight` verdict.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct Preflight {
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub license_files: Vec<String>,
    #[serde(default)]
    pub policy_type: Option<String>,
    #[serde(default)]
    pub gated: Option<GatedCheck>,
    #[serde(default)]
    pub stats: Option<StatsCheck>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GatedCheck {
    pub repo: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StatsCheck {
    pub ok: bool,
    pub detail: String,
}

/// The steps that touch Python, so the order of checks is testable with a fake.
pub(crate) trait ExportHost {
    fn facts(&mut self) -> HostFacts;
    /// The small environment for the metadata checks.
    fn prepare_preflight(&mut self) -> Result<()>;
    fn preflight(&mut self, model: &str) -> Result<Preflight>;
    /// The full export environment.
    fn prepare_export(&mut self) -> Result<()>;
    fn export(&mut self, args: &ExportPolicyArgs, preflight: &Preflight) -> Result<()>;
}

pub(crate) fn uv_install_hint() -> String {
    if cfg!(windows) {
        r#"powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex""#
            .into()
    } else {
        "curl -LsSf https://astral.sh/uv/install.sh | sh".into()
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// The first host-level reason to refuse, in the design's order.
pub(crate) fn host_refusal(f: &HostFacts) -> Option<String> {
    if f.uv.is_none() {
        return Some(format!(
            "export-policy needs uv, which is not on PATH. It builds its own Python environment \
             with it. Install uv:\n  {}\nthen open a new shell and run this again.",
            uv_install_hint()
        ));
    }
    if f.git.is_none() {
        return Some(
            "export-policy needs git, which is not on PATH: two of its packages \
             (physical-ai-studio and the physicalai runtime) are pinned to git commits, and uv \
             fetches them with git."
                .into(),
        );
    }
    let room = f.available_ram.saturating_add(f.free_swap);
    if room < MIN_AVAILABLE {
        return Some(format!(
            "not enough memory to export: {} of available RAM plus free swap, and an export needs \
             32 GiB (a π0.5 export peaked at 14.3 GB resident plus 10 GB of swap; a busy machine \
             can freeze). Close other work, add swap, or export on a disposable Linux machine.",
            gib(room)
        ));
    }
    if f.total_ram < MIN_PHYSICAL {
        return Some(format!(
            "not enough memory to export: this host reports {} of physical RAM, and an export \
             needs a machine with at least 16 GiB (15 GiB as the OS reports it), swap or not.",
            gib(f.total_ram)
        ));
    }
    serving_process(&f.processes, f.own_pid)
}

/// A process that serves models on this host: a cascadia node, OVMS, or `cascadia run`/`worker`.
pub(crate) fn serving_process(procs: &[ProcessFacts], own_pid: u32) -> Option<String> {
    let found = procs.iter().filter(|p| p.pid != own_pid).find_map(|p| {
        let lower = p.name.to_ascii_lowercase();
        let name = lower.strip_suffix(".exe").unwrap_or(&lower);
        match name {
            "ovms" => Some(format!("OVMS (pid {})", p.pid)),
            "cascadia-node" => Some(format!("a cascadia node (pid {})", p.pid)),
            "cascadia" => p
                .args
                .get(1)
                .filter(|a| matches!(a.as_str(), "run" | "worker" | "run-placement"))
                .map(|a| format!("`cascadia {a}` (pid {})", p.pid)),
            _ => None,
        }
    })?;
    Some(format!(
        "{found} is serving on this host. An export takes most of the machine's memory for many \
         minutes and would starve it. Export on a disposable Linux machine instead (x86-64, \
         32 GiB of RAM plus swap) and copy the package over."
    ))
}

/// Local argument checks that need no network.
pub(crate) fn check_inputs(args: &ExportPolicyArgs) -> Result<()> {
    if args.out.exists() {
        let mut entries = std::fs::read_dir(&args.out)
            .with_context(|| format!("--out {}: not a readable directory", args.out.display()))?;
        if entries.next().is_some() {
            return Err(anyhow!(
                "--out {} is not empty; pick a new directory",
                args.out.display()
            ));
        }
    }
    if let Some(path) = &args.stats {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("--stats {}", path.display()))?;
        let doc: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("--stats {}: not JSON", path.display()))?;
        let has = |k: &str| doc.get(k).is_some_and(serde_json::Value::is_object);
        if !((has("state") || has("observation.state")) && has("action")) {
            return Err(anyhow!(
                "--stats {}: expected {{\"state\": {{...}}, \"action\": {{...}}}}, each with \
                 mean/std, q01/q99 or min/max",
                path.display()
            ));
        }
    }
    Ok(())
}

/// The reason to refuse after the metadata checks, or None to export.
pub(crate) fn preflight_refusal(p: &Preflight, model: &str, stats_given: bool) -> Option<String> {
    if let Some(e) = &p.error {
        return Some(e.clone());
    }
    match p.policy_type.as_deref() {
        Some(t) if SUPPORTED.contains(&t) => {}
        Some(t) => {
            return Some(format!(
                "{model} is a `{t}` policy; export-policy exports smolvla and pi05 checkpoints \
                 (what physical-ai-studio 8f132416 exports)"
            ))
        }
        None => return Some(format!("{model}: config.json names no policy type")),
    }
    if let Some(g) = p.gated.as_ref().filter(|g| !g.ok) {
        return Some(format!(
            "{model} needs the tokenizer of {repo}, a gated repository under the Gemma licence \
             ({detail}). To get access: sign in at https://huggingface.co, open \
             https://huggingface.co/{repo} and accept the Gemma licence there, then log in on \
             this machine with `hf auth login` (or set HF_TOKEN to a token of that account) and \
             run this again. The Gemma Terms of Use: {GEMMA_TERMS}",
            repo = g.repo,
            detail = g.detail,
        ));
    }
    let stats_ok = p.stats.as_ref().is_some_and(|s| s.ok);
    if !stats_ok && !stats_given {
        let detail = p
            .stats
            .as_ref()
            .map_or("they were not checked", |s| s.detail.as_str());
        return Some(format!(
            "{model} ships no usable normalization statistics: {detail}. An export without them \
             gets identity placeholders, which `cascadia run` refuses. Pass them with --stats \
             <file>: {{\"state\": {{...}}, \"action\": {{...}}}}, each with mean/std, q01/q99 or \
             min/max."
        ));
    }
    None
}

/// Whether the Gemma Terms of Use apply to the package.
pub(crate) fn gemma_terms_apply(p: &Preflight) -> bool {
    p.policy_type.as_deref() == Some("pi05")
        || p.license
            .as_deref()
            .is_some_and(|l| l.to_ascii_lowercase().contains("gemma"))
}

/// The package's `NOTICE`, which the engine serves as the metadata's `attribution`.
pub(crate) fn notice_text(p: &Preflight, model: &str) -> String {
    let mut n = format!(
        "This policy package was exported by `cascadia export-policy` from the Hugging Face \
         checkpoint {model} (revision {}) with physical-ai-studio 8f132416 and the physicalai \
         runtime e3dd944.\n\nLicence of the checkpoint, from its model card: {}.\n",
        p.revision.as_deref().unwrap_or("unknown"),
        p.license.as_deref().unwrap_or("not stated"),
    );
    if !p.license_files.is_empty() {
        n.push_str(&format!(
            "The checkpoint's licence files are included: {}.\n",
            p.license_files.join(", ")
        ));
    }
    if gemma_terms_apply(p) {
        n.push_str(&format!(
            "\nThis model is built on PaliGemma, a Gemma model. Its use is subject to the Gemma \
             Terms of Use ({GEMMA_TERMS}) and the Gemma Prohibited Use Policy \
             ({GEMMA_PROHIBITED_USE}).\nGemma is provided under and subject to the Gemma Terms \
             of Use found at ai.google.dev/gemma/terms.\n"
        ));
    }
    n
}

/// One line per set of terms, printed before the export starts and after it ends.
pub(crate) fn terms_summary(p: &Preflight) -> Vec<String> {
    let mut lines = vec![format!(
        "checkpoint licence (model card): {}",
        p.license.as_deref().unwrap_or("not stated")
    )];
    if !p.license_files.is_empty() {
        lines.push(format!(
            "licence files copied into the package: {}",
            p.license_files.join(", ")
        ));
    }
    if gemma_terms_apply(p) {
        lines.push(format!(
            "the Gemma Terms of Use apply (PaliGemma backbone): {GEMMA_TERMS}, prohibited uses: \
             {GEMMA_PROHIBITED_USE}"
        ));
    }
    lines
}

fn print_terms(p: &Preflight) {
    eprintln!("Terms that apply to this package (also in its NOTICE):");
    for l in terms_summary(p) {
        eprintln!("  {l}");
    }
}

/// Runs the export against `host`; every refusal comes before `prepare_export`.
pub(crate) fn run_export(args: &ExportPolicyArgs, host: &mut dyn ExportHost) -> Result<()> {
    if let Some(why) = host_refusal(&host.facts()) {
        return Err(anyhow!(why));
    }
    check_inputs(args)?;
    host.prepare_preflight()?;
    let pre = host.preflight(&args.model)?;
    if let Some(why) = preflight_refusal(&pre, &args.model, args.stats.is_some()) {
        return Err(anyhow!(why));
    }
    print_terms(&pre);
    host.prepare_export()?;
    host.export(args, &pre)?;
    std::fs::write(args.out.join("NOTICE"), notice_text(&pre, &args.model))
        .with_context(|| format!("writing {}", args.out.join("NOTICE").display()))?;
    eprintln!("\nPackage written to {}.", args.out.display());
    print_terms(&pre);
    eprintln!(
        "Serve it with:\n  cascadia run {} --engine ov-policy",
        args.out.display()
    );
    Ok(())
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let file = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|d| d.join(&file))
        .find(|p| p.is_file())
}

/// FNV-1a: a stable id for an environment's requirements, so a new pin set gets a new venv.
fn requirements_id(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn venv_python(dir: &Path) -> PathBuf {
    if cfg!(windows) {
        dir.join("Scripts").join("python.exe")
    } else {
        dir.join("bin").join("python")
    }
}

/// The real host: uv environments under the Cascadia cache, the driver in a temp dir.
struct UvHost {
    uv: PathBuf,
    root: PathBuf,
    work: tempfile::TempDir,
    preflight_python: Option<PathBuf>,
    export_python: Option<PathBuf>,
}

impl UvHost {
    fn new() -> Result<Self> {
        let root = dirs::cache_dir()
            .ok_or_else(|| anyhow!("no cache directory on this platform"))?
            .join("cascadia")
            .join("export-policy");
        let work = tempfile::Builder::new()
            .prefix("cascadia-export-policy-")
            .tempdir()
            .context("creating a temp dir for the export driver")?;
        std::fs::write(work.path().join("export_policy.py"), DRIVER)
            .context("writing the embedded export driver")?;
        Ok(Self {
            uv: find_on_path("uv").unwrap_or_else(|| "uv".into()),
            root,
            work,
            preflight_python: None,
            export_python: None,
        })
    }

    fn uv(&self, args: &[&std::ffi::OsStr]) -> Result<()> {
        let status = std::process::Command::new(&self.uv)
            .args(args)
            .stdin(std::process::Stdio::null())
            .status()
            .with_context(|| format!("running {}", self.uv.display()))?;
        if !status.success() {
            return Err(anyhow!("uv exited with {status}; see its output above"));
        }
        Ok(())
    }

    /// The environment for `requirements`, built once and reused while the pins stay the same.
    fn ensure_env(&self, kind: &str, requirements: &str, torch_index: bool) -> Result<PathBuf> {
        let id = requirements_id(requirements);
        let dir = self.root.join(format!("{kind}-{id}"));
        let python = venv_python(&dir);
        let stamp = dir.join(".cascadia-ready");
        if stamp.is_file() && python.is_file() {
            return Ok(python);
        }
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("creating {}", self.root.display()))?;
        let reqs = self.root.join(format!("{kind}-{id}.txt"));
        std::fs::write(&reqs, requirements)
            .with_context(|| format!("writing {}", reqs.display()))?;
        eprintln!("Preparing the {kind} environment in {}", dir.display());
        if !python.is_file() {
            self.uv(&[
                "venv".as_ref(),
                "--python".as_ref(),
                PYTHON_VERSION.as_ref(),
                dir.as_os_str(),
            ])?;
        }
        let mut sync: Vec<&std::ffi::OsStr> = vec![
            "pip".as_ref(),
            "sync".as_ref(),
            "--python".as_ref(),
            python.as_os_str(),
            "--verify-hashes".as_ref(),
        ];
        if torch_index {
            // Exact pins plus hashes, so taking each wheel from whichever index has it is safe;
            // first-index matching would look for every package on the torch index only.
            sync.extend::<[&std::ffi::OsStr; 4]>([
                "--index-strategy".as_ref(),
                "unsafe-best-match".as_ref(),
                "--extra-index-url".as_ref(),
                TORCH_CPU_INDEX.as_ref(),
            ]);
        }
        sync.push(reqs.as_os_str());
        self.uv(&sync)?;
        std::fs::write(&stamp, &id).with_context(|| format!("writing {}", stamp.display()))?;
        Ok(python)
    }

    fn driver(&self, python: &Path) -> std::process::Command {
        let mut cmd = std::process::Command::new(python);
        cmd.arg("-I")
            .arg("-u")
            .arg(self.work.path().join("export_policy.py"))
            .stdin(std::process::Stdio::null());
        cmd
    }

    fn preflight_file(&self) -> PathBuf {
        self.work.path().join("preflight.json")
    }
}

impl ExportHost for UvHost {
    fn facts(&mut self) -> HostFacts {
        use sysinfo::{
            MemoryRefreshKind, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System,
            UpdateKind,
        };
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        );
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_cmd(UpdateKind::Always),
        );
        let processes = sys
            .processes()
            .iter()
            .map(|(pid, p)| ProcessFacts {
                pid: pid.as_u32(),
                name: p.name().to_string_lossy().into_owned(),
                args: p
                    .cmd()
                    .iter()
                    .map(|a| a.to_string_lossy().into_owned())
                    .collect(),
            })
            .collect();
        HostFacts {
            uv: find_on_path("uv"),
            git: find_on_path("git"),
            total_ram: sys.total_memory(),
            available_ram: sys.available_memory(),
            free_swap: sys.free_swap(),
            processes,
            own_pid: std::process::id(),
        }
    }

    fn prepare_preflight(&mut self) -> Result<()> {
        self.preflight_python =
            Some(self.ensure_env("preflight", PREFLIGHT_REQUIREMENTS, false)?);
        Ok(())
    }

    fn preflight(&mut self, model: &str) -> Result<Preflight> {
        let python = self
            .preflight_python
            .clone()
            .ok_or_else(|| anyhow!("preflight environment not prepared"))?;
        let result = self.preflight_file();
        let status = self
            .driver(&python)
            .args(["preflight", "--model", model, "--result"])
            .arg(&result)
            .status()
            .context("running the export-policy preflight")?;
        if !status.success() {
            return Err(anyhow!(
                "the preflight exited with {status}; see its output above"
            ));
        }
        let text = std::fs::read_to_string(&result).context("reading the preflight result")?;
        serde_json::from_str(&text).context("parsing the preflight result")
    }

    fn prepare_export(&mut self) -> Result<()> {
        self.export_python = Some(self.ensure_env("env", REQUIREMENTS, true)?);
        Ok(())
    }

    fn export(&mut self, args: &ExportPolicyArgs, _preflight: &Preflight) -> Result<()> {
        let python = self
            .export_python
            .clone()
            .ok_or_else(|| anyhow!("export environment not prepared"))?;
        let mut cmd = self.driver(&python);
        cmd.args(["export", "--model", &args.model, "--preflight"])
            .arg(self.preflight_file())
            .arg("--out")
            .arg(&args.out);
        if args.rtc {
            cmd.arg("--rtc");
        }
        if let Some(stats) = &args.stats {
            cmd.arg("--stats").arg(stats);
        }
        if args.verify {
            cmd.arg("--verify");
        }
        let status = cmd.status().context("running the export driver")?;
        if !status.success() {
            return Err(anyhow!(
                "the export exited with {status}; see its output above"
            ));
        }
        Ok(())
    }
}

pub fn cmd_export_policy(args: ExportPolicyArgs) -> Result<()> {
    run_export(&args, &mut UvHost::new()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> std::result::Result<crate::Cli, clap::Error> {
        use clap::Parser as _;
        crate::Cli::try_parse_from(argv)
    }

    fn args(model: &str, out: &Path) -> ExportPolicyArgs {
        ExportPolicyArgs {
            model: model.into(),
            out: out.into(),
            rtc: false,
            stats: None,
            verify: false,
        }
    }

    fn roomy() -> HostFacts {
        HostFacts {
            uv: Some("/usr/bin/uv".into()),
            git: Some("/usr/bin/git".into()),
            total_ram: 16 * GIB,
            available_ram: 10 * GIB,
            free_swap: 48 * GIB,
            processes: vec![],
            own_pid: 42,
        }
    }

    fn proc(pid: u32, name: &str, args: &[&str]) -> ProcessFacts {
        ProcessFacts {
            pid,
            name: name.into(),
            args: args.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn passing(kind: &str) -> Preflight {
        Preflight {
            policy_type: Some(kind.into()),
            revision: Some("a".repeat(40)),
            license: Some("apache-2.0".into()),
            stats: Some(StatsCheck {
                ok: true,
                detail: "present".into(),
            }),
            gated: (kind == "pi05").then(|| GatedCheck {
                repo: "google/paligemma-3b-pt-224".into(),
                ok: true,
                detail: "access granted".into(),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn parses_every_flag() {
        let cli = parse(&[
            "cascadia",
            "export-policy",
            "lerobot/smolvla_libero",
            "--out",
            "pkg",
            "--rtc",
            "--stats",
            "s.json",
            "--verify",
        ])
        .unwrap();
        let crate::Command::ExportPolicy(a) = cli.cmd else {
            panic!("not export-policy")
        };
        assert_eq!(a.model, "lerobot/smolvla_libero");
        assert_eq!(a.out, PathBuf::from("pkg"));
        assert!(a.rtc && a.verify);
        assert_eq!(a.stats, Some(PathBuf::from("s.json")));
    }

    #[test]
    fn defaults_and_required_out() {
        let cli = parse(&["cascadia", "export-policy", "m/x", "--out", "p"]).unwrap();
        let crate::Command::ExportPolicy(a) = cli.cmd else {
            panic!("not export-policy")
        };
        assert!(!a.rtc && !a.verify && a.stats.is_none());
        assert!(parse(&["cascadia", "export-policy", "m/x"]).is_err());
        assert!(parse(&["cascadia", "export-policy", "--out", "p"]).is_err());
    }

    #[test]
    fn refuses_without_uv_and_names_the_install_command() {
        let f = HostFacts {
            uv: None,
            ..roomy()
        };
        let msg = host_refusal(&f).unwrap();
        assert!(msg.contains("uv"), "{msg}");
        assert!(msg.contains(&uv_install_hint()), "{msg}");
        assert!(uv_install_hint().contains("astral.sh/uv/install"));
    }

    #[test]
    fn uv_is_checked_before_anything_else() {
        let f = HostFacts {
            uv: None,
            total_ram: GIB,
            available_ram: 0,
            free_swap: 0,
            processes: vec![proc(7, "ovms", &[])],
            ..roomy()
        };
        assert!(host_refusal(&f).unwrap().contains("uv"));
    }

    #[test]
    fn refuses_without_git() {
        let f = HostFacts {
            git: None,
            ..roomy()
        };
        assert!(host_refusal(&f).unwrap().contains("git"));
    }

    #[test]
    fn refuses_low_available_memory_plus_swap() {
        let f = HostFacts {
            total_ram: 64 * GIB,
            available_ram: 20 * GIB,
            free_swap: 11 * GIB,
            ..roomy()
        };
        let msg = host_refusal(&f).unwrap();
        assert!(msg.contains("31.0 GiB") && msg.contains("32 GiB"), "{msg}");
        // exactly 32 GiB passes
        let f = HostFacts {
            free_swap: 12 * GIB,
            ..f
        };
        assert_eq!(host_refusal(&f), None);
    }

    #[test]
    fn refuses_low_physical_memory_even_with_swap() {
        // an 8 GiB box with 64 GiB swap
        let f = HostFacts {
            total_ram: 8 * GIB,
            available_ram: 6 * GIB,
            free_swap: 64 * GIB,
            ..roomy()
        };
        let msg = host_refusal(&f).unwrap();
        assert!(msg.contains("physical"), "{msg}");
        // a 16 GiB VM reports about 15.6 GiB and passes
        let f = HostFacts {
            total_ram: 15 * GIB + 600 * (1 << 20),
            ..roomy()
        };
        assert_eq!(host_refusal(&f), None);
    }

    #[test]
    fn refuses_on_a_serving_host() {
        for p in [
            proc(9, "ovms", &["ovms", "--config_path", "c.json"]),
            proc(9, "OVMS.exe", &[]),
            proc(9, "cascadia-node.exe", &[]),
            proc(
                9,
                "cascadia",
                &["cascadia", "run", "pkg", "--engine", "ov-policy"],
            ),
            proc(
                9,
                "cascadia.exe",
                &["cascadia.exe", "worker", "--rank", "0"],
            ),
        ] {
            let f = HostFacts {
                processes: vec![proc(1, "bash", &[]), p.clone()],
                ..roomy()
            };
            let msg = host_refusal(&f).unwrap_or_else(|| panic!("{p:?} not refused"));
            assert!(msg.contains("pid 9"), "{msg}");
            assert!(msg.contains("disposable Linux machine"), "{msg}");
        }
    }

    #[test]
    fn ignores_itself_and_non_serving_cascadia_commands() {
        let procs = vec![
            proc(42, "cascadia", &["cascadia", "export-policy", "m/x"]),
            proc(43, "cascadia", &["cascadia", "doctor"]),
            proc(44, "cascadia", &["cascadia", "worker"]),
        ];
        assert_eq!(serving_process(&procs[..2], 42), None);
        // the same command line under another pid is a serving worker
        assert!(serving_process(&procs, 42).unwrap().contains("pid 44"));
        // our own pid never counts, whatever it runs
        assert_eq!(
            serving_process(&[proc(42, "cascadia", &["cascadia", "run"])], 42),
            None
        );
    }

    #[test]
    fn input_checks() {
        let dir = tempfile::tempdir().unwrap();
        // missing and empty output dirs are fine
        check_inputs(&args("m/x", &dir.path().join("new"))).unwrap();
        check_inputs(&args("m/x", dir.path())).unwrap();
        std::fs::write(dir.path().join("f"), "x").unwrap();
        let err = check_inputs(&args("m/x", dir.path()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not empty"), "{err}");

        let out = dir.path().join("pkg");
        let mut a = args("m/x", &out);
        a.stats = Some(dir.path().join("missing.json"));
        assert!(check_inputs(&a)
            .unwrap_err()
            .to_string()
            .contains("--stats"));
        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, "[1, 2]").unwrap();
        a.stats = Some(bad);
        assert!(check_inputs(&a)
            .unwrap_err()
            .to_string()
            .contains("\"state\""));
        let good = dir.path().join("good.json");
        std::fs::write(
            &good,
            r#"{"state": {"mean": [0], "std": [1]}, "action": {"q01": [0], "q99": [1]}}"#,
        )
        .unwrap();
        a.stats = Some(good);
        check_inputs(&a).unwrap();
    }

    #[test]
    fn refuses_a_gated_tokenizer_with_the_gemma_licence_steps() {
        let mut p = passing("pi05");
        p.gated.as_mut().unwrap().ok = false;
        p.gated.as_mut().unwrap().detail = "gated".into();
        let msg = preflight_refusal(&p, "lerobot/pi05_libero_finetuned_v044", false).unwrap();
        for want in [
            "google/paligemma-3b-pt-224",
            "https://huggingface.co/google/paligemma-3b-pt-224",
            "Gemma",
            "hf auth login",
            "HF_TOKEN",
        ] {
            assert!(msg.contains(want), "missing {want:?} in {msg}");
        }
    }

    #[test]
    fn refuses_missing_stats_unless_given() {
        let mut p = passing("pi05");
        p.stats = Some(StatsCheck {
            ok: false,
            detail: "policy_preprocessor.json: normalizer_processor lists no STATE feature".into(),
        });
        let msg = preflight_refusal(&p, "lerobot/pi05_droid", false).unwrap();
        assert!(msg.contains("normalization statistics"), "{msg}");
        assert!(msg.contains("--stats"), "{msg}");
        assert!(msg.contains("lists no STATE feature"), "{msg}");
        assert_eq!(preflight_refusal(&p, "lerobot/pi05_droid", true), None);
    }

    #[test]
    fn refuses_unsupported_kinds_and_driver_errors() {
        let mut p = passing("smolvla");
        p.policy_type = Some("act".into());
        let msg = preflight_refusal(&p, "lerobot/act_x", false).unwrap();
        assert!(
            msg.contains("act") && msg.contains("smolvla") && msg.contains("pi05"),
            "{msg}"
        );
        let p = Preflight {
            error: Some("nope/x: no such model".into()),
            ..Default::default()
        };
        assert!(preflight_refusal(&p, "nope/x", false)
            .unwrap()
            .contains("no such model"));
        assert_eq!(preflight_refusal(&passing("smolvla"), "m/x", false), None);
        assert_eq!(preflight_refusal(&passing("pi05"), "m/x", false), None);
    }

    #[test]
    fn notice_carries_the_licence_and_gemma_terms_for_pi05() {
        let mut p = passing("pi05");
        p.license = Some("gemma".into());
        let n = notice_text(&p, "lerobot/pi05_libero_finetuned_v044");
        assert!(gemma_terms_apply(&p));
        for want in [
            "lerobot/pi05_libero_finetuned_v044",
            &"a".repeat(40),
            "8f132416",
            "e3dd944",
            "gemma",
            "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms",
            GEMMA_PROHIBITED_USE,
        ] {
            assert!(n.contains(want), "missing {want:?} in {n}");
        }
        let t = terms_summary(&p).join("\n");
        assert!(
            t.contains("Gemma Terms of Use") && t.contains(GEMMA_TERMS),
            "{t}"
        );
    }

    #[test]
    fn notice_for_smolvla_has_no_gemma_terms() {
        let mut p = passing("smolvla");
        p.license_files = vec!["LICENSE".into()];
        let n = notice_text(&p, "lerobot/smolvla_libero");
        assert!(!gemma_terms_apply(&p));
        assert!(n.contains("apache-2.0") && n.contains("LICENSE"), "{n}");
        assert!(!n.contains("Gemma"), "{n}");
        let t = terms_summary(&p).join("\n");
        assert!(t.contains("apache-2.0") && !t.contains("Gemma"), "{t}");
        // an unstated licence is said so, not guessed
        p.license = None;
        assert!(notice_text(&p, "m/x").contains("not stated"));
    }

    #[test]
    fn embedded_driver_and_hashed_requirements_are_present() {
        assert!(DRIVER.contains("def preflight(") && DRIVER.contains("def export("));
        assert!(DRIVER.contains("compress_to_fp16=True") && DRIVER.contains("dtype=\"float32\""));
        for (name, text) in [
            ("requirements.txt", REQUIREMENTS),
            ("requirements-preflight.txt", PREFLIGHT_REQUIREMENTS),
        ] {
            let joined = text.replace("\\\n", " ");
            let reqs: Vec<&str> = joined
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();
            assert!(!reqs.is_empty(), "{name} is empty");
            for r in reqs {
                if r.contains(" @ git+") {
                    // git sources cannot be hashed; they are pinned by full commit
                    let rev = r.rsplit('@').next().unwrap();
                    let rev = rev.split('#').next().unwrap();
                    assert!(
                        rev.len() == 40 && rev.chars().all(|c| c.is_ascii_hexdigit()),
                        "{name}: {r} is not pinned to a commit"
                    );
                } else {
                    assert!(
                        r.contains("==") && r.contains("--hash=sha256:"),
                        "{name}: {r}"
                    );
                }
            }
        }
        for want in [
            "physicalai-train @ git+https://github.com/open-edge-platform/physical-ai-studio@8f1324161ece2b24f2d38765b6e943ed1961258c#subdirectory=library",
            "physicalai @ git+https://github.com/openvinotoolkit/physicalai@e3dd9445a188f99c30afd92077b22a927c16b2bb",
            "\nopenvino==2026.4.1 ",
            "\nopenvino-tokenizers==2026.4.1.0 ",
            "\ntorch==2.11.0+cpu ",
        ] {
            assert!(REQUIREMENTS.contains(want), "requirements.txt lacks {want:?}");
        }
        let hub = REQUIREMENTS
            .lines()
            .find(|l| l.starts_with("huggingface-hub=="))
            .unwrap();
        assert!(
            PREFLIGHT_REQUIREMENTS.contains(hub),
            "preflight hub differs: {hub}"
        );
    }

    /// Records the calls, so a test sees what ran before a refusal.
    struct FakeHost {
        facts: HostFacts,
        preflight: Preflight,
        calls: Vec<&'static str>,
    }

    impl ExportHost for FakeHost {
        fn facts(&mut self) -> HostFacts {
            self.calls.push("facts");
            self.facts.clone()
        }
        fn prepare_preflight(&mut self) -> Result<()> {
            self.calls.push("prepare_preflight");
            Ok(())
        }
        fn preflight(&mut self, _model: &str) -> Result<Preflight> {
            self.calls.push("preflight");
            Ok(self.preflight.clone())
        }
        fn prepare_export(&mut self) -> Result<()> {
            self.calls.push("prepare_export");
            Ok(())
        }
        fn export(&mut self, a: &ExportPolicyArgs, _p: &Preflight) -> Result<()> {
            self.calls.push("export");
            std::fs::create_dir_all(&a.out)?;
            std::fs::write(a.out.join("manifest.json"), "{}")?;
            Ok(())
        }
    }

    fn run_fake(
        facts: HostFacts,
        pre: Preflight,
        a: &ExportPolicyArgs,
    ) -> (Result<()>, Vec<&'static str>) {
        let mut host = FakeHost {
            facts,
            preflight: pre,
            calls: vec![],
        };
        let r = run_export(a, &mut host);
        (r, host.calls)
    }

    #[test]
    fn host_refusals_stop_before_any_environment() {
        let dir = tempfile::tempdir().unwrap();
        let a = args("lerobot/smolvla_libero", &dir.path().join("pkg"));
        for facts in [
            HostFacts {
                uv: None,
                ..roomy()
            },
            HostFacts {
                available_ram: GIB,
                free_swap: 0,
                ..roomy()
            },
            HostFacts {
                processes: vec![proc(5, "ovms", &[])],
                ..roomy()
            },
        ] {
            let (r, calls) = run_fake(facts, passing("smolvla"), &a);
            assert!(r.is_err());
            assert_eq!(calls, ["facts"]);
        }
    }

    #[test]
    fn metadata_refusals_stop_before_the_export_environment() {
        let dir = tempfile::tempdir().unwrap();
        let a = args("lerobot/pi05_droid", &dir.path().join("pkg"));
        let mut gated = passing("pi05");
        gated.gated.as_mut().unwrap().ok = false;
        let mut no_stats = passing("pi05");
        no_stats.stats.as_mut().unwrap().ok = false;
        for pre in [gated, no_stats] {
            let (r, calls) = run_fake(roomy(), pre, &a);
            assert!(r.is_err());
            assert_eq!(calls, ["facts", "prepare_preflight", "preflight"]);
            assert!(!a.out.exists());
        }
    }

    #[test]
    fn a_passing_run_exports_and_writes_the_notice() {
        let dir = tempfile::tempdir().unwrap();
        let a = args(
            "lerobot/pi05_libero_finetuned_v044",
            &dir.path().join("pkg"),
        );
        let (r, calls) = run_fake(roomy(), passing("pi05"), &a);
        r.unwrap();
        assert_eq!(
            calls,
            [
                "facts",
                "prepare_preflight",
                "preflight",
                "prepare_export",
                "export"
            ]
        );
        let notice = std::fs::read_to_string(a.out.join("NOTICE")).unwrap();
        assert!(notice.contains("Gemma Terms of Use"), "{notice}");
    }
}
