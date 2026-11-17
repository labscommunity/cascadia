//! The host checks the two exporters share: memory headroom and a process serving models.
//! `export-policy` refuses on them, `shard` warns.

use crate::export_policy::{HostFacts, ProcessFacts};

pub(crate) const GIB: u64 = 1 << 30;

pub(crate) fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// This host's memory and processes; the tool fields of [`HostFacts`] stay unset.
pub(crate) fn read_host() -> HostFacts {
    use sysinfo::{
        MemoryRefreshKind, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, UpdateKind,
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
        total_ram: sys.total_memory(),
        available_ram: sys.available_memory(),
        free_swap: sys.free_swap(),
        processes,
        own_pid: std::process::id(),
        ..HostFacts::default()
    }
}

/// Available RAM plus free swap.
pub(crate) fn headroom(f: &HostFacts) -> u64 {
    f.available_ram.saturating_add(f.free_swap)
}

/// The first process other than this one that serves models: OVMS, a cascadia node, or a
/// `cascadia run` / `worker` / `run-placement`, as "OVMS (pid 7)".
pub(crate) fn serving_process(procs: &[ProcessFacts], own_pid: u32) -> Option<String> {
    procs.iter().filter(|p| p.pid != own_pid).find_map(|p| {
        let lower = p.name.to_ascii_lowercase();
        let name = lower.strip_suffix(".exe").unwrap_or(&lower);
        match name {
            "ovms" => Some(format!("OVMS (pid {})", p.pid)),
            "cascadia-node" => Some(format!("a cascadia node (pid {})", p.pid)),
            "cascadia" => subcommand(&p.args)
                .filter(|a| matches!(*a, "run" | "worker" | "run-placement"))
                .map(|a| format!("`cascadia {a}` (pid {})", p.pid)),
            _ => None,
        }
    })
}

/// The subcommand of a cascadia command line: the first word after the global flags.
fn subcommand(args: &[String]) -> Option<&str> {
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == "--log-level" {
            it.next();
        } else if !a.starts_with('-') {
            return Some(a);
        }
    }
    None
}

/// Free memory under which `cascadia shard` warns: the export loads the whole model in torch,
/// at 2 bytes a parameter in fp16 (16 GB for an 8B model) before any quantization.
pub(crate) const SHARD_WARN_HEADROOM: u64 = 16 * GIB;

/// `cascadia shard`'s warnings from the shared checks; it exports anyway.
pub(crate) fn shard_warnings(f: &HostFacts) -> Vec<String> {
    let mut out = Vec::new();
    if headroom(f) < SHARD_WARN_HEADROOM {
        out.push(format!(
            "only {} of available RAM plus free swap: the export loads the whole model in torch \
             (2 bytes a parameter at fp16, 16 GB for an 8B model) and may swap hard or be killed",
            gib(headroom(f))
        ));
    }
    if let Some(found) = serving_process(&f.processes, f.own_pid) {
        out.push(format!(
            "{found} is serving on this host; the export takes much of the machine's memory and \
             can starve it"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proc(pid: u32, name: &str, args: &[&str]) -> ProcessFacts {
        ProcessFacts {
            pid,
            name: name.into(),
            args: args.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn shard_warns_on_low_memory_and_a_serving_process() {
        let roomy = HostFacts {
            total_ram: 64 * GIB,
            available_ram: 40 * GIB,
            processes: vec![proc(1, "bash", &[])],
            own_pid: 2,
            ..HostFacts::default()
        };
        assert!(shard_warnings(&roomy).is_empty());
        let tight = HostFacts {
            available_ram: 6 * GIB,
            free_swap: 2 * GIB,
            processes: vec![proc(3, "cascadia", &["cascadia", "run", "m"])],
            ..roomy
        };
        let w = shard_warnings(&tight);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("8.0 GiB"), "{w:?}");
        assert!(w[1].contains("`cascadia run` (pid 3)"), "{w:?}");
    }

    /// `serve` is not a cascadia subcommand, so it never marks a serving process.
    #[test]
    fn serve_is_not_a_subcommand() {
        assert_eq!(
            serving_process(&[proc(9, "cascadia", &["cascadia", "serve"])], 1),
            None
        );
    }
}
