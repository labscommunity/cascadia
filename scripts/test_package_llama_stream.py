"""Linux ELF packaging regressions (no Intel SDK or GPU required).

    python3 -m unittest scripts/test_package_llama_stream.py

Requires a C compiler, binutils, patchelf, and the standard Linux loader tools.
The fixtures really link and dlopen a SYCL-style adapter with a transitive
UMF dependency; packaging and relocated execution use the actual ELF tools.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("package-llama-stream.sh")
TOOLS = ("cc", "readelf", "patchelf", "ldd", "tar")


@unittest.skipUnless(
    sys.platform.startswith("linux") and all(shutil.which(tool) for tool in TOOLS),
    "Linux packaging tests require cc, readelf, patchelf, ldd, and tar",
)
class PackageLlamaStream(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cascadia-package-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "build-tree"
        self.bin = self.build / "build" / "bin"
        self.bin.mkdir(parents=True)
        self.runtime = self.root / "oneapi" / "compiler" / "selected" / "lib"
        self.runtime.mkdir(parents=True)
        self.umf_dir = self.root / "oneapi" / "umf" / "selected" / "lib"
        self.umf_dir.mkdir(parents=True)
        self.archive = self.root / "bundle.tar.gz"
        self.env = os.environ.copy()
        self.env.pop("LD_PRELOAD", None)
        self.env["LD_LIBRARY_PATH"] = f"{self.runtime}:{self.umf_dir}"
        self.compile(
            self.umf_dir / "libumf.so.1.2",
            "int umf_value(void) { return 42; }",
            "-shared", "-fPIC", "-Wl,-soname,libumf.so.1",
        )
        (self.umf_dir / "libumf.so.1").symlink_to("libumf.so.1.2")
        (self.umf_dir / "libumf.so").symlink_to("libumf.so.1")
        self.compile(
            self.runtime / "libur_adapter_level_zero.so.0.12",
            "extern int umf_value(void); int adapter_value(void) { return umf_value(); }",
            "-shared", "-fPIC", "-Wl,-soname,libur_adapter_level_zero.so.0",
            f"-L{self.umf_dir}", "-lumf",
        )
        (self.runtime / "libur_adapter_level_zero.so.0").symlink_to(
            "libur_adapter_level_zero.so.0.12"
        )
        (self.runtime / "libur_adapter_level_zero.so").symlink_to(
            "libur_adapter_level_zero.so.0"
        )
        self.compile_sycl("libur_adapter_level_zero.so.0")
        self.compile_server()

    def compile(self, output, source, *flags):
        subprocess.run(
            ["cc", "-x", "c", "-", "-o", str(output), *flags],
            input=source, text=True, capture_output=True, check=True,
        )

    def compile_sycl(self, plugin):
        self.compile(
            self.runtime / "libsycl.so.9",
            r'''
#include <dlfcn.h>
int sycl_value(void) {
    void *adapter = dlopen("PLUGIN", RTLD_NOW);
    if (!adapter) return -1;
    int (*value)(void) = (int (*)(void))dlsym(adapter, "adapter_value");
    return value ? value() : -2;
}
'''.replace("PLUGIN", plugin),
            "-shared", "-fPIC", "-Wl,-soname,libsycl.so.9", "-ldl",
        )
        if not (self.runtime / "libsycl.so").exists():
            (self.runtime / "libsycl.so").symlink_to("libsycl.so.9")

    def compile_server(self, require_environment=False):
        condition = '!getenv("LD_LIBRARY_PATH") || ' if require_environment else ""
        self.compile(
            self.bin / "llama-server",
            '#include <stdio.h>\n#include <stdlib.h>\n'
            'extern int sycl_value(void);\n'
            f'int main(void) {{ if ({condition}sycl_value() != 42) return 73; '
            'puts("fixture server 42"); return 0; }\n',
            f"-L{self.runtime}", "-lsycl", f"-Wl,-rpath,{self.runtime}",
        )

    def package(self, *options):
        return subprocess.run(
            ["bash", str(SCRIPT), str(self.build), str(self.archive), *options],
            env=self.env, text=True, capture_output=True,
        )

    def link_server_against_missing(self, soname):
        """Rebuild the server with a DT_NEEDED on soname, then delete that library."""
        stub = self.root / "stub"
        stub.mkdir(exist_ok=True)
        name = soname.split(".so")[0][len("lib"):]
        self.compile(stub / soname, "int stub_value(void) { return 1; }",
                     "-shared", "-fPIC", f"-Wl,-soname,{soname}")
        (stub / f"lib{name}.so").symlink_to(soname)
        self.compile(
            self.bin / "llama-server",
            "extern int sycl_value(void); extern int stub_value(void);\n"
            "int main(void) { return sycl_value() + stub_value() == 43 ? 0 : 73; }\n",
            f"-L{self.runtime}", "-lsycl", f"-Wl,-rpath,{self.runtime}",
            f"-L{stub}", f"-l{name}",
        )
        (stub / soname).unlink()
        ldd = subprocess.run(["ldd", str(self.bin / "llama-server")], env=self.env,
                             text=True, capture_output=True).stdout
        self.assertIn(f"{soname} => not found", ldd)

    def archive_names(self):
        with tarfile.open(self.archive) as archive:
            return {Path(name).name for name in archive.getnames()}

    def unpack_and_remove_sources(self):
        destination = self.root / "relocated"
        destination.mkdir()
        with tarfile.open(self.archive) as archive:
            archive.extractall(destination)
        shutil.rmtree(self.build)
        shutil.rmtree(self.root / "oneapi")
        return destination / "cascadia-llama-sycl"

    def run_relocated(self, bundle):
        env = self.env.copy()
        env.pop("LD_LIBRARY_PATH", None)
        return subprocess.run(
            [str(bundle / "llama-server"), "--version"],
            env=env, text=True, capture_output=True,
        )

    def test_plugin_transitive_dependency_and_sonames_survive_relocation(self):
        # A second compiler installation must never be swept into this bundle.
        other = self.root / "oneapi" / "compiler" / "unselected" / "lib"
        other.mkdir(parents=True)
        (other / "libur_adapter_unrelated.so").write_text("not the active runtime")
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        bundle = self.unpack_and_remove_sources()
        self.assertTrue((bundle / "libumf.so.1").is_symlink())
        self.assertEqual((bundle / "libumf.so.1").readlink(), Path("libumf.so.1.2"))
        self.assertTrue((bundle / "libur_adapter_level_zero.so").is_symlink())
        self.assertFalse((bundle / "libur_adapter_unrelated.so").exists())
        relocated = self.run_relocated(bundle)
        self.assertEqual(relocated.returncode, 0, relocated.stdout + relocated.stderr)
        self.assertIn("fixture server 42", relocated.stdout)
        self.assertIn("fixture server 42", (bundle / "VERSION.txt").read_text())

    def test_missing_plugin_transitive_dependency_aborts_packaging(self):
        shutil.rmtree(self.umf_dir)
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("libumf.so.1", result.stderr)
        self.assertIn("unresolved dependency", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_legacy_pi_plugin_is_included_with_its_dependencies(self):
        self.compile(
            self.runtime / "libpi_level_zero.so",
            "extern int umf_value(void); int adapter_value(void) { return umf_value(); }",
            "-shared", "-fPIC", "-Wl,-soname,libpi_level_zero.so",
            f"-L{self.umf_dir}", "-lumf",
        )
        for path in self.runtime.glob("libur_adapter_level_zero.so*"):
            path.unlink()
        self.compile_sycl("libpi_level_zero.so")
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        bundle = self.unpack_and_remove_sources()
        self.assertTrue((bundle / "libpi_level_zero.so").is_file())
        result = subprocess.run(
            ["ldd", str(bundle / "libpi_level_zero.so")],
            env={key: value for key, value in self.env.items() if key != "LD_LIBRARY_PATH"},
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(bundle / "libumf.so.1"), result.stdout)
        self.assertNotIn("not found", result.stdout)
        relocated = self.run_relocated(bundle)
        self.assertEqual(relocated.returncode, 0, relocated.stdout + relocated.stderr)

    def test_relocated_version_failure_aborts_packaging(self):
        # The build executable succeeds in its build environment, but not with
        # the clean loader environment required of the staged executable.
        self.compile_server(require_environment=True)
        original = subprocess.run(
            [str(self.bin / "llama-server"), "--version"], env=self.env,
            text=True, capture_output=True,
        )
        self.assertEqual(original.returncode, 0, original.stdout + original.stderr)
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("relocated llama-server --version failed", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_patchelf_failure_is_fatal(self):
        tools = self.root / "tools"
        tools.mkdir()
        patchelf = tools / "patchelf"
        patchelf.write_text("#!/bin/sh\necho forced-patchelf-failure >&2\nexit 19\n")
        patchelf.chmod(0o755)
        self.env["PATH"] = f"{tools}:{self.env['PATH']}"
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forced-patchelf-failure", result.stderr)
        self.assertIn("cannot relocate", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_ineffective_patching_cannot_pass_using_build_rpath(self):
        tools = self.root / "tools"
        tools.mkdir()
        patchelf = tools / "patchelf"
        patchelf.write_text("#!/bin/sh\nexit 0\n")
        patchelf.chmod(0o755)
        self.env["PATH"] = f"{tools}:{self.env['PATH']}"
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside bundle", result.stderr)
        self.assertFalse(self.archive.exists())

    # A GPU-less release runner: the SYCL runtime aborts before --version, so
    # the run is skipped while every relocation check still applies.
    def test_no_run_check_packages_without_running_the_server(self):
        self.compile(self.bin / "llama-server", "int main(void) { return 73; }")
        self.assertNotEqual(self.package().returncode, 0)
        result = self.package("--no-run-check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("run check skipped", result.stdout)
        self.assertNotIn("VERSION.txt", self.archive_names())

    def test_missing_host_library_is_tolerated_and_not_bundled(self):
        # The runner has no GPU driver; the destination provides the loader.
        self.link_server_against_missing("libze_loader.so.1")
        result = self.package("--no-run-check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("libze_loader.so.1", self.archive_names())

    def test_missing_non_host_library_still_aborts_packaging(self):
        self.link_server_against_missing("libnothost.so.1")
        result = self.package("--no-run-check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unresolved dependency", result.stderr)
        self.assertIn("libnothost.so.1", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_unknown_option_is_rejected(self):
        result = self.package("--bogus")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown option: --bogus", result.stderr)
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
