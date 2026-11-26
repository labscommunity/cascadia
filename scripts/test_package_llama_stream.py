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
        self.tcm_dir = self.root / "oneapi" / "tcm" / "selected" / "lib"
        self.compiler_docs = self.runtime.parent / "share" / "doc" / "compiler"
        self.umf_docs = self.umf_dir.parent / "share" / "doc" / "umf"
        for path in (
            self.build / "LICENSE",
            self.build / "licenses" / "LICENSE-jsonhpp",
            self.build / "vendor" / "cpp-httplib" / "LICENSE",
            self.compiler_docs / "licensing" / "c" / "third-party-programs.txt",
            self.umf_docs / "LICENSE.TXT",
        ):
            self.write_text(path)

    def write_text(self, path, text="license text\n"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

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

    def archive_licenses(self):
        prefix = "cascadia-llama-sycl/licenses/"
        with tarfile.open(self.archive) as archive:
            return {member.name[len(prefix):] for member in archive.getmembers()
                    if member.isfile() and member.name.startswith(prefix)}

    def link_server_against(self, *libraries):
        """Rebuild the server with an extra DT_NEEDED on each (directory, soname)."""
        flags, calls = [], []
        for index, (library_dir, soname) in enumerate(libraries):
            library_dir.mkdir(parents=True, exist_ok=True)
            name = soname.split(".so")[0][len("lib"):]
            self.compile(library_dir / soname, f"int extra_value{index}(void) {{ return 1; }}",
                         "-shared", "-fPIC", f"-Wl,-soname,{soname}")
            if not (library_dir / f"lib{name}.so").exists():
                (library_dir / f"lib{name}.so").symlink_to(soname)
            flags += [f"-L{library_dir}", f"-l{name}", f"-Wl,-rpath,{library_dir}"]
            calls.append(index)
        self.compile(
            self.bin / "llama-server",
            "#include <stdio.h>\nextern int sycl_value(void);\n"
            + "".join(f"extern int extra_value{i}(void);\n" for i in calls)
            + "int main(void) { if (sycl_value()"
            + "".join(f" + extra_value{i}()" for i in calls)
            + f' != {42 + len(calls)}) return 73; puts("fixture server 42"); return 0; }}\n',
            f"-L{self.runtime}", "-lsycl", f"-Wl,-rpath,{self.runtime}", *flags,
        )

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

    def test_licenses_are_exactly_the_required_texts(self):
        # Decoys: notices of components not used directly, the OpenMP notice
        # with no OpenMP runtime bundled, and an unbundled component.
        self.write_text(self.compiler_docs / "licensing" / "c" / "LICENSE")
        self.write_text(self.umf_docs / "licensing" / "third-party-programs.txt")
        self.write_text(self.umf_docs / "licensing" / "extra-notice.txt")
        self.write_text(self.compiler_docs / "licensing" / "openmp" / "third-party-programs.txt")
        self.write_text(self.root / "oneapi" / "mkl" / "selected" / "share" / "doc" / "mkl"
                        / "licensing" / "license.txt")
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.archive_licenses(), {
            "llama.cpp-LICENSE",
            "nlohmann-json-LICENSE",
            "cpp-httplib-LICENSE",
            "oneapi-compiler-third-party-programs.txt",
            "umf-LICENSE.txt",
        })

    def test_bundled_openmp_runtime_adds_its_notice(self):
        self.write_text(self.compiler_docs / "licensing" / "openmp" / "third-party-programs.txt")
        self.link_server_against((self.runtime, "libiomp5.so"))
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("libiomp5.so", self.archive_names())
        self.assertIn("oneapi-openmp-third-party-programs.txt",
                      self.archive_licenses())

    def test_bundled_hwloc_without_openmp_aborts_packaging(self):
        self.link_server_against((self.tcm_dir, "libhwloc.so.15"))
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hwloc", result.stderr)
        self.assertIn("OpenMP", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_bundled_hwloc_with_openmp_ships_no_tcm_text(self):
        self.write_text(self.compiler_docs / "licensing" / "openmp" / "third-party-programs.txt")
        self.write_text(self.tcm_dir.parent / "share" / "doc" / "tcm" / "licensing"
                        / "third-party-programs.txt")
        self.link_server_against((self.tcm_dir, "libhwloc.so.15"), (self.runtime, "libiomp5.so"))
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("libhwloc.so.15", self.archive_names())
        licenses = self.archive_licenses()
        self.assertEqual(licenses, {
            "llama.cpp-LICENSE",
            "nlohmann-json-LICENSE",
            "cpp-httplib-LICENSE",
            "oneapi-compiler-third-party-programs.txt",
            "oneapi-openmp-third-party-programs.txt",
            "umf-LICENSE.txt",
        })

    def move_umf_into_compiler(self):
        """Ship UMF from the compiler's lib directory, as some releases do."""
        for path in self.umf_dir.iterdir():
            shutil.move(str(path), self.runtime / path.name)
        self.umf_dir.rmdir()

    def test_umf_bundled_from_another_component_ships_umf_license(self):
        self.move_umf_into_compiler()
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("libumf.so.1.2", self.archive_names())
        self.assertIn("umf-LICENSE.txt", self.archive_licenses())
        self.assertIn("[compiler; +umf license]", result.stdout)

    def test_umf_bundled_from_another_component_without_umf_license_aborts(self):
        self.move_umf_into_compiler()
        (self.umf_docs / "LICENSE.TXT").unlink()
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UMF is bundled from", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_missing_oneapi_license_text_aborts_packaging(self):
        (self.umf_docs / "LICENSE.TXT").unlink()
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required license text missing", result.stderr)
        self.assertIn(str(self.umf_docs / "LICENSE.TXT"), result.stderr)
        self.assertFalse(self.archive.exists())

    def test_missing_llama_license_text_aborts_packaging(self):
        (self.build / "vendor" / "cpp-httplib" / "LICENSE").unlink()
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("vendor/cpp-httplib/LICENSE", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_bundled_library_without_license_mapping_aborts_packaging(self):
        self.link_server_against((self.root / "other", "libfoo.so.1"))
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no license mapping", result.stderr)
        self.assertIn("libfoo.so.1", result.stderr)
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
