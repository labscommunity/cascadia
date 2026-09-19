# Assembling the SSD

The SSD carries this folder as `inkling-deploy/` next to the export
`inkling/out/` (549 GB int4 export + `attn_ov/` int8 attention IRs for all
layers + `head_ov/`). Besides these scripts it needs:

```
bin/linux/cascadia            cargo build --release -p cascadia --features openvino
                              with INTEL_OPENVINO_DIR = the extracted ubuntu24 GenAI archive
                              (sourced setupvars.sh), built on Ubuntu 24.04
bin/windows/cascadia-ov.exe   same on Windows (MSVC) against the Windows GenAI archive
bin/windows/cascadia.exe      cargo build --release -p cascadia (no OpenVINO; CPU-only ranks)
runtime/openvino_genai_ubuntu24_2026.3.1.0_x86_64.tar.gz
runtime/openvino_genai_ubuntu22_2026.3.1.0_x86_64.tar.gz
runtime/openvino_genai_windows_2026.3.1.0_x86_64.zip
    from https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.3.1/{linux,windows}/
runtime/python-3.12.10-embed-amd64.zip      (python.org; the Windows boxes' private Python)
runtime/vc_redist.x64.exe                   (https://aka.ms/vs/17/release/vc_redist.x64.exe; the executables and every
                                            OpenVINO DLL import the Visual C++ runtime, which the archive does not bundle)
gpu-debs/   intel/compute-runtime 26.35.39758.10 (intel-opencl-icd, libze-intel-gpu1, intel-ocloc,
            libigdgmm12), intel/intel-graphics-compiler 2.41.5 (intel-igc-core-2, intel-igc-opencl-2),
            oneapi-src/level-zero 1.33.1 (libze1 u22.04 + u24.04)   — GitHub release assets
            ocl-icd-libopencl1 (Ubuntu archive, `apt-get download ocl-icd-libopencl1` on 24.04): the OpenCL
            ICD loader. OpenVINO GenAI's library links libOpenCL.so.1, so the binary does not start without
            it, and a default Ubuntu install does not always have it
wheels/     openvino==2026.3.1 for cp310..cp314 manylinux_2_28 + cp312 win_amd64; numpy for the same
            (pip download --only-binary=:all: --platform manylinux_2_28_x86_64 --python-version X)
tools/      inkling_attn_ov.py inkling_moe_layer_ov.py inkling_expert_ov.py glm5_expert_ov.py
```

The SSD is ext4, which Windows cannot read: the two Windows boxes install over the LAN
(`serve.py` on an Ubuntu box that holds the SSD, `bootstrap.ps1` on the Windows box; see README).

The copy on the miner's external SSD (`/mnt/external_ssd/inkling-deploy`)
is the one assembled and tested on 2026-09-18.
