#!/usr/bin/env python3
"""
setup.py – FRAMEWORK-C Python extension
======================================

Builds the ultra-lightweight C neural-network core with SIMD/BLAS
acceleration and exposes it to Python via a single extension module.

    python -m pip install .
"""

from setuptools import setup, Extension
import subprocess, platform, pathlib, sys
import numpy as np

# ───────────────────────────── probe best -mcpu on Apple silicon ──
def best_apple_mcpu() -> str | None:
    """
    Return the newest “-mcpu=apple-*” string accepted by the host Clang.
    Falls back to None if *none* are recognised.
    """
    clang = "clang"
    # If the build is being driven by a non-Clang compiler, bail out.
    if "clang" not in subprocess.getoutput(f"{clang} --version"):
        return None

    for target in ("apple-m4", "apple-m3", "apple-m2", "apple-m1"):
        cmd = [clang, f"-mcpu={target}", "-x", "c", "-", "-c", "-o", "/dev/null"]
        try:
            subprocess.run(cmd, input=b"", check=True, stderr=subprocess.DEVNULL)
            return target                # first one that succeeds
        except subprocess.CalledProcessError:
            continue
    return None

# ───────────────────────────── common compiler & linker flags ─────
compile_args = [
    "-Ofast",            # aggressive optimisation
    "-ffast-math",       # relaxed IEEE rules
    "-funroll-loops",
    "-fno-strict-overflow",
    "-fno-common",
    "-Wall",
    "-Wextra",
]

link_args = []

system  = platform.system()
machine = platform.machine().lower()

if system == "darwin" and machine == "arm64":       # Apple silicon macOS
    mcpu_flag = best_apple_mcpu()
    compile_args += [f"-mcpu={mcpu_flag or 'native'}"]
    compile_args += ["-DACCELERATE_NEW_LAPACK"]
    link_args    += ["-framework", "Accelerate"]    # CBLAS + vDSP
else:                                               # Linux, Windows, Intel Mac
    compile_args += ["-march=native"]
    link_args    += ["-lopenblas"]                  # rely on OpenBLAS

# ───────────────────────────── extension definition ───────────────
ext_modules = [
    Extension(
        name="frameworkc",
        sources=[
            "src/my_module.c",
            "src/nn.c",
            "src/utils.c",
            "src/data_split.c",
            "src/model_selection.c",
            "src/elas.c",
        ],
        include_dirs=["src", np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

# ───────────────────────────── long description for PyPI ──────────
readme_path = pathlib.Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# ───────────────────────────── diagnostics banner ─────────────────
print(f"[setup] Building on {system}/{machine}", file=sys.stderr)
print(f"[setup]  compile flags: {' '.join(compile_args)}", file=sys.stderr)
print(f"[setup]  link flags   : {' '.join(link_args)}", file=sys.stderr)

# ───────────────────────────── setuptools invocation ──────────────
setup(
    name="frameworkc",
    version="0.2.1",
    description="Pure-C neural-network core with Python bindings (SIMD/BLAS-accelerated)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Defalt",
    url="https://github.com/",
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    zip_safe=False,
)
