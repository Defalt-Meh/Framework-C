#!/usr/bin/env python3
from setuptools import setup, Extension
import subprocess, platform, pathlib, sys, os, shutil
import numpy as np

mac = sys.platform == "darwin"

def is_clang(cc):
    try:
        out = subprocess.check_output([cc, "--version"], stderr=subprocess.STDOUT, text=True)
        return "clang" in out.lower()
    except Exception:
        return False

def best_apple_mcpu(cc="clang"):
    if not is_clang(cc):
        return None
    for target in ("apple-m4", "apple-m3", "apple-m2", "apple-m1"):
        try:
            subprocess.run([cc, f"-mcpu={target}", "-x", "c", "-", "-c", "-o", os.devnull],
                           input=b"", check=True, stderr=subprocess.DEVNULL)
            return target
        except subprocess.CalledProcessError:
            pass
    return None

def has_openblas():
    if os.environ.get("OPENBLAS_DIR"):
        return True
    try:
        subprocess.check_call(["pkg-config", "--exists", "openblas"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

system  = platform.system().lower()
machine = platform.machine().lower()
compiler = os.environ.get("CC", "clang" if system=="darwin" else "gcc")
msvc = (system == "windows") and ("cl" in compiler or "cl.exe" in compiler or os.environ.get("MSVC", ""))

# ---------------- profiles ----------------
profile = os.environ.get("FWC_PROFILE", "fast").lower()  # "fast" or "accurate"
native  = os.environ.get("FWC_NATIVE", "0") == "1"       # add native/arch flags

# BLAS + ReLU switches (env overrides allowed)
use_blas_env = os.environ.get("FWC_USE_BLAS")
relu_on_env  = os.environ.get("FWC_RELU", "1")           # default ReLU ON
oat_env      = os.environ.get("FWC_OAT", "1")            # default OAT ON

use_blas = has_openblas() or (system == "darwin") if use_blas_env is None else use_blas_env
use_blas = bool(int(use_blas)) if use_blas in ("0", "1") else bool(use_blas)

relu_on  = bool(int(relu_on_env)) if relu_on_env in ("0", "1") else bool(relu_on_env)
oat_on   = bool(int(oat_env)) if oat_env in ("0","1") else bool(oat_env)

# ---------------- compiler flags/macros ----------------
compile_args = []
define_macros = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_19_API_VERSION"),
]

# OAT (Online Alpha–Beta Tuning) default ON
if oat_on:
    define_macros += [("FWC_OAT", "1")]
else:
    define_macros += [("FWC_OAT", "0")]

# ReLU macro (avoid duplicate definitions)
define_macros += [("FWC_RELU_HID", "1" if relu_on else "0")]

if msvc:
    if profile == "fast":
        compile_args += ["/O2", "/fp:fast"]
    else:
        compile_args += ["/O2", "/fp:precise"]
    if native:
        compile_args += ["/arch:AVX2"]
else:
    if profile == "fast":
        compile_args += ["-O3", "-ffast-math", "-funroll-loops", "-fno-common",
                         "-fno-math-errno", "-fno-trapping-math"]
    else:
        compile_args += ["-O3", "-fno-common"]
    if native:
        if system == "darwin" and machine == "arm64":
            mcpu = best_apple_mcpu(compiler) or "native"
            compile_args += [f"-mcpu={mcpu}"]
        else:
            compile_args += ["-march=native"]

# Warnings
compile_args += ["/W3"] if msvc else ["-Wall", "-Wextra"]

# BLAS/Accelerate linkage (optional)
libraries = []
library_dirs = []
extra_link_args = []

if use_blas:
    define_macros += [("FWC_USE_BLAS", "1")]
    if system == "darwin":
        define_macros += [("ACCELERATE_NEW_LAPACK", "1")]
        extra_link_args += ["-framework", "Accelerate"]
    elif msvc:
        ob_dir = os.environ.get("OPENBLAS_DIR")
        if ob_dir:
            library_dirs += [os.path.join(ob_dir, "lib")]
        libraries += ["openblas"]
    else:
        try:
            cflags = subprocess.check_output(["pkg-config", "--cflags", "openblas"], text=True).strip().split()
            libs   = subprocess.check_output(["pkg-config", "--libs", "openblas"], text=True).strip().split()
            compile_args += cflags
            extra_link_args += libs
        except Exception:
            libraries += ["openblas"]
else:
    define_macros += [("FWC_USE_BLAS", "0")]

sources = [
    "src/my_module.c",
    "src/nn.c",
    "src/utils.c",
    "src/data_split.c",
    "src/model_selection.c",
    "src/elas.c",
]

ext_modules = [
    Extension(
        name="frameworkc",
        sources=sources,
        include_dirs=["src", np.get_include()],
        define_macros=define_macros,
        extra_compile_args=compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        library_dirs=library_dirs,
    )
]

readme_path = pathlib.Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

print(f"[setup] Building on {system}/{machine}", file=sys.stderr)
print(f"[setup] profile={profile} native={native} blas={use_blas} relu={relu_on} oat={oat_on}", file=sys.stderr)
print(f"[setup]  cc={compiler}", file=sys.stderr)
print(f"[setup]  cflags: {' '.join(compile_args)}", file=sys.stderr)
print(f"[setup]  macros: {define_macros}", file=sys.stderr)
print(f"[setup]  link  : libs={libraries} extra={extra_link_args}", file=sys.stderr)

setup(
    name="frameworkc",
    version="0.2.3",  # bump to avoid cached wheel
    description="Pure-C neural-network core with Python bindings (SIMD/BLAS-optional)",
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
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    zip_safe=False,
)
