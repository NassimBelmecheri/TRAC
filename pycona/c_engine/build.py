#!/usr/bin/env python3
"""
Cross-platform build script for the TRAC/pycona C++ acquisition engine.

The engine is a self-contained C++ library (no third-party dependencies, C++17,
standard library only) compiled into a single shared library that ``c_interface.py``
loads via ``ctypes``:

    * Windows        -> ``c_engine.dll``
    * Linux          -> ``c_engine.so``
    * macOS          -> ``c_engine.so``   (kept as ``.so`` because the loader
                                            in ``c_interface.py`` expects that name)

Usage
-----
    python build.py                 # auto-detect a compiler and build in place
    python build.py --compiler g++  # force a specific compiler (cl/g++/clang++)
    python build.py --debug         # unoptimised build with assertions/symbols
    python build.py --verify        # after building, load the library via ctypes

The three translation units (``c_api.cpp``, ``quacq.cpp``, ``solver.cpp``) are
compiled and linked into one shared object. On Windows the GCC/Clang builds are
linked statically against libstdc++/libgcc so the resulting DLL has no runtime
dependency on the MinGW/Clang runtime DLLs (it loads inside any Python process).
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["c_api.cpp", "quacq.cpp", "solver.cpp"]


def _output_name() -> str:
    # c_interface.py loads "c_engine.dll" on win32 and "c_engine.so" everywhere else.
    return "c_engine.dll" if sys.platform == "win32" else "c_engine.so"


def _find_compiler(preferred: str = None):
    """Return (kind, exe) where kind is 'msvc' or 'gcc' ('gcc' == gcc/clang CLI)."""
    if preferred:
        exe = shutil.which(preferred)
        if not exe:
            sys.exit(f"error: requested compiler '{preferred}' not found on PATH")
        kind = "msvc" if os.path.basename(exe).lower().startswith("cl") else "gcc"
        return kind, exe

    # Preference order: native toolchain first on each platform.
    if sys.platform == "win32":
        candidates = [("msvc", "cl"), ("gcc", "g++"), ("gcc", "clang++")]
    elif sys.platform == "darwin":
        candidates = [("gcc", "clang++"), ("gcc", "g++")]
    else:
        candidates = [("gcc", "g++"), ("gcc", "clang++")]

    for kind, name in candidates:
        exe = shutil.which(name)
        if exe:
            return kind, exe

    sys.exit(
        "error: no C++ compiler found. Install one of:\n"
        "  * Windows : Visual Studio Build Tools (cl) or MinGW-w64 (g++)\n"
        "  * Linux   : g++ (build-essential) or clang++\n"
        "  * macOS   : clang++ (Xcode command line tools)"
    )


def _command(kind: str, exe: str, out: str, debug: bool):
    srcs = [os.path.join(HERE, s) for s in SOURCES]
    out_path = os.path.join(HERE, out)

    if kind == "msvc":
        # MSVC: /LD -> DLL, /EHsc -> C++ exceptions, /std:c++17, /O2 (or /Od /Zi debug).
        opt = ["/Od", "/Zi"] if debug else ["/O2"]
        return [exe, "/nologo", "/std:c++17", "/EHsc", *opt, "/LD", *srcs,
                f"/Fe:{out_path}"]

    # GCC / Clang command line.
    opt = ["-O0", "-g"] if debug else ["-O3"]
    cmd = [exe, "-std=c++17", "-shared", *opt, "-o", out_path]
    if sys.platform == "win32":
        # Self-contained DLL: no dependency on libstdc++-6/libgcc_s DLLs at runtime.
        cmd += ["-static", "-static-libgcc", "-static-libstdc++"]
    else:
        cmd += ["-fPIC"]
    cmd += srcs
    return cmd


def _verify(out: str):
    import ctypes
    lib = ctypes.CDLL(os.path.join(HERE, out))
    lib.quacq_create.argtypes = [ctypes.c_int]
    lib.quacq_create.restype = ctypes.c_void_p
    lib.quacq_free.argtypes = [ctypes.c_void_p]
    lib.quacq_free.restype = None
    handle = lib.quacq_create(9)
    if not handle:
        raise RuntimeError("quacq_create returned NULL")
    lib.quacq_free(handle)
    print(f"verify: {out} loads and runs correctly.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the pycona C++ acquisition engine.")
    ap.add_argument("--compiler", help="force a compiler executable (cl, g++, clang++)")
    ap.add_argument("--debug", action="store_true", help="unoptimised build with symbols")
    ap.add_argument("--verify", action="store_true", help="load the library via ctypes after building")
    args = ap.parse_args()

    out = _output_name()
    kind, exe = _find_compiler(args.compiler)
    cmd = _command(kind, exe, out, args.debug)

    print(f"compiler : {exe} ({kind})")
    print(f"target   : {os.path.join(HERE, out)}")
    print("command  : " + " ".join(cmd))

    # MSVC drops .obj files next to the sources; build in a scratch cwd and clean up.
    proc = subprocess.run(cmd, cwd=HERE)
    if proc.returncode != 0:
        return proc.returncode

    if kind == "msvc":
        for junk in os.listdir(HERE):
            if junk.endswith((".obj", ".exp", ".lib")):
                try:
                    os.remove(os.path.join(HERE, junk))
                except OSError:
                    pass

    print(f"built    : {out}")
    if args.verify:
        _verify(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
