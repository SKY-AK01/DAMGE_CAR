#!/usr/bin/env python3
"""
build_rust_dataloader.py
========================
Compiles the `rust_dataloader` PyO3 crate and installs the resulting
.pyd / .so into the project root so `import rust_dataloader` works
from any training script.

Requirements
------------
  - Rust toolchain (stable) with `cargo` in PATH
  - `maturin` Python package  (pip install maturin)

Usage
-----
    python build_rust_dataloader.py             # compile + install
    python build_rust_dataloader.py --check     # only check if already built
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
CRATE_DIR    = PROJECT_ROOT / "rust_dataloader"
DEST_DIR     = PROJECT_ROOT  # import rust_dataloader from project root

def _run(cmd: list, cwd: Path):
    print(f"[CMD] {' '.join(cmd)}")
    env = os.environ.copy()
    bin_path = r"C:\Users\Aakash\.cargo\w64devkit\w64devkit\bin"
    lib_path1 = r"C:\Users\Aakash\.cargo\w64devkit\w64devkit\lib"
    lib_path2 = r"C:\Users\Aakash\.cargo\w64devkit\w64devkit\lib\gcc\x86_64-w64-mingw32\16.2.0"
    
    if os.path.exists(bin_path) and bin_path not in env.get("PATH", ""):
        env["PATH"] = bin_path + os.path.pathsep + env.get("PATH", "")
    
    lib_paths = [p for p in [lib_path1, lib_path2] if os.path.exists(p)]
    if lib_paths:
        env["LIBRARY_PATH"] = os.path.pathsep.join(lib_paths) + (os.path.pathsep + env.get("LIBRARY_PATH", "") if env.get("LIBRARY_PATH") else "")
        flags = " ".join([f"-L native={p}" for p in lib_paths])
        env["RUSTFLAGS"] = flags + " " + env.get("RUSTFLAGS", "")
    
    # Avoid Windows antivirus file locking race conditions
    env["CARGO_BUILD_JOBS"] = "2"

    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        print(f"[FAILED] Command exited with code {result.returncode}")
        sys.exit(result.returncode)

def check_tools():
    ok = True
    if shutil.which("cargo") is None:
        print("[ERROR] `cargo` not found.  Install Rust from https://rustup.rs/")
        ok = False
    if shutil.which("maturin") is None:
        print("[WARN]  `maturin` not found — trying `pip install maturin` ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
    return ok

def find_built_lib() -> Path | None:
    """Find the compiled .pyd/.so in the target directory."""
    ext = ".pyd" if platform.system() == "Windows" else ".so"
    for path in (CRATE_DIR / "target").rglob(f"rust_dataloader*{ext}"):
        if "release" in path.parts or "debug" in path.parts:
            return path
    return None

def build():
    if not check_tools():
        sys.exit(1)

    print("\n[*] Compiling rust_dataloader (PyO3 extension) in release mode ...")
    print(f"    Crate : {CRATE_DIR}")
    print(f"    Output: {DEST_DIR}\n")

    # Build a wheel using maturin build (works without a virtualenv)
    wheel_dir = CRATE_DIR / "target" / "wheels"
    _run(
        [sys.executable, "-m", "maturin", "build", "--release",
         "--manifest-path", str(CRATE_DIR / "Cargo.toml"),
         "--out", str(wheel_dir)],
        cwd=PROJECT_ROOT,
    )

    # Find the built wheel and install it
    wheels = list(wheel_dir.glob("rust_dataloader*.whl"))
    if not wheels:
        print("[ERROR] No wheel found after build. Check maturin output above.")
        sys.exit(1)
    latest_wheel = sorted(wheels)[-1]
    print(f"\n[*] Installing wheel: {latest_wheel.name}")
    _run([sys.executable, "-m", "pip", "install", str(latest_wheel), "--force-reinstall"], cwd=PROJECT_ROOT)

    # Also try to copy the .pyd/.so to project root for Azure ML jobs
    lib = find_built_lib()
    if lib:
        dest = DEST_DIR / lib.name
        shutil.copy2(lib, dest)
        print(f"\n[OK] Copied {lib.name} → {dest}")
    else:
        print("\n[WARN] Could not locate compiled library; it may be installed into your venv directly.")

    print("\n[OK] rust_dataloader built successfully!")
    print("     You can now `import rust_dataloader` in any training script.\n")

def main():
    parser = argparse.ArgumentParser(description="Build the Rust DataLoader PyO3 extension.")
    parser.add_argument("--check", action="store_true",
                        help="Only check if the extension is already built, don't compile.")
    args = parser.parse_args()

    if args.check:
        try:
            import rust_dataloader  # noqa: F401
            print("[OK] rust_dataloader is available.")
        except ImportError:
            print("[NOT BUILT] rust_dataloader not found.  Run `python build_rust_dataloader.py` to compile.")
            sys.exit(1)
        return

    build()

if __name__ == "__main__":
    main()
