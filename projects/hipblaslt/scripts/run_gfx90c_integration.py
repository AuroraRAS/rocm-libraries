#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build and run the clean gfx90c:xnack+ integration gate."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


TARGET = "gfx90c:xnack+"
BASE_ARCH = "gfx90c"
GTEST_FILTER = "*gfx90c_integration*"
EXPECTED_TESTS = 3


def run(command, *, cwd=None, env=None, capture=False):
    print("+", " ".join(map(str, command)), flush=True)
    result = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if result.returncode:
        if capture and result.stdout:
            print(result.stdout, end="")
        result.check_returncode()
    return result


def require_clean_build_dir(build_dir: Path) -> None:
    if build_dir.exists() and any(build_dir.iterdir()):
        raise RuntimeError(f"build directory must not exist or must be empty: {build_dir}")
    build_dir.mkdir(parents=True, exist_ok=True)


def verify_elf_target(readobj: Path, code_object: Path) -> None:
    output = run([readobj, "--file-headers", code_object], capture=True).stdout
    required = (
        "EF_AMDGPU_MACH_AMDGCN_GFX90C",
        "EF_AMDGPU_FEATURE_XNACK_ON_V4",
    )
    missing = [flag for flag in required if flag not in output]
    if missing:
        raise RuntimeError(f"{code_object} is missing target flags: {', '.join(missing)}")
    forbidden = (
        "EF_AMDGPU_FEATURE_XNACK_ANY_V4",
        "EF_AMDGPU_FEATURE_XNACK_OFF_V4",
    )
    present = [flag for flag in forbidden if flag in output]
    if present:
        raise RuntimeError(f"{code_object} has incompatible target flags: {', '.join(present)}")


def verify_device_library(library_dir: Path, rocm_root: Path) -> None:
    required_catalogs = (
        "TensileLibrary_lazy_gfx90c.dat.zlib",
        "TensileLiteLibrary_lazy_gfx90c_Mapping.dat.zlib",
    )
    for name in required_catalogs:
        if not (library_dir / name).is_file():
            raise RuntimeError(f"missing device-library catalog: {library_dir / name}")

    solution_bundles = sorted(library_dir.glob("TensileLibrary_*_gfx90c.co"))
    required_families = ("_SS_SS_", "_HH_HH_Type_HH_", "_HH_HH_Type_HH_HPA_")
    for family in required_families:
        if not any(family in path.name for path in solution_bundles):
            raise RuntimeError(f"missing gfx90c solution bundle family: {family}")

    fp8_artifacts = [
        path.name
        for path in library_dir.iterdir()
        if re.search(r"(?:^|_)(?:F8|B8|F8N|B8N)(?:_|$)", path.name)
    ]
    if fp8_artifacts:
        raise RuntimeError(f"unexpected FP8 artifacts without an emulation contract: {fp8_artifacts}")

    bundler = rocm_root / "bin" / "clang-offload-bundler"
    readobj = rocm_root / "bin" / "llvm-readobj"
    target_tag = "hipv4-amdgcn-amd-amdhsa--gfx90c:xnack+"
    with tempfile.TemporaryDirectory(prefix="gfx90c-code-objects-") as tmp:
        tmp_dir = Path(tmp)
        for index, bundle in enumerate(solution_bundles):
            targets = run([bundler, "--type=o", f"--input={bundle}", "-list"], capture=True).stdout
            if target_tag not in targets.splitlines():
                raise RuntimeError(f"{bundle} does not contain exact target {target_tag}:\n{targets}")
            raw = tmp_dir / f"solution-{index}.co"
            run(
                [
                    bundler,
                    "--type=o",
                    f"--targets={target_tag}",
                    f"--input={bundle}",
                    f"--output={raw}",
                    "--unbundle",
                ]
            )
            verify_elf_target(readobj, raw)

    helper = library_dir / "Kernels.so-000-gfx90c-xnack+.hsaco"
    if not helper.is_file():
        raise RuntimeError(f"missing xnack+ helper code object: {helper}")
    verify_elf_target(readobj, helper)


def find_test_binary(build_dir: Path) -> Path:
    matches = [path for path in build_dir.rglob("hipblaslt-test") if path.is_file() and os.access(path, os.X_OK)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one hipblaslt-test under {build_dir}, found: {matches}")
    return matches[0]


def run_numerical_tests(test_binary: Path, library_dir: Path) -> None:
    enumerator = shutil.which("rocm_agent_enumerator") or shutil.which("amdgpu-arch")
    if not enumerator:
        raise RuntimeError("rocm_agent_enumerator or amdgpu-arch is required")
    detected = run([enumerator], capture=True).stdout.splitlines()
    if BASE_ARCH not in {line.strip().split(":", 1)[0] for line in detected}:
        raise RuntimeError(f"this test requires a gfx90c GPU, detected: {detected}")

    env = os.environ.copy()
    env["HIPBLASLT_TENSILE_LIBPATH"] = str(library_dir)
    result = run([test_binary, f"--gtest_filter={GTEST_FILTER}"], env=env, capture=True)
    print(result.stdout, end="")
    passed = re.search(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.", result.stdout)
    if not passed or int(passed.group(1)) != EXPECTED_TESTS:
        raise RuntimeError(f"expected {EXPECTED_TESTS} passing gfx90c integration tests")
    if re.search(r"\[\s*SKIPPED\s*\]", result.stdout):
        raise RuntimeError("gfx90c integration tests must not be skipped")


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True, help="new or empty CMake build directory")
    parser.add_argument("--rocm-root", type=Path, default=Path("/usr/lib64/rocm/llvm"))
    parser.add_argument("--python", type=Path, default=root / "build" / "venv" / "bin" / "python")
    parser.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    return parser.parse_args(), root


def main() -> None:
    args, root = parse_args()
    build_dir = args.build_dir.resolve()
    rocm_root = args.rocm_root.resolve()
    require_clean_build_dir(build_dir)

    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("cmake is required")
    configure = [
        cmake,
        "-S", root,
        "-B", build_dir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={rocm_root / 'bin' / 'amdclang'}",
        f"-DCMAKE_CXX_COMPILER={rocm_root / 'bin' / 'amdclang++'}",
        f"-DCMAKE_ASM_COMPILER={rocm_root / 'bin' / 'amdclang'}",
        f"-DCMAKE_PREFIX_PATH={rocm_root}",
        f"-DROCM_PATH={rocm_root}",
        f"-DPython_EXECUTABLE={args.python.absolute()}",
        f"-DPython3_EXECUTABLE={args.python.absolute()}",
        f"-DGPU_TARGETS={TARGET}",
        "-DHIPBLASLT_ENABLE_DEVICE=ON",
        "-DHIPBLASLT_ENABLE_HOST=ON",
        "-DHIPBLASLT_ENABLE_CLIENT=ON",
        "-DHIPBLASLT_BUILD_TESTING=ON",
        "-DHIPBLASLT_ENABLE_BLIS=OFF",
        "-DHIPBLASLT_ENABLE_ROCROLLER=OFF",
        "-DHIPBLASLT_ENABLE_EXTOPS=OFF",
        "-DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF",
        "-DHIPBLASLT_ENABLE_SAMPLES=OFF",
        "-DTENSILELITE_LOGIC_FILTER=gfx90c/Equality/*",
        "-DTENSILELITE_BUILD_PARALLEL_LEVEL=1",
    ]
    run(configure)
    build_env = os.environ.copy()
    build_env["TENSILE_DISABLE_HELPER_CACHE"] = "1"
    run(
        [cmake, "--build", build_dir, "--target", "tensilelite-device-libraries", "--parallel", args.jobs],
        env=build_env,
    )
    run([cmake, "--build", build_dir, "--target", "hipblaslt-test", "--parallel", args.jobs])

    library_dir = build_dir / "Tensile" / "library" / BASE_ARCH
    verify_device_library(library_dir, rocm_root)
    run_numerical_tests(find_test_binary(build_dir), library_dir)
    print("gfx90c:xnack+ clean-build integration: PASS")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
