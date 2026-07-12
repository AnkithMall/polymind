#!/usr/bin/env python3
"""
PolyMind Test Runner — runs unit, integration, and coverage checks
in one shot, then prints a detailed summary.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
PYTHON = sys.executable


def banner(msg: str) -> None:
    sep = "─" * 72
    print(f"\n{sep}\n  {msg}\n{sep}")


def run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or ROOT)


def main() -> int:
    start = time.monotonic()
    exit_code = 0
    sections: list[dict] = []

    # ── 1. Unit tests ────────────────────────────────────────────
    banner("UNIT TESTS — individual feature modules")
    unit_files = sorted(
        p for p in TESTS.iterdir()
        if p.name.startswith("test_") and p.suffix == ".py"
    )
    unit_passed = unit_failed = unit_skipped = 0
    unit_results: list[tuple[str, str]] = []
    for f in unit_files:
        r = run([PYTHON, "-m", "pytest", str(f), "-q", "--tb=short", "-p", "no:cacheprovider"])
        if r.returncode == 0:
            unit_passed += 1
            unit_results.append((f.name, "PASS"))
        elif r.returncode == 5:
            unit_skipped += 1
            unit_results.append((f.name, "SKIP"))
        else:
            unit_failed += 1
            unit_results.append((f.name, "FAIL"))
            exit_code = 1
        # print last line of output
        last = [l for l in r.stdout.strip().split("\n") if l][-2:] if r.stdout else []
        for l in last:
            print(f"  {f.name}: {l[:100]}")

    # ── 2. Integration tests ─────────────────────────────────────
    banner("INTEGRATION TESTS — cross-module flows")
    int_dir = TESTS / "integration"
    if int_dir.exists():
        r = run([PYTHON, "-m", "pytest", str(int_dir), "-v", "--tb=short", "-p", "no:cacheprovider"])
        print(r.stdout or r.stderr)
        if r.returncode == 0:
            int_result = "PASS"
        else:
            int_result = "FAIL"
            exit_code = 1
    else:
        int_result = "N/A"

    # ── 3. Summary ───────────────────────────────────────────────
    elapsed = time.monotonic() - start
    banner("SUMMARY")

    total_passed = unit_passed
    total_failed = unit_failed + (1 if int_result == "FAIL" else 0)
    total = total_passed + total_failed

    print(f"  {'Unit tests':30s} {unit_passed:3d} passed, {unit_failed:3d} failed")
    print(f"  {'Integration tests':30s} {int_result}")
    print(f"  {'Total duration':30s} {elapsed:.1f}s")
    print(f"  {'Exit code':30s} {exit_code}")

    # ── 4. Per-file results table ────────────────────────────────
    print()
    name_w = max(len(n) for n, _ in unit_results) + 2
    print(f"  {'File':{name_w}s} Status")
    print(f"  {'─' * name_w} ──────")
    for name, status in unit_results:
        tag = "[green]PASS[/]" if status == "PASS" else "[red]FAIL[/]"
        # plain text for terminal
        plain = "PASS" if status == "PASS" else "FAIL"
        print(f"  {name:{name_w}s} {plain}")

    if int_result != "N/A":
        print(f"  {'integration/':{name_w}s} {int_result}")

    print(f"\n  {'─' * 60}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
