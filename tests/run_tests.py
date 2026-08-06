#!/usr/bin/env python3
"""Run the whole test suite without pytest.

    python tests/run_tests.py

Two conventions coexist in this directory and both are supported:

  * script-style — the file does its work under `if __name__ == "__main__"` and reports by
    exit code. This is what most of the suite uses.
  * function-style — the file defines `test_*` functions and nothing else, so pytest can
    collect them. Those files call `run_module_tests(globals())` from their own `__main__`.

Every file is run as a SUBPROCESS. That is not incidental: most of this suite needs the
upstream lingbot-va or cosmos-framework trees, and importing those in-process means one
missing checkout takes down the whole run instead of skipping one file.

A missing THIRD-PARTY module is reported as SKIP. A missing `instinctwm` is reported as FAIL,
because that means the repo cannot import itself — which is exactly what a stale absolute
sys.path entry looks like, and it stayed invisible for a while by looking like a skip.
"""
from __future__ import annotations

import re
import subprocess
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MISSING_RE = re.compile(r"No module named '([^']+)'")
#: Evidence that the file got far enough to RUN tests, in either convention: `ok test_x` /
#: `FAIL test_x` from the function-style runner, and the indented `OK  ` / `FAIL` markers the
#: script-style files print per assertion.
RAN_RE = re.compile(r"^(?:ok|FAIL)\s+test_\w|^\s+(?:OK|FAIL)\s", re.M)


def run_module_tests(namespace: dict) -> int:
    """Run every `test_*` callable in `namespace`. Returns a process exit code.

    Used by the function-style files so they behave like the script-style ones when executed
    directly, while staying collectable by pytest.
    """
    failures = 0
    for name in sorted(namespace):
        if not name.startswith("test_"):
            continue
        fn = namespace[name]
        if not callable(fn):
            continue
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"FAIL {name}")
            failures += 1
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


def classify(proc: subprocess.CompletedProcess) -> tuple[str, str]:
    """-> (verdict, detail)."""
    if proc.returncode == 0:
        return "PASS", ""
    missing = MISSING_RE.search(proc.stderr or "")
    if missing:
        mod = missing.group(1)
        root = mod.split(".")[0]
        if root == "instinctwm":
            return "FAIL", f"cannot import {mod} — the repo cannot import itself"
        # A missing third-party module is a SKIP only if it stopped the file from running at
        # all. If tests EXECUTED and some failed, the failure is real and must be reported --
        # otherwise one test that touches the lingbot tree hides every other failure in the
        # file behind a reassuring "needs modules". That is not hypothetical: it hid a genuine
        # `install_plan` regression, which read as a skip for as long as nobody ran the file
        # directly.
        if not RAN_RE.search(proc.stdout or ""):
            return "SKIP", f"needs {root}"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return "FAIL", tail[-1] if tail else f"exit {proc.returncode}"


def main() -> int:
    results = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        proc = subprocess.run(
            [sys.executable, str(path)], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=600,
        )
        verdict, detail = classify(proc)
        results.append((path.name, verdict, detail, proc))
        print(f"{verdict:5s} {path.name:34s} {detail}")

    failed = [r for r in results if r[1] == "FAIL"]
    for name, _, _, proc in failed:
        print("\n" + "=" * 72)
        print(name)
        print((proc.stdout or "") + (proc.stderr or ""))

    passed = sum(1 for r in results if r[1] == "PASS")
    skipped = sum(1 for r in results if r[1] == "SKIP")
    print(f"\n{passed} passed, {skipped} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
