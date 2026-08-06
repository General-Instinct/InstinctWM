#!/usr/bin/env python3
"""Two RoboTwin runs -> one certificate.json. The end of the Layer 1 workflow.

    checkpoint -> paired evaluation on pinned seeds -> episode JSONL -> certificate.json -> PASS/FAIL

The margin is a REQUIRED argument and is recorded in the certificate. A non-inferiority threshold
chosen after seeing the delta is a narrative, not a gate.

    python certify_run.py --teacher t.jsonl --student s.jsonl --margin -0.05 -o certificate.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root from this file, not written down. The hardcoded "/home/ubuntu/InstinctWM" this
# replaces has not existed since the tree moved to /home/ubuntu/Code/InstinctWM; it only kept
# working because `instinctwm` happened to resolve some other way, and it would have failed as
# an import error at the END of a multi-hour certification run. serve_variant.py carries the
# same note for the same reason.
IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

from instinctwm.certify import NotCertifiable, certify, load_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--margin", type=float, required=True,
                    help="acceptable success-rate LOSS, negative. -0.05 = 'may be 5 points worse'")
    ap.add_argument("-o", "--out", default="certificate.json")
    ap.add_argument("--teacher-hash", default="?")
    ap.add_argument("--student-hash", default="?")
    ap.add_argument("--recipe", default="?")
    ap.add_argument("--harness", default="robotwin-2.0")
    a = ap.parse_args()

    try:
        cert = certify(
            load_jsonl(a.teacher), load_jsonl(a.student), margin=a.margin,
            teacher_hash=a.teacher_hash, student_hash=a.student_hash,
            recipe=a.recipe, harness=a.harness,
            seeds="official LingBot-VA protocol, st_seed = 10000*(1+seed)")
    except NotCertifiable as e:
        print(f"NOT CERTIFIABLE: {e}", file=sys.stderr)
        return 2

    print(cert)
    print("\nper-task (a macro average can hide a task that collapsed):")
    print(cert.per_task_table())
    with open(a.out, "w") as f:
        f.write(cert.to_json())
    print(f"\nwrote {a.out}")
    return 0 if cert.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
