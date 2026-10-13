#!/usr/bin/env python3
"""
Week 7 Lab - student self-check.  Run this BEFORE you submit.

    python selfcheck.py

It checks that your answer sheet is COMPLETE, your files are present, and
your four TODOs are filled in. It deliberately does NOT tell you whether
your numbers are right - that is what you are being marked on.
"""
import os
import re
import sys

REQUIRED_KEYS = [
    "STUDENT_NAME", "STUDENT_ID", "GPU_NAME", "PEAK_BW_GB_S",
    "E1_CPU_FLOAT_SUM", "E1_CPU_DOUBLE_SUM", "E1_GPU_SUM",
    "E1_BYTES_READ", "E1_AI", "E1_FLOOR_MS",
    "E1_MS", "E1_GBS", "E1_PCT_PEAK",
    "E2_B100_SUM", "E2_B100_LANES", "E2_B100_PCT", "E2_GRID50_SUM",
    "E2_NOSYNC_SUM", "E2_NOSYNC_RUNS_AGREE",
    "E3_INTERLEAVED_SUM", "E3_INTERLEAVED_MS", "E3_INTERLEAVED_SLOWDOWN",
    "E3_COARSE1_MS", "E3_COARSE1_GBS",
    "E3_BEST_BLOCK", "E3_BEST_COARSE", "E3_BEST_GBS", "E3_BEST_PCT_PEAK",
]
FILES = ["answers.txt", "reduction.cu", "README.md"]
YESNO = ["E2_NOSYNC_RUNS_AGREE"]
PLACEHOLDER = "<--"
N = 1 << 26

fails = warns = 0


def fail(m, hint=""):
    global fails
    fails += 1
    print("  [FAIL] " + m)
    if hint:
        print("         " + hint)


def warn(m, hint=""):
    global warns
    warns += 1
    print("  [WARN] " + m)
    if hint:
        print("         " + hint)


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", (s or "").replace(",", ""))
    return float(m.group(0)) if m else None


def main():
    print("Week 7 self-check\n")
    if not os.path.exists("answers.txt"):
        fail("answers.txt not found",
             "Copy answers_template.txt to answers.txt and fill it in.")
        return 1

    vals = {}
    for line in open("answers.txt", encoding="utf-8", errors="replace"):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()

    print("answer sheet")
    missing = [k for k in REQUIRED_KEYS if k not in vals]
    blank = [k for k in REQUIRED_KEYS if not vals.get(k, "")]
    if missing:
        fail("keys deleted from the sheet: " + ", ".join(missing),
             "Do not remove keys. Re-copy them from answers_template.txt.")
    if blank:
        fail("%d value(s) left blank: %s" % (len(blank), ", ".join(blank)),
             "Every blank is marks you cannot score.")
    if not missing and not blank:
        print("  [PASS] all %d keys present and filled" % len(REQUIRED_KEYS))

    for k in YESNO:
        v = vals.get(k, "").lower()
        if v and not (v.startswith("y") or v.startswith("n")):
            fail("%s = %r must be yes or no" % (k, vals.get(k)))

    # unit sanity, not correctness
    bw = num(vals.get("PEAK_BW_GB_S"))
    if bw is not None and not (100 <= bw <= 3000):
        warn("PEAK_BW_GB_S = %g looks off" % bw,
             "GB/s, e.g. 360 for an RTX 3060 - not TB/s, not bytes.")
    pct = num(vals.get("E1_PCT_PEAK"))
    if pct is not None and pct > 100:
        warn("E1_PCT_PEAK = %g is above 100%%" % pct,
             "No kernel beats its own memory bandwidth. Check PEAK_BW_GBS "
             "in the control panel against your Week 4 output.")
    gpu = num(vals.get("E1_GPU_SUM"))
    if gpu is not None and gpu != N:
        warn("E1_GPU_SUM = %g, but the BASELINE run should be exact (%d)"
             % (gpu, N),
             "Exercise 1 is the run with the control panel untouched.")
    isum = num(vals.get("E3_INTERLEAVED_SUM"))
    if isum is not None and isum != N:
        warn("E3_INTERLEAVED_SUM = %g" % isum,
             "Run 5 only changes SPEED - the answer should still be exact. "
             "Did another control-panel line get left changed?")
    for k in ("E2_B100_SUM", "E2_GRID50_SUM"):
        x = num(vals.get(k))
        if x is not None and x == N:
            warn("%s is the fully correct answer" % k,
                 "That run is supposed to LOSE data. Check you rebuilt with "
                 "'make' after editing the control panel.")

    print("\nfiles")
    for f in FILES:
        print("  [PASS] " + f) if os.path.exists(f) else fail(f + " missing")
    if not any(os.path.exists(r) for r in
               ("report.pdf", "report.md", "report.docx")):
        fail("report.* missing", "Q1-Q5 live there.")

    print("\nCUDA TODOs")
    if os.path.exists("reduction.cu"):
        left = open("reduction.cu", encoding="utf-8",
                    errors="replace").read().count(PLACEHOLDER)
        if left:
            fail("reduction.cu: %d TODO placeholder(s) still in the file" % left,
                 "Replace every line marked '<--'.")
        else:
            print("  [PASS] reduction.cu: TODOs filled in")

    print("\n" + "=" * 60)
    print("  %d failures, %d warnings" % (fails, warns))
    if fails:
        print("  Fix the failures before submitting.")
    elif warns:
        print("  No failures. Read the warnings - each is a common way to "
              "lose easy marks.")
    else:
        print("  Complete. This checks COMPLETENESS, not correctness -")
        print("  your numbers are still your own responsibility.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
