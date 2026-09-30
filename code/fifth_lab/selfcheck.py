#!/usr/bin/env python3
"""
Week 6 Lab - student self-check.

Run this BEFORE you submit.

    python selfcheck.py

It checks that your answer sheet is COMPLETE and your files are present.
It deliberately does NOT tell you whether your derived numbers are right --
that is the part you are being marked on. It catches the cheap mistakes:
a key you left blank, a value in the wrong units, a file you forgot.
"""
import os
import re
import sys

REQUIRED_KEYS = [
    "STUDENT_NAME", "STUDENT_ID",
    "GPU_NAME", "PEAK_BW_GB_S", "PEAK_FP32_TFLOPS",
    "THREADS_PER_SM", "BLOCKS_PER_SM",
    "E1_AI_R1", "E1_OPS_R2", "E1_BYTES_R2", "E1_AI_R2", "E1_AI_R3",
    "E1_AI_LIMIT", "E1_ROOF_GFLOPS", "E1_BOUND",
    "E1_MEASURED_GFLOPS_R2", "E1_EXCEEDS_ROOF", "E1_BREAK_DIFFS",
    "E2_OUT_TILE_8", "E2_HALO_PCT_8", "E2_OUT_TILE_16", "E2_HALO_PCT_16",
    "E2_OUT_TILE_32", "E2_HALO_PCT_32",
    "E2_THREADS_PER_BLOCK_32", "E2_BLOCKS_RESIDENT_32",
    "E2_LOADS_PER_OUT_V1", "E2_LOADS_PER_OUT_V3_32", "E2_REDUCTION",
    "E2_TILED_AI_32", "E2_SPEEDUP_V2_V1", "E2_SPEEDUP_V3_V1",
    "E2_FASTEST_TILE", "E2_BREAKB_DIFFS",
    "E2_BREAKC_WRONG", "E2_BREAKC_RUNS_AGREE",
    "E3_AI_C1", "E3_AI_C16", "E3_AI_C64", "E3_RIDGE", "E3_CROSSOVER_C",
    "E3_PEAK_GFLOPS", "E3_FLATTEN_C",
    "E3_INPLACE_DIFFS", "E3_INPLACE_RUNS_AGREE",
]
FILES = ["answers.txt", "conv_basic.csv", "conv_tiled.csv",
         "conv_layer.csv", "README.md",
         "conv_basic.cu", "conv_tiled.cu", "conv_layer.cu"]
YESNO = ["E1_EXCEEDS_ROOF", "E2_BREAKC_RUNS_AGREE", "E3_INPLACE_RUNS_AGREE"]

# Every placeholder in the student kit contains this marker. If it is still
# there, that TODO was never done.
PLACEHOLDER = "<--"
CUDA_TODOS = {"conv_basic.cu": 2, "conv_tiled.cu": 2, "conv_layer.cu": 2}

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
    print("Week 6 self-check\n")

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
    blank = [k for k in REQUIRED_KEYS if not vals.get(k, "")]
    missing = [k for k in REQUIRED_KEYS if k not in vals]
    if missing:
        fail("keys deleted from the sheet: " + ", ".join(missing),
             "Do not remove keys. Re-copy them from answers_template.txt.")
    if blank:
        fail("%d value(s) left blank: %s" % (len(blank), ", ".join(blank)),
             "Every blank is marks you cannot score. Write SKIP only if you "
             "truly could not obtain it.")
    if not missing and not blank:
        print("  [PASS] all %d keys present and filled" % len(REQUIRED_KEYS))

    for k in YESNO:
        v = vals.get(k, "").lower()
        if v and not (v.startswith("y") or v.startswith("n")):
            fail("%s = %r must be yes or no" % (k, vals.get(k)))

    b = vals.get("E1_BOUND", "").lower()
    if b and not (b.startswith("mem") or b.startswith("comp")):
        fail("E1_BOUND = %r must be 'memory' or 'compute'" % vals.get("E1_BOUND"))

    # unit sanity, not correctness
    bw = num(vals.get("PEAK_BW_GB_S"))
    if bw is not None and not (100 <= bw <= 3000):
        warn("PEAK_BW_GB_S = %g looks off" % bw,
             "This should be GB/s, e.g. 360 for an RTX 3060 - not bytes, "
             "not TB/s.")
    tf = num(vals.get("PEAK_FP32_TFLOPS"))
    if tf is not None and not (2 <= tf <= 200):
        warn("PEAK_FP32_TFLOPS = %g looks off" % tf,
             "TFLOP/s, e.g. 12.74 - not GFLOP/s.")
    tps = num(vals.get("THREADS_PER_SM"))
    if tps == 2048:
        warn("THREADS_PER_SM = 2048 is a DATACENTER value",
             "Consumer RTX cards report 1536. Use the banner your programs "
             "print, not the lecture slides.")

    for k in ("E1_BREAK_DIFFS", "E2_BREAKB_DIFFS", "E2_BREAKC_WRONG",
              "E3_INPLACE_DIFFS"):
        x = num(vals.get(k))
        if x is not None and x == 0:
            warn("%s = 0" % k,
                 "A break experiment that changed nothing usually means the "
                 "broken variant was not actually run. Re-check the command.")

    print("\nfiles")
    for f in FILES:
        if os.path.exists(f):
            print("  [PASS] " + f)
        else:
            fail(f + " missing")
    if not any(os.path.exists(p) for p in
               ("layer_plot.png", "layer_plot.pdf", "layer_plot.jpg")):
        fail("layer_plot.* missing", "Exercise 3 requires the plot.")
    if not any(os.path.exists(r) for r in
               ("report.pdf", "report.md", "report.docx")):
        fail("report.* missing",
             "Your 4 prose answers and 4 break explanations live here.")
    if not any(f.endswith(".py") and f not in ("selfcheck.py",)
               for f in os.listdir(".")):
        warn("no plot script (.py) found", "It is worth marks.")

    print("\nCUDA TODOs")
    for f, n in CUDA_TODOS.items():
        if not os.path.exists(f):
            continue
        src = open(f, encoding="utf-8", errors="replace").read()
        left = src.count(PLACEHOLDER)
        if left:
            fail("%s: %d of %d TODO placeholder(s) still in the file"
                 % (f, left, n),
                 "Replace every line marked '<--'. The program will not "
                 "produce results until all of them are right.")
        else:
            print("  [PASS] %s: TODOs filled in" % f)

    print("\n" + "=" * 60)
    print("  %d failures, %d warnings" % (fails, warns))
    if fails:
        print("  Fix the failures before submitting.")
    elif warns:
        print("  No failures. Read the warnings - each one is a common way "
              "to lose easy marks.")
    else:
        print("  Complete. Note this checks COMPLETENESS, not correctness -")
        print("  your derived numbers are still your own responsibility.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
