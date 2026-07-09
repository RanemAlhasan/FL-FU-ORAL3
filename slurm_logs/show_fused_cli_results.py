#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from datetime import datetime

def grab(pattern, text, default=""):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else default

def grab_float(pattern, text, default=None):
    v = grab(pattern, text, "")
    try:
        return float(v)
    except:
        return default

def fmt(v):
    if v is None:
        return "-"
    return f"{v:.4f}"

def parse_log(path):
    text = path.read_text(errors="ignore")

    job_id = grab(r"SLURM job ID:\s*(\d+)", text)
    source_run = grab(r"Source run:\s*(\S+)", text)
    run_id = grab(r"Run started:\s*(\S+)", text)
    algorithm = grab(r"algorithm=([a-zA-Z0-9_+-]+)", text)
    forgot = grab(r"forgot\s+([A-Za-z_]+Dataset)", text)
    if not forgot:
        forgot = grab(r"Forgetting hospital:\s*([A-Za-z_]+)", text)

    start = grab(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] === Run started", text)
    end = grab(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] === Run finished", text)
    elapsed = grab_float(r"elapsed\s+([\d.]+)s", text)

    final_acc = grab_float(r"Final unlearned model:.*?acc=([\d.]+)", text)
    overall_acc = grab_float(r"\[eval\] overall_acc=([\d.]+)", text)
    overall_loss = grab_float(r"\[eval\] overall_acc=[\d.]+ overall_loss=([\d.]+)", text)

    cls_acc = grab_float(r"\[eval/classification\] accuracy=([\d.]+)", text)
    macro_f1 = grab_float(r"\[eval/classification\].*?macro_f1=([\d.]+)", text)
    weighted_f1 = grab_float(r"\[eval/classification\].*?weighted_f1=([\d.]+)", text)

    remember_acc = grab_float(r"\[eval/unlearning/remember_classification\] accuracy=([\d.]+)", text)
    forget_acc = grab_float(r"\[eval/unlearning/forget_classification\] accuracy=([\d.]+)", text)

    ra = grab_float(r"\[eval/unlearning\] RA=([\d.]+)", text)
    fa = grab_float(r"\[eval/unlearning\].*?FA=([\d.]+)", text)
    rea = grab_float(r"\[eval/unlearning\].*?ReA=([\d.]+)", text)
    quick_mia = grab_float(r"\[eval/unlearning\].*?MIA_acc=([\d.]+)", text)
    shadow_mia = grab_float(r"Shadow-model MIA accuracy = ([\d.]+)", text)

    per_hospital = grab(r"per_hospital=\{([^}]+)\}", text)

    status = "DONE" if "Run finished" in text else "INCOMPLETE"
    if "Traceback" in text or "Error" in text or "Exception" in text:
        status = "ERROR"

    return {
        "file": path.name,
        "job_id": job_id,
        "run_id": run_id,
        "source_run": source_run,
        "algorithm": algorithm,
        "forgot": forgot.replace("_Dataset", ""),
        "status": status,
        "start": start,
        "end": end,
        "elapsed_h": elapsed / 3600 if elapsed else None,
        "acc": cls_acc or final_acc or overall_acc,
        "loss": overall_loss,
        "RA": ra or remember_acc,
        "FA": fa or forget_acc,
        "ReA": rea,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "quick_mia": quick_mia,
        "shadow_mia": shadow_mia,
        "per_hospital": per_hospital,
    }

def print_table(rows):
    headers = [
        "Job", "Forgot", "Alg", "Status", "Acc", "RA", "FA", "ReA",
        "MacroF1", "WeightedF1", "QuickMIA", "ShadowMIA", "Hours"
    ]

    table = []
    for r in rows:
        table.append([
            r["job_id"],
            r["forgot"],
            r["algorithm"],
            r["status"],
            fmt(r["acc"]),
            fmt(r["RA"]),
            fmt(r["FA"]),
            fmt(r["ReA"]),
            fmt(r["macro_f1"]),
            fmt(r["weighted_f1"]),
            fmt(r["quick_mia"]),
            fmt(r["shadow_mia"]),
            fmt(r["elapsed_h"]),
        ])

    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line():
        print("-" * (sum(widths) + 3 * len(widths) + 1))

    line()
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    line()
    for row in table:
        print("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    line()

def print_details(rows):
    print("\nDETAILED JOB INFO")
    print("=" * 80)

    for r in rows:
        print(f"\nJob {r['job_id']} | {r['forgot']} | {r['algorithm']} | {r['status']}")
        print(f"  File       : {r['file']}")
        print(f"  Run ID     : {r['run_id']}")
        print(f"  Source run : {r['source_run']}")
        print(f"  Start      : {r['start']}")
        print(f"  End        : {r['end']}")
        print(f"  Runtime    : {fmt(r['elapsed_h'])} hours")
        if r["per_hospital"]:
            print(f"  Per hospital accuracy: {{ {r['per_hospital']} }}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python show_fused_cli_results.py <fused_cli_or_fused_cli_backbone_folder>")
        sys.exit(1)

    log_dir = Path(sys.argv[1])

    if not log_dir.exists():
        print(f"Error: folder not found: {log_dir}")
        sys.exit(1)

    # Works for both:
    # fused_cli/          -> output_fu_cli_canada_fedavg_342714.txt
    # fused_cli_backbone/ -> output_fu_cli_canada_fedbn_bb_342726.txt
    files = sorted(log_dir.glob("output_fu_cli_*.txt"))

    if not files:
        print(f"No output_fu_cli_*.txt files found in: {log_dir}")
        sys.exit(1)

    rows = [parse_log(f) for f in files]

    rows.sort(key=lambda r: (
        r["forgot"],
        r["algorithm"],
        int(r["job_id"]) if r["job_id"].isdigit() else 999999999
    ))

    title = "FUSED CLI BACKBONE RESULTS SUMMARY" if "backbone" in log_dir.name.lower() else "FUSED CLI RESULTS SUMMARY"

    print(f"\n{title}")
    print("=" * 80)
    print_table(rows)
    print_details(rows)

    print("\nLegend:")
    print("  RA  = Remember accuracy")
    print("  FA  = Forget accuracy")
    print("  ReA = Relearn accuracy")
    print("  QuickMIA = MIA value printed before shadow-model MIA")
    print("  ShadowMIA = final shadow-model MIA accuracy")

if __name__ == "__main__":
    main()
