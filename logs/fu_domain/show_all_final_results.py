#!/usr/bin/env python3
import json
from pathlib import Path

METRICS = {
    "Acc": "eval/overall/acc",
    "MacroF1": "eval/classification/macro_f1",
    "WeightedF1": "eval/classification/weighted_f1",
    "RA": "eval/unlearning/RA",
    "FA": "eval/unlearning/FA",
    "ReA": "eval/unlearning/ReA",
    "MIA": "eval/unlearning/MIA_acc",
    "RemF1": "eval/unlearning/remember_macro_f1",
    "ForF1": "eval/unlearning/forget_macro_f1",
}

METHODS = ["fedavg", "fedbn", "fedmoon", "fedprox"]
DOMAINS = ["canada", "india", "spain"]

def parse_job_name(name):
    method = next((m for m in METHODS if f"_{m}_" in name), "-")
    domain = next((d for d in DOMAINS if f"_{d}_" in name), "-")
    setting = "backbone" if "_backbone_" in name else "oral" if "_oral_" in name else "-"
    return method.upper(), domain.capitalize(), setting.capitalize()

def fmt(x):
    if x is None:
        return "-"
    return f"{x:.4f}"

def load_rows(root):
    rows = []

    for path in sorted(Path(root).glob("*/metrics.json")):
        job = path.parent.name

        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"[SKIP] {job}: {e}")
            continue

        final = data.get("final", {})
        method, domain, setting = parse_job_name(job)

        row = {
            "Job": job,
            "Method": method,
            "Forget": domain,
            "Setting": setting,
        }

        for label, key in METRICS.items():
            row[label] = fmt(final.get(key))

        rows.append(row)

    return rows

def print_table(rows):
    headers = ["Method", "Forget", "Setting"] + list(METRICS.keys())

    widths = {
        h: max(len(h), *(len(str(r[h])) for r in rows))
        for h in headers
    }

    line = "─" * (sum(widths.values()) + 3 * (len(headers) - 1))

    print(line)
    print("   ".join(h.ljust(widths[h]) for h in headers))
    print(line)

    for r in rows:
        print("   ".join(str(r[h]).ljust(widths[h]) for h in headers))

    print(line)

def print_grouped(rows):
    for setting in ["Backbone", "Oral"]:
        group = [r for r in rows if r["Setting"] == setting]
        if not group:
            continue

        print(f"\n========== {setting.upper()} RESULTS ==========\n")
        print_table(group)

def print_best(rows):
    print("\n========== BEST RESULTS ==========\n")

    for metric in ["Acc", "MacroF1", "WeightedF1", "RA", "FA", "ReA", "MIA"]:
        valid = []
        for r in rows:
            try:
                valid.append((float(r[metric]), r))
            except:
                pass

        if not valid:
            continue

        best_value, best_row = max(valid, key=lambda x: x[0])

        print(
            f"Best {metric:<10}: {best_value:.4f}  "
            f"{best_row['Method']} | {best_row['Forget']} | {best_row['Setting']}"
        )

def main():
    rows = load_rows(".")

    if not rows:
        print("No metrics.json files found.")
        return

    rows.sort(key=lambda r: (r["Setting"], r["Forget"], r["Method"]))

    print_grouped(rows)
    print_best(rows)

if __name__ == "__main__":
    main()
