#!/usr/bin/env python3
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python show_fused_final_results.py <run_folder>")
    sys.exit(1)

input_path = Path(sys.argv[1])

if not input_path.exists():
    print(f"Error: folder not found: {input_path}")
    sys.exit(1)

if not input_path.is_dir():
    print(f"Error: input must be a directory, not a file: {input_path}")
    sys.exit(1)


def find_output_log(run_dir):
    """
    Finds the most likely FUSED output log inside the given run directory.
    """
    candidates = []

    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".txt", ".out", ".log", ""]:
            continue

        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue

        score = 0
        if "FUSED LoRA unlearning" in text:
            score += 3
        if "FUSED forget_client_train" in text:
            score += 3
        if "ReA (post-relearn forget-client accuracy)" in text:
            score += 3
        if "Final unlearned model" in text:
            score += 3
        if "LoRA trainable params" in text:
            score += 2
        if "=== Run started:" in text:
            score += 1

        if score > 0:
            candidates.append((score, path, text))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


log_path, text = find_output_log(input_path)

if text is None:
    print(f"Error: no FUSED output log found inside: {input_path}")
    sys.exit(1)


def get_match(pattern, default=None, cast=str):
    match = re.search(pattern, text)
    if not match:
        return default
    value = match.group(1)
    return cast(value)


def percent(x):
    if x is None:
        return "N/A"
    return f"{x:.4f}  ({x * 100:.2f}%)"


run_id = get_match(r"=== Run started:\s*(\S+)\s*===")
source_run = get_match(r"Forking from source FL run:\s*(\S+)")
slurm_job_id = get_match(r"SLURM job ID:\s*(\S+)")
working_dir = get_match(r"Working dir:\s*(.+)")
python_path = get_match(r"Python:\s*(.+)")

forget_hospital = get_match(r"Forgetting hospital:\s*(\S+)")
forget_index = get_match(r"Forgetting hospital:\s*\S+\s*\(index\s*(\d+)", cast=int)

global_epoch = get_match(r"global_epoch=(\d+)", cast=int)
local_epoch = get_match(r"local_epoch=(\d+)", cast=int)
batch_size = get_match(r"batch_size=(\d+)", cast=int)

trainable_params = get_match(r"LoRA trainable params:\s*(\d+)", cast=int)
total_params = get_match(r"LoRA trainable params:\s*\d+/(\d+)", cast=int)
trainable_percent = get_match(r"LoRA trainable params:\s*\d+/\d+\s*\(([0-9.]+)%\)", cast=float)

final_loss = get_match(r"Final unlearned model:\s*global test loss=([0-9.]+)", cast=float)
final_acc = get_match(r"Final unlearned model:\s*global test loss=[0-9.]+,\s*acc=([0-9.]+)", cast=float)

rea = get_match(r"ReA\s*\(post-relearn forget-client accuracy\)\s*=\s*([0-9.]+)", cast=float)

saved_model = get_match(r"Saved unlearned model to\s*(.+)")
elapsed = get_match(r"elapsed\s*([0-9.]+)s", cast=float)

fused_epochs = re.findall(
    r"\[FUSED forget_client_train\]\s*Epoch=(\d+),\s*avg_r_acc=([0-9.]+),\s*avg_f_acc=([0-9.]+)",
    text,
)

relearn_rounds = re.findall(
    r"\[Relearn\]\s*Round=(\d+),\s*avg_f_acc=([0-9.]+)\s*\(ReA\),\s*avg_r_acc=([0-9.]+)",
    text,
)

# Remove duplicate lines caused by timestamped + non-timestamped prints
fused_dict = {}
for epoch, r_acc, f_acc in fused_epochs:
    fused_dict[int(epoch)] = {
        "ra": float(r_acc),
        "fa": float(f_acc),
    }

relearn_dict = {}
for round_id, f_acc, r_acc in relearn_rounds:
    relearn_dict[int(round_id)] = {
        "rea": float(f_acc),
        "ra": float(r_acc),
    }


print("\n========== FUSED LoRA FINAL RESULTS ==========\n")

print(f"Input Folder        : {input_path}")
print(f"Detected Log File   : {log_path}")
print(f"Run ID              : {run_id or 'N/A'}")
print(f"Source FL Run       : {source_run or 'N/A'}")
print(f"SLURM Job ID        : {slurm_job_id or 'N/A'}")
print(f"Forgotten Hospital  : {forget_hospital or 'N/A'}")

if forget_index is not None:
    print(f"Forget Index        : {forget_index}")

print(f"Working Directory   : {working_dir or 'N/A'}")
print(f"Python              : {python_path or 'N/A'}")

print("\n--- Config ---")
print(f"Global Epochs       : {global_epoch if global_epoch is not None else 'N/A'}")
print(f"Local Epochs        : {local_epoch if local_epoch is not None else 'N/A'}")
print(f"Batch Size          : {batch_size if batch_size is not None else 'N/A'}")

print("\n--- Final Unlearned Model ---")
print(f"Global Accuracy     : {percent(final_acc)}")

if final_loss is not None:
    print(f"Global Loss         : {final_loss:.4f}")
else:
    print("Global Loss         : N/A")

print("\n--- LoRA Adapter Size ---")
if trainable_params is not None and total_params is not None:
    print(
        f"Trainable Params    : {trainable_params:,} / "
        f"{total_params:,} ({trainable_percent:.4f}%)"
    )
else:
    print("Trainable Params    : N/A")

print("\n--- FUSED Unlearning Summary ---")
if fused_dict:
    first_epoch = min(fused_dict)
    last_epoch = max(fused_dict)

    first = fused_dict[first_epoch]
    last = fused_dict[last_epoch]

    best_ra_epoch = max(fused_dict, key=lambda e: fused_dict[e]["ra"])
    lowest_fa_epoch = min(fused_dict, key=lambda e: fused_dict[e]["fa"])

    print(f"Epochs Found        : {len(fused_dict)}")
    print(f"Initial RA          : {percent(first['ra'])}")
    print(f"Initial FA          : {percent(first['fa'])}")
    print(f"Final RA            : {percent(last['ra'])}")
    print(f"Final FA            : {percent(last['fa'])}")
    print(f"Best RA             : {percent(fused_dict[best_ra_epoch]['ra'])} at epoch {best_ra_epoch}")
    print(f"Lowest FA           : {percent(fused_dict[lowest_fa_epoch]['fa'])} at epoch {lowest_fa_epoch}")
else:
    print("No FUSED epoch records found.")

print("\n--- Relearn Probe Summary ---")
if relearn_dict:
    first_round = min(relearn_dict)
    last_round = max(relearn_dict)

    first = relearn_dict[first_round]
    last = relearn_dict[last_round]

    best_rea_round = max(relearn_dict, key=lambda r: relearn_dict[r]["rea"])

    print(f"Rounds Found        : {len(relearn_dict)}")
    print(f"Initial ReA         : {percent(first['rea'])}")
    print(f"Final ReA           : {percent(rea if rea is not None else last['rea'])}")
    print(f"Final Retained Acc. : {percent(last['ra'])}")
    print(f"Best ReA            : {percent(relearn_dict[best_rea_round]['rea'])} at round {best_rea_round}")
else:
    print("No relearn records found.")

print("\n--- Output Files ---")
print(f"Saved Model         : {saved_model or 'N/A'}")

if elapsed is not None:
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Elapsed Time        : {hours}h {minutes}m {seconds}s")
else:
    print("Elapsed Time        : N/A")

print("\n==============================================\n")
