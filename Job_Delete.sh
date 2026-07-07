#!/usr/bin/env bash
set -euo pipefail

JOBID="${1:-}"

if [[ -z "$JOBID" ]]; then
  echo "Usage: $0 <jobid>"
  exit 1
fi

echo "Searching for files/directories related to job ID: $JOBID"
echo "Skipping ./dataset"
echo

mapfile -d '' MATCHES < <(
  find . \
    -path "./dataset" -prune -o \
    -name "*${JOBID}*" -print0
)

if [[ ${#MATCHES[@]} -eq 0 ]]; then
  echo "No files or directories found for job ID: $JOBID"
  exit 0
fi

echo "The following files/directories will be deleted:"
printf '%s\n' "${MATCHES[@]}"
echo

read -r -p "Type 'yes' to permanently delete these files: " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
  echo "Deletion cancelled."
  exit 0
fi

printf '%s\0' "${MATCHES[@]}" | xargs -0 rm -rf

echo "Deleted ${#MATCHES[@]} item(s) related to job ID: $JOBID"
