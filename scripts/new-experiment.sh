#!/usr/bin/env bash
# Usage: scripts/new-experiment.sh <project> <short-name>
set -euo pipefail

project="${1:?project name required}"
name="${2:?short experiment name required}"
root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
dir="$root/projects/$project/experiments/$(date +%Y-%m-%d)-$name"

if [[ ! -d "$root/projects/$project" ]]; then
  echo "No such project: $project" >&2
  exit 1
fi
if [[ -e "$dir" ]]; then
  echo "Already exists: $dir" >&2
  exit 1
fi

mkdir -p "$dir"
cat > "$dir/config.yaml" <<EOF
seed: 0
# parameters for this run
EOF
cat > "$dir/README.md" <<EOF
# $name

**Hypothesis:**

**Setup:** see config.yaml. Commit: \`$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo none)\`

**Result:**

**Verdict:**
EOF
echo "Created $dir"
