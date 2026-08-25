#!/usr/bin/env bash
# Clones the CLEF source (it isn't published as a pip package) and downloads its pretrained
# checkpoint from Zenodo (DOI 10.5281/zenodo.17572734).
# BSD-3-Clause-Clear licensed, fully open — no account or access request needed.
#
# Defaults to the "medium" size (~370MB) to match ECGFounder's parameter count for
# comparison. Pass "small" or "large" to fetch a different size instead.
set -euo pipefail

SIZE="${1:-medium}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/external"
if [ ! -d "$ROOT/external/ecg-foundation-model" ]; then
  git clone --depth 1 https://github.com/Nokia-Bell-Labs/ecg-foundation-model "$ROOT/external/ecg-foundation-model"
fi

declare -A FILENAMES=(
  [small]="clef_small.ckpt"
  [medium]="clef_medium.ckpt"
  [large]="clef_largel.ckpt"  # typo is Zenodo's own filename, not ours
)
FILENAME="${FILENAMES[$SIZE]}"

mkdir -p "$ROOT/weights/clef"
wget -c -q --show-progress \
  "https://zenodo.org/records/17572734/files/${FILENAME}?download=1" \
  -O "$ROOT/weights/clef/${FILENAME}"

echo "CLEF source cloned to $ROOT/external/ecg-foundation-model"
echo "CLEF ($SIZE) weights downloaded to $ROOT/weights/clef/${FILENAME}"
