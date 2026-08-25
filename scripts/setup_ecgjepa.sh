#!/usr/bin/env bash
# Clones the ECG-JEPA source (not published as a pip package) and downloads its multi-block-
# masking checkpoint from Google Drive (~326MB). MIT licensed, fully open.
#
# CAUTION: ECG-JEPA needs `timm`, which by default pulls in a CUDA build of torch and an
# incompatible torchvision, silently breaking an existing CPU-only torch install. This script
# installs torchvision from the CPU wheel index with torch version-pinned via a constraints
# file first, then installs timm/einops/gdown/beautifulsoup4 with --no-deps so nothing else
# gets pulled in. Verify `python3 -c "import torch; print(torch.__version__)"` still shows
# your expected build afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/external"
if [ ! -d "$ROOT/external/ecg-jepa" ]; then
  git clone --depth 1 https://github.com/sehunfromdaegu/ECG_JEPA "$ROOT/external/ecg-jepa"
fi

TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
CONSTRAINTS="$(mktemp)"
echo "torch==${TORCH_VERSION}" > "$CONSTRAINTS"
pip install --index-url https://download.pytorch.org/whl/cpu -c "$CONSTRAINTS" torchvision -q
pip install --no-deps timm einops gdown beautifulsoup4 soupsieve -q
rm -f "$CONSTRAINTS"
python3 -c "import torch; assert torch.__version__.split('+')[0] == '${TORCH_VERSION}', 'torch version changed! check the log above'"

mkdir -p "$ROOT/weights/ecgjepa"
gdown "https://drive.google.com/uc?id=1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l" \
  -O "$ROOT/weights/ecgjepa/multiblock_epoch100.pth"

echo "ECG-JEPA source cloned to $ROOT/external/ecg-jepa"
echo "ECG-JEPA (multi-block masking) weights downloaded to $ROOT/weights/ecgjepa/"
