#!/usr/bin/env bash
# Clones the ECGFounder source (it isn't published as a pip package, so the model class has to
# come from the repo itself) and downloads its two pretrained checkpoints from Hugging Face.
# MIT licensed, fully open — no HF account or access request needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/external"
if [ ! -d "$ROOT/external/ECGFounder" ]; then
  git clone --depth 1 https://github.com/PKUDigitalHealth/ECGFounder "$ROOT/external/ECGFounder"
fi

mkdir -p "$ROOT/weights/ecgfounder"
python3 - "$ROOT/weights/ecgfounder" <<'PY'
import sys
from huggingface_hub import hf_hub_download

dest = sys.argv[1]
for filename in ("12_lead_ECGFounder.pth", "1_lead_ECGFounder.pth"):
    path = hf_hub_download(repo_id="PKUDigitalHealth/ECGFounder", filename=filename, local_dir=dest)
    print(f"downloaded {path}")
PY

echo "ECGFounder source cloned to $ROOT/external/ECGFounder"
echo "ECGFounder weights downloaded to $ROOT/weights/ecgfounder"
