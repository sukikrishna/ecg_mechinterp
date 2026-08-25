#!/usr/bin/env bash
# Downloads PTB-XL 1.0.3 from PhysioNet into data/raw/ptb-xl/.
# Fully open access — no PhysioNet account or agreement needed.
#
# Uses the single prebuilt ZIP rather than mirroring the ~43k individual record files: the
# per-file recursive download (wget -r over files/ptb-xl/1.0.3/) is dramatically slower in
# practice (thousands of tiny HTTP requests) than one bulk transfer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/ptb-xl"
ZIP_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
ZIP_PATH="$ROOT/data/raw/ptb-xl.zip"

mkdir -p "$ROOT/data/raw"
wget -c -q --show-progress "$ZIP_URL" -O "$ZIP_PATH"

rm -rf "$DEST"
TMP_EXTRACT="$(mktemp -d)"
# Uses Python's zipfile instead of the `unzip` binary, which isn't installed everywhere.
python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP_PATH" "$TMP_EXTRACT"
mv "$TMP_EXTRACT"/*/ "$DEST"
rmdir "$TMP_EXTRACT"
rm "$ZIP_PATH"

echo "PTB-XL downloaded to $DEST"
