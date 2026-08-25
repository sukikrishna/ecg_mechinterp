#!/usr/bin/env bash
# Downloads the PTB Diagnostic ECG Database 1.0.0 from PhysioNet into data/raw/ptbdb/.
# Fully open access — no PhysioNet account or agreement needed.
#
# Uses the single prebuilt ZIP rather than mirroring individual record files: the per-file
# recursive download (wget -r over files/ptbdb/1.0.0/) is dramatically slower in practice
# (many tiny HTTP requests) than one bulk transfer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/ptbdb"
ZIP_URL="https://physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip"
ZIP_PATH="$ROOT/data/raw/ptbdb.zip"

mkdir -p "$ROOT/data/raw"
wget -c -q --show-progress "$ZIP_URL" -O "$ZIP_PATH"

rm -rf "$DEST"
TMP_EXTRACT="$(mktemp -d)"
# Uses Python's zipfile instead of the `unzip` binary, which isn't installed everywhere.
python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP_PATH" "$TMP_EXTRACT"
mv "$TMP_EXTRACT"/*/ "$DEST"
rmdir "$TMP_EXTRACT"
rm "$ZIP_PATH"

echo "PTB Diagnostic ECG Database downloaded to $DEST"
