# Data: making sure every analysis pulled the same corpus

Three prior analyses each downloaded PTB-XL and/or PTB-DB independently, on different
machines, at different times. Before treating any two results as comparable — which is the
entire premise of consolidating them — this page pins exactly what "PTB-XL" and "PTB-DB" mean
here, and `scripts/verify_data.py` checks a fresh pull against it automatically.

## PTB-XL

- **Version**: 1.0.3, from `https://physionet.org/content/ptb-xl/1.0.3/`.
  `scripts/download_ptbxl.sh` pulls this exact version-pinned URL, not "latest" — re-running
  it on any machine gets the same bytes.
- **Expected size**: 21,799 records (`ptbxl_database.csv` row count). The commonly cited
  figure is 21,801; the earlier multi-model probing/CKA study found 2 fewer and left this
  unexplained but negligible. If a fresh pull gives exactly 21,801, PhysioNet's distribution
  has changed since — note it here rather than silently treating the two counts as
  interchangeable.
- **Verify**: `python3 scripts/verify_data.py ptbxl` checks the record count and, if
  `SHA256SUMS.txt` is present in the pull (PhysioNet ships one), checksums
  `ptbxl_database.csv`/`scp_statements.csv`/`LICENSE.txt` against it directly.
- **Not yet used for**: any PTB-DB-style analysis (SAE training, causal ablation, the
  demographic-pathway test) — see docs/findings.md's top recommendation. PTB-XL's own
  `ptbxl_database.csv` already carries age/sex (`PTBXL.demographics()` in
  `data/ptbxl.py`), so the confound check that mattered so much on PTB-DB is directly
  runnable here; it just hasn't been run yet.

## PTB Diagnostic ECG Database (PTB-DB)

- **Version**: 1.0.0, from `https://physionet.org/content/ptbdb/1.0.0/`.
  `scripts/download_ptbdb.sh` pulls this exact version-pinned URL.
- **Expected size**: 549 records, 290 subjects. Both the CLEF SAE convergence study (reading
  from a locally-downloaded zip) and the RQ1-4 stability notebook (reading via `wfdb`,
  streamed or local) independently reported these exact same numbers — since they used
  different acquisition code paths, that agreement is itself evidence the two pulls were the
  same corpus, not a coincidence to take on faith.
- **Expected diagnostic breakdown** (subject level): 148 MI, 52 HC, 22 with no free-text
  summary at all (`reason` comment missing or unparseable). The 22 no-summary subjects are
  used identically across the two PTB-DB analyses in this repo's lineage: included in
  unsupervised SAE token training (they're still real ECG windows), excluded from every
  *labeled* probe or ablation (they have no diagnosis to label). `data/ptbdb.py`'s
  `map_group` reproduces this by mapping `"n/a"` to `None`, so any downstream concept mask
  that filters on `dx_group.isin([...])` already excludes them the same way.
- **Verify**: `python3 scripts/verify_data.py ptbdb` walks every `.hea` file under the pull,
  parses the free-text summary the same way `data/ptbdb.py` does, and reports record count,
  subject count, and the per-group subject breakdown against the numbers above.
- **Known demographic skew** (this is a property of the *cohort*, not of any model — see
  docs/findings.md): MI subjects are older (59.4y mean) and more often male than HC subjects
  (40.2y mean); `data/ptbdb.py`'s `mi_vs_hc_confound_test` reproduces the exact chi-square/
  Welch-t check that established this.

## Running a new pull

```bash
bash scripts/download_ptbxl.sh   # -> data/raw/ptb-xl/
bash scripts/download_ptbdb.sh   # -> data/raw/ptbdb/
python3 scripts/verify_data.py ptbxl
python3 scripts/verify_data.py ptbdb
```

Run the verify step on every machine a new result will be produced on (a labmate's laptop,
Kaggle, a lab server) before comparing its numbers to anyone else's — see
[docs/provenance.md](provenance.md) for why this matters for the specific numbers being
carried forward from the three source analyses.
