# ecg_mechinterp

Mechanistic interpretability of ECG foundation models — shared representations across
encoders, sparse-autoencoder feature/subspace stability, and whether a clinical concept is
partly a demographic one, measured causally.

This repo **consolidates three prior, independently-run analyses** into one dataset-agnostic,
model-agnostic library, so a new result (a new model, a new dataset, a new concept) is written
once instead of adapted from whichever of the three happens to be closest. See
[docs/provenance.md](docs/provenance.md) for what each prior analysis contributed and why
this repo's design picks the pieces it does, and [docs/findings.md](docs/findings.md) for
what they found, where they agree, and where they don't yet.

## Start here

- **[docs/findings.md](docs/findings.md)** — consolidated results across all three prior
  analyses: what's load-bearing, what's still an open discrepancy, and what to run next.
- **[docs/methodology.md](docs/methodology.md)** — the standing rules this codebase enforces
  (patient-level splits, SAE convergence checks, similarity/stability floors, demographic
  confound checks) and *why*, each backed by a specific result that would have looked solid
  without it.
- **[docs/data.md](docs/data.md)** — exactly which dataset versions this consolidates, and
  how to check a fresh pull matches them (`scripts/verify_data.py`).
- **[docs/models.md](docs/models.md)** — which models are wrapped, what's verified about each,
  and what's still open (ECG-JEPA's random-init control; other candidate models to wrap next).
- **[docs/provenance.md](docs/provenance.md)** — where each module's code came from, and one
  real bug this consolidation found and fixed while generalizing across models.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Data (fully open access, no PhysioNet account needed):
bash scripts/download_ptbxl.sh
bash scripts/download_ptbdb.sh
python3 scripts/verify_data.py ptbxl
python3 scripts/verify_data.py ptbdb

# Models (each clones its source repo and fetches weights):
bash scripts/setup_ecgfounder.sh
bash scripts/setup_clef.sh medium   # or: small / large
bash scripts/setup_ecgjepa.sh
```

Run the test suite (pure-math/pure-data checks, no data or model download needed):

```bash
pytest tests/
```

## Structure

```
src/ecg_mechinterp/
  config.py            one Config for a run: dataset, encoders, SAE/analysis hyperparameters
  models/               ECGModel interface + ECGFounder/CLEF/ECG-JEPA wrappers + the encoder
                         registry (build_encoder_registry, EncoderWrapper, "_rand" controls)
  data/                  PTB-XL / PTB-DB loading, windowing, patient-level splits
  activation/             forward-hook activation extraction (pooled and token-level)
  probing/                 linear probing with patient-level bootstrap CIs
  similarity/               CKA/Procrustes/SVCCA/kNN, subspace overlap + stable rank, controls
  sae/                       TopK/L1 sparse autoencoder, training, cross-seed stability
  causal/                     gradient attribution + feature ablation
  bias/                        the demographic-pathway test
  evaluation.py                 window -> record -> patient score aggregation
scripts/
  download_ptbxl.sh / download_ptbdb.sh       version-pinned dataset downloads
  setup_clef.sh / setup_ecgfounder.sh / setup_ecgjepa.sh    model source + weights
  verify_data.py                                 checks a pull against docs/data.md
  run_depth_profile.py                            per-layer, per-concept probing
  run_similarity_sweep.py                          RQ1: cross-encoder representation similarity
  train_sae_sweep.py                                trains the SAE grid + convergence check
  run_stability_suite.py                             RQ2/RQ3: feature + subspace stability
  run_causal_ablation.py                              RQ4: stability vs. causal importance
  run_demographic_bias_check.py                        the flagship bias/demographic test
```

## Adding a new model

Follow the pattern in `src/ecg_mechinterp/models/ecgfounder.py`: implement the `ECGModel`
interface (`models/base.py`), verify preprocessing/hook points against the model's actual
source before writing the wrapper — don't guess layer names or input format — add it to
`models/registry.py::_BUILDERS`, and add a row to [docs/models.md](docs/models.md) with its
license/access status. This is the concrete next step for loading additional models and
checking whether they reproduce the findings in docs/findings.md.

## Adding a new dataset

Implement a `load_<name>_windows(cfg) -> (X, win_meta)` function in `data/loader.py` following
`load_ptbdb_windows`/`load_ptbxl_windows`'s contract (see `data/loader.py`'s module docstring
for the exact shape), and register it in `load_windows`'s dispatch.

## Status

This is a from-scratch consolidation, not a copy — most modules are ported and generalized
from the three source analyses (docs/provenance.md), with one real cross-model bug found and
fixed in the process (also docs/provenance.md). The pure-math/pure-data-structure logic
(similarity metrics, subspace comparison, SAE training/stability, probing, splits,
evaluation) is covered by `tests/` and passes. The model-loading and full pipeline scripts
are wired up and pass a structural smoke test (encoder construction, hooking, and activation
extraction shapes, verified against the CLEF SAE convergence study's independently-documented
per-stage shapes and parameter counts — see docs/provenance.md) but have **not yet been run
end-to-end against real downloaded weights and data** in this environment — do that before
trusting a number out of `scripts/` for the first time on a new machine, the same as you would
for any freshly assembled pipeline.
