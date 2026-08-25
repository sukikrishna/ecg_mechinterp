# Models

## Wrapped and wired up in this repo

| Model | Venue | Lab | Code / weights | License | Weight size |
|---|---|---|---|---|---|
| **CLEF** | arXiv | Nokia Bell Labs | [github.com/Nokia-Bell-Labs/ecg-foundation-model](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) | BSD-3-Clause-Clear | Small 448K / Medium 30.7M / Large 296M params |
| **ECGFounder** | NEJM AI 2025 | PKU Digital Health | [github.com/PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) · HF `PKUDigitalHealth/ECGFounder` | MIT | ~370MB per checkpoint (12-lead and 1-lead variants) |
| **ECG-JEPA** | arXiv | S. Kim | [github.com/sehunfromdaegu/ECG_JEPA](https://github.com/sehunfromdaegu/ECG_JEPA) | MIT | 326MB (encoder alone 85.4M params) |

None of the three require an account, application, or signed data-use agreement — verified
directly against each repo.

**CLEF and ECGFounder are the identical backbone.** Both instantiate the exact same `Net1D`
configuration (same `filter_list`/`m_blocks_list` at every stage, verified against both repos'
source) — they differ in training data/objective (ECGFounder: supervised 150-class diagnostic
classification; CLEF: SimCLR-style contrastive pretraining on MIMIC-IV-ECG), not architecture.
Any "shared representation" finding between the two is evidence about training-objective
effects on a fixed architecture, **not** architecture-independence.

**ECG-JEPA is confirmed (by actually instantiating it and checking for `Conv1d`/`Conv2d`) to be
a genuine transformer/JEPA architecture** — multi-head attention blocks, patch embedding via a
plain linear layer (`W_P`), no convolutions anywhere. This is what makes it the model that can
turn the CLEF/ECGFounder comparison into an actual architecture-independence test, instead of a
same-backbone one (see docs/findings.md's discrepancy #1). It expects a specific 8-lead subset
(I, II, V1-V6), 2500 samples (10s at an effective 250Hz), no z-scoring.

**Dependency caution (ECG-JEPA)**: installing its `timm` requirement naively pulls in a CUDA
build of torch and an incompatible torchvision, silently breaking an existing CPU-only install
— `scripts/setup_ecgjepa.sh` pins torch via a constraints file and installs the rest with
`--no-deps`; verify `python3 -c "import torch; print(torch.__version__)"` still shows the
expected build after running it.

## Random-initialization controls

`models/registry.py`'s `<name>_rand` convention (see docs/methodology.md rule 3) needs an
architecture-matched random-init build for each model:

- **ECGFounder**: free — its wrapper already separates building the architecture
  (`__init__`) from loading weights (`load`), so skipping `load()` is already a valid control.
- **CLEF**: `models/clef.py::CLEF.load_random` instantiates the same `net1d.Net1D` class
  directly with CLEF's published per-size config, bypassing the pretrained-loading factory
  entirely. Cross-checked against the real checkpoint's parameter count (30,654,000 for
  medium) in the smoke test that validated this repo — see docs/provenance.md.
- **ECG-JEPA**: **not implemented.** `load_encoder()` conflates building the transformer with
  loading its weights, and this repo hasn't independently verified the transformer's real
  constructor hyperparameters (embed dim, depth, heads, masking config) against its source.
  `ECGJEPA.load_random` raises `NotImplementedError` with this explanation rather than
  guessing at an architecture. Wiring this is real follow-up work: read `models.py` in
  `external/ecg-jepa` directly, instantiate the same encoder class with random weights, and
  verify it against the pretrained checkpoint's parameter count the same way CLEF's was.

## CLEF weight source

Zenodo record [10.5281/zenodo.17572734](https://zenodo.org/records/17572734)
(`clef_small.ckpt` 5.5MB, `clef_medium.ckpt` 368MB, `clef_largel.ckpt` ~3.6GB — the `l` typo
in "largel" is Zenodo's own filename).

## Other candidate models (not yet wrapped)

For Azmine/Advith's next step of loading additional models and checking reproduction —
license/access already verified, none require an account:

| Model | Venue | Lab | Code / weights | License | Notes |
|---|---|---|---|---|---|
| ST-MEM | ICLR 2024 | VUNO Inc. | github.com/vuno/ST-MEM | **Proprietary — VUNO, all rights reserved** | Redistribution/derivative use barred without permission; usable for private read-only analysis at most |
| MERL | ICML 2024 | Imperial College London | github.com/cheliu-computation/MERL-ICML2024 | MIT | Weight size not stated |
| ECG-FM | JAMIA Open 2025 | Bo Wang Lab, U. Toronto / Vector Institute | github.com/bowang-lab/ECG-FM · HF `wanglab/ecg-fm` | MIT | Needs their fairseq-based loader, not vanilla `transformers`; ~1.09GB pretrained, 90.9M params |
| HuBERT-ECG | arXiv | E. Coppola et al. | HF `Edoardo-BS/hubert-ecg-base` | **CC-BY-NC-4.0 — non-commercial only** | Fine for academic publication, flag before any commercial use |
| HeartLang | arXiv | PKU Digital Health | github.com/PKUDigitalHealth/HeartLang | Not yet checked | — |

Before wrapping a new model, follow the pattern in `src/ecg_mechinterp/models/ecgfounder.py`:
implement the `ECGModel` interface (`models/base.py`), verify preprocessing/hook points
against the model's actual source before writing the wrapper — don't guess layer names or
input format — and add a row to this table with its license/access status.

## Interpretability tooling referenced by this project's design

| Library | Use | Link |
|---|---|---|
| `dictionary_learning` (Marks) | TopK/other SAE trainers, activation buffers, HF-hosted dictionaries | github.com/saprmarks/dictionary_learning |
| OpenAI `sparse_autoencoder` | TopK SAEs + feature visualizer | github.com/openai/sparse_autoencoder |
| SAELens | SAE training/analysis suite | github.com/jbloomAus/SAELens |
| `concept-erasure` (EleutherAI) | Closed-form linear concept erasure (LEACE) | github.com/EleutherAI/concept-erasure |
