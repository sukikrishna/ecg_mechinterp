# Provenance: where this codebase came from

This package consolidates three prior, independently-run analyses into one dataset-agnostic,
model-agnostic library. None of the three is "the" reference implementation — each got
something importantly right that the others didn't, which is the actual reason to merge them
rather than pick one. This page names them descriptively (by what they studied) rather than
by codename, and maps every module here back to which analysis it came from and why.

## The three source analyses

- **The PTB-XL multi-model study** — probing, cross-model CKA, and a first SAE pass across
  three encoders (ECGFounder, CLEF-medium, ECG-JEPA) on the full PTB-XL dataset (21,799
  records). Its `src/ecg_interp/` package is the direct ancestor of this repo's `models/`,
  `data/ptbxl.py`, `activation/extraction.py`'s simple path, and `probing/probe.py`'s
  single-split `linear_probe`. Strongest at: large-N probing/CKA with real statistical power,
  and the most careful per-model preprocessing verification (see docs/models.md).
- **The CLEF SAE convergence study** — a single-model (CLEF-medium), single-dataset (PTB
  Diagnostic ECG Database, 549 records / 290 subjects) deep dive into whether SAE features
  reproduce across seeds. Its `src/` is the direct ancestor of this repo's
  `sae/model.py`'s geometric-median init and AuxK loss, `sae/train.py::track_convergence`,
  `data/windows.py`'s validated filter chain, and `data/ptbdb.py`'s demographic-confound
  check. Strongest at: the single most load-bearing methodological finding this repo
  encodes — that reconstruction quality is not a valid SAE convergence criterion (see
  docs/methodology.md rule 2) — established with an actual training-budget sweep on real
  data, not asserted.
- **The RQ1-4 stability-and-bias notebook** — a structured investigation (shared
  representation -> feature stability -> subspace stability -> causal importance -> public-
  health reading) across CLEF's three sizes plus random-init controls, on the same PTB-DB
  cohort. Its cells are the direct ancestor of this repo's `similarity/metrics.py` (linear
  CKA, RBF CKA, Procrustes, SVCCA, kNN overlap — the most complete similarity toolkit of the
  three), `similarity/subspace.py`'s stable-rank machinery, `causal/` (gradient attribution +
  ablation), and `bias/demographic.py`. Strongest at: methodological rigor — it is the only
  one of the three with random-encoder controls, null-adjusted metrics, and an actual causal
  (not just correlational) test throughout.

See [docs/findings.md](findings.md) for what each one found and where they agreed/disagreed;
this page is about the code, not the results.

## Specific facts this codebase relies on, and where they were checked

- **CLEF-medium's exact Net1D config and 30,654,000-parameter count**
  (`models/clef.py::_NET1D_CONFIGS["medium"]`): independently confirmed by the CLEF SAE
  convergence study via a strict `state_dict` load (`strict=True`, zero missing/unexpected
  keys) against the real checkpoint, an exact parameter-count assertion, and a "weights
  actually changed after load" check. Re-verified when this repo was assembled: instantiating
  `net1d.Net1D` with this config gives exactly 30,654,000 parameters (see the smoke test this
  repo's `models/registry.py` and `models/clef.py` were built against).
- **CLEF-small/-large's configs** (448K / 296M params): from the CLEF paper (Appendix C.3,
  Table S7), as transcribed by the RQ1-4 notebook — not independently checked against a real
  checkpoint the way medium was, since this repo doesn't have a validated strict-loader for
  those sizes yet. Treat clef_s/clef_l pretrained loading as unverified until someone repeats
  the medium-style strict-load check on those checkpoints specifically.
- **CLEF's per-stage output shapes** at a (B, 1, 5000) input (`(64,1250)`, `(160,625)`, ...,
  `(1024,20)`): from the CLEF SAE convergence study's own shape verification. Reproduced
  exactly by this repo's `EncoderWrapper` in the smoke test used to validate the
  `models/registry.py` rewrite (see the ECGFounder-1lead shapes, which follow the same Net1D
  config).
- **The PTB-DB record/subject counts (549/290) and MI/HC/no-summary breakdown
  (148/52/22)**: reported independently by both PTB-DB analyses via different acquisition
  code paths (a streamed zip reader vs. `wfdb` against a local or PhysioNet-streamed copy) —
  see docs/data.md. `scripts/verify_data.py` checks a fresh pull against these numbers.

## A bug this consolidation found and fixed

Generalizing the RQ1-4 notebook's `EncoderWrapper` (which only ever wrapped CLEF, whose
pretrained loader swaps its classification head for `nn.Identity`) to also wrap ECGFounder
(whose wrapper never does that swap) exposed that the notebook's convention — treating the
encoder's raw `forward()` output as "the pooled embedding" — is only correct for CLEF. Applied
naively to ECGFounder, it would have silently probed on 150-way classification logits instead
of the backbone feature. `models/registry.py::EncoderWrapper.forward` now derives the pooled
embedding from the last hooked layer's activation (time-pooled) instead, which is correct
regardless of whether a model's head was swapped, and was confirmed in this repo's smoke tests
to reproduce CLEF's original behavior exactly while fixing ECGFounder's. Worth remembering
when wrapping a fourth model: check what its raw `forward()` actually returns before trusting
it as "the embedding" anywhere in `activation/` or `causal/`.

## Known follow-up work (not yet done, deliberately not guessed at)

1. **ECG-JEPA random-init control** — see docs/models.md. Needs reading `models.py` in the
   real ECG-JEPA repo, not guessing at transformer hyperparameters.
2. **Re-test the "architecture doesn't determine representational similarity" claim
   properly** — the PTB-XL multi-model study's ECGFounder-vs-ECG-JEPA CKA result never had a
   random-encoder-floor control; the RQ1-4 notebook's random-encoder-floor methodology never
   touched a truly different architecture (its "cross-encoder" comparisons are all CLEF-S vs.
   CLEF-M vs. CLEF-L, one architecture family). Combining the two — random-floor CKA on the
   real ECGFounder/ECG-JEPA pair, at PTB-XL scale — is the actual test and hasn't been run.
   See docs/findings.md's discrepancy #1.
3. **Re-run the demographic-pathway test on PTB-XL**, for MI and LVH (both age-skewed
   diagnoses), now that `data/ptbxl.py::PTBXL.demographics` and `bias/demographic.py` support
   it generically. See docs/findings.md's discrepancy #2 for why the PTB-XL multi-model
   study's depth-decline explanation (temporal pooling) was never checked against this
   alternative (demographic confound) explanation.
4. **Confirm the RQ1-4 notebook's reported SAE stability numbers (e.g. cross-seed MMCS
   null-adjusted 0.66) against the training-budget check in rule 2** — cross-reference the
   step count those numbers were trained at against `sae/train.py::track_convergence`'s
   divergence curve before treating them as converged. See docs/findings.md's discrepancy #3.
5. **CLEF-small/-large strict-load verification** (see above) — currently unverified.
