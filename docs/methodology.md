# Standing methodological rules

These are not style preferences — each one exists because a specific prior result looked
solid and turned out to be an artifact once checked. The code in this repo enforces or
documents each rule at the point it applies; this page is the one place they're stated
together with the reasoning, so a new analysis can be checked against them before its numbers
get quoted anywhere. See [docs/findings.md](findings.md) for the results that established
each rule.

## 1. Every split is at the patient level, never the window level

A probe or SAE that trains on some of a patient's windows and is evaluated on other windows
from the *same* patient measures memorization, not generalization — consecutive windows from
one recording are close to identical. `data/splits.py` enforces this for every split this
package produces; `evaluation.py` and `probing/probe.py::patient_bootstrap` enforce it again
at the *scoring* step, since even a correctly patient-split probe reports a misleadingly tight
confidence interval if its uncertainty is estimated by resampling windows instead of patients.

## 2. Reconstruction quality is not a valid SAE convergence criterion

`EV > 0.85` (equivalently, `FVU < 0.15`) is a common SAE training gate, and it saturates long
before cross-seed feature *identity* does. In the reference sweep
(`sae/train.py::track_convergence`, run on real data in the CLEF SAE convergence study),
across a 30x increase in training steps, EV moved 0.012 while cross-seed MMCS moved 0.297 —
the exact region where a stability claim gets decided is the region reconstruction error is
blind to. Concretely: **run `track_convergence` on any new encoder/layer/dataset before
picking a training budget for a stability study.** Don't reuse a budget validated on a
different setup without rechecking.

## 3. A raw similarity/stability number needs a floor, always

- **Random-encoder floor** (`models/registry.py`'s `_rand` convention): two
  architecture-matched, randomly-initialized encoders already share substantial CKA purely
  from architecture — the RQ1-4 notebook measured 0.864 between two random CLEF variants.  A
  pretrained cross-model CKA has to be read against this, not against 0.
- **Random-dictionary floor** (`sae/stability.py::random_dictionary_null`): two untrained SAE
  dictionaries of the same shape already share nontrivial MMCS purely from dimensionality.
- **Permutation/shuffled-token null** (`sae/stability.py::permuted_pairs`,
  `similarity/controls.py`): the floor for any matched-pair correlation once sample order is
  destroyed.
- **Within-encoder ceiling** (`similarity/controls.py::within_encoder_ceiling`): the same
  encoder/layer against itself on two disjoint patient halves — the *highest* similarity
  achievable given finite data. Cross-encoder similarity is best read as a fraction of this,
  not as an absolute number.

A number from `similarity/` or `sae/stability.py` without at least one of these attached
should not be quoted in a write-up.

## 4. A causal claim needs a density-matched control, not just a before/after

Removing *any* set of active SAE features tends to move a probe's score somewhat. That a
targeted ablation (e.g. the features most attributed to a *different* concept) moves it *more*
than an equally-sized, density-matched random ablation is the actual evidence — see
`causal/ablation.py::density_matched_control` and `bias/demographic.py`. Report both rows
together; a targeted-ablation number alone is not interpretable.

## 5. Demographic confounds get checked before a clinical-concept result is trusted

Both PTB-DB analyses independently found the same pattern for MI: a large chunk of a
"clinical" probe's accuracy was actually age (and, causally, sex), not cardiology-specific
signal. Any new clinical-concept probe on any dataset should get the same check before its
number is treated as a disease-specific finding:
1. `data/ptbdb.py::mi_vs_hc_confound_test` (or the PTB-XL equivalent, using
   `data/ptbxl.py::PTBXL.demographics`) — is the concept confounded with age/sex in this
   cohort at all?
2. If yes: age/sex-match a subset and re-run the probe (correlational check), and run
   `bias/demographic.py::demographic_pathway_test` (causal check) before describing the
   result as disease-specific.

## 6. Individual-direction stability and subspace stability are different questions

A direction (an SAE feature, a probe weight vector) can be unstable across seeds/resamples
while the *subspace* spanned by many such directions is highly reproducible, or the reverse.
Always compute both (`probing/probe.py::probe_direction_stability` +
`similarity/subspace.py::stable_rank_table`) rather than assuming one predicts the other — see
docs/findings.md's RQ3 discussion for a case where they diverged sharply and a case where a
subspace-instability result turned out to be a sample-size artifact, not a real finding.
