# Consolidated findings

This page merges the results of three prior analyses — see [docs/provenance.md](provenance.md)
for what each one is and which code came from where. They are referred to here by what they
studied, not by codename:

- **the PTB-XL multi-model study**: ECGFounder + CLEF-medium + ECG-JEPA, probing/CKA/SAE, on
  the full PTB-XL dataset (21,799 records, 5 concepts).
- **the CLEF SAE convergence study**: CLEF-medium only, on PTB-DB (549 records / 290
  subjects), a deep dive into SAE feature reproducibility and training-budget effects.
- **the RQ1-4 stability-and-bias notebook**: CLEF-S/M/L plus random-init controls, on the
  same PTB-DB cohort, a structured shared-representation -> feature-stability ->
  subspace-stability -> causal-importance -> demographic-bias investigation.

## Where all three agree (the load-bearing findings)

### 1. A clinical concept can be substantially a demographic concept — correlationally and now causally

The single strongest, most escalating result across the three. The CLEF SAE convergence
study found MI-vs-HC decoding on PTB-DB was substantially an age effect: age-matching a
subset (HC 40.2y vs. MI 59.4y, unmatched) removed about 0.11 AUROC at every layer. The RQ1-4
notebook then made this causal: ablating the SAE features most attributed to a *sex* probe
dropped MI-probe AUROC by 0.27, versus only 0.11 for a density-matched random control — a
demographic pathway measured directly, not inferred from a confound table. See
docs/methodology.md rule 5 and `bias/demographic.py`, which generalizes this exact test.

### 2. Reconstruction quality (EV/FVU) is not a valid SAE convergence criterion

The CLEF SAE convergence study's central finding, from an actual training-budget sweep
(12/40/120/360 epochs): EV moved from 0.986 to 0.998 (a 0.012 change) while cross-seed MMCS
moved from 0.267 to 0.564 (a 0.297 change) over the same range — reconstruction saturates in
exactly the region a stability claim gets decided. The PTB-XL multi-model study's own SAE
pass (2,500-record sample) independently flagged its own result as likely undertrained for
the same reason, without yet having run the sweep that would confirm it. See
docs/methodology.md rule 2 and `sae/train.py::track_convergence`, which operationalizes this
check as a reusable step rather than something to re-derive per study.

### 3. Stability lives in a subspace, but only the *right* subspace

The CLEF SAE convergence study found individual SAE features reach 0.564 cross-seed cosine
agreement (well-trained) while a *probe-selected* feature subspace stays near-orthogonal
(69°, chance ~78°) — but PCA subspaces on the identical activations (no label-driven
selection) reproduce to within 2.4°. The RQ1-4 notebook's stable-rank result (32 at a PCA-
defined, cross-encoder overlap threshold) is the same idea extended across encoders. The
PTB-XL multi-model study's Finding 3 (individual feature match ~0.18, subspace angle ~77.6°,
"not clearly more stable than individual features") used a probe/decoder-selected subspace at
only 2,500 samples — the same setup the other two analyses found gives a misleadingly
unstable-looking answer. See docs/methodology.md rule 6.

### 4. Raw CKA needs a random-init floor before it means anything

The PTB-XL multi-model study found the counter-intuitive result that a genuinely different
architecture (ECGFounder vs. ECG-JEPA) scores *higher* CKA than same-architecture pairs
(ECGFounder vs. CLEF), with no random-init control run alongside it. The RQ1-4 notebook added
exactly that control and found two random-init CLEF variants already share CKA 0.864 — most
of what a pretrained cross-model CKA number looks like is architecture prior, not learned
content. This strengthens the caution the PTB-XL study already had about over-reading CKA,
but see discrepancy #1 below for why it doesn't yet settle the architecture question.

### 5. Small-N cohorts (PTB-DB) are underpowered for exactly the claims that matter most

Both PTB-DB analyses flag this explicitly (fold sd ≈ 0.11 at n=54 age-matched; "non-monotonic
and simply noisy at n=200"). The PTB-XL multi-model study's probing/CKA numbers don't have
this problem (n=21,799) — but its SAE pass does (still on a 2,500-record sample). See
docs/data.md and the recommendation below.

## Where they disagree — not yet resolved, and why

### Discrepancy 1 — "cross-encoder" means different things in different analyses

The RQ1-4 notebook's rigorous random-encoder-floor methodology never actually tests the PTB-XL
multi-model study's most surprising claim: its "cross-encoder" comparisons are CLEF-S vs.
CLEF-M vs. CLEF-L — one architecture family at different sizes — never a truly different
architecture. The PTB-XL study's "different architecture" claim is ECGFounder (CNN) vs.
ECG-JEPA (transformer), and that comparison has never had a random-init floor run against it.
**Resolution requires**: running the notebook's random-floor method on the real
ECGFounder/ECG-JEPA pair — see docs/provenance.md's follow-up list item 2, blocked today only
by ECG-JEPA's missing random-init control (docs/models.md).

### Discrepancy 2 — the depth-vs-decodability shape conflicts on the same model

On PTB-XL (5 concepts), CLEF's first_conv layer is near chance (~0.50 AUROC) and AUROC rises
sharply through early layers before a mild late decline — the PTB-XL study's "temporal
pooling causes late decline" explanation. On PTB-DB (MI-vs-HC only), the CLEF SAE convergence
study found CLEF's first-conv (unmatched) actually *beats* its last layer (0.956 vs. 0.924),
and that inversion disappears entirely once age-matched — i.e., a demographic-confound
explanation, not a pooling one, for what looked like a depth effect. The PTB-XL study never
ran the age/sex-confound check on its own depth-decline claim for MI (or LVH, similarly
age-skewed). **These aren't necessarily contradictory** — different datasets, different label
granularity — but the pooling explanation for MI/LVH specifically hasn't survived the same
confound check that overturned the naive depth story on PTB-DB. **Resolution requires**:
rerunning the PTB-XL depth profile for MI and LVH on an age/sex-matched subset — now directly
supported by `data/ptbxl.py::PTBXL.demographics` and `bias/demographic.py` (docs/provenance.md
follow-up item 3).

### Discrepancy 3 — SAE stability magnitudes may not be apples-to-apples

The CLEF SAE convergence study's most-trained (360-epoch) result is MMCS = 0.564 raw. The
RQ1-4 notebook's reported cross-seed MMCS null-adjusted is 0.66 (raw 0.877, null 0.217) — a
different normalization, and not confirmed to be trained to the same budget. **Resolution
requires**: checking the notebook's SAE training step count against
`sae/train.py::track_convergence`'s divergence curve before treating both numbers as
comparably converged (docs/provenance.md follow-up item 4).

### Discrepancy 4 — a stable-rank result may itself be a sample-size artifact

The RQ1-4 notebook's own within-encoder stable-rank result (0, on disjoint n=290 patient
halves) is flagged in the notebook as likely a data-size artifact rather than a real finding.
The PTB-XL study's Finding 3 (n=2,500) has the identical vulnerability and hasn't been rerun
at full PTB-XL scale either. **Resolution requires**: rerunning both the SAE pass and the
stable-rank comparison on full PTB-XL (docs/data.md).

## Recommendation

1. **Lead with finding #1** (demographic entanglement) — it's correlational and causal, the
   strongest result available.
2. **Gate every SAE number** (from any of the three analyses) through the convergence check in
   docs/methodology.md rule 2 before quoting it — including the RQ1-4 notebook's own numbers
   (discrepancy #3).
3. **Move the SAE, causal-ablation, and demographic-bias pipeline to full PTB-XL** — resolves
   discrepancies #2 and #4 at once, and is independently motivated by PTB-DB's small-N
   problem (finding #5). `data/ptbxl.py` and `bias/demographic.py` already support this; it
   hasn't been run yet.
4. **Finish ECG-JEPA's random-init control**, then rerun the random-floor CKA check on the
   real ECGFounder/ECG-JEPA pair — resolves discrepancy #1, the actual test of whether
   architecture matters that's been missing from both analyses that touched it.
