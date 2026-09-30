# Results: ECGFounder (1-lead) on PTB-DB

The SAE half of the pipeline, run on PTB-DB 1.0.0 (549 records, 290 patients, 5,724 ten-second
lead I windows, patient-level splits). The depth profile and similarity results for this
encoder are in `../results_clef_m/`, since those scripts were run with all encoders together.

- Encoder: `ecgfounder_1lead`, layer `stage_list.4` (relative depth 0.75)
- SAEs: TopK, k = 32, 4x expansion (1,600 features), 5 seeds x 34,920 steps. Same
  configuration as the CLEF-medium run, so the two are directly comparable.

| File | Script |
|---|---|
| `sae_convergence_check_ecgfounder_1lead.csv` | `sae.train.track_convergence`, budgets 1,164 / 3,492 / 11,640 / 34,920, seeds 0 and 1 |
| `sae_quality.csv` | `train_sae_sweep.py --skip-convergence-check` |
| `feature_stability_cross_seed.csv`, `subspace_overlap.csv` | `run_stability_suite.py` |
| `causal_effects.csv` | `run_causal_ablation.py --concept is_mi` |
| `demographic_pathway.csv` | `run_demographic_bias_check.py --protected sex_male --target is_mi` |

The trained SAEs (`sae_store.pt`, 24 MB) are not committed, since `*.pt` is gitignored.

## Caveats

- **Invalid rows.** Every `within_encoder_data_split` row in `subspace_overlap.csv` compares two
  disjoint patient halves row by row. The metric needs the same examples on both sides, so
  these values fall to chance by construction and should not be read.
- **MI target.** `is_mi` is MI vs every other window: 928 healthy-control windows plus other
  cardiac diagnoses and 175 windows from the 22 subjects with no recorded diagnosis. It is not
  MI vs healthy controls.
- **Environment.** Run on an Apple M1 GPU (MPS). That needed small local patches (a `--device`
  flag, and float64 accumulators moved to the host) that are not part of this commit. They do
  not change the math.
- **Scope.** One dataset, one layer, one SAE configuration.
