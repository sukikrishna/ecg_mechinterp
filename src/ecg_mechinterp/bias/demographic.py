"""The demographic-pathway test: is a clinical concept partly a demographic concept?

This is the flagship finding across all three source analyses (docs/findings.md) — first
correlational (age-matching removes a large chunk of MI decodability on PTB-DB), then causal
(ablating the features most attributed to a *protected* concept measurably drops a *target*
clinical concept's AUROC, more than a density-matched control does). Generalized here from the
original sex-vs-MI-on-PTB-DB test to any (protected_concept, target_concept) pair so it can be
re-run on PTB-XL against MI and LVH — both age-skewed diagnoses that were never checked this
way in the original PTB-XL study (see docs/findings.md's discrepancy #2).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from ecg_mechinterp.causal.ablation import ablation_effect, density_matched_control
from ecg_mechinterp.causal.attribution import TorchProbe
from ecg_mechinterp.models.registry import EncoderWrapper
from ecg_mechinterp.sae.model import SAE


def demographic_pathway_test(
    enc: EncoderWrapper, sae: SAE, layer: str,
    protected_density: np.ndarray, top_features: Sequence[int],
    probes: Dict[str, TorchProbe], eval_windows: Dict[str, np.ndarray], eval_labels: Dict[str, np.ndarray],
    subgroup: np.ndarray = None, time_pool: int = 8, device: str = "cpu", seed: int = 0,
) -> pd.DataFrame:
    """`top_features`: the protected concept's top-attribution feature ids (from
    causal.attribution.feature_attribution on the *protected* probe). `probes`/`eval_windows`/
    `eval_labels`: one entry per concept to *measure* after ablation (must include the
    protected concept itself, as the check that the intervention worked at all, and the
    target clinical concept(s) that are the actual question).

    Returns one row per (ablated set, measured concept): `ablate_top_<protected>_features` vs.
    `ablate_density_matched_control`, each measured on every concept in `probes`. Read it as:
    - the protected-concept row with a large `auroc_drop` confirms the intervention worked;
      a small drop there means the rest of the table is uninformative.
    - a target-concept row with a larger `auroc_drop` than its density-matched-control row is
      evidence that target concept's decoding partly runs through the protected pathway.
    - `auroc_drop_<subgroup>` columns (if `subgroup` is given) show whether that pathway is
      used asymmetrically across subgroups — the fairness-relevant asymmetry.
    """
    control = density_matched_control(top_features, protected_density, seed=seed)
    rows = []
    for label, feats in [("ablate_top_features", list(top_features)),
                         ("ablate_density_matched_control", control)]:
        for concept, probe in probes.items():
            x, y = eval_windows[concept], eval_labels[concept]
            eff = ablation_effect(enc, probe, sae, layer, feats, x, y, time_pool=time_pool, device=device)
            row = dict(ablated=label, measured_on=concept, **{k: round(v, 4) for k, v in eff.items()})
            if subgroup is not None and len(np.unique(y)) > 1:
                for grp in np.unique(subgroup):
                    m = subgroup == grp
                    if m.sum() > 20 and len(np.unique(y[m])) > 1:
                        base_grp = ablation_effect(enc, probe, sae, layer, [], x[m], y[m],
                                                   time_pool=time_pool, device=device)
                        abl_grp = ablation_effect(enc, probe, sae, layer, feats, x[m], y[m],
                                                  time_pool=time_pool, device=device)
                        if "auroc_base" in base_grp and "auroc_base" in abl_grp:
                            row[f"auroc_drop_{grp}"] = round(base_grp["auroc_base"] - abl_grp["auroc_base"], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y after removing the linear effect of rank(z) — used to
    check whether a stability-vs-causal-importance correlation (RQ4) survives controlling for
    activation density, since density is a plausible common cause of both."""
    from scipy import stats as sps

    rx, ry, rz = (sps.rankdata(v) for v in (x, y, z))

    def resid(a, b):
        fit = np.polyfit(b, a, 1)
        return a - np.polyval(fit, b)

    return float(sps.spearmanr(resid(rx, rz), resid(ry, rz)).statistic)
