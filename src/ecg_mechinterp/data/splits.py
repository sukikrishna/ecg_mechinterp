"""Patient-level splitting.

Every split in this package is at the patient level, never the window level: a probe or SAE
that saw a patient's other windows during training and is then evaluated on a held-out window
from the *same* patient measures memorization, not generalization. Both source studies
independently arrived at this rule (see docs/methodology.md) and it is treated here as
non-negotiable, not a tunable choice.

`make_patient_splits` generalizes the RQ1-4 notebook's stratify-on-(diagnosis, sex) train/
val/test split to an arbitrary set of stratification columns, and adds two disjoint halves of
the training patients (`train_a`/`train_b`) for the within-encoder noise ceiling every
similarity/subspace comparison in `similarity/` is calibrated against.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_patient_splits(
    win_meta: pd.DataFrame,
    patient_col: str = "patient",
    stratify_cols: Sequence[str] = ("dx_group", "sex"),
    seed: int = 0,
    frac: Tuple[float, float, float] = (0.6, 0.15, 0.25),
) -> Dict[str, np.ndarray]:
    """Returns {split_name: array of row indices into win_meta}, split_name in
    {"train", "val", "test", "train_a", "train_b"}. Rare strata (fewer than 4 patients) are
    pooled into a single "RARE" bucket so `train_test_split`'s stratification doesn't error."""
    pat = win_meta.drop_duplicates(patient_col)[[patient_col, *stratify_cols]].copy()
    pat["strat"] = pat[list(stratify_cols)].astype(str).agg("_".join, axis=1)
    counts = pat["strat"].value_counts()
    pat.loc[pat["strat"].isin(counts[counts < 4].index), "strat"] = "RARE"

    train_pat, tmp_pat = train_test_split(
        pat, train_size=frac[0], stratify=pat["strat"], random_state=seed
    )
    rel = frac[1] / (frac[1] + frac[2])
    counts2 = tmp_pat["strat"].value_counts()
    strat2 = tmp_pat["strat"].where(tmp_pat["strat"].isin(counts2[counts2 >= 2].index), "RARE")
    val_pat, test_pat = train_test_split(tmp_pat, train_size=rel, stratify=strat2, random_state=seed)

    sets = {"train": set(train_pat[patient_col]), "val": set(val_pat[patient_col]),
            "test": set(test_pat[patient_col])}
    splits = {k: win_meta.index[win_meta[patient_col].isin(v)].to_numpy() for k, v in sets.items()}

    tr_pats = np.array(sorted(sets["train"]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr_pats)
    half_a, half_b = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
    splits["train_a"] = win_meta.index[win_meta[patient_col].isin(half_a)].to_numpy()
    splits["train_b"] = win_meta.index[win_meta[patient_col].isin(half_b)].to_numpy()

    assert not (sets["train"] & sets["test"]), "train/test patient overlap"
    assert not (sets["train"] & sets["val"]), "train/val patient overlap"
    return splits


def held_out_split(
    win_meta: pd.DataFrame,
    patient_col: str = "patient",
    stratify_col: Optional[str] = None,
    held_out_frac: float = 0.20,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Simpler two-way (train/held_out) patient split, for SAE-only work that doesn't need a
    separate validation set. Equivalent to the CLEF SAE convergence study's `build_split`."""
    pat = win_meta.drop_duplicates(patient_col)
    strata = pat[stratify_col] if stratify_col else pd.Series("all", index=pat.index)
    rng = np.random.default_rng(seed)
    train, held = [], []
    for _, grp in strata.groupby(strata):
        subs = np.array(sorted(pat.loc[grp.index, patient_col]))
        rng.shuffle(subs)
        n_held = max(1, int(round(len(subs) * held_out_frac)))
        held += subs[:n_held].tolist()
        train += subs[n_held:].tolist()
    return {
        "train": win_meta.index[win_meta[patient_col].isin(train)].to_numpy(),
        "held_out": win_meta.index[win_meta[patient_col].isin(held)].to_numpy(),
    }
