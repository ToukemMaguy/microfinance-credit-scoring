"""Reusable helpers for the credit scoring notebook: binning, WOE/IV, validation metrics, scorecard points."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- binning, WOE and IV
def make_bins(x: pd.Series, edges: list[float] | None = None, n_bins: int = 5) -> list[float]:
    """Right-closed bin edges: given edges, or quantiles of x. Always open on both ends."""
    if edges is None:
        qs = np.unique(np.nanquantile(x.dropna(), np.linspace(0, 1, n_bins + 1)))
        edges = list(qs[1:-1])
    return [-np.inf] + sorted(edges) + [np.inf]


def to_bins(x: pd.Series, bins: list[float]) -> pd.Series:
    """Label of the bin of each value; missing values get their own class, 'Missing'."""
    labels = pd.cut(x, bins).astype(str)
    return labels.where(x.notna(), "Missing")


def woe_table(bin_labels: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Weight of evidence and information value per bin (0.5 smoothing to avoid empty cells)."""
    t = pd.DataFrame({"bin": bin_labels.values, "y": y.values}).groupby("bin")["y"].agg(n="size", bad="sum")
    t["good"] = t["n"] - t["bad"]
    k = len(t)
    pct_good = (t["good"] + 0.5) / (t["good"].sum() + 0.5 * k)
    pct_bad = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5 * k)
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log(pct_good / pct_bad)
    t["iv"] = (pct_good - pct_bad) * t["woe"]
    return t


def iv_strength(iv: float) -> str:
    return "useless" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3 else "strong" if iv < 0.5 else "suspicious"


# --------------------------------------------------------------------------- validation metrics
def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def gini_ci(y, p, n_boot: int = 300, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval of the Gini coefficient."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    values = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            values.append(gini(y[i], p[i]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """Population stability index between the training and test score distributions."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges).value_counts(normalize=True, sort=False) + 1e-6
    a = pd.cut(actual, edges).value_counts(normalize=True, sort=False) + 1e-6
    return float(((a - e) * np.log(a / e)).sum())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval of a proportion k / n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


# --------------------------------------------------------------------------- scorecard scaling
def scaling(base_score: int = 600, base_odds: float = 10, pdo: int = 20) -> tuple[float, float]:
    """Factor and offset such that base_score points = base_odds goods per bad, and +pdo points doubles the odds."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset
