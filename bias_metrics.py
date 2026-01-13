from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else float("-inf")


def _normalize_probs(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    total = float(p.sum())
    if total <= 0:
        return p * 0.0
    return p / total


def jensen_shannon_divergence(p: pd.Series, q: pd.Series) -> float:
    keys = sorted(set(p.index).union(set(q.index)))
    p2 = _normalize_probs(p.reindex(keys, fill_value=0.0))
    q2 = _normalize_probs(q.reindex(keys, fill_value=0.0))
    m = 0.5 * (p2 + q2)

    def kl(a: pd.Series, b: pd.Series) -> float:
        s = 0.0
        for k in a.index:
            ai = float(a[k])
            bi = float(b[k])
            if ai <= 0:
                continue
            if bi <= 0:
                return float("inf")
            s += ai * (_safe_log(ai) - _safe_log(bi))
        return float(s)

    return 0.5 * kl(p2, m) + 0.5 * kl(q2, m)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    eps = 1e-12
    ps = float(np.sum(p))
    qs = float(np.sum(q))
    if ps <= 0 or qs <= 0:
        return float("nan")
    p2 = p / ps
    q2 = q / qs
    p2 = np.clip(p2, eps, 1.0)
    q2 = np.clip(q2, eps, 1.0)
    p2 = p2 / float(np.sum(p2))
    q2 = q2 / float(np.sum(q2))

    m = 0.5 * (p2 + q2)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a) - np.log(b))))

    return 0.5 * kl(p2, m) + 0.5 * kl(q2, m)


def cramer_v(table: pd.DataFrame) -> float:
    if table is None or table.empty:
        return float("nan")

    obs = table.to_numpy(dtype=float)
    n = float(obs.sum())
    if n <= 0:
        return float("nan")

    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = (row_sum @ col_sum) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(((obs - expected) ** 2) / expected)

    r, k = obs.shape
    denom = n * float(max(1, min(r - 1, k - 1)))
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(chi2 / denom))


def mutual_information(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x.astype(str), "y": y.astype(str)})
    joint = df.value_counts(normalize=True)
    px = df["x"].value_counts(normalize=True)
    py = df["y"].value_counts(normalize=True)

    mi = 0.0
    for (xi, yi), pxy in joint.items():
        pxi = float(px.get(xi, 0.0))
        pyi = float(py.get(yi, 0.0))
        if pxy <= 0 or pxi <= 0 or pyi <= 0:
            continue
        mi += float(pxy) * (_safe_log(float(pxy)) - _safe_log(pxi) - _safe_log(pyi))
    return float(mi)


@dataclass(frozen=True)
class AdherenceResult:
    n: int
    accuracy: float


def adherence_rate(*, specified: pd.Series, predicted: pd.Series, na_value: str = "N/A") -> AdherenceResult:
    s = specified.fillna("").astype(str).str.strip()
    p = predicted.fillna("").astype(str).str.strip()
    mask = s.ne("") & s.ne(na_value) & p.ne("") & p.ne(na_value)
    if int(mask.sum()) == 0:
        return AdherenceResult(n=0, accuracy=float("nan"))
    correct = (s[mask].str.lower() == p[mask].str.lower()).sum()
    n = int(mask.sum())
    return AdherenceResult(n=n, accuracy=float(correct) / float(n))
