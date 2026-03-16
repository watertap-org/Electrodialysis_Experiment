from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any

import numpy as np
import pandas as pd


ArrayLike = Union[pd.Series, pd.Index, np.ndarray, Sequence[float]]


# Data alignment / conversion
def _to_1d_array(a: ArrayLike, name: str) -> np.ndarray:
    if isinstance(a, pd.Series):
        arr = a.to_numpy(dtype=float)
    elif isinstance(a, pd.Index):
        arr = a.to_numpy(dtype=float)
    else:
        arr = np.asarray(a, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape}")
    return arr


def align_xy(
    observed: ArrayLike, predicted: ArrayLike, *, dropna: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If both inputs are pandas Series, align on the intersection of indices.
    Otherwise, treat them as array-like and require equal length.
    Drops non-finite pairs by default.
    """
    if isinstance(observed, pd.Series) and isinstance(predicted, pd.Series):
        df = pd.concat(
            [observed.rename("obs"), predicted.rename("pred")], axis=1, join="inner"
        )
        if dropna:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df["obs"].to_numpy(dtype=float), df["pred"].to_numpy(dtype=float)

    x = _to_1d_array(observed, "observed")
    y = _to_1d_array(predicted, "predicted")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"observed and predicted must have same length; got {x.shape[0]} vs {y.shape[0]}"
        )

    if dropna:
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
    return x, y


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


# Common metrics


def r2(observed: ArrayLike, predicted: ArrayLike) -> float:
    """R^2 = 1 - SSE/SST, SST computed against observed mean."""
    x, y = align_xy(observed, predicted)
    if x.size == 0:
        return float("nan")
    sse = float(np.sum((y - x) ** 2))
    sst = float(np.sum((x - np.mean(x)) ** 2))
    return float("nan") if sst == 0 else 1.0 - sse / sst


def mse(observed: ArrayLike, predicted: ArrayLike) -> float:
    x, y = align_xy(observed, predicted)
    return float("nan") if x.size == 0 else float(np.mean((y - x) ** 2))


def rmse(observed: ArrayLike, predicted: ArrayLike) -> float:
    x, y = align_xy(observed, predicted)
    return float("nan") if x.size == 0 else float(np.sqrt(np.mean((y - x) ** 2)))


def mae(observed: ArrayLike, predicted: ArrayLike) -> float:
    x, y = align_xy(observed, predicted)
    return float("nan") if x.size == 0 else float(np.mean(np.abs(y - x)))


def bias(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Mean error (pred - obs)."""
    x, y = align_xy(observed, predicted)
    return float("nan") if x.size == 0 else float(np.mean(y - x))


def mape_pct(observed: ArrayLike, predicted: ArrayLike, *, eps: float = 1e-12) -> float:
    """Mean absolute percentage error, in %."""
    x, y = align_xy(observed, predicted)
    if x.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(x), eps)
    return float(np.mean(np.abs((y - x) / denom)) * 100.0)


def smape_pct(
    observed: ArrayLike, predicted: ArrayLike, *, eps: float = 1e-12
) -> float:
    """Symmetric MAPE, in %."""
    x, y = align_xy(observed, predicted)
    if x.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(x) + np.abs(y), eps)
    return float(np.mean(2.0 * np.abs(y - x) / denom) * 100.0)


def pearson_r(observed: ArrayLike, predicted: ArrayLike) -> float:
    x, y = align_xy(observed, predicted)
    if x.size < 2:
        return float("nan")
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    den = float(np.sqrt(np.sum(x0**2) * np.sum(y0**2)))
    return _safe_div(float(np.sum(x0 * y0)), den)


def spearman_r(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Spearman rank correlation (average ranks for ties; no scipy)."""
    x, y = align_xy(observed, predicted)
    if x.size < 2:
        return float("nan")

    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, a.size + 1, dtype=float)
        sorted_a = a[order]
        i = 0
        while i < sorted_a.size:
            j = i
            while j + 1 < sorted_a.size and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                avg = (i + 1 + j + 1) / 2.0
                ranks[order[i : j + 1]] = avg
            i = j + 1
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_r(rx, ry)


def regression_line(predicted: ArrayLike, observed: ArrayLike) -> Tuple[float, float]:
    """
    Fit predicted = intercept + slope * observed (OLS).
    Returns (intercept, slope).
    """
    x, y = align_xy(observed, predicted)
    if x.size < 2:
        return float("nan"), float("nan")
    xm, ym = float(np.mean(x)), float(np.mean(y))
    num = float(np.sum((x - xm) * (y - ym)))
    den = float(np.sum((x - xm) ** 2))
    slope = _safe_div(num, den)
    intercept = ym - slope * xm
    return intercept, slope


def concordance_ccc(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Lin's concordance correlation coefficient."""
    x, y = align_xy(observed, predicted)
    if x.size < 2:
        return float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    return _safe_div(2.0 * cov, vx + vy + (mx - my) ** 2)


def coverage_within_abs(observed: ArrayLike, predicted: ArrayLike, tol: float) -> float:
    """Fraction with |pred-obs| <= tol."""
    x, y = align_xy(observed, predicted)
    return float("nan") if x.size == 0 else float(np.mean(np.abs(y - x) <= tol))


def coverage_within_pct(
    observed: ArrayLike, predicted: ArrayLike, pct: float, *, eps: float = 1e-12
) -> float:
    """Fraction with |pred-obs|/|obs| <= pct (pct as decimal, e.g., 0.1 for 10%)."""
    x, y = align_xy(observed, predicted)
    if x.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(x), eps)
    return float(np.mean(np.abs(y - x) / denom <= pct))


# Metric registry (selective compute)

MetricFn = Callable[[np.ndarray, np.ndarray], float]

METRICS: Dict[str, MetricFn] = {
    "r2": lambda x, y: r2(x, y),
    "rmse": lambda x, y: rmse(x, y),
    "mse": lambda x, y: mse(x, y),
    "mae": lambda x, y: mae(x, y),
    "bias": lambda x, y: bias(x, y),
    "mape_pct": lambda x, y: mape_pct(x, y),
    "smape_pct": lambda x, y: smape_pct(x, y),
    "pearson_r": lambda x, y: pearson_r(x, y),
    "spearman_r": lambda x, y: spearman_r(x, y),
    "ccc": lambda x, y: concordance_ccc(x, y),
    "slope": lambda x, y: regression_line(y, x)[1],  # slope of predicted vs observed
    "intercept": lambda x, y: regression_line(y, x)[
        0
    ],  # intercept of predicted vs observed
}


def list_metrics() -> List[str]:
    return sorted(METRICS.keys())


def compute_metrics(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    metrics: Optional[Sequence[str]] = None,
    dropna: bool = True,
    extra: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
) -> Dict[str, float]:
    """
    Compute selected metrics (or all if metrics=None).

    - Returns a dict with 'n' always included.
    - extra allows adding custom metric functions at call time.
    """
    x, y = align_xy(observed, predicted, dropna=dropna)
    out: Dict[str, float] = {"n": float(x.size)}

    registry: Dict[str, MetricFn] = dict(METRICS)
    if extra:
        registry.update(extra)

    names = list(registry.keys()) if metrics is None else list(metrics)
    unknown = [m for m in names if m not in registry]
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}. Available: {sorted(registry.keys())}"
        )

    for m in names:
        out[m] = float(registry[m](x, y))
    return out


def metrics_table(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    metrics: Optional[Sequence[str]] = None,
    dropna: bool = True,
    extra: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
) -> pd.DataFrame:
    """1-row DataFrame for compute_metrics()."""
    return pd.DataFrame(
        [
            compute_metrics(
                observed, predicted, metrics=metrics, dropna=dropna, extra=extra
            )
        ]
    )


@dataclass(frozen=True)
class RegressionStats:
    n: int
    r2: float
    rmse: float
    mae: float
    mse: float
    mape_pct: float
    smape_pct: float
    bias: float
    pearson_r: float
    spearman_r: float
    ccc: float
    slope: float
    intercept: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n": float(self.n),
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mse": self.mse,
            "mape_pct": self.mape_pct,
            "smape_pct": self.smape_pct,
            "bias": self.bias,
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
            "ccc": self.ccc,
            "slope": self.slope,
            "intercept": self.intercept,
        }


def compute_regression_stats(
    observed: ArrayLike, predicted: ArrayLike, *, dropna: bool = True
) -> RegressionStats:
    """
    Compute the standard "all metrics" bundle.
    If you want a subset, use compute_metrics(..., metrics=[...]).
    """
    x, y = align_xy(observed, predicted, dropna=dropna)
    intercept, slope = regression_line(y, x)
    return RegressionStats(
        n=int(x.size),
        r2=r2(x, y),
        rmse=rmse(x, y),
        mae=mae(x, y),
        mse=mse(x, y),
        mape_pct=mape_pct(x, y),
        smape_pct=smape_pct(x, y),
        bias=bias(x, y),
        pearson_r=pearson_r(x, y),
        spearman_r=spearman_r(x, y),
        ccc=concordance_ccc(x, y),
        slope=slope,
        intercept=intercept,
    )


def summarize_samples(
    data,
    *,
    var_names=None,
    ddof: int = 1,
    cv_eps: float = 1e-12,
    quantiles=(0.25, 0.75),
    dropna: bool = True,
    sort_by: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Compute summary statistics for one or more variables.

    data: DataFrame, Series, dict[str, Series], or sequence of Series/arrays
    var_names: optional subset/order of variables
    Returns columns: n, mean, sd, cv, median, q25, q75, iqr, min, max
    """
    # ---- normalize input to DataFrame ----
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if var_names is not None:
            missing = [c for c in var_names if c not in df.columns]
            if missing:
                raise ValueError(f"var_names not found in DataFrame columns: {missing}")
            df = df[list(var_names)]

    elif isinstance(data, pd.Series):
        name = data.name or "value"
        df = data.to_frame(name=name)
        if var_names is not None and len(var_names) == 1:
            df = df.rename(columns={name: var_names[0]})

    elif isinstance(data, dict):
        keys = list(data.keys()) if var_names is None else list(var_names)
        missing = [k for k in keys if k not in data]
        if missing:
            raise ValueError(f"var_names not found in dict keys: {missing}")
        df = pd.DataFrame(
            {
                k: (data[k] if isinstance(data[k], pd.Series) else pd.Series(data[k]))
                for k in keys
            }
        )

    else:
        seq = list(data)
        names = (
            [getattr(s, "name", None) for s in seq]
            if var_names is None
            else list(var_names)
        )
        if any(n is None for n in names):
            raise ValueError("Series in sequence must have .name or provide var_names.")
        if len(names) != len(seq):
            raise ValueError("var_names length must match number of variables.")
        df = pd.DataFrame(
            {
                n: (s if isinstance(s, pd.Series) else pd.Series(s))
                for n, s in zip(names, seq)
            }
        )

    # ---- numeric coercion & cleaning ----
    df = df.apply(pd.to_numeric, errors="coerce")
    if dropna:
        df = df.replace([np.inf, -np.inf], np.nan)

    q_lo, q_hi = quantiles
    stats = {}

    # ---- compute statistics ----
    for col in df.columns:
        s = df[col].dropna() if dropna else df[col]
        n = int(s.size)

        if n == 0:
            stats[col] = dict(
                n=0,
                mean=np.nan,
                sd=np.nan,
                cv=np.nan,
                median=np.nan,
                q25=np.nan,
                q75=np.nan,
                iqr=np.nan,
                min=np.nan,
                max=np.nan,
            )
            continue

        mean_ = float(s.mean())
        sd_ = float(s.std(ddof=ddof)) if n > 1 else 0.0
        cv_ = float(sd_ / max(abs(mean_), cv_eps)) if np.isfinite(mean_) else np.nan
        q1 = float(s.quantile(q_lo))
        q3 = float(s.quantile(q_hi))

        stats[col] = dict(
            n=n,
            mean=mean_,
            sd=sd_,
            cv=cv_,
            median=float(s.median()),
            q25=q1,
            q75=q3,
            iqr=float(q3 - q1),
            min=float(s.min()),
            max=float(s.max()),
        )

    out = pd.DataFrame.from_dict(stats, orient="index")

    # ---- optional sorting ----
    if sort_by is not None:
        if sort_by not in out.columns:
            raise ValueError(f"sort_by must be one of {list(out.columns)}")
        out = out.sort_values(by=sort_by, ascending=ascending)

    return out


# Example usage

#     obs = pd.Series([15, 20, 25, 30], index=[0, 1, 2, 3], name="Observed")
#     pred = pd.Series([15.2, 19.4, 25.5, 29.8], index=[0, 1, 2, 3], name="Predicted")
#
#     print("Available metrics:", list_metrics())
#     print(compute_metrics(obs, pred, metrics=["r2", "rmse", "ccc"]))
#     print(metrics_table(obs, pred, metrics=["r2", "rmse"]))
#     print(compute_regression_stats(obs, pred).to_dict())
