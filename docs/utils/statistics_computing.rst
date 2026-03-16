``statistics_computing.py``
===========================

Location
--------

``src/electrodialysis_experiment/utils/statistics_computing.py``

Purpose
-------

Reusable statistical metrics and summary helpers for observed-vs-predicted
analysis.

Core capabilities
-----------------

Data alignment:

- ``align_xy(...)`` aligns paired data (including index-aware alignment for
  pandas Series) and optional finite-value filtering.

Point-estimate metrics:

- ``r2``, ``mse``, ``rmse``, ``mae``, ``bias``,
- ``mape_pct``, ``smape_pct``,
- ``pearson_r``, ``spearman_r``,
- ``concordance_ccc``,
- regression slope/intercept via ``regression_line``.

Coverage metrics:

- ``coverage_within_abs`` and ``coverage_within_pct``.

Registry-based computation:

- ``list_metrics()``,
- ``compute_metrics(...)`` for selected/all metrics,
- ``metrics_table(...)`` for one-row DataFrame output.

Bundled regression summary:

- ``RegressionStats`` dataclass,
- ``compute_regression_stats(...)``.

Variable-level descriptive statistics:

- ``summarize_samples(...)`` returns per-variable ``n``, mean, SD, CV,
  median/quantiles, IQR, min/max.

How this fits the repository
----------------------------

These functions support model evaluation workflows (including surrogate and
process calibration) while keeping metric definitions centralized and
consistent.
