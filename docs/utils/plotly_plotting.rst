``plotly_plotting.py``
======================

Location
--------

``src/electrodialysis_experiment/utils/plotly_plotting.py``

Purpose
-------

Provides standardized Plotly figure builders for exploratory analysis and model
evaluation. The module emphasizes consistent visual style (white background,
boxed axes, controlled ticks) across plot types.

Main plotting utilities
-----------------------

``scatter_2d_series(x, y, ...)``
  2D scatter with optional ``y=x`` diagonal and nice tick-spacing helpers.

``scatter_3d_series(x, y, z, ...)``
  3D scatter with optional depth-based color encoding.

``violin_box_params(params, ...)``
  Parameter distribution plots for DataFrame/dict/series collections.

``zscore_heatmap(data, ...)``
  Column-wise z-score heatmap with clipping and display-label controls.

``mean_pm_nstd_plot(data, ...)``
  Mean with ``± n_std * SD`` error bars per variable.

Uncertainty visualization
-------------------------

``summarize_predictions(...)`` / ``_summarize_predictions(...)``
  Reduce draws to center + quantile band.

``plot_prediction_with_uncertainty_band(...)``
  Plot center line with uncertainty envelope and optional observations.

``observed_vs_pred_with_uncertainty(...)``
  Observed-vs-predicted parity view with per-point uncertainty bars.

Design intent
-------------

This module centralizes plotting logic so analysis scripts can reuse a common
visual language instead of reimplementing style and axis-formatting each time.
