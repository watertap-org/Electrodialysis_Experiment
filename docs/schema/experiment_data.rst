``experiment/data.py``
======================

Location
--------

``src/electrodialysis_experiment/schema/experiment/data.py``

Purpose
-------

Provides typed experiment-data records and conversion utilities that transform
tabular datasets (typically parquet-derived) into payloads expected by process
wrappers.

This is the key bridge between measured campaign data and model inputs.

Core data models
----------------

``FluidCondition``
  Feed-state payload for ``initialize_process(...)``:

- ``flow_vol_phase`` (for example ``{"Liq": 2.5e-5}``)
- ``conc_mol_phase_comp`` keyed by ``(phase, component)``
- ``get_state_dict()`` helper to produce ``calculate_state(...)``-compatible
  dict format.

``UpdateParam`` / ``UpdateParam_stg`` / ``UpdateParam_4s``
  Typed update payloads for one-stage, stage-indexed, and legacy four-stage
  update patterns.

``UpdateStgConstCurr``
  Stage-indexed constant-current payload.

``TargetVariable``
  Regression/estimation target wrapper with ``name``, ``value``, ``weight``.

Column mappings
---------------

``COLUMN_MAPPING`` centralizes expected data-column names (for example
``CfNa``, ``Curr``, ``Volt_I``), and ``TargetVariableMapping`` maps model
target paths to dataframe columns for supervised workflows.

Main preparation functions
--------------------------

Feed/state construction:

- ``prepare_fluid_cond_dt(df)``
- ``prepare_fluid_cond_dt_compatible_to_calculate_state(fluid_cond_dt)``

Stage/config update payloads:

- ``prepare_upd_param_dt_cv(df)``: constant-voltage single-stage payloads.
- ``prepare_upd_param_dt_cc(df)``: constant-current single-stage payloads.
- ``prepare_upd_param_dt_cc_sv(df)``: constant-current with measured voltage.
- ``prepare_upd_param_dt_tssp_cv(df, stage_list=(1, 2))``: two-stage payloads.
- ``prepare_upd_param_dt_tssp_i_ii(df, stage_1=1, stage_2=2)``:
  backward-compatible two-stage wrapper.
- ``prepare_upd_param_dt_fssp_cv_stg(df)`` and ``prepare_upd_param_dt_fssp_cv(df)``:
  four-stage update helpers.
- ``prepare_const_curr_4st(df)``: four-stage constant-current payloads.

Transport-number estimators:

- ``prepare_cation_cem_transport_number_estimate(df)``
- ``prepare_cation_cem_transport_number_estimate_vas(df, feed_name, product_name)``
- ``prepare_cation_cem_transport_number_estimate_tssp(df, ...)``:
  stage-wise two-stage transport-number estimates.

Target extraction and I/O helpers:

- ``prepare_target_variable_dt(df)``
- ``prepare_cation_product_conc(df)``
- ``csv_to_parquet(...)``

Typical workflows
-----------------

Single-stage training/simulation input:

.. code-block:: python

   import pandas as pd
   from electrodialysis_experiment.schema.experiment.data import (
       prepare_fluid_cond_dt,
       prepare_upd_param_dt_cc,
       prepare_cation_cem_transport_number_estimate,
   )

   df = pd.read_parquet("src/electrodialysis_experiment/data/raw/dt_SEDv4_021125.parquet")
   fluid_conditions = prepare_fluid_cond_dt(df)
   update_params = prepare_upd_param_dt_cc(df)
   trans_numbers = prepare_cation_cem_transport_number_estimate(df)

Two-stage input:

.. code-block:: python

   from electrodialysis_experiment.schema.experiment.data import (
       prepare_upd_param_dt_tssp_cv,
       prepare_cation_cem_transport_number_estimate_tssp,
   )

   upd = prepare_upd_param_dt_tssp_cv(df, stage_list=[1, 2])
   t_est = prepare_cation_cem_transport_number_estimate_tssp(df, stage_1=1, stage_2=2)

``csv_to_parquet`` utility
--------------------------

Use ``csv_to_parquet(...)`` to standardize raw CSV input before running
preparation utilities:

- optional column subset/ordering,
- optional numeric coercion,
- optional per-column multipliers,
- optional NaN filtering,
- writes parquet and optionally returns processed ``DataFrame``.

Operational note
----------------

``load_parquet_and_prepare(path)`` currently calls ``prepare_upd_param_dt(df)``,
which is not defined in this module. Prefer the explicit preparation functions
above (``prepare_upd_param_dt_cc``, ``prepare_upd_param_dt_cv``, etc.) for
reliable workflows.
