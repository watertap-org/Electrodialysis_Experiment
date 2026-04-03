``experiment.py``
=================

Location
--------

``src/electrodialysis_experiment/experiment.py``

Purpose
-------

Defines ``MasterExperimentBuilder``, the orchestration layer for multi-sample
calibration experiments built from repeated process blocks.

Core architecture
-----------------

``MasterExperimentBuilder`` creates:

- ``model.sample_set``:
  sample indices ``0..N-1``.
- ``model.sample_blk[i]``:
  one ``OneStageSinglePass`` process per sample.

This creates one large NLP with repeated physics blocks and shared calibration
degrees of freedom.

Key methods used in training
----------------------------

``proc_config_from_yaml(path, sample_size)``
  Validate process config (Pydantic) and construct the experiment model.

``initialize_individual_sample_blks(...)``
  For each sample: apply scaling/init config, apply per-sample updates,
  optionally set estimated transport numbers, initialize/solve process.

``add_cation_cem_transport_number_simulator(...)``
  Attach stage-wise surrogate blocks indexed by ``sample_set``.
  The selected ``SurrogateType`` may be any registered transport-number
  surrogate, including ``SOFTMAX_COVARIATES``.

``add_log_linear_surr_coef_constraint()``
  Constrain surrogate coefficients to be equal across all sample simulators.
  This creates a single global coefficient set shared by all samples.

``free_cation_transport_numbers_in_cem()``
  Unfix cation CEM transport-number variables so they are solved under surrogate
  constraints during calibration.

``add_sse_objective_of_selected_variables(variables_weights, data)``
  Build weighted SSE objective over active sample blocks and selected target
  variable paths.

``save_model_hdf(filename)`` / ``load_model_data(filename)``
  Serialize and restore variables/parameters/fixed-status/scaling values.

Theoretical role in calibration
-------------------------------

The experiment builder turns many steady-state process simulations into one
joint estimation problem:

- local sample-specific process states are resolved per sample block,
- shared surrogate parameters are estimated globally,
- all sample residuals contribute to one objective.

This is equivalent to pooled nonlinear regression with process constraints.

Surrogate interface note
------------------------

``add_cation_cem_transport_number_simulator(...)`` dispatches surrogate build
logic through the registered surrogate interface exposed by
``CationCemTransportNumberSimulator``. Surrogate-specific keyword arguments are
passed through unchanged. This matters for the softmax surrogate, whose
initializer requires ``feature_data`` in addition to measured transport-number
targets. As a result, the experiment builder can support both simpler
concentration-ratio surrogates and richer covariate-based surrogates without
changing the higher-level experiment workflow.

Snapshot mechanism
------------------

What is saved:

- variable values, fixed status, scaling,
- parameter values,
- constraint and expression scaling metadata.

Why it matters:

- expensive initialization can be reused,
- iterative tuning becomes much faster,
- calibration can resume from a numerically conditioned state.

Operational cautions
--------------------

- ``save_model_hdf`` writes in ``mode='w'`` and overwrites existing files.
- ``load_model_data`` expects component paths to match the current model
  structure; schema/config changes can invalidate old snapshots.
