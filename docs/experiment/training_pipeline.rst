``scripts/model_training.py``
=============================

Location
--------

``scripts/model_training.py``

Purpose
-------

Calibrate a one-stage ED process model over multiple experimental samples,
using a trainable surrogate for cation transport numbers in CEM and a weighted
SSE objective on selected product concentrations.

Module map used by this script
------------------------------

- ``schema.experiment.data`` (documented in ``docs/schema/experiment_data.rst``):
  converts parquet rows into typed model inputs/targets.
- ``experiment.MasterExperimentBuilder``:
  creates and manages the multi-sample optimization model.
- ``surrogates.transport_number_membrane.*``:
  adds transport-number surrogate variables/constraints and initialization.
- ``utils.solver_configuring.config_ipopt_solver`` (documented in
  ``docs/utils/solver_configuring.rst``):
  loads solver options from YAML into an IPOPT solver instance.
- ``processes.one_stage_single_pass`` (documented in ``docs/processes``):
  physical process model instantiated per sample.

Training formulation (conceptual)
---------------------------------

For each sample ``i``:

1. Build one process block ``sample_blk[i].proc``.
2. Set feed and process settings from measured data.
3. Solve/condition once (or restore from snapshot).
4. Add surrogate constraints linking membrane transport numbers to local
   concentration ratios.

Global coupling across all samples:

- surrogate coefficient equality constraints force one shared coefficient set;
- weighted SSE objective aggregates prediction error across all active samples.

Objective shape
---------------

The script builds:

.. math::

   \min \sum_{i \in \text{samples}} \sum_{v \in \text{targets}}
   w_v \left(\hat{y}_{i,v} - y_{i,v}^{\text{exp}}\right)^2

where ``w_v`` are user-specified variable weights.

Runtime strategy
----------------

The script supports two initialization paths:

- full initialization path:
  ``initialize_individual_sample_blks(...)`` (slow but complete),
- fast-restart path:
  ``load_model_data(...)`` from an HDF snapshot (fast for iterative training).

Use the full path when model structure/config changes; use snapshot load for
repeated surrogate/objective tuning runs.

Important practical notes
-------------------------

- Snapshot save uses HDF store write mode; target files are overwritten.
- Relative paths depend on runtime current working directory; prefer repo-root
  absolute paths in notebooks.
- Transport-number variables are unfixed before training and controlled by
  surrogate constraints and bounds.
