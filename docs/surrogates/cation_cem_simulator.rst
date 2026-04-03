``cation_cem_simulator.py``
===========================

Location
--------

``src/electrodialysis_experiment/surrogates/transport_number_membrane/cation_cem_simulator.py``

Purpose
-------

Defines the process block wrapper that attaches transport-number surrogate
equations to each sample block in the experiment model.

Core components
---------------

``SurrogateType``
  Enum selecting surrogate form:

- ``LOG_LINEAR_POLYNOMIAL``
- ``LOG_LINEAR_LOG``
- ``SOFTMAX_COVARIATES``

``CationCemTransportNumberSimulator``
  Indexed ``ProcessBlock`` expected to be keyed by experiment ``sample_set``.

Build flow
----------

On build, each simulator block:

1. validates context (sample blocks exist and contain ED flowsheets),
2. exposes common ``cation_set`` reference,
3. dispatches surrogate equation construction through registry lookup.

Initialization entry point
--------------------------

``initiate_surrogate(...)`` performs offline coefficient fitting using the
registered initialization routine for the selected surrogate type, then writes
fitted coefficient values into block variables.

For surrogates that require additional covariates beyond concentration or
transport-number targets, such as ``SOFTMAX_COVARIATES``, the initializer also
passes surrogate-specific keyword data through to the registered fitting
routine. In the current softmax implementation, this includes precomputed
``feature_data`` built from local concentration ratios and current density.

Role in training
----------------

This block is the coupling interface between:

- measured concentration/transport-number data (for initial coefficient fit),
- process-state concentrations in each sample block, and
- trainable surrogate coefficients used in the global NLP.

It allows surrogate equations to be embedded directly in Pyomo while still
bootstrapping coefficients from data-driven regression.
