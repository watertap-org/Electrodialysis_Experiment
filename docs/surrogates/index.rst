Transport-Number Surrogates
===========================

Location
--------

``src/electrodialysis_experiment/surrogates/transport_number_membrane/``

Purpose
-------

These modules define the surrogate infrastructure used in
``scripts/model_training.py`` to relate cation transport numbers in CEM to
local concentration ratios.

Contents
--------

.. toctree::
   :maxdepth: 1

   cation_cem_simulator
   log_linear_conc_ratio
   softmax_covariates
   registry
