``schema`` package
==================

Location
--------

``src/electrodialysis_experiment/schema/``

Purpose
-------

The ``schema`` package is the typed interface layer between raw input files and
the process model code. It does three jobs:

- validates process-configuration YAML (stack options, ion data, solver options),
- validates value/scaling update YAML payloads, and
- converts tabular experiment data into structured objects used by simulation
  scripts.

Package structure
-----------------

- ``schema/config/process_config_schema.py``:
  Pydantic models for full process YAMLs consumed by
  ``OneStageSinglePass.from_yaml(...)`` and ``KStageSinglePass.from_yaml(...)``.
- ``schema/config/comp_value_schema.py``:
  Pydantic models for variable value assignment YAML consumed by
  ``apply_value_updates_from_yaml``.
- ``schema/config/scaling_schema.py``:
  Pydantic models for user scaling YAML consumed by
  ``apply_scaling_from_yaml``.
- ``schema/experiment/data.py``:
  Pydantic records and data-preparation helpers that map experimental tables
  into model-ready payloads.

``__init__.py`` files
---------------------

The package ``__init__.py`` files are currently minimal/empty namespace files.
Import concrete models/functions from submodules directly, for example:

.. code-block:: python

   from electrodialysis_experiment.schema.config.process_config_schema import (
       KStageSinglePassConfig,
   )
   from electrodialysis_experiment.schema.experiment.data import (
       FluidCondition,
       prepare_upd_param_dt_tssp_cv,
   )

Typical usage flow
------------------

1. Read a process YAML into a schema model (directly or via ``from_yaml`` in a
   process wrapper).
2. Read optional init-value/scaling YAMLs into schema payloads through utility
   functions.
3. Convert experiment data (parquet/csv) into typed update payloads using
   ``schema.experiment.data`` helpers.
4. Apply those payloads to process models.
