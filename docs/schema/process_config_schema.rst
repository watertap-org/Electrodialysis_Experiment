``process_config_schema.py``
============================

Location
--------

``src/electrodialysis_experiment/schema/config/process_config_schema.py``

Purpose
-------

Defines the main Pydantic schema models for process-configuration YAML files.
These models are used to validate process settings before building:

- ``OneStageSinglePass``,
- ``FourStageSinglePass``, and
- ``KStageSinglePass``.

Validation behavior
-------------------

- Most models use ``extra="forbid"``. Unknown keys in YAML raise validation
  errors.
- Enum fields accept case-insensitive enum names via helper
  ``validate_enum(...)``.
- ``IonConfig`` accepts either tuple-keyed dicts or YAML-friendly nested dicts
  for transport properties and normalizes them internally.

Core models
-----------

``ProcessConfig``
  Process-level runtime options:
  ``build_costing``, ``tee``, ``output_dir``.

``IPOPTconfig``
  IPOPT defaults such as ``tol``, ``max_iter``, ``linear_solver``,
  ``mu_strategy``, and user-scaling mode.

``EDStackConfig``
  Electrodialysis unit options passed into ``ED_base``:
  operation mode, pressure-drop method, limiting-current method, discretization
  settings, and other stack toggles.

``IonConfig``
  Chemistry/transport data for ``MCASParameterBlock``:
  ``solute_list``, ``mw_data``, ``charge``, and optional diffusivity/mobility/
  transport-number maps.

``SolutionConfig``
  Property-calculation options for solution transport/conductivity.

``OneStageSinglePassConfig`` / ``FourStageSinglePassConfig`` /
``KStageSinglePassConfig``
  Top-level process schemas.

K-stage specifics
-----------------

``KStageSinglePassConfig`` supports two ways to define stage stacks:

- explicit ``ed_stacks: {1: ..., 2: ...}``, and
- legacy keys like ``ed_stack_1``, ``ed_stack_2``, etc.

The model validator:

- migrates legacy ``ed_stack_N`` keys into ``ed_stacks``,
- infers ``num_stages`` from the largest legacy stage index when not provided,
- normalizes stage keys to integers, and
- enforces stage-index bounds: ``1 <= stage <= num_stages``.

Minimal YAML example
--------------------

.. code-block:: yaml

   ion:
     solute_list: [Na_+, Ca_2+, Mg_2+, Cl_-]
     mw_data: {H2O: 0.01801528, Na_+: 0.02299, Ca_2+: 0.04008, Mg_2+: 0.024305, Cl_-: 0.03545}
     charge: {Na_+: 1, Ca_2+: 2, Mg_2+: 2, Cl_-: -1}
   solution:
     electrical_mobility_calculation: none
     equivalent_conductivity_calculation: ElectricalMobility
   process:
     build_costing: false
   ipopt:
     tol: 1e-8
   ed_stack:
     operation_mode: Constant_Current
     finite_elements: 10

Usage example
-------------

.. code-block:: python

   import yaml
   from electrodialysis_experiment.schema.config.process_config_schema import (
       KStageSinglePassConfig,
   )

   with open("src/electrodialysis_experiment/configs/two_stage_single_pass.yml") as f:
       raw = yaml.safe_load(f)
   cfg = KStageSinglePassConfig(**raw)

Common errors to watch
----------------------

- stage key typo (for example, ``"one"`` instead of ``1``),
- unknown YAML key under strict models,
- missing required ``ion`` fields, and
- stage indices outside ``num_stages`` in ``ed_stacks``.
