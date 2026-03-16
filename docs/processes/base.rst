``base.py``
===========

Location
--------

``src/electrodialysis_experiment/processes/base.py``

Purpose
-------

``base.py`` defines the low-level one-dimensional electrodialysis unit model
used by the higher-level process wrappers. In this repository, it is the core
stack model from which the single-stage and multi-stage process flowsheets are
assembled.

The central class is ``ED_base`` (implemented as ``ED1DData`` through
``declare_process_block_class``). It represents one electrodialysis stack with
distributed behavior along the flow direction.

What this module contains
-------------------------

- Configuration enums for important modeling choices:
  ``LimitingCurrentDensityMethod``, ``ElectricalOperationMode``,
  ``PressureDropMethod``, ``FrictionFactorMethod``, and
  ``HydraulicDiameterMethod``.
- The base electrodialysis unit model class ``ED_base`` / ``ED1DData``.
- The distributed transport, electrical, and optional pressure-drop logic used
  inside each stack.

Role in the codebase
--------------------

This module is not usually the direct script entry point. Instead:

- ``one_stage_single_pass.py`` creates one ``ED_base`` instance.
- ``k_stage_single_pass.py`` creates one ``ED_base`` instance per stage.

Because of that, changes in ``base.py`` propagate upward into nearly every
process simulation workflow.

Key configuration concepts
--------------------------

The class exposes many model-form choices through IDAES/Pyomo configuration.
The most important ones are:

- ``operation_mode``:
  chooses constant-current versus constant-voltage operation.
- ``limiting_current_density_method``:
  selects how limiting current is estimated.
- ``has_pressure_change`` and the pressure-drop-related methods:
  turn hydraulic pressure loss on or off and select the correlation used.
- ``has_nonohmic_potential_membrane``:
  enables additional membrane-potential contributions beyond the basic ohmic
  description.
- ``has_Nernst_diffusion_layer``:
  enables concentration-polarization diffusion layer modeling.

These options determine both physics included in the unit model and numerical
difficulty during initialization and solve.

Main implementation blocks
--------------------------

``build()``
  Validates the configuration and constructs the electrodialysis unit structure.

``_make_performance()``
  Builds the main transport and electrical-performance equations for the base
  stack model.

``_make_performance_nonohm_mem()``
  Adds non-ohmic membrane-potential contributions when enabled.

``_make_performance_dl_polarization()``
  Adds diffusion-layer concentration-polarization effects when enabled.

``_pressure_drop_calculation()``
  Builds the hydraulic pressure-loss calculations according to the selected
  pressure-drop method.

``initialize_build()``
  Executes initialization of the stack model.

``calculate_scaling_factors()``
  Applies model-specific scaling logic for improved nonlinear solver behavior.

Operational outputs
-------------------

The base unit provides the quantities that the process wrappers use most often:

- diluate and concentrate outlet states,
- applied current or applied voltage behavior,
- current-density and power relationships,
- membrane transport variables, including ion transport numbers, and
- performance and stream-table summaries.

When to edit this file
----------------------

Edit ``base.py`` when the change is fundamentally about stack physics or stack
numerics, for example:

- changing electrodialysis transport equations,
- adding a new pressure-drop correlation,
- introducing a new electrical operation feature, or
- modifying stack-level initialization behavior.

Do not start here if you only need to change how complete process stages are
connected together. Those changes usually belong in the process-wrapper modules.

Practical caution
-----------------

This module sits at the highest-risk layer for regressions. Even a small change
can affect all flowsheets that depend on ``ED_base``. It is best treated as the
unit-model layer, not as a script-level customization point.
