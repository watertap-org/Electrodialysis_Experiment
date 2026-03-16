``one_stage_single_pass.py``
============================

Location
--------

``src/electrodialysis_experiment/processes/one_stage_single_pass.py``

Purpose
-------

``one_stage_single_pass.py`` defines the process wrapper for a single-stage,
single-pass electrodialysis simulation. It packages one electrodialysis stack
into a complete flowsheet with boundary-condition blocks, pumps, product and
disposal outlets, process-level expressions, and convenience methods for
initialization and reporting.

The central class is ``OneStageSinglePass`` (implemented as
``OneStageSinglePassData`` through ``declare_process_block_class``).

How the flowsheet is assembled
------------------------------

The wrapper builds a complete flowsheet around one ``ED_base`` unit:

- ``Feed``:
  inlet boundary condition.
- ``Separator``:
  splits the incoming stream into the diluate-side and concentrate-side inlets.
- Two ``Pump`` blocks:
  drive the split streams into the stack inlets.
- ``ED_base``:
  the actual electrodialysis stack model.
- ``Product`` and ``Product``:
  collect the diluate-side product and the concentrate-side disposal stream.

The arcs are then expanded with ``TransformationFactory("network.expand_arcs")``
to create the connected steady-state process model.

Configuration source
--------------------

The wrapper is built from a validated Pydantic config object,
``OneStageSinglePassConfig``. The usual entry point is
``OneStageSinglePass.from_yaml(...)``, which reads a YAML file, validates it,
and returns a ready-to-build process block.

YAML files for this module
--------------------------

To author or modify YAML inputs, use:

- :doc:`../utils/yaml_configuration_guide` for practical templates and examples.
- :doc:`../schema/process_config_schema` for exact schema fields and defaults.

In practice, one-stage runs commonly use:

- process config: ``src/electrodialysis_experiment/configs/one_stage_single_pass.yml``
- initialization values: ``src/electrodialysis_experiment/configs/ossp_init_config.yml``
- scaling factors: ``src/electrodialysis_experiment/configs/scaling.yml``

Core responsibilities
---------------------

This module handles the process-layer concerns that sit above the raw stack:

- constructing the shared property package (``MCASParameterBlock``),
- instantiating and connecting the process units,
- adding process-level expressions such as water recovery, stack-average current
  density, voltage, salinity, and membrane area,
- optional costing integration through ``WaterTAPCosting``, and
- exposing convenience methods for initialization, solving, parameter fixing,
  and reporting.

Important public methods
------------------------

``from_yaml(path)``
  Build the process wrapper from a YAML file.

``import_scaling_config(path)``
  Apply user-defined scaling factors from YAML.

``import_init_value_config(path)``
  Apply initial values and fixed values from YAML.

``initialize_process(...)``
  Load the feed state, initialize units in sequence, propagate states through
  arcs, and optionally solve the full flowsheet.

``solve(model, solver=None, tee=True)``
  Solve the model using the supplied solver or default IPOPT.

``update_cation_cem_transport_number(...)``
  Fix cation transport-number values in the cation-exchange membrane.

``update_var_values(...)``
  Fix process variables from a dictionary or Pydantic model of updates.

``display_selected_model_metrics(...)``
  Print selected process-level performance values for inspection.

``plot_lengthwise_profile(...)``
  Generate plots of spatial profiles along the electrodialysis stack.

Initialization pattern
----------------------

The wrapper follows a staged initialization strategy:

1. Apply the selected feed state to the feed block.
2. Initialize the feed.
3. Propagate the feed state to the separator.
4. Initialize the separator.
5. Initialize the pumps.
6. Propagate states to the ED stack.
7. Initialize the stack.
8. Propagate outlet states to the product and disposal blocks.
9. Initialize the terminal blocks.
10. Solve the full connected flowsheet.

This makes the file the main process-level bridge between experimental input
data and a fully solved one-stage simulation.

When to use this module
-----------------------

Use ``OneStageSinglePass`` when:

- one electrodialysis stack is enough to represent the process,
- you want a compact process model for testing or calibration, or
- you want the clearest baseline before moving to a multi-stage model.

When not to use this module
---------------------------

If each stage requires its own operating conditions, transport-number estimates,
or stack configuration, the single-stage wrapper becomes too restrictive. In
that case, ``k_stage_single_pass.py`` is the more appropriate process layer.
