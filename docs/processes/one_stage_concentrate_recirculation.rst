``one_stage_concentrate_recirculation.py``
==========================================

Location
--------

``src/electrodialysis_experiment/processes/one_stage_concentrate_recirculation.py``

Purpose
-------

``one_stage_concentrate_recirculation.py`` defines a one-stage electrodialysis
process wrapper with a concentrate recycle loop. It extends the single-pass
wrapper into a feed-and-bleed configuration where part of the final
concentrate stream is recycled to the concentrate inlet of the stack.

The central class is ``OneStageConcentrateRecirculation`` (implemented as
``OneStageConcentrateRecirculationData`` through
``declare_process_block_class``).

How the flowsheet is assembled
------------------------------

The wrapper builds a complete flowsheet around one ``ED_base`` unit:

- ``Feed``:
  inlet boundary condition.
- ``Separator`` ``sepa0``:
  splits the feed into the diluate branch and the fresh concentrate branch.
- ``Mixer`` ``mix0``:
  combines fresh concentrate-side feed with recycled concentrate.
- Two ``Pump`` blocks:
  drive the diluate and concentrate branches into the stack.
- ``ED_base``:
  the electrodialysis stack.
- ``Separator`` ``sepa1``:
  splits final concentrate into a bleed stream and a recycle stream.
- ``Product`` and ``Product``:
  collect the diluate-side product and the concentrate-side bleed stream.

Compared with the single-pass module, the key structural change is the recycle
loop:

``EDstack.outlet_concentrate -> sepa1 -> mix0 -> pump0 -> EDstack.inlet_concentrate``

Configuration source
--------------------

The wrapper uses the same validated Pydantic config object family as the
single-pass version, with ``OneStageSinglePassConfig`` as the process config.
The usual entry point is ``OneStageConcentrateRecirculation.from_yaml(...)``.

YAML files for this module
--------------------------

To author or modify YAML inputs, use:

- :doc:`../utils/yaml_configuration_guide` for practical templates and examples.
- :doc:`../schema/process_config_schema` for exact schema fields and defaults.

In practice, one-stage concentrate-recycle runs commonly use:

- process config: ``src/electrodialysis_experiment/configs/one_stage_conc_recir.yml``
- initialization values: ``src/electrodialysis_experiment/configs/oscr_init_config.yml``
- scaling factors: ``src/electrodialysis_experiment/configs/scaling.yml``

Core responsibilities
---------------------

This module handles the process-layer concerns specific to one-stage
concentrate recycle operation:

- constructing the shared property package,
- instantiating and connecting recycle-specific process units,
- replacing the inherited recovery expression with a recoverable
  ``Var + Constraint`` form so water recovery can be fixed directly,
- adding recycle and bleed fraction expressions,
- supporting staged initialization through the recycle loop, and
- exposing solve, reporting, and configuration helper methods.

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
  the recycle loop, and optionally solve the full flowsheet.

``set_feed_split_to_diluate(split_fraction, fix=True)``
  Set or fix the first separator split between the diluate and fresh
  concentrate branches.

``set_concentrate_recycle_fraction(recycle_fraction, fix=True)``
  Set or fix the final concentrate recycle fraction.

``display_selected_model_metrics(...)``
  Print selected process-level performance values for inspection.

Recovery and recycle closure
----------------------------

Unlike ``one_stage_single_pass.py``, this module defines
``fs.recovery_vol_H2O`` as a ``Var`` with an explicit constraint:

.. code-block:: python

   recovery_vol_H2O * feed_flow - product_flow == 0

This is intentional. In a feed-and-bleed recirculation process, users may want
to fix either:

- water recovery, or
- recycle split fraction.

The class keeps that flexibility while still allowing the recycle fraction to
be temporarily fixed during separator initialization if the caller has not
already fixed it.

Initialization pattern
----------------------

The wrapper follows a recycle-aware staged initialization strategy:

1. Apply the selected feed state to the feed block.
2. Initialize the feed.
3. Propagate the feed state to ``sepa0``.
4. Initialize ``sepa0``.
5. Initialize the diluate-side pump.
6. Initialize the concentrate-side pump from a propagated inlet state.
7. Propagate states to the ED stack.
8. Initialize the stack.
9. Initialize the product block.
10. Initialize ``sepa1`` with a temporary recycle split if needed.
11. Initialize the disposal block.
12. Propagate recycle state back through the loop.
13. Initialize the recycle mixer.

This sequence mirrors the logic of the original standalone recirculation
flowsheet while packaging it into a reusable class-based process module.

When to use this module
-----------------------

Use ``OneStageConcentrateRecirculation`` when:

- one electrodialysis stack is enough to represent the treatment section,
- concentrate recycle is required to represent feed-and-bleed operation,
- water recovery must be controlled directly, or
- you want a one-stage recycle baseline before building a multi-stage recycle
  process.

When not to use this module
---------------------------

If the process requires several electrodialysis stages in sequence, this
one-stage wrapper becomes too restrictive. In that case,
``k_stage_concentrate_recirculation.py`` is the more appropriate process
layer.
