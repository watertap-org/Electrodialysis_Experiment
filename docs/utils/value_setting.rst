``value_setting.py``
====================

Location
--------

``src/electrodialysis_experiment/utils/value_setting.py``

Purpose
-------

Applies YAML-defined variable assignments (fix/set/unfix and optional bounds)
to Pyomo/IDAES models.

Schema dependency
-----------------

Assignment payloads are validated with:

- ``schema.config.comp_value_schema.VarValueConfig``,
- ``VarAssignment``,
- ``IndexedVarItem``.

Core API
--------

``apply_value_updates_from_yaml(m, yaml_path, key=\"InitialValue\")``
  Load, validate, and apply YAML update block.

``apply_value_updates(m, cfg)``
  Apply an already-validated ``VarValueConfig`` object.

``resolve_path(m, path)``
  Resolve dotted/indexed path strings to model components.

Assignment modes
----------------

- ``fix``: ``var.fix(value)``,
- ``set``: ``var.set_value(value)`` with optional ``lb``/``ub``,
- ``unfix``: ``var.unfix()``.

Supports both:

- scalar assignments (``scalarVar``), and
- indexed assignments (``indexedVar`` + ``items`` or broadcast ``value``).

How this fits the workflow
--------------------------

This module is the bridge between config files (for initialization/calibration
setups) and concrete model variable states. It enables reproducible model
conditioning without editing scripts for each parameter update.

YAML Authoring Guidance
-----------------------

Use :doc:`yaml_configuration_guide` as the main how-to page for writing config
files. For this utility specifically, the expected block is:

.. code-block:: yaml

   InitialValue:
     variables:
       - scalarVar: fs.feed.properties[0].pressure
         value: 101325
         mode: fix

       - indexedVar: fs.EDstack.membrane_thickness
         items:
           - index: ["cem"]
             value: 5.7e-4
           - index: ["aem"]
             value: 6.35e-4
         mode: fix

Full schema details are documented in
:doc:`../schema/comp_value_schema`.
