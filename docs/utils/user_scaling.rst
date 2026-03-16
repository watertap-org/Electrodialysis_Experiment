``user_scaling.py``
===================

Location
--------

``src/electrodialysis_experiment/utils/user_scaling.py``

Purpose
-------

Applies model scaling from YAML (validated via
``schema.config.scaling_schema.ScalingConfig``) to IDAES/Pyomo models.

Core API
--------

``apply_scaling_from_yaml(m, yaml_path)``
  Load, validate, and apply scaling config from YAML.

``apply_scaling(m, cfg)``
  Apply scaling payload directly from a ``ScalingConfig`` object.

``resolve_path(m, path)``
  Resolve dotted paths with optional bracket indices to concrete model
  components.

``check_badly_scaled_vars(model, ...)``
  Diagnostic printout for badly scaled variables.

Scaling actions supported
-------------------------

- default property scaling via ``set_default_scaling``,
- component scaling via ``set_scaling_factor``,
- constraint scaling via ``constraint_scaling_transform``,
- global scaling-factor calculation after updates.

Why this matters
----------------

Nonlinear ED process solves are sensitive to variable magnitude. This module
provides a reproducible, externalized way to improve conditioning and solver
performance without hardcoding scaling logic inside process modules.

YAML Authoring Guidance
-----------------------

Use :doc:`yaml_configuration_guide` as the canonical reference for YAML
structure. The scaling block consumed by this module is:

.. code-block:: yaml

   scaling:
     properties:
       - var: flow_mol_phase_comp
         index: ["Liq", "Na_+"]
         factor: 2000
     components:
       - target: fs.EDstack.cell_width
         factor: 5
     constraints:
       - target: fs.eq_electrodialysis_equal_flow
         factor: 10

Full field definitions are in :doc:`../schema/scaling_schema`.
