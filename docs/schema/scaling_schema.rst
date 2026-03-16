``scaling_schema.py``
=====================

Location
--------

``src/electrodialysis_experiment/schema/config/scaling_schema.py``

Purpose
-------

Defines the schema for user scaling YAML consumed by
``electrodialysis_experiment.utils.user_scaling.apply_scaling_from_yaml``.

Core models
-----------

``SolutionDefaultScaling``
  Default scaling for property-package variables via
  ``set_default_scaling(var, factor, index=...)``.

- ``var``: property variable name (for example ``flow_mol_phase_comp``).
- ``factor``: scaling factor.
- ``index``: optional index list (for example ``["Liq", "Na_+"]``).

``ComponentScaling``
  Targeted component scaling via
  ``iscale.set_scaling_factor(resolve_path(target), factor)``.

- ``target``: dotted model path string.
- ``factor``: scaling factor.

``ConstraintScaling``
  Constraint scaling transform via
  ``iscale.constraint_scaling_transform(resolve_path(target), factor)``.

- ``target``: dotted model constraint path.
- ``factor``: scaling multiplier.

``ScalingConfig``
  Top-level object with three lists:
  ``properties``, ``components``, ``constraints``.

YAML shape
----------

.. code-block:: yaml

   scaling:
     properties:
       - var: flow_mol_phase_comp
         index: [Liq, Na_+]
         factor: 2000
       - var: flow_vol_phase
         index: [Liq]
         factor: 50000

     components:
       - target: fs.EDstack.cell_width
         factor: 5
       - target: fs.pump0.control_volume.work
         factor: 10

     constraints:
       - target: fs.eq_electrodialysis_equal_flow
         factor: 10

Usage example
-------------

.. code-block:: python

   from electrodialysis_experiment.utils.user_scaling import apply_scaling_from_yaml

   apply_scaling_from_yaml(model.proc, "src/electrodialysis_experiment/configs/scaling.yml")

Practical notes
---------------

- Paths are resolved at runtime; invalid ``target`` strings fail during apply.
- Keep this schema focused on scaling-only changes. Variable fixing belongs in
  ``comp_value_schema.py`` payloads.
