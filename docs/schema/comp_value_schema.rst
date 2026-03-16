``comp_value_schema.py``
========================

Location
--------

``src/electrodialysis_experiment/schema/config/comp_value_schema.py``

Purpose
-------

Defines the schema for YAML payloads that set/fix/unfix model variable values.
These payloads are consumed by
``electrodialysis_experiment.utils.value_setting.apply_value_updates_from_yaml``.

Core models
-----------

``IndexedVarItem``
  One index/value pair for an indexed variable.

- ``index``: scalar or list (list is interpreted as a tuple index).
- ``value``: numeric value.

``VarAssignment``
  One assignment operation.

- scalar form:
  ``scalarVar`` + ``value``
- indexed form:
  ``indexedVar`` + ``items`` (or broadcast with ``value``)
- behavior control:
  ``mode`` in ``{"fix", "set", "unfix"}`` (default ``fix``)
- optional bounds:
  ``lb`` and ``ub`` (applied when mode is ``set``)

``VarValueConfig``
  Top-level list container for assignment entries.

YAML shape
----------

The utility expects a block containing ``variables`` entries. Example:

.. code-block:: yaml

   InitialValue:
     variables:
       - scalarVar: fs.feed.properties[0].pressure
         value: 101325
         mode: fix

       - indexedVar: fs.EDstack.membrane_thickness
         items:
           - index: [cem]
             value: 5.7e-4
           - index: [aem]
             value: 6.35e-4
         mode: fix

       - scalarVar: fs.ocv
         value: 4.0
         mode: set
         lb: 0
         ub: 10

How to choose scalar vs indexed form
------------------------------------

- Use ``scalarVar`` when path resolution ends at one scalar variable/data item.
- Use ``indexedVar`` when path resolution ends at an indexed component and you
  want explicit per-index assignments.
- Use ``indexedVar`` + top-level ``value`` (no ``items``) when you need to
  broadcast a single value to all indices.

Practical notes
---------------

- Path strings are resolved dynamically against the model object (for example,
  ``fs.EDstack[1].unit.cell_width``).
- Schema-level validation enforces field structure; runtime utility validation
  enforces that resolved objects are writable Pyomo variables.
- Field names are case-sensitive; use exactly ``scalarVar`` and ``indexedVar``.
