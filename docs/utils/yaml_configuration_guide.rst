YAML Configuration Guide
========================

This project relies on YAML files for four distinct configuration tasks:

1. process model construction (process config schema),
2. variable initialization/fixing (``value_setting``),
3. scaling factors (``user_scaling``), and
4. standalone solver option tuning (``solver_configuring``).

Use this page as the canonical authoring reference.

Where Each YAML Is Used
-----------------------

- Process config YAML:
  consumed by ``OneStageSinglePass.from_yaml(...)`` and
  ``KStageSinglePass.from_yaml(...)``.
- Initial value YAML:
  consumed by ``import_init_value_config(...)`` via
  ``utils.value_setting.apply_value_updates_from_yaml``.
- Scaling YAML:
  consumed by ``import_scaling_config(...)`` via
  ``utils.user_scaling.apply_scaling_from_yaml``.
- Solver option YAML:
  consumed by ``utils.solver_configuring.config_ipopt_solver(...)``.

Process Config YAML (One-Stage)
-------------------------------

Minimal shape:

.. code-block:: yaml

   ed_stack:
     operation_mode: Constant_Current
     finite_elements: 100
     has_pressure_change: true
     pressure_drop_method: experimental
     has_nonohmic_potential_membrane: true
     has_Nernst_diffusion_layer: true
     limiting_current_density_method: InitialValue
     limiting_current_density_data: 500

   ion:
     solute_list: ["Na_+", "Ca_2+", "Mg_2+", "Cl_-"]
     mw_data:
       H2O: 0.018
       Na_+: 0.023
       Ca_2+: 0.040078
       Mg_2+: 0.024305
       Cl_-: 0.0355
     charge:
       Na_+: 1
       Ca_2+: 2
       Mg_2+: 2
       Cl_-: -1

   solution:
     electrical_mobility_calculation: EinsteinRelation
     equivalent_conductivity_calculation: Onsager_Falkenhagen

   process:
     build_costing: false

   ipopt:
     solver_name: ipopt-watertap
     tol: 1e-8
     max_iter: 3000
     linear_solver: ma27
     bound_push: 1e-5
     mu_strategy: monotone
     nlp_scaling_method: user-scaling

Notes:

- ``solver_name`` selects the solver used by process initialization/solve paths.
- In current schema, IPOPT option fields are optional; omitted fields do not
  override solver defaults.

Process Config YAML (K-Stage)
-----------------------------

For multi-stage models, define either a shared ``ed_stack`` plus stage overrides
in ``ed_stacks`` or provide stage blocks mapped into ``ed_stacks`` by the
builder.

Common pattern (as used by two-stage workflows):

.. code-block:: yaml

   ed_stack_1:
     operation_mode: Constant_Voltage
     finite_elements: 100

   ed_stack_2:
     operation_mode: Constant_Voltage
     finite_elements: 100

   ion:
     solute_list: ["Na_+", "Ca_2+", "Mg_2+", "Cl_-"]
     mw_data: {H2O: 0.018, Na_+: 0.023, Ca_2+: 0.040078, Mg_2+: 0.024305, Cl_-: 0.0355}
     charge: {Na_+: 1, Ca_2+: 2, Mg_2+: 2, Cl_-: -1}

   solution:
     electrical_mobility_calculation: EinsteinRelation
     equivalent_conductivity_calculation: Onsager_Falkenhagen

   process:
     build_costing: false

   ipopt:
     solver_name: ipopt-watertap

Initialization Value YAML (``value_setting``)
---------------------------------------------

Top-level shape:

.. code-block:: yaml

   InitialValue:
     variables:
       - scalarVar: fs.feed.properties[0].pressure
         value: 101325
         mode: fix

       - scalarVar: fs.ocv
         value: 4.0
         mode: set
         lb: 0
         ub: 10

       - indexedVar: fs.EDstack.membrane_thickness
         items:
           - index: ["cem"]
             value: 5.7e-4
           - index: ["aem"]
             value: 6.35e-4
         mode: fix

Supported fields:

- ``scalarVar`` or ``indexedVar`` (choose one per entry),
- ``value`` (scalar or broadcast value),
- ``items`` for explicit indexed assignments,
- ``mode`` in ``fix``, ``set``, ``unfix`` (default is ``fix``),
- optional ``lb`` and ``ub`` (applied for ``set`` mode).

Scaling YAML (``user_scaling``)
-------------------------------

Top-level shape:

.. code-block:: yaml

   scaling:
     properties:
       - var: flow_mol_phase_comp
         index: ["Liq", "Na_+"]
         factor: 2000
       - var: flow_vol_phase
         index: ["Liq"]
         factor: 50000

     components:
       - target: fs.EDstack.cell_width
         factor: 5
       - target: fs.pump0.control_volume.work
         factor: 10

     constraints:
       - target: fs.eq_electrodialysis_equal_flow
         factor: 10

Rules:

- ``properties`` entries map to
  ``m.fs.properties.set_default_scaling(var, factor, index=...)``.
- ``components`` and ``constraints`` paths must be resolvable on the built model.

Solver Option YAML (``solver_configuring``)
-------------------------------------------

Top-level shape:

.. code-block:: yaml

   solver:
     ipopt:
       tol: 1e-8
       max_iter: 1000
       linear_solver: ma27
       bound_push: 1e-5
       mu_strategy: monotone
       nlp_scaling_method: user-scaling

Notes:

- ``config_ipopt_solver(...)`` applies only options under ``solver.ipopt``.
- Solver selection itself should be handled by process config ``ipopt.solver_name``
  or by explicitly requesting a solver in code.

Practical Authoring Checklist
-----------------------------

- Keep path strings exact (for example ``fs.EDstack[1].unit.cell_width``).
- Validate index tokens and ion names for consistency with the built model.
- Start from existing files in ``src/electrodialysis_experiment/configs/`` and
  modify incrementally.
