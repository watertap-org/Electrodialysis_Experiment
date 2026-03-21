``base_dynamic.py``
===================

Location
--------

``src/electrodialysis_experiment/processes/base_dynamic.py``

Purpose
-------

``base_dynamic.py`` adds a dynamic electrodialysis stack model that extends the
steady ``ED_base`` concept to include time accumulation of species in both
channels while preserving the 1D axial transport formulation.

This model is intended for transient studies where current density, flux, and
channel concentrations evolve over time.

Relationship to the WaterTAP ED1D theory page
----------------------------------------------

The governing transport/electrical terms are aligned with the same ED1D
framework documented by WaterTAP:

- https://watertap.readthedocs.io/en/stable/technical_reference/unit_models/electrodialysis_1D.html

The key difference is that this module explicitly enables dynamic control
volumes (time accumulation terms), while the current steady ``base.py`` is
configured for ``dynamic=False``.

Model scope and assumptions
---------------------------

Current implementation scope:

- Dynamic, 1D control volumes in both diluate and concentrate channels.
- Constant-current and constant-voltage operating modes.
- Core ion/water transport terms from the steady model:
  migration, membrane diffusion, and electro-osmotic/osmotic water transfer.
- Spatial discretization by Pyomo DAE finite difference/collocation.
- Time discretization by Pyomo DAE finite difference through a helper method.

Simplifications in this first dynamic version:

- Non-ohmic membrane potential terms are not included.
- Nernst diffusion-layer polarization terms are not included.
- Pressure-drop submodels are not included.

These simplifications are intentional to provide a robust transient baseline
that is consistent with core ED physics and easier to initialize/solve.

Governing equations (high-level)
--------------------------------

1) Dynamic species balance in each channel (after spatial discretization):

.. math::

   \frac{\partial n_{j}}{\partial t} + \frac{\partial F_{j}}{\partial x} = \Gamma_{j}

where :math:`\Gamma_j` is the membrane mass-transfer source term.

2) Diluate source terms use the same structure as the steady model (simplified
without diffusion-layer/non-ohmic terms):

- Ions: migration + diffusion
- Water: electro-osmosis + osmotic permeation
- Neutral solutes: diffusion only

3) Concentrate source terms enforce conservation across membranes:

.. math::

   \Gamma_{j}^{conc} = -\Gamma_{j}^{dil}

4) Electrical closure:

.. math::

   i(x,t) R_{area}(x,t) = V_{applied}(t) \quad \text{(constant-voltage mode)}

or

.. math::

   i(x,t) A = I_{applied}(t) \quad \text{(constant-current mode)}

5) Electrical power profile along stack length:

.. math::

   \frac{dP}{dx} = V(x,t)\, i(x,t)\, W\,L

where :math:`W` is stack width and :math:`L` is stack length scaling used by
the control-volume geometry convention.

Main class
----------

- ``ED_base_dynamic`` (implemented as ``EDDynamic1DData`` via
  ``declare_process_block_class``)

Key methods:

- ``build()``:
  constructs dynamic 1D control volumes and ED transport/electrical equations.
- ``apply_time_discretization(...)``:
  applies finite-difference discretization to the model time domain
  (on the parent flowsheet block).
- ``apply_spatial_discretization()``:
  applies the configured length-domain discretization on both channels.
- ``discretize_dynamic_model(...)``:
  convenience method that enforces the safe order: time then space.
- ``initialize_build(...)``:
  initializes both channels and performs an initialization solve.
- ``calculate_scaling_factors()``:
  applies baseline scaling to key variables.

Expected usage workflow
-----------------------

1. Build a dynamic flowsheet (time domain present).
2. Construct properties.
3. Add ``ED_base_dynamic`` unit.
4. Fix design/operating variables and boundary conditions.
5. Discretize dynamics with ``discretize_dynamic_model`` (time then space).
6. Initialize and solve.

Minimal code skeleton
---------------------

.. code-block:: python

   from pyomo.environ import ConcreteModel, units as pyunits
   from idaes.core import FlowsheetBlock
   from idaes.core.solvers import get_solver

   from electrodialysis_experiment.processes.solution_dynamic import MCASParameterBlock
   from electrodialysis_experiment.processes.base import ElectricalOperationMode
   from electrodialysis_experiment.processes.base_dynamic import ED_base_dynamic

   m = ConcreteModel()
   m.fs = FlowsheetBlock(dynamic=True, time_units=pyunits.s, time_set=[0, 3600])

   m.fs.properties = MCASParameterBlock(
       solute_list=["Na_+", "Ca_2+", "Mg_2+", "Cl_-"],
       # ... other required property arguments ...
   )

   m.fs.ed = ED_base_dynamic(
       property_package=m.fs.properties,
       operation_mode=ElectricalOperationMode.Constant_Voltage,
       finite_elements=10,
       collocation_points=2,
   )

   # Fix design and boundary conditions (examples)
   m.fs.ed.cell_length.fix(0.5)
   m.fs.ed.cell_width.fix(0.1)
   m.fs.ed.cell_pair_num.fix(50)
   m.fs.ed.voltage_applied[:].fix(20)

   # Discretize in robust order (time first, then space)
   m.fs.ed.discretize_dynamic_model(time_nfe=40, time_scheme="BACKWARD")

   # Optional unit initialization, then solve
   m.fs.ed.initialize_build()
   solver = get_solver("ipopt-watertap")
   res = solver.solve(m, tee=True)

Notes on configuration
----------------------

- This dynamic model enforces ``dynamic=True`` and ``has_holdup=True``.
- If your parent process wrapper was originally steady, instantiate this unit in
  a new dynamic flowsheet setup rather than replacing the steady unit in-place.
- For stiff transient problems, start with modest time resolution and increase
  ``nfe`` gradually.

Numerical guidance
------------------

- Provide realistic initial states before initialization.
- Use consistent scaling (including property-package scaling).
- Prefer small transient horizons first to validate setup, then extend the
  simulated time window.

Current limitations and extension path
--------------------------------------

To approach full parity with ``base.py``, next additions would typically be:

- non-ohmic membrane potential terms,
- diffusion-layer polarization terms,
- pressure-drop correlations and coupling to momentum losses,
- enhanced dynamic initialization strategy for large transient studies.

Status in this repository
-------------------------

This page is intentionally standalone documentation for the new dynamic module.
If desired, it can be linked into the docs toctree in a follow-up change.
