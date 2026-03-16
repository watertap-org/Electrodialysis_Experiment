``solver_configuring.py``
=========================

Location
--------

``src/electrodialysis_experiment/utils/solver_configuring.py``

Purpose
-------

Provides one helper, ``config_ipopt_solver(...)``, to load IPOPT options from
YAML and apply them to a solver instance.

Public function
---------------

``config_ipopt_solver(config_yaml, solver=None)``

- reads ``solver.ipopt`` options from YAML,
- creates default IPOPT solver if none is provided,
- updates ``solver.options`` and returns the configured solver.

Role in training workflow
-------------------------

In ``model_training.py``, this function is used to keep solver tuning external
to code. This improves reproducibility and makes solver settings easy to adjust
between runs without modifying script logic.

Practical notes
---------------

- Keys under ``solver.ipopt`` are passed directly to IPOPT.
- If an option is unsupported by the local IPOPT build, solve may fail at
  runtime.
- The helper configures options only; it does not validate problem scaling or
  convergence quality.

YAML Authoring Guidance
-----------------------

Use :doc:`yaml_configuration_guide` for the complete format. For this utility,
the expected shape is:

.. code-block:: yaml

   solver:
     ipopt:
       tol: 1e-8
       max_iter: 3000
       linear_solver: ma27
       nlp_scaling_method: user-scaling

Solver selection vs solver options
----------------------------------

- ``config_ipopt_solver(...)`` configures IPOPT options only.
- Process modules select the solver instance through
  ``idaes.core.solvers.get_solver(...)`` using process-config ``ipopt.solver_name``.
- For this repository, the recommended starting solver is
  ``ipopt-watertap`` (set in process YAML under ``ipopt.solver_name``).
