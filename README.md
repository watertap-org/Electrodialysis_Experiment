# Electrodialysis Experiment

A Python package for simulating electrodialysis experiments to assist electrodialysis process design, control, and optimization. 

## Start Here: Solver + Environment (Recommended)

If you use IDAES-managed solver binaries (recommended for this project), do this in a fresh conda environment:

```bash
conda create -n edexp python=3.11 -y
conda activate edexp
pip install -e ".[dev]"
idaes get-extensions
```

For this package, use the WaterTAP IPOPT wrapper for process initialization and NLP solves:

```python
from idaes.core.solvers import get_solver

solver = get_solver("ipopt-watertap")
```

This is the package default expectation for initialization and NLP solves.

Verify IPOPT + MA27 from Python:

```bash
python - <<'PY'
import watertap_solvers
from idaes.core.solvers import get_solver
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, minimize

s = get_solver("ipopt-watertap")
print("IPOPT available:", s.available(exception_flag=False))

m = ConcreteModel()
m.x = Var(initialize=1.0)
m.obj = Objective(expr=(m.x - 2.0)**2, sense=minimize)
m.c = Constraint(expr=m.x >= 0)

s.options["linear_solver"] = "ma27"
res = s.solve(m, tee=False)
print("Termination:", res.solver.termination_condition)
PY
```

Notes:
- If you use `idaes get-extensions`, you generally do not need a separate `conda install ipopt`.

## Installation

If your environment is already prepared, install in editable mode:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Prerequisites

- Python: `3.9`-`3.11` (see `pyproject.toml`)
- IPOPT available on PATH (preferred linear solver: `ma27`)
- Fast tests: `pytest -m "not slow"`
- Full tests: `pytest`

## Demo Files

All current demo files in `scripts/demos/`:
- `scripts/demos/model_training_walkthrough.ipynb`
- `scripts/demos/demo_ossp_proc.py`
- `scripts/demos/demo_kssp_proc_k2.py`

## Documentation

Project documentation lives in `docs/`.

- Main entry: `docs/index.rst`
- Workflow overview: `docs/simulation-workflows.md`
- Docs map: `docs/README.md`

## Project Structure

```
ELECTRODIALYSIS_EXPERIMENT/
├─ src/electrodialysis_experiment/   # Core package code
├─ scripts/                          # Runner scripts
├─ docs/                             # Project documentation
├─ tests/                            # Unit tests
├─ pyproject.toml                    # Build and dependency info
└─ README.md                         # Project description
```
## License

This project uses a WaterTAP-style license and copyright model.
See `LICENSE.md` and `COPYRIGHT.md` in the repository root.

## Contact

Main contact: Xiangyu Bi (xbi@lbl.gov)
