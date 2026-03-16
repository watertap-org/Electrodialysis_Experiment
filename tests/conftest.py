from __future__ import annotations

import pytest
from idaes.core.solvers import get_solver


@pytest.fixture(scope="session")
def watertap_solver():
    """Return ipopt-watertap solver or skip tests that require it."""
    try:
        solver = get_solver("ipopt-watertap")
    except Exception as err:  # pragma: no cover - env-dependent path
        pytest.skip(f"ipopt-watertap is not available: {err}")

    if hasattr(solver, "available") and not solver.available(exception_flag=False):
        pytest.skip("ipopt-watertap solver is not available in this environment.")

    return solver
