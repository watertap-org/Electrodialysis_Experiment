#################################################################################
# Electrodialysis_Experiment is part of the WaterTAP software platform.
#
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Laboratory of the Rockies, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
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
