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
from pyomo.environ import SolverFactory
from pathlib import Path
import yaml
import idaes.logger as idaeslogger
from idaes.core.solvers import get_solver

_log = idaeslogger.getIdaesLogger(__name__)
def config_ipopt_solver(config_yaml: str|Path, solver=None): 
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)
    ipopt_config=config["solver"].get("ipopt", {})
    if solver is None:
        solver = get_solver("ipopt-watertap")
        _log.info("No IPOPT solver instance provided; using WaterTAP's default IPOPT solver.")
    solver.options.update(ipopt_config)
    return solver