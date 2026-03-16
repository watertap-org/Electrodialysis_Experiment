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