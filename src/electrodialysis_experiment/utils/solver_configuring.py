from pyomo.environ import SolverFactory
from pathlib import Path
import yaml
import idaes.logger as idaeslogger

_log = idaeslogger.getIdaesLogger(__name__)
def config_ipopt_solver(config_yaml: str|Path, solver=None): 
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)
    ipopt_config=config["solver"].get("ipopt", {})
    if solver is None:
        solver = SolverFactory("ipopt")
        _log.info("No IPOPT solver instance provided; using Pyomo SolverFactory's default IPOPT solver.")
        solver = SolverFactory("ipopt")
    solver.options.update(ipopt_config)
    return solver