from pyomo.environ import SolverFactory
from pathlib import Path
import yaml

def get_ipopt_configed_solver(config_yaml: str|Path) :
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)
    ipopt_config=config["solver"].get("ipopt", {})
    solver = SolverFactory("ipopt")
    solver.options.update(ipopt_config)
    return solver