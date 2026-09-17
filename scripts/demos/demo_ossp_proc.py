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

"""
Standalone OneStageSinglePass demo.

Purpose
-------
Give a minimal, reproducible example of building and solving a
single-stage electrodialysis (ED) process model without parquet input.

Modeling workflow in this demo
------------------------------
1) Define one feed-state boundary condition as `FluidCondition`.
2) Build `OneStageSinglePass` from process YAML (unit structure + options).
3) Apply scaling and initial-value YAMLs (numerics + fixed operating/design vars).
4) Call `initialize_process(...)`, which initializes units and solves at the
   provided feed condition.

What is hardcoded vs. YAML-driven
---------------------------------
- Hardcoded here: feed flow and feed ion concentrations.
- YAML-driven: ED stack design/operation defaults, membrane parameters,
  currents/voltages/OCV initial values, and scaling factors.
"""

from pathlib import Path

import pyomo.environ as pyo
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom

from electrodialysis_experiment.processes.one_stage_single_pass import (
    OneStageSinglePass,
)
from electrodialysis_experiment.schema.experiment.data import FluidCondition


# Process topology/configuration for OneStageSinglePass.
ONE_STAGE_CFG = Path("src/electrodialysis_experiment/configs/one_stage_single_pass.yml")
# User scaling factors to improve solver conditioning.
SCALING_CFG = Path("src/electrodialysis_experiment/configs/scaling.yml")
# Initial values/fixes for model variables (including ED-specific parameters).
INIT_CFG = Path("src/electrodialysis_experiment/configs/ossp_init_config.yml")


def build_demo_fluid_condition() -> FluidCondition:
    """
    Build one feed boundary-condition object for `initialize_process`.

    The process property package in this project is configured for
    Na+, Ca2+, Mg2+, and Cl-. We provide Na/Ca/Mg directly, and compute Cl-
    from electroneutrality (assuming chloride is the balancing anion):

        Cl- = Na+ + 2*Ca2+ + 2*Mg2+

    Returns
    -------
    FluidCondition
        A state dictionary wrapper containing volumetric flow rate and
        solute molar concentrations used to initialize the feed unit.
    """
    feed = {
        "flow": 2.5e-5,  # m^3/s
        "Na_+": 32.174,  # mol/m^3
        "Ca_2+": 4.5500,  # mol/m^3
        "Mg_2+": 1.875,  # mol/m^3
    }
    cl = feed["Na_+"] + 2 * feed["Ca_2+"] + 2 * feed["Mg_2+"]

    return FluidCondition(
        flow_vol_phase={"Liq": feed["flow"]},
        conc_mol_phase_comp={
            ("Liq", "Na_+"): feed["Na_+"],
            ("Liq", "Ca_2+"): feed["Ca_2+"],
            ("Liq", "Mg_2+"): feed["Mg_2+"],
            ("Liq", "Cl_-"): cl,
        },
    )


def main():
    """
    Build, initialize, and solve a one-stage ED process with a hardcoded feed.

    Notes on sequence:
    - The process block is created first from YAML.
    - Scaling and initial values are imported before solve.
    - `initialize_process` both initializes units and solves the full process.
    """
    test_cond = build_demo_fluid_condition()
    print("Demo fluid condition:", test_cond)

    m = pyo.ConcreteModel()
    # Attach the process block at m.proc; flowsheet is available as m.proc.fs.
    m.proc = OneStageSinglePass.from_yaml(ONE_STAGE_CFG)
    # Apply numerical scaling and variable initial values/fixes.
    m.proc.import_scaling_config(SCALING_CFG)
    m.proc.import_init_value_config(INIT_CFG)

    # DOF check before initialization/solve (useful sanity check).
    dof_before = degrees_of_freedom(m.proc.fs)
    print(f"DOF before initialize_process: {dof_before}")

    # Performs staged unit initialization, state propagation, and process solve.
    m.proc.initialize_process(fluid_condition=test_cond, solver=get_solver(), tee=True)

    # Optional post-solve summary table from the process helper method.
    solutes = m.proc.fs.properties.solute_set
    m.proc.display_selected_model_metrics(solutes)


if __name__ == "__main__":
    main()
