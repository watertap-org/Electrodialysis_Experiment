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
Standalone KStageSinglePass demo (k=2).

Purpose
-------
Provide a minimal, user-facing example of building and solving a two-stage
single-pass electrodialysis (ED) process using `KStageSinglePass`.

Modeling workflow in this demo
------------------------------
1) Define one feed-state boundary condition as `FluidCondition`.
2) Build a k=2 process config from the existing four-stage YAML by taking
   `ed_stack_1` and `ed_stack_2`.
3) Apply stage-aware init/scaling YAML files prepared for indexed ED stacks.
4) Apply stage-specific operating updates (e.g., stack voltage and cell-pair count).
5) Apply stage-specific cation transport-number estimates in CEM.
6) Initialize and solve with `initialize_process(...)`.

"""

from pathlib import Path
import yaml

import pyomo.environ as pyo
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom

from electrodialysis_experiment.processes.k_stage_single_pass import KStageSinglePass
from electrodialysis_experiment.schema.config.process_config_schema import (
    KStageSinglePassConfig,
)
from electrodialysis_experiment.schema.experiment.data import (
    FluidCondition,
    UpdateParam_stg,
)


# Source YAML that already defines ED stack options per stage.
BASE_CFG_PATH = Path("src/electrodialysis_experiment/configs/two_stage_single_pass.yml")
# K=2-specific init/scaling files (indexed ED paths: fs.EDstack[stage].unit...).
INIT_CFG_PATH = Path("src/electrodialysis_experiment/configs/tssp_init_config.yml")
SCALING_CFG_PATH = Path("src/electrodialysis_experiment/configs/scaling_tssp.yml")


def build_k2_config_from_four_stage_yaml(path: Path) -> KStageSinglePassConfig:
    """
    Build a KStageSinglePassConfig (num_stages=2) from the existing 4-stage YAML.

    This keeps your existing configuration source and reuses only stage 1 and 2
    ED stack definitions.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if "ed_stack_1" not in raw or "ed_stack_2" not in raw:
        raise KeyError("Expected both 'ed_stack_1' and 'ed_stack_2' in base YAML.")

    ed1 = raw["ed_stack_1"]
    ed2 = raw["ed_stack_2"]
    return KStageSinglePassConfig(
        num_stages=2,
        ed_stack=ed1,
        ed_stacks={1: ed1, 2: ed2},
        ion=raw["ion"],
        solution=raw.get("solution", {}),
        process=raw.get("process", {}),
        ipopt=raw.get("ipopt", {}),
    )


def build_demo_fluid_condition() -> FluidCondition:
    """
    Build one feed boundary-condition object for initialize_process.

    Cl- is computed by electroneutrality:
        Cl- = Na+ + 2*Ca2+ + 2*Mg2+
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
    Build, initialize, and solve a two-stage KStageSinglePass process model.
    """
    test_cond = build_demo_fluid_condition()
    print("Demo feed condition:", test_cond)

    m = pyo.ConcreteModel()
    m.proc = KStageSinglePass(
        process_cfg=build_k2_config_from_four_stage_yaml(BASE_CFG_PATH)
    )
    m.proc.import_init_value_config(INIT_CFG_PATH)
    m.proc.import_scaling_config(SCALING_CFG_PATH)

    dof_before = degrees_of_freedom(m.proc.fs)
    print(f"DOF before initialize_process: {dof_before}")

    m.proc.initialize_process(fluid_condition=test_cond, solver=get_solver(), tee=True)

    dof_after = degrees_of_freedom(m.proc.fs)
    print(f"DOF after initialize_process: {dof_after}")

    # Optional post-solve summary table.
    solutes = m.proc.fs.properties.solute_set
    m.proc.display_selected_model_metrics(solutes)


if __name__ == "__main__":
    main()
