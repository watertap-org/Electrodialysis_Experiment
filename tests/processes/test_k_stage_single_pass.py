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

from pathlib import Path

import pyomo.environ as pyo
import pytest
import yaml
from pyomo.environ import value

from electrodialysis_experiment.processes.k_stage_single_pass import KStageSinglePass
from electrodialysis_experiment.schema.config.process_config_schema import (
    KStageSinglePassConfig,
)
from electrodialysis_experiment.schema.experiment.data import FluidCondition


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/two_stage_single_pass.yml"
SCALING_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/scaling_tssp.yml"
INIT_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/tssp_init_config.yml"

pytestmark = pytest.mark.requires_solver

def _build_k2_config_from_base_yaml(path: Path) -> KStageSinglePassConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

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


def _build_demo_fluid_condition() -> FluidCondition:
    feed = {
        "flow": 2.5e-5,
        "Na_+": 32.174,
        "Ca_2+": 4.55,
        "Mg_2+": 1.875,
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


def test_k_stage_single_pass_k2_demo_solution_has_expected_tds_split(watertap_solver):
    model = pyo.ConcreteModel()
    model.proc = KStageSinglePass(process_cfg=_build_k2_config_from_base_yaml(BASE_CFG_PATH))
    model.proc.import_init_value_config(INIT_CFG_PATH)
    model.proc.import_scaling_config(SCALING_CFG_PATH)

    model.proc.initialize_process(
        fluid_condition=_build_demo_fluid_condition(),
        solver=watertap_solver,
        tee=False,
    )

    feed_tds = value(model.proc.fs.feed_salinity)
    prod_tds = value(model.proc.fs.prod_salinity)
    disp_tds = value(model.proc.fs.disp_salinity)

    # Reference values from demo_kssp_proc_k2-style solve.
    assert prod_tds == pytest.approx(1.7793, rel=1e-4)
    assert disp_tds == pytest.approx(3.4775, rel=1e-4)

    # Stage-level checks corresponding to display_selected_model_metrics output.
    assert value(model.proc.fs.voltage_avg_stage[1]) == pytest.approx(8.0, rel=1e-6)
    assert value(model.proc.fs.voltage_avg_stage[2]) == pytest.approx(9.0, rel=1e-6)
    assert value(model.proc.fs.EDstack[1].unit.ocv) == pytest.approx(2.0, rel=1e-6)
    assert value(model.proc.fs.EDstack[2].unit.ocv) == pytest.approx(1.0, rel=1e-6)

    assert prod_tds < feed_tds < disp_tds
