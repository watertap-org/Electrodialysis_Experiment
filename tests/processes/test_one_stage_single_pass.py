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
from pyomo.environ import value

from electrodialysis_experiment.processes.one_stage_single_pass import (
    OneStageSinglePass,
)
from electrodialysis_experiment.schema.experiment.data import FluidCondition


REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/one_stage_single_pass.yml"
SCALING_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/scaling.yml"
INIT_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/ossp_init_config.yml"

pytestmark = pytest.mark.requires_solver

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


def test_one_stage_single_pass_demo_solution_has_expected_tds_split(watertap_solver):
    model = pyo.ConcreteModel()
    model.proc = OneStageSinglePass.from_yaml(PROC_CFG_PATH)
    model.proc.import_scaling_config(SCALING_CFG_PATH)
    model.proc.import_init_value_config(INIT_CFG_PATH)

    model.proc.initialize_process(
        fluid_condition=_build_demo_fluid_condition(),
        solver=watertap_solver,
        tee=False,
    )

    feed_tds = value(model.proc.fs.feed_salinity)
    prod_tds = value(model.proc.fs.prod_salinity)
    disp_tds = value(model.proc.fs.disp_salinity)

    # Reference values from demo_ossp_proc-style solve.
    assert prod_tds == pytest.approx(2.5081, rel=1e-4)
    assert disp_tds == pytest.approx(2.7595, rel=1e-4)

    # Basic physical sanity checks: product is less saline, disposal is more saline.
    assert prod_tds < feed_tds < disp_tds
    assert value(model.proc.fs.prod.properties[0].flow_vol_phase["Liq"]) < value(
        model.proc.fs.feed.properties[0].flow_vol_phase["Liq"]
    )
