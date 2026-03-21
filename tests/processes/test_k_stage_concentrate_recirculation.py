from __future__ import annotations

from pathlib import Path

import pyomo.environ as pyo
import pytest
import yaml
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import value

from electrodialysis_experiment.processes.k_stage_concentrate_recirculation import (
    KStageConcentrateRecirculation,
)
from electrodialysis_experiment.schema.config.process_config_schema import (
    KStageSinglePassConfig,
)
from electrodialysis_experiment.schema.experiment.data import FluidCondition


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/two_stage_conc_recir.yml"
SCALING_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/scaling_ts_cc.yml"
INIT_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/tscr_init_config.yml"

pytestmark = pytest.mark.requires_solver


def _build_k2_config(path: Path) -> KStageSinglePassConfig:
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


def _build_demo_model():
    model = pyo.ConcreteModel()
    model.proc = KStageConcentrateRecirculation(process_cfg=_build_k2_config(BASE_CFG_PATH))
    model.proc.import_init_value_config(INIT_CFG_PATH)
    model.proc.import_scaling_config(SCALING_CFG_PATH)
    model.proc.set_feed_split_to_diluate(0.50, fix=False)
    model.proc.set_concentrate_recycle_fraction(0.50, fix=False)
    return model


def test_k_stage_concentrate_recirculation_k2_demo_solution_matches_reference(
    watertap_solver,
):
    model = _build_demo_model()

    model.proc.initialize_process(
        fluid_condition=_build_demo_fluid_condition(),
        solve_after_init=False,
        solver=watertap_solver,
        tee=False,
    )

    assert degrees_of_freedom(model.proc.fs) == 0

    result = model.proc.solve(model.proc.fs, solver=watertap_solver, tee=False)
    assert str(result.solver.termination_condition).lower() == "optimal"
    assert degrees_of_freedom(model.proc.fs) == 0

    fs = model.proc.fs

    expected_stream_values = {
        "feed_flow": 2.5e-5,
        "prod_flow": 1.25e-5,
        "disp_flow": 1.25e-5,
        "feed_tds": 2.633904,
        "prod_tds": 2.286195,
        "disp_tds": 2.981613,
        "feed_na": 32.174000,
        "prod_na": 26.707982,
        "disp_na": 37.640018,
        "feed_ca": 4.550000,
        "prod_ca": 4.485690,
        "disp_ca": 4.614310,
        "feed_mg": 1.875000,
        "prod_mg": 1.700448,
        "disp_mg": 2.049552,
        "feed_cl": 45.024000,
        "prod_cl": 39.074230,
        "disp_cl": 50.973770,
    }

    observed_stream_values = {
        "feed_flow": value(fs.feed.properties[0].flow_vol_phase["Liq"]),
        "prod_flow": value(fs.prod.properties[0].flow_vol_phase["Liq"]),
        "disp_flow": value(fs.disp.properties[0].flow_vol_phase["Liq"]),
        "feed_tds": value(fs.feed_salinity),
        "prod_tds": value(fs.prod_salinity),
        "disp_tds": value(fs.disp_salinity),
        "feed_na": value(fs.feed.properties[0].conc_mol_phase_comp["Liq", "Na_+"]),
        "prod_na": value(fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]),
        "disp_na": value(fs.disp.properties[0].conc_mol_phase_comp["Liq", "Na_+"]),
        "feed_ca": value(fs.feed.properties[0].conc_mol_phase_comp["Liq", "Ca_2+"]),
        "prod_ca": value(fs.prod.properties[0].conc_mol_phase_comp["Liq", "Ca_2+"]),
        "disp_ca": value(fs.disp.properties[0].conc_mol_phase_comp["Liq", "Ca_2+"]),
        "feed_mg": value(fs.feed.properties[0].conc_mol_phase_comp["Liq", "Mg_2+"]),
        "prod_mg": value(fs.prod.properties[0].conc_mol_phase_comp["Liq", "Mg_2+"]),
        "disp_mg": value(fs.disp.properties[0].conc_mol_phase_comp["Liq", "Mg_2+"]),
        "feed_cl": value(fs.feed.properties[0].conc_mol_phase_comp["Liq", "Cl_-"]),
        "prod_cl": value(fs.prod.properties[0].conc_mol_phase_comp["Liq", "Cl_-"]),
        "disp_cl": value(fs.disp.properties[0].conc_mol_phase_comp["Liq", "Cl_-"]),
    }

    for name, expected in expected_stream_values.items():
        assert observed_stream_values[name] == pytest.approx(expected, rel=1e-4, abs=1e-8)

    expected_stage_values = {
        "s1_dil_na": 29.700545,
        "s2_dil_na": 26.707982,
        "s1_conc_na": 34.660576,
        "s2_conc_na": 37.640018,
        "s1_removal_na": 7.687744,
        "s2_removal_na": 10.075784,
        "s1_dil_ca": 4.520863,
        "s2_dil_ca": 4.485690,
        "s1_conc_ca": 4.579292,
        "s2_conc_ca": 4.614310,
        "s1_removal_ca": 0.640379,
        "s2_removal_ca": 0.778005,
        "s1_dil_mg": 1.796005,
        "s2_dil_mg": 1.700448,
        "s1_conc_mg": 1.954414,
        "s2_conc_mg": 2.049552,
        "s1_removal_mg": 4.213046,
        "s2_removal_mg": 5.320536,
        "s1_dil_cl": 42.331553,
        "s2_dil_cl": 39.074230,
        "s1_conc_cl": 47.730729,
        "s2_conc_cl": 50.973770,
        "s1_removal_cl": 5.980025,
        "s2_removal_cl": 7.694788,
    }

    ed1 = fs.EDstack[1].unit
    ed2 = fs.EDstack[2].unit
    observed_stage_values = {
        "s1_dil_na": value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Na_+"]),
        "s2_dil_na": value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Na_+"]),
        "s1_conc_na": value(ed1.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Na_+"]),
        "s2_conc_na": value(ed2.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Na_+"]),
        "s1_removal_na": (
            (
                value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Na_+"])
                - value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Na_+"])
            )
            / value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Na_+"])
            * 100
        ),
        "s2_removal_na": (
            (
                value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Na_+"])
                - value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Na_+"])
            )
            / value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Na_+"])
            * 100
        ),
        "s1_dil_ca": value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Ca_2+"]),
        "s2_dil_ca": value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Ca_2+"]),
        "s1_conc_ca": value(ed1.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Ca_2+"]),
        "s2_conc_ca": value(ed2.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Ca_2+"]),
        "s1_removal_ca": (
            (
                value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Ca_2+"])
                - value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Ca_2+"])
            )
            / value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Ca_2+"])
            * 100
        ),
        "s2_removal_ca": (
            (
                value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Ca_2+"])
                - value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Ca_2+"])
            )
            / value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Ca_2+"])
            * 100
        ),
        "s1_dil_mg": value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Mg_2+"]),
        "s2_dil_mg": value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Mg_2+"]),
        "s1_conc_mg": value(ed1.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Mg_2+"]),
        "s2_conc_mg": value(ed2.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Mg_2+"]),
        "s1_removal_mg": (
            (
                value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Mg_2+"])
                - value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Mg_2+"])
            )
            / value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Mg_2+"])
            * 100
        ),
        "s2_removal_mg": (
            (
                value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Mg_2+"])
                - value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Mg_2+"])
            )
            / value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Mg_2+"])
            * 100
        ),
        "s1_dil_cl": value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Cl_-"]),
        "s2_dil_cl": value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Cl_-"]),
        "s1_conc_cl": value(ed1.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Cl_-"]),
        "s2_conc_cl": value(ed2.concentrate.properties[0, 1].conc_mol_phase_comp["Liq", "Cl_-"]),
        "s1_removal_cl": (
            (
                value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Cl_-"])
                - value(ed1.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Cl_-"])
            )
            / value(ed1.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Cl_-"])
            * 100
        ),
        "s2_removal_cl": (
            (
                value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Cl_-"])
                - value(ed2.diluate.properties[0, 1].conc_mol_phase_comp["Liq", "Cl_-"])
            )
            / value(ed2.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Cl_-"])
            * 100
        ),
    }

    for name, expected in expected_stage_values.items():
        assert observed_stage_values[name] == pytest.approx(expected, rel=1e-4, abs=1e-8)

    expected_performance_values = {
        "recovery_vol_H2O": 0.5,
        "mem_area": 0.4418,
        "stage1_cell_pair_num": 10.0,
        "stage2_cell_pair_num": 10.0,
        "stage1_channel_height": 0.00076,
        "stage2_channel_height": 0.00076,
        "stage1_cell_length": 0.47,
        "stage2_cell_length": 0.47,
        "stage1_cell_width": 0.047,
        "stage2_cell_width": 0.047,
        "stage1_voltage": 1.503979,
        "stage2_voltage": 1.914028,
        "stage1_voltage_per_cp": 0.150398,
        "stage2_voltage_per_cp": 0.191403,
        "stage1_current": 0.33,
        "stage2_current": 0.4,
        "stage1_current_utilization": 1.0,
        "stage2_current_utilization": 1.0,
    }

    observed_performance_values = {
        "recovery_vol_H2O": value(fs.recovery_vol_H2O),
        "mem_area": value(fs.mem_area),
        "stage1_cell_pair_num": value(ed1.cell_pair_num),
        "stage2_cell_pair_num": value(ed2.cell_pair_num),
        "stage1_channel_height": value(ed1.channel_height),
        "stage2_channel_height": value(ed2.channel_height),
        "stage1_cell_length": value(ed1.cell_length),
        "stage2_cell_length": value(ed2.cell_length),
        "stage1_cell_width": value(ed1.cell_width),
        "stage2_cell_width": value(ed2.cell_width),
        "stage1_voltage": value(fs.voltage_avg_stage[1]),
        "stage2_voltage": value(fs.voltage_avg_stage[2]),
        "stage1_voltage_per_cp": value(fs.voltage_per_cp_stage[1]),
        "stage2_voltage_per_cp": value(fs.voltage_per_cp_stage[2]),
        "stage1_current": value(ed1.current_applied[0]),
        "stage2_current": value(ed2.current_applied[0]),
        "stage1_current_utilization": value(ed1.current_utilization),
        "stage2_current_utilization": value(ed2.current_utilization),
    }

    for name, expected in expected_performance_values.items():
        assert observed_performance_values[name] == pytest.approx(
            expected, rel=1e-4, abs=1e-8
        )

    expected_pt_values = {
        "feed_pressure": 101325.0,
        "pump0_in_pressure": 101325.0,
        "pump1_in_pressure": 101325.0,
        "pump0_out_pressure": 242325.0,
        "pump1_out_pressure": 242325.0,
        "ed1_in_dil_pressure": 242325.0,
        "ed1_in_conc_pressure": 242325.0,
        "ed1_out_dil_pressure": 171825.0,
        "ed1_out_conc_pressure": 171825.0,
        "ed2_in_dil_pressure": 171825.0,
        "ed2_in_conc_pressure": 171825.0,
        "ed2_out_dil_pressure": 101325.0,
        "ed2_out_conc_pressure": 101325.0,
        "prod_pressure": 101325.0,
        "disp_pressure": 101325.0,
        "feed_temperature": 298.15,
        "pump0_in_temperature": 298.15,
        "pump1_in_temperature": 298.15,
        "pump0_out_temperature": 298.15,
        "pump1_out_temperature": 298.15,
        "ed1_in_dil_temperature": 298.15,
        "ed1_in_conc_temperature": 298.15,
        "ed1_out_dil_temperature": 298.15,
        "ed1_out_conc_temperature": 298.15,
        "ed2_in_dil_temperature": 298.15,
        "ed2_in_conc_temperature": 298.15,
        "ed2_out_dil_temperature": 298.15,
        "ed2_out_conc_temperature": 298.15,
        "prod_temperature": 298.15,
        "disp_temperature": 298.15,
    }

    observed_pt_values = {
        "feed_pressure": value(fs.feed.outlet.pressure[0]),
        "pump0_in_pressure": value(fs.pump0.inlet.pressure[0]),
        "pump1_in_pressure": value(fs.pump1.inlet.pressure[0]),
        "pump0_out_pressure": value(fs.pump0.outlet.pressure[0]),
        "pump1_out_pressure": value(fs.pump1.outlet.pressure[0]),
        "ed1_in_dil_pressure": value(ed1.inlet_diluate.pressure[0]),
        "ed1_in_conc_pressure": value(ed1.inlet_concentrate.pressure[0]),
        "ed1_out_dil_pressure": value(ed1.outlet_diluate.pressure[0]),
        "ed1_out_conc_pressure": value(ed1.outlet_concentrate.pressure[0]),
        "ed2_in_dil_pressure": value(ed2.inlet_diluate.pressure[0]),
        "ed2_in_conc_pressure": value(ed2.inlet_concentrate.pressure[0]),
        "ed2_out_dil_pressure": value(ed2.outlet_diluate.pressure[0]),
        "ed2_out_conc_pressure": value(ed2.outlet_concentrate.pressure[0]),
        "prod_pressure": value(fs.prod.inlet.pressure[0]),
        "disp_pressure": value(fs.disp.inlet.pressure[0]),
        "feed_temperature": value(fs.feed.outlet.temperature[0]),
        "pump0_in_temperature": value(fs.pump0.inlet.temperature[0]),
        "pump1_in_temperature": value(fs.pump1.inlet.temperature[0]),
        "pump0_out_temperature": value(fs.pump0.outlet.temperature[0]),
        "pump1_out_temperature": value(fs.pump1.outlet.temperature[0]),
        "ed1_in_dil_temperature": value(ed1.inlet_diluate.temperature[0]),
        "ed1_in_conc_temperature": value(ed1.inlet_concentrate.temperature[0]),
        "ed1_out_dil_temperature": value(ed1.outlet_diluate.temperature[0]),
        "ed1_out_conc_temperature": value(ed1.outlet_concentrate.temperature[0]),
        "ed2_in_dil_temperature": value(ed2.inlet_diluate.temperature[0]),
        "ed2_in_conc_temperature": value(ed2.inlet_concentrate.temperature[0]),
        "ed2_out_dil_temperature": value(ed2.outlet_diluate.temperature[0]),
        "ed2_out_conc_temperature": value(ed2.outlet_concentrate.temperature[0]),
        "prod_temperature": value(fs.prod.inlet.temperature[0]),
        "disp_temperature": value(fs.disp.inlet.temperature[0]),
    }

    for name, expected in expected_pt_values.items():
        assert observed_pt_values[name] == pytest.approx(expected, rel=1e-5, abs=1e-8)

    assert value(ed1.ocv) == pytest.approx(2.0, rel=1e-6)
    assert value(ed2.ocv) == pytest.approx(1.0, rel=1e-6)

    assert observed_stream_values["prod_tds"] < observed_stream_values["feed_tds"]
    assert observed_stream_values["disp_tds"] > observed_stream_values["feed_tds"]
