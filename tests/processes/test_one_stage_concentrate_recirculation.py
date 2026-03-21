from __future__ import annotations

from pathlib import Path

import pyomo.environ as pyo
import pytest
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import value

from electrodialysis_experiment.processes.one_stage_concentrate_recirculation import (
    OneStageConcentrateRecirculation,
)
from electrodialysis_experiment.schema.experiment.data import FluidCondition


REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_CFG_PATH = (
    REPO_ROOT / "src/electrodialysis_experiment/configs/one_stage_conc_recir.yml"
)
SCALING_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/scaling.yml"
INIT_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/oscr_init_config.yml"

pytestmark = pytest.mark.requires_solver


def _build_demo_fluid_condition() -> FluidCondition:
    feed = {
        "flow": 1 / 60000,
        "Na_+": 32.043478,
        "Ca_2+": 0.4525,
        "Mg_2+": 1.833333,
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
    model.proc = OneStageConcentrateRecirculation.from_yaml(PROC_CFG_PATH)
    model.proc.import_scaling_config(SCALING_CFG_PATH)
    model.proc.import_init_value_config(INIT_CFG_PATH)
    model.proc.set_feed_split_to_diluate(0.50, fix=False)
    model.proc.set_concentrate_recycle_fraction(0.50, fix=False)
    return model


def test_one_stage_concentrate_recirculation_demo_solution_matches_reference(
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
        "feed_flow": 1.0 / 60000.0,
        "prod_flow": 0.0000083333333333,
        "disp_flow": 0.0000083333333333,
        "feed_tds": 2.141986,
        "prod_tds": 1.419440,
        "disp_tds": 2.864532,
        "feed_na": 32.043478,
        "prod_na": 20.923966,
        "disp_na": 43.162990,
        "feed_ca": 0.452500,
        "prod_ca": 0.329312,
        "disp_ca": 0.575688,
        "feed_mg": 1.833333,
        "prod_mg": 1.340669,
        "disp_mg": 2.325997,
        "feed_cl": 36.615144,
        "prod_cl": 24.263928,
        "disp_cl": 48.966360,
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

    expected_performance_values = {
        "recovery_vol_H2O": 0.50000,
        "mem_area": 0.22090,
        "cell_pair_num": 10.00000,
        "channel_height": 0.00076,
        "cell_length": 0.47000,
        "cell_width": 0.04700,
        "experimental_voltage": 10.00000,
        "voltage_avg": 5.65000,
        "voltage_per_cp": 0.56500,
        "current_applied": 1.00746,
        "current_utilization": 1.00000,
        "ocv": 4.35000,
    }

    observed_performance_values = {
        "recovery_vol_H2O": value(fs.recovery_vol_H2O),
        "mem_area": value(fs.mem_area),
        "cell_pair_num": value(fs.EDstack.cell_pair_num),
        "channel_height": value(fs.EDstack.channel_height),
        "cell_length": value(fs.EDstack.cell_length),
        "cell_width": value(fs.EDstack.cell_width),
        "experimental_voltage": value(fs.experimental_voltage),
        "voltage_avg": value(fs.voltage_avg),
        "voltage_per_cp": value(fs.voltage_per_cp),
        "current_applied": value(fs.EDstack.current_applied[0]),
        "current_utilization": value(fs.EDstack.current_utilization),
        "ocv": value(fs.ocv),
    }

    for name, expected in expected_performance_values.items():
        assert observed_performance_values[name] == pytest.approx(
            expected, rel=1e-4, abs=1e-8
        )

    expected_pt_values = {
        "feed_pressure": 101325.00,
        "pump0_in_pressure": 101325.00,
        "pump1_in_pressure": 101325.00,
        "pump0_out_pressure": 171825.00,
        "pump1_out_pressure": 171825.00,
        "ed_in_dil_pressure": 171825.00,
        "ed_in_conc_pressure": 171825.00,
        "ed_out_dil_pressure": 101325.00,
        "ed_out_conc_pressure": 101325.00,
        "prod_pressure": 101325.00,
        "disp_pressure": 101325.00,
        "feed_temperature": 298.15,
        "pump0_in_temperature": 298.15,
        "pump1_in_temperature": 298.15,
        "pump0_out_temperature": 298.15,
        "pump1_out_temperature": 298.15,
        "ed_in_dil_temperature": 298.15,
        "ed_in_conc_temperature": 298.15,
        "ed_out_dil_temperature": 298.15,
        "ed_out_conc_temperature": 298.15,
        "prod_temperature": 298.15,
        "disp_temperature": 298.15,
    }

    observed_pt_values = {
        "feed_pressure": value(fs.feed.outlet.pressure[0]),
        "pump0_in_pressure": value(fs.pump0.inlet.pressure[0]),
        "pump1_in_pressure": value(fs.pump1.inlet.pressure[0]),
        "pump0_out_pressure": value(fs.pump0.outlet.pressure[0]),
        "pump1_out_pressure": value(fs.pump1.outlet.pressure[0]),
        "ed_in_dil_pressure": value(fs.EDstack.inlet_diluate.pressure[0]),
        "ed_in_conc_pressure": value(fs.EDstack.inlet_concentrate.pressure[0]),
        "ed_out_dil_pressure": value(fs.EDstack.outlet_diluate.pressure[0]),
        "ed_out_conc_pressure": value(fs.EDstack.outlet_concentrate.pressure[0]),
        "prod_pressure": value(fs.prod.inlet.pressure[0]),
        "disp_pressure": value(fs.disp.inlet.pressure[0]),
        "feed_temperature": value(fs.feed.outlet.temperature[0]),
        "pump0_in_temperature": value(fs.pump0.inlet.temperature[0]),
        "pump1_in_temperature": value(fs.pump1.inlet.temperature[0]),
        "pump0_out_temperature": value(fs.pump0.outlet.temperature[0]),
        "pump1_out_temperature": value(fs.pump1.outlet.temperature[0]),
        "ed_in_dil_temperature": value(fs.EDstack.inlet_diluate.temperature[0]),
        "ed_in_conc_temperature": value(fs.EDstack.inlet_concentrate.temperature[0]),
        "ed_out_dil_temperature": value(fs.EDstack.outlet_diluate.temperature[0]),
        "ed_out_conc_temperature": value(fs.EDstack.outlet_concentrate.temperature[0]),
        "prod_temperature": value(fs.prod.inlet.temperature[0]),
        "disp_temperature": value(fs.disp.inlet.temperature[0]),
    }

    for name, expected in expected_pt_values.items():
        assert observed_pt_values[name] == pytest.approx(expected, rel=1e-5, abs=1e-8)

    assert observed_stream_values["prod_tds"] < observed_stream_values["feed_tds"]
    assert observed_stream_values["disp_tds"] > observed_stream_values["feed_tds"]
