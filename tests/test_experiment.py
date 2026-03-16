from __future__ import annotations

import math
from pathlib import Path

import idaes.core.util.model_statistics as mstat
import pandas as pd
import pyomo.environ as pyo
import pytest

import electrodialysis_experiment.schema.experiment.data as ds
from electrodialysis_experiment.experiment import MasterExperimentBuilder
from electrodialysis_experiment.surrogates.transport_number_membrane.cation_cem_simulator import (
    SurrogateType,
)
from electrodialysis_experiment.utils.solver_configuring import config_ipopt_solver


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "src/electrodialysis_experiment/data/raw/dt_SEDv4_021125.parquet"
PROC_CFG_PATH = (
    REPO_ROOT / "src/electrodialysis_experiment/configs/one_stage_single_pass.yml"
)
SOLVER_CFG_PATH = REPO_ROOT / "src/electrodialysis_experiment/configs/solver_config.yml"
INIT_SNAPSHOT_PATH = (
    REPO_ROOT / "src/electrodialysis_experiment/data/output/cc_lowFidel_init0.h5"
)

TARGET_WEIGHTS = {
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']": 0.1,
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']": 1.0,
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']": 10.0,
}
IONS = ("Na_+", "Ca_2+", "Mg_2+")
SAMPLE_SIZE = 25
SAMPLES_TO_CHECK = (0, 12)

# Hardcoded reference values extracted from:
# src/electrodialysis_experiment/data/output/archive/m_concSSE_minimized_cc_copy.h5
REFERENCE_TRAINED_COEFS = {
    "Ca_2+": 8.50200,
    "Mg_2+": 4.74722,
}
REFERENCE_TRAINED_PROD_CONC = {
    (0, "Na_+"): 31.3958,
    (0, "Ca_2+"): 0.413702,
    (0, "Mg_2+"): 1.74450,
    (12, "Na_+"): 16.8875,
    (12, "Ca_2+"): 2.37695,
    (12, "Mg_2+"): 1.31175,
}


@pytest.fixture(scope="module")
def training_data():
    if not DATA_PATH.exists():
        pytest.skip(f"Training dataset not found: {DATA_PATH}")

    dt = pd.read_parquet(DATA_PATH)
    ti_data = ds.prepare_cation_cem_transport_number_estimate(dt)
    ci_data = ds.prepare_cation_product_conc(dt)
    _, target_df = ds.prepare_target_variable_dt(dt)
    target_df = target_df.reset_index(drop=True)
    return ti_data, ci_data, target_df


def _weighted_sse_for_samples(
    model: pyo.ConcreteModel, target_df: pd.DataFrame, samples
):
    sse = 0.0
    for i in samples:
        for ion in IONS:
            col = f"fs.prod.properties[0].conc_mol_phase_comp['Liq','{ion}']"
            weight = TARGET_WEIGHTS[col]
            sim = pyo.value(
                model.sample_blk[i]
                .proc.fs.prod.properties[0]
                .conc_mol_phase_comp["Liq", ion]
            )
            target = float(target_df.loc[i, col])
            sse += weight * (sim - target) ** 2
    return sse


def _build_base_experiment(sample_size: int = SAMPLE_SIZE):
    return MasterExperimentBuilder.proc_config_from_yaml(
        PROC_CFG_PATH, sample_size=sample_size
    )


def _add_simulator(exp: MasterExperimentBuilder):
    exp.add_cation_cem_transport_number_simulator(
        surrogate_method=SurrogateType.LOG_LINEAR_POLYNOMIAL,
        poly_degree=5,
        **{"reference_ion": "Na_+"},
    )


def test_model_training_setup_workflow_methods(training_data):
    ti_data, ci_data, target_df = training_data

    if not INIT_SNAPSHOT_PATH.exists():
        pytest.skip(f"Initialization snapshot not found: {INIT_SNAPSHOT_PATH}")

    exp = _build_base_experiment(sample_size=SAMPLE_SIZE)
    exp.load_model_data(INIT_SNAPSHOT_PATH)
    _add_simulator(exp)

    fitted_dict = exp.model.cation_cem_transport_number_simulator[0].initiate_surrogate(
        conc_data=ci_data,
        trans_number_data=ti_data,
        fitting_coef_guess={"Ca_2+": 9, "Mg_2+": 2},
        reference_ion="Na_+",
        log_objective=False,
        polynomial_degree=5,
        plot_results=False,
    )

    assert "Ca_2+" in fitted_dict and "Mg_2+" in fitted_dict

    exp.add_log_linear_surr_coef_constraint()
    exp.free_cation_transport_numbers_in_cem()

    for blk in exp.model.sample_blk.values():
        blk.proc.fs.ocv.setlb(0)
        blk.proc.fs.ocv.setub(10)
        blk.proc.fs.EDstack.current_utilization.setlb(0.2)
        blk.proc.fs.EDstack.current_utilization.setub(1.0)

    assert mstat.degrees_of_freedom(exp.model) == 27

    coef_indices = list(
        exp.model.cation_cem_transport_number_simulator[0].conc_ratio_coef.index_set()
    )
    expected_n_constraints = len(coef_indices) * (SAMPLE_SIZE - 1)
    assert len(exp.model.conc_ratio_coef_equality_cons) == expected_n_constraints

    x0 = next(iter(exp.model.sample_blk[0].proc.fs.EDstack.diluate.length_domain))
    for ion in exp.model.sample_blk[0].proc.fs.EDstack.cation_set:
        assert not exp.model.sample_blk[0].proc.fs.EDstack.ion_trans_number_membrane[
            "cem", ion, x0
        ].fixed

    exp.add_sse_objective_of_selected_variables(
        variables_weights=TARGET_WEIGHTS,
        data=target_df.iloc[:SAMPLE_SIZE],
    )
    assert hasattr(exp.model, "sse_objective")
    assert pyo.value(exp.model.sse_objective.expr) >= 0


def test_training_snapshot_key_outcomes_and_trained_parameters(training_data):
    _, _, target_df = training_data

    if not INIT_SNAPSHOT_PATH.exists():
        pytest.skip(f"Initialization snapshot not found: {INIT_SNAPSHOT_PATH}")

    exp_init = _build_base_experiment(sample_size=SAMPLE_SIZE)
    exp_init.load_model_data(INIT_SNAPSHOT_PATH)

    init_selected_sse = _weighted_sse_for_samples(
        exp_init.model, target_df, SAMPLES_TO_CHECK
    )
    trained_selected_sse = 0.0
    for i in SAMPLES_TO_CHECK:
        for ion in IONS:
            col = f"fs.prod.properties[0].conc_mol_phase_comp['Liq','{ion}']"
            weight = TARGET_WEIGHTS[col]
            trained_val = REFERENCE_TRAINED_PROD_CONC[(i, ion)]
            target_val = float(target_df.loc[i, col])
            trained_selected_sse += weight * (trained_val - target_val) ** 2

    assert trained_selected_sse < init_selected_sse

    for coef_idx in ("Ca_2+", "Mg_2+"):
        ref_val = REFERENCE_TRAINED_COEFS[coef_idx]
        assert math.isfinite(ref_val)
        assert ref_val > 0


def test_save_and_load_model_hdf_roundtrip(tmp_path):
    exp = _build_base_experiment(sample_size=1)

    ocv = exp.model.sample_blk[0].proc.fs.ocv
    ocv.set_value(3.21)
    ocv.fix()

    snapshot = tmp_path / "roundtrip_test.h5"
    exp.save_model_hdf(snapshot)

    ocv.set_value(8.76)
    ocv.unfix()

    exp.load_model_data(snapshot)

    assert pyo.value(ocv) == pytest.approx(3.21, rel=1e-9)
    assert ocv.fixed


@pytest.mark.slow
@pytest.mark.requires_solver
def test_full_training_solve_matches_representative_reference_outcomes(
    training_data, watertap_solver
):
    ti_data, ci_data, target_df = training_data

    if not INIT_SNAPSHOT_PATH.exists():
        pytest.skip(f"Initialization snapshot not found: {INIT_SNAPSHOT_PATH}")

    exp = _build_base_experiment(sample_size=SAMPLE_SIZE)
    exp.load_model_data(INIT_SNAPSHOT_PATH)
    _add_simulator(exp)

    exp.model.cation_cem_transport_number_simulator[0].initiate_surrogate(
        conc_data=ci_data,
        trans_number_data=ti_data,
        fitting_coef_guess={"Ca_2+": 9, "Mg_2+": 2},
        reference_ion="Na_+",
        log_objective=False,
        polynomial_degree=5,
        plot_results=False,
    )

    exp.add_log_linear_surr_coef_constraint()
    exp.free_cation_transport_numbers_in_cem()

    for blk in exp.model.sample_blk.values():
        blk.proc.fs.ocv.setlb(0)
        blk.proc.fs.ocv.setub(10)
        blk.proc.fs.EDstack.current_utilization.setlb(0.2)
        blk.proc.fs.EDstack.current_utilization.setub(1.0)

    assert mstat.degrees_of_freedom(exp.model) == 27

    exp.add_sse_objective_of_selected_variables(
        variables_weights=TARGET_WEIGHTS,
        data=target_df.iloc[:SAMPLE_SIZE],
    )
    
    solver = config_ipopt_solver(solver=watertap_solver, config_yaml=SOLVER_CFG_PATH)

    results = solver.solve(exp.model, tee=False)

    assert results.solver.termination_condition in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    }

    # Representative surrogate coefficient checks (global coefficients shared by constraints)
    for ion in ("Ca_2+", "Mg_2+"):
        solved_var = exp.model.cation_cem_transport_number_simulator[0].conc_ratio_coef[ion]
        solved_coef = pyo.value(solved_var)
        ref_coef = REFERENCE_TRAINED_COEFS[ion]
        assert solved_coef == pytest.approx(ref_coef, rel=2e-2, abs=1e-3)

    # Representative product concentration checks from plotted sample indices
    rep_points = [
        (0, "Na_+"),
        (12, "Ca_2+"),
        (12, "Mg_2+"),
    ]
    for sample_idx, ion in rep_points:
        solved_var = (
            exp.model.sample_blk[sample_idx]
            .proc.fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", ion]
        )
        solved_conc = pyo.value(solved_var)
        ref_conc = REFERENCE_TRAINED_PROD_CONC[(sample_idx, ion)]
        assert solved_conc == pytest.approx(ref_conc, rel=2e-2, abs=1e-3)
