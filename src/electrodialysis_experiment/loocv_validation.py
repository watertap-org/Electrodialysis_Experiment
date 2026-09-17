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
import pyomo.environ as pyo
from idaes.core.util import model_statistics as mstat
from electrodialysis_experiment.experiment import MasterExperimentBuilder
from electrodialysis_experiment.surrogates.transport_number_membrane.cation_cem_simulator import (
    CationCemTransportNumberSimulator,
    SurrogateType,
)
from idaes.core.solvers import get_solver
import idaes.logger as idaeslogger
import electrodialysis_experiment.schema.experiment.data as ds
from typing import Iterable, List, Tuple, Union
import math
import pandas as pd

_log = idaeslogger.getLogger(__name__)


def build_exp(
    proc_config_path: str,
    init_model_h5_path: str,
    surrogate_method: SurrogateType,
    total_sample_size: int,
    sample_group_list: List[List[int]] = None,
    poly_degree: int = 5,
):

    exp = MasterExperimentBuilder.proc_config_from_yaml(
        proc_config_path,
        sample_size=total_sample_size,
    )

    exp.load_model_data(init_model_h5_path)
    model = exp.model

    exp.add_cation_cem_transport_number_simulator(
        surrogate_method=surrogate_method,
        poly_degree=poly_degree,
        **{"reference_ion": "Na_+"},
    )
    if sample_group_list is None:
        exp.add_log_linear_surr_coef_constraint()
    if sample_group_list is not None:
        exp.add_log_linear_surr_coef_constraint_grouped(
            group_list=sample_group_list,
            require_full_coverage=True,
        )
    exp.free_cation_transport_numbers_in_cem()

    for blk in model.sample_blk.values():
        blk.proc.fs.ocv.setlb(0)
        blk.proc.fs.ocv.setub(10)
        blk.proc.fs.EDstack.current_utilization.setlb(0.2)
        blk.proc.fs.EDstack.current_utilization.setub(1.0)

    return exp


def deactivate_all_objectives(model):
    for obj in model.component_objects(pyo.Objective, active=True):
        obj.deactivate()


def set_sse_objective_subset(
    exp,
    model: pyo.ConcreteModel = None,
    target_df_subset: pd.DataFrame = None,
    variable_weights: dict = None,
):
    """
    Must build SSE only for rows in target_df_subset.
    """
    if model is None:
        model = exp.model
    deactivate_all_objectives(model)
    if variable_weights is None:
        raise ValueError("Provide variable_weights dict.")
    exp.add_sse_objective_of_selected_variables(
        model=model,
        variables_weights=variable_weights,
        data=target_df_subset,
    )


def get_global_coef(model):
    sim = model.cation_cem_transport_number_simulator[0]
    return {
        "Ca_2+": pyo.value(sim.conc_ratio_coef["Ca_2+"]),
        "Mg_2+": pyo.value(sim.conc_ratio_coef["Mg_2+"]),
    }


def get_groupped_coef(model, group_list: List[List[int]]):
    coef_dict = {}
    for group_id, group in enumerate(group_list):
        sim = model.cation_cem_transport_number_simulator[group[0]]
        coef_dict[f"group_{group_id}_Ca_2+"] = pyo.value(sim.conc_ratio_coef["Ca_2+"])
        coef_dict[f"group_{group_id}_Mg_2+"] = pyo.value(sim.conc_ratio_coef["Mg_2+"])
    return coef_dict


def find_samples_at_nearest_applied_current(
    model,
    blk_ids,
    current_applied_value,
    near_sample_number=2,
):
    vals = []

    for i in blk_ids:
        cur = pyo.value(model.sample_blk[i].proc.fs.EDstack.current_applied[0])
        vals.append((i, cur, abs(cur - current_applied_value)))

    vals.sort(key=lambda t: t[2])
    return vals[:near_sample_number]


def get_prod_ion_conc_mol_simu(model, k):

    blk = model.sample_blk[k]
    out = {}
    out["Na"] = pyo.value(
        blk.proc.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
    )
    out["Ca"] = pyo.value(
        blk.proc.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Ca_2+"]
    )
    out["Mg"] = pyo.value(
        blk.proc.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Mg_2+"]
    )
    return out


def get_prod_ion_conc_mol_true(target_df, k):
    row = target_df.loc[k]
    out = {}
    out["Na"] = row["fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']"]
    out["Ca"] = row["fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']"]
    out["Mg"] = row["fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']"]
    return out


def weighted_sse(y_true, y_pred, metric_weights):
    return sum(
        metric_weights[sp] * (y_pred[sp] - y_true[sp]) ** 2 for sp in metric_weights
    )


def loocv(
    exp,
    total_sample_data: pd.DataFrame,
    solver=None,
    group_list=None,
    poly_degree=5,
    concIon_obj_weights: dict = {"Na_+": 1.0, "Ca_2+": 1.0, "Mg_2+": 1.0},
):

    if solver is None:
        _log.warning("No solver provided, a default solver will be set up and used.")
        sol_ipopt_idaes = get_solver()
        solver = config_ipopt_solver(
            solver=sol_ipopt_idaes,
            config_yaml="src/electrodialysis_experiment/configs/solver_config.yml",
        )
        solver.options["linear_solver"] = "ma57"

    target_var_list, target_df = ds.prepare_target_variable_dt(total_sample_data)
    ci_data = ds.prepare_cation_product_conc(total_sample_data)
    ti_data = ds.prepare_cation_cem_transport_number_estimate(total_sample_data)

    rows = []
    sample_ids = list(exp.model.sample_blk.keys())

    for k_test in sample_ids:
        train_ids = [k for k in sample_ids if k != k_test]
        _log.info(f"LOOCV: Test sample {k_test}, training on {train_ids}.")
        target_weights = {
            "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']": concIon_obj_weights[
                "Na_+"
            ],
            "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']": concIon_obj_weights[
                "Ca_2+"
            ],
            "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']": concIon_obj_weights[
                "Mg_2+"
            ],
        }

        # ---- TRAIN ----
        mod_k = exp.model.clone()
        mod_k.sample_blk[k_test].deactivate()
        mod_k.cation_cem_transport_number_simulator[k_test].deactivate()
        set_sse_objective_subset(exp, mod_k, target_df.loc[train_ids], target_weights)

        active_ci_data = [ci_data[i] for i in train_ids]
        active_ti_data = [ti_data[i] for i in train_ids]

        mod_k.cation_cem_transport_number_simulator[train_ids[0]].initiate_surrogate(
            conc_data=active_ci_data,
            trans_number_data=active_ti_data,
            fitting_coef_guess={"Ca_2+": 9, "Mg_2+": 2},
            reference_ion="Na_+",
            log_objective=False,
            polynomial_degree=poly_degree,
            plot_results=False,
        )

        dof = mstat.degrees_of_freedom(mod_k)
        print(f"DOF of the current training model ={dof}")

        solver.solve(mod_k, tee=True)
        if group_list is None:
            coef = get_global_coef(mod_k)
            _log.info(
                f"Fitted global coefficients with testing sample Test sample {k_test} excluded : {coef}"
            )
        else:
            coef = get_groupped_coef(mod_k, group_list)
            _log.info(
                f"Fitted grouped coefficients with testing sample Test sample {k_test} excluded : {coef}"
            )
        exp.save_model_hdf(
            model=mod_k,
            filename=f"src/electrodialysis_experiment/data/output/loocv_2_grouped/m_concSSE_trained_excl_sample_{k_test}.h5",
        )

        # ---- TEST ----
        if group_list is not None:
            for group_id, group in enumerate(group_list):
                if k_test not in group:
                    mod_k.cation_cem_transport_number_simulator[
                        group[0]
                    ].conc_ratio_coef.fix()
        testing_sample_blk = mod_k.sample_blk[k_test]
        testing_sample_blk.activate()
        mod_k.cation_cem_transport_number_simulator[k_test].activate()
        for k in train_ids:
            mod_k.sample_blk[k].deactivate()
            mod_k.cation_cem_transport_number_simulator[k].deactivate()

        _log.info(f"Testing on sample block {k_test}.")
        mod_k.cation_cem_transport_number_simulator[k_test].conc_ratio_coef.pprint()
        mod_k.cation_cem_transport_number_simulator[k_test].conc_ratio_coef.fix()

        near_current_samples = find_samples_at_nearest_applied_current(
            mod_k,
            train_ids,
            pyo.value(testing_sample_blk.proc.fs.EDstack.current_applied[0]),
            near_sample_number=2,
        )
        near_curr_sample_ids = [i[0] for i in near_current_samples]
        curr_util_test = sum(
            pyo.value(mod_k.sample_blk[j].proc.fs.EDstack.current_utilization)
            for j in near_curr_sample_ids
        ) / len(near_curr_sample_ids)
        testing_sample_blk.proc.fs.EDstack.current_utilization.fix(curr_util_test)

        # test_sample_dof = mstat.degrees_of_freedom(testing_sample_b)
        _log.info(
            f"DOF at the testing sample {k_test} stage ={mstat.degrees_of_freedom(mod_k)}"
        )
        solver.solve(mod_k, tee=True)

        exp.save_model_hdf(
            model=mod_k,
            filename=f"src/electrodialysis_experiment/data/output/loocv_2_grouped/m_concSSE_loocv_tested_sample_{k_test}.h5",
        )

        y_pred = get_prod_ion_conc_mol_simu(mod_k, k_test)
        y_true = get_prod_ion_conc_mol_true(target_df, k_test)

        rows.append(
            {
                "k_test": k_test,
                "surr_param_Ca": pyo.value(
                    mod_k.cation_cem_transport_number_simulator[k_test].conc_ratio_coef[
                        "Ca_2+"
                    ]
                ),
                "surr_param_Mg": pyo.value(
                    mod_k.cation_cem_transport_number_simulator[k_test].conc_ratio_coef[
                        "Mg_2+"
                    ]
                ),
                "prod_conc_mol_Na_pred": y_pred["Na"],
                "prod_conc_mol_Ca_pred": y_pred["Ca"],
                "prod_conc_mol_Mg_pred": y_pred["Mg"],
                "prod_conc_mol_Na_true": y_true["Na"],
                "prod_conc_mol_Ca_true": y_true["Ca"],
                "prod_conc_mol_Mg_true": y_true["Mg"],
                "current_utlization_test": pyo.value(
                    testing_sample_blk.proc.fs.EDstack.current_utilization
                ),
            }
        )

    return rows
