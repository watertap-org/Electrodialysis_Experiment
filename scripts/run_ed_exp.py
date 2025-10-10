# A script to run the electrodialysis experiment model with surrogate for cation-cem transport number

import pandas as pd
import electrodialysis_experiment.schema.data as ds
from electrodialysis_experiment.experiment import MasterExperimentBuilder
import IPython as ipy
import pyomo.environ as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import string
from IPython.display import display
import idaes.core.util.scaling as iscale
import idaes.core.util.model_statistics as mstat
import numpy as np
from electrodialysis_experiment.surrogates.transport_number_membrane.cation_cem_simulator import (
    CationCemTransportNumberSimulator,
    SurrogateType,
)
from electrodialysis_experiment.surrogates.transport_number_membrane.registry import (
    SURROGATE_IDENTITY,
)
from electrodialysis_experiment.surrogates.transport_number_membrane.log_linear_conc_ratio import (
    predict_ti_by_surrogate,
)


def plot_run():
    """
    A function to plot the results after running the experiment model.
    """
    dt = pd.read_parquet(
        "src/electrodialysis_experiment/data/raw/dt_x_y_4_061025.parquet"
    )
    display(dt)
    size = 25
    ion = _get_ion_dict()

    # Prepare data for the experiment
    target_var_list, target_df = ds.prepare_target_variable_dt(dt)
    # Build the experiment
    exp = MasterExperimentBuilder(ion=ion, sample_size=size, finite_diff_elements=100)
    exp.add_cation_cem_transport_number_simulator(
        surrogate_method=SurrogateType.LOG_LINEAR_POLYNOMIAL,
        poly_degree=5,
        **{"reference_ion": "Na_+"},
    )
    exp.add_log_linear_surr_coef_constraint()
    exp.add_equal_ocv_constraint()
    model = exp.model

    # Load the saved model for plotting
    exp.load_model_data(
        "src/electrodialysis_experiment/data/output/m_concSSE_minimized_slkocvcu_with_surrloglin.h5"
    )
    sim_Na = [
        pyo.value(
            model.sample_blk[i].fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
        )
        for i in model.sample_set
    ]
    exp_Na = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']"
    ].tolist()
    sim_Ca = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Ca_2+"]
        )
        for i in model.sample_set
    ]
    exp_Ca = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']"
    ].tolist()
    sim_Mg = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Mg_2+"]
        )
        for i in model.sample_set
    ]
    exp_Mg = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']"
    ].tolist()

    fig1 = plot_ion(exp_Na, sim_Na, "Na⁺", "blue", "triangle-up")
    fig2 = plot_ion(exp_Ca, sim_Ca, "Ca²⁺", "red", "triangle-up")
    fig3 = plot_ion(exp_Mg, sim_Mg, "Mg²⁺", "green", "triangle-up")
    combined_fig = panel_from_figs([fig1, fig2, fig3], rows=1, cols=3)
    combined_fig.show()
    ## Other plotting options
    # combined_fig.write_image("src/electrodialysis_experiment/data/derived/training_surr_091025.pdf")
    # fig4 = plot_trans_number_plotly(exp.model, 0)
    # fig4.write_image("src/electrodialysis_experiment/data/derived/transport_number_0_021025.pdf")
    # fig5 = plot_trans_number_plotly(exp.model, 12)
    # #fig5.write_image("src/electrodialysis_experiment/data/derived/transport_number_12_021025.pdf")
    # combined_fig2 = panel_from_figs([fig4, fig5], rows=1, cols=2)
    # combined_fig2.show()
    # combined_fig2.write_image("data/temp_figs_2909/combined_transport_numbers_021025.pdf")
    pass


def prepare_experiment():
    dt = pd.read_parquet(
        "src/electrodialysis_experiment/data/raw/dt_x_y_4_061025.parquet"
    )
    # display(dt)
    size = 25
    ion = _get_ion_dict()
    # Prepare data for the experiment
    fl_dt, param_dt, ti_data, ci_data = (
        ds.prepare_fluid_cond_dt(dt),
        ds.prepare_upd_param_dt(dt),
        ds.prepare_cation_cem_tranport_numbre_estimate(dt),
        ds.prepare_cation_product_conc(dt),
    )
    fl_dt_compatible = ds.prepare_fluid_cond_dt_compatible_to_calculate_state(fl_dt)
    target_var_list, target_df = ds.prepare_target_variable_dt(dt)

    # Create the experiment
    exp = MasterExperimentBuilder(ion=ion, sample_size=size, finite_diff_elements=100)

    # Condition the individual experiments; this solves each sed block at DOF=0 at the best-estiated conditions recorded in the param yaml.

    # exp.condition_individual_experiments(
    #     fluid_cond=fl_dt_compatible,
    #     param_dt_upd=param_dt,
    #     cation_cem_transport_number=ti_data,
    #     param_yaml="src/electrodialysis_experiment/configs/sed_single_pass_param.yaml",
    #     max_iter=500,
    # )

    # The model snapshot is saved after this step. This can be used to skip the conditioning step above, provided that a conditioned model snapshot has been obtained.
    # exp.save_model_hdf("src/electrodialysis_experiment/data/output/m_conditioned_dof0.h5")

    # Load the saved model snapshot; this can be used to skip the conditioning step above, provided that a conditioned model snapshot has been obtained.
    exp.load_model_data(
        "src/electrodialysis_experiment/data/output/m_conditioned_dof0.h5"
    )

    iscale.calculate_scaling_factors(exp.model)

    # Add the cation_cem_transport_number_simulator block to the experiment model
    exp.add_cation_cem_transport_number_simulator(
        surrogate_method=SurrogateType.LOG_LINEAR_POLYNOMIAL,
        poly_degree=5,
        **{"reference_ion": "Na_+"},
    )

    # Initiate the surrogate model by fitting the surrogate model to the experimentally-estimated transport numbers on block 0
    fitted_dict = exp.model.cation_cem_transport_number_simulator[0].initiate_surrogate(
        conc_data=ci_data,
        trans_number_data=ti_data,
        fitting_coef_guess={"Ca_2+": 9, "Mg_2+": 2},
        reference_ion="Na_+",
        log_objective=False,
        polynomial_degree=5,
    )

    # Optional additional initialization routines, at the user's discretion. These, however, may yield an improved or feasible initial point for the subsequent optimization step.

    # Routine 1. Set the surrogate initial param values and seek a better initial point by solving all individual sed blocks
    # for i, v in exp.model.cation_cem_transport_number_simulator.items():
    #     if i:
    #         v.conc_ratio_coef["Ca_2+"].set_value(fitted_dict["Ca_2+"])
    #         v.conc_ratio_coef["Mg_2+"].set_value(fitted_dict["Mg_2+"])

    # Fix all cation-cem transport numbers to values predicted from the initially-trained surrogate model
    # The "initially-trained" means the model was trained using cation-cem transport number data experimentally estimated by a empirical equation (\deltaCi/deltaCtotal)

    # model = exp.model
    # cation_set = model.sample_blk[0].fs.properties.cation_set
    # surr_fn = SURROGATE_IDENTITY["log_linear_polynomial"]
    # length_domain = model.sample_blk[0].fs.EDstack.diluate.length_domain
    # for i, v in model.sample_blk.items():
    #     dependent_conc = {
    #         j: np.array(
    #             [
    #                 pyo.value(
    #                     v.fs.EDstack.diluate.properties[0, x].conc_mol_phase_comp[
    #                         "Liq", j
    #                     ]
    #                 )
    #                 for x in length_domain
    #             ]
    #         )
    #         for j in cation_set
    #     }

    #     ion_trans_pred, _ = predict_ti_by_surrogate(
    #         ions=cation_set,
    #         surrogate_fn=surr_fn,
    #         reference_ion="Na_+",
    #         fitted_coef_dict=fitted_dict,
    #         dependent_conc=dependent_conc,
    #         degree=5,
    #         eps=1e-12,
    #     )
    #     assert len(ion_trans_pred["Na_+"]) == len(length_domain)
    #     for ion in cation_set:
    #         for ind, x in enumerate(length_domain):
    #             v.fs.EDstack.ion_trans_number_membrane["cem", ion, x].fix(
    #                 ion_trans_pred[ion][ind]
    #             )

    # #exp.free_cation_transport_numbers_in_cem()
    # exp.solve_individual_blocks(linear_solver="ma27", max_iter=500, tee=True)
    # check_badly_scaled_vars(exp.model)
    # exp.save_model_hdf("src/electrodialysis_experiment/data/output/m_conditioned_dof0_with_surr_ti.h5")

    # Rountine 2. Solve the entire model at DOF=0 to get a better initial point.
    model = exp.model
    for i, v in exp.model.cation_cem_transport_number_simulator.items():
        v.conc_ratio_coef["Ca_2+"].fix(fitted_dict["Ca_2+"])
        v.conc_ratio_coef["Mg_2+"].fix(fitted_dict["Mg_2+"])
    exp.free_cation_transport_numbers_in_cem()
    # exp.add_equal_ocv_constraint()

    for edfs in model.sample_blk.values():
        edfs.fs.ocv.unfix()
        edfs.fs.ocv.setlb(0)
        edfs.fs.ocv.setub(5.5)
        edfs.fs.EDstack.slack_resistance.fix(0)
        # edfs.fs.EDstack.current_utilization.fix(1)
        edfs.fs.EDstack.current_utilization.unfix()
        edfs.fs.EDstack.current_utilization.setlb(0.2)
        edfs.fs.EDstack.current_utilization.setub(1.0)
    print(f"DOF={mstat.degrees_of_freedom(model)}")
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 1000  # Set maximum iterations
    solver.options["mu_strategy"] = "adaptive"
    # solver.options["halt_on_ampl_error"] = "yes"
    solver.options["nlp_scaling_method"] = (
        "user-scaling"  # Use user-defined scaling user-scaling
    )
    solver.options["linear_solver"] = "mumps"

    try:
        results = solver.solve(model, tee=True)
    except KeyboardInterrupt:
        print("\n[!] Solver interrupted by user. Saving snapshot...")
    finally:
        # This runs both after success and after Ctrl-C
        exp.save_model_hdf(
            "src/electrodialysis_experiment/data/output/m_solved_slkocvcu_with_surr.h5"
        )

    ## Plotting
    sim_Na = [
        pyo.value(
            model.sample_blk[i].fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
        )
        for i in model.sample_set
    ]
    exp_Na = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']"
    ].tolist()
    sim_Ca = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Ca_2+"]
        )
        for i in model.sample_set
    ]
    exp_Ca = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']"
    ].tolist()
    sim_Mg = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Mg_2+"]
        )
        for i in model.sample_set
    ]
    exp_Mg = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']"
    ].tolist()

    fig1 = plot_ion(exp_Na, sim_Na, "Na⁺", "blue", "triangle-up")
    fig2 = plot_ion(exp_Ca, sim_Ca, "Ca²⁺", "red", "triangle-up")
    fig3 = plot_ion(exp_Mg, sim_Mg, "Mg²⁺", "green", "triangle-up")
    combined_fig = panel_from_figs([fig1, fig2, fig3], rows=1, cols=3)
    combined_fig.show()


def run_experiment():
    dt = pd.read_parquet(
        "src/electrodialysis_experiment/data/raw/dt_x_y_4_061025.parquet"
    )
    size = 25
    ion = _get_ion_dict()
    # Build the experiment
    exp = MasterExperimentBuilder(ion=ion, sample_size=size, finite_diff_elements=100)
    # Add the cation_cem_transport_number_simulator block to the experiment model
    exp.add_cation_cem_transport_number_simulator(
        surrogate_method=SurrogateType.LOG_LINEAR_POLYNOMIAL,
        poly_degree=5,
        **{"reference_ion": "Na_+"},
    )
    # Additional constraints across individual experiments
    exp.add_log_linear_surr_coef_constraint()
    exp.add_equal_ocv_constraint()

    # Load a saved model snapshot as the initial point; this can be from the prepare_experiment() function above or another saved model snapshot that is believed to be a good initial point.
    exp.load_model_data(
        "src/electrodialysis_experiment/data/output/m_solved_slkocvcu_with_surr.h5"
    )

    model = exp.model

    # Unfix some variables to become slack variables
    for edfs in model.sample_blk.values():
        if edfs.fs.ocv.is_fixed():
            edfs.fs.ocv.unfix()
        edfs.fs.ocv.setlb(0)
        edfs.fs.ocv.setub(5.5)
        edfs.fs.EDstack.slack_resistance.fix(0)
        if edfs.fs.EDstack.current_utilization.is_fixed():
            edfs.fs.EDstack.current_utilization.unfix()
        edfs.fs.EDstack.current_utilization.setlb(0.2)
        edfs.fs.EDstack.current_utilization.setub(1.0)

    for i, v in model.cation_cem_transport_number_simulator.items():
        v.conc_ratio_coef["Ca_2+"].unfix()
        v.conc_ratio_coef["Mg_2+"].unfix()

    print(f"DOF={mstat.degrees_of_freedom(model)}")
    assert mstat.degrees_of_freedom(model) == 28

    target_weights = {
        "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']": 0.1,
        "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']": 1,
        "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']": 10,
        "fs.current_density_avg": 0.05,
    }
    target_var_list, target_df = ds.prepare_target_variable_dt(dt)
    exp.add_sse_objective_of_selected_variables(
        variables_weights=target_weights, data=target_df[:size]
    )

    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 1000  # Set maximum iterations
    # solver.options["mu_strategy"] = "adaptive"
    # solver.options["halt_on_ampl_error"] = "yes"
    solver.options["nlp_scaling_method"] = (
        "user-scaling"  # Use user-defined scaling user-scaling
    )
    solver.options["linear_solver"] = "ma57"  # mumps Use MUMPS for linear solver ma57

    try:
        results = solver.solve(model, tee=True)
    except KeyboardInterrupt:
        print("\n[!] Solver interrupted by user. Saving snapshot...")
    finally:
        # This runs both after success and after Ctrl-C
        exp.save_model_hdf(
            "src/electrodialysis_experiment/data/output/m_conc_cd05_SSE_minimized_slkocvcu_with_surrloglin.h5"
        )

    for k, edfs in model.sample_blk.items():
        print(f"OCV of sample {k}: {pyo.value(edfs.fs.ocv)} V")
        print(
            f"Current utilization of sample {k}: {pyo.value(edfs.fs.EDstack.current_utilization)}"
        )

    for i, v in model.cation_cem_transport_number_simulator.items():
        v.conc_ratio_coef.pprint()

    ## Plotting
    sim_Na = [
        pyo.value(
            model.sample_blk[i].fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
        )
        for i in model.sample_set
    ]
    exp_Na = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']"
    ].tolist()
    sim_Ca = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Ca_2+"]
        )
        for i in model.sample_set
    ]
    exp_Ca = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']"
    ].tolist()
    sim_Mg = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Mg_2+"]
        )
        for i in model.sample_set
    ]
    exp_Mg = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']"
    ].tolist()

    sim_CurrD = [
        pyo.value(model.sample_blk[i].fs.current_density_avg) for i in model.sample_set
    ]
    exp_CurrD = target_df.loc[:size, "fs.current_density_avg"].tolist()

    fig1 = plot_ion(exp_Na, sim_Na, "Na⁺", "blue", "square")
    fig2 = plot_ion(exp_Ca, sim_Ca, "Ca²⁺", "red", "square")
    fig3 = plot_ion(exp_Mg, sim_Mg, "Mg²⁺", "green", "square")
    combined_fig = panel_from_figs([fig1, fig2, fig3], rows=1, cols=3)
    combined_fig.show()
    fig4 = plot_current_dens(exp_CurrD, sim_CurrD, "Current Density", "purple", "cross")
    fig4.show()


def _get_ion_dict():
    return {
        "solute_list": ["Na_+", "Ca_2+", "Mg_2+", "Cl_-"],
        "mw_data": {
            "H2O": 18e-3,
            "Na_+": 23e-3,
            "Mg_2+": 24.305e-3,
            "Ca_2+": 40.078e-3,
            "Cl_-": 35.5e-3,
        },
        "diffusivity_data": {
            ("Liq", "Na_+"): 1.33e-9,
            ("Liq", "Ca_2+"): 0.793e-9,
            ("Liq", "Mg_2+"): 0.705e-9,
            ("Liq", "Cl_-"): 2.03e-9,
        },  # referenced from https://www.aqion.de/site/diffusion-coefficients
        "charge": {"Na_+": 1, "Mg_2+": 2, "Ca_2+": 2, "Cl_-": -1},
        "membrane_transport_number": {
            "Na_+": {"cem": 0.916, "aem": 0},
            "Ca_2+": {"cem": 0.024, "aem": 0},
            "Mg_2+": {"cem": 0.059, "aem": 0},
            "Cl_-": {"cem": 0, "aem": 1},
        },
    }


def check_badly_scaled_vars(model):
    found = False
    for var, val in iscale.badly_scaled_var_generator(model, small=1e-4, large=1e4):
        print(f"Badly scaled var: {var}, value: {val}")
        found = True
    if not found:
        print("No badly scaled variables found.")


def plot_results(model, target_df, size):
    sim_Na = [
        pyo.value(
            model.sample_blk[i].fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
        )
        for i in model.sample_set
    ]
    exp_Na = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']"
    ].tolist()
    sim_Ca = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Ca_2+"]
        )
        for i in model.sample_set
    ]
    exp_Ca = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']"
    ].tolist()
    sim_Mg = [
        pyo.value(
            model.sample_blk[i]
            .fs.prod.properties[0]
            .conc_mol_phase_comp["Liq", "Mg_2+"]
        )
        for i in model.sample_set
    ]
    exp_Mg = target_df.loc[
        :size, "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']"
    ].tolist()
    sim_CurrD = [
        pyo.value(model.sample_blk[i].fs.current_density_avg) for i in model.sample_set
    ]
    exp_CurrD = target_df.loc[:size, "fs.current_density_avg"].tolist()

    plot_ion(exp_Na, sim_Na, "Na⁺", "blue", "circle")
    plot_ion(exp_Ca, sim_Ca, "Ca²⁺", "red", "square")
    plot_ion(exp_Mg, sim_Mg, "Mg²⁺", "green", "diamond")
    plot_current_dens(exp_CurrD, sim_CurrD, "Current Density", "purple", "cross")

    for i in range(0, size):
        plot_trans_number_plotly(model, i)


def data_cvs2parquet():

    # Prepare data
    df = pd.read_csv("data/SED_ref_v4_tnupd.csv", skiprows=1)
    df = df.dropna(how="all")
    # display(df)
    x_fields = [
        "fl",
        "CfNa",
        "CfCa",
        "CfMg",
        "Volt",
        "r_cem",
        "k_cem",
        "r_aem",
        "k_aem",
        "Dcem",
        "Daem",
    ]
    y_fields = ["CpNa", "CpCa", "CpMg", "CurrD"]
    # x_data_original=df[x_fields]
    # display(x_data_original)
    x_data = df[x_fields].apply(pd.to_numeric, errors="coerce")
    x_data.drop(0, inplace=True)
    print(x_data.loc[1, "CfNa"])
    print(type(x_data.loc[1, "CfNa"]))
    # x_data['CfNa']=x_data['CfNa']*1000
    # print(x_data['CfNa'])
    # print(type(x_data.loc[1,'CfNa']))
    # Convert mol/L to mol/m3
    x_data[["CfNa", "CfCa", "CfMg"]] = x_data[["CfNa", "CfCa", "CfMg"]] * 1000
    x_data["fl"] = x_data["fl"] / 60000  # Convert flow rate from L/h to m3/s
    y_data = df[y_fields].apply(pd.to_numeric, errors="coerce")
    y_data.drop(0, inplace=True)
    # Convert mol/L to mol/m3
    y_data[["CpNa", "CpCa", "CpMg"]] = y_data[["CpNa", "CpCa", "CpMg"]] * 1000
    y_data["CurrD"] = (
        y_data["CurrD"] * 10
    )  # Convert current density from A/m2 to A/m3 (assuming 1 m2 area for simplicity)

    # All data is in SI thus far.

    display(x_data)
    display(y_data)
    dt_x_y = pd.concat([x_data, y_data], axis=1)
    dt_x_y.to_parquet("data/dt_x_y_4_061025.parquet", index=False)


def plot_ion(exp, sim, ion_name, color, marker):
    min_val = np.floor(min(exp + sim))
    max_val = np.ceil(max(exp + sim)) + 0.5
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exp,
            y=sim,
            mode="markers",
            # name=ion_name,
            marker=dict(color=color, symbol=marker, size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            # name="y = x",
            line=dict(dash="dash", color="black"),
        )
    )
    fig.update_layout(
        # title=f"Simulated vs Experimental {ion_name} Concentration in the Product Solution",
        title=dict(text=ion_name, x=0.5, y=0.8, xanchor="center"),
        xaxis_title="Experimental (mol/m³)",
        yaxis_title="Simulated (mol/m³)",
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[min_val, max_val],
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[min_val, max_val],
        ),
        showlegend=False,
        # legend_title="Ion",
        # legend=dict(x=.9, y=.9),
        width=600,
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    # fig.show()
    return fig


def plot_current_dens(exp, sim, ion_name, color, marker):
    min_val = min(exp + sim)
    max_val = max(exp + sim)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exp,
            y=sim,
            mode="markers",
            name=ion_name,
            marker=dict(color=color, symbol=marker),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="y = x",
            line=dict(dash="dash", color="black"),
        )
    )
    fig.update_layout(
        title=f"Simulated vs Experimental {ion_name} Concentration",
        xaxis_title="Experimental (A/m²)",
        yaxis_title="Simulated (A/m²)",
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[min_val, max_val],
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[min_val, max_val],
        ),
        showlegend=False,
        width=600,
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.show()
    return fig


def plot_trans_number_plotly(model, sample_idx):
    ions = ["Na_+", "Ca_2+", "Mg_2+"]
    x_vals = list(model.length_domain)
    fig = go.Figure()
    for ion in ions:
        y_vals = [
            pyo.value(
                model.sample_blk[sample_idx].fs.EDstack.ion_trans_number_membrane[
                    "cem", ion, x
                ]
            )
            for x in x_vals
        ]
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=ion))
    fig.update_layout(
        title=f"Ion Transport Numbers along Length Domain (Sample {sample_idx})",
        xaxis_title="Length Domain (x)",
        yaxis_title="Ion Transport Number (CEM)",
        legend_title="Ion",
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[0, 1],
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[0, 1],
        ),
        width=700,
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.show()
    return fig


def compare_csvs(file1, file2, check_dtype=True, ignore_index_order=False):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if ignore_index_order:
        df1 = (
            df1.sort_index(axis=1)
            .sort_values(by=df1.columns.tolist())
            .reset_index(drop=True)
        )
        df2 = (
            df2.sort_index(axis=1)
            .sort_values(by=df2.columns.tolist())
            .reset_index(drop=True)
        )

    equal = df1.equals(df2) if check_dtype else df1.astype(str).equals(df2.astype(str))

    if equal:
        print("✅ The CSV files are identical.")
    else:
        print("❌ The CSV files differ.")

    return equal


def panel_from_figs(
    figs, rows, cols, titles=None, label_panels=True, hide_legends=True
):
    """
    Combine prebuilt Plotly figures into a single multi-panel figure.
    - figs: list of go.Figure
    - rows, cols: grid layout
    - titles: optional list of subplot titles
    - label_panels: add (a), (b), (c)... at top-left of each subplot
    - hide_legends: hide legends from all subplots (common for panels)
    """
    n = len(figs)
    assert n <= rows * cols, "More figures than grid slots."

    fig_panel = make_subplots(
        rows=rows, cols=cols, subplot_titles=(titles if titles else [None] * n)
    )

    # Add each figure's traces and carry over axis titles/ranges
    for i, f in enumerate(figs):
        r = i // cols + 1
        c = i % cols + 1

        # Add traces
        for tr in f.data:
            # (Optional) hide legends to keep the panel clean
            tr = tr.__class__(**tr.to_plotly_json())  # shallow copy
            if hide_legends:
                tr.showlegend = False
            fig_panel.add_trace(tr, row=r, col=c)

        # Carry over axis titles/ranges if present
        xa = f.layout.xaxis
        ya = f.layout.yaxis
        fig_panel.update_xaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_text=getattr(xa.title, "text", None),
            range=xa.range if hasattr(xa, "range") else None,
            row=r,
            col=c,
        )
        fig_panel.update_yaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_text=getattr(ya.title, "text", None),
            range=ya.range if hasattr(ya, "range") else None,
            row=r,
            col=c,
        )

        # Panel labels: (a), (b), ...
        if label_panels:
            label = f"({string.ascii_lowercase[i]})"
            fig_panel.add_annotation(
                text=label,
                xref="x domain",
                yref="y domain",
                x=0.0,
                y=1.08,  # top-left just above frame
                showarrow=False,
                font=dict(size=16),
                row=r,
                col=c,
            )

    # Global layout tweaks (optional)
    fig_panel.update_layout(
        width=1000,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=60, b=60),
        showlegend=not hide_legends,
    )

    return fig_panel


if __name__ == "__main__":
    run_experiment()
    # prepare_experiment()
    # single_experiment_test()
    # random_test()
    # sample_12_test()
    # plot_run()
    # data_cvs2parquet()
