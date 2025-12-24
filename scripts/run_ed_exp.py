# A script to run the electrodialysis experiment model with surrogate for cation-cem transport number

import pandas as pd
import electrodialysis_experiment.schema.experiment.data as ds
from electrodialysis_experiment.experiment import MasterExperimentBuilder
import IPython as ipy
import pyomo.environ as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import string
from IPython.display import display
import idaes.core.util.scaling as iscale
from idaes.core.solvers import get_solver
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
from electrodialysis_experiment.processes.one_stage_single_pass import (
    OneStageSinglePass,
)
from electrodialysis_experiment.utils.solver_configuring import (
    config_ipopt_solver,
)


def main():
    dt = pd.read_parquet(
        "src/electrodialysis_experiment/data/raw/dt_SEDv4_021125.parquet"
    )
    size = 25

    # Prepare data for the experiment
    fl_dt, param_dt, ti_data, ci_data = (
        ds.prepare_fluid_cond_dt(dt),
        ds.prepare_upd_param_dt_cc_sv(dt),
        ds.prepare_cation_cem_transport_number_estimate(dt),
        ds.prepare_cation_product_conc(dt),
    )
    target_var_list, target_df = ds.prepare_target_variable_dt(dt)

    # Prepare the solver
    sol_ipopt_idaes = get_solver()
    solver = config_ipopt_solver(
        solver=sol_ipopt_idaes,
        config_yaml="src/electrodialysis_experiment/configs/solver_config.yml",
    )

    # Create the experiment
    exp = MasterExperimentBuilder.proc_config_from_yaml(
        "src/electrodialysis_experiment/configs/one_stage_single_pass.yml",
        sample_size=size,
    )

    # Initialize the individual sample blocks.

    # exp.initialize_individual_sample_blks(
    #     scaling_cfg_path="src/electrodialysis_experiment/configs/scaling.yml",
    #     process_init_cfg_path="src/electrodialysis_experiment/configs/ossp_init_config.yml",
    #     fluid_condition=fl_dt,
    #     exp_setup_param=param_dt,
    #     t_est=ti_data,
    #     solver=solver,
    # )
    # Save the model snapshot afterinitialization
    # exp.save_model_hdf("src/electrodialysis_experiment/data/output/cc_lowFidel_init0.h5")

    ## OR
    # Load the saved model snapshot; this can be used to skip the conditioning step above, provided that a conditioned model snapshot has been obtained.
    exp.load_model_data(
        "src/electrodialysis_experiment/data/output/cc_lowFidel_init0.h5"
    )

    model = exp.model

    # Plotting
    ions = [
        {"comp": "Na_+", "label": "Na⁺", "color": "blue", "marker": "square"},
        {"comp": "Ca_2+", "label": "Ca²⁺", "color": "red", "marker": "square"},
        {"comp": "Mg_2+", "label": "Mg²⁺", "color": "green", "marker": "square"},
    ]

    figs = typical_plot(
        model=model,
        target_df=target_df,
        size=size,
        ions=ions,
        sample_indices=(0, 12),
        show=True,
    )

    check_badly_scaled_vars(exp.model)

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
        plot_results=False,
    )
    model = exp.model
    exp.add_log_linear_surr_coef_constraint()
    exp.free_cation_transport_numbers_in_cem()
    for blk in model.sample_blk.values():
        blk.proc.fs.ocv.setlb(0)
        blk.proc.fs.ocv.setub(10)
        blk.proc.fs.EDstack.current_utilization.setlb(0.2)
        blk.proc.fs.EDstack.current_utilization.setub(1.0)

    print(f"DOF={mstat.degrees_of_freedom(model)}")
    assert mstat.degrees_of_freedom(model) == 27

    target_weights = {
        "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']": 0.1,
        "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']": 1,
        "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']": 10,
    }
    exp.add_sse_objective_of_selected_variables(
        variables_weights=target_weights, data=target_df[:size]
    )

    # Config solver for the optimization
    solver.options["linear_solver"] = "ma57"  # mumps Use MUMPS for linear solver ma57

    try:
        results = solver.solve(model, tee=True)
    except KeyboardInterrupt:
        print("\n[!] Solver interrupted by user. Saving snapshot...")
    finally:
        # This runs both after success and after Ctrl-C
        exp.save_model_hdf(
            "src/electrodialysis_experiment/data/output/m_concSSE_minimized_cc_copy.h5"
        )

    ## Plotting
    ions = [
        {"comp": "Na_+", "label": "Na⁺", "color": "blue", "marker": "circle"},
        {"comp": "Ca_2+", "label": "Ca²⁺", "color": "red", "marker": "circle"},
        {"comp": "Mg_2+", "label": "Mg²⁺", "color": "green", "marker": "circle"},
    ]
    figs = typical_plot(
        model=model,
        target_df=target_df,
        size=size,
        ions=ions,
        sample_indices=(0, 12),
        show=True,
    )


def typical_plot(model, target_df, size, ions, sample_indices=(0), show=True):
    """
    Generate ion concentration and transport-number plots.

    Parameters
    ----------
    model : Pyomo model
    target_df : pd.DataFrame
    size : int
    exp_model : object
        Model used in plot_trans_number_plotly.
    ions : list of dict
        Each dict: {"comp": str, "label": str, "color": str, "marker": str}
    transport_indices : tuple of int
    show : bool

    Returns
    -------
    dict of Plotly figures
    """

    ion_figs = []
    for ion in ions:
        comp = ion["comp"]
        sim_vals = [
            pyo.value(
                model.sample_blk[i]
                .proc.fs.prod.properties[0]
                .conc_mol_phase_comp["Liq", comp]
            )
            for i in model.sample_set
        ]
        col = f"fs.prod.properties[0].conc_mol_phase_comp['Liq','{comp}']"
        exp_vals = target_df.loc[:size, col].tolist()
        fig = plot_ion(exp_vals, sim_vals, ion["label"], ion["color"], ion["marker"])
        ion_figs.append(fig)

    combined_ions = panel_from_figs(ion_figs, rows=1, cols=len(ion_figs))
    if show:
        combined_ions.show()

    transport_figs = [plot_trans_number_plotly(model, idx) for idx in sample_indices]
    combined_transport = panel_from_figs(
        transport_figs, rows=1, cols=len(transport_figs)
    )
    if show:
        combined_transport.show()

    return {
        "ion_figs": ion_figs,
        "combined_ions": combined_ions,
        "transport_figs": transport_figs,
        "combined_transport": combined_transport,
    }


def check_badly_scaled_vars(model):
    found = False
    for var, val in iscale.badly_scaled_var_generator(model, small=1e-4, large=1e4):
        print(f"Badly scaled var: {var}, value: {val}")
        found = True
    if not found:
        print("No badly scaled variables found.")


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
        legend_title="current density, average",
        width=700,
        height=500,
    )
    fig.show()


def plot_voltage(exp, sim, color, marker):
    min_val = min(exp + sim)
    max_val = max(exp + sim)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exp,
            y=sim,
            mode="markers",
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
        xaxis_title="Experimental (V)",
        yaxis_title="Simulated (V)",
        legend_title="Applied voltage",
        width=700,
        height=500,
    )
    fig.show()


def plot_trans_number_plotly(model, sample_idx):
    ions = ["Na_+", "Ca_2+", "Mg_2+"]
    x_vals = list(model.sample_blk[0].proc.fs.EDstack.diluate.length_domain)
    fig = go.Figure()
    for ion in ions:
        y_vals = [
            pyo.value(
                model.sample_blk[sample_idx].proc.fs.EDstack.ion_trans_number_membrane[
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
    main()
