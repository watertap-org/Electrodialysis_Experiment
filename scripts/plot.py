import pyomo.environ as pyo
import pandas as pd
from IPython.display import display
from electrodialysis_experiment.experiment import MasterExperimentBuilder
import electrodialysis_experiment.schema.experiment.data as ds
from electrodialysis_experiment.surrogates.transport_number_membrane.cation_cem_simulator import SurrogateType
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import string

def main():
    dt = pd.read_parquet(
        "src/electrodialysis_experiment/data/raw/dt_SEDv4_021125.parquet"
    )
    size = 25
    # Build the experiment skeleton
    exp = MasterExperimentBuilder.proc_config_from_yaml(
        "src/electrodialysis_experiment/configs/one_stage_single_pass.yml",
        sample_size=size,
    )
    exp.add_cation_cem_transport_number_simulator(
        surrogate_method=SurrogateType.LOG_LINEAR_POLYNOMIAL,
        poly_degree=5,
        **{"reference_ion": "Na_+"},
    )
    exp.add_log_linear_surr_coef_constraint()
    exp.add_equal_ocv_constraint()

    # Load the solved experiment outcome.
    exp.load_model_data(
        "src/electrodialysis_experiment/data/output/m_concSSE_minimized_slkocvcu_with_surrloglin_cc_init1_cuun.h5")

    target_var_list, target_df = ds.prepare_target_variable_dt(dt)
    model = exp.model
    #plot_conc_ion_prod_mod_exp()
    #plot_trans_number_plotly(model, 0)
    ions = [
        {"comp": "Na_+", "label": "Na⁺", "color": "blue", "marker": "triangle-up"},
        {"comp": "Ca_2+", "label": "Ca²⁺", "color": "red", "marker": "triangle-up"},
        {"comp": "Mg_2+", "label": "Mg²⁺", "color": "green", "marker": "triangle-up"},
    ]
    figs = typical_plot(
        model=model,
        target_df=target_df,
        size=size,
        ions=ions,
        sample_indices=(0, 24),
        show=True,
    )

    exp_voltage = target_df.loc[:size, "fs.voltage_avg"].tolist()
    sim_voltage = [
        pyo.value(model.sample_blk[i].proc.fs.voltage_avg) for i in model.sample_set
    ]
    fig_voltage = plot_voltage(exp_voltage, sim_voltage, "blue", "circle")


def plot_conc_ion_prod_mod_exp():

    # dt = pd.read_parquet(
    #     "src/electrodialysis_experiment/data/raw/dt_SEDv4_021125.parquet"
    # )
    # size = 25
    # # Build the experiment skeleton
    # exp = MasterExperimentBuilder.proc_config_from_yaml(
    #     "src/electrodialysis_experiment/configs/one_stage_single_pass.yml",
    #     sample_size=size,
    # )
    # exp.add_cation_cem_transport_number_simulator(
    #     surrogate_method=SurrogateType.LOG_LINEAR_POLYNOMIAL,
    #     poly_degree=5,
    #     **{"reference_ion": "Na_+"},
    # )
    # exp.add_log_linear_surr_coef_constraint()
    # exp.add_equal_ocv_constraint()

    # # Load the solved experiment outcome.
    # exp.load_model_data("src/electrodialysis_experiment/data/output/m_concSSE_minimized_slkocvcu_with_surrloglin_cc_init1_cuun.h5")

    # target_var_list, target_df = ds.prepare_target_variable_dt(dt)
    # model = exp.model
    ions = [
        {"comp": "Na_+", "label": "Na⁺", "color": "blue", "marker": "triangle-up"},
        {"comp": "Ca_2+", "label": "Ca²⁺", "color": "red", "marker": "triangle-up"},
        {"comp": "Mg_2+", "label": "Mg²⁺", "color": "green", "marker": "triangle-up"},
    ]
    figs = typical_plot(
        model=model,
        target_df=target_df,
        size=size,
        ions=ions,
        sample_indices=(0, 24),
        show=True,
    )

    exp_voltage = target_df.loc[:size, "fs.voltage_avg"].tolist()
    sim_voltage = [
        pyo.value(model.sample_blk[i].proc.fs.voltage_avg) for i in model.sample_set
    ]
    fig_voltage = plot_voltage(exp_voltage, sim_voltage, "blue", "circle")


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
        title=f"Simulated vs Experimental Voltage",
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
    fig4 = plot_trans_number_plotly(exp.model, 0)
    # fig4.write_image("src/electrodialysis_experiment/data/derived/transport_number_0_021025.pdf")
    fig5 = plot_trans_number_plotly(exp.model, 12)
    # fig5.write_image("src/electrodialysis_experiment/data/derived/transport_number_12_021025.pdf")
    combined_fig2 = panel_from_figs([fig4, fig5], rows=1, cols=2)
    combined_fig2.show()
    # combined_fig2.write_image("data/temp_figs_2909/combined_transport_numbers_021025.pdf")
    pass

if __name__ == "__main__":
    main()