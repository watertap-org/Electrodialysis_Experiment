from electrodialysis_experiment.processes.one_stage_single_pass import (
    OneStageSinglePass,
)
from electrodialysis_experiment.schema.config.process_config_schema import (
    OneStageSinglePassConfig,
)
import electrodialysis_experiment.schema.experiment.data as dt
import pandas as pd
import pyomo.environ as pyo
from IPython.display import display
from idaes.core.solvers import get_solver


def main():
    edt = pd.read_parquet(
        "src/electrodialysis_experiment/data/raw/dt_SEDv4_021125.parquet"
    )
    ind=22
    display(edt)
    fluid_conditions = dt.prepare_fluid_cond_dt(edt)
    exp_setup_params = dt.prepare_upd_param_dt_cc(edt)
    t_est = dt.prepare_cation_cem_transport_number_estimate(edt)
    print(exp_setup_params[ind])
    test_cond = fluid_conditions[ind]
    print(test_cond)
    m = pyo.ConcreteModel()

    m.proc = OneStageSinglePass.from_yaml(
        "src/electrodialysis_experiment/configs/one_stage_single_pass.yml"
    )

    m.proc.import_scaling_config("src/electrodialysis_experiment/configs/scaling.yml")

    m.proc.import_init_value_config(
        "src/electrodialysis_experiment/configs/ossp_init_config.yml"
    )
    m.proc.update_var_values(exp_setup_params[ind])
    m.proc.update_cation_cem_transport_number(t_est[ind])
    m.proc.fs.EDstack.current_utilization.unfix()
    m.proc.fs.EDstack.current_utilization.setlb(0.5)
    m.proc.fs.EDstack.current_utilization.setub(1.0)
    m.proc.initialize_process( fluid_condition=test_cond, solver=get_solver() )
    solutes = m.proc.fs.properties.solute_set
    m.proc.display_selected_model_metrics(solutes)

    m.proc.plot_lengthwise_profile("current_density_x", precision=3)

if __name__ == "__main__":
    main()
