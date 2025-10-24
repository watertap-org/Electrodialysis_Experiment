from electrodialysis_experiment.processes.one_stage_single_pass import (
    OneStageSinglePass,
)
from electrodialysis_experiment.schema.config.process_config_schema import (
    OneStageSinglePassConfig,
)
import electrodialysis_experiment.schema.experiment.data as dt
import pandas as pd
import pyomo.environ as pyo


def main():
    edt = pd.read_parquet(
        "src/electrodialysis_experiment/data/raw/dt_x_y_4_061025.parquet"
    )
    fluid_conditions = dt.prepare_fluid_cond_dt(edt)
    test_cond = fluid_conditions[0]
    print(test_cond)
    m = pyo.ConcreteModel()

    m.proc = OneStageSinglePass.from_yaml(
        "src/electrodialysis_experiment/configs/one_stage_single_pass.yml"
    )
    # print(m.proc.__dict__)
    # m.proc.display()

    m.proc.import_scaling_config("src/electrodialysis_experiment/configs/scaling.yml")
    m.proc.import_init_value_config(
        "src/electrodialysis_experiment/configs/ossp_init_config.yml"
    )
    m.proc.initialize_process(fluid_condition=test_cond)
    solutes = m.proc.fs.properties.solute_set
    m.proc.display_selected_model_metrics(solutes)


if __name__ == "__main__":
    main()
