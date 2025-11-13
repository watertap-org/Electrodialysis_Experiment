from typing import Dict, List, Tuple, Optional
import pandas as pd
from pydantic import BaseModel
from IPython.display import display

# -------------------
# Schema Definitions
# -------------------


class FluidCondition(BaseModel):
    flow_vol_phase: Dict[str, float]  # e.g., {"Liq": 1e-5}
    conc_mol_phase_comp: Dict[Tuple[str, str], float]  # e.g., {("Liq", "Na_+"): 0.1}

    def get_state_dict(self) -> Dict:
        """
        Convert this FluidCondition instance into a format
        compatible with calculate_state.
        """
        entry = {("flow_vol_phase", ("Liq",)): self.flow_vol_phase["Liq"]}
        for (phase, comp), conc in self.conc_mol_phase_comp.items():
            entry[("conc_mol_phase_comp", (phase, comp))] = conc
        return entry


class UpdateParam(BaseModel):
    experimental_voltage: Optional[float] = None
    current_applied: Optional[Dict[float, float]] = None
    membrane_thickness: Dict[str, float]
    membrane_areal_resistance_const: Dict[str, float]
    membrane_areal_resistance_coef: Dict[str, float]


class TargetVariable(BaseModel):
    name: str
    value: float
    weight: float


# -------------------
# Mapping Config (Optional)
# -------------------

COLUMN_MAPPING = {
    "flow": "fl",
    "feed_Na": "CfNa",
    "feed_Ca": "CfCa",
    "feed_Mg": "CfMg",
    "voltage": "Volt",
    "current": "Curr",
    "r_cem": "r_cem",
    "k_cem": "k_cem",
    "r_aem": "r_aem",
    "k_aem": "k_aem",
    "Dcem": "Dcem",
    "Daem": "Daem",
}

TargetVariableMapping = {
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']": "CpNa",
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']": "CpCa",
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']": "CpMg",
    "fs.current_density_avg": "CurrD",
}

# -------------------
# Data Preparation Functions
# -------------------


def prepare_fluid_cond_dt(df: pd.DataFrame) -> List[FluidCondition]:
    fluid_cond_list = []
    for _, r in df.iterrows():
        flow_vol_phase = {"Liq": r[COLUMN_MAPPING["flow"]]}
        conc_mol_phase_comp = {
            ("Liq", "Na_+"): r[COLUMN_MAPPING["feed_Na"]],
            ("Liq", "Ca_2+"): r[COLUMN_MAPPING["feed_Ca"]],
            ("Liq", "Mg_2+"): r[COLUMN_MAPPING["feed_Mg"]],
            ("Liq", "Cl_-"): r[COLUMN_MAPPING["feed_Na"]]
            + 2 * r[COLUMN_MAPPING["feed_Ca"]]
            + 2 * r[COLUMN_MAPPING["feed_Mg"]],
        }
        fluid_cond_list.append(
            FluidCondition(
                flow_vol_phase=flow_vol_phase, conc_mol_phase_comp=conc_mol_phase_comp
            )
        )
    return fluid_cond_list


def prepare_fluid_cond_dt_compatible_to_calculate_state(
    fluid_cond_dt: List[FluidCondition],
) -> List[Dict]:
    """
    Convert FluidCondition list to a format compatible with calculate_state.
    """
    fluid_cond_compatible = []
    for cond in fluid_cond_dt:
        entry = {("flow_vol_phase", ("Liq")): cond.flow_vol_phase["Liq"]}
        for (phase, comp), conc in cond.conc_mol_phase_comp.items():
            entry[("conc_mol_phase_comp", (phase, comp))] = conc
        fluid_cond_compatible.append(entry)
    return fluid_cond_compatible


def prepare_upd_param_dt_cv(df: pd.DataFrame) -> List[UpdateParam]:
    param_list = []
    for _, r in df.iterrows():
        param_list.append(
            UpdateParam(
                experimental_voltage=r[COLUMN_MAPPING["voltage"]],
                membrane_thickness={
                    "cem": r[COLUMN_MAPPING["Dcem"]],
                    "aem": r[COLUMN_MAPPING["Daem"]],
                },
                membrane_areal_resistance_const={
                    "cem": r[COLUMN_MAPPING["r_cem"]],
                    "aem": r[COLUMN_MAPPING["r_aem"]],
                },
                membrane_areal_resistance_coef={
                    "cem": r[COLUMN_MAPPING["k_cem"]],
                    "aem": r[COLUMN_MAPPING["k_aem"]],
                },
            )
        )
    return param_list


def prepare_upd_param_dt_cc(df: pd.DataFrame) -> List[UpdateParam]:
    param_list = []
    for _, r in df.iterrows():
        param_list.append(
            UpdateParam(
                current_applied={0: r[COLUMN_MAPPING["current"]]},
                membrane_thickness={
                    "cem": r[COLUMN_MAPPING["Dcem"]],
                    "aem": r[COLUMN_MAPPING["Daem"]],
                },
                membrane_areal_resistance_const={
                    "cem": r[COLUMN_MAPPING["r_cem"]],
                    "aem": r[COLUMN_MAPPING["r_aem"]],
                },
                membrane_areal_resistance_coef={
                    "cem": r[COLUMN_MAPPING["k_cem"]],
                    "aem": r[COLUMN_MAPPING["k_aem"]],
                },
            )
        )
    return param_list


def prepare_target_variable_dt(
    df: pd.DataFrame,
) -> Tuple[List[TargetVariable], pd.DataFrame]:
    target_var_list = []
    structured_rows = []
    for _, r in df.iterrows():
        row = {}
        for name, col in TargetVariableMapping.items():
            value = r[col]
            target_var_list.append(TargetVariable(name=name, value=value, weight=1.0))
            row[name] = value  # Use full variable path as column name
        structured_rows.append(row)

    target_df = pd.DataFrame(structured_rows)
    return target_var_list, target_df


# def prepare_target_variable_dt(df: pd.DataFrame) -> List[TargetVariable]:
#     target_var_list = []
#     for _, r in df.iterrows():
#         for name, col in TargetVariableMapping.items():
#             target_var_list.append(
#                 TargetVariable(
#                     name=name,
#                     value=r[col],
#                     weight=1.0  # Default weight, can be adjusted as needed
#                 )
#             )
#     return target_var_list


def prepare_cation_cem_transport_number_estimate(
    df: pd.DataFrame,
) -> List[Dict[str, float]]:
    """Compute transport numbers based on charge balance."""
    trans_list = []
    for _, row in df.iterrows():
        total_charge_in = row["CfNa"] + 2 * row["CfCa"] + 2 * row["CfMg"]
        total_charge_out = row["CpNa"] + 2 * row["CpCa"] + 2 * row["CpMg"]
        total_charge_removed = total_charge_in - total_charge_out

        if total_charge_removed == 0:
            # Avoid division by zero — handle as needed (e.g., skip or set to zero)
            t_Na = t_Ca = t_Mg = 0.0
        else:
            t_Na = (row["CfNa"] - row["CpNa"]) / total_charge_removed
            t_Ca = (row["CfCa"] - row["CpCa"]) * 2 / total_charge_removed
            t_Mg = (row["CfMg"] - row["CpMg"]) * 2 / total_charge_removed

        trans_list.append(
            {
                "Na_+": t_Na,
                "Ca_2+": t_Ca,
                "Mg_2+": t_Mg,
            }
        )

    return trans_list


def prepare_cation_product_conc(df: pd.DataFrame) -> List[Dict[str, float]]:
    """Compute cation product concentrations."""
    product_conc_list = []
    for _, row in df.iterrows():
        product_conc_list.append(
            {
                "Na_+": row["CpNa"],
                "Ca_2+": row["CpCa"],
                "Mg_2+": row["CpMg"],
            }
        )
    return product_conc_list


# -------------------
# Example Usage
# -------------------


def load_parquet_and_prepare(path: str):
    df = pd.read_parquet(path)
    fluid_conditions = prepare_fluid_cond_dt(df)
    update_params = prepare_upd_param_dt(df)
    trans_numbers = prepare_cation_cem_transport_number_estimate(df)
    return fluid_conditions, update_params, trans_numbers


def testing_fun():
    data = pd.read_parquet("dt/dt_x_y_1.parquet")
    display(data)
    df = data.iloc[[0, 1, 12]]
    display(df)
    fluid_conditions = prepare_fluid_cond_dt(df)
    print("===fulid conditions===")
    print(fluid_conditions)
    fc4cs = prepare_fluid_cond_dt_compatible_to_calculate_state(fluid_conditions)
    print("===fulid conditions for calculate_state()===")
    print(fc4cs)
    update_params = prepare_upd_param_dt(df)
    print("===updata_param_data===")
    print(update_params)
    trans_numbers = prepare_cation_cem_transport_number_estimate(df)
    print("===transport_num_init_data===")
    print(trans_numbers)


if __name__ == "__main__":
    testing_fun()
    # target_var_list, target_df= prepare_target_variable_dt(pd.read_parquet("dt/dt_x_y_1.parquet"))
    # print("Target Variables:")
    # print(target_var_list)
    # print(target_df)

    # for var in target_var_list:
    #     print(var)
    #     print(var.name)
