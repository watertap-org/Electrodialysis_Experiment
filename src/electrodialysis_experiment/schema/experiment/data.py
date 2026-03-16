from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Sequence
import pandas as pd
from pydantic import BaseModel
from IPython.display import display
from pathlib import Path

# Schema Definitions


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


class UpdateParam_4s(BaseModel):
    experimental_voltage_12: Optional[float] = None
    experimental_voltage_34: Optional[float] = None
    current_applied_: Optional[
        Union[Dict[float, float], Dict[str, Dict[float, float]]]
    ] = None
    cell_width: Optional[float] = None
    cell_length: Optional[float] = None
    cell_pair_num: Optional[float] = None
    membrane_thickness: Dict[str, float]
    membrane_areal_resistance_const: Dict[str, float]
    membrane_areal_resistance_coef: Dict[str, float]


class UpdateParam(BaseModel):
    experimental_voltage: Optional[float] = None
    current_applied: Optional[Dict[float, float]] = None
    membrane_thickness: Dict[str, float]
    membrane_areal_resistance_const: Dict[str, float]
    membrane_areal_resistance_coef: Dict[str, float]


class UpdateParam_stg(BaseModel):
    experimental_voltage: Optional[Dict[int, float]] = None
    voltage_applied: Optional[Dict[int, Dict[float, float]]] = None
    current_applied: Optional[Dict[int, Dict[float, float]]] = None
    cell_width: Optional[Dict[int, float]] = None
    cell_length: Optional[Dict[int, float]] = None
    cell_pair_num: Optional[Dict[int, float]] = None
    membrane_thickness: Optional[Dict[int, Dict[str, float]]] = None
    membrane_areal_resistance_const: Optional[Dict[int, Dict[str, float]]] = None
    membrane_areal_resistance_coef: Optional[Dict[int, Dict[str, float]]] = None


class UpdateStgConstCurr(BaseModel):
    current_applied: Dict[int, Dict[float, float]]


class TargetVariable(BaseModel):
    name: str
    value: float
    weight: float


# Mapping Config

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
    "Leff": "Leff",
    "Weff": "Weff",
    "voltage_I": "Volt_I",
    "voltage_II": "Volt_II",
    "current_I": "Curr_I",
    "current_II": "Curr_II",
    "cpn_I": "cpn_I",
    "cpn_II": "cpn_II",
    "wr": "WR",
}

TargetVariableMapping = {
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Na_+']": "CpNa",
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Ca_2+']": "CpCa",
    "fs.prod.properties[0].conc_mol_phase_comp['Liq','Mg_2+']": "CpMg",
    "fs.current_density_avg": "CurrD",
    "fs.voltage_avg": "Volt",
}

# Data Preparation Functions


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


def prepare_upd_param_dt_fssp_cv_stg(df: pd.DataFrame) -> List[UpdateParam_stg]:
    param_list = []
    for _, r in df.iterrows():
        param_list.append(
            UpdateParam_stg(
                cell_pair_num={
                    1: r[COLUMN_MAPPING["cpn_I"]] / 2,
                    2: r[COLUMN_MAPPING["cpn_I"]] / 2,
                    3: r[COLUMN_MAPPING["cpn_II"]] / 2,
                    4: r[COLUMN_MAPPING["cpn_II"]] / 2,
                },
            )
        )
    return param_list


def _get_shared_row_value(
    row: pd.Series,
    mapping_keys: Sequence[str],
    field_name: str,
    atol: float = 1e-12,
) -> float:
    """
    Resolve one scalar value from one or more candidate mapped columns.
    If multiple candidate columns are present, they must be numerically equal.
    """
    values = []
    cols = []

    for key in mapping_keys:
        col = COLUMN_MAPPING.get(key, key)
        if col in row.index and pd.notna(row[col]):
            values.append(float(row[col]))
            cols.append(col)

    if not values:
        mapped_cols = [COLUMN_MAPPING.get(key, key) for key in mapping_keys]
        raise KeyError(
            f"Cannot prepare '{field_name}': none of the candidate columns exist with non-null values. "
            f"Tried {mapped_cols}."
        )

    ref = values[0]
    for val, col in zip(values[1:], cols[1:]):
        if abs(val - ref) > atol:
            raise ValueError(
                f"Conflicting values for '{field_name}' across columns {cols}: {values}. "
                f"Provide one shared value for a universal stage update."
            )
    return ref


def prepare_upd_param_dt_tssp_cv(
    df: pd.DataFrame,
    stage_list: Sequence[int] = (1, 2),
) -> List[UpdateParam_stg]:
    """
    Prepare stage-wise update payload for a two-stage single-pass process:
      - stage_list[0] <- *_I columns (voltage_I, cpn_I)
      - stage_list[1] <- *_II columns (voltage_II, cpn_II)

    Shared geometry/membrane properties are applied to both stages.
    """
    stages = [int(s) for s in stage_list]
    if len(stages) != 2:
        raise ValueError(
            "prepare_upd_param_dt_tssp_cv expects exactly two stages in stage_list."
        )
    s1, s2 = stages

    param_list = []
    for _, r in df.iterrows():
        voltage_1 = float(r[COLUMN_MAPPING["voltage_I"]])
        voltage_2 = float(r[COLUMN_MAPPING["voltage_II"]])
        cpn_1 = float(r[COLUMN_MAPPING["cpn_I"]])
        cpn_2 = float(r[COLUMN_MAPPING["cpn_II"]])

        cell_width = float(r[COLUMN_MAPPING["Weff"]])
        cell_length = float(r[COLUMN_MAPPING["Leff"]])
        d_cem = float(r[COLUMN_MAPPING["Dcem"]])
        d_aem = float(r[COLUMN_MAPPING["Daem"]])
        r_cem = float(r[COLUMN_MAPPING["r_cem"]])
        r_aem = float(r[COLUMN_MAPPING["r_aem"]])
        k_cem = float(r[COLUMN_MAPPING["k_cem"]])
        k_aem = float(r[COLUMN_MAPPING["k_aem"]])

        param_list.append(
            UpdateParam_stg(
                experimental_voltage={
                    s1: voltage_1,
                    s2: voltage_2,
                },
                cell_width={s1: cell_width, s2: cell_width},
                cell_length={s1: cell_length, s2: cell_length},
                cell_pair_num={s1: cpn_1, s2: cpn_2},
                membrane_thickness={
                    s1: {"cem": d_cem, "aem": d_aem},
                    s2: {"cem": d_cem, "aem": d_aem},
                },
                membrane_areal_resistance_const={
                    s1: {"cem": r_cem, "aem": r_aem},
                    s2: {"cem": r_cem, "aem": r_aem},
                },
                membrane_areal_resistance_coef={
                    s1: {"cem": k_cem, "aem": k_aem},
                    s2: {"cem": k_cem, "aem": k_aem},
                },
            )
        )
    return param_list


def prepare_upd_param_dt_tssp_i_ii(
    df: pd.DataFrame,
    stage_1: int = 1,
    stage_2: int = 2,
) -> List[UpdateParam_stg]:
    """Backward-compatible wrapper around prepare_upd_param_dt_tssp_cv."""
    return prepare_upd_param_dt_tssp_cv(df, stage_list=[stage_1, stage_2])


def prepare_cation_cem_transport_number_estimate_tssp(
    df: pd.DataFrame,
    stage_1: int = 1,
    stage_2: int = 2,
    stage_1_feed_name: list = ["CfNa", "CfCa", "CfMg"],
    stage_1_product_name: list = ["CpNa_I", "CpCa_I", "CpMg_I"],
    stage_2_feed_name: list = ["CpNa_I", "CpCa_I", "CpMg_I"],
    stage_2_product_name: list = ["CpNa_II", "CpCa_II", "CpMg_II"],
) -> Dict[int, List[Dict[str, float]]]:
    """
    Compute stage-wise cation transport-number estimates for a 2-stage process:
      - stage_1: feed -> stage-I product
      - stage_2: stage-I product -> stage-II product
    """
    t_est_stage_1 = prepare_cation_cem_transport_number_estimate_vas(
        df,
        feed_name=stage_1_feed_name,
        product_name=stage_1_product_name,
    )
    t_est_stage_2 = prepare_cation_cem_transport_number_estimate_vas(
        df,
        feed_name=stage_2_feed_name,
        product_name=stage_2_product_name,
    )
    return {int(stage_1): t_est_stage_1, int(stage_2): t_est_stage_2}


def prepare_upd_param_dt_fssp_cv(df: pd.DataFrame) -> List[UpdateParam]:
    param_list = []
    for _, r in df.iterrows():
        param_list.append(
            UpdateParam_4s(
                experimental_voltage_12=r[COLUMN_MAPPING["voltage_I"]],
                experimental_voltage_34=r[COLUMN_MAPPING["voltage_II"]],
                cell_width=r[COLUMN_MAPPING["Weff"]],
                cell_length=r[COLUMN_MAPPING["Leff"]],
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


def prepare_const_curr_4st(df: pd.DataFrame) -> List[UpdateStgConstCurr]:
    param_list = []
    for _, r in df.iterrows():
        param_list.append(
            UpdateStgConstCurr(
                current_applied={
                    1: {0: r[COLUMN_MAPPING["current_I"]]},
                    2: {0: r[COLUMN_MAPPING["current_I"]]},
                    3: {0: r[COLUMN_MAPPING["current_II"]]},
                    4: {0: r[COLUMN_MAPPING["current_II"]]},
                }
            )
        )
    return param_list


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


def prepare_upd_param_dt_cc_sv(df: pd.DataFrame) -> List[UpdateParam]:
    """
    Constant Current mode with a steady voltage recorded.
    """
    param_list = []
    for _, r in df.iterrows():
        param_list.append(
            UpdateParam(
                current_applied={0: r[COLUMN_MAPPING["current"]]},
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


def prepare_cation_cem_transport_number_estimate_vas(
    df: pd.DataFrame,
    feed_name: list = ["CfNa", "CfCa", "CfMg"],
    product_name: list = ["CpNa", "CpCa", "CpMg"],
) -> List[Dict[str, float]]:
    """Compute transport numbers based on charge balance."""
    trans_list = []
    for _, row in df.iterrows():
        total_charge_in = sum(
            [row[name] * charge for name, charge in zip(feed_name, [1, 2, 2])]
        )
        total_charge_out = sum(
            [row[name] * charge for name, charge in zip(product_name, [1, 2, 2])]
        )
        total_charge_removed = total_charge_in - total_charge_out

        if total_charge_removed == 0:
            # Avoid division by zero — handle as needed (e.g., skip or set to zero)
            t_Na = t_Ca = t_Mg = 0.0
        else:
            t_Na = (row[feed_name[0]] - row[product_name[0]]) / total_charge_removed
            t_Ca = (row[feed_name[1]] - row[product_name[1]]) * 2 / total_charge_removed
            t_Mg = (row[feed_name[2]] - row[product_name[2]]) * 2 / total_charge_removed

        trans_list.append(
            {
                "Na_+": t_Na,
                "Ca_2+": t_Ca,
                "Mg_2+": t_Mg,
            }
        )

    return trans_list


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


def load_parquet_and_prepare(path: str):
    df = pd.read_parquet(path)
    fluid_conditions = prepare_fluid_cond_dt(df)
    update_params = prepare_upd_param_dt(df)
    trans_numbers = prepare_cation_cem_transport_number_estimate(df)
    return fluid_conditions, update_params, trans_numbers


def csv_to_parquet(
    *,
    csv_path: Union[str, Path],
    parquet_path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
    skiprows: int = 0,
    drop_all_nan_rows: bool = True,
    drop_first_row: bool = False,
    numeric_coerce: bool = True,
    multipliers: Optional[Dict[str, float]] = None,
    dropna_subset: Optional[Sequence[str]] = None,
    index: bool = False,
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Read a CSV file, optionally select columns, coerce numeric values,
    apply per-column scaling, and save the result as a Parquet file.

    Parameters
    ----------
    csv_path, parquet_path
        Input CSV path and output Parquet path.
    columns
        Columns to keep (and order). If None, all columns are kept.
    skiprows
        Passed to pd.read_csv(skiprows=...).
    drop_all_nan_rows
        Drop rows where all values are NaN.
    drop_first_row
        Drop the first data row after loading (useful for stray headers).
    numeric_coerce
        Convert values to numeric using errors="coerce".
    multipliers
        Dict mapping column name -> multiplicative factor.
        (Use values < 1 for division.)
    dropna_subset
        Drop rows with NaNs in these columns (after scaling).
    index
        Write parquet with index.
    return_df
        Return the processed DataFrame if True.

    Returns
    -------
    Optional[pd.DataFrame]
        Processed DataFrame if return_df=True, else None.

    Example
    -------
    >>> csv_to_parquet(
    ...     csv_path="data/raw.csv",
    ...     parquet_path="data/processed.parquet",
    ...     columns=["A", "B", "C"],
    ...     multipliers={"A": 1000, "B": 0.01},
    ... )
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, skiprows=skiprows)

    if drop_all_nan_rows:
        df = df.dropna(how="all")

    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        df = df[list(columns)].copy()
    else:
        df = df.copy()

    if numeric_coerce:
        df = df.apply(pd.to_numeric, errors="coerce")

    if drop_first_row:
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if multipliers is not None:
        for col, factor in multipliers.items():
            if col not in df.columns:
                raise ValueError(f"Multiplier references unknown column: {col}")
            df[col] *= factor

    if dropna_subset is not None:
        df = df.dropna(subset=list(dropna_subset))

    df.to_parquet(parquet_path, index=index)

    return df if return_df else None
